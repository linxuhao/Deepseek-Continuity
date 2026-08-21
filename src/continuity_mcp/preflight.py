# ==========================================
# 装机前的体检 —— 三个数字决定这台机器能装成什么样。
#
#   显存  硬门槛。够 8 GB 装全套; 不够就只装音频那半 (而不是装个装不动的全套)。
#   内存  软门槛。决定抠图默认走哪个模型 —— 这是唯一真正"自适应"的一项。
#   磁盘  硬门槛。权重是十几 GB, 下到一半没空间比一开始就拒绝糟糕得多。
#
# 显存只有一个门槛, 没有分档。8 GiB 以上一律同一套行为 —— 因为运行时的规则是
# "开工前把不是这件活要用的模型全卸掉", 峰值因此恒等于单个最大模型 (6.80 GiB),
# 与卡多大、与调用顺序都无关。大卡上留着模型不卸能省几秒, 但那要多一个配置项、
# 多一条只在大卡上走的代码路径, 而且会让"峰值多少"重新变成一个要看历史的问题。
#
# 8 GiB 以下不自动降级成"只装音频"那种另一个产品, 而是说清楚再问一句 ——
# 用户知道一些我不知道的事 (要不要升级显卡, 是不是只想要配音)。
#
# 换更小的生图模型省不下显存, 这是量出来的: Q4_0 (2.29 GiB 权重) 峰值 6.60 GiB,
# Q8_0 (4.01 GiB) 峰值 6.59 GiB, 一模一样; 降分辨率也一样 (512/768/1024 峰值相同,
# 只有耗时变)。瓶颈是那个 8 GiB 不量化的 4B 文本编码器, 不是扩散模型。
# ==========================================
import os
import re
import shutil
import subprocess
from pathlib import Path

GIB = 1024 ** 3

# 各项的门槛 (GiB)
IMAGE_PEAK_GIB = 6.80         # 生图实测峰值 (与分辨率和量化都无关)
VRAM_FOR_IMAGE = 8.0
VRAM_FOR_AUDIO = 4.0          # 单个音频模型实测峰值 3.62 GiB
RAM_FOR_BEST_CUTOUT = 12.0    # remove_bg quality=best 实测峰值 7.74 GB
# 磁盘按"装的过程中"算, 不是按"装完之后" —— 峰值出现在编译还没被回收的时候, 实测:
#   权重  生图 10.10 + 音频 7.33 GiB
#   运行镜像               2.11 GiB
#   构建中间层 (可回收)     8.47 GiB   <- 装完可以 docker builder prune 掉
# 全装峰值 ~28 GiB, 回收后 ~19.5 GiB; 只装音频峰值 ~18 GiB, 回收后 ~9.5 GiB。
# 卡在峰值上而不是终值上: 下到一半没空间比一开始就被拒绝糟糕得多。
DISK_FULL = 30.0
DISK_AUDIO_ONLY = 20.0


class PreflightError(RuntimeError):
    pass


def _run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)
    except (OSError, subprocess.SubprocessError):
        return None


# ---- docker ----

def check_docker():
    if not shutil.which("docker"):
        raise PreflightError(
            "找不到 docker。引擎以容器运行, 请先安装 Docker Engine 或 Docker Desktop:\n"
            "  https://docs.docker.com/engine/install/")
    r = _run(["docker", "compose", "version"])
    if r is None or r.returncode != 0:
        raise PreflightError(
            "docker 在, 但没有 compose 插件 (docker compose version 失败)。\n"
            "  Debian/Ubuntu: sudo apt install docker-compose-plugin")
    r = _run(["docker", "info"])
    if r is None or r.returncode != 0:
        raise PreflightError(
            "docker 守护进程连不上 (docker info 失败)。要么它没起来, 要么当前用户不在 "
            "docker 组:\n  sudo usermod -aG docker $USER  然后重新登录")
    return (r.stdout or "").strip().splitlines()[0] if r.stdout else "docker ok"


# ---- 显卡 ----

_DEV_RE = re.compile(r"^GPU(\d+):")
_FIELD_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\s*$")


def parse_vulkaninfo(text):
    """从 vulkaninfo 的完整输出里读出每块设备的 (index, name, type, 显存, 当前可用)。

    显存取 DEVICE_LOCAL 堆里最大的那个 size; free 取同一个堆的 budget
    (VK_EXT_memory_budget, 不是所有驱动都报, 报不出来就是 None)。
    两个数字用途不同, 混用会出错:
      size   是硬件属性 —— "这张卡够不够格" 只能问它。用 free 判会把一张正在打游戏
             的 16 GB 卡判成不合格。
      budget 是此刻的可用量 —— "该用哪张卡" 只能问它。实测本机 GPU0 总量 24 GiB 但
             budget 只剩 1.35 GiB (被另一个进程占着), 按总量选就会选中它然后 OOM。

    跳过 PHYSICAL_DEVICE_TYPE_CPU —— llvmpipe 这类软件光栅器会报出一个非常好看的
    数字 (实测 30.94 GiB, 就是主机内存), 照它选设备的话整套东西会在 CPU 上跑,
    慢到没法用, 而且看起来完全正常。
    """
    devices, cur = [], None
    heap = None
    for line in text.splitlines():
        m = _DEV_RE.match(line)
        if m:
            if cur:
                devices.append(cur)
            cur = {"index": int(m.group(1)), "name": "?", "type": "?",
                   "vram_gib": 0.0, "free_gib": None}
            heap = None
            continue
        if cur is None:
            continue
        f = _FIELD_RE.match(line)
        if f:
            k, v = f.group(1), f.group(2)
            if k == "deviceName":
                cur["name"] = v
            elif k == "deviceType":
                cur["type"] = v.replace("PHYSICAL_DEVICE_TYPE_", "")
            elif k == "size":
                heap = {"size": _bytes(v)}
            elif k == "budget" and heap is not None:
                heap["budget"] = _bytes(v)
        elif "MEMORY_HEAP_DEVICE_LOCAL_BIT" in line and heap and heap.get("size"):
            if heap["size"] / GIB > cur["vram_gib"]:
                cur["vram_gib"] = heap["size"] / GIB
                b = heap.get("budget")
                cur["free_gib"] = (b / GIB) if b else None
            heap = None
    if cur:
        devices.append(cur)
    return devices


def _bytes(v):
    n = re.match(r"(\d+)", v)
    return int(n.group(1)) if n else None


def dri_gids():
    """/dev/dri 上的真实 gid。

    必须用数字, 不能用组名: 容器的 /etc/group 里没有 video/render 这两个名字,
    `--group-add render` 直接被 daemon 拒掉 ("unable to find group render")。
    而 gid 本身在各发行版并不一致 (44 / 992 / 993 都见过), 所以只能从设备节点上读。
    """
    gids = {}
    dri = Path("/dev/dri")
    if dri.is_dir():
        for node in sorted(dri.iterdir()):
            try:
                gid = os.stat(node).st_gid
            except OSError:
                continue
            gids.setdefault("render" if node.name.startswith("renderD") else "video", gid)
    return gids.get("video", 44), gids.get("render", 993)


def looks_like_nvidia():
    if shutil.which("nvidia-smi") or Path("/dev/nvidiactl").exists():
        return True
    r = _run(["lspci"])
    return bool(r and r.stdout and "NVIDIA" in r.stdout)


def _dri_args():
    v, rd = dri_gids()
    return ["--device", "/dev/dri", "--group-add", str(v), "--group-add", str(rd)]


def _nvidia_args():
    # N 卡的 ICD 在宿主机驱动里, 由 nvidia-container-toolkit 注入。默认的
    # DRIVER_CAPABILITIES 只有 compute+utility —— 那样 CUDA 能用而 Vulkan 看不到设备,
    # 所以必须显式要 graphics。
    return ["--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility"]


def vulkan_devices(image=None):
    """跑一次 vulkaninfo, 返回 (设备表, 来源, 接入方式)。

    优先在引擎镜像里跑 —— 那才是引擎实际看到的东西: 它同时验了 docker 能不能把设备
    递进去、容器里有没有 ICD、卡认不认得出来。主机上的 vulkaninfo 只能算一个近似。

    两种接入方式要分别试: A 卡/Intel 靠 /dev/dri + 镜像里的 mesa ICD;
    N 卡靠 nvidia-container-toolkit 注入宿主机驱动, 只递 /dev/dri 是一块卡都看不见的。
    """
    if image:
        modes = [("dri", _dri_args())]
        if looks_like_nvidia():
            # 探到 N 卡就把 nvidia 方式排前面, 省一次注定失败的尝试
            modes.insert(0, ("nvidia", _nvidia_args()))
        for mode, args in modes:
            r = _run(["docker", "run", "--rm"] + args + [image, "vulkaninfo"])
            if r and r.returncode == 0 and "GPU0:" in (r.stdout or ""):
                return parse_vulkaninfo(r.stdout), "容器内", mode
    if shutil.which("vulkaninfo"):
        r = _run(["vulkaninfo"])
        if r and r.stdout and "GPU0:" in r.stdout:
            return parse_vulkaninfo(r.stdout), "主机", "nvidia" if looks_like_nvidia() else "dri"
    return [], "不可用", "dri"


def pick_device(devices):
    """选一块卡: 排除软件渲染, 然后按"此刻可用"最多的挑 (报不出可用量就退回按总量)。
    返回 (设备, 被跳过的)。"""
    real = [d for d in devices if d["type"] != "CPU"]
    skipped = [d for d in devices if d["type"] == "CPU"]
    if not real:
        return None, skipped
    return max(real, key=lambda d: (d["free_gib"] if d["free_gib"] is not None
                                    else d["vram_gib"])), skipped


# ---- 内存 / 磁盘 ----

def total_ram_gib():
    try:
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / GIB
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except (OSError, ValueError, AttributeError):
        return 0.0


def free_disk_gib(path):
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / GIB


# ---- 汇总 ----

def run(state_dir, image=None, want_image=True, want_audio=True):
    """跑全部体检, 返回一份计划。硬门槛不过就抛 PreflightError。"""
    report = {"docker": check_docker()}

    devices, source, mode = vulkan_devices(image)
    report["vulkan_source"] = source
    report["runtime_mode"] = mode
    report["devices"] = devices
    if not devices:
        raise PreflightError(
            "看不到任何 Vulkan 设备。可能是:\n"
            "  - 没装 Vulkan 驱动 (AMD: mesa-vulkan-drivers / NVIDIA: 官方驱动自带 / "
            "Intel: mesa-vulkan-drivers)\n"
            "  - /dev/dri 没递进容器, 或当前用户不在 video/render 组 (A 卡 / Intel)\n"
            "  - 没装 nvidia-container-toolkit (N 卡: 它的 Vulkan 驱动只能靠这个注进容器,\n"
            "    光递 /dev/dri 是看不见卡的)\n"
            "本插件不需要 CUDA 也不需要 ROCm, 但 Vulkan 是必须的。")

    dev, skipped = pick_device(devices)
    report["skipped_devices"] = skipped
    if dev is None:
        raise PreflightError(
            f"只看到软件渲染设备 ({', '.join(d['name'] for d in skipped)}), 没有真正的 GPU。"
            f"在它上面能跑通, 但慢到没有意义。")
    report["device"] = dev
    report["other_devices"] = [d for d in devices
                               if d["type"] != "CPU" and d["index"] != dev["index"]]

    vram = dev["vram_gib"]
    # 门槛只对"本机要装的那一半"生效。早先写成 `enable_audio = want_audio and 够显存`,
    # 然后 not enable_audio 就抛错 —— 于是 BYO 音频 (want_audio=False) 时必然抛,
    # 而且抛出来的是 "16.0 GiB 显存, 低于最低要求 4 GiB" 这种自相矛盾的话。
    # --audio-server 因此从一开始就是坏的, 而我只测过 --sd-server。
    if want_audio and vram < VRAM_FOR_AUDIO:
        raise PreflightError(
            f"{dev['name']} 只有 {vram:.1f} GiB 显存, 装音频那半至少要 {VRAM_FOR_AUDIO:.0f} GiB。\n"
            f"换更小的模型省不下这部分显存 —— 实测 Q4 与 Q8 的峰值相同 (6.60 / 6.59 GiB), "
            f"降分辨率也一样, 瓶颈是文本编码器。\n"
            f"音频后端在别处的话, 用 --audio-server <url>。")
    enable_audio = want_audio
    enable_image = want_image and vram >= VRAM_FOR_IMAGE
    if want_image and not enable_image:
        # 不自作主张换成另一个产品 —— setup 会把这段摆出来问一句
        report["image_warning"] = (
            f"{dev['name']} 只有 {vram:.1f} GiB, 而生图实测峰值 {IMAGE_PEAK_GIB:.2f} GiB, "
            f"需要 {VRAM_FOR_IMAGE:.0f} GiB。\n"
            f"          换更小的生图模型省不下这部分 (Q4 与 Q8 峰值相同 6.60 / 6.59), "
            f"降分辨率也不行 —— 瓶颈是那个 8 GiB 不量化的文本编码器。\n"
            f"          音频那半仍然可以装: 铸声/配音/音乐/音效/抠图都能用, 4 GiB 就够。")
    report["enable_image"] = enable_image
    report["enable_audio"] = enable_audio

    ram = total_ram_gib()
    report["ram_gib"] = ram
    report["cutout_quality"] = "best" if ram >= RAM_FOR_BEST_CUTOUT else "fast"
    if report["cutout_quality"] == "fast":
        report["cutout_reason"] = (
            f"本机内存 {ram:.1f} GiB < {RAM_FOR_BEST_CUTOUT:.0f} GiB, 抠图默认档设为 fast "
            f"(峰值 1.33 GB, 而 best 是 7.74 GB)。单次调用仍可传 quality='best' 覆盖。")

    need = DISK_FULL if enable_image else DISK_AUDIO_ONLY
    free = free_disk_gib(state_dir)
    report["disk_free_gib"] = free
    report["disk_need_gib"] = need
    if free < need:
        raise PreflightError(
            f"{state_dir} 所在分区只剩 {free:.1f} GiB, 需要 {need:.0f} GiB "
            f"(权重 {'17.4' if enable_image else '7.3'} + 运行镜像 2.1 + 编译中间层 8.5 GiB;"
            f" 装完 docker builder prune 能回收 8.5 GiB)。\n"
            f"换个盘: CONTINUITY_STATE_DIR=/别的/路径 continuity-setup")
    return report


def format_report(r):
    out = ["体检结果:"]
    d = r["device"]
    free = f", 此刻可用 {d['free_gib']:.1f} GiB" if d.get("free_gib") is not None else ""
    out.append(f"  GPU     {d['name']}  ({d['vram_gib']:.1f} GiB{free}, {d['type']}, "
               f"vulkan device {d['index']}, 来源: {r['vulkan_source']})")
    for o in r.get("other_devices", []):
        of = f", 此刻可用 {o['free_gib']:.1f} GiB" if o.get("free_gib") is not None else ""
        out.append(f"          未选 {o['name']} ({o['vram_gib']:.1f} GiB{of})")
    for s in r.get("skipped_devices", []):
        out.append(f"          跳过 {s['name']} —— 软件渲染, 不是真显卡")
    out.append(f"  内存    {r['ram_gib']:.1f} GiB")
    out.append(f"  磁盘    {r['disk_free_gib']:.1f} GiB 可用 / 需要 {r['disk_need_gib']:.0f} GiB")
    out.append(f"  生图    {'启用' if r['enable_image'] else '显存不足'}")
    if r.get("image_warning"):
        out.append(f"          {r['image_warning']}")
    out.append(f"  音频    {'启用' if r['enable_audio'] else '未启用'}")
    out.append(f"  抠图默认档  {r['cutout_quality']}")
    if r.get("cutout_reason"):
        out.append(f"          {r['cutout_reason']}")
    if r.get("runtime_mode") == "nvidia":
        out.append("  接入方式  nvidia-container-toolkit (⚠️ 这条路径没有在 N 卡上实测过)")
    return "\n".join(out)
