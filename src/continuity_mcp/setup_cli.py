# ==========================================
# continuity-setup —— 一条命令装好本机后端。
#
# 顺序是刻意的, 而且是按"最贵的一步放最后"排的:
#   1. 便宜的体检 (docker / 内存 / 磁盘)          秒级
#   2. 编引擎镜像                                  十几分钟
#   3. 在镜像里探 Vulkan -> 显存 -> 定下要不要装生图  秒级
#   4. 只下这台机器用得上的权重                     10 或 17 GiB
#   5. 起服务并验活
#
# 第 3 步必须在第 4 步之前: 显存不够的机器不装生图, 那是 10 GiB 的下载量差别 ——
# 先下完再告诉用户"你这卡用不了", 是最差的一种顺序。
# 第 3 步必须在第 2 步之后: 在引擎镜像里探到的才是引擎实际看得见的东西, 它同时验了
# /dev/dri 有没有递进去、容器里有没有 ICD。主机上探只是个近似。
# ==========================================
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import preflight

DEPLOY = Path(__file__).resolve().parent / "deploy"
ENGINES_IMAGE = "continuity-engines:latest"


def say(msg=""):
    print(msg, flush=True)


def step(n, total, msg):
    say(f"\n[{n}/{total}] {msg}")


def die(msg, code=1):
    say(f"\n✗ {msg}")
    sys.exit(code)


def sh(cmd, **kw):
    say("    $ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def build_engines(no_cache=False):
    cmd = ["docker", "build", "-t", ENGINES_IMAGE, str(DEPLOY)]
    if no_cache:
        cmd.insert(2, "--no-cache")
    say("    从源码编译 stable-diffusion.cpp + audio.cpp (Vulkan 后端)。")
    say("    这一步慢 —— 十几分钟很正常, 编的是 shader 和 host 代码。")
    if sh(cmd).returncode != 0:
        die("引擎镜像构建失败。上面的日志里通常直接写着缺哪个包。")


def verify_manifest(todo):
    """下载之前先把每个 URL 都 HEAD 一遍。

    存在的意义: 清单里的路径写错时, 原先要下到那一个才炸 —— 前面几 GiB 白下, 而且
    只报第一个错。一次性验完, 一次说清哪几条不对。
    (这条检查是有来历的: 0.1.1 的清单把 stable-audio 的路径少写了一层子目录,
     而我历次测试都用硬链接预置好了权重, 从来没真正走过下载这条路。)"""
    import urllib.error
    import urllib.request
    bad = []
    for m in todo:
        url = f"https://huggingface.co/{m['repo_id']}/resolve/main/{m['filename']}"
        req = urllib.request.Request(url, method="HEAD")
        try:
            urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 307, 308):
                bad.append(f"{m['repo_id']}/{m['filename']} -> HTTP {e.code}")
        except Exception as e:
            bad.append(f"{m['repo_id']}/{m['filename']} -> {e}")
    return bad


def download_models(models_dir, groups):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("缺 huggingface_hub。装法: pip install huggingface_hub\n"
            "  (或者用 uvx 跑本工具: uvx --from dsh-continuity continuity-setup)")
    manifest = json.loads((DEPLOY / "models.json").read_text(encoding="utf-8"))
    todo = [m for m in manifest["models"] if m["group"] in groups]
    total = sum(m["size_bytes"] for m in todo) / preflight.GIB
    missing = [m for m in todo
               if not (models_dir / m["dest"]).exists()
               or (models_dir / m["dest"]).stat().st_size != m["size_bytes"]]
    if missing:
        say(f"    先核对 {len(missing)} 个待下载文件的地址...")
        bad = verify_manifest(missing)
        if bad:
            die("权重清单里的这些地址取不到:\n  " + "\n  ".join(bad) +
                "\n上游仓库可能改了路径。请开 issue: "
                "https://github.com/linxuhao/Deepseek-Continuity/issues")
    say(f"    需要 {len(todo)} 个文件, 共 {total:.1f} GiB。已经下过的会跳过。")
    for i, m in enumerate(todo, 1):
        dest = models_dir / m["dest"]
        if dest.exists() and dest.stat().st_size == m["size_bytes"]:
            say(f"    [{i}/{len(todo)}] 已有 {m['dest']}")
            continue
        say(f"    [{i}/{len(todo)}] {m['dest']}  ({m['size_bytes'] / preflight.GIB:.2f} GiB)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            # local_dir: 直接下到目标目录, 不经过 ~/.cache/huggingface。
            # 原先是"下到缓存再 copy 一份", 同一块盘上同时存在两份 —— 17.4 GiB 的
            # 安装实际要 35 GiB, 而体检只按一份算, 于是空间刚够的机器会在编完镜像、
            # 下到一半时 ENOSPC。而且那次 copy 在 try 之外, 报出来是条裸 traceback。
            got = hf_hub_download(repo_id=m["repo_id"], filename=m["filename"],
                                  local_dir=str(models_dir / "_hf"))
            Path(got).replace(dest)
        except Exception as e:
            die(f"下载 {m['repo_id']}/{m['filename']} 失败: {e}\n"
                f"  网络不通或仓库改名了。可以手动放到 {dest} 再重跑本命令。")
    shutil.rmtree(models_dir / "_hf", ignore_errors=True)
    return total


def write_config(state_dir, models_dir, report, args, byo=None):
    for sub in ("actors", "subjects", "generated"):
        (state_dir / sub).mkdir(parents=True, exist_ok=True)
    dev_index = report.get("device", {}).get("index", 0)
    (state_dir / "audio_server.json").write_text(
        (DEPLOY / "audio_server.json.tmpl").read_text(encoding="utf-8")
        .replace("__DEVICE__", str(dev_index)), encoding="utf-8")

    byo = byo or {}
    video_gid, render_gid = preflight.dri_gids()
    env = {
        "CONTINUITY_STATE_DIR": str(state_dir),
        "CONTINUITY_MODELS_DIR": str(models_dir),
        "VULKAN_DEVICE": str(dev_index),
        "VIDEO_GID": str(video_gid),
        "RENDER_GID": str(render_gid),
        "SD_PORT": str(args.sd_port),
        "AUDIO_PORT": str(args.audio_port),
        # MCP server 要连的地址: BYO 那半是用户给的, 其余是本机起的
        "SD_SERVER": byo.get("image") or f"http://127.0.0.1:{args.sd_port}",
        "AUDIO_SERVER": byo.get("audio") or f"http://127.0.0.1:{args.audio_port}",
        "SD_DIFFUSION_MODEL": "flux-2-klein-4b-Q4_0.gguf",
    }
    (state_dir / "compose.env").write_text(
        "\n".join(f"{k}={v}" for k, v in env.items()) + "\n", encoding="utf-8")
    return env


def compose_files(mode):
    files = ["-f", str(DEPLOY / "docker-compose.yml")]
    if mode == "nvidia":
        files += ["-f", str(DEPLOY / "docker-compose.nvidia.yml")]
    return files


def compose(state_dir, env, profiles, *args_, mode="dri"):
    cmd = ["docker", "compose"] + compose_files(mode) + [
           "--env-file", str(state_dir / "compose.env"), "-p", "continuity"]
    for p in profiles:
        cmd += ["--profile", p]
    return sh(cmd + list(args_), env={**os.environ, **env})


def wait_healthy(env, profiles, timeout=300):
    import time
    os.environ.setdefault("SD_SERVER", f"http://127.0.0.1:{env['SD_PORT']}")
    os.environ.setdefault("AUDIO_SERVER", f"http://127.0.0.1:{env['AUDIO_PORT']}")
    from . import engines
    engines.SD_SERVER = env["SD_SERVER"]
    engines.AUDIO_SERVER = env["AUDIO_SERVER"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, down = engines.health()
        # 只装了音频时 sd-server 本来就不该在
        down = [d for d in down if not (d.startswith("sd_server") and "image" not in profiles)]
        if not down:
            return True, []
        time.sleep(3)
    return False, down


def cmd_check(args):
    state_dir = Path(args.state_dir).expanduser()
    image = ENGINES_IMAGE if _image_exists() else None
    try:
        r = preflight.run(state_dir, image=image,
                          want_image=not args.no_image, want_audio=True)
    except preflight.PreflightError as e:
        die(str(e))
    say(preflight.format_report(r))
    if image is None:
        say("\n  (引擎镜像还没构建, 显卡信息来自主机的 vulkaninfo 或不可用 ——\n"
            "   跑 continuity-setup 时会在镜像里重新探一次, 那次才是引擎真正看到的。)")
    return r


def _image_exists():
    r = subprocess.run(["docker", "image", "inspect", ENGINES_IMAGE], capture_output=True)
    return r.returncode == 0


def _image_is_current():
    """镜像里该有的东西都在吗。

    不比版本号, 比能力 —— 缺什么就直接查什么。目前查 model_specs/: 少了它,
    /v1/audio/speech 会报 "model contract spec not found for family ...", 而那是唯一
    能收内联参考音的端点, 于是配音整条路不通。
    这一条是踩出来的: 0.1.4 之前的镜像没有它, 而 --skip-build 的人会一直留着旧镜像,
    表现是装机一切正常、一配音就 500。"""
    if not _image_exists():
        return False
    r = subprocess.run(["docker", "run", "--rm", "--entrypoint", "test", ENGINES_IMAGE,
                        "-f", "/opt/continuity/model_specs/qwen3_tts.json"],
                       capture_output=True)
    return r.returncode == 0


def cmd_install(args):
    """装本机后端。

    "BYO 某一半" 是一等公民: --sd-server / --audio-server 给了地址, 就表示那一半你自己
    提供 —— 这一半的权重不下、引擎不起、显存门槛也不查, 但工具照常注册。
    早先没有这两个选项, BYO 生图的人仍然会下 10.1 GiB 权重并起一个永远用不到的
    sd-server, 而且还得自己知道事后把 CONTINUITY_ENABLE_IMAGE 覆写回 1。
    """
    # resolve(): 相对路径原样写进 compose.env 的话, compose 会按 compose 文件所在目录
    # (site-packages 里的 deploy/) 去解释它 —— docker 会在那儿建一个空的 root 目录,
    # 而权重好端端躺在 $PWD 下, 引擎报"文件不存在"。
    state_dir = Path(args.state_dir).expanduser().resolve()
    models_dir = (Path(args.models_dir).expanduser().resolve()
                  if args.models_dir else state_dir / "models")
    byo = {"image": args.sd_server, "audio": args.audio_server}
    local = {"image": not byo["image"] and not args.no_image, "audio": not byo["audio"]}
    if not any(local.values()):
        say("    两半都是 BYO —— 本机不装任何引擎, 只写配置。")

    total = 5
    step(1, total, "体检")
    try:
        preflight.check_docker() if any(local.values()) else None
    except preflight.PreflightError as e:
        die(str(e))
    ram = preflight.total_ram_gib()
    free = preflight.free_disk_gib(state_dir)
    say(f"    内存 {ram:.1f} GiB, {state_dir} 所在分区可用 {free:.1f} GiB")
    for cap, url in byo.items():
        if url:
            say(f"    {cap}: BYO -> {url} (本机不装这一半)")

    report = {"ram_gib": ram, "runtime_mode": "dri",
              "cutout_quality": "best" if ram >= preflight.RAM_FOR_BEST_CUTOUT else "fast"}
    if any(local.values()):
        step(2, total, "构建引擎镜像")
        if args.skip_build and _image_exists() and not _image_is_current():
            say("    已有镜像, 但它缺 model_specs/ —— 那样配音会在 /v1/audio/speech 上 500。")
            say("    忽略 --skip-build, 重编一次。")
            build_engines(args.no_cache)
        elif args.skip_build and _image_exists():
            say("    已有镜像, 跳过 (--skip-build)")
        else:
            build_engines(args.no_cache)

        step(3, total, "探测显卡")
        try:
            report = preflight.run(state_dir, image=ENGINES_IMAGE,
                                   want_image=local["image"], want_audio=local["audio"])
        except preflight.PreflightError as e:
            die(str(e))
        report["byo"] = {k: v for k, v in byo.items() if v}
        say(preflight.format_report(report))
        if local["image"] and report.get("image_warning") and not args.yes:
            # 不替用户决定: 他知道一些我不知道的事 (要不要换显卡, 是不是本来就只想要配音)
            say("")
            say("  ⚠️ 这张卡装不了生图那半。")
            say("     (生图后端在别处的话, 用 --sd-server <url> 重跑 —— 那样工具照常可用。)")
            if input("     只装音频那半 (铸声/配音/音乐/音效/抠图)? [y/N] ").strip().lower() \
                    not in ("y", "yes"):
                die("已取消。换一张 8 GiB 以上的卡, 或用 --sd-server 指向你自己的后端。", 0)
            local["image"] = False
        local["image"] = local["image"] and report.get("enable_image", False)
    else:
        say("\n[2-3/5] 不装本机引擎, 跳过镜像构建和显卡探测")

    groups = {c for c, v in local.items() if v}
    if groups and not args.yes:
        manifest = json.loads((DEPLOY / "models.json").read_text(encoding="utf-8"))
        gib = sum(m["size_bytes"] for m in manifest["models"]
                  if m["group"] in groups) / preflight.GIB
        say(f"\n    接下来要下载 {gib:.1f} GiB 权重到 {models_dir}。")
        if input("    继续? [Y/n] ").strip().lower() in ("n", "no"):
            die("已取消。", 0)

    step(4, total, "下载权重")
    if groups:
        models_dir.mkdir(parents=True, exist_ok=True)
        download_models(models_dir, groups)
    else:
        say("    没有要下的 —— 两半都是 BYO。")

    step(5, total, "启动引擎")
    # 能力开关看的是"有没有后端", 不是"本机装没装" —— BYO 那半照样要注册工具
    report["enable_image"] = bool(local["image"] or byo["image"])
    report["enable_audio"] = bool(local["audio"] or byo["audio"])
    env = write_config(state_dir, models_dir, report, args, byo)
    profiles = sorted(groups)
    mode = report.get("runtime_mode", "dri")
    if profiles:
        if compose(state_dir, env, profiles, "up", "-d", mode=mode).returncode != 0:
            die("docker compose up 失败。")
        say("    等引擎读完权重 (冷启动要从磁盘读十几 GB, 可能几十秒)...")
        ok, down = wait_healthy(env, profiles)
        if not ok:
            die("引擎起来了但没通过健康检查:\n  " + "\n  ".join(down) +
                "\n看日志: docker compose -p continuity logs")
    else:
        say("    没有本机引擎要起。")

    _print_done(state_dir, report, env, profiles, mode, byo)


def _print_done(state_dir, report, env, profiles, mode="dri", byo=None):
    say("\n" + "=" * 68)
    say("装好了。")
    say("=" * 68)
    byo = byo or {}
    if report.get("device"):
        say(f"  显卡      {report['device']['name']} ({report['device']['vram_gib']:.1f} GiB)")
    caps = []
    for cap, label in (("image", "生图"), ("audio", "音频")):
        if byo.get(cap):
            caps.append(f"{label}(BYO {byo[cap]})")
        elif cap in profiles:
            caps.append(f"{label}(本机)")
    say(f"  能力      {' + '.join(caps) if caps else '(无)'}")
    say(f"  抠图默认  {report['cutout_quality']}")
    say(f"  资产目录  {state_dir}   ← 参考音和定妆图在这里, 不可复现, 记得备份")
    say("")
    say("把它接到 DeepSeek Harness —— 往你 profile 的 cordis.patch.yml 里加一行:")
    say("")
    say("  - insert:")
    say("      - id: continuity")
    say("        name: '@deepseek-ai/dsh-mcp-client'")
    say("        config:")
    say("          serverName: continuity")
    say("          transport: stdio")
    say("          command: uvx")
    say("          args: ['--from', 'dsh-continuity', 'continuity-mcp']")
    say("          env:  # 下面这些")
    say("")
    say("本机对应的环境变量:")
    say("")
    say(f"  CONTINUITY_STATE_DIR={state_dir} \\")
    # 名字必须和 cordis.patch.yml 里读的一致。patch 读 CONTINUITY_SD_SERVER,
    # 这里原先印的是 SD_SERVER —— 照着 export 的人会让 patch 拿到空串, 回落到
    # 127.0.0.1 默认值, 于是 BYO 明明配好了却报"连不上一个我没配过的地址"。
    say(f"  CONTINUITY_SD_SERVER={env['SD_SERVER']} \\")
    say(f"  CONTINUITY_AUDIO_SERVER={env['AUDIO_SERVER']} \\")
    # 按"有没有后端"判, 不是按"本机装没装" —— BYO 的那半没在 profiles 里, 但它是启用的。
    # 早先这里看 profiles, 于是 BYO 生图时同一段输出上面写"生图(BYO)"、下面写
    # ENABLE_IMAGE=0, 照着抄就把刚接好的后端关掉了。
    if not report.get("enable_image", True):
        say("  CONTINUITY_ENABLE_IMAGE=0 \\")
    say(f"  CONTINUITY_CUTOUT_QUALITY={report['cutout_quality']} \\")

    say("  uvx --from dsh-continuity continuity-mcp")
    say("")
    say("引擎的开关:")
    say(f"  docker compose -p continuity {' '.join(f'--profile {p}' for p in profiles)} stop|start")
    if mode == "nvidia":
        say("")
        say("  ⚠️ 你这台是 N 卡, 走的是 nvidia-container-toolkit 注入驱动那条路径。")
        say("     它按 NVIDIA 官方文档实现, 但作者手头只有 A 卡, 没有实测过。")
        say("     跑通或跑不通都欢迎去 issue 里说一声。")
    say("")
    say("显存: 空闲时应该回到接近 0 —— 生图权重不常驻, 音频模型 120s 无调用后卸载。")


def main():
    ap = argparse.ArgumentParser(
        prog="continuity-setup",
        description="装好 Continuity 的本机后端 (两个 ggml/Vulkan 引擎 + 权重)。")
    ap.add_argument("--check", action="store_true",
                    help="只体检, 什么都不改")
    ap.add_argument("--state-dir", default=os.getenv("CONTINUITY_STATE_DIR", "~/.continuity"),
                    help="资产 + 配置目录 (默认 ~/.continuity)")
    ap.add_argument("--models-dir", default=os.getenv("CONTINUITY_MODELS_DIR"),
                    help="权重目录 (默认 <state-dir>/models; 权重大, 可以放别的盘)")
    ap.add_argument("--no-image", action="store_true",
                    help="不装生图那半, 并且也不用别处的 (工具不注册)")
    ap.add_argument("--sd-server", metavar="URL",
                    help="生图后端你自己提供 —— 本机不装这一半 (权重不下, 引擎不起), 工具照常可用")
    ap.add_argument("--audio-server", metavar="URL",
                    help="音频后端你自己提供 —— 本机不装这一半, 工具照常可用")
    ap.add_argument("--skip-build", action="store_true", help="已有引擎镜像就不重编")
    ap.add_argument("--no-cache", action="store_true", help="强制重编引擎镜像")
    ap.add_argument("--sd-port", type=int, default=9020)
    ap.add_argument("--audio-port", type=int, default=9021)
    ap.add_argument("-y", "--yes", action="store_true", help="不问, 直接装")
    args = ap.parse_args()
    if args.check:
        cmd_check(args)
    else:
        cmd_install(args)


if __name__ == "__main__":
    main()
