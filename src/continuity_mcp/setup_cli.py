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


def download_models(models_dir, groups):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("缺 huggingface_hub。装法: pip install huggingface_hub\n"
            f"  (或者用 uvx 跑本工具: uvx --from git+https://github.com/linxuhao/Deepseek-Continuity continuity-setup)")
    manifest = json.loads((DEPLOY / "models.json").read_text(encoding="utf-8"))
    todo = [m for m in manifest["models"] if m["group"] in groups]
    total = sum(m["size_bytes"] for m in todo) / preflight.GIB
    say(f"    需要 {len(todo)} 个文件, 共 {total:.1f} GiB。已经下过的会跳过。")
    for i, m in enumerate(todo, 1):
        dest = models_dir / m["dest"]
        if dest.exists() and dest.stat().st_size == m["size_bytes"]:
            say(f"    [{i}/{len(todo)}] 已有 {m['dest']}")
            continue
        say(f"    [{i}/{len(todo)}] {m['dest']}  ({m['size_bytes'] / preflight.GIB:.2f} GiB)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            got = hf_hub_download(repo_id=m["repo_id"], filename=m["filename"])
        except Exception as e:
            die(f"下载 {m['repo_id']}/{m['filename']} 失败: {e}\n"
                f"  网络不通或仓库改名了。可以手动放到 {dest} 再重跑本命令。")
        # 复制而不是软链: 软链指向 HF 缓存, 缓存被清理时引擎会突然读不到权重,
        # 而那时的报错是"文件不存在", 与网络问题长得一模一样。
        shutil.copyfile(got, dest)
    return total


def write_config(state_dir, models_dir, report, args):
    for sub in ("actors", "subjects", "generated"):
        (state_dir / sub).mkdir(parents=True, exist_ok=True)
    dev_index = report["device"]["index"]
    (state_dir / "audio_server.json").write_text(
        (DEPLOY / "audio_server.json.tmpl").read_text(encoding="utf-8")
        .replace("__DEVICE__", str(dev_index)), encoding="utf-8")

    video_gid, render_gid = preflight.dri_gids()
    env = {
        "CONTINUITY_STATE_DIR": str(state_dir),
        "CONTINUITY_MODELS_DIR": str(models_dir),
        "VULKAN_DEVICE": str(dev_index),
        "VIDEO_GID": str(video_gid),
        "RENDER_GID": str(render_gid),
        "SD_PORT": str(args.sd_port),
        "AUDIO_PORT": str(args.audio_port),
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
    engines.SD_SERVER = f"http://127.0.0.1:{env['SD_PORT']}"
    engines.AUDIO_SERVER = f"http://127.0.0.1:{env['AUDIO_PORT']}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, down = engines.health()
        # 只装了音频时 sd-server 本来就不该在
        down = [d for d in down if not (d.startswith("sd-server") and "image" not in profiles)]
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
    r = subprocess.run(["docker", "image", "inspect", ENGINES_IMAGE],
                       capture_output=True)
    return r.returncode == 0


def cmd_install(args):
    total = 5
    state_dir = Path(args.state_dir).expanduser()
    models_dir = Path(args.models_dir).expanduser() if args.models_dir else state_dir / "models"

    step(1, total, "体检")
    try:
        preflight.check_docker()
    except preflight.PreflightError as e:
        die(str(e))
    ram = preflight.total_ram_gib()
    free = preflight.free_disk_gib(state_dir)
    say(f"    内存 {ram:.1f} GiB, {state_dir} 所在分区可用 {free:.1f} GiB")
    if free < preflight.DISK_AUDIO_ONLY:
        die(f"磁盘只剩 {free:.1f} GiB, 连最小安装 ({preflight.DISK_AUDIO_ONLY:.0f} GiB) 都不够。\n"
            f"  换个盘: continuity-setup --state-dir /别的/路径")

    step(2, total, "构建引擎镜像")
    if args.skip_build and _image_exists():
        say("    已有镜像, 跳过 (--skip-build)")
    else:
        build_engines(args.no_cache)

    step(3, total, "探测显卡")
    try:
        report = preflight.run(state_dir, image=ENGINES_IMAGE,
                               want_image=not args.no_image, want_audio=True)
    except preflight.PreflightError as e:
        die(str(e))
    say(preflight.format_report(report))
    if report.get("image_warning") and not args.yes and not args.no_image:
        # 不替用户决定: 他知道一些我不知道的事 (要不要换显卡, 是不是只要配音)
        say("")
        say("  ⚠️ 这张卡装不了生图那半。")
        if input("     只装音频那半 (铸声/配音/音乐/音效/抠图)? [y/N] ").strip().lower() \
                not in ("y", "yes"):
            die("已取消。换一张 8 GiB 以上的卡, 或者加 --no-image 明确只装音频。", 0)
    if not args.yes:
        groups = ["audio"] + (["image"] if report["enable_image"] else [])
        manifest = json.loads((DEPLOY / "models.json").read_text(encoding="utf-8"))
        gib = sum(m["size_bytes"] for m in manifest["models"]
                  if m["group"] in groups) / preflight.GIB
        say(f"\n    接下来要下载 {gib:.1f} GiB 权重到 {models_dir}。")
        if input("    继续? [Y/n] ").strip().lower() in ("n", "no"):
            die("已取消。", 0)

    step(4, total, "下载权重")
    groups = {"audio"} | ({"image"} if report["enable_image"] else set())
    models_dir.mkdir(parents=True, exist_ok=True)
    download_models(models_dir, groups)

    step(5, total, "启动引擎")
    env = write_config(state_dir, models_dir, report, args)
    profiles = sorted(groups)
    mode = report.get("runtime_mode", "dri")
    if compose(state_dir, env, profiles, "up", "-d", mode=mode).returncode != 0:
        die("docker compose up 失败。")
    say("    等引擎读完权重 (冷启动要从磁盘读十几 GB, 可能几十秒)...")
    ok, down = wait_healthy(env, profiles)
    if not ok:
        die("引擎起来了但没通过健康检查:\n  " + "\n  ".join(down) +
            f"\n看日志: docker compose -p continuity logs")

    _print_done(state_dir, report, env, profiles, mode)


def _print_done(state_dir, report, env, profiles, mode="dri"):
    say("\n" + "=" * 68)
    say("装好了。")
    say("=" * 68)
    say(f"  显卡      {report['device']['name']} ({report['device']['vram_gib']:.1f} GiB)")
    say(f"  能力      {'生图 + 音频' if 'image' in profiles else '仅音频 (显存不足以装生图)'}")
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
    say("          args: ['--from', 'git+https://github.com/linxuhao/Deepseek-Continuity',")
    say("                 'continuity-mcp']")
    say("          env:  # 下面这些")
    say("")
    say("本机对应的环境变量:")
    say("")
    say(f"  CONTINUITY_STATE_DIR={state_dir} \\")
    say(f"  SD_SERVER=http://127.0.0.1:{env['SD_PORT']} \\")
    say(f"  AUDIO_SERVER=http://127.0.0.1:{env['AUDIO_PORT']} \\")
    if "image" not in profiles:
        say("  CONTINUITY_ENABLE_IMAGE=0 \\")
    say(f"  CONTINUITY_CUTOUT_QUALITY={report['cutout_quality']} \\")

    say(f"  uvx --from git+https://github.com/linxuhao/Deepseek-Continuity continuity-mcp")
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
                    help="不装生图那半 (只要配音/音乐/音效/抠图)")
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
