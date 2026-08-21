# ==========================================
# 两个 ggml/Vulkan 引擎的客户端 + 显存生命周期。
#
# 显存策略是这个插件"不用的时候跟没装过一样"的全部实现:
#   生图  sd-server 开 --offload-to-cpu, 权重根本不常驻显存, 出图完 2s 内回到 0.21 GB
#   音频  引擎会把用过的模型留在显存里, 所以两条规则:
#         1. 同一时刻只留一个 (三个全常驻实测峰值 15.70/16.00 GB, 再叠一张生图就 device lost)
#         2. 空闲 AUDIO_IDLE_UNLOAD_S 之后全卸掉
# 重载实测 4.3 s 且权重在 page cache 里, 这个代价只在切换模型时付。
# ==========================================
import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request

from .config import (SD_SERVER, AUDIO_SERVER, AUDIO_MODELS, JOB_TIMEOUT_S, ENGINE_WAIT_S,
                     AUDIO_IDLE_UNLOAD_S, SPEECH_TOKENS_PER_CHAR, SPEECH_MAX_TOKENS)

log = logging.getLogger("continuity")


def _http(req, timeout, tag, retry_s=0.0):
    """引擎冷启动时会拒连: sd-server 要读十几 GB 权重, 宿主机重启后 page cache 是冷的,
    可能要几十秒才 listen。对连接错误重试, 对 HTTP 错误立即失败 (那是真错)。"""
    deadline = time.time() + retry_s
    delay = 0.5
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            if time.time() >= deadline:
                raise EngineUnreachable(f"{tag} 不可达 (已重试 {retry_s:.0f}s): {e}") from e
            log.warning("[%s] 未就绪, %.1fs 后重试: %s", tag, delay, e)
            time.sleep(delay)
            delay = min(delay * 2, 5.0)


class EngineUnreachable(RuntimeError):
    """引擎没起来 —— 多半是 docker compose 没 up, 或者还在读权重。"""


def post(url, payload, tag, timeout=None, retry_s=None):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return _http(req, timeout or JOB_TIMEOUT_S, tag,
                 retry_s=ENGINE_WAIT_S if retry_s is None else retry_s)


def get(url, timeout=30, tag="engine", retry_s=0.0):
    return _http(urllib.request.Request(url), timeout, tag, retry_s=retry_s)


# ---- 生图 ----

def sd_generate(prompt, width, height, steps=4, cfg_scale=1.0, seed=None, ref_b64=None):
    """向 sd-server 提交一张图并等它出来, 返回 (图片字节, 后缀)。"""
    payload = {"prompt": prompt, "width": width, "height": height,
               "steps": steps or 4, "cfg_scale": cfg_scale or 1.0}
    if seed is not None:
        payload["seed"] = seed
    if ref_b64:
        payload["ref_images"] = [ref_b64]             # base64 参考图 -> 图生图
    sub = post(f"{SD_SERVER}/sdcpp/v1/img_gen", payload, "sd-server", timeout=60)
    poll = f"{SD_SERVER}{sub['poll_url']}"
    deadline = time.time() + JOB_TIMEOUT_S
    while True:
        st = get(poll)
        if st.get("error"):
            raise RuntimeError(f"sd-server: {st['error']}")
        if st.get("result"):
            break
        if time.time() > deadline:
            raise TimeoutError(f"sd-server job {sub.get('id')} exceeded {JOB_TIMEOUT_S}s")
        time.sleep(0.5)
    imgs = st["result"].get("images") or []
    if not imgs:
        raise RuntimeError("sd-server returned no images")
    return (base64.b64decode(imgs[0]["b64_json"]),
            st["result"].get("output_format") or "png")


# ---- 音频显存生命周期 ----
_audio_state = {"last_use": 0.0, "loaded": False}
_busy = threading.Event()
_idle_thread = None


def mark_busy():
    """任务开始 —— 空闲卸载绝不能打断正在跑的任务。

    顺手保证空闲看守已经起来: 早先它只在 server.main() 里启动, 于是任何不走 main()
    的入口 (测试、把本模块当库用) 都会永远不卸载, 而那种"忘了启动"从外部看
    与"卸载失效"完全一样。"""
    start_idle_watch()
    _busy.set()


def mark_idle():
    """任务结束 —— 空闲计时从这里算起, 不是从任务开始。"""
    _audio_state["last_use"] = time.time()
    _busy.clear()


def unload_all_audio(reason="explicit"):
    try:
        post(f"{AUDIO_SERVER}/v1/tasks/unload_models", {"model_ids": sorted(AUDIO_MODELS)},
             "audiocpp-server", timeout=120, retry_s=0)
        _audio_state["loaded"] = False
        log.info("音频模型已全部卸载 (%s), 显卡回到零常驻", reason)
        return True
    except Exception:
        log.warning("卸载失败, 下次再试", exc_info=True)
        return False


def use_audio_model(keep):
    """同一时刻只让一个音频模型占着显存。

    三个音频模型 (音乐 / VoiceDesign / Base 克隆) 全常驻时实测峰值 15.70/16.00 GB,
    再叠一张 1024 生图就把 GPU 挤挂了。而它们的显存是"用过就留着"的: 三个都推理过
    是 11.09 GB, 卸掉两个降到 7.90, 全卸掉降到 3.62。
    卸载失败不让任务失败: 那只是少省一点显存, 不是错误。"""
    _audio_state.update(last_use=time.time(), loaded=True)
    others = sorted(AUDIO_MODELS - {keep})
    if not others:
        return
    try:
        post(f"{AUDIO_SERVER}/v1/tasks/unload_models", {"model_ids": others},
             "audiocpp-server", timeout=120)
    except Exception:
        log.warning("卸载 %s 失败, 继续", others, exc_info=True)


def _idle_loop():
    """空闲够久就把显存还给用户 —— 他要拿这张卡打游戏。"""
    while True:
        time.sleep(5)
        if AUDIO_IDLE_UNLOAD_S <= 0 or not _audio_state["loaded"] or _busy.is_set():
            continue
        if time.time() - _audio_state["last_use"] >= AUDIO_IDLE_UNLOAD_S:
            unload_all_audio(f"空闲 {AUDIO_IDLE_UNLOAD_S:.0f}s")


def start_idle_watch():
    global _idle_thread
    if _idle_thread is None and AUDIO_IDLE_UNLOAD_S > 0:
        _idle_thread = threading.Thread(target=_idle_loop, daemon=True)
        _idle_thread.start()


def shutdown():
    """MCP server 退出时把显存还回去。

    dsh 通过 stdio 拉起本进程, 也会直接把它杀掉 —— 那一刻 audiocpp 容器还活着,
    显存还占着, 而再没有人会来卸它了。没有这一步, "关掉 agent" 不等于 "还回显存"。"""
    if _audio_state["loaded"]:
        unload_all_audio("shutdown")


# ---- 语音 ----

def tts(model_id, req, tag="audiocpp-server"):
    use_audio_model(model_id)
    req["max_tokens"] = max(64, min(SPEECH_MAX_TOKENS,
                                    int(len(req["text"]) * SPEECH_TOKENS_PER_CHAR)))
    res = post(f"{AUDIO_SERVER}/v1/tasks/run", {"model": model_id, "request": req}, tag)
    b64 = res.get("audio")
    if not b64:
        raise RuntimeError(f"{model_id} returned no audio: {str(res)[:300]}")
    return base64.b64decode(b64), res.get("timing") or {}


def health():
    """两个引擎在不在。返回 (ok, [坏掉的描述])。"""
    down = []
    for name, url in (("sd-server", f"{SD_SERVER}/sdcpp/v1/capabilities"),
                      ("audiocpp-server", f"{AUDIO_SERVER}/health")):
        try:
            get(url, timeout=5, tag=name)
        except Exception as e:
            down.append(f"{name}: {e}")
    return (not down), down
