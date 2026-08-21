# ==========================================
# 两个 ggml/Vulkan 引擎的客户端 + 显存生命周期。
#
# 显存只有一条规则: 任务是串行的, 同一时刻只需要一个模型, 所以每次开工前把不是它的
# 都还回去 —— release_all_but(需要的那个)。
#
# 这条规则换来的是一个常数: 峰值 = 单个最大模型 = 6.80 GiB, 与调用顺序无关。
# 之前是三套拼起来的规则 (音频之间互卸 / 空闲卸 / 大卡不卸生图前那次), 于是峰值
# 取决于你之前干了什么 —— 实测 "配一句台词再画张图" 这种最普通的顺序会到 10.94 GiB,
# 8 GiB 的卡根本跑不了。而 8 GiB 正是本插件承诺的门槛。
#
# 代价实测: 交替调用时, 紧跟在生图后面的那次配音要重载, +3.4 s (5.0 vs 1.6)。
# 连着配十句不受影响 —— 中间没有别的模型来抢, 就不会被卸。
# 权重是 mmap 的且在 page cache 里, 所以重载只是重新建 session, 不是重读磁盘。
#
# 生图那半不需要这套: sd-server 开着 --offload-to-cpu, 权重根本不常驻显存,
# 出图完 2 s 内自己回到 0.21 GiB。
#
# 空闲计时器仍然保留, 但它管的是另一件事: 最后一个任务之后没有"下一个任务"来触发
# 卸载, 所以由它来收尾。给 120 s 宽限而不是立刻卸, 是因为连着配十句台词的人
# 不该每句都付重载。
# ==========================================
import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import (SD_SERVER, AUDIO_SERVER, AUDIO_MODELS, JOB_TIMEOUT_S, ENGINE_WAIT_S,
                     AUDIO_IDLE_UNLOAD_S, SPEECH_TOKENS_PER_CHAR, SPEECH_MAX_TOKENS,
                     STATE_DIR, DEFAULT_SD_SERVER, DEFAULT_AUDIO_SERVER)

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
        except urllib.error.HTTPError as he:
            # 一定要把引擎自己说的那句读出来。原先这里直接 raise, 于是
            #   {"error":{"message":"could not open WAV input: /actors/xxx.wav"}}
            # 变成了调用方看到的 "HTTP Error 500: Internal Server Error" ——
            # 一句能直接定位问题的话, 被换成了一句什么都没说的话。
            raise EngineError(f"{tag}: {_http_detail(he)}") from he
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            if time.time() >= deadline:
                waited = f" (已重试 {retry_s:.0f}s)" if retry_s else ""
                hint = "" if setup_was_run() else "\n  " + SETUP_HINT
                raise EngineUnreachable(f"{tag} 连不上{waited}: {e}{hint}") from e
            log.warning("[%s] 未就绪, %.1fs 后重试: %s", tag, delay, e)
            time.sleep(delay)
            delay = min(delay * 2, 5.0)


class EngineUnreachable(RuntimeError):
    """引擎连不上。"""


class EngineError(RuntimeError):
    """引擎收到了请求但拒绝/失败了 —— 消息里带着它自己的说法。"""


def _http_detail(he):
    """把引擎返回体里的错误消息取出来。"""
    try:
        body = he.read().decode("utf-8", "replace")
    except Exception:
        return f"HTTP {he.code} {he.reason}"
    try:
        msg = (json.loads(body).get("error") or {}).get("message")
    except Exception:
        msg = None
    return f"HTTP {he.code}: {(msg or body or he.reason)[:400]}"


SETUP_HINT = (
    "看起来后端还没装 —— 这个插件只是前半截, 两个 ggml/Vulkan 引擎要先装起来:\n"
    "    uvx --from dsh-continuity continuity-setup\n"
    "  它会体检本机、编引擎镜像、下权重并把引擎跑起来。装好之后本插件的工具才有东西可用。"
)


def _engines_are_default():
    return (SD_SERVER == DEFAULT_SD_SERVER) and (AUDIO_SERVER == DEFAULT_AUDIO_SERVER)


def setup_was_run():
    """这台机器上把后端准备好了没有。

    用来区分两件长得一样、处理方式完全相反的事:
      准备过但连不上  -> 多半是冷启动在读十几 GB 权重, 值得等满 ENGINE_WAIT_S
      压根没准备过    -> 等三分钟也不会有人来, 立刻失败并告诉他去装
    早先两者是一个分支, 于是"只装了 dsh 插件、没跑 setup"的人每调一次工具要白等 3 分钟,
    而 npm 那条安装路径恰好最容易走成这样。

    "准备过"有两种: 跑过 continuity-setup (state_dir 里留下 compose.env), 或者自己
    把引擎地址指到了别处 (自建/远程后端)。后一种也要算, 否则会对着一个自带后端的人
    喊"去跑 continuity-setup" —— 那对他是错的建议, 而错的建议比没有建议更糟。"""
    return (STATE_DIR / "compose.env").is_file() or not _engines_are_default()


def engine_retry_budget():
    return ENGINE_WAIT_S if setup_was_run() else 0.0


def post(url, payload, tag, timeout=None, retry_s=None):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return _http(req, timeout or JOB_TIMEOUT_S, tag,
                 retry_s=engine_retry_budget() if retry_s is None else retry_s)


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


def release_all_but(keep=None, reason=""):
    """把除 keep 之外的模型全还回显存池。keep=None 表示全卸 (生图前就是这种)。

    卸载失败不让任务失败: 那只是少省一点显存, 不是错误 —— 除非显存真的不够,
    而那种情况下引擎自己会报 OOM, 报得比我们清楚。"""
    if keep:
        _audio_state.update(last_use=time.time(), loaded=True)
    others = sorted(AUDIO_MODELS - ({keep} if keep else set()))
    if not others:
        return True
    try:
        # retry_s=0: 这是 best-effort 的清理, 失败只是少省一点显存。早先它跟着默认值
        # 重试 180s, 于是引擎不在时, 每个任务在真正开始之前先白等三分钟。
        post(f"{AUDIO_SERVER}/v1/tasks/unload_models", {"model_ids": others},
             "audiocpp-server", timeout=15, retry_s=0)
        if keep is None:
            _audio_state["loaded"] = False
            log.info("音频模型已全部卸载%s, 显卡回到零常驻", f" ({reason})" if reason else "")
        return True
    except Exception as e:
        # 只取第一行: 引擎不在时异常里带着"去跑 continuity-setup"的整段指引, 而这里
        # 只是一次可有可无的清理 —— 该说那段话的是真正失败的那次调用, 不是它。
        log.warning("卸载 %s 失败, 继续: %s", others, str(e).splitlines()[0])
        return False


def unload_all_audio(reason="explicit"):
    return release_all_but(None, reason)


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
    release_all_but(model_id)
    req["max_tokens"] = max(64, min(SPEECH_MAX_TOKENS,
                                    int(len(req["text"]) * SPEECH_TOKENS_PER_CHAR)))
    res = post(f"{AUDIO_SERVER}/v1/tasks/run", {"model": model_id, "request": req}, tag)
    b64 = res.get("audio")
    if not b64:
        raise RuntimeError(f"{model_id} returned no audio: {str(res)[:300]}")
    return base64.b64decode(b64), res.get("timing") or {}


def audio_engine_is_remote():
    """音频引擎是不是在别的机器上。

    用来判断"参考音这条路径引擎大概率看不见" —— 它是引擎自己去 open 的, 跨机就一定
    找不到, 除非两边挂了同一个目录。"""
    host = urllib.parse.urlparse(AUDIO_SERVER).hostname or ""
    return host not in ("127.0.0.1", "localhost", "::1", "")


SHARED_DIR_HINT = (
    "克隆用的参考音是音频引擎自己去打开的, 不是本进程读了发过去的。"
    "引擎在另一台机器上时, 它看不见本机的 actors 目录 —— 铸声会成功 (那一步不读文件), "
    "而之后每一句台词都会失败。\n"
    "  要么把引擎跑在本机 (uvx --from dsh-continuity continuity-setup), "
    "要么让引擎挂载同一个 actors 目录, 并用 CONTINUITY_ENGINE_ACTORS_DIR 告诉本插件"
    "它在引擎那边的路径。"


)


def health():
    """两个引擎在不在。返回 (ok, [坏掉的名字])。

    只返回名字, 不返回长报错: 每条都带一遍"去跑 continuity-setup"的话, 两个引擎
    就是两遍, 而 continuity_status 自己还会再说一遍。提示只该出现一次。"""
    down = []
    for name, url in (("sd-server", f"{SD_SERVER}/sdcpp/v1/capabilities"),
                      ("audiocpp-server", f"{AUDIO_SERVER}/health")):
        try:
            get(url, timeout=5, tag=name)
        except Exception:
            down.append(name)
    return (not down), down
