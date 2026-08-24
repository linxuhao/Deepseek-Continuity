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
import uuid

from .config import (SD_SERVER, AUDIO_SERVER, AUDIO_MODELS, JOB_TIMEOUT_S, ENGINE_WAIT_S,
                     AUDIO_IDLE_UNLOAD_S, SPEECH_TOKENS_PER_CHAR, SPEECH_MAX_TOKENS,
                     STATE_DIR, DEFAULT_SD_SERVER, DEFAULT_AUDIO_SERVER,
                     ASR_SERVER, ASR_API_KEY, ASR_IS_REMOTE,
                     IMAGE_API_SERVER, IMAGE_API_KEY, IMAGE_API_MODEL, IMAGE_API_SIZES,
                     IMAGE_VIA_API)

log = logging.getLogger("continuity")


def _http(req, timeout, tag, retry_s=0.0, retry_timeouts=True):
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
            # socket 超时也是 OSError。对"提交一个任务"这种非幂等请求重发, 会让引擎
            # 排上三四份同样的活, 而我们只轮询最后一个 poll_url —— 用户等好几轮,
            # 更糟的是引擎若并行跑, "峰值=单个模型"这条 8 GiB 设计的地基就没了。
            if not retry_timeouts and isinstance(e, TimeoutError):
                raise EngineUnreachable(f"{tag} 超时 ({timeout:.0f}s): {e}") from e
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
    # error 可能是字符串也可能是对象: sd_server 回 {"error": "invalid generation parameters"},
    # 而对 str 调 .get 会抛 AttributeError —— 以前那版就是这么被 except 吞掉的, 于是
    # "取出错误消息"这件事对这个引擎从来没成功过, 却一声不响地吐回整个 JSON。
    msg = None
    try:
        d = json.loads(body)
        for k in ("error", "message", "detail"):
            v = d.get(k)
            if isinstance(v, dict):
                v = v.get("message") or v.get("detail")
            if isinstance(v, str) and v.strip():
                msg = v.strip()
                break
    except Exception:
        pass
    # 无冒号 —— 和 media-gen 的 "HTTP 400 <msg>" 逐字一致。外壳两边都要解析这一行,
    # 格式漂开一个冒号就够让它判错。
    return f"HTTP {he.code} {(msg or body or he.reason)[:400]}"


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


def post(url, payload, tag, timeout=None, retry_s=None, retry_timeouts=True, api_key=None):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    return _http(req, timeout or JOB_TIMEOUT_S, tag,
                 retry_s=engine_retry_budget() if retry_s is None else retry_s,
                 retry_timeouts=retry_timeouts)


def post_multipart(url, fields, file_part, tag, timeout=None, retry_s=None,
                   retry_timeouts=True, api_key=None):
    """multipart/form-data 版的 post。

    只有一个端点用它 (/v1/audio/transcriptions), 但那个端点没有别的选择, 见 transcribe()。
    自己拼而不引 requests: 全包到这里为止没有任何第三方 HTTP 依赖, 为一个 form 破例
    不值当 —— 这段拼接是死的, 二十行, 没有需要维护的分支。
    """
    name, filename, data, ctype = file_part
    b = uuid.uuid4().hex
    buf = bytearray()
    for k, v in fields.items():
        buf += (f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n').encode()
    buf += (f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f'Content-Type: {ctype}\r\n\r\n').encode()
    buf += data + b"\r\n" + f"--{b}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={b}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=bytes(buf), headers=headers)
    return _http(req, timeout or JOB_TIMEOUT_S, tag,
                 retry_s=engine_retry_budget() if retry_s is None else retry_s,
                 retry_timeouts=retry_timeouts)


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
    # 提交是非幂等的: 连不上可以重试 (还没到引擎), 但超时不行 (可能已经排上了)。
    sub = post(f"{SD_SERVER}/sdcpp/v1/img_gen", payload, "sd_server", timeout=60,
               retry_timeouts=False)
    poll = f"{SD_SERVER}{sub['poll_url']}"
    deadline = time.time() + JOB_TIMEOUT_S
    while True:
        st = get(poll)
        if st.get("error"):
            raise RuntimeError(f"sd_server: {st['error']}")
        if st.get("result"):
            break
        if time.time() > deadline:
            raise TimeoutError(f"sd_server job {sub.get('id')} exceeded {JOB_TIMEOUT_S}s")
        time.sleep(0.5)
    imgs = st["result"].get("images") or []
    if not imgs:
        raise RuntimeError("sd_server returned no images")
    return (base64.b64decode(imgs[0]["b64_json"]),
            st["result"].get("output_format") or "png")


def _get_bytes(url, timeout=120):
    """把一个 URL 的内容取回来 (标准 API 可能回 url 而不是 b64)。"""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _pick_size(width, height):
    """从后端允许的那份枚举里挑长宽比最接近的。返回 (size 字符串, 说明或 None)。"""
    want = width / height
    best = min(IMAGE_API_SIZES,
               key=lambda z: abs(int(z.split("x")[0]) / int(z.split("x")[1]) - want))
    if best == f"{width}x{height}":
        return best, None
    # 措辞不能写成"落盘前会缩回你要的尺寸": 那只对 generate_image 成立 (它有 _fit_size),
    # 定妆图存的就是这里挑出来的尺寸。第一版写成了前者, 而定妆那条路上它是句假话。
    return best, (f"标准生图 API 的尺寸是固定枚举, {width}x{height} 被换成长宽比最接近的 "
                  f"{best} —— 长宽比对不上时会有形变。generate_image 落盘前会缩回你要的"
                  f"尺寸, 定妆图存的就是 {best}。后端支持别的尺寸就改 IMAGE_API_SIZES。")


def image_api(prompt, width, height, seed=None, ref_b64=None):
    """别人家的 /v1/images/generations —— OpenAI 那套形状。返回 (bytes, ext, notes)。

    ⚠️ 这条路是对着标准的**形状**写的, 在一个本地 mock 上打通过, 但没有对任何一家真实
    服务实测过。各家在参考图上的语义差得最远 (见下), 换后端时先拿 generate_image
    试一张再说。

    notes 里的每一条都是"你要的东西这里给不了"。它们会一路走到工具返回的 warnings ——
    这条路能用, 但不能假装它和本机引擎是同一个东西:
      - seed 没有对应参数, 所以"同 seed 可复现"在这条路上不成立;
      - steps / cfg_scale 无处可放, 丢掉;
      - 尺寸被吸附到后端的枚举上。
    """
    notes = []
    size, note = _pick_size(width, height)
    if note:
        notes.append(note)
    if seed is not None:
        notes.append("标准生图 API 没有 seed 参数, 这次的 seed 被忽略了 —— "
                     "同 seed 复现同一张图在这条路上不成立。要可复现就用本机引擎。")
    if ref_b64:
        # 参考图走 edits: 各家对它的理解差得最远 (蒙版编辑 / 多图参考 / 风格迁移),
        # 所以定妆一致性在这条路上是后端的性质, 不是我们的保证。
        notes.append("参考图走的是标准 API 的 edits 端点, 各家语义不同 —— "
                     "定妆的一致性在这条路上由那个后端决定, 不由这里保证。")
        res = post_multipart(f"{IMAGE_API_SERVER}/v1/images/edits",
                             {"model": IMAGE_API_MODEL, "prompt": prompt, "size": size, "n": "1"},
                             ("image", "reference.png", base64.b64decode(ref_b64), "image/png"),
                             "image_api", api_key=IMAGE_API_KEY or None)
    else:
        res = post(f"{IMAGE_API_SERVER}/v1/images/generations",
                   {"model": IMAGE_API_MODEL, "prompt": prompt, "size": size, "n": 1},
                   "image_api", api_key=IMAGE_API_KEY or None)
    items = res.get("data") or []
    if not items:
        raise RuntimeError(f"image_api 没返回图片: {str(res)[:300]}")
    first = items[0]
    if first.get("b64_json"):
        data = base64.b64decode(first["b64_json"])
    elif first.get("url"):
        # 不强制 response_format=b64_json: 有的实现认这个参数, 有的认了会报未知参数,
        # 而两种回法我们都收得下, 那就没必要为它挑一边。
        data = _get_bytes(first["url"])
    else:
        raise RuntimeError(f"image_api 的返回里既没有 b64_json 也没有 url: {str(first)[:200]}")
    ext = "png" if data[:4] == b"\x89PNG" else ("jpg" if data[:2] == b"\xff\xd8" else "png")
    return data, ext, notes


def draw(prompt, width, height, steps=4, cfg_scale=1.0, seed=None, ref_b64=None):
    """出一张图 —— 自己的引擎还是别人家的 API, 由 IMAGE_API_SERVER 决定。

    返回 (bytes, ext, notes)。notes 只有 API 那条路才会非空。
    """
    if IMAGE_VIA_API:
        return image_api(prompt, width, height, seed, ref_b64)
    data, ext = sd_generate(prompt, width, height, steps, cfg_scale, seed, ref_b64)
    return data, ext, []


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


# 退出时卸载的超时。dsh 的收尾是三段: 先关我们的 stdin 等 2 秒 (正常退出, atexit 会跑),
# 不退就 SIGTERM 再等 2 秒 (Python 默认不跑 atexit), 还不退就 SIGKILL。
# 也就是说"退出时把显存还回去"总共只有 2 秒。默认的 120 秒超时在引擎稍慢时就会冲过去,
# 被 SIGTERM 掉 —— 显存留着不还, 而日志上什么都看不出来。
SHUTDOWN_UNLOAD_TIMEOUT_S = 1.5


def release_all_but(keep=None, reason="", timeout=120):
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
             "audiocpp_server", timeout=timeout, retry_s=0)
        if keep is None:
            _audio_state["loaded"] = False
            log.info("音频模型已全部卸载%s, 显卡回到零常驻", f" ({reason})" if reason else "")
        return True
    except Exception as e:
        # 只取第一行: 引擎不在时异常里带着"去跑 continuity-setup"的整段指引, 而这里
        # 只是一次可有可无的清理 —— 该说那段话的是真正失败的那次调用, 不是它。
        log.warning("卸载 %s 失败, 继续: %s", others, str(e).splitlines()[0])
        return False


def unload_all_audio(reason="explicit", timeout=15):
    return release_all_but(None, reason, timeout=timeout)


def _idle_loop():
    """空闲够久就把显存还给用户 —— 他要拿这张卡打游戏。

    和 shutdown() 一样, 不看 _audio_state["loaded"]: 那是本进程的记账, 而显存是
    引擎持有的。上一代进程被 SIGKILL 或超时后, 模型还在引擎里占着, 新一代记账为空,
    于是"120s 后自动释放"这句话对重连之后的所有进程都是空头支票 —— 而
    continuity_status 还在照常打印它。改成: 起来先卸一次 (幂等), 之后按空闲时间卸。
    卸载失败只 warn, 不影响任何任务。

    每个空闲段只卸一次 (unloaded_for 记住是为哪一次 last_use 卸的)。原先只看
    "距 last_use 是不是超了 120s" —— 而卸载并不更新 last_use, 所以条件一旦成立就
    再也不会不成立: 之后每 5 秒往引擎发一次 unload_models, 一天一万七千次。
    stdio 下看不出来, 进程跟着 dsh 会话一起没了; 而 --http 是常驻的, 它会一直发。
    """
    first = True
    unloaded_for = None                     # 已经为哪一次 last_use 卸过了
    while True:
        time.sleep(5 if not first else 1)
        if AUDIO_IDLE_UNLOAD_S <= 0 or _busy.is_set():
            continue
        if first:
            # 接管上一代可能留下的显存。本进程还没跑过任何任务, 卸掉不会打断谁。
            first = False
            release_all_but(None, "接管上一代", timeout=5)
            unloaded_for = _audio_state["last_use"]
            continue
        last_use = _audio_state["last_use"]
        if last_use != unloaded_for and time.time() - last_use >= AUDIO_IDLE_UNLOAD_S:
            unload_all_audio(f"空闲 {AUDIO_IDLE_UNLOAD_S:.0f}s")
            unloaded_for = last_use


def start_idle_watch():
    global _idle_thread
    if _idle_thread is None and AUDIO_IDLE_UNLOAD_S > 0:
        _idle_thread = threading.Thread(target=_idle_loop, daemon=True)
        _idle_thread.start()


def shutdown():
    """MCP server 退出时把显存还回去。

    dsh 通过 stdio 拉起本进程, 退出时先关我们的 stdin —— 那一刻 audiocpp 容器还活着,
    显存还占着, 而再没有人会来卸它了。没有这一步, "关掉 agent" 不等于 "还回显存"。
    但这一步只有 2 秒 (见 SHUTDOWN_UNLOAD_TIMEOUT_S), 所以宁可放弃也不能拖着不退:
    拖过去会被 SIGTERM, 那时 atexit 已经不会跑了, 反而一点都卸不掉。"""
    # 不看 _audio_state["loaded"]: 那是本进程的记账, 而显存是引擎持有的, 跨进程共享。
    # dsh 断线重连会换一个新进程, 它记账为空 —— 于是"退出时还显存"对重连之后的那一代
    # 永远不会触发, 而引擎那边模型还占着。卸载本身是幂等的, 没有就是个空操作, 直接发。
    if True:
        if not unload_all_audio("shutdown", timeout=SHUTDOWN_UNLOAD_TIMEOUT_S):
            log.warning("退出时没来得及卸载显存 —— 引擎仍占着音频模型, "
                        "它会在 %.0fs 空闲后自己卸掉", AUDIO_IDLE_UNLOAD_S)


# ---- 语音 ----

def tts(model_id, text, voice_ref_b64=None, reference_text=None, instructions=None,
        seed=None, speaking_rate=None, tag="audiocpp_server"):
    """合成一句语音。

    走 /v1/audio/speech 而不是 /v1/tasks/run, 因为只有前者能收内联的参考音 ——
    后者的 voice_ref 只认路径, 而路径只有引擎自己解析得了, 于是引擎一旦不在本机,
    克隆就整个不成立。这曾被我当成引擎的限制写进文档, 其实是选错了端点。
    (该端点需要 model_specs/<family>.json, 见 deploy/Dockerfile。)
    """
    release_all_but(model_id)
    body = {
        "model": model_id,
        "input": text,
        "response_format": "b64_json",
        # 字数上限只挡"输入长", 挡不住"输出跑飞": 一句短台词退化成循环照样能生成几分钟
        # 并拖挂 GPU。按字数推 token 预算, 让跑飞的请求早早自己停下。
        "max_tokens": max(64, min(SPEECH_MAX_TOKENS,
                                  int(len(text) * SPEECH_TOKENS_PER_CHAR))),
    }
    if voice_ref_b64 is not None:
        body["voice_ref"] = {"type": "base64", "data": voice_ref_b64}
    if reference_text:
        body["reference_text"] = reference_text
    if instructions:
        body["instructions"] = instructions
    if seed is not None:
        body["seed"] = seed
    if speaking_rate:
        body["options"] = {"speaking_rate": speaking_rate}
    res = post(f"{AUDIO_SERVER}/v1/audio/speech", body, tag)
    b64 = res.get("audio")
    if not b64:
        raise RuntimeError(f"{model_id} returned no audio: {str(res)[:300]}")
    return base64.b64decode(b64), res.get("timing") or {}


# ---- 听写 ----
#
# 两条路, 故意分开写:
#
#   transcribe()      我们自己那个 audiocpp_server。它的模型是我们选的 (Qwen3-ASR,
#                     16 kHz), 显存也归我们管, 所以该卸的卸、该降采样的降采样。
#   transcribe_api()  别人家的 OpenAI 形状端点。我们对它一无所知 —— 什么模型、
#                     内部多少赫兹、要不要鉴权, 全是它自己的事。所以这一路只做三件
#                     必须做的: 按标准形状发、带上 key、把 {"text"} 取出来。
#
# 合成一个函数会逼出一堆 "if 是不是自己的引擎" 的分支, 而这两件事本来就不共享假设。


def _text_of(res, who):
    text = (res.get("text") or "").strip()
    if not text:
        # 空转写不当成"成功但没内容": 调用它的两处 (transcribe 工具、import_actor 自动
        # 听写) 拿到空字符串都没法往下走, 而静默的空串会一路变成一个 transcript 为空
        # 的角色 —— 那正是克隆对不齐的起点。
        raise RuntimeError(f"{who} 听不出内容 (是不是静音或纯噪声?): {str(res)[:200]}")
    return text, res.get("timing") or {}


def transcribe(model_id, audio_bytes, filename="audio.wav", language=None,
               tag="audiocpp_server"):
    """自己那个引擎。音频已经是 16 kHz 单声道 —— 它的模型就在那个采样率上算。

    走 multipart 而不是 JSON, 和 tts() 走 /v1/audio/speech 是同一件事: 这个端点的
    JSON 形式三个字段 (audio / audio_path / file) 收的都是**引擎本机的路径**, 而引擎
    不一定和我们在同一台机器上。只有 multipart 是把音频本身发过去的, 引擎一行文件
    系统都不用碰 —— 跨机因此成立。
    (引擎目前只收 WAV: "only WAV audio uploads are currently supported for transcription"。)

    实测 (Qwen3-ASR 1.7B Q8_0, RX 7800 XT, 9.5 s 参考音): 冷启 3.9 s (其中加载约 2.4 s),
    热调 0.4 s, RTF 0.042。
    """
    release_all_but(model_id)
    fields = {"model": model_id}
    if language:
        fields["language"] = language
    res = post_multipart(f"{AUDIO_SERVER}/v1/audio/transcriptions", fields,
                         ("file", filename, audio_bytes, "audio/wav"), tag)
    return _text_of(res, model_id)


def transcribe_api(model_id, audio_bytes, filename="audio.wav", language=None):
    """别人家的 /v1/audio/transcriptions —— OpenAI 那套形状。

    这里不碰本机显存: 那边的模型不在我们的卡上, 卸本机的模型不省任何东西, 只让下一次
    配音白付一次重载 (和 engines_share_a_gpu() 对生图的判断同一个理由)。

    音频原样发。该不该降采样是那个后端的事 —— 我们连它跑的是什么模型都不知道,
    而替一个吃 48 kHz 的服务先降到 16 kHz 是在丢它要用的信息。挑食的后端由调用方
    按 400 重试一次 (见 jobs._hear)。
    """
    fields = {"model": model_id}
    if language:
        fields["language"] = language
    res = post_multipart(f"{ASR_SERVER}/v1/audio/transcriptions", fields,
                         ("file", filename, audio_bytes, "audio/wav"), "asr_server",
                         api_key=ASR_API_KEY or None)
    return _text_of(res, model_id)


def engines_share_a_gpu():
    """两个引擎是不是在同一台机器上。

    "开工前把不是这件活要用的模型卸掉" 的前提是它们抢同一张卡。生图后端被指到别的
    机器时这个前提就没了 —— 那时卸掉本机的音频模型不省任何东西, 只是让下一次配音
    白付一次重载。实测混合部署 (音频本机核显 / 生图远程独显) 时, 每张图前面都白卸一次。
    同机就当成共用一张卡: 自带的 compose 里两个引擎本来就指向同一个 VULKAN_DEVICE。
    """
    return (urllib.parse.urlparse(SD_SERVER).hostname
            == urllib.parse.urlparse(AUDIO_SERVER).hostname)


def health():
    """两个引擎在不在。返回 (ok, [坏掉的名字])。

    只返回名字, 不返回长报错: 每条都带一遍"去跑 continuity-setup"的话, 两个引擎
    就是两遍, 而 continuity_status 自己还会再说一遍。提示只该出现一次。"""
    down = []
    for name, url in (("sd_server", f"{SD_SERVER}/sdcpp/v1/capabilities"),
                      ("audiocpp_server", f"{AUDIO_SERVER}/health")):
        try:
            get(url, timeout=5, tag=name)
        except Exception:
            down.append(name)
    return (not down), down
