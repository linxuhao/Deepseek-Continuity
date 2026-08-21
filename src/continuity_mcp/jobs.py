# ==========================================
# 用得着 GPU 的活, 全部经过这里, 而且一次只跑一件。
#
# 串行不是为了简单, 是显存要求的: 峰值 = 单个最大模型 (6.80 GB) 而不是它们的和,
# 8 GB 的卡装得下整套, 靠的就是"同一时刻只有一件事在跑"。并发会把这个前提取消掉。
# ==========================================
import base64
import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from . import engines, store
from .config import (GENERATED_DIR, MAX_IMAGE_SIZE, SUBJECT_FRAMING, DEFAULT_SUBJECT_KIND,
                     MUSIC_MODEL_ID, DESIGN_MODEL_ID, CLONE_MODEL_ID, DEFAULT_VOICE,
                     MAX_SPEECH_CHARS, MAX_AUDIO_SECONDS, NAME_RE,
                     REF_MIN_S, REF_MAX_S, MAX_SAMPLE_CHARS)
from .verify import check_image, check_audio

log = logging.getLogger("continuity")

# 全局串行锁。跨线程 (MCP 的 to_thread) 生效。
_gpu_lock = threading.Lock()


class UserError(ValueError):
    """给调用方 (LLM) 看的指令式报错 —— 告诉它下一步该干什么, 不只是"失败了"。"""


def _new_name(prefix, ext):
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"


def _out(name):
    return GENERATED_DIR / name


def _ref_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _fit_size(path, want_w, want_h):
    """Make the file the size that was ASKED for, not the one the model felt like.

    The engine snaps a request onto its own latent grid (and to a minimum edge),
    so a 40x32 sprite comes back as 256x256. Callers that need an exact canvas —
    a game asset pipeline validating sprite dimensions — then reject every image
    and ship nothing, which is exactly what happened. width/height are documented
    as the output size, so honour them here instead of leaking the grid.

    LANCZOS going down (almost always the case: the grid minimum is far above a
    sprite), NEAREST going up so an upscaled pixel sprite keeps hard edges."""
    try:
        img = Image.open(path)
        if (img.width, img.height) == (want_w, want_h):
            return
        shrinking = want_w * want_h < img.width * img.height
        img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB").resize(
            (want_w, want_h), Image.LANCZOS if shrinking else Image.NEAREST).save(path)
        log.info("resized %dx%d -> %dx%d", img.width, img.height, want_w, want_h)
    except Exception:
        log.warning("resize to %dx%d failed; keeping engine canvas", want_w, want_h, exc_info=True)


def _clamp_size(w, h):
    """两个不同的尺寸, 混为一谈正是每张精灵图形状都不对的原因:
    want_* 是调用方拿到的 (只有上限起作用, 最后 _fit_size 缩放),
    width/height 是问引擎要的, 它有 256 的下限。只有上限值得回报, 下限是内部实现。"""
    want_w, want_h = min(w, MAX_IMAGE_SIZE), min(h, MAX_IMAGE_SIZE)
    return want_w, want_h, max(256, want_w), max(256, want_h)


def _run(fn, *a, needs=None, **kw):
    """串行执行一件用 GPU 的活。

    needs 是这件活要用的模型 id; None 表示它不用音频模型 (生图)。开工前把不是它的
    都卸掉 —— 这样峰值恒等于单个最大模型, 与调用顺序无关。
    """
    with _gpu_lock:
        engines.mark_busy()
        try:
            if needs is not None or engines.engines_share_a_gpu():
                engines.release_all_but(needs, "为生图腾显存" if needs is None else "")
            return fn(*a, **kw)
        finally:
            engines.mark_idle()


# ---- 图像 ----

def generate_image(prompt, width=1024, height=1024, seed=None, ref_b64=None,
                   steps=None, cfg_scale=1.0):
    want_w, want_h, w, h = _clamp_size(width, height)
    def work():
        data, ext = engines.sd_generate(prompt, w, h, steps, cfg_scale, seed, ref_b64)
        name = _new_name("img", ext)
        _out(name).write_bytes(data)
        _fit_size(_out(name), want_w, want_h)
        check_image(_out(name))
        return name
    name = _run(work, needs=None)
    clamped = {"width": want_w, "height": want_h} if (want_w, want_h) != (width, height) else None
    return {"file": name, "path": str(_out(name)), "clamped": clamped}


def subject_image(subject, scene, width=512, height=512, seed=None):
    """场景图: 外观由定妆图决定, scene 只管场景/动作/视角。"""
    s = store.load_subject(subject)
    if s is None:
        raise UserError(
            f"subject '{subject}' 不存在 —— 先调 create_character / create_animal / "
            f"create_object (name='{subject}', appearance='一段外观描述') 定妆, "
            f"再用它出场景图。现有: {store.subject_names() or '(还没有)'}")
    png, _ = store.subject_paths(subject)
    return generate_image(f'{s["appearance"]}, {scene}', width, height, seed, _ref_b64(png))


def create_subject(name, appearance, kind=DEFAULT_SUBJECT_KIND, width=512, height=512,
                   seed=None, force=False):
    """定妆: 生成一张参考图存成 subject。和铸声一样占 GPU, 所以走同一把锁。"""
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise UserError("subject 名只能是字母/数字/下划线/连字符/中文, 1~40 字")
    if kind not in SUBJECT_FRAMING:
        raise UserError(f"kind 必须是 {'/'.join(SUBJECT_FRAMING)}")
    if not (appearance or "").strip():
        raise UserError("appearance 是外观描述, 不能为空")
    if store.load_subject(name) is not None and not force:
        raise UserError(f"subject '{name}' 已存在。定妆一次用一辈子, 覆盖会让它之前所有"
                        f"场景图的外观对不上 —— 确实要重定就传 force=True。")
    _, _, w, h = _clamp_size(width, height)
    png, meta_path = store.subject_paths(name)

    def work():
        t = time.time()
        data, _ = engines.sd_generate(f"{appearance}, {SUBJECT_FRAMING[kind]}", w, h, seed=seed)
        png.write_bytes(data)
        check_image(png)      # 退化的定妆图会污染这个 subject 的每一张场景图
        meta = store.save_meta(meta_path, {"name": name, "kind": kind, "appearance": appearance,
                                           "reference_path": str(png), "seed": seed})
        log.info("[subject] 定妆 %s (%s) 完成 (%.1fs)", name, kind, time.time() - t)
        return meta
    return _run(work, needs=None)


# ---- 音频 ----

def generate_music(prompt, seed=None, duration=30.0, num_inference_steps=None):
    secs = max(1.0, min(float(duration), MAX_AUDIO_SECONDS))
    req = {"task_route": "text2music", "text": prompt, "duration_seconds": secs}
    if seed is not None:
        req["seed"] = seed
    if num_inference_steps:
        req["num_inference_steps"] = num_inference_steps

    def work():
        t = time.time()
        res = engines.post(f"{engines.AUDIO_SERVER}/v1/tasks/run",
                           {"model": MUSIC_MODEL_ID, "request": req}, "audiocpp-server")
        b64 = res.get("audio")
        if not b64:
            raise RuntimeError(f"audiocpp-server returned no audio: {str(res)[:300]}")
        name = _new_name("music", "wav")
        _out(name).write_bytes(base64.b64decode(b64))
        check_audio(_out(name))
        log.info("[music] ok in %.1fs", time.time() - t)
        return name
    name = _run(work, needs=MUSIC_MODEL_ID)
    return {"file": name, "path": str(_out(name)), "duration": secs,
            "clamped": {"duration": secs} if secs != duration else None}


def _clip_text(text):
    text = (text or "").strip()
    if not text:
        raise UserError("台词不能为空")
    return text[:MAX_SPEECH_CHARS], (len(text) > MAX_SPEECH_CHARS)


def create_actor(name, voice, sample_text=None, seed=None, force=False):
    """用 VoiceDesign 铸一句参考音, 存成角色。"""
    from .config import DEFAULT_SAMPLE_TEXT
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise UserError("actor 名只能是字母/数字/下划线/连字符/中文, 1~40 字")
    if not (voice or "").strip():
        raise UserError("create_actor 必须给 voice (声音的自然语言描述)")
    if store.load_actor(name) is not None and not force:
        raise UserError(f"actor '{name}' 已存在。铸声一次用一辈子, 覆盖会让它之前所有台词的"
                        f"音色对不上 —— 确实要重铸就传 force=True。")
    raw = (sample_text or DEFAULT_SAMPLE_TEXT).strip()
    text, clipped = raw[:MAX_SAMPLE_CHARS], len(raw) > MAX_SAMPLE_CHARS
    wav, meta_path = store.actor_paths(name)

    def work():
        t = time.time()
        audio, tm = engines.tts(DESIGN_MODEL_ID, text, instructions=voice, seed=seed)
        wav.write_bytes(audio)
        # 退化的参考音会污染这个角色的每一句台词
        _, secs = check_audio(wav)
        meta = store.save_meta(meta_path, {
            "name": name, "voice": voice,
            "transcript": text,                     # Base 克隆要参考音的文字
            "reference_path": str(wav), "ref_seconds": round(secs, 1), "seed": seed})
        log.info("[actor] 铸声 %s 完成 (%.1fs, rtf=%s)", name, time.time() - t, tm.get("rtf"))
        return meta
    meta = _run(work, needs=DESIGN_MODEL_ID)
    meta["clipped"] = clipped
    # 字数卡不准时长 (实测 60 字 -> 19.1 秒), 所以铸完按真实时长再复核一次。
    # 不重铸也不失败 —— 这段参考音本身是好的, 只是之后每句台词都会比必要的贵。
    if meta["ref_seconds"] > REF_MAX_S:
        meta["too_long"] = (
            f"这段参考音 {meta['ref_seconds']:.1f}s, 超过本机上限 {REF_MAX_S:.0f}s。"
            f"它能用, 但之后每一句台词都要多付显存 (约 0.19 GiB/秒), 显存紧的机器上"
            f"可能因此失败。想省事就用更短的 sample_text 重铸一次 —— 3~10 秒足够定住音色。")
    return meta


def speak(text, actor=None, voice=None, seed=None, speaking_rate=None):
    if actor:
        a = store.load_actor(actor)
        if a is None:
            # 指令式报错: 调用方是 LLM, 告诉它下一步该干什么
            raise UserError(f"actor '{actor}' 不存在 —— 先调 create_actor(name='{actor}', "
                            f"voice='一段声音描述') 铸声, 再用它说台词。"
                            f"现有角色: {store.actor_names() or '(还没有)'}")
        model_id = CLONE_MODEL_ID
        # 参考音随请求一起发过去, 而不是给引擎一个路径让它自己去开。
        #
        # 早先给的是路径, 那条路径只有引擎解析得了 —— 引擎在别的机器上时它必然找不到,
        # 表现是"铸声成功、之后每一句台词都 HTTP 500"。当时我把它当成引擎的限制写进了
        # 文档, 其实是我选错了端点: /v1/audio/speech 一直支持
        # {"type":"base64","data":...} (audio.cpp runtime.cpp), 上限 5 MiB。
        # 我们的参考音是 24 kHz 单声道 16-bit, 15 秒也才 720 KB。
        # 换成内联之后, "两边必须挂同一个目录"这个要求整个消失了。
        wav, _ = store.actor_paths(actor)
        raw = wav.read_bytes()
        if len(raw) > MAX_VOICE_REF_BYTES:
            raise UserError(
                f"参考音 {len(raw)/2**20:.1f} MiB, 超过引擎 {MAX_VOICE_REF_BYTES/2**20:.0f} MiB 的上限。"
                f"用更短的参考音重铸 —— 3~10 秒就足够定住音色。")
        kw = {"voice_ref_b64": base64.b64encode(raw).decode(),
              "reference_text": a["transcript"]}
    else:
        # 一次性旁白: VoiceDesign 直接从描述生成, 不保证跨句音色一致
        model_id = DESIGN_MODEL_ID
        kw = {"instructions": voice or DEFAULT_VOICE}
    clipped_text, clipped = _clip_text(text)

    def work():
        t = time.time()
        audio, tm = engines.tts(model_id, clipped_text, seed=seed,
                                speaking_rate=speaking_rate, **kw)
        name = _new_name("speech", "wav")
        _out(name).write_bytes(audio)
        check_audio(_out(name))
        log.info("[%s] speech ok in %.1fs (rtf=%s)", model_id, time.time() - t, tm.get("rtf"))
        return name
    name = _run(work, needs=model_id)
    return {"file": name, "path": str(_out(name)), "clipped": clipped}


# ---- 导入外部素材 ----
# 定妆/铸声的本质是"一份参考产物 + 一段描述", 参考产物是我们自己生成的还是别处来的
# 并不重要。所以导入走的是同一条路, 只是把生成那一步换成校验 + 落盘 ——
# 用别的工具做好的角色照样能在这里保持一致。

REF_RATE = 24000            # 克隆参考音的采样率
# 引擎对内联参考音的硬上限 (audio.cpp: kMaxVoiceRefBytes)。我们的 REF_MAX_S 让实际
# 大小停在 720 KB 左右, 这个检查只是为了在超限时说人话而不是 HTTP 500。
MAX_VOICE_REF_BYTES = 5 * 1024 * 1024


def _normalize_ref_wav(src, dst):
    """把任意 WAV 转成 24 kHz 单声道 16-bit —— 参考音的规格。

    自己转而不是要求调用方转: 用户手上的素材大多是 44.1k 立体声, 而"格式不对"在引擎
    那边的表现是音色古怪或直接失败, 都不会说是采样率的事。不引入 ffmpeg 依赖, 线性
    重采样对一段人声参考音足够。
    """
    import wave as _wave
    with _wave.open(str(src)) as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise UserError(f"参考音必须是 16-bit PCM WAV (这个是 {width * 8}-bit)。"
                        f"转一下: ffmpeg -i 原文件 -acodec pcm_s16le -ac 1 -ar 24000 新文件.wav")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    secs = x.size / rate
    if secs < REF_MIN_S:
        raise UserError(f"参考音只有 {secs:.1f}s, 至少要 {REF_MIN_S:.0f}s —— 太短克隆不出音色。")
    if secs > REF_MAX_S:
        # 这里刻意不自动截断。图片缩小仍然是同一张图, 而音频截掉后半段之后,
        # transcript 描述的就不再是这段音频了 —— 而音文对齐正是克隆的依据。
        # 静默截断会得到一个"成功导入"却音色不对的角色, 那比报错糟糕得多。
        raise UserError(
            f"参考音 {secs:.1f}s, 上限 {REF_MAX_S:.0f}s。这里不替你截断: 截了之后 "
            f"transcript 描述的就不是这段音频了, 而音文对齐正是克隆的依据, "
            f"结果会是一个'导入成功'但音色不对的角色。\n"
            f"请自己剪一段 {REF_MIN_S:.0f}~{REF_MAX_S:.0f}s 的, 并给出那一段对应的文字。"
            f"3~10 秒就足够定住音色。\n"
            f"  ffmpeg -i {{原文件}} -t {REF_MAX_S:.0f} -acodec pcm_s16le -ac 1 -ar 24000 ref.wav\n"
            f"(上限来自显存: 参考音每秒约 0.19 GiB, {REF_MAX_S:.0f}s 时峰值已达 6.6 GiB。"
            f"显存宽裕可设 CONTINUITY_REF_MAX_S 放宽。)")
    if rate != REF_RATE:
        m = int(round(x.size * REF_RATE / rate))
        x = np.interp(np.linspace(0, x.size - 1, m), np.arange(x.size), x)
    pcm = np.clip(np.round(x), -32768, 32767).astype("<i2")
    with _wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(REF_RATE)
        w.writeframes(pcm.tobytes())
    return secs, rate, ch


def import_actor(name, audio_path, transcript, force=False):
    """用一段现成的录音铸声 —— 别处做好的声音也能在这里保持一致。"""
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise UserError("actor 名只能是字母/数字/下划线/连字符/中文, 1~40 字")
    if not (transcript or "").strip():
        raise UserError("必须给 transcript —— 那段录音里念的是什么。克隆模型要拿它对齐"
                        "音频和文字, 写错了音色会明显不对。")
    src = Path(audio_path).expanduser()
    if not src.is_file():
        raise UserError(f"找不到 {src}")
    if store.load_actor(name) is not None and not force:
        raise UserError(f"actor '{name}' 已存在。覆盖会让它之前所有台词的音色对不上 —— "
                        f"确实要换就传 force=True。")
    wav, meta_path = store.actor_paths(name)
    try:
        secs, rate, ch = _normalize_ref_wav(src, wav)
    except UserError:
        raise
    except Exception as e:
        raise UserError(f"读不了这个 WAV ({e})。只收 16-bit PCM WAV; 其它格式先转:\n"
                        f"  ffmpeg -i {src} -acodec pcm_s16le -ac 1 -ar 24000 ref.wav")
    check_audio(wav)
    meta = store.save_meta(meta_path, {
        "name": name, "voice": f"(导入自 {src.name})", "transcript": transcript.strip(),
        "reference_path": str(wav),
        "imported_from": str(src), "source_format": f"{rate} Hz / {ch} ch / {secs:.1f}s",
        "ref_seconds": round(secs, 1)})
    # 升采样补不回丢掉的高频。不拦, 但一定要说 —— 否则就是又一个"导入成功、
    # 音色却比原声闷"的静默降级。
    if rate < REF_RATE:
        meta["lowband"] = (
            f"这段录音只有 {rate} Hz, 低于克隆用的 {REF_RATE} Hz。升采样补不回丢掉的高频, "
            f"音色会比原声闷。有更高采样率的原始文件就换那个。")
    log.info("[actor] 导入 %s 完成 (%s)", name, meta["source_format"])
    return meta


def import_subject(name, image_path, appearance, kind=DEFAULT_SUBJECT_KIND, force=False):
    """用一张现成的图定妆 —— 别处画好的角色/物件也能在这里保持一致。"""
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise UserError("subject 名只能是字母/数字/下划线/连字符/中文, 1~40 字")
    if kind not in SUBJECT_FRAMING:
        raise UserError(f"kind 必须是 {'/'.join(SUBJECT_FRAMING)}")
    if not (appearance or "").strip():
        raise UserError("必须给 appearance —— 它会被拼进之后每一张场景图的提示词。"
                        "只有参考图而没有文字描述时, 模型对'这是什么'没有着落, 外观照样会漂。")
    src = Path(image_path).expanduser()
    if not src.is_file():
        raise UserError(f"找不到 {src}")
    if store.load_subject(name) is not None and not force:
        raise UserError(f"subject '{name}' 已存在。覆盖会让它之前所有场景图的外观对不上 —— "
                        f"确实要换就传 force=True。")
    png, meta_path = store.subject_paths(name)
    try:
        img = Image.open(src)
        w, h = img.size
        # 缩小而不是拒绝: 缩过的图还是同一张图, 语义没变 (音频不是这样, 见上面)。
        # 但一定要把缩成了什么报回去 —— 静默改掉调用方给的东西是另一种坑。
        if max(w, h) > MAX_IMAGE_SIZE:
            sc = MAX_IMAGE_SIZE / max(w, h)
            img = img.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
        # 存成 RGB: 带 alpha 的参考图交给引擎, 透明区会被当成黑色实心块
        img.convert("RGB").save(png, format="PNG")
    except UserError:
        raise
    except Exception as e:
        raise UserError(f"读不了这张图: {e}")
    check_image(png)
    meta = store.save_meta(meta_path, {
        "name": name, "kind": kind, "appearance": appearance.strip(),
        "reference_path": str(png), "imported_from": str(src),
        "source_size": f"{w}x{h}", "stored_size": f"{img.width}x{img.height}",
        "resized": (img.width, img.height) != (w, h)})
    log.info("[subject] 导入 %s (%s) 完成, 原图 %dx%d -> %dx%d",
             name, kind, w, h, img.width, img.height)
    return meta
