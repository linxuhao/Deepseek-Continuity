# ==========================================
# 输出校验 —— 本插件存在的第二个理由。
#
# 一个算错的后端不会报错: 它返回 HTTP 200 和一个格式完全合法的全 0 WAV, 或者一张
# 纯灰 PNG。调用方 (LLM) 看到 status=done 就继续往下走, 把垃圾装进游戏里。
# 这里的规则很粗糙, 但它们抓的正是"退化"这一类, 而不是"画得不好看"那一类。
# ==========================================
import logging
import os
import wave

import numpy as np
from PIL import Image

from .config import MIN_IMAGE_STD, MIN_AUDIO_RMS_DBFS

log = logging.getLogger("continuity")


class DegenerateOutput(RuntimeError):
    """后端返回了格式合法但内容退化的产物。"""


def check_image(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    std = float(a.std())
    if std < MIN_IMAGE_STD:
        raise DegenerateOutput(
            f"degenerate image: std={std:.3f} < {MIN_IMAGE_STD} — 输出接近纯色, "
            f"多半是后端算错而不是提示词问题")
    log.info("image ok: %s std=%.1f", os.path.basename(path), std)
    return std


def check_audio(path):
    with wave.open(str(path)) as w:
        n, rate = w.getnframes(), w.getframerate()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        raise DegenerateOutput("empty audio")
    if not np.all(np.isfinite(x)):
        raise DegenerateOutput("audio contains non-finite samples")
    rms = float(np.sqrt((x ** 2).mean()))
    dbfs = 20 * np.log10(rms) if rms > 0 else -999.0
    if dbfs < MIN_AUDIO_RMS_DBFS:
        raise DegenerateOutput(f"degenerate audio: rms={dbfs:.1f} dBFS < {MIN_AUDIO_RMS_DBFS}")
    secs = n / rate
    log.info("audio ok: %s %.1fs rms=%.1f dBFS", os.path.basename(path), secs, dbfs)
    return dbfs, secs
