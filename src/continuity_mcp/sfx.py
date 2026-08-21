# ==========================================
# 程序化音效 (sfxr/jsfxr 风格, 纯 numpy)。
#
# 刻意不走 generate_music: 那是扩散模型, 几十秒起步、没有循环点、出来的是宽带糊音。
# 游戏音效是 10~200ms 的瞬态, 要的是精确、即时、可复现 —— 一次合成 ~10ms,
# 同 seed 逐字节相同, 而且完全不碰 GPU。
# ==========================================
import wave
from dataclasses import dataclass, asdict

import numpy as np

from .config import SFX_RATE, MAX_SFX_SECONDS

SFX_WAVES = ("square", "saw", "sine", "triangle", "noise")


@dataclass
class SfxParams:
    wave: str = "square"          # square / saw / sine / triangle / noise
    # 包络 (秒): attack 0->1, decay 1->sustain_level, sustain 保持, release ->0
    attack: float = 0.005
    decay: float = 0.03
    sustain: float = 0.06
    sustain_level: float = 0.8
    release: float = 0.10
    base_freq: float = 440.0      # Hz (noise 波形下是采样保持的刷新率)
    freq_slide: float = 0.0       # 八度/秒
    delta_slide: float = 0.0      # 八度/秒^2
    duty: float = 0.5             # 方波占空比
    duty_sweep: float = 0.0       # 占空比变化/秒
    vibrato_depth: float = 0.0    # 半音
    vibrato_speed: float = 0.0    # Hz
    arp_mult: float = 1.0         # arp_time 之后基频乘以它 (金币的两段音阶)
    arp_time: float = 0.0         # 秒
    lpf: float = 1.0              # 低通截止, 归一化 (1 = 不滤)
    lpf_sweep: float = 0.0        # 截止的八度/秒
    hpf: float = 0.0              # 高通截止, 归一化 (0 = 不滤)
    volume: float = 0.95          # 归一化后的峰值


SFX_PRESETS = {
    "jump":      dict(wave="square", base_freq=360, freq_slide=3.0, attack=0.005, decay=0.03,
                      sustain=0.07, sustain_level=0.85, release=0.09, duty=0.5, duty_sweep=0.5),
    "coin":      dict(wave="square", base_freq=988, arp_mult=1.5, arp_time=0.06, attack=0.002,
                      decay=0.012, sustain=0.10, sustain_level=0.9, release=0.22, duty=0.35),
    "hit":       dict(wave="noise", base_freq=3000, freq_slide=-2.5, attack=0.001, decay=0.02,
                      sustain=0.02, sustain_level=0.5, release=0.13, lpf=0.55, lpf_sweep=-2.5),
    "explosion": dict(wave="noise", base_freq=44100, freq_slide=-0.9, attack=0.002, decay=0.10,
                      sustain=0.25, sustain_level=0.8, release=0.85, lpf=0.95, lpf_sweep=-0.5),
    "powerup":   dict(wave="square", base_freq=320, freq_slide=1.2, arp_mult=1.25, arp_time=0.20,
                      attack=0.01, decay=0.05, sustain=0.24, sustain_level=0.85, release=0.24,
                      vibrato_depth=0.35, vibrato_speed=13.0),
    "laser":     dict(wave="saw", base_freq=1500, freq_slide=-3.6, attack=0.001, decay=0.02,
                      sustain=0.05, sustain_level=0.7, release=0.15, hpf=0.02),
    "select":    dict(wave="square", base_freq=880, attack=0.002, decay=0.012, sustain=0.03,
                      sustain_level=0.8, release=0.045, duty=0.25),
    "hurt":      dict(wave="saw", base_freq=520, freq_slide=-1.5, attack=0.002, decay=0.03,
                      sustain=0.05, sustain_level=0.6, release=0.17, lpf=0.7),
}

# seed 抖动的是参数, 不是采样点 —— 同一个 preset 听起来还是它自己, 只是每次略有不同。
_SFX_JITTER = {"base_freq": 0.10, "freq_slide": 0.15, "duty": 0.12, "decay": 0.15,
               "sustain": 0.15, "release": 0.15, "arp_mult": 0.04, "arp_time": 0.15,
               "vibrato_depth": 0.20, "lpf": 0.08}


def sfx_params(preset, seed=None, overrides=None):
    if preset not in SFX_PRESETS:
        raise ValueError(f"未知 preset: {preset} (可用: {', '.join(SFX_PRESETS)})")
    p = SfxParams(**SFX_PRESETS[preset])
    rng = np.random.default_rng(seed)
    if seed is not None:
        for field, amount in _SFX_JITTER.items():
            v = getattr(p, field)
            if v:
                setattr(p, field, float(v) * float(1.0 + rng.uniform(-amount, amount)))
    for k, v in (overrides or {}).items():
        if not hasattr(p, k):
            raise ValueError(f"未知参数: {k}")
        setattr(p, k, v if k == "wave" else float(v))
    if p.wave not in SFX_WAVES:
        raise ValueError(f"wave 必须是 {'/'.join(SFX_WAVES)}")
    total = p.attack + p.decay + p.sustain + p.release
    if not (0.005 <= total <= MAX_SFX_SECONDS):
        raise ValueError(f"总时长 {total:.3f}s 超出 0.005~{MAX_SFX_SECONDS}s")
    return p, rng


def sfx_envelope(p, n):
    na = max(1, int(p.attack * SFX_RATE))
    nd = max(1, int(p.decay * SFX_RATE))
    ns = max(0, int(p.sustain * SFX_RATE))
    nr = max(1, n - na - nd - ns)
    sl = float(np.clip(p.sustain_level, 0.0, 1.0))
    env = np.concatenate([
        np.linspace(0.0, 1.0, na, endpoint=False),
        np.linspace(1.0, sl, nd, endpoint=False),
        np.full(ns, sl),
        np.linspace(sl, 0.0, nr),
    ])
    return env[:n] if env.size >= n else np.concatenate([env, np.zeros(n - env.size)])


def sfx_filters(x, p, t):
    """一阶低通 (可扫频) + 一阶高通。逐样本递归, 无法向量化, 但音效最长几万个样本。"""
    if p.lpf >= 1.0 and not p.lpf_sweep and p.hpf <= 0.0:
        return x
    cut = np.clip(p.lpf * np.exp2(p.lpf_sweep * t), 0.001, 1.0).tolist()
    a = float(np.clip(1.0 - p.hpf, 0.0, 1.0))
    xs, out = x.tolist(), []
    lp = hp = hp_prev_in = 0.0
    for i, s in enumerate(xs):
        lp += cut[i] * (s - lp)
        hp = a * (hp + lp - hp_prev_in)
        hp_prev_in = lp
        out.append(hp)
    return np.asarray(out, dtype=np.float64)


def sfx_render(p, rng):
    """按参数合成一段单声道 float64 波形 (峰值归一化到 p.volume)。"""
    n = max(2, int(round((p.attack + p.decay + p.sustain + p.release) * SFX_RATE)))
    t = np.arange(n, dtype=np.float64) / SFX_RATE
    octaves = p.freq_slide * t + 0.5 * p.delta_slide * t * t
    if p.vibrato_depth and p.vibrato_speed:
        octaves = octaves + (p.vibrato_depth / 12.0) * np.sin(2 * np.pi * p.vibrato_speed * t)
    f = p.base_freq * np.exp2(octaves)
    if p.arp_mult != 1.0 and p.arp_time > 0:
        f = np.where(t >= p.arp_time, f * p.arp_mult, f)
    # noise 的"频率"是采样保持的刷新率, 可以一直到采样率 (刷新率 = 采样率就是白噪声)
    f = np.clip(f, 10.0, SFX_RATE if p.wave == "noise" else 0.45 * SFX_RATE)
    phase = np.cumsum(f) / SFX_RATE
    frac = phase - np.floor(phase)
    if p.wave == "sine":
        x = np.sin(2 * np.pi * frac)
    elif p.wave == "saw":
        x = 2.0 * frac - 1.0
    elif p.wave == "triangle":
        x = 4.0 * np.abs(frac - 0.5) - 1.0
    elif p.wave == "noise":
        table = rng.uniform(-1.0, 1.0, size=n + 8)     # 够长, 不会在一次音效里循环
        x = table[np.floor(phase).astype(np.int64) % table.size]
    else:                                              # square
        duty = np.clip(p.duty + p.duty_sweep * t, 0.01, 0.99)
        x = np.where(frac < duty, 1.0, -1.0)
    x = sfx_filters(x * sfx_envelope(p, n), p, t)
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x * (p.volume / peak)
    return x


def write_wav(path, x):
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SFX_RATE)
        w.writeframes(pcm.tobytes())

