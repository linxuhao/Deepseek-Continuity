# ==========================================
# 配置 —— 全部来自环境变量, 默认值面向"本机单卡, 引擎跑在 docker 里"。
#
# 目录有两套路径, 分清楚很重要:
#   本进程看到的   {STATE_DIR}/actors/王五.wav
#   音频引擎看到的  /actors/王五.wav        (compose 把同一个目录 bind 进容器)
# 克隆用的 voice_ref 是交给引擎去打开的, 所以那条路径必须写引擎的视角。
# 早期版本两边混用, 表现是"铸声成功、说台词报文件不存在"。
# ==========================================
import os
import re
from pathlib import Path


def _env(name, default=None):
    """把空字符串当成"没设"。

    dsh 的 cordis patch 里 env 是 z.dict(String), 值不能是 undefined —— 所以那边写的是
    `process.env.X ?? ''`, 没设的变量会以空串传进来。os.getenv(name, default) 只在
    "键不存在"时才给默认值, 空串会原样返回, 于是 SD_SERVER 会变成 '' 而不是本机地址。
    默认值只留一份, 在这里。
    """
    return os.getenv(name) or default


def _dir(env, default):
    p = Path(_env(env, default)).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


STATE_DIR = Path(_env("CONTINUITY_STATE_DIR", "~/.continuity")).expanduser()
# 参考音和定妆图是不可复现的长期资产 —— 重铸出来是另一个人。它们和有保留期清理的
# generated/ 分开放, 不是为了整洁, 是为了不让清理逻辑哪天顺手删掉整部戏的角色。
ACTORS_DIR = _dir("CONTINUITY_ACTORS_DIR", STATE_DIR / "actors")
SUBJECTS_DIR = _dir("CONTINUITY_SUBJECTS_DIR", STATE_DIR / "subjects")
GENERATED_DIR = _dir("CONTINUITY_GENERATED_DIR", STATE_DIR / "generated")

# 引擎容器里的挂载点。远程引擎 / 自定义挂载时改这两个。
ENGINE_ACTORS_DIR = _env("CONTINUITY_ENGINE_ACTORS_DIR", "/actors")

# 默认值单独留一份: engines.setup_was_run() 要靠"地址是不是还是默认的"来判断
# 用户有没有自己接了后端。
DEFAULT_SD_SERVER = "http://127.0.0.1:9020"
DEFAULT_AUDIO_SERVER = "http://127.0.0.1:9021"
SD_SERVER = _env("SD_SERVER", DEFAULT_SD_SERVER)
AUDIO_SERVER = _env("AUDIO_SERVER", DEFAULT_AUDIO_SERVER)

MUSIC_MODEL_ID = _env("MUSIC_MODEL_ID", "stable-audio")
DESIGN_MODEL_ID = _env("DESIGN_MODEL_ID", "qwen3-tts")        # VoiceDesign: 描述 -> 声音
CLONE_MODEL_ID = _env("CLONE_MODEL_ID", "qwen3-tts-base")     # Base: 参考音 -> 声音
AUDIO_MODELS = {MUSIC_MODEL_ID, DESIGN_MODEL_ID, CLONE_MODEL_ID}

MAX_IMAGE_SIZE = int(_env("MAX_IMAGE_SIZE", "1024"))
# 引擎自己在 120s 硬截断: 请求 180/240/300/480 都返回 120001 ms 且不报错。
# 保留这个上限不是防炸 (显存/耗时都与时长无关), 而是把引擎的"静默截断"变成响应里
# 显式的 clamped 字段, 让调用方知道自己被截了。
MAX_AUDIO_SECONDS = float(_env("MAX_AUDIO_SECONDS", "120"))
JOB_TIMEOUT_S = float(_env("JOB_TIMEOUT_S", "900"))
# 引擎冷启动的等待上限 (宿主机重启后要从磁盘重读十几 GB 权重)
ENGINE_WAIT_S = float(_env("ENGINE_WAIT_S", "180"))

# 文本长度上限 —— 这是护栏, 不是礼貌性的限制。实测:
#   201 字 -> 45.7 s 音频, 生成 16 s   OK
#   402 字 -> 90.3 s 音频, 生成 42 s   OK
#   600 字 -> ~135 s 音频, 生成 ~63 s  GPU 挂死 (amdgpu GPU reset(6), 连带另一张卡上的
#             进程 SIGSEGV) —— 单次 Vulkan 提交扛不住这么长的持续计算。
# 200 字取在已知安全值 (402) 的一半, 而且 200 字已经是 45 秒旁白 —— 游戏里一句 NPC
# 台词通常 10~40 字, 这个上限不会碰到。更长的文本请分多次调用。
MAX_SPEECH_CHARS = int(_env("MAX_SPEECH_CHARS", "200"))
# 字数上限只挡住"输入长", 挡不住"输出跑飞": 引擎默认 max_tokens=2048 (~170 s 音频),
# 一句短台词一旦退化成循环, 照样能生成几分钟并拖挂 GPU。按字数推 token 预算,
# 让跑飞的请求早早自己停下。实测 ~2.7 token/字, 取 4.0 留余量。
SPEECH_TOKENS_PER_CHAR = float(_env("SPEECH_TOKENS_PER_CHAR", "4.0"))
SPEECH_MAX_TOKENS = int(_env("SPEECH_MAX_TOKENS", "900"))

# 参考音时长上限 —— 实测出来的, 不是拍的。克隆时参考音是要进模型的, 显存代价随它
# 线性涨 (基线 ~3 GiB + 约 0.19 GiB/秒):
#     5s 3.92    8s 4.58    12s 5.50    15s 6.59    18s 7.31    20s 7.58    30s 9.04
# 取 15s 是因为它是最后一个还压在生图峰值 (6.80 GiB) 之下的档 —— 再长, 配音就取代
# 生图成了整套东西的显存天花板, 8 GiB 的卡就装不下了。
# 大卡不单独放宽: 30s 的参考音对音色没有额外收益 (3~10 秒就足够定住), 为此多一个
# 按显存分档的配置项, 换来的是"同一段录音在你机器上能导入、在别人机器上不能"。
# 想要就自己设 CONTINUITY_REF_MAX_S。
REF_MIN_S = float(_env("CONTINUITY_REF_MIN_S", "2"))
REF_MAX_S = float(_env("CONTINUITY_REF_MAX_S", "15"))
# 铸声台词的字数上限, 单独于 MAX_SPEECH_CHARS —— 后者管的是"说一句台词"(200 字/45 秒),
# 而铸声产出的是参考音, 它会被之后每一句台词反复吃进显存。200 字铸出来是 45 秒的
# 参考音, 那之后每句话都要付 11 GiB 以上。60 字约 13 秒, 落在上面那张表的安全档里。
MAX_SAMPLE_CHARS = int(_env("CONTINUITY_MAX_SAMPLE_CHARS", "45"))

DEFAULT_VOICE = _env("DEFAULT_VOICE", "A neutral adult narrator, clear and natural")
# 铸声用的台词: 覆盖面尽量广, 时长 ~7 s (克隆参考音的常用区间)
DEFAULT_SAMPLE_TEXT = _env(
    "DEFAULT_SAMPLE_TEXT",
    "江湖路远，人心难测。今日一别，山高水长，来日方长，后会有期。")

# 空闲这么久之后把音频模型全卸掉, 让整张卡回到零常驻。
# 不"用完立刻卸"是因为重载要 4.3 s —— 连着配十句台词的人不该每句都付这个钱;
# 也不能不卸, 否则这张卡在用户不用我们的时候仍被占着 2 GB, 打不了游戏。
# 0 = 关闭空闲卸载 (始终常驻)。
AUDIO_IDLE_UNLOAD_S = float(_env("AUDIO_IDLE_UNLOAD_S", "120"))

# 生成产物保留天数; 0 表示不清理。actors/ 和 subjects/ 永不清理。
RETENTION_DAYS = float(_env("RETENTION_DAYS", "30"))
CLEANUP_INTERVAL_S = float(_env("CLEANUP_INTERVAL_S", "21600"))   # 6 小时

# ---- 输出校验 ----
# 最贵的一课: 后端算错时会写出一个全 0 的 WAV 并返回 200。任何"看起来成功但内容
# 退化"的输出都必须让调用显式失败。
MIN_IMAGE_STD = float(_env("MIN_IMAGE_STD", "3.0"))     # 纯灰图实测 std=0.5
MIN_AUDIO_RMS_DBFS = float(_env("MIN_AUDIO_RMS_DBFS", "-60"))

NAME_RE = re.compile(r"^[\w一-鿿-]{1,40}$")

# 三类要盯的东西不一样, 所以取景也不一样 —— 定妆图上没留下的信息, 出场景图时
# 模型只能自己编, 而它每次编得都不一样。
SUBJECT_FRAMING = {
    # 人: 脸和衣着是识别点, 要正面全身看全
    "character": "full body character reference, neutral standing pose, facing viewer, "
                 "plain flat background, clean game art",
    # 动物: 体型比例和花纹分布是识别点, 四分之三站姿同时给出侧面轮廓和正面头部
    "animal": "full body animal reference, standing in three-quarter view, head visible, "
              "plain flat background, no scenery, clean game art",
    # 物件: 几何是识别点, 且最容易漂 —— 正投影看不出体积, 换个角度就没有可对齐的信息
    "object": "single game asset reference, three-quarter view, centered, isolated, "
              "plain flat background, no scenery, clean game art",
}
DEFAULT_SUBJECT_KIND = "character"

# rembg 模型。同机实测 (512x512, 热推理):
#   u2netp                0.16s  软边最多
#   isnet-general-use     1.38s
#   birefnet-general-lite 5.89s  <- 默认
#   bria-rmbg            10.85s  同样干净, 翻倍耗时只换来边缘一点点提升
# best 的代价不在时间而在内存: 峰值 ~6.8 GB 常驻内存 (与输入尺寸无关)。
# 16 GB 以下的机器由 continuity-setup 把默认值改成 fast。
REMBG_MODELS = {"best": "birefnet-general-lite", "fast": "u2netp"}
DEFAULT_CUTOUT_QUALITY = _env("CONTINUITY_CUTOUT_QUALITY", "best")

# 棋盘格判定阈值 (见 cutout.looks_like_checkerboard)
CHECKER_MIN_CAND = float(_env("CHECKER_MIN_CAND", "0.05"))
CHECKER_MIN_GAP = float(_env("CHECKER_MIN_GAP", "15"))
CHECKER_MAX_VALLEY = float(_env("CHECKER_MAX_VALLEY", "0.30"))
CHECKER_MIN_RUNS = float(_env("CHECKER_MIN_RUNS", "0.60"))

# alpha 退化判定 (见 cutout.alpha_report)
ALPHA_MAX_TRANSPARENT = float(_env("ALPHA_MAX_TRANSPARENT", "0.95"))
ALPHA_MIN_TRANSPARENT = float(_env("ALPHA_MIN_TRANSPARENT", "0.02"))
ALPHA_MIN_BLOB = float(_env("ALPHA_MIN_BLOB", "0.02"))
ALPHA_MAX_HOLES = float(_env("ALPHA_MAX_HOLES", "0.05"))
ALPHA_MAX_BG_DETAIL = float(_env("ALPHA_MAX_BG_DETAIL", "0.5"))

SFX_RATE = 44100
MAX_SFX_SECONDS = float(_env("MAX_SFX_SECONDS", "5"))

# 能力开关。显存不够时 continuity-setup 会关掉生图那半而保留音频那半 ——
# 见 preflight.py: 换模型省不下显存 (Q4 与 Q8 实测同为 6.6 GB), 能省的只有"不装"。
ENABLE_IMAGE = _env("CONTINUITY_ENABLE_IMAGE", "1") not in ("0", "false", "False")
ENABLE_AUDIO = _env("CONTINUITY_ENABLE_AUDIO", "1") not in ("0", "false", "False")
