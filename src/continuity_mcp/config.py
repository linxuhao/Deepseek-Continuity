# ==========================================
# 配置 —— 全部来自环境变量, 默认值面向"本机单卡, 引擎跑在 docker 里"。
#
# 所有目录都只有本进程会读写。参考音是读出来随请求内联发给引擎的, 引擎不需要
# 看见任何一个本地目录 —— 这让"引擎跑在别的机器上"变成一件不需要额外配置的事。
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

# 默认值单独留一份: engines.setup_was_run() 要靠"地址是不是还是默认的"来判断
# 用户有没有自己接了后端。
DEFAULT_SD_SERVER = "http://127.0.0.1:9020"
DEFAULT_AUDIO_SERVER = "http://127.0.0.1:9021"
SD_SERVER = _env("SD_SERVER", DEFAULT_SD_SERVER)
AUDIO_SERVER = _env("AUDIO_SERVER", DEFAULT_AUDIO_SERVER)

# 听写单独一个后端地址, 默认跟着 AUDIO_SERVER 走 —— 不设它, 一切和以前一样。
#
# 为什么只有听写有这一档: 它的端点是 OpenAI 那套 (multipart file + model, 回 {"text"}),
# 于是"别人家的 ASR"是真实存在的东西 —— vLLM、任何 OpenAI 兼容的服务、别人的机器。
# 配音和音乐没有这个待遇: /v1/audio/speech 收的是 voice_ref 内联 base64 + reference_text,
# 音乐走的是 /v1/tasks/run, 两个都是 audio.cpp 自己的形状, 没有第三方讲这套话。
# 它们的"BYO"只能是"另一台 audiocpp_server", 而那正是 AUDIO_SERVER 已经在做的事。
ASR_SERVER = _env("ASR_SERVER", "") or AUDIO_SERVER
ASR_API_KEY = _env("ASR_API_KEY", "")

# 发给别人家 API 的采样率: native (原样发) | 16000。
#
# 默认 native, 因为"该不该降采样"是后端的事, 不是我们的事。Whisper 那一系确实在
# 16 kHz 上算, 但不是所有服务都是 —— 一些厂商的模型吃到 48 kHz 并明确建议发原生音频,
# 替他们降一道是在丢他们要用的信息。我们知道的只有自己那个引擎的模型。
#
# 而有的 OpenAI 兼容服务反过来只收 16 kHz (实测 vLLM 对 22.05k/24k 一律 400
# "Invalid or unsupported audio file", 只字不提采样率)。所以标准那条路遇到 400 会
# 自动按 16 kHz 重发一次; 后端确定是这一类时把这个设成 16000, 省掉那次白跑的往返。
ASR_SEND_RATE = _env("ASR_SEND_RATE", "native").strip().lower()


# 听写是不是由"我们自己那个 audiocpp_server"提供的。
#
# 判据是同不同一个**引擎**, 不是同不同一台机器 —— 一开始我按 hostname 比, 于是
# 把 ASR_SERVER 指到同机另一个端口 (127.0.0.1:9000 的 vLLM) 被判成"本机的",
# 而那个进程的模型表里根本没有 qwen3-asr。同机不同进程也是别人家的引擎。
#
# 不是自己那个引擎时:
#   - 它不进 AUDIO_MODELS (否则每次配音都要为一个本机引擎没有的模型发一次卸载)
#   - 听写开工前也不卸本机的模型 (那台/那个进程的显存不归我们管, 卸了不省任何东西,
#     只让下一次配音白付一次重载 —— 和 engines_share_a_gpu() 对生图的判断同一个理由)
ASR_IS_REMOTE = ASR_SERVER.rstrip("/") != AUDIO_SERVER.rstrip("/")

# 生图接别人家的标准 API。设了它就走那条路, SD_SERVER 一眼都不看。
#
# 这里和听写那边形状不同, 是故意的: 听写两边说的是同一套端点
# (/v1/audio/transcriptions), 所以一个 ASR_SERVER 就够, 地址不同即"别人家的"。
# 生图不是 —— 我们自己的引擎说的是 sd.cpp 的 /sdcpp/v1/img_gen + 轮询, 标准 API 说的是
# /v1/images/generations, 两套协议。所以这个变量的职责是**选协议**, 不只是给地址,
# 那它就必须是单独一个, 不能靠"和 SD_SERVER 不一样"推出来。
IMAGE_API_SERVER = _env("IMAGE_API_SERVER", "")
IMAGE_API_KEY = _env("IMAGE_API_KEY", "")
IMAGE_API_MODEL = _env("IMAGE_API_MODEL", "gpt-image-1")
# 标准 API 的尺寸是一份固定枚举, 各家不同, 所以是配置不是常量。
# 请求的尺寸会被换成这里面**长宽比最接近**的那一个, 落盘前再由 _fit_size 修回你要的
# 尺寸 (那个函数本来就在干这件事: 引擎也会把请求吸附到自己的网格上)。
IMAGE_API_SIZES = [x.strip() for x in
                   _env("IMAGE_API_SIZES", "1024x1024,1536x1024,1024x1536").split(",") if x.strip()]
IMAGE_VIA_API = bool(IMAGE_API_SERVER)

MUSIC_MODEL_ID = _env("MUSIC_MODEL_ID", "stable-audio")
DESIGN_MODEL_ID = _env("DESIGN_MODEL_ID", "qwen3-tts")        # VoiceDesign: 描述 -> 声音
CLONE_MODEL_ID = _env("CLONE_MODEL_ID", "qwen3-tts-base")     # Base: 参考音 -> 声音
ASR_MODEL_ID = _env("ASR_MODEL_ID", "qwen3-asr")              # ASR: 声音 -> 文字
# 进这个集合才受"开工前把不是这件活要用的全卸掉"管 —— 空闲卸载、互斥卸载、退出释放
# 三件事都是从这里派生的。ASR 实测常驻 3.05 GB, 低于生图的 6.80 GiB 峰值, 所以
# "峰值 = 单个最大模型"这条不因为多一个模型而变。
AUDIO_MODELS = ({MUSIC_MODEL_ID, DESIGN_MODEL_ID, CLONE_MODEL_ID}
                | (set() if ASR_IS_REMOTE else {ASR_MODEL_ID}))

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

# 空闲卸载不再由本进程做, 改由引擎的 idle_unload_ms 负责 (deploy/audio_server.json.tmpl,
# 默认 120000 ms)。原因不是省一个线程: 本进程的计时器按自己的 last_use 判, 所以任何
# 不经过它的加载 —— 另一个客户端直连引擎、上一代被 SIGKILL 后留下的常驻模型 ——
# 它永远不会去卸。实测过: 直连引擎加载一个模型, continuity 在跑、等满 120s,
# 显存纹丝不动。引擎的计时器按 session 是否真在判, 不在乎是谁加载的。
# 需要模型常驻就把 idle_unload_ms 设成 0。

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

# 定妆图随工具结果一起回传, 让 agent 自己看一眼。
#
# 本插件不带 VLM, 也不打算带: 一个视觉模型要自己的显存, 会毁掉"峰值=单个最大模型"
# 这条性质。而 harness 自己的模型多半已经能看图 —— 那就把图给它, 别再养一个。
# 只有定妆/铸声这类"定完该先确认"的工具回传, subject_image 不回传 (它会被调很多次,
# 每次塞一张图进上下文不划算)。
# 缩到 INLINE_IMAGE_MAX px 再传: 定妆图默认 512, 确认长相够用了。
INLINE_IMAGES = _env("CONTINUITY_INLINE_IMAGES", "1") not in ("0", "false", "False")
INLINE_IMAGE_MAX = int(_env("CONTINUITY_INLINE_IMAGE_MAX", "512"))

# 传输方式。默认 stdio —— dsh 是把本进程当子进程拉起来的, 换掉默认值等于把插件弄坏。
# streamable-http 是给"不由 dsh 拉起"的调用方用的: 一个常驻进程, 多个客户端接上来。
#
# 默认只听 127.0.0.1。这个服务没有任何鉴权, 而它能往本机磁盘写文件、能删 actor/subject,
# 绑到 0.0.0.0 等于把这些交给同网段的任何人 —— 要对外必须自己在前面放反向代理。
# 端口 9030: 9020/9021 是两个引擎, 别撞上。
TRANSPORT = _env("CONTINUITY_TRANSPORT", "stdio")
HTTP_HOST = _env("CONTINUITY_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(_env("CONTINUITY_HTTP_PORT", "9030"))
HTTP_PATH = _env("CONTINUITY_HTTP_PATH", "/mcp")

# 能力开关。显存不够时 continuity-setup 会关掉生图那半而保留音频那半 ——
# 见 preflight.py: 换模型省不下显存 (Q4 与 Q8 实测同为 6.6 GB), 能省的只有"不装"。
ENABLE_IMAGE = _env("CONTINUITY_ENABLE_IMAGE", "1") not in ("0", "false", "False")
ENABLE_AUDIO = _env("CONTINUITY_ENABLE_AUDIO", "1") not in ("0", "false", "False")
