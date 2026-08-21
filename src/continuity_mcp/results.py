# ==========================================
# 结构化返回 —— 给程序看的那一半。
#
# 每个工具都回两份内容, 描述的是同一件事:
#   content            人话 (中文, 带 ⚠️)。读它的是 LLM, 它读到什么就会做什么。
#   structured_content 本文件里的模型。读它的是程序 (HTTP 外壳、批处理脚本)。
#
# 为什么要分开: 早先只有人话, 想拿到文件路径就只能去正则那段中文
# ("图片已生成: /path/x.png")。而那段中文是 prompt 的一部分, 是会被改写的 ——
# 改一次措辞, 下游的正则就静默地匹配不到, 没有任何一层会报错, 拿到的是空路径。
# 提示词的措辞和机器接口的契约本来就不该是同一个东西。
#
# 用 Annotated[CallToolResult, <模型>] 标注返回值, 于是:
#   - 模型的 JSON Schema 会作为 outputSchema 出现在 tools/list 里, 调用方能先看形状;
#   - mcp 每次调用都拿这个模型校验 structured_content, 对不上直接 ToolError ——
#     人话和结构化两份内容漂开时是响的, 不是静默的。
#
# 约定:
#   ok=False 时只保证 ok / error / warnings 三个字段有意义, 其余一律 None/空。
#   所以调用方的第一件事永远是看 ok, 而不是看 path 是不是空字符串。
#   路径一律是本机绝对路径 (和人话里那份是同一个)。
# ==========================================
from pydantic import BaseModel, Field


class Result(BaseModel):
    """公共信封。失败时人话照旧是那段指令式中文, 这里给出无歧义的 ok=False。"""
    ok: bool
    error: str | None = None
    # 人话里每一条 ⚠️ 在这里都有一条对应的字符串 —— 警告不能只存在于散文里。
    warnings: list[str] = Field(default_factory=list)


class ArtifactResult(Result):
    """产出一个文件的工具。"""
    path: str | None = None


class ImageResult(ArtifactResult):
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    clamped: bool = False                 # 请求尺寸超上限, 已被限制到 width/height
    subject: str | None = None            # subject_image 才有
    scene: str | None = None


class SpeechResult(ArtifactResult):
    actor: str | None = None              # actor_tts 才有
    voice: str | None = None              # generate_speech 才有
    truncated: bool = False               # 台词超 max_chars 已截断
    max_chars: int | None = None
    consistent: bool = False              # 音色是否跨句可复现 (actor_tts 真, 旁白假)


class MusicResult(ArtifactResult):
    duration: float | None = None         # 实测时长, 不是请求时长
    requested_duration: float | None = None
    clamped: bool = False                 # 请求秒数超上限, 已被限制
    truncated: bool = False               # 引擎在自己的上限处静默截短了


class SfxResult(ArtifactResult):
    preset: str | None = None
    seed: int | None = None
    duration: float | None = None
    wave: str | None = None
    base_freq: float | None = None


class CutoutResult(ArtifactResult):
    mode_used: str | None = None          # checker / rembg
    model: str | None = None              # rembg 分支用的模型, checker 分支为 None
    transparent_ratio: float | None = None


class SliceResult(Result):
    paths: list[str] = Field(default_factory=list)
    frames: int = 0


class ActorResult(Result):
    """铸声 (create_actor / import_actor) 的结果。"""
    name: str | None = None
    kind: str = "actor"
    reference_path: str | None = None     # 参考音 / 试音片段
    ref_seconds: float | None = None
    transcript: str | None = None
    voice: str | None = None
    seed: int | None = None
    source_format: str | None = None      # import_actor: 原始录音的规格
    imported_from: str | None = None
    truncated: bool = False               # create_actor: 铸声台词被截断


class SubjectResult(Result):
    """定妆 (create_character / create_animal / create_object / import_subject) 的结果。"""
    name: str | None = None
    kind: str | None = None               # character / animal / object
    reference_path: str | None = None     # 定妆图
    appearance: str | None = None
    seed: int | None = None
    source_size: str | None = None        # import_subject: 原图尺寸 "WxH"
    stored_size: str | None = None
    resized: bool = False
    imported_from: str | None = None


class ActorInfo(BaseModel):
    name: str
    voice: str | None = None
    created: str | None = None
    ref_seconds: float | None = None
    reference_path: str | None = None


class ActorListResult(Result):
    count: int = 0
    actors: list[ActorInfo] = Field(default_factory=list)


class SubjectInfo(BaseModel):
    name: str
    kind: str | None = None
    appearance: str | None = None
    created: str | None = None
    reference_path: str | None = None


class SubjectListResult(Result):
    count: int = 0
    subjects: list[SubjectInfo] = Field(default_factory=list)


class DeleteResult(Result):
    name: str | None = None
    kind: str | None = None
    files_removed: int = 0


class EngineInfo(BaseModel):
    url: str
    enabled: bool                         # 这半边能力装没装 (显存不够时会只装音频那半)
    reachable: bool                       # 现在连不连得上


class StatusResult(Result):
    engines_ok: bool = False
    engines_down: list[str] = Field(default_factory=list)
    image: EngineInfo | None = None
    audio: EngineInfo | None = None
    state_dir: str | None = None
    actors: int = 0
    subjects: int = 0
    cutout_quality: str | None = None
    idle_unload_s: float | None = None    # <=0 表示模型常驻
    setup_needed: bool = False            # 引擎连不上且看起来没跑过 continuity-setup
