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
#   ok=False 时只保证 ok / error / error_code / warnings 四个字段有意义, 其余一律 None/空。
#   所以调用方的第一件事永远是看 ok, 而不是看 path 是不是空字符串。
#   路径一律是本机绝对路径 (和人话里那份是同一个)。
# ==========================================
from pydantic import BaseModel, Field, model_validator

from .errors import CONFLICT, ENGINE_ERROR, INVALID, NOT_FOUND

ERROR_CODES = (INVALID, NOT_FOUND, CONFLICT, ENGINE_ERROR)


class Result(BaseModel):
    """公共信封。失败时人话照旧是那段指令式中文, 这里给出无歧义的 ok=False。"""
    ok: bool
    error: str | None = None
    # 失败的机器可读分类, 取值只有 ERROR_CODES 那四个。
    # error 那段中文是 prompt 的一部分 (会被改写), 所以它不能是调用方判断失败类型的依据 ——
    # HTTP 外壳要把失败还原成 400/404/409/500, 它读的是这个字段。
    error_code: str | None = None
    # 人话里每一条 ⚠️ 在这里都有一条对应的字符串 —— 警告不能只存在于散文里。
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _code_matches_ok(self):
        """ok 和 error_code 必须互相自洽 —— 不自洽是响的, 不是静默的。

        (和 outputSchema 校验同一个道理: 漏填的那一次要当场炸, 而不是让外壳拿到一个
        ok=False 却没有 code 的信封, 再退回去正则那段中文。)
        """
        if self.ok:
            if self.error_code is not None:
                raise ValueError(f"ok=True 不该带 error_code ({self.error_code})")
        elif self.error_code not in ERROR_CODES:
            raise ValueError(f"ok=False 必须给 error_code, 且取值属于 {ERROR_CODES} "
                             f"(拿到 {self.error_code!r})")
        return self


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
    # int | float 而不是 float: preset 里的基频写的是整数 (select 是 880), 标成 float
    # 会被 pydantic 强转成 880.0, 于是同一个数在这里和 params 里长得不一样。
    base_freq: int | float | None = None
    # sfx.sfx_params() 算出来的完整参数对象 (SfxParams 的全部字段, 扁平 dict)。
    # 顶层只挑了 wave/base_freq 两个 —— 想复现或微调这枚音效的调用方要的是全部,
    # 而它本来就在手上, 只是以前没往外给。
    params: dict | None = None


class CutoutResult(ArtifactResult):
    mode_used: str | None = None          # checker / rembg
    model: str | None = None              # rembg 分支用的模型, checker 分支为 None
    transparent_ratio: float | None = None
    # 抠图质量报告: transparent_ratio / solid_ratio / hole_ratio /
    # largest_blob_ratio / bg_detail_ratio。警告那句话就是从这几个数得出来的 ——
    # 只给结论不给依据, 调用方就没法自己定阈值。
    metrics: dict | None = None
    # 走没走 checker 分支的结构证据 (cand_ratio / tones / cell / reason ...);
    # mode=rembg 时没有取证, 为 None。
    checker_evidence: dict | None = None


class SfxPresetsResult(Result):
    """gen_sfx 能用的 preset 名单 + 全部可覆盖字段的默认值。"""
    presets: list[str] = Field(default_factory=list)
    params: dict | None = None            # SfxParams 的全部字段及其默认值
    rate: int | None = None               # 采样率 Hz


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


class TranscriptResult(Result):
    """听写 (transcribe) 的结果。"""
    text: str | None = None
    language: str | None = None           # 请求时指定的语种; 没指定就是 None (引擎自己判)
    audio_seconds: float | None = None    # 引擎报的音频时长
    source: str | None = None             # 听的是哪个文件 (本机绝对路径)


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
