# ==========================================
# MCP server —— dsh 通过 stdio 拉起本进程, 工具以 mcp__continuity__<name> 出现。
#
# 工具的 docstring 就是产品本身: 调用方是 LLM, 它读到什么就会做什么。所以这里的
# 说明写的是"为什么要有这一步"和"哪些信息必须写死", 而不是参数表 —— 参数表它看得懂,
# "不定妆就会得到三个不同的人"它看不出来。
#
# 返回的是本机绝对路径, 不是 URL: agent 拿到之后要把文件放进游戏工程里, 路径直接可用。
#
# 每个工具回两份内容: 人话 (content) 给 LLM 看, 结构化 (structured_content, 模型在
# results.py) 给程序看。别去正则那段中文取路径 —— 措辞是会改的, 而正则失配是静默的。
# ==========================================
import asyncio
import atexit
import signal
import argparse
import base64
import io
import logging
import os
import sys
import threading
import time
from typing import Annotated

from mcp.server.mcpserver import Image, MCPServer
from mcp_types import CallToolResult, TextContent
from PIL import Image as PILImage

from . import cutout, engines, jobs, results, sfx, store
from .config import (GENERATED_DIR, ACTORS_DIR, SUBJECTS_DIR, STATE_DIR, ENABLE_IMAGE,
                     ENABLE_AUDIO, DEFAULT_CUTOUT_QUALITY, RETENTION_DAYS, CLEANUP_INTERVAL_S,
                     MAX_SPEECH_CHARS, AUDIO_IDLE_UNLOAD_S, SD_SERVER, AUDIO_SERVER,
                     INLINE_IMAGES, INLINE_IMAGE_MAX, TRANSPORT, HTTP_HOST, HTTP_PORT, HTTP_PATH)

logging.basicConfig(level=os.getenv("CONTINUITY_LOG_LEVEL", "INFO"), stream=sys.stderr)
log = logging.getLogger("continuity")

mcp = MCPServer("continuity")


def _res(_model, _text, _image=None, **fields):
    """把一次调用的两份表述打包成一个 CallToolResult。

    _text   原样是以前那个返回值 —— 措辞一个字没改, LLM 读到的东西不变。
    fields  结构化那份。mcp 会拿返回标注里的模型校验它, 字段名写错/类型不对当场报错,
            所以两份内容漂开是响的。
    _image  要内联的定妆图路径 (只有定妆/导入那四个工具给)。

    前三个形参一律带下划线前缀: 它们和 **fields 共用一个命名空间, 而 fields 的键是
    模型字段名。remove_bg 有个字段就叫 model —— 撞上时报的是
    "got multiple values for argument 'model'", 跟结构化返回半点关系没有。
    pydantic 字段不允许下划线开头, 所以加前缀之后这类碰撞不可能再发生。
    """
    content = [TextContent(type="text", text=_text)]
    if _image is not None:
        block = _image_block(_image)
        if block is not None:
            content.append(block)
    return CallToolResult(content=content,
                          structured_content=_model(**fields).model_dump(mode="json"))


def _fail(_model, prefix, e):
    """失败: 人话照旧是那句指令式中文, 结构化那份给出无歧义的 ok=False。"""
    return _res(_model, f"{prefix}: {e}", ok=False, error=str(e))


def _image_block(path):
    """把定妆图和说明一起回传 —— 说明"先看一眼"而不给图, 那句话是没法照做的。

    缩过再传: 一张 512px PNG 已经足够确认长相, 而原图可能是 1024。
    关掉: CONTINUITY_INLINE_IMAGES=0 (模型不认图片内容时)。
    """
    if not INLINE_IMAGES:
        return None
    try:
        img = PILImage.open(path)
        if max(img.size) > INLINE_IMAGE_MAX:
            k = INLINE_IMAGE_MAX / max(img.size)
            img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))),
                             PILImage.LANCZOS)
        buf = io.BytesIO()
        # JPEG 而不是 PNG: 这一份只是给模型看一眼, 存下来的那张 PNG 原样不动。
        # 实测 512px 定妆图 PNG 约 200 KB, JPEG q85 约 40 KB —— 进上下文的东西便宜五倍,
        # 而"这是不是我要的那个人"根本不需要无损。
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return Image(data=buf.getvalue(), format="jpeg").to_image_content()
    except Exception:
        log.warning("定妆图内联失败, 只回文字", exc_info=True)
        return None


async def _go(fn, *a, **kw):
    """把同步的 GPU 活丢到线程里, 别堵住 MCP 的事件循环。"""
    return await asyncio.to_thread(fn, *a, **kw)


# =========================================================
# 铸声 —— 让同一个 NPC 每句台词都是同一个人
# =========================================================

if ENABLE_AUDIO:

    @mcp.tool()
    async def create_actor(name: str, voice: str, sample_text: str = None, seed: int = None,
                           force: bool = False) -> Annotated[CallToolResult, results.ActorResult]:
        """给一个角色铸声(定妆), 之后用 actor_tts 让他说任意台词都保持同一个音色。

        为什么要有这一步: 直接用文字描述合成语音, 那段描述只圈定一个大致的音色区间,
        区间内每句台词各漂各的 —— 实测同 voice 同 seed 四句台词基频极差 125 Hz,
        关掉采样走贪心反而涨到 242 Hz(音色是文本的函数, 不是采样随机性, 锁 seed 或
        temperature 都锁不住)。本工具先用 voice 描述生成一段参考音, 之后所有台词
        改由克隆模型照着这段参考音说, 音色与台词内容无关。实测极差降到 5 Hz。

        重要: 铸完请先听那段试音, 确认是不是你要的那个人。铸砸了会把整个角色锁死在
        错的音色上, 而且它之后每一句都错得很一致。不满意就 force=True 重铸。

        参数:
            name: 角色名(字母/数字/下划线/连字符/中文, 1~40 字), 之后 actor_tts 用它指代
            voice: 声音的自然语言描述, 英文效果最佳。写年龄/性别/音色/语速/情绪,
                   例如 "An elderly Chinese man, gravelly chest voice, commanding"
            sample_text: 铸声用的台词(可选)。默认用一段覆盖面较广的中文
            seed: 随机种子(可选)
            force: 覆盖已有角色。会让该角色之前所有台词的音色对不上, 慎用

        返回: 试音片段的本机路径 —— 先听再用
        """
        try:
            m = await _go(jobs.create_actor, name, voice, sample_text, seed, force)
        except Exception as e:
            return _fail(results.ActorResult, "铸声失败", e)
        warns = []
        if m.get("clipped"):
            warns.append("铸声台词已截断到 %d 字 —— 参考音越长, 之后每一句台词的显存代价越高。"
                         % len(m["transcript"]))
        for k in ("too_long", "remote_engine"):
            if m.get(k):
                warns.append(m[k])
        clip = "".join(f"\n⚠️ {w}" for w in warns)
        return _res(results.ActorResult,
                    f"角色 '{m['name']}' 已铸声 (参考音 {m['ref_seconds']:.1f}s)。\n"
                    f"试音: {m['reference_path']}\n"
                    f"(念的是: {m['transcript']}){clip}\n"
                    f"先听一遍确认是不是你要的人, 不满意用 create_actor(..., force=True) 重铸。",
                    ok=True, warnings=warns, name=m["name"], voice=m.get("voice"),
                    reference_path=m["reference_path"], ref_seconds=m["ref_seconds"],
                    transcript=m["transcript"], seed=m.get("seed"),
                    truncated=bool(m.get("clipped")))

    @mcp.tool()
    async def import_actor(name: str, audio_path: str, transcript: str, force: bool = False
                           ) -> Annotated[CallToolResult, results.ActorResult]:
        """用一段现成的录音铸声 —— 声音是别处做的(真人录音 / 其它 TTS)也照样能保持一致。

        和 create_actor 得到的东西完全一样, 只是参考音由你提供而不是我们生成。之后
        actor_tts 让他说任意台词, 音色都来自这段录音。

        参数:
            name: 角色名, 之后 actor_tts 用它指代
            audio_path: 录音的本机路径。16-bit PCM WAV, 2~30 秒, 单人清唱式的干净人声
                        最好(没有背景音乐和混响)。采样率/声道数会自动转成 24 kHz 单声道。
                        其它格式先转: ffmpeg -i 原文件 -acodec pcm_s16le -ac 1 -ar 24000 ref.wav
            transcript: 那段录音里念的是什么 —— 必填。克隆模型要拿它对齐音频和文字,
                        写错了音色会明显不对。
            force: 覆盖已有角色

        注意: 你有权使用这段声音才导入它。克隆一个真人的嗓子在很多地方是需要本人同意的。
        """
        try:
            m = await _go(jobs.import_actor, name, audio_path, transcript, force)
        except Exception as e:
            return _fail(results.ActorResult, "导入失败", e)
        warns = [m["lowband"]] if m.get("lowband") else []
        warn = f"\n⚠️ {m['lowband']}" if m.get("lowband") else ""
        return _res(results.ActorResult,
                    f"角色 '{m['name']}' 已从录音铸声 ({m['source_format']})。\n"
                    f"参考音: {m['reference_path']}\n"
                    f"先用 actor_tts 试一句, 确认克隆出来的音色对不对。{warn}",
                    ok=True, warnings=warns, name=m["name"], voice=m.get("voice"),
                    reference_path=m["reference_path"], ref_seconds=m.get("ref_seconds"),
                    transcript=m.get("transcript"), source_format=m.get("source_format"),
                    imported_from=m.get("imported_from"))

    @mcp.tool()
    async def actor_tts(actor: str, text: str, speaking_rate: float = None, seed: int = None
                        ) -> Annotated[CallToolResult, results.SpeechResult]:
        """让某个已铸声的角色说一句台词, 音色与他之前每一句都一致。

        做游戏 NPC 对白用这个, 不要用 generate_speech —— 后者每句音色会漂。
        角色不存在会告诉你先去 create_actor。

        参数:
            actor: 角色名(create_actor 时定的)
            text: 台词, 上限 200 字(约 45 秒), 超出截断
            speaking_rate: 语速倍率(可选)
            seed: 随机种子(可选)

        返回: 24 kHz 单声道 WAV 的本机路径
        """
        try:
            r = await _go(jobs.speak, text, actor, None, seed, speaking_rate)
        except Exception as e:
            return _fail(results.SpeechResult, "配音失败", e)
        w = (f"台词超过 {MAX_SPEECH_CHARS} 字已截断, 长文本请分多次调用。") if r["clipped"] else None
        tail = f"\n⚠️ {w}" if w else ""
        return _res(results.SpeechResult, f"{actor} 的台词已生成: {r['path']}{tail}",
                    ok=True, warnings=[w] if w else [], path=r["path"], actor=actor,
                    truncated=r["clipped"], max_chars=MAX_SPEECH_CHARS, consistent=True)

    @mcp.tool()
    async def list_actors() -> Annotated[CallToolResult, results.ActorListResult]:
        """列出已铸声的角色(名字 + 当初的声音描述 + 铸声时间)。"""
        actors = store.listing("actor")
        info = [results.ActorInfo(name=a["name"], voice=a.get("voice"), created=a.get("created"),
                                  ref_seconds=a.get("ref_seconds"),
                                  reference_path=a.get("reference_path")) for a in actors]
        if not actors:
            return _res(results.ActorListResult,
                        "还没有角色。用 create_actor(name='...', voice='...') 铸一个。",
                        ok=True, count=0, actors=[])
        return _res(results.ActorListResult,
                    "\n".join(f"- {a['name']}: {a['voice']} (铸于 {a['created']})" for a in actors),
                    ok=True, count=len(info), actors=info)

    @mcp.tool()
    async def delete_actor(name: str) -> Annotated[CallToolResult, results.DeleteResult]:
        """删掉一个已铸声的角色。不可逆: 参考音不可复现, 重铸出来是另一个人。"""
        try:
            n = store.drop(store.actor_paths(name), "actor", name)
        except Exception as e:
            return _fail(results.DeleteResult, "删除失败", e)
        return _res(results.DeleteResult, f"已删除 actor '{name}' ({n} 个文件)",
                    ok=True, name=name, kind="actor", files_removed=n)

    @mcp.tool()
    async def generate_speech(text: str, voice: str = None, seed: int = None,
                              speaking_rate: float = None
                              ) -> Annotated[CallToolResult, results.SpeechResult]:
        """一次性旁白 —— 不保证跨句音色一致, 会重复出现的角色请用 create_actor + actor_tts。

        适合系统提示音、一次性播报这类"说完就没了"的语音。任何会说第二句的角色都不该
        用这个: 同一段 voice 描述的两句台词不是同一个人。

        参数:
            text: 要念的台词, 上限 200 字
            voice: 声音的自然语言描述(英文效果最佳), 不传则用中性旁白嗓
            seed / speaking_rate: 可选

        返回: 24 kHz 单声道 WAV 的本机路径
        """
        try:
            r = await _go(jobs.speak, text, None, voice, seed, speaking_rate)
        except Exception as e:
            return _fail(results.SpeechResult, "语音生成失败", e)
        w = f"台词超过 {MAX_SPEECH_CHARS} 字已截断。" if r["clipped"] else None
        tail = f"\n⚠️ {w}" if w else ""
        return _res(results.SpeechResult,
                    f"语音已生成: {r['path']}{tail}\n"
                    f"提示: 这个音色不会在下一次调用中重现。角色请用 create_actor。",
                    ok=True, warnings=[w] if w else [], path=r["path"], voice=voice,
                    truncated=r["clipped"], max_chars=MAX_SPEECH_CHARS, consistent=False)

    @mcp.tool()
    async def generate_music(prompt: str, seed: int = None, duration: float = 30.0,
                             num_inference_steps: int = 100
                             ) -> Annotated[CallToolResult, results.MusicResult]:
        """生成一段背景音乐(Stable Audio), 返回 WAV 的本机路径。

        参数:
            prompt: 音乐描述(风格/乐器/情绪, 英文效果最佳)
            seed: 随机种子(可选)
            duration: 秒数, 上限 120
            num_inference_steps: 推理步数(默认 100, 越多越慢)

        注意: 出来的是一段有头有尾的音乐, 没有做无缝循环点。要循环播放的 BGM
        请自己在编辑器里找循环点, 或把它当作一次性过场音乐。
        """
        try:
            r = await _go(jobs.generate_music, prompt, seed, duration, num_inference_steps)
        except Exception as e:
            return _fail(results.MusicResult, "音乐生成失败", e)
        tail = f" (请求被限制到 {r['clamped']['duration']:.0f}s)" if r["clamped"] else ""
        warns = []
        if r.get("truncated"):
            want, got = r["truncated"]
            warns.append(f"请求 {want:.0f}s, 引擎只给了 {got:.0f}s —— 它在自己的上限处"
                         f"静默截断且不报错。需要更长就分段生成。")
            tail += f"\n⚠️ {warns[0]}"
        return _res(results.MusicResult,
                    f"音乐已生成: {r['path']} ({r['duration']:.0f}s){tail}",
                    ok=True, warnings=warns, path=r["path"], duration=r["duration"],
                    requested_duration=duration, clamped=bool(r["clamped"]),
                    truncated=bool(r.get("truncated")))


# =========================================================
# 定妆 —— 让同一个角色/物件每张图都长一个样
# =========================================================

if ENABLE_IMAGE:

    async def _pin(name, appearance, kind, width, height, seed, force, label):
        try:
            m = await _go(jobs.create_subject, name, appearance, kind, width, height, seed, force)
        except Exception as e:
            return _fail(results.SubjectResult, "定妆失败", e)
        return _res(results.SubjectResult,
                    f"{label} '{m['name']}' 已定妆。\n定妆图: {m['reference_path']}\n"
                    f"先看一眼确认是不是你要的, 不满意用 force=True 重定。",
                    _image=m["reference_path"],
                    ok=True, name=m["name"], kind=m["kind"], appearance=m["appearance"],
                    reference_path=m["reference_path"], seed=m.get("seed"))

    # 这四个工具回的是 [文字, 定妆图] 两块内容, 所以不能标 `-> str` —— mcp 2.0 会按返回
    # 标注建结构化输出模型, 标了 str 就只能回一个字符串。原先的办法是索性不标注, 代价是
    # 它们连 outputSchema 都没有。改标 Annotated[CallToolResult, 模型]: 自己造
    # CallToolResult 就能同时塞图片块和 structured_content, 模型照样进 outputSchema
    # 并逐次校验。图片没丢, 结构化也有了。
    @mcp.tool()
    async def create_character(name: str, appearance: str, width: int = 512, height: int = 512,
                               seed: int = None, force: bool = False
                               ) -> Annotated[CallToolResult, results.SubjectResult]:
        """给一个人物定妆(生成并存下参考图), 之后 subject_image 出的每张图长相都一致。

        为什么要有这一步: generate_image 每次给的是"长得不一样的人"。同一个角色的头像 /
        战斗立绘 / 地图小人, 直接用文字描述生成出来是三个人。

        appearance 分两部分, 分清楚很重要:

        (1) 身份 —— 必须写死, 漏掉的每一项模型都会自己编, 而且每张编得不一样:
          - 年龄段 + 体型(高瘦/魁梧/矮壮)
          - 脸: 脸型、显著特征(疤/须/眉眼)
          - 发型 + 发色 + 束发方式
          - 辨识物: 跟着这个人走、换装也不摘的东西(独眼罩/佩剑/护腕/胎记)

        (2) 默认服装 —— 只是个基线, 不是身份的一部分。照样写进 appearance,
            但 subject_image 的 scene 里写新衣服就能换掉(实测: 定妆穿布袍, scene 写
            "wearing heavy red armor" 能换成甲胄而脸不变)。所以一个角色不需要按套装
            定妆很多次。

        不要写场景、动作、表情 —— 那些留给 subject_image 的 scene。

        定完先看返回的定妆图确认是不是你要的人; 定砸了会把整个角色锁死在错的长相上,
        不满意就 force=True 重定。
        """
        return await _pin(name, appearance, "character", width, height, seed, force, "人物")

    @mcp.tool()
    async def create_animal(name: str, appearance: str, width: int = 512, height: int = 512,
                            seed: int = None, force: bool = False
                            ) -> Annotated[CallToolResult, results.SubjectResult]:
        """给一只动物/坐骑/灵兽定妆, 之后每张图它都是同一只。

        appearance 里必须写死这几样:
          - 物种 + 体型比例(腿长/身长/头身比)
          - 毛色/羽色 + 花纹的分布位置(不是只说"有斑点", 要说斑点在哪)
          - 耳朵、尾巴、翅膀的形状
          - 显著特征(独角/断尾/眼色)
        鞍具、缰绳这类可穿卸的东西和人物的服装同理: 写进 appearance 只是默认值,
        scene 里可以换掉。不要写场景和动作。

        定完先看定妆图, 不满意 force=True 重定。
        """
        return await _pin(name, appearance, "animal", width, height, seed, force, "动物")

    @mcp.tool()
    async def create_object(name: str, appearance: str, width: int = 512, height: int = 512,
                            seed: int = None, force: bool = False
                            ) -> Annotated[CallToolResult, results.SubjectResult]:
        """给一件道具/物件定妆, 之后每张图它都长一个样, 换角度也不变。

        物件最容易漂的是**几何**, 不是材质配色 —— 实测一个宝箱, 材质配色五金件都对得上,
        盖子却一会儿是平的方的、一会儿是拱的圆的, 因为原始描述里压根没写盖子什么形状。

        appearance 里必须写死这几样:
          - 整体轮廓 + 比例(长方/立方/圆桶, 宽高比)
          - 关键几何: 盖子平的还是拱的、边角方的还是圆的、侧面直的还是弧的、有没有底座
          - 材质 + 主次配色
          - 五金件/纹饰及其位置(锁扣、包角、铆钉在哪)
        不要写场景和角度 —— 角度留给 subject_image 的 scene。

        定完先看定妆图, 不满意 force=True 重定。
        """
        return await _pin(name, appearance, "object", width, height, seed, force, "物件")

    @mcp.tool()
    async def import_subject(name: str, image_path: str, appearance: str,
                             kind: str = "character", force: bool = False
                             ) -> Annotated[CallToolResult, results.SubjectResult]:
        """用一张现成的图定妆 —— 角色/物件是别处画的也照样能保持一致。

        和 create_character / create_object 得到的东西完全一样, 只是定妆图由你提供。
        之后 subject_image 让它出任意场景图, 外观都来自这张图。

        参数:
            name: 名字, 之后 subject_image 用它指代
            image_path: 参考图的本机路径。要求和我们自己生成的定妆图一样: 单个主体、
                        正面或四分之三视角、背景干净、看得全。一张有场景有动作的插画
                        当参考图, 场景会跟着一起被复制过去。
            appearance: 这是什么的文字描述 —— 必填, 它会被拼进之后每一张场景图的提示词。
                        只给参考图而不给描述, 模型对"这是什么"没有着落, 外观照样会漂。
                        写法同 create_character / create_animal / create_object 的要求。
            kind: character / animal / object
            force: 覆盖已有的

        大图会缩到 1024 以内; 带透明通道的图会转成 RGB(透明区交给引擎会变成黑块)。
        """
        try:
            m = await _go(jobs.import_subject, name, image_path, appearance, kind, force)
        except Exception as e:
            return _fail(results.SubjectResult, "导入失败", e)
        size = (f"原图 {m['source_size']} → 存为 {m['stored_size']}"
                if m["resized"] else f"{m['source_size']}")
        return _res(results.SubjectResult,
                    f"{m['kind']} '{m['name']}' 已用现成图定妆 ({size})。\n"
                    f"定妆图: {m['reference_path']}\n"
                    f"先用 subject_image 出一张试试, 确认外观跟得住。",
                    _image=m["reference_path"],
                    ok=True, name=m["name"], kind=m["kind"], appearance=m["appearance"],
                    reference_path=m["reference_path"], source_size=m["source_size"],
                    stored_size=m["stored_size"], resized=m["resized"],
                    imported_from=m.get("imported_from"))

    @mcp.tool()
    async def subject_image(subject: str, scene: str, width: int = 512, height: int = 512,
                            seed: int = None) -> Annotated[CallToolResult, results.ImageResult]:
        """让某个已定妆的角色或物件出一张新图, 外观与它之前每一张都一致。

        做游戏素材用这个, 不要用 generate_image —— 后者每张长相会变。
        subject 不存在会告诉你先去 create_character / create_animal / create_object。

        参数:
            subject: 名字(定妆时定的)
            scene: 这张图里它在干什么 / 在哪 / 什么角度 —— 只写场景动作视角,
                   身份由定妆图决定。例如 "opened, seen from behind, on a stone floor"。
                   人物/动物还可以在这里换装: "wearing heavy red armor" 会换掉定妆图
                   里那身衣服而保住脸。
            width/height: 上限 1024
            seed: 随机种子(可选)

        返回: 本机路径
        """
        try:
            r = await _go(jobs.subject_image, subject, scene, width, height, seed)
        except Exception as e:
            return _fail(results.ImageResult, "出图失败", e)
        return _res(results.ImageResult, f"{subject} 的新图已生成: {r['path']}",
                    ok=True, path=r["path"], width=r["width"], height=r["height"], seed=seed,
                    clamped=bool(r["clamped"]), subject=subject, scene=scene)

    @mcp.tool()
    async def list_subjects() -> Annotated[CallToolResult, results.SubjectListResult]:
        """列出已定妆的角色、动物和物件。"""
        subs = store.listing("subject")
        info = [results.SubjectInfo(name=s["name"], kind=s.get("kind"),
                                    appearance=s.get("appearance"), created=s.get("created"),
                                    reference_path=s.get("reference_path")) for s in subs]
        if not subs:
            return _res(results.SubjectListResult,
                        "还没有定妆过的东西。用 create_character / create_animal / "
                        "create_object 定一个。", ok=True, count=0, subjects=[])
        return _res(results.SubjectListResult,
                    "\n".join(f"- [{s.get('kind')}] {s['name']}: {s['appearance']} "
                              f"(定于 {s['created']})" for s in subs),
                    ok=True, count=len(info), subjects=info)

    @mcp.tool()
    async def delete_subject(name: str) -> Annotated[CallToolResult, results.DeleteResult]:
        """删掉一个已定妆的角色或物件。不可逆: 定妆图不可复现, 重定出来是另一个。"""
        try:
            n = store.drop(store.subject_paths(name), "subject", name)
        except Exception as e:
            return _fail(results.DeleteResult, "删除失败", e)
        return _res(results.DeleteResult, f"已删除 subject '{name}' ({n} 个文件)",
                    ok=True, name=name, kind="subject", files_removed=n)

    @mcp.tool()
    async def generate_image(prompt: str, width: int = 1024, height: int = 1024, seed: int = None,
                             reference_image_path: str = None
                             ) -> Annotated[CallToolResult, results.ImageResult]:
        """生成一张一次性图片 —— 不保证与任何其它图一致。

        会重复出现的角色/物件请先 create_character / create_object 定妆, 再用
        subject_image 出图。这个工具适合背景板、UI 底图这类只出现一次的东西。

        参数:
            prompt: 图片描述(英文效果最佳)
            width/height: 上限 1024
            seed: 随机种子(可选)
            reference_image_path: 参考图的本机路径(可选), 传了就是图生图

        返回: 本机路径
        """
        ref = None
        if reference_image_path:
            try:
                with open(reference_image_path, "rb") as f:
                    ref = base64.b64encode(f.read()).decode()
            except OSError as e:
                return _fail(results.ImageResult, "读不到参考图", e)
        try:
            r = await _go(jobs.generate_image, prompt, width, height, seed, ref)
        except Exception as e:
            return _fail(results.ImageResult, "图片生成失败", e)
        tail = f" (尺寸被限制到 {r['clamped']['width']}x{r['clamped']['height']})" if r["clamped"] else ""
        return _res(results.ImageResult,
                    f"图片已生成: {r['path']}{tail}\n"
                    f"提示: 这张图里的东西不会在下一次调用中重现。角色/物件请用定妆流程。",
                    ok=True, path=r["path"], width=r["width"], height=r["height"], seed=seed,
                    clamped=bool(r["clamped"]))


# =========================================================
# 后处理 —— 不碰 GPU
# =========================================================

def _open_image(path=None, b64=None):
    if path:
        return PILImage.open(path)
    if b64:
        return PILImage.open(io.BytesIO(base64.b64decode(b64)))
    raise ValueError("必须提供 image_path 或 image_base64")


@mcp.tool()
async def remove_bg(image_path: str = None, image_base64: str = None, mode: str = "auto",
                    quality: str = None) -> Annotated[CallToolResult, results.CutoutResult]:
    """抠掉图片背景, 输出真正带 alpha 通道的 RGBA PNG。

    生图模型画不出 alpha: 你让它画"透明背景", 它是把 PS 那种灰白棋盘格当成不透明
    像素画出来的。做游戏精灵图必须用本工具把它转成真的 RGBA。

    参数:
        image_path: 图片的本机路径(其它工具返回的路径可直接用)
        image_base64: 或者直接给 base64
        mode: auto (默认, 按结构证据判断是不是棋盘格: 恰好两级灰度 + 周期方格; 不是就走
              通用抠图) / checker (强制只抠棋盘格) / rembg (强制通用显著物体抠图, CPU)
        quality: best (birefnet-general-lite, ~7s, 峰值内存 ~6.8 GB) /
                 fast (u2netp, ~0.6s, 峰值 ~1.3 GB)。只影响 rembg 分支。
                 不传则用安装时按本机内存定下的默认值。

    返回: RGBA PNG 的本机路径, 附带实际走的分支与透明像素占比。抠出来明显不对
    (几乎全透明 / 几乎没抠掉 / 碎成一堆小块 / 主体被啃出洞) 时会附一行 ⚠️ 警告 ——
    那种结果别直接用。
    """
    try:
        img = _open_image(image_path, image_base64)
        path, data = await asyncio.to_thread(
            cutout.remove_bg, img, mode, quality or DEFAULT_CUTOUT_QUALITY,
            jobs._new_name("cut", "png"), GENERATED_DIR)
    except Exception as e:
        return _fail(results.CutoutResult, "抠图失败", e)
    model = f"/{data['model']}" if data.get("model") else ""
    out = (f"背景已移除: {path} (分支 {data['mode_used']}{model}, "
           f"透明像素占比 {data['transparent_ratio']:.1%})")
    warns = [data["warning"]] if data.get("warning") else []
    if data.get("warning"):
        out += f"\n⚠️ {data['warning']}"
    return _res(results.CutoutResult, out, ok=True, warnings=warns, path=str(path),
                mode_used=data["mode_used"], model=data.get("model"),
                transparent_ratio=data["transparent_ratio"])


@mcp.tool()
async def slice_sheet(image_path: str = None, image_base64: str = None,
                      rows: int = None, cols: int = None, frame_width: int = None,
                      frame_height: int = None, trim: bool = True
                      ) -> Annotated[CallToolResult, results.SliceResult]:
    """把排成网格的 sprite sheet 切成单帧 PNG。

    让生图模型画"4 帧动画"时它会摆成 2x2 网格而不是 4 张图, 用本工具切开。

    参数:
        image_path / image_base64: 同其它图片工具
        rows, cols: 网格行列数(二者都给)
        frame_width, frame_height: 单帧像素尺寸(与 rows+cols 二选一)
        trim: 是否把每帧裁到非透明/非背景的外接框(默认 True)

    返回: 各帧 PNG 的本机路径
    """
    try:
        img = _open_image(image_path, image_base64)
        paths = await asyncio.to_thread(
            cutout.slice_sheet, img, rows, cols, frame_width, frame_height, trim,
            lambda: jobs._new_name("frame", "png"), GENERATED_DIR)
    except Exception as e:
        return _fail(results.SliceResult, "切图失败", e)
    return _res(results.SliceResult,
                f"已切出 {len(paths)} 帧:\n" + "\n".join(str(p) for p in paths),
                ok=True, paths=[str(p) for p in paths], frames=len(paths))


@mcp.tool()
async def gen_sfx(preset: str = "select", seed: int = None, base_freq: float = None,
                  wave: str = None, overrides: dict = None
                  ) -> Annotated[CallToolResult, results.SfxResult]:
    """合成一枚 sfxr/jsfxr 风格的游戏音效(纯程序化, 不用模型), 返回 WAV 路径。

    不要用 generate_music 做音效: 那是扩散模型, 出来的是几十秒的宽带糊音。游戏音效
    是 10~200ms 的瞬态, 要精确、即时、可复现 —— 本工具毫秒级出结果, 同 seed 逐字节可复现,
    而且完全不占显存。

    参数:
        preset: jump / coin / hit / explosion / powerup / laser / select / hurt
        seed: 随机种子。给了就在 preset 周围抖动参数(不是抖采样点), 同 seed 结果完全相同;
              不给则严格使用 preset 的原始参数。
        base_freq: 覆盖基频 Hz (noise 波形下是采样保持的刷新率)
        wave: 覆盖波形 square / saw / sine / triangle / noise
        overrides: 覆盖任意合成参数的字典, 例如
                   {"freq_slide": -3.0, "release": 0.4, "lpf": 0.5, "duty": 0.25}

    返回: 44.1kHz 16bit 单声道 WAV 的本机路径
    """
    ov = dict(overrides or {})
    if base_freq is not None:
        ov["base_freq"] = base_freq
    if wave is not None:
        ov["wave"] = wave
    try:
        p, rng = sfx.sfx_params(preset, seed, ov or None)
        x = sfx.sfx_render(p, rng)
    except Exception as e:
        return _fail(results.SfxResult, "音效合成失败", e)
    path = GENERATED_DIR / jobs._new_name("sfx", "wav")
    sfx.write_wav(str(path), x)
    return _res(results.SfxResult,
                f"音效已生成: {path} (preset {preset}, seed {seed}, "
                f"时长 {x.size / sfx.SFX_RATE:.3f}s, 波形 {p.wave}, 基频 {p.base_freq:.0f}Hz)",
                ok=True, path=str(path), preset=preset, seed=seed,
                duration=x.size / sfx.SFX_RATE, wave=p.wave, base_freq=p.base_freq)


@mcp.tool()
async def continuity_status() -> Annotated[CallToolResult, results.StatusResult]:
    """本插件当前的状态: 两个引擎在不在、哪些能力开着、资产存在哪。

    某个工具"不存在"时先看这里 —— 显存不够的机器上生图那半是被安装脚本关掉的,
    不是坏了。
    """
    ok, down = await asyncio.to_thread(engines.health)
    actors, subjects = store.actor_names(), store.subject_names()
    lines = [f"引擎: {'全部在线' if ok else '连不上 —— ' + '、'.join(down)}",
             f"  sd-server      {SD_SERVER}    {'启用' if ENABLE_IMAGE else '未启用 (显存不足, 只装了音频那半)'}",
             f"  audiocpp-server {AUDIO_SERVER}  {'启用' if ENABLE_AUDIO else '未启用'}",
             f"资产目录: {STATE_DIR}",
             f"  角色 {len(actors)} 个, 定妆 {len(subjects)} 个",
             f"抠图默认档: {DEFAULT_CUTOUT_QUALITY}",
             f"空闲卸载: {'关闭 (模型常驻)' if AUDIO_IDLE_UNLOAD_S <= 0 else f'{AUDIO_IDLE_UNLOAD_S:.0f}s 后释放显存'}"]
    setup_needed = not ok and not engines.setup_was_run()
    if setup_needed:
        lines.append("")
        lines.append(engines.SETUP_HINT)
    return _res(results.StatusResult, "\n".join(lines), ok=True,
                engines_ok=ok, engines_down=down,
                image=results.EngineInfo(url=SD_SERVER, enabled=ENABLE_IMAGE,
                                         reachable="sd-server" not in down),
                audio=results.EngineInfo(url=AUDIO_SERVER, enabled=ENABLE_AUDIO,
                                         reachable="audiocpp-server" not in down),
                state_dir=str(STATE_DIR), actors=len(actors), subjects=len(subjects),
                cutout_quality=DEFAULT_CUTOUT_QUALITY, idle_unload_s=AUDIO_IDLE_UNLOAD_S,
                setup_needed=setup_needed)


# =========================================================
def _cleanup_loop():
    """删除超过 RETENTION_DAYS 的生成产物。actors/ 和 subjects/ 不在这个目录下 ——
    它们不可复现, 永远不清理。"""
    while True:
        try:
            if RETENTION_DAYS > 0:
                cutoff = time.time() - RETENTION_DAYS * 86400
                freed = n = 0
                for f in GENERATED_DIR.iterdir():
                    try:
                        if f.is_file() and f.stat().st_mtime < cutoff:
                            freed += f.stat().st_size
                            f.unlink()
                            n += 1
                    except OSError:
                        pass
                if n:
                    log.info("cleanup: 删除 %d 个超过 %.0f 天的文件, 释放 %.1f MiB",
                             n, RETENTION_DAYS, freed / 2**20)
        except Exception:
            log.exception("cleanup 失败")
        time.sleep(CLEANUP_INTERVAL_S)


def _parse_args(argv=None):
    """命令行只覆盖传输方式 —— 别的一律走环境变量 (dsh 那边只能传 env)。

    默认值来自 config, 所以 `continuity-mcp` 不带参数时和以前一模一样: stdio。
    """
    ap = argparse.ArgumentParser(
        prog="continuity-mcp",
        description="continuity MCP server。默认 stdio (由 dsh 拉起); --http 改成常驻的 "
                    "streamable-http 服务, 给不走 stdio 的调用方用。")
    ap.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default=TRANSPORT,
                    help="传输方式 (默认 %(default)s, 或 CONTINUITY_TRANSPORT)")
    ap.add_argument("--http", action="store_const", const="streamable-http", dest="transport",
                    help="--transport streamable-http 的简写")
    ap.add_argument("--host", default=HTTP_HOST,
                    help="HTTP 监听地址 (默认 %(default)s, 或 CONTINUITY_HTTP_HOST)。"
                         "没有鉴权, 绑非本机地址前先想清楚")
    ap.add_argument("--port", type=int, default=HTTP_PORT,
                    help="HTTP 端口 (默认 %(default)s, 或 CONTINUITY_HTTP_PORT)")
    ap.add_argument("--path", default=HTTP_PATH,
                    help="HTTP 端点路径 (默认 %(default)s, 或 CONTINUITY_HTTP_PATH)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    for d in (STATE_DIR, ACTORS_DIR, SUBJECTS_DIR, GENERATED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    engines.start_idle_watch()
    atexit.register(engines.shutdown)      # 关掉 agent 也要还回显存
    # SIGTERM 默认直接终止, atexit 不跑; 而 dsh 关掉 stdin 之后 2 秒就发 SIGTERM。
    # 处理器里不能只 sys.exit —— SystemExit 会被 asyncio 事件循环吞掉, 进程反而挂住,
    # 拖满 2 秒再被 SIGKILL, 比不装处理器还糟 (实测 10 秒没退)。
    # 所以在处理器里直接把显存还掉再硬退, 那次调用自己有 1.5 秒上限。
    def _bail(signum, _frame):
        log.info("收到信号 %s, 释放显存后退出", signum)
        try:
            engines.shutdown()
        finally:
            os._exit(0)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _bail)
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    log.info("continuity: state=%s image=%s audio=%s", STATE_DIR, ENABLE_IMAGE, ENABLE_AUDIO)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return
    # 显存那套约束在 HTTP 下一个字都没变: 生图和 TTS 仍然共用同一把 _gpu_lock,
    # 所以多个客户端同时接上来也只是排队, 不会两个模型同时占卡。
    log.info("continuity: 监听 http://%s:%s%s (transport=%s)",
             args.host, args.port, args.path, args.transport)
    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port,
                streamable_http_path=args.path)


if __name__ == "__main__":
    main()
