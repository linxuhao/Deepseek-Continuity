# ==========================================
# actor / subject 的存储 —— 一个名字对应一份参考产物 (wav 或 png) + 一份 json。
#
# 这两样东西是本插件的全部价值所在, 也是唯一不可复现的东西: 重铸/重定出来的
# 是"另一个人"。所以这里只做最笨的事 —— 文件落盘, 不加缓存, 不做数据库。
#
# 参考音只由本进程读, 读出来随请求内联发给引擎。引擎不需要看见这个目录 ——
# 早先它需要, 那是个设计错误, 代价是跨机部署下克隆整个不能用。
# ==========================================
import json
import os
import time

from .config import ACTORS_DIR, SUBJECTS_DIR, NAME_RE
from .errors import NotFound, UserError


def _safe(name):
    """名字必须先过 NAME_RE 才准拼进路径。

    创建那几条路径一直有这个校验, 删除那两条没有 —— 而 Path.unlink() 对 ".." 一视同仁,
    于是 delete_actor(name="../../../某工程/package") 会删掉别人的 package.json,
    然后回一句"已删除"。调用方是 LLM, 一个幻觉出来的名字就够了, 不需要有人使坏。
    收在这里而不是各个工具里, 是因为所有路径都从这一个函数出去。"""
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise UserError("名字只能是字母/数字/下划线/连字符/中文, 1~40 字")
    return name


def _paths(base, name, ext):
    name = _safe(name)
    return base / f"{name}.{ext}", base / f"{name}.json"


def actor_paths(name):
    return _paths(ACTORS_DIR, name, "wav")


def subject_paths(name):
    return _paths(SUBJECTS_DIR, name, "png")


def _load(meta_path):
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_actor(name):
    return _load(actor_paths(name)[1])


def load_subject(name):
    return _load(subject_paths(name)[1])


def _names(base):
    try:
        return sorted(f[:-5] for f in os.listdir(base) if f.endswith(".json"))
    except OSError:
        return []


def actor_names():
    return _names(ACTORS_DIR)


def subject_names():
    return _names(SUBJECTS_DIR)


def save_meta(meta_path, meta):
    meta["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def drop(paths, kind, name):
    """删掉一个角色/物件。不可逆 —— 参考产物不可复现。"""
    gone = [p for p in paths if p.is_file()]
    if not gone:
        # NotFound 而不是 FileNotFoundError: "这个名字没有"和"这个名字已经有了"是调用方
        # 要区分对待的两件事 (404 / 409), 而只有这里知道是哪一件。判断留在知道答案的地方。
        raise NotFound(f"{kind} '{name}' 不存在")
    for p in gone:
        p.unlink()
    return len(gone)


def listing(kind):
    """给 LLM 看的一行行摘要。"""
    if kind == "actor":
        return [a for a in (load_actor(n) for n in actor_names()) if a]
    return [s for s in (load_subject(n) for n in subject_names()) if s]
