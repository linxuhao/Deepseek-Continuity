# ==========================================
# actor / subject 的存储 —— 一个名字对应一份参考产物 (wav 或 png) + 一份 json。
#
# 这两样东西是本插件的全部价值所在, 也是唯一不可复现的东西: 重铸/重定出来的
# 是"另一个人"。所以这里只做最笨的事 —— 文件落盘, 不加缓存, 不做数据库。
# ==========================================
import json
import os
import time

from .config import ACTORS_DIR, SUBJECTS_DIR, ENGINE_ACTORS_DIR


def _paths(base, name, ext):
    return base / f"{name}.{ext}", base / f"{name}.json"


def actor_paths(name):
    return _paths(ACTORS_DIR, name, "wav")


def subject_paths(name):
    return _paths(SUBJECTS_DIR, name, "png")


def engine_voice_ref(name):
    """参考音在音频引擎眼里的路径。voice_ref 是交给引擎去 open 的, 不是本进程。"""
    return f"{ENGINE_ACTORS_DIR.rstrip('/')}/{name}.wav"


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
        raise FileNotFoundError(f"{kind} '{name}' 不存在")
    for p in gone:
        p.unlink()
    return len(gone)


def listing(kind):
    """给 LLM 看的一行行摘要。"""
    if kind == "actor":
        return [a for a in (load_actor(n) for n in actor_names()) if a]
    return [s for s in (load_subject(n) for n in subject_names()) if s]
