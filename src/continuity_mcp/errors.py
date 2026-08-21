# ==========================================
# 失败的分类 —— 给程序看的那一半。
#
# 人话里的失败一直是一段中文, 而调用方 (HTTP 外壳) 要把它还原成 400/404/409/500。
# 靠正则匹配那段中文是不行的: 措辞是 prompt 的一部分, 改一次下游就静默地判错。
# 所以分类必须在"知道自己为什么失败"的那一层做出来 —— 也就是抛异常的地方 ——
# 再由 results.Result.error_code 原样带出去。
#
# 只有四个取值, 刻意不扩张:
#   invalid       入参不合法 (名字非法 / 描述为空 / mode 不认识 / 参考音规格不对 / 图读不了)
#   not_found     指名的 actor / subject 不存在
#   conflict      同名已存在, 而调用方没传 force
#   engine_error  引擎调用失败, 以及其它意料之外的异常
#
# NotFound / Conflict 都从 UserError 派生: 它们本来就是"调用方该改的事", 而且这样
# 既有的 `except UserError: raise` 不用动。
# ==========================================

INVALID = "invalid"
NOT_FOUND = "not_found"
CONFLICT = "conflict"
ENGINE_ERROR = "engine_error"


class UserError(ValueError):
    """给调用方 (LLM) 看的指令式报错 —— 告诉它下一步该干什么, 不只是"失败了"。"""


class NotFound(UserError):
    """点名的 actor / subject 不存在。"""


class Conflict(UserError):
    """同名的 actor / subject 已经有了, 而调用方没说要覆盖。"""


def error_code(e):
    """把一个异常归到四类之一。

    兜底是 engine_error 而不是 invalid: 猜不出来的东西是我们这边的意外, 不是调用方
    传错了参数 —— 把意外报成 400 会让外壳的调用方去改一个本来就没错的请求。
    """
    if isinstance(e, NotFound):
        return NOT_FOUND
    if isinstance(e, Conflict):
        return CONFLICT
    if isinstance(e, (UserError, ValueError)):
        return INVALID
    return ENGINE_ERROR
