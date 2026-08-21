# ==========================================
# 抠图 / 切图 —— 纯 CPU, 不碰 GPU, 所以不走 jobs.py 那把串行锁。
#
# FLUX 生成不出 alpha: 你让它画"透明背景", 它把 PS 那种灰白棋盘格当成不透明像素画
# 出来。做精灵图必须转成真 RGBA。两条路径:
#   checker  抠掉假透明棋盘格 (按结构证据判定, 不按"抠掉了多少")
#   rembg    通用显著物体抠图 (onnxruntime, CPU)
# 外加一份质量报告: 抠成一堆碎片 / 主体被啃出洞 / 压根没有前景主体, 都会附警告。
# ==========================================
import logging

import numpy as np
from PIL import Image, ImageDraw

from .config import (REMBG_MODELS, CHECKER_MIN_CAND, CHECKER_MIN_GAP, CHECKER_MAX_VALLEY,
                     CHECKER_MIN_RUNS, ALPHA_MAX_TRANSPARENT, ALPHA_MIN_TRANSPARENT,
                     ALPHA_MIN_BLOB, ALPHA_MAX_HOLES, ALPHA_MAX_BG_DETAIL)

log = logging.getLogger("continuity")

_rembg_sessions = {}          # 每个模型一个常驻 session (onnxruntime 初始化很贵)


def checker_candidates(a, bright=195, neutral=20):
    """又亮又接近中性灰的像素 —— 棋盘格的必要条件, 但远不是充分条件。"""
    return (a.min(2) >= bright) & ((a.max(2) - a.min(2)) <= neutral)


def key_checkerboard(img, bright=195, neutral=20):
    """把 FLUX 画出来的"透明棋盘格"抠掉, 返回 alpha 数组。

    FLUX.2 Klein 不会输出 alpha 通道: 你要"透明背景", 它就把 PS 那种灰白格子当成
    不透明像素画出来。这些格子的特征是又亮又接近中性灰 (实测 255 与 220 两种方块)。
    只按颜色判定会连鸟肚子上的白色一起抠掉, 所以再加一条: 必须与画面边缘连通。
    """
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    cand = checker_candidates(a, bright, neutral)
    # Image.fromarray 返回的是只读 buffer, floodfill 的写入会被静默丢弃, 必须 copy()
    m = Image.fromarray(np.where(cand, 255, 0).astype(np.uint8)).copy()
    w, h = img.size
    for xy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if m.getpixel(xy) == 255:
            ImageDraw.floodfill(m, xy, 128, thresh=0)
    return np.where(np.array(m) == 128, 0, 255).astype(np.uint8)


def two_tone(grey, cand):
    """候选区的灰度是不是"两级"。返回 (峰1, 峰2, 谷/峰)。

    真棋盘格实测并不是干净的两个值 (FLUX 画出来带噪): 255/254 一簇, 219~224 一簇。
    所以用平滑直方图找两个峰, 再看两峰之间的谷有多深 —— 连续渐变的摄影背景填满谷,
    两级方格则谷接近 0。
    """
    hist = np.bincount(grey[cand], minlength=256).astype(np.float64)
    sm = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    total = sm.sum()
    if total <= 0:
        return 0, 0, 1.0
    sm /= total
    p1 = int(np.argmax(sm))
    far = np.ones(256, dtype=bool)
    far[max(0, p1 - 10):p1 + 11] = False          # 第二个峰必须离第一个足够远
    p2 = int(np.argmax(np.where(far, sm, -1.0)))
    lo, hi = sorted((p1, p2))
    peak = float(min(sm[p1], sm[p2]))
    valley = float(sm[lo + 1:hi].min()) if hi - lo > 1 else peak
    return p1, p2, (valley / peak if peak > 0 else 1.0)


def tone_runs(tone, cand):
    """沿行统计交替方块的游程长度。

    只看候选像素占多数的行, 并丢掉每段两端不完整的游程 —— 半个方块会污染中位数。
    """
    out = []
    for r in range(tone.shape[0]):
        row = cand[r]
        if row.mean() < 0.5:
            continue
        idx = np.flatnonzero(row)
        if idx.size < 32:
            continue
        for span in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if span.size < 32:
                continue
            t = tone[r, span]
            bounds = np.concatenate(([0], np.flatnonzero(np.diff(t)) + 1, [t.size]))
            rl = np.diff(bounds)
            if rl.size > 2:
                out.extend(rl[1:-1].tolist())
    return np.asarray(out, dtype=np.float64)


def looks_like_checkerboard(img, bright=195, neutral=20):
    """这张图的亮中性区域到底是不是 FLUX 的"假透明"棋盘格。返回 (bool, 证据)。

    旧实现按"抠掉了多少"来判 (yield): 先抠一遍, 抠掉 >5% 就认为是棋盘格。那是错的 ——
    任何明亮中性的摄影背景都满足候选条件。实测白猫 + 浅灰影棚背景被判成 checker (0.662),
    floodfill 从边缘连通进猫身体, 把猫身上和头上啃出大洞。

    改判结构证据: 棋盘格是恰好两级灰度 (实测 253 与 221) 铺成的固定边长方块。
    两个条件都成立才算; 否则一律走 rembg。
    """
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    cand = checker_candidates(a, bright, neutral)
    ev = {"cand_ratio": round(float(cand.mean()), 4)}
    if cand.mean() < CHECKER_MIN_CAND:
        return False, dict(ev, reason="亮中性区域太小, 没有假透明背景")
    grey = a.mean(2).astype(np.uint8)
    p1, p2, valley = two_tone(grey, cand)
    ev.update(tones=[p1, p2], tone_gap=abs(p1 - p2), valley_ratio=round(valley, 3))
    if abs(p1 - p2) < CHECKER_MIN_GAP or valley > CHECKER_MAX_VALLEY:
        return False, dict(ev, reason="灰度是连续渐变而非两级, 像摄影背景")
    tone = grey >= (p1 + p2) / 2.0
    rows, cols = tone_runs(tone, cand), tone_runs(tone.T, cand.T)
    if rows.size < 16 or cols.size < 16:
        return False, dict(ev, reason="没有成片的候选区可判周期")
    cell_r, cell_c = float(np.median(rows)), float(np.median(cols))
    cons_r = float((np.abs(rows - cell_r) <= max(1.0, 0.25 * cell_r)).mean())
    cons_c = float((np.abs(cols - cell_c) <= max(1.0, 0.25 * cell_c)).mean())
    ev.update(cell=[round(cell_r, 1), round(cell_c, 1)],
              run_consistency=[round(cons_r, 3), round(cons_c, 3)])
    if cons_r < CHECKER_MIN_RUNS or cons_c < CHECKER_MIN_RUNS:
        return False, dict(ev, reason="方块边长不规则, 不是周期网格")
    if not (3 <= cell_r <= 128 and 3 <= cell_c <= 128):
        return False, dict(ev, reason="方块尺寸不合理")
    if abs(cell_r - cell_c) > 0.25 * max(cell_r, cell_c):
        return False, dict(ev, reason="方块不是正方形")
    return True, dict(ev, reason="两级灰度 + 周期方格 = FLUX 假透明")


def rembg_alpha(img, quality="best"):
    """通用显著物体抠图。放 CPU (onnxruntime) —— GPU 留给两个 ggml 引擎。"""
    import onnxruntime as ort
    from rembg import new_session, remove
    model = REMBG_MODELS.get(quality, REMBG_MODELS["best"])
    sess = _rembg_sessions.get(model)
    if sess is None:
        log.info("rembg: 首次加载 %s", model)
        # onnxruntime 的 CPU memory arena 把推理峰值变成常驻内存, 且永不归还:
        # 实测 1024x1024 两次调用后 RSS 0.06 -> 7.5 -> 12.1 GB 封顶不动
        # (30 GB 的机器凭空少掉 40% 内存, available 只剩 2 GB)。
        # 关掉 arena 后常驻 0.69 GB, alpha 输出逐位相同 —— 纯粹是分配器行为。
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        sess = _rembg_sessions[model] = new_session(model, sess_opts=so)
    return model, np.array(remove(img.convert("RGB"), session=sess))[:, :, 3]

def largest_blob_ratio(mask, max_side=192):
    """最大不透明连通块占整图的比例。

    缩到 <=192px 再做标签传播 (取 4 邻域最大值直到不动): 这个数只用来判"抠出来的东西
    碎成了渣", 不需要像素级精度, 而全分辨率的纯 python 连通域太慢。
    """
    h, w = mask.shape
    s = max_side / max(h, w)
    if s < 1.0:
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST)) > 127
    else:
        m = mask
    if not m.any():
        return 0.0
    lab = np.where(m, np.arange(1, m.size + 1).reshape(m.shape), 0)
    for _ in range(4 * max(m.shape)):
        nb = lab.copy()
        nb[1:] = np.maximum(nb[1:], lab[:-1])
        nb[:-1] = np.maximum(nb[:-1], lab[1:])
        nb[:, 1:] = np.maximum(nb[:, 1:], lab[:, :-1])
        nb[:, :-1] = np.maximum(nb[:, :-1], lab[:, 1:])
        nb = np.where(m, nb, 0)
        if np.array_equal(nb, lab):
            break
        lab = nb
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return float(counts.max() / m.size)


def hole_ratio(alpha):
    """主体轮廓内部被抠出来的洞, 占不透明面积的比例。

    白猫那次翻车不是比例算错 —— 0.662 对那张图完全是个合理的数字; 真正错的是
    猫身上和头上被啃出了洞。洞是"透明、但从画面边缘走不到"的像素, 用 key_checkerboard
    已经在用的 floodfill 就能数出来 (外面先套一圈透明, 让所有贴边的透明区连成一片,
    一次 floodfill 就够), 不必为此引入 scipy。
    """
    opaque = alpha > 0
    n = int(opaque.sum())
    if n == 0:
        return 0.0
    h, w = opaque.shape
    pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
    pad[1:-1, 1:-1] = np.where(opaque, 255, 0)
    m = Image.fromarray(pad).copy()          # fromarray 的 buffer 只读, 必须 copy
    ImageDraw.floodfill(m, (0, 0), 128, thresh=0)
    return float((np.asarray(m)[1:-1, 1:-1] == 0).sum() / n)


def bg_detail_ratio(img, alpha):
    """被抠掉的那片区域, 细节密度相对于留下来的主体有多高。

    合格的抠图, 去掉的是背景: 影棚灰、白桌面、绿幕、棋盘格, 都很平。当"背景"和主体
    一样满是边缘时 (集市那张: 0.90), 说明模型只是从一整幅场景里随手挑了几个人留下,
    这种结果多半不是调用方想要的。返回 (相对比值, 去掉区域的平均梯度)。
    """
    g = np.asarray(img.convert("L"), dtype=np.float32)
    gx, gy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    def mean_in(mask):
        mx, my = mask[:, :-1] & mask[:, 1:], mask[:-1] & mask[1:]
        v = np.concatenate([gx[mx], gy[my]])
        return float(v.mean()) if v.size else 0.0
    removed, kept = mean_in(alpha == 0), mean_in(alpha > 127)
    return (removed / kept if kept > 1.0 else 0.0), removed


def alpha_report(img, alpha, extra=()):
    """算出抠图质量指标, 并在明显不对时给一句话警告 (不失败, 只是别再默默报成功)。"""
    tr = float((alpha == 0).mean())
    solid = float((alpha > 127).mean())
    holes = hole_ratio(alpha)
    blob = largest_blob_ratio(alpha > 127)
    detail, removed_grad = bg_detail_ratio(img, alpha)
    m = {"transparent_ratio": round(tr, 4), "solid_ratio": round(solid, 4),
         "hole_ratio": round(holes, 4), "largest_blob_ratio": round(blob, 4),
         "bg_detail_ratio": round(detail, 3)}
    w = list(extra)
    if tr > ALPHA_MAX_TRANSPARENT:
        w.append(f"{tr:.1%} 的像素被抠成了透明, 主体几乎整个没了")
    if tr < ALPHA_MIN_TRANSPARENT:
        w.append(f"只抠掉了 {tr:.1%}, 基本什么都没去掉")
    if blob < ALPHA_MIN_BLOB:
        w.append(f"最大的不透明连通块只占整图 {blob:.1%}, 抠出来的是碎片不是主体")
    if holes > ALPHA_MAX_HOLES:
        w.append(f"主体轮廓内部有 {holes:.1%} 的面积被挖成了洞 (相对不透明面积), "
                 f"多半是背景色和主体撞色被啃穿了")
    if detail > ALPHA_MAX_BG_DETAIL and removed_grad > 5.0:
        w.append(f"被去掉的区域细节密度是主体的 {detail:.0%}, 那不是背景而是画面的一部分, "
                 f"这张图没有明确的前景主体")
    return m, ("抠图结果很可能不对: " + "; ".join(w) if w else None)


def trim_frame(frame):
    """裁到非透明/非棋盘格的外接框。"""
    if frame.mode == "RGBA":
        box = frame.getchannel("A").getbbox()
    else:
        box = Image.fromarray(key_checkerboard(frame)).getbbox()
    return frame.crop(box) if box else frame



def remove_bg(img, mode="auto", quality="best", new_name=None, out_dir=None):
    """抠掉背景, 写出真正带 alpha 的 RGBA PNG。返回 (路径, 结果字典)。"""
    if mode not in ("auto", "checker", "rembg"):
        raise ValueError("mode 必须是 auto/checker/rembg")
    if quality not in REMBG_MODELS:
        raise ValueError(f"quality 必须是 {'/'.join(REMBG_MODELS)}")
    evidence, is_checker, extra = None, None, []
    if mode == "auto":
        # 按结构证据路由, 不按"抠掉了多少" —— 后者会把浅色影棚背景误判成棋盘格,
        # floodfill 从边缘连通进主体, 在猫身上和头上啃出洞。
        is_checker, evidence = looks_like_checkerboard(img)
        used = "checker" if is_checker else "rembg"
    else:
        used = mode
        if used == "checker":
            # 手动指定也照样取证: 强行对一张照片走 checker 正是上面那条翻车路径,
            # 抠出来的洞常常与背景连通 (不是闭合的洞), hole_ratio 抓不到,
            # 只有"这压根不是棋盘格"这个证据抓得到。
            is_checker, evidence = looks_like_checkerboard(img)
    if used == "checker":
        alpha, model = key_checkerboard(img), None
        if is_checker is False:
            extra.append(f"你指定了 mode=checker, 但这张图不是 FLUX 的假透明棋盘格 "
                         f"({evidence.get('reason')}) —— 亮色背景会顺着边缘连通吃进主体, "
                         f"改用 mode=auto 或 mode=rembg")
    else:
        model, alpha = rembg_alpha(img, quality)
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(alpha))
    path = out_dir / new_name
    out.save(path, format="PNG")
    metrics, warning = alpha_report(img, alpha, extra)
    res = {"transparent_ratio": metrics["transparent_ratio"], "mode_used": used,
           "model": model, "metrics": metrics}
    if evidence:
        res["checker_evidence"] = evidence
    if warning:
        log.warning("remove_bg %s: %s", new_name, warning)
        res["warning"] = warning
    return path, res


def slice_sheet(img, rows=None, cols=None, frame_width=None, frame_height=None,
                trim=True, name_fn=None, out_dir=None):
    """把排成网格的 sprite sheet 切成单帧 PNG。返回路径列表。"""
    W, H = img.size
    if rows and cols:
        fw, fh = W // cols, H // rows
    elif frame_width and frame_height:
        fw, fh = frame_width, frame_height
        rows, cols = H // fh, W // fw
    else:
        raise ValueError("必须提供 rows+cols 或 frame_width+frame_height")
    paths = []
    for r in range(rows):
        for c in range(cols):
            f = img.crop((c * fw, r * fh, (c + 1) * fw, (r + 1) * fh))
            if trim:
                f = trim_frame(f)
            p = out_dir / name_fn()
            f.save(p, format="PNG")
            paths.append(p)
    return paths
