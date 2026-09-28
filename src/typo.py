# 文字：Pillow で描いて RGBA（float）にする。同じ文字・大きさはキャッシュする
#   フォントは OS に入っているものから探す。見つからないときは環境変数で指定する（下の FONTS のキー名）
import os
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

FONTS = {
    "OKUJOU_FONT_JP": [        # 歌詞の太字
        "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
        "C:/Windows/Fonts/YuGothB.ttc"],
    "OKUJOU_FONT_JP_THIN": [   # 歌詞の細字
        "/System/Library/Fonts/ヒラギノ角ゴシック W2.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc",
        "C:/Windows/Fonts/YuGothL.ttc"],
    "OKUJOU_FONT_MINCHO": [    # 縦書きの漢字（ttc の中の番号を #n で指定できる。2＝W6）
        "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc#2",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "C:/Windows/Fonts/yumindb.ttf"],
    "OKUJOU_FONT_TALL": [      # 画面より大きい縦長の英字
        "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"],
    "OKUJOU_FONT_WIDE": [      # DANCE・CREW の太い英字
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "OKUJOU_FONT_SERIF": [     # 紋章の文字
        "/System/Library/Fonts/Supplemental/Copperplate.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
    "OKUJOU_FONT_HUD": [       # HUD の小さな英数字
        "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}

def find_font(env):
    path = os.environ.get(env)
    if path:
        if not os.path.exists(path.split("#")[0]): raise FileNotFoundError(f"{env}={path} が見つかりません")
        return path
    for p in FONTS[env]:
        if os.path.exists(p.split("#")[0]): return p
    raise FileNotFoundError(f"フォントが見つかりません。環境変数 {env} にフォントのパスを指定してください")

JP, JP_THIN, MINCHO, TALL, WIDE, SERIF, HUD = (find_font(k) for k in FONTS)

@lru_cache(maxsize=64)
def _font(path, px):
    f, _, idx = path.partition("#")
    return ImageFont.truetype(f, px, index=int(idx or 0))

@lru_cache(maxsize=512)
def word(s, px, font, fill=(255, 255, 255), stroke=None, sw=0, stretch=1.0, spacing=0):
    """文字列を描いた RGBA（uint8）。fill=None でフチだけ（中は透明）。stretch で横に伸ばす。spacing は字間（px）"""
    f = _font(font, px)
    chars = list(s) if spacing else [s]
    l, t, r, b = f.getbbox(s, stroke_width=sw)
    asc, desc = f.getmetrics()
    tw = (sum(f.getlength(c) for c in chars) + spacing * (len(chars) - 1)) if spacing else r - l
    im = Image.new("RGBA", (int(tw) + 2 * sw + 16, asc + desc + 2 * sw + 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(im); x = 8 + sw - (0 if spacing else l)
    for c in chars:
        d.text((x, 8 + sw), c, font=f, fill=tuple(fill) + (255,) if fill else (0, 0, 0, 0),
               stroke_width=sw, stroke_fill=tuple(stroke) + (255,) if stroke else None)
        if spacing: x += f.getlength(c) + spacing
    a = np.array(im)
    ys, xs = np.nonzero(a[..., 3])
    a = a[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]   # 余白を詰める（文字の実際の形で位置を決める）
    if stretch != 1.0: a = cv2.resize(a, (int(a.shape[1] * stretch), a.shape[0]), interpolation=cv2.INTER_AREA)
    return a

@lru_cache(maxsize=64)
def vertical(s, px, font, fill=(255, 255, 255), gap=1.18):
    """縦書き（1文字ずつ縦に積む）"""
    ims = [word(c, px, font, fill) for c in s]
    w = max(i.shape[1] for i in ims); step = int(px * gap)
    out = np.zeros((step * len(s), w, 4), np.uint8)
    for k, i in enumerate(ims):
        y = k * step + (step - i.shape[0]) // 2; x = (w - i.shape[1]) // 2
        out[y:y + i.shape[0], x:x + i.shape[1]] = i
    return out

def paste(canvas, rgba, x, y, alpha=1.0, scale=1.0, anchor="lt", mask=None, add=False):
    """canvas（float RGB）の (x, y) に rgba を重ねる。anchor：l/m/r ＋ t/m/b。mask（画面と同じ大きさ HxW）を掛けられる
    add=True でスクリーン合成（光る文字）"""
    if scale != 1.0:
        rgba = cv2.resize(rgba, (max(1, int(rgba.shape[1] * scale)), max(1, int(rgba.shape[0] * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    h, w = rgba.shape[:2]
    x = x - {"l": 0, "m": w / 2, "r": w}[anchor[0]]; y = y - {"t": 0, "m": h / 2, "b": h}[anchor[1]]
    x, y = int(round(x)), int(round(y)); H, W = canvas.shape[:2]
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if X1 <= X0 or Y1 <= Y0: return
    src = rgba[Y0 - y:Y1 - y, X0 - x:X1 - x].astype(np.float32)
    a = src[..., 3:4] / 255 * alpha
    if mask is not None: a = a * mask[Y0:Y1, X0:X1, None]
    dst = canvas[Y0:Y1, X0:X1]
    if add: dst[:] = 255 - (255 - dst) * (255 - src[..., :3] * a) / 255
    else: dst *= (1 - a); dst += src[..., :3] * a

def _shadow_of(im):
    sh = cv2.copyMakeBorder(im, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)); sh[..., :3] = (15, 25, 55)
    sh = cv2.GaussianBlur(sh, (0, 0), 3); sh[..., 3] = np.clip(sh[..., 3].astype(np.int32) * 2, 0, 255).astype(np.uint8)
    return sh[6:-6, 6:-6]

def text(canvas, s, x, y, px, font, fill=(255, 255, 255), anchor="lt", alpha=1.0, shadow=0.0, **kw):
    """文字を置く。shadow … 下に敷く暗いにじみの濃さ（明るい空の上でも読めるように）"""
    im = word(s, px, font, tuple(fill) if fill else None, **kw)
    if shadow: paste(canvas, _shadow_of(im), x + 1, y + 2, alpha * shadow, anchor=anchor)
    paste(canvas, im, x, y, alpha, anchor=anchor)
    return im.shape[1], im.shape[0]
