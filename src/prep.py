# 下ごしらえ：商品の缶を切り抜く・紋章を描く。どちらも cache/ に RGBA の PNG で置く
#   python src/prep.py
import math, os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from key import key_frame
import typo

NAVY, WHITE, PINK, CYAN = (22, 30, 58), (255, 255, 255), (240, 70, 150), (40, 210, 235)

def cut_can(src="assets/can_green.png", out="cache/can.png"):
    """グリーンバックの缶を切り抜いて、缶の形ぴったりに詰める"""
    bgr = cv2.imread(src)
    if bgr is None: sys.exit(f"缶の画像がありません：{src}")
    rgba = key_frame(bgr)
    a = rgba[..., 3]; a[a < 40] = 0
    # 缶の下に落ちている影（暗い緑）も抜けるので、缶そのもの＝いちばん大きな塊だけ残す
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 128).astype(np.uint8))
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    keep = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8))
    rgba[..., 3] = a * keep
    # 床に落ちた影も同じ塊につながるので、缶の上半分が写っている列だけ残す
    top = rgba[: rgba.shape[0] // 2, :, 3] > 128
    cols = np.nonzero(top.mean(0) > 0.25)[0]
    rgba[:, : cols.min()] = 0; rgba[:, cols.max() + 1:] = 0
    ys, xs = np.nonzero(rgba[..., 3] > 8)
    cv2.imwrite(out, cv2.cvtColor(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1], cv2.COLOR_RGBA2BGRA))
    print(out, xs.max() - xs.min() + 1, "x", ys.max() - ys.min() + 1)

def shield(cx, top, w, h):
    """盾の形（上が平ら、下が尖る）の頂点"""
    pts = [(cx - w / 2, top + h * 0.06), (cx - w * 0.2, top), (cx, top + h * 0.03), (cx + w * 0.2, top), (cx + w / 2, top + h * 0.06)]
    for k in range(1, 21):   # 右の辺 → 下の尖り
        u = k / 20; pts.append((cx + w / 2 * (1 - u ** 2.2), top + h * 0.06 + (h * 0.94) * math.sin(u * math.pi / 2) ** 1.0 * u ** 0.4))
    for k in range(19, 0, -1):
        u = k / 20; pts.append((cx - w / 2 * (1 - u ** 2.2), top + h * 0.06 + (h * 0.94) * math.sin(u * math.pi / 2) ** 1.0 * u ** 0.4))
    return pts

def crest(out="cache/crest.png", S=4):
    """紋章：盾・交差するヘアピン（キャラの X 字ヘアピン）・星・帯に「ROOFTOP」・リボンに「DANCE CREW」"""
    Wc, Hc = 900 * S, 760 * S
    im = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx = Wc / 2
    # 盾
    pts = shield(cx, 40 * S, 420 * S, 560 * S)
    d.polygon(pts, fill=NAVY + (215,)); d.line(pts + [pts[0]], fill=WHITE + (255,), width=10 * S)
    inner = shield(cx, 62 * S, 372 * S, 512 * S); d.line(inner + [inner[0]], fill=WHITE + (255,), width=3 * S)
    # 交差するヘアピン（X）：ピンクとシアン
    L = 150 * S; y = 250 * S
    for col, ang in ((PINK, 45), (CYAN, -45)):
        a = math.radians(ang); dx, dy = math.cos(a) * L, math.sin(a) * L
        d.line([(cx - dx, y - dy), (cx + dx, y + dy)], fill=WHITE + (255,), width=34 * S)
        d.line([(cx - dx, y - dy), (cx + dx, y + dy)], fill=col + (255,), width=22 * S)
    # 星
    def star(x, y, r):
        p = [(x + r * (1 if k % 2 == 0 else 0.42) * math.sin(k * math.pi / 5), y - r * (1 if k % 2 == 0 else 0.42) * math.cos(k * math.pi / 5)) for k in range(10)]
        d.polygon(p, fill=WHITE + (255,))
    for k, x in enumerate((-110, 0, 110)): star(cx + x * S, (92 if k != 1 else 80) * S, (18 if k != 1 else 24) * S)
    # 帯（盾の真ん中を横切る）に ROOFTOP
    by = 430 * S; bh = 104 * S; bw = 620 * S
    band = [(cx - bw / 2, by - bh / 2), (cx + bw / 2, by - bh / 2), (cx + bw / 2 - 26 * S, by), (cx + bw / 2, by + bh / 2), (cx - bw / 2, by + bh / 2), (cx - bw / 2 + 26 * S, by)]
    d.polygon(band, fill=NAVY + (255,)); d.line(band + [band[0]], fill=WHITE + (255,), width=6 * S)
    t = Image.fromarray(typo.word("ROOFTOP", 86 * S, typo.SERIF, WHITE, spacing=6 * S))
    im.alpha_composite(t, (int(cx - t.width / 2), int(by - t.height / 2)))
    # リボン（下）に DANCE CREW
    ry = 560 * S; rw = 420 * S; rh = 60 * S
    rib = [(cx - rw / 2, ry - rh / 2), (cx + rw / 2, ry - rh / 2), (cx + rw / 2 + 40 * S, ry + rh / 2 + 10 * S), (cx - rw / 2 - 40 * S, ry + rh / 2 + 10 * S)]
    d.polygon(rib, fill=WHITE + (255,))
    t = Image.fromarray(typo.word("DANCE CREW", 40 * S, typo.SERIF, NAVY, spacing=5 * S))
    im.alpha_composite(t, (int(cx - t.width / 2), int(ry + 5 * S - t.height / 2)))
    t = Image.fromarray(typo.word("EST. 2026", 30 * S, typo.SERIF, WHITE, spacing=6 * S))
    im.alpha_composite(t, (int(cx - t.width / 2), int(655 * S)))
    im = im.resize((Wc // S, Hc // S), Image.LANCZOS)
    a = np.array(im); ys, xs = np.nonzero(a[..., 3] > 4)
    Image.fromarray(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]).save(out)
    print(out, a.shape)

if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    cut_can(); crest()
