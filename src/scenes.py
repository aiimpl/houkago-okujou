# カット割り：frame(秒) が1コマ（RGB uint8, 1920×1080）を返す。16秒でループする
#
#   カットの秒は、曲（ACE-Step の歌入り）で歌詞が歌われる秒に合わせてある（gen/CHOICE.json の _song と conte/CONTE.md）
#   秒          背景         内容
#   0.0–1.0    ローアングル  商品カット：手前に缶2本、右上に太陽。0.5秒で歌詞の1行目
#   1.0–7.0    正面          カメラが右へ流れる。左に歌詞が1行ずつ積まれる（キャラの後ろ）。チャイム → 階段 → 駆け上がる
#   7.0–11.0   斜め          頭上に名札、四角のグリッドが明滅。右下に歌詞2行。空を指さす → 床 →「ステージ」
#   11.0–12.0  建屋          画面より大きい「ROOFTOP」が右から左へ流れる（キャラの前は通らない）
#   12.0–14.0  正面2         白フラッシュ。「Dance」「Dance」と歌う所で DANCE が塗りに、キャラに赤青のにじみ
#   14.0–15.0  正面2         「CREW」拍ごとに切り替え、カメラが揺れてブレ・グリッチ
#   15.0–16.0  俯瞰          「Rooftop Dance Crew」で寄り。腕組みの X、手前に紋章、左右に縦書き
import numpy as np, cv2
from comp import (W, H, SPB, DUR, Plate, Cam, Dancer, dancer_layer, cast_shadow, put_shadow, put_dancer, over,
                  screen_add, sun_flare, bloom, vignette, chroma, motion_blur, glitch, clamp01, out_cubic, out_back, in_out, punch)
import typo
from typo import paste, text, word

WHITE = (255, 255, 255)

# 描く層（作り方の動画で、背景 → ＋キャラ → ＋文字 → ＋仕上げ と1つずつ足す絵に使う）
LAYERS = {"bg", "dancer", "text", "fx"}
PINK, CYAN = (240, 70, 150), (40, 210, 235)

# 背景：ファイル・床の高さ・地平線の高さ（絵のピクセル、方眼で測った）・カメラの高さが身長の何割か
PLATE_DEFS = {
    "low":    ("assets/bg_low.png",    1180, 946, 0.28),   # 床から40cmのローアングル
    "front":  ("assets/bg_front.png",  1420, 814, 0.81),
    "diag":   ("assets/bg_diag.png",   1400, 841, 0.81),
    "build":  ("assets/bg_build.png",  1480, 949, 0.81),
    "front2": ("assets/bg_front2.png", 1430, 824, 0.81),
    "top":    ("assets/bg_top.png",    1400, 0, 0.81),     # 俯瞰（寄りの背景にだけ使う）
}
_plates = {}
def plate(k):
    if k not in _plates:
        path, fy, hy, eye = PLATE_DEFS[k]
        _plates[k] = Plate(path, fy, hy); _plates[k].eye = eye
    return _plates[k]

_assets = {}
def asset(name):
    if name not in _assets:
        im = cv2.imread(f"cache/{name}.png", cv2.IMREAD_UNCHANGED)
        if im is None: raise FileNotFoundError(f"cache/{name}.png がありません（python src/prep.py を先に）")
        _assets[name] = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    return _assets[name]

# ---------- ダンス ----------
# カットごとに生成したダンス（cache/cuts/<名前>/）と、動画の秒 → 素材の秒 の対応。
# 対応は区間ごとに線形（決めのコマが拍に来るよう、区間ごとに再生速度を変える）。秒は各カットの頭から
DANCE = {
    "product": ("c1_open",   [(0.0, 2 / 16), (0.5, 18 / 16), (1.0, 30 / 16)]),                   # 「六時間目の」で X から腕を開く
    "lyrics1": ("c2b_chime", [(0.0, 0.0), (0.55, 3 / 16), (1.4, 14 / 16), (2.25, 30 / 16), (3.0, 42 / 16), (4.0, 57 / 16), (4.3, 63 / 16), (6.0, 96 / 16)]),  # 「チャイムで」で指、「階段を」「駆け上がる」でもも上げ
    "lyrics2": ("c3_sky",    [(0.0, 14 / 16), (0.3, 24 / 16), (1.6, 33 / 16), (2.3, 40 / 16), (3.3, 48 / 16), (4.0, 60 / 16)]),  # 「空に」で空、「近い」で床、「ステージ」で腰に手
    "build":   ("c4b_swing", [(0.0, 42 / 16), (0.5, 51 / 16), (1.0, 60 / 16)]),                  # 跳んで着地
    "dance":   ("c5_dance",  [(0.0, 6 / 16), (0.4, 17 / 16), (1.06, 41 / 16), (1.68, 56 / 16), (2.0, 64 / 16)]),  # 「Dance」で頭上の X、次の「Dance」で手をたたく、「Forever」で V
    "crew":    ("c6_punch",  [(0.0, 6 / 16), (1.0, 30 / 16)]),                                   # こぶしを交互に突く（1.5倍速）
    "crest":   ("c8o_wink",  [(0.0, 18 / 16), (0.35, 29 / 16), (1.0, 44 / 16)]),                # 目を閉じて笑う →「Rooftop」で目を開け、カメラを見て終わる
}
CLOSE_H = 1.25   # 最後の寄り（腰から上のクリップ）の大きさ

SHOT_START = {"product": 0.0, "lyrics1": 1.0, "lyrics2": 7.0, "build": 11.0, "dance": 12.0, "crew": 14.0, "crest": 15.0}

# 歌詞が歌われる秒（曲を文字起こしして測った。faster-whisper medium の単語の時刻）→ 画面の歌詞もこの秒に出す
L_ROKU, L_CHIME, L_KAIDAN, L_KAKE = 0.05, 1.55, 3.25, 5.1        # 六時間目の / チャイムで / 階段を / 駆け上がる
L_SORA, L_STAGE = 7.15, 9.2                                       # 空に一番 / 近いステージ
L_DANCE_FILL = [(12.4, 12.95), (13.06, 13.6)]                     # 「Dance」「Dance」

class Cast:
    """カットごとのダンスを必要になったときに読み込む"""
    def __init__(self, root="cache/cuts"):
        self.root, self.d = root, {}

    def at(self, shot, t):
        """動画の秒 t のときの (Dancer, 素材のコマ番号)"""
        name, keys = DANCE[shot]
        if name not in self.d: self.d[name] = Dancer(f"{self.root}/{name}")
        D = self.d[name]; u = t - SHOT_START[shot]
        s = float(np.interp(u, [k[0] for k in keys], [k[1] for k in keys]))
        return D, min(D.n - 1, max(0, int(s * D.fps)))

def body_height(p):
    """床に立つ人の身長（絵のピクセル）。地平線がカメラの高さ＝身長の p.eye の所に来る"""
    return (p.floor_y - p.horizon_y) / p.eye

def frame_on(p, fx, body_frac, foot_y, screen_x, rot=0.0):
    """キャラ（床の点 fx）の身長が画面の高さの body_frac、足が画面の foot_y（割合）、横が screen_x（割合）に来るカメラ"""
    s = body_frac * H / body_height(p)
    return Cam(p, fx - (screen_x - 0.5) * W / s, p.floor_y - (foot_y - 0.5) * H / s, s * p.w / W, rot)

# ---------- HUD ----------
def hud(c, t, full=True, logo_outline=False):
    a = 0.92
    # 画面の縁の目盛り
    for x in range(40, W - 30, 48):
        cv2.line(c, (x, 8), (x, 14 if x % 240 else 20), (235, 240, 255), 1)
    for y in range(120, H - 90, 48):
        cv2.line(c, (8, y), (14 if y % 240 else 20, y), (235, 240, 255), 1); cv2.line(c, (W - 9, y), (W - (15 if y % 240 else 21), y), (235, 240, 255), 1)
    text(c, "ROOFTOP DANCE CREW", 48, 38, 24, typo.HUD, alpha=a, spacing=2, shadow=0.5)
    for k in range(10):   # 点の列
        x = 48 + k * 19; on = (int(t * 4) + k) % 3 != 0
        cv2.rectangle(c, (x, 72), (x + 11, 79), (255, 255, 255) if on else (150, 165, 195), -1)
    xs = np.arange(0, 200, 5); hts = (np.abs(np.sin(xs * 0.37 + t * 7)) * 10 + 2).astype(int)   # 小さなレベルメーター
    for x, h_ in zip(xs, hts): cv2.line(c, (48 + x, 104), (48 + x, 104 - h_), (235, 240, 255), 2)
    x0 = 382
    w, _ = text(c, "RT", x0, 30, 62, typo.WIDE, alpha=a, shadow=0.5)
    text(c, "DC", x0 + w + 4, 30, 62, typo.WIDE, fill=None if logo_outline else WHITE, stroke=WHITE, sw=2, alpha=a)
    # 右上：♥ BPM ▶ 時間 と 波形（▶ は絵で描く。フォントに無いことがある）
    sec = int(t) % 60
    w2, _ = text(c, f"00:00:{sec:02d}", W - 48, 38, 24, typo.HUD, anchor="rt", alpha=a, shadow=0.5)
    tx = W - 48 - w2 - 30
    cv2.fillPoly(c, [np.array([[tx, 40], [tx, 60], [tx + 16, 50]], np.int32)], (255, 255, 255), cv2.LINE_AA)
    w3, _ = text(c, "♥ 120 BPM", tx - 22, 38, 24, typo.JP, anchor="rt", alpha=a, shadow=0.5)
    xs = np.arange(0, 190, 2); ph = t * 9
    ys = 78 + (np.sin(xs * 0.19 + ph) * 5 + np.sin(xs * 0.07 - ph * 0.6) * 4) * (0.5 + 0.5 * np.abs(np.sin(xs * 0.05 + t * 3)))
    pts = np.stack([W - 48 - 190 + xs, ys + 14], 1).astype(np.int32)
    cv2.polylines(c, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
    if full:   # 下の細い線と、進む印
        y = H - 64; cv2.line(c, (44, y), (W - 44, y), (235, 240, 255), 1, cv2.LINE_AA)
        x = int(44 + (W - 88) * (t / DUR)); cv2.rectangle(c, (x - 1, y - 5), (x + 1, y + 5), (255, 255, 255), -1)

def name_tag(c, hx, hy, t0, t):
    """キャラの頭の上の名札（細い線＋小さな文字）"""
    u = out_cubic((t - t0) / 0.3)
    if u <= 0: return
    # 頭の右横から線を引き出して、先に名前を置く
    x, y = int(hx + 110), int(hy + 70)
    L = int(260 * u)
    cv2.line(c, (x - 40, y + 40), (x, y), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(c, (x, y), (x + L, y), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(c, (x - 40, y + 40), 4, (255, 255, 255), -1, cv2.LINE_AA)
    text(c, "01 — PIN / CENTER", x + 8, y - 34, 26, typo.HUD, alpha=u, spacing=1, shadow=0.7)
    text(c, "ROOFTOP DANCE CREW  ·  ♪ 120", x + 8, y + 10, 17, typo.JP, alpha=u * 0.9, spacing=1, shadow=0.7)

def square_grid(c, t, x0, y0, cols, rows, mask_fn=None, seed=3):
    """小さな四角のグリッド（線の四角と、ところどころ塗った四角）。0.25秒ごとに点く所が入れ替わる"""
    r = np.random.RandomState(seed + int(t * 4))
    sz, g = 28, 12
    ov = np.zeros((H, W), np.float32)
    for j in range(rows):
        for i in range(cols):
            p = r.rand()
            if p < 0.35: continue
            fade = mask_fn(i, j) if mask_fn else 1.0
            if fade <= 0.02: continue
            x, y = x0 + i * (sz + g), y0 + j * (sz + g)
            if p > 0.88: cv2.rectangle(ov, (x, y), (x + sz, y + sz), 0.55 * fade, -1)
            else: cv2.rectangle(ov, (x, y), (x + sz, y + sz), 0.8 * fade, 2)
    c += (255 - c) * ov[..., None]

def lyric(c, s, x, y, px, t0, t, thin=False, anchor="lt"):
    """歌詞の1行：少し下から浮き上がって出る"""
    u = out_cubic((t - t0) / 0.22)
    if u <= 0: return
    font = typo.JP_THIN if thin else typo.JP
    im = word(s, px, font, WHITE, spacing=int(px * 0.02))
    # 薄い影を先に
    sh = im.copy(); sh[..., :3] = (20, 30, 60)
    sh = cv2.GaussianBlur(sh, (0, 0), 6)
    paste(c, sh, x + 4, y + 6 + 24 * (1 - u), 0.35 * u, anchor=anchor)
    paste(c, im, x, y + 24 * (1 - u), 0.96 * u, anchor=anchor)

# ---------- 共通：背景＋後ろの文字＋キャラ ----------
def stage(t, C, shot, key, fx, body_frac, foot_y, screen_x, *, back=None, rim=0.55, glow=None, shadow=True, rot=0.0):
    p = plate(key); cam = frame_on(p, fx, body_frac, foot_y, screen_x, rot)
    c = cam.render()
    if back and "text" in LAYERS: back(c, cam)
    D, i = C.at(shot, t)
    lay, fsy, s, sx = dancer_layer(D, i, cam, fx, body_height(p))
    if "dancer" in LAYERS:
        if shadow: put_shadow(c, cast_shadow(lay, fsy, s))
        put_dancer(c, lay, rim=rim, glow=glow if "fx" in LAYERS else None)
    return c, cam, lay

def head_of(lay):
    a = lay[..., 3] > 128
    rows = np.nonzero(a.any(1))[0]
    if not len(rows): return W / 2, H / 3
    top = rows.min(); xs = np.nonzero(a[top:top + 40].any(0))[0]
    return xs.mean(), top

# ---------- カット ----------
def shot_product(t, D):
    u = t / 1.0
    p = plate("low")
    c, cam, lay = stage(t, D, "product", "low", p.w * 0.5, 0.74 + 0.03 * u, 0.90, 0.5)
    sun_flare(c, *cam.to_screen(p.w * 0.93, p.h * 0.06), k=0.9)
    # 手前の缶2本（カメラより速く動かして手前に見せる）
    can = asset("can")
    for x, hh, blur in ((W * 0.15 - 70 * u, 0.98, 1.6), (W * 0.86 + 70 * u, 0.90, 2.4)):
        sc = H * hh / can.shape[0] * (1 + 0.06 * u)
        im = cv2.resize(can, (int(can.shape[1] * sc), int(can.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        if blur: im = cv2.GaussianBlur(im, (0, 0), blur)
        paste(c, im, x, H + 90, anchor="mb")
    lyric(c, "六時間目の", 110, 130, 160, L_ROKU, t)   # 商品カットだけは缶の手前
    return c

def shot_lyrics1(t, D):
    u = (t - 1.0) / 6.0
    def back(c, cam):
        lyric(c, "六時間目の", 110, 130, 160, L_ROKU, t)
        lyric(c, "チャイムで", 200, 330, 160, L_CHIME, t)
        lyric(c, "階段を", 120, 540, 112, L_KAIDAN, t, thin=True)
        lyric(c, "かけ上がる", 100, 700, 170, L_KAKE, t)
    c, cam, lay = stage(t, D, "lyrics1", "front", 1500, 0.84, 0.96, 0.68 - 0.08 * in_out(u), back=back)
    return c, lay

def shot_lyrics2(t, D):
    u = (t - 7.0) / 4.0
    p = plate("diag")
    fx = p.w * 0.55
    def back(c, cam):
        hx, hy = cam.to_screen(fx, p.floor_y)
        square_grid(c, t, int(hx - 470), 230, 8, 16, lambda i, j: clamp01(1 - abs(j - 7) / 10) * clamp01((i + 1) / 3))
        square_grid(c, t, int(hx + 190), 330, 5, 10, lambda i, j: clamp01(1 - abs(j - 5) / 7) * clamp01((5 - i) / 3), seed=11)
        lyric(c, "空にいちばん", W - 90, 600, 150, L_SORA, t, anchor="rt")
        lyric(c, "近いステージ", W - 90, 790, 150, L_STAGE, t, anchor="rt")
    c, cam, lay = stage(t, D, "lyrics2", "diag", fx, 0.95 + 0.05 * u, 1.14, 0.34 + 0.04 * u, back=back)
    hx, hy = head_of(lay)
    name_tag(c, hx, hy, 7.4, t)
    return c, lay

def shot_build(t, D):
    p = plate("build")
    u = (t - 11.0) / 1.0
    fx = p.w * 0.5
    def back(c, cam):
        big = word("ROOFTOP", 1900, typo.TALL, WHITE, spacing=40)
        v = (t - 11.0) / 1.0
        x = W + 150 - v * (big.shape[1] + W * 0.35)
        paste(c, big, x, H * 0.95, 0.97, anchor="lb")
    c, cam, lay = stage(t, D, "build", "build", fx, 0.64 + 0.04 * u, 0.95, 0.52 - 0.04 * u, back=back)
    return c

def big_word(c, s, t, t0, fill, y=0.50, px=370, stretch=1.2):
    """DANCE・CREW：塗り（fill=True）と中抜きを拍で切り替え、赤と青をずらして重ねる"""
    im = word(s, px, typo.WIDE, WHITE if fill else None, stroke=WHITE, sw=0 if fill else 5, stretch=stretch)
    frac = ((t - t0) % SPB) / SPB
    d = int(10 + 18 * punch(frac, 6))
    layer = np.zeros((H, W, 3), np.float32)
    paste(layer, im, W / 2, H * y, 1.0, anchor="mm")
    a = layer.max(2, keepdims=True) / 255
    red = np.roll(layer, -d, axis=1) * np.array([1, 0.25, 0.45], np.float32)
    blue = np.roll(layer, d, axis=1) * np.array([0.25, 0.6, 1], np.float32)
    screen_add(c, red, 0.55); screen_add(c, blue, 0.55)
    over(c, layer, a * (0.95 if fill else 0.9))

def shot_dance(t, D):
    p = plate("front2")
    u = (t - 12.0) / 3.0
    fx = p.w * 0.5
    crew = t >= 14.0
    word_ = "CREW" if crew else "DANCE"
    t0 = 14.0 if crew else 12.0
    k = int((t - t0) / SPB)
    # DANCE は「Dance」と歌っている間だけ塗り、CREW は拍ごとに切り替え
    fill = (k % 2 == 1) if crew else any(a <= t < b for a, b in L_DANCE_FILL)
    # 揺れ（CREW の場面）：拍ごとにカメラを振る
    sx = sy = 0.0; blur = 0
    if crew:
        frac = ((t - 14.0) % SPB) / SPB; r = np.random.RandomState(k)
        amp = punch(frac, 5) * 40
        sx, sy = (r.rand() - 0.5) * 2 * amp, (r.rand() - 0.5) * amp
        blur = amp * 0.9
    def back(c, cam): big_word(c, word_, t, t0, fill)
    frac0 = ((t - t0) % SPB) / SPB
    c, cam, lay = stage(t, D, "crew" if crew else "dance", "front2", fx, 1.12 + 0.08 * u + 0.03 * punch(frac0, 7), 1.18 + 0.07 * u + sy / H, 0.5 + sx / W,
                        back=back, glow=(1.0, 12 + int(10 * punch(frac0, 5))), shadow=False)
    if blur > 3: c = motion_blur(c, blur)
    if crew and frac0 < 0.16: glitch(c, 100 + k, 0.8)
    if t < 12.35: c += (255 - c) * (1 - (t - 12.0) / 0.35) ** 1.5 * 0.95   # 白フラッシュ
    return c

def shot_crest(t, D):
    p = plate("top")
    u = (t - 15.0) / 1.0
    cam = Cam(p, p.w * 0.5, p.h * 0.5, 1.05 + 0.04 * u)
    c = cam.render(blur=1.2)
    # キャラ：腰から上を大きく（素材の画面の高さを 2300px に）
    D, i = D.at("crest", t)
    im = D.img[i]; m = D.meta[i]
    if DANCE["crest"][0].startswith("c8"):
        # 腰から上で生成したクリップ：素材の画面の高さを画面の CLOSE_H 倍に、頭の上に少し空きを残す
        k = CLOSE_H * H / D.fh; top = D.meta[0]["y"]
        M = np.array([[k, 0, W / 2 - D.fw / 2 * k + m["x"] * k], [0, k, 60 - 30 * u - top * k + m["y"] * k]], np.float32)
    else:
        k = 2160 / (D.body * D.fh)                 # 全身のクリップ：身長が画面の2倍（腰から上が画面に入る）
        M = np.array([[k, 0, W / 2 - D.fw / 2 * k + m["x"] * k], [0, k, 70 - 40 * u - D.head * k + m["y"] * k]], np.float32)
    lay = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0)).astype(np.float32)
    # 画面の下で切れる所はそのまま（寄りの画）
    put_dancer(c, lay, rim=0.4, wrap=0.3)
    # 紋章
    v = out_back((t - 15.05) / 0.3)
    if v > 0:
        cr = asset("crest")
        # 胸の前の X を隠さないよう、腰の高さに小さめに置く
        paste(c, cv2.GaussianBlur(cr, (0, 0), 10) * np.array([0, 0, 0, 0.5], np.float32), W / 2 + 6, H * 0.84 + 10, 1.0, scale=0.44 * v, anchor="mm")
        paste(c, cr, W / 2, H * 0.84, 1.0, scale=0.44 * v, anchor="mm")
    # 縦書き
    a = clamp01((t - 15.15) / 0.25)
    vt = typo.vertical("屋上舞踊団", 78, typo.JP)
    sh = vt.copy(); sh[..., :3] = (20, 25, 50); paste(c, cv2.GaussianBlur(sh, (0, 0), 5), W - 64, 176, a * 0.5, anchor="rt")
    paste(c, vt, W - 70, 170, a, anchor="rt")
    lv = word("ROOFTOP  DANCE  CREW   ·   EST. 2026", 18, typo.HUD, WHITE, spacing=4)
    paste(c, np.ascontiguousarray(np.rot90(lv, -1)), 36, 150, a * 0.85)
    # 少し色を落として暖かく
    g = c.mean(2, keepdims=True); c[:] = c * 0.85 + g * 0.15
    c *= np.array([1.04, 1.0, 0.95], np.float32)
    return c

SHOTS = [(0.0, 1.0, "product"), (1.0, 7.0, "lyrics1"), (7.0, 11.0, "lyrics2"), (11.0, 12.0, "build"),
         (12.0, 15.0, "dance"), (15.0, 16.0, "crest")]

def frame(t, D):
    t = t % DUR
    name = next(n for a, b, n in SHOTS if a <= t < b)
    if name == "product": c = shot_product(t, D)
    elif name == "lyrics1": c, _ = shot_lyrics1(t, D)
    elif name == "lyrics2": c, _ = shot_lyrics2(t, D)
    elif name == "build": c = shot_build(t, D)
    elif name == "dance": c = shot_dance(t, D)
    else: c = shot_crest(t, D)
    if "fx" in LAYERS:
        bloom(c, 205, 0.3)
        vignette(c, 0.15)
        hud(c, t, full=True, logo_outline=t >= 12.0)
        chroma(c, 1)
    return np.clip(c, 0, 255).astype(np.uint8)
