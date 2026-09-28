# 合成の部品：背景の絵をカメラで切り取る・キャラを床に立たせる（影・逆光のふち・光のにじみ）・光の演出・仕上げ
#   座標は「背景の絵のピクセル」で考える。カメラが絵のどこをどの倍率で映すかを決め、キャラも文字も同じカメラで画面に写す
import json, math
import numpy as np, cv2

W, H = 1920, 1080
FPS = 30
BPM = 120; SPB = 60 / BPM          # 1拍 0.5秒。32拍でちょうど16秒
DUR = 16.0

# ---------- イージング ----------
def clamp01(u): return min(1.0, max(0.0, u))
def out_cubic(u): u = clamp01(u); return 1 - (1 - u) ** 3
def in_out(u): u = clamp01(u); return u * u * (3 - 2 * u)
def out_back(u, c=1.7):
    u = clamp01(u); return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2
def punch(frac, k=7.0):
    """拍の瞬間に 1 になって減衰する"""
    return math.exp(-max(0.0, frac) * k)

# ---------- 背景の絵とカメラ ----------
class Plate:
    def __init__(self, path, floor_y, horizon_y):
        """floor_y … キャラを立たせる床の高さ（絵のピクセル）。horizon_y … 地平線の高さ（影の向きの目安）"""
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None: raise FileNotFoundError(path)
        self.img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img.shape[:2]
        self.floor_y, self.horizon_y = floor_y, horizon_y

class Cam:
    """絵の (cx, cy) を画面の中央に、倍率 zoom（1＝絵の横幅が画面の横幅）で映す。rot は度"""
    def __init__(self, plate, cx, cy, zoom=1.0, rot=0.0):
        self.p = plate; self.s = W / plate.w * zoom; self.cx, self.cy, self.rot = cx, cy, rot
        a = math.radians(rot); c, s = math.cos(a) * self.s, math.sin(a) * self.s
        # 絵 → 画面：画面 = R·S·(絵 − c) + 画面の中央
        self.M = np.array([[c, -s, W / 2 - c * cx + s * cy], [s, c, H / 2 - s * cx - c * cy]], np.float32)

    def to_screen(self, x, y):
        return self.M[0, 0] * x + self.M[0, 1] * y + self.M[0, 2], self.M[1, 0] * x + self.M[1, 1] * y + self.M[1, 2]

    def render(self, blur=0.0):
        out = cv2.warpAffine(self.p.img, self.M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if blur > 0.3: out = cv2.GaussianBlur(out, (0, 0), blur)
        return out.astype(np.float32)

# ---------- キャラ ----------
class Dancer:
    """key.py が書き出した切り抜き（cache/rgba/）を全コマ読み込んでおく
    body … 素材の画面の高さのうち、頭のてっぺんから足の先までの割合。省略すると最初のコマ（立ち姿）で測る"""
    def __init__(self, cache="cache", body=None):
        m = json.load(open(f"{cache}/rgba.json"))
        self.fps, self.fw, self.fh = m["fps"], m["w"], m["h"]
        self.meta = m["frames"]; self.n = len(self.meta)
        self.img = [cv2.cvtColor(cv2.imread(f"{cache}/rgba/{i:04d}.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA) for i in range(self.n)]
        # 床の高さ＝立っているコマの足の先の中央値（跳んだコマはこれより上に足がある）
        self.floor = float(np.median([f["foot"] for f in self.meta]))
        self.body = body or (self.floor - self.meta[0]["y"]) / self.fh
        self.head = self.floor - self.body * self.fh          # 立ち姿の頭のてっぺん（素材の画面の座標）

def over(dst, rgb, a):
    """dst（float RGB）に rgb を透明度 a（0〜1, HxWx1）で重ねる"""
    dst *= (1 - a); dst += rgb * a

def screen_add(dst, rgb, a=1.0):
    """スクリーン合成（明るくするだけ）"""
    dst[:] = 255 - (255 - dst) * (255 - rgb * a) / 255

def _warp(rgba, M):
    return cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def dancer_layer(D, i, cam, fx, body):
    """素材のコマ i を、床の点 (fx, 絵の floor_y) に立たせて画面に写した RGBA（float）と、画面での床の高さ・倍率を返す
    body … 身長（頭から足まで）を絵の何ピクセルにするか"""
    k = body / (D.body * D.fh)
    im = D.img[i]; m = D.meta[i]
    # 素材の画面の座標 → 絵の座標：床の中央（素材の横の中央, D.floor）が (fx, floor_y) に来る
    ox = fx - D.fw / 2 * k + m["x"] * k; oy = cam.p.floor_y - D.floor * k + m["y"] * k
    A = np.array([[k, 0, ox], [0, k, oy], [0, 0, 1]], np.float32)
    M = cam.M.astype(np.float32) @ A
    lay = _warp(im, M).astype(np.float32)
    sx, sy = cam.to_screen(fx, cam.p.floor_y)
    return lay, sy, cam.s * k, sx

def cast_shadow(lay, floor_sy, scale, shear=0.38, squash=0.30, blur=4, dark=0.5, ao=0.65):
    """床に落ちる影（float, HxWx1 の暗さ）。太陽は右上の奥なので、シルエットを床の線で折り返して左手前へ倒す
    scale … 素材1ピクセルが画面の何ピクセルか。ao … 足元の濃い影（接地感）：床から少し上までの靴の形を、ぼかして少し手前へずらしたもの"""
    a = lay[..., 3] / 255
    def fold(sh, sq):  # 影の点 (x', y') = (x − sh·(y − 床), 床 + sq·(床 − y))
        M = np.array([[1, sh, -sh * floor_sy], [0, -sq, (1 + sq) * floor_sy]], np.float32)
        return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    out = cv2.GaussianBlur(fold(shear, squash), (0, 0), blur) * dark
    if ao:
        band = a.copy(); band[: max(0, int(floor_sy - 45 * scale))] = 0
        d = max(1, int(8 * scale)); band = np.roll(band, d, axis=0)
        c = cv2.GaussianBlur(cv2.dilate(band, np.ones((1, max(3, int(24 * scale))), np.uint8)), (0, 0), max(1.5, 7 * scale))
        out = np.maximum(out, np.clip(c * 1.5, 0, 1) * ao)
    out[: max(0, int(floor_sy) - 8)] = 0                    # 床より上には落とさない
    return out[..., None]

SHADOW_TINT = np.array([0.50, 0.56, 0.72], np.float32)     # 影は空の青を映して少し青い

def put_shadow(canvas, sh):
    canvas *= 1 - sh * (1 - SHADOW_TINT)

def put_dancer(canvas, lay, *, rim=0.55, wrap=0.35, grade=(1.02, 1.0, 0.97), alpha=1.0, glow=None):
    """キャラを重ねる。rim … 右上からの日差しでふちを明るく。wrap … 背景の明るさを輪郭ににじませて馴染ませる
    glow=(強さ, ずらし量) … 赤と青のにじみ（色収差のようなオーラ）を後ろに敷く"""
    a = lay[..., 3:4] / 255 * alpha
    if a.max() <= 0: return
    rgb = lay[..., :3] * np.array(grade, np.float32)
    if glow:
        # 赤と青のふち（くっきり）＋ にじみ（ぼかし）を左右にずらして敷く
        g, d = glow
        a0 = lay[..., 3] / 255
        soft = (cv2.GaussianBlur(a0, (0, 0), 3) * 0.9 + cv2.GaussianBlur(a0, (0, 0), 16) * 0.6) * g * alpha
        red = np.roll(soft, -d, axis=1)[..., None]; blue = np.roll(soft, d, axis=1)[..., None]
        screen_add(canvas, np.clip(np.concatenate([red * 255, red * 60, red * 130], 2), 0, 255))
        screen_add(canvas, np.clip(np.concatenate([blue * 50, blue * 140, blue * 255], 2), 0, 255))
    if rim:
        # 輪郭の内側のうち、右上を向いている所だけ明るくする（シルエットを左下へずらして引いた差）
        a0 = lay[..., 3] / 255
        sh = np.zeros_like(a0); sh[6:, :-6] = a0[:-6, 6:]
        edge = np.clip(a0 - sh, 0, 1); edge = cv2.GaussianBlur(edge, (0, 0), 1.6)[..., None]
        rgb = rgb + (np.array([255, 246, 225], np.float32) - rgb) * edge * rim
    if wrap:
        bgsoft = cv2.GaussianBlur(canvas, (0, 0), 12)
        inner = 1 - cv2.GaussianBlur(lay[..., 3] / 255, (0, 0), 3)[..., None]   # 輪郭の内側だけ 1 に近い
        rgb = rgb * (1 - inner * wrap) + np.maximum(rgb, bgsoft) * inner * wrap
    over(canvas, rgb, a)

# ---------- 光・仕上げ ----------
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)

def sun_flare(canvas, x, y, k=1.0):
    """画面内の太陽：芯・大きなにじみ・光の筋・逆側へ並ぶゴースト"""
    d = np.hypot(_xx - x, _yy - y)
    glow = (np.exp(-d / 60) * 1.0 + np.exp(-d / 320) * 0.45) * k
    ang = np.arctan2(_yy - y, _xx - x)
    rays = (np.abs(np.sin(ang * 7 + 0.4)) ** 40 * 0.5 + np.abs(np.sin(ang * 11 + 1.3)) ** 60 * 0.35) * np.exp(-d / 700) * k
    L = (glow + rays)[..., None] * np.array([255, 244, 220], np.float32)
    # ゴースト：太陽と画面の中央を結ぶ線上に淡い輪
    for t, r, col in ((0.55, 70, (120, 200, 255)), (0.8, 30, (255, 200, 150)), (1.25, 110, (140, 255, 220)), (1.6, 45, (200, 170, 255))):
        gx, gy = x + (W / 2 - x) * t, y + (H / 2 - y) * t
        g = np.clip(1 - np.abs(np.hypot(_xx - gx, _yy - gy) - r * 0.7) / (r * 0.35), 0, 1) * 0.10 * k
        L += g[..., None] * np.array(col, np.float32)
    screen_add(canvas, np.clip(L, 0, 255))

def bloom(canvas, thresh=200, k=0.35, sigma=18):
    """明るい所をぼかして足す（全体をふわっと馴染ませる）"""
    b = np.clip(canvas - thresh, 0, None) * (255 / (255 - thresh))
    b = cv2.resize(cv2.GaussianBlur(cv2.resize(b, (W // 4, H // 4)), (0, 0), sigma / 4), (W, H))
    screen_add(canvas, b * k)

def vignette(canvas, k=0.18):
    d = ((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2
    canvas *= (1 - k * np.clip(d - 0.3, 0, None))[..., None]

def chroma(canvas, d):
    """色収差：赤と青だけ左右にずらす"""
    if d:
        canvas[..., 0] = np.roll(canvas[..., 0], d, axis=1); canvas[..., 2] = np.roll(canvas[..., 2], -d, axis=1)

def motion_blur(canvas, dx, dy=0):
    """(dx, dy) 方向に流すブレ（カメラを振ったとき）"""
    n = int(max(abs(dx), abs(dy)))
    if n < 2: return canvas
    k = np.zeros((2 * n + 1, 2 * n + 1), np.float32)
    cv2.line(k, (n - int(dx / 2), n - int(dy / 2)), (n + int(dx / 2), n + int(dy / 2)), 1.0, 1)
    return cv2.filter2D(canvas, -1, k / k.sum())

def glitch(canvas, seed, amount=1.0, band=(10, 60)):
    """横の帯に切って、帯ごとに左右へずらす"""
    r = np.random.RandomState(seed); y = 0
    while y < H:
        h = r.randint(*band)
        if r.rand() < 0.5 * amount:
            canvas[y:y + h] = np.roll(canvas[y:y + h], int((r.rand() - 0.5) * 120 * amount), axis=1)
        y += h
