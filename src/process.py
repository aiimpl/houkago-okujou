# 作り方の動画（20秒）：表題 → ① コンテ → ② ダンス生成 → ③ 曲 → ④ 歌に合わせる → ⑤ 合成 → 完成
#   絵はどれも本番の素材・本番のコードから出す（⑤の層を足す絵は scenes.LAYERS を切り替えて描く）
#   先に本編を書き出しておく（make）。python src/process.py [out/process.mp4]
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from comp import W, H, FPS, out_cubic, clamp01
import typo
from typo import paste, word
import scenes
from scenes import Cast, DANCE

DUR = 20.0
SONG = json.load(open("gen/CHOICE.json"))["_song"]
REPO = "github.com/aiimpl/houkago-okujou"

STEPS = [  # (始まり, 終わり, 見出し, 補足)
    (0.0, 2.0, "", ""),
    (2.0, 4.0, "① コンテ：カットごとに決めの振りを1つ", "キャラのX字ヘアピンがモチーフ。決めは歌詞の秒に置く"),
    (4.0, 7.0, "② ダンスは1カットずつ生成（グリーンバック）", "Wan2.2・RunPod の A100 で25本作って、7本を選んだ"),
    (7.0, 10.0, "③ 曲は歌入りで生成、歌詞の秒を測る", "ACE-Step 1.5。文字起こしで、どの秒に何を歌うかを測った"),
    (10.0, 12.0, "④ 決めのコマを、歌詞の秒へ", "区間ごとに再生速度を変える（頭上のX・手をたたく・V）"),
    (12.0, 15.0, "⑤ 背景 → キャラ → 文字と仕上げ", "背景は GPT Image 2.5 の6枚。合成・文字・HUD は Python"),
    (15.0, 20.0, "全部コードです", REPO),
]
BG = np.array([16, 16, 20], np.float32)

# ---------- 部品 ----------
class Video:
    """動画を前から順に読む（戻るときだけ開き直す）"""
    def __init__(self, path): self.path = path; self._open()
    def _open(self): self.cap = cv2.VideoCapture(self.path); self.i = -1; self.f = None
    def at(self, n):
        if n < self.i: self._open()
        while self.i < n:
            ok, f = self.cap.read()
            if not ok: break
            self.f = f[..., ::-1].astype(np.float32); self.i += 1
        return self.f

def caption(c, head, sub, u):
    """左下の見出し（juuen・byodoin と同じ形：下に暗いグラデーション＋白の太字＋小さな補足）"""
    if not head: return
    a = out_cubic(u / 0.3)
    g = np.clip((np.arange(H) - (H - 300)) / 300, 0, 1)[:, None, None] * 0.75 * a
    c *= 1 - g
    paste(c, word(head, 50, typo.JP), 70 - 40 * (1 - a), H - 165, a)
    paste(c, word(sub, 28, typo.JP_THIN), 72 - 40 * (1 - a), H - 92, a * 0.9)

def label(c, s, x, y, px=26, anchor="mt", a=1.0, font=None, pill=False):
    im = word(s, px, font or typo.JP)
    if pill:   # 文字の下に暗い帯（絵の上に置くとき）
        h, w = im.shape[:2]; x0 = int(x - w / 2 - 22 if anchor[0] == "m" else x - 22); y0 = int(y - 12)
        sub = c[max(0, y0):y0 + h + 24, max(0, x0):x0 + w + 44]; sub *= 1 - 0.6 * a
    paste(c, im, x, y, a, anchor=anchor)

def rgba_frame(D, i, h):
    """切り抜いたコマを、素材の画面ごと高さ h に縮めた RGBA"""
    im = D.img[i]; m = D.meta[i]; k = h / D.fh
    out = np.zeros((int(h), int(D.fw * k), 4), np.uint8)
    sm = cv2.resize(im, (max(1, int(im.shape[1] * k)), max(1, int(im.shape[0] * k))), interpolation=cv2.INTER_AREA)
    x, y = int(m["x"] * k), int(m["y"] * k)
    hh, ww = min(sm.shape[0], out.shape[0] - y), min(sm.shape[1], out.shape[1] - x)
    out[y:y + hh, x:x + ww] = sm[:hh, :ww]
    return out

# ---------- 場面 ----------
FINAL = Video("out/okujou.mp4")
CAST = Cast()

def s_title(t):
    f = FINAL.at(int(15.6 * FPS)).copy()
    c = cv2.GaussianBlur(f, (0, 0), 14) * 0.45
    a = out_cubic(t / 0.5)
    label(c, "放課後の屋上で踊る16秒", W / 2, H * 0.38, 76, a=a)
    label(c, "振付・ダンス・曲まで生成して、歌に合わせた工程", W / 2, H * 0.38 + 110, 34, a=a * 0.85, font=typo.JP_THIN)
    return c

BOARD = [  # (カット名, コマ, 秒と歌, 振り)
    ("c1_open", 2, "0.0  六時間目の", "腕のX"),
    ("c2b_chime", 6, "1.6  チャイムで", "指でチャイム"),
    ("c2b_chime", 66, "5.1  駆け上がる", "もも上げ"),
    ("c3_sky", 26, "7.2  空に一番", "空を指さす"),
    ("c5_dance", 17, "12.4  Dance", "頭上でX"),
    ("c5_dance", 56, "13.7  Forever", "大きくV"),
    ("c8o_wink", 44, "15.0  Rooftop", "Xで笑顔"),
]

def s_board(t):
    c = np.zeros((H, W, 3), np.float32) + BG
    n = len(BOARD); pw = 250; gap = (W - 120 - n * pw) / (n - 1)
    for k, (name, i, when, move) in enumerate(BOARD):
        a = out_cubic((t - 0.12 * k) / 0.3)
        if a <= 0: continue
        D = CAST.d.get(name) or CAST.at(next(s for s, v in DANCE.items() if v[0] == name), 0)[0]
        x = 60 + k * (pw + gap); y = 150 + 30 * (1 - a)
        cv2.rectangle(c, (int(x), int(y)), (int(x + pw), int(y + 560)), (34, 36, 44), -1)
        im = rgba_frame(D, i, 520)
        if im.shape[1] > pw - 10: im = cv2.resize(im, (pw - 10, int(im.shape[0] * (pw - 10) / im.shape[1])), interpolation=cv2.INTER_AREA)
        paste(c, im, x + pw / 2, y + 540, a, anchor="mb")
        label(c, when, x + pw / 2, y + 580, 26, a=a)
        label(c, move, x + pw / 2, y + 624, 30, a=a)
    return c

CLIPS = ["c1_open", "c2b_chime", "c3_sky", "c4b_swing", "c5_dance", "c6_punch", "c8o_wink"]
_raw = {}
def raw_clip(name):
    if name not in _raw:
        f = json.load(open("gen/CHOICE.json"))[name]["file"]
        cap = cv2.VideoCapture(f); fr = []
        while True:
            ok, x = cap.read()
            if not ok: break
            fr.append(cv2.resize(x, (250, 365), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32))
        _raw[name] = fr
    return _raw[name]

def s_gen(t):
    c = np.zeros((H, W, 3), np.float32) + BG
    n = len(CLIPS); gap = (W - 120 - n * 250) / (n - 1)
    for k, name in enumerate(CLIPS):
        a = out_cubic((t - 0.08 * k) / 0.3)
        if a <= 0: continue
        fr = raw_clip(name); im = fr[int(t * 16) % len(fr)]
        x = int(60 + k * (250 + gap)); y = int(200 + 30 * (1 - a))
        c[y:y + 365, x:x + 250] = c[y:y + 365, x:x + 250] * (1 - a) + im * a
        label(c, name.split("_")[0].upper(), x + 125, y + 380, 24, a=a, font=typo.HUD)
    label(c, "832×1216・16fps・1本 3〜6秒", W / 2, 110, 30, a=out_cubic(t / 0.4), font=typo.JP_THIN)
    return c

WORDS = [(0.0, "六時間目の"), (1.58, "チャイムで"), (3.26, "階段を"), (5.14, "駆け上がる"), (7.18, "空に一番"),
         (9.22, "近いステージ"), (12.40, "Dance"), (13.06, "Dance"), (13.68, "Forever"), (15.00, "Rooftop Dance Crew")]
_wave = None
def s_song(t):
    global _wave
    c = np.zeros((H, W, 3), np.float32) + BG
    if _wave is None:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", SONG["file"], "-f", "s16le", "-ac", "1", "-ar", "8000", "-"], capture_output=True, check=True).stdout
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
        x = x[int(SONG["start"] * 8000):int((SONG["start"] + 16) * 8000)]
        cols = W - 160; _wave = np.array([np.abs(x[int(j * len(x) / cols):int((j + 1) * len(x) / cols)]).max() for j in range(cols)])
    cy = 430; x0 = 80; sweep = clamp01(t / 2.6) * 16          # 3秒で16秒ぶんを走らせる
    for j, v in enumerate(_wave):
        sec = j / len(_wave) * 16; hgt = int(v * 260)
        col = (120, 210, 255) if sec <= sweep else (70, 74, 88)
        cv2.line(c, (x0 + j, cy - hgt), (x0 + j, cy + hgt), col, 1)
    px = int(x0 + sweep / 16 * len(_wave)); cv2.line(c, (px, cy - 300), (px, cy + 300), (255, 255, 255), 2)
    for k, (sec, w) in enumerate(WORDS):
        if sec > sweep: continue
        x = x0 + sec / 16 * len(_wave); up = k % 2 == 0
        a = out_cubic((sweep - sec) / 1.2)
        cv2.line(c, (int(x), cy - 280 if up else cy + 280), (int(x), cy - 320 if up else cy + 320), (255, 255, 255), 2)
        label(c, w, x, cy - 380 if up else cy + 330, 30, anchor="lt", a=a)
        label(c, f"{sec:.1f}", x, cy - 336 if up else cy + 375, 20, anchor="lt", a=a * 0.8, font=typo.HUD)
    for s in range(0, 17, 2):
        label(c, f"{s}s", x0 + s / 16 * len(_wave), cy + 410, 20, a=0.6, font=typo.HUD)
    return c

def s_sync(t):
    c = np.zeros((H, W, 3), np.float32) + BG
    u = (t % 2.0)                                              # 2秒をくり返す
    # 左：生成したまま（16fps をそのまま）、右：本編の12〜14秒（歌に合わせた後）
    fr = raw_clip("c5_dance"); left = fr[min(len(fr) - 1, int((6 / 16 + u) * 16))]
    left = cv2.resize(left, (500, 730))
    right = FINAL.at(int((12.0 + u) * FPS))
    right = cv2.resize(right, (1000, 562))
    c[150:880, 170:670] = left; c[234:796, 760:1760] = right
    label(c, "生成したまま", 420, 90, 32); label(c, "歌に合わせた後", 1260, 90, 32)
    for sec, w in WORDS[6:9]:
        if 12.0 + u >= sec: last = w
    if 12.0 + u >= 12.4: label(c, "♪ " + last, 1260, 820, 40)
    return c

def s_layers(t):
    k = min(3, int(t / 0.75))                                  # 0.75秒ごとに1層ずつ
    sets = [{"bg"}, {"bg", "dancer"}, {"bg", "dancer", "text"}, {"bg", "dancer", "text", "fx"}]
    names = ["背景（生成した絵）", "＋キャラ（影・逆光のふち）", "＋文字", "＋にじみ・HUD・色収差"]
    scenes.LAYERS = sets[k]
    try: c = scenes.frame(12.6, CAST).astype(np.float32)
    finally: scenes.LAYERS = sets[3]
    label(c, names[k], W / 2, 40, 40, pill=True)
    return c

def s_final(t):
    return FINAL.at(min(int((12.0 + t) * FPS), int(16 * FPS) - 1)).copy()

SCENES = [s_title, s_board, s_gen, s_song, s_sync, s_layers, s_final]

def frame(t):
    k = next(i for i, s in enumerate(STEPS) if s[0] <= t < s[1])
    a, b, head, sub = STEPS[k]
    c = SCENES[k](t - a)
    if t - a < 0.15 and k: c *= 0.4 + 0.6 * (t - a) / 0.15    # 場面の頭を少し暗くして切り替えを見せる
    caption(c, head, sub, t - a)
    return np.clip(c, 0, 255).astype(np.uint8)

def audio(out):
    """曲：頭から15秒 → 完成の場面は本編の12秒目からの音に戻す（映像と歌詞がそろうように）"""
    s = SONG["start"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SONG["file"], "-filter_complex",
                    f"[0]atrim={s}:{s + 15.04},asetpts=N/SR/TB[a];[0]atrim={s + 11.96}:{s + 17},asetpts=N/SR/TB,afade=t=out:st=4.2:d=0.8[b];"
                    "[a][b]acrossfade=d=0.08[o]", "-map", "[o]", "-t", str(DUR), out], check=True)

def main(out="out/process.mp4"):
    tmp = out + ".v.mp4"; wav = out + ".a.wav"
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-profile:v", "high", "-crf", "17", "-vf", "scale=out_color_matrix=bt709:in_range=pc:out_range=tv,format=yuv420p",
                           "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", tmp],
                          stdin=subprocess.PIPE)
    n1 = int(DUR * FPS)
    for n in range(n1):
        ff.stdin.write(frame(n / FPS).tobytes())
        if n % 30 == 0: print(f"\r{n}/{n1}", end="", flush=True)
    ff.stdin.close(); ff.wait()
    audio(wav)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); os.remove(wav); print("\r" + out)

if __name__ == "__main__":
    if sys.argv[1:2] == ["--sheet"]:
        ts = [0.8, 3.5, 5.5, 9.4, 11.0, 12.3, 13.1, 13.9, 14.7, 17.0]
        tiles = [cv2.resize(frame(s), (640, 360)) for s in ts]
        cv2.imwrite("out/process_sheet.png", cv2.cvtColor(np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]), cv2.COLOR_RGB2BGR))
        print("out/process_sheet.png")
    else:
        main(*sys.argv[1:2])
