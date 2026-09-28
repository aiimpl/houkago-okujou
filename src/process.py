# 作り方の動画（18秒）
#   曲は頭から途中で切らずに流し、どの場面も「その瞬間に歌っている所」と中身をそろえる
#     0–1   六時間目の             完成版の頭に表題
#     1–4   チャイムで・階段を      ① コンテ：決めの振り7つ
#     4–7   駆け上がる             ② ダンス生成：プロンプト → グリーンバックのクリップ → 7本
#     7–11  空に一番・近いステージ  ③ 曲：今の歌の位置で波形が流れ、歌われた瞬間に歌詞と秒が出る
#     11–12 （伸ばし）             ④ 合成：背景 → キャラ → 文字 → 仕上げ を半拍ずつ
#     12–14 Dance Dance Forever   ⑤ 合わせる：生成したままの動きと完成版を並べる（音と塗りと振りがそろう）
#     14–15                       完成版の CREW
#     15–18 Rooftop Dance Crew    最後の寄り → 使ったものとリポジトリ
#   絵はどれも本番の素材・本番のコードから出す（④は scenes.LAYERS を切り替えて描く）
#   先に本編を書き出しておく（make）。python src/process.py [out/process.mp4]　／ --sheet で見本の一覧
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from comp import W, H, FPS, SPB, out_cubic, out_back, clamp01
import typo
from typo import paste, word
import scenes
from scenes import Cast, DANCE

DUR = 18.0
SONG = json.load(open("gen/CHOICE.json"))["_song"]
REPO = "github.com/aiimpl/houkago-okujou"
STEPS = [  # (始まり, 終わり, 見出し, 補足)
    (0.0, 1.0, "", ""),
    (1.0, 4.0, "① コンテ：カットごとに決めの振りを1つ", "キャラのX字ヘアピンがモチーフ。決めは歌詞の秒に置く"),
    (4.0, 7.0, "② ダンスは1カットずつ生成", "Wan2.2・グリーンバック。RunPod の A100 で25本作って7本を選んだ"),
    (7.0, 11.0, "③ 曲も歌入りで生成して、歌詞の秒を測る", "ACE-Step 1.5 → 文字起こしで、どの秒に何を歌うか"),
    (11.0, 12.0, "④ 背景 → キャラ → 文字 → 仕上げ", "背景は GPT Image 2.5。合成は Python"),
    (12.0, 14.0, "⑤ 決めのコマを、歌詞の秒へ", "区間ごとに再生速度を変える"),
    (14.0, 15.0, "", ""),
    (15.0, 18.0, "", ""),
]
BG = np.array([14, 15, 20], np.float32)

def prompt_of(name):
    """gen/build.py に書いたそのカットのプロンプト（動き）"""
    ns = {}; exec(open("gen/build.py").read().split('if __name__')[0], ns)
    return ns["CUTS"][name][2]

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

FINAL = Video("out/okujou.mp4")
CAST = Cast()
def final(t): return FINAL.at(min(int(t * FPS), int(16 * FPS) - 1)).copy()

def backdrop(t, dark=0.28, blur=18):
    """その秒の完成版をぼかして暗くした背景（場面がつながって見えるように）"""
    return cv2.GaussianBlur(final(t), (0, 0), blur) * dark + BG * (1 - dark)

def caption(c, head, sub, u):
    """左下の見出し：下に暗いグラデーション＋白の太字＋補足"""
    if not head: return
    a = out_cubic(u / 0.25)
    g = np.clip((np.arange(H) - (H - 330)) / 330, 0, 1)[:, None, None] * 0.8 * a
    c *= 1 - g
    paste(c, word(head, 56, typo.JP), 72 - 50 * (1 - a), H - 178, a)
    paste(c, word(sub, 32, typo.JP_THIN), 74 - 50 * (1 - a), H - 98, a * 0.92)

def label(c, s, x, y, px=26, anchor="mt", a=1.0, font=None, pill=0.0):
    if not s.strip(): return
    im = word(s, max(8, int(px)), font or typo.JP)
    if pill:   # 文字の下に暗い帯
        h, w = im.shape[:2]; x0 = int(x - w / 2 - 22 if anchor[0] == "m" else x - 22 if anchor[0] == "l" else x - w - 22); y0 = int(y - 12)
        sub = c[max(0, y0):max(0, y0 + h + 24), max(0, x0):max(0, x0 + w + 44)]; sub *= 1 - pill * a
    paste(c, im, x, y, a, anchor=anchor)

def pose(name, i, body_px):
    """切り抜いたコマ（キャラの外接矩形）を、身長が body_px になる大きさで"""
    D = CAST.d.get(name) or CAST.at(next(s for s, v in DANCE.items() if v[0] == name), 0)[0]
    k = body_px / (D.body * D.fh); im = D.img[i]
    return cv2.resize(im, (max(1, int(im.shape[1] * k)), max(1, int(im.shape[0] * k))), interpolation=cv2.INTER_AREA)

_raw = {}
def raw(name, size):
    """生成したままのクリップ（緑のまま）"""
    key = (name, size)
    if key not in _raw:
        cap = cv2.VideoCapture(json.load(open("gen/CHOICE.json"))[name]["file"]); fr = []
        while True:
            ok, x = cap.read()
            if not ok: break
            fr.append(cv2.resize(x, size, interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32))
        _raw[key] = fr
    return _raw[key]

def pop(t, t0, d=0.22): return out_back((t - t0) / d)

# ---------- 場面 ----------
def s_open(t):
    c = final(t)
    a = clamp01(1 - (t - 0.75) / 0.25)
    c *= 1 - 0.35 * a
    label(c, "放課後の屋上で踊る16秒", W / 2, H * 0.60, 92, a=a, pill=0.55)
    label(c, "その作り方", W / 2, H * 0.60 + 136, 44, a=a, pill=0.55)
    return c

BOARD = [  # (カット名, コマ, 秒, 歌, 振り)
    ("c1_open", 2, "0.0", "六時間目の", "腕のX"),
    ("c2b_chime", 6, "1.6", "チャイムで", "指でチャイム"),
    ("c2b_chime", 66, "5.1", "駆け上がる", "もも上げ"),
    ("c3_sky", 26, "7.2", "空に一番", "空を指さす"),
    ("c5_dance", 17, "12.4", "Dance", "頭上でX"),
    ("c5_dance", 56, "13.7", "Forever", "大きくV"),
    ("c1_open", 2, "15.0", "Rooftop", "Xで終わる"),
]
def s_board(t):
    c = backdrop(1.0 + t)
    n = len(BOARD); cw = (W - 120) / n
    for k, (name, i, sec, lyr, move) in enumerate(BOARD):
        s = pop(t, 0.25 * k)                                    # 半拍ずつ出す
        if s <= 0: continue
        x = 60 + cw * (k + 0.5)
        if k: cv2.line(c, (int(x - cw / 2), 150), (int(x - cw / 2), 740), (60, 64, 80), 1)
        paste(c, pose(name, i, 420 * s), x, 630, min(1, s), anchor="mb")
        label(c, move, x, 652, 34, a=min(1, s))
        label(c, f"{sec}s  {lyr}", x, 704, 24, a=min(1, s) * 0.85, font=typo.JP_THIN)
    return c

def s_gen(t):
    c = backdrop(4.0 + t)
    if t < 1.6:                                                  # 1本を大きく：プロンプトが打ち込まれて、動き出す
        fr = raw("c5_dance", (520, 760))
        text = prompt_of("c5_dance"); n = int(len(text) * clamp01(t / 1.1))
        lines, cur = [], ""
        for w in text[:n].split(" "):
            if len(cur) + len(w) > 44: lines.append(cur); cur = w
            else: cur = (cur + " " + w).strip()
        lines.append(cur)
        label(c, "prompt", 150, 150, 26, anchor="lt", font=typo.HUD, a=0.7)
        y = 200
        for ln in lines[-12:]:
            label(c, ln, 150, y, 32, anchor="lt", font=typo.HUD); y += 46
        im = fr[min(len(fr) - 1, int(max(0, t - 0.5) * 16 * 2))]    # 打ち込みの途中から2倍速で動く
        a = clamp01((t - 0.35) / 0.25)
        c[140:900, 1250:1770] = c[140:900, 1250:1770] * (1 - a) + im * a
        label(c, "832×1216・16fps", 1510, 912, 22, font=typo.HUD, a=a * 0.8)
        return c
    u = t - 1.6                                                  # 7本を画面いっぱいに
    names = ["c1_open", "c2b_chime", "c3_sky", "c4b_swing", "c5_dance", "c6_punch", "c8o_wink"]
    cw, ch = 262, 383
    for k, name in enumerate(names):
        s = clamp01((u - 0.08 * k) / 0.2)
        if s <= 0: continue
        fr = raw(name, (cw, ch)); im = fr[int(u * 16 * 1.5 + k * 7) % len(fr)]
        x = 35 + k * (cw + 8); y = 250
        c[y:y + ch, x:x + cw] = c[y:y + ch, x:x + cw] * (1 - s) + im * s
    return c

WORDS = [(0.0, "六時間目の"), (1.58, "チャイムで"), (3.26, "階段を"), (5.14, "駆け上がる"), (7.18, "空に一番"),
         (9.22, "近いステージ"), (12.40, "Dance"), (13.06, "Dance"), (13.68, "Forever"), (15.00, "Rooftop Dance Crew")]
_env = None
def s_song(t):
    """曲の今の位置（7〜11秒）を真ん中に波形を流す。歌われた瞬間に歌詞と秒が出る"""
    global _env
    if _env is None:
        rawb = subprocess.run(["ffmpeg", "-v", "error", "-i", SONG["file"], "-f", "s16le", "-ac", "1", "-ar", "8000", "-"], capture_output=True, check=True).stdout
        x = np.frombuffer(rawb, np.int16).astype(np.float32) / 32768
        x = x[int(SONG["start"] * 8000):]
        hop = 80; n = len(x) // hop
        _env = np.abs(x[:n * hop]).reshape(n, hop).max(1)       # 10ms ごとの振れ幅
    c = backdrop(7.0 + t, dark=0.2)
    now = 7.0 + t; span = 6.0; cy = 470
    pps = W / span; left = now - span * 0.5
    for j in range(0, W, 3):
        sec = left + j / pps
        if sec < 0: continue
        v = _env[min(len(_env) - 1, int(sec * 100))]; hgt = int(v * 300)
        cv2.line(c, (j, cy - hgt), (j, cy + hgt), (130, 215, 255) if sec <= now else (70, 76, 92), 2)
    cv2.line(c, (W // 2, cy - 330), (W // 2, cy + 330), (255, 255, 255), 2)
    for s in range(int(left), int(left + span) + 2):
        xs = int((s - left) * pps)
        if 0 <= xs < W: label(c, f"{s}s", xs, cy + 345, 20, a=0.5, font=typo.HUD)
    for sec, w in WORDS:
        if sec > now or sec < left - 2: continue
        xs = (sec - left) * pps; a = clamp01((now - sec) / 0.15)
        cv2.line(c, (int(xs), cy - 330), (int(xs), cy - 250), (255, 220, 120), 3)
        label(c, w, xs + 12, cy - 330, 64 * min(1.1, pop(now, sec, 0.25)), anchor="lt", a=a)
        label(c, f"{sec:.2f}s", xs + 14, cy - 250, 28, anchor="lt", a=a, font=typo.HUD)
    return c

def s_layers(t):
    k = min(3, int(t / (SPB / 2)))                               # 半拍ごとに1層ずつ
    sets = [{"bg"}, {"bg", "dancer"}, {"bg", "dancer", "text"}, {"bg", "dancer", "text", "fx"}]
    names = ["背景", "＋キャラ", "＋文字", "＋仕上げ"]
    scenes.LAYERS = sets[k]
    try: c = scenes.frame(11.35, CAST).astype(np.float32)
    finally: scenes.LAYERS = sets[3]
    label(c, names[k], W / 2, 36, 44, pill=0.6)
    return c

def s_sync(t):
    """左：生成したまま（一定の速さ）、右：完成版の12〜14秒（音と一緒に）"""
    c = np.zeros((H, W, 3), np.float32) + BG
    fr = raw("c5_dance", (430, 628))
    c[150:778, 60:490] = fr[min(len(fr) - 1, int((6 / 16 + t) * 16))]
    c[100:831, 560:1860] = cv2.resize(final(12.0 + t), (1300, 731), interpolation=cv2.INTER_AREA)
    label(c, "生成したまま", 275, 92, 30, a=0.9); label(c, "歌に合わせた後", 1210, 40, 30, a=0.9)
    now = 12.0 + t; hit = [(s, w) for s, w in WORDS[6:9] if s <= now]
    if hit:
        s, w = hit[-1]
        label(c, "♪ " + w, 1210, 745, 56 * min(1.15, pop(now, s, 0.2)), pill=0.5)
    return c

def s_crew(t): return final(14.0 + t)

def s_end(t):
    if t < 1.0: return final(15.0 + t)
    f = final(15.99); z = 1 + 0.03 * (t - 1.0)
    c = cv2.warpAffine(f, cv2.getRotationMatrix2D((W / 2, H * 0.4), 0, z), (W, H), borderMode=cv2.BORDER_REFLECT)
    a = clamp01((t - 1.0) / 0.4); c *= 1 - 0.55 * a
    label(c, "全部コードです", W / 2, 330, 70, a=a)
    label(c, REPO, W / 2, 440, 40, a=a, font=typo.HUD)
    label(c, "ダンス Wan2.2 ／ 曲 ACE-Step 1.5 ／ 背景 GPT Image 2.5 ／ 合成・演出 Claude Opus 5.5", W / 2, 540, 30,
          a=clamp01((t - 1.4) / 0.4) * 0.9, font=typo.JP_THIN)
    return c

SCENES = [s_open, s_board, s_gen, s_song, s_layers, s_sync, s_crew, s_end]

def frame(t):
    k = next(i for i, s in enumerate(STEPS) if s[0] <= t < s[1])
    a, b, head, sub = STEPS[k]
    c = SCENES[k](t - a)
    u = t - a
    if 0 < k and u < 0.1: c = c + (255 - c) * 0.5 * (1 - u / 0.1)   # 場面の頭を白く光らせる（拍の頭）
    caption(c, head, sub, u)
    if t > DUR - 0.6: c *= (DUR - t) / 0.6
    return np.clip(c, 0, 255).astype(np.uint8)

def audio(out):
    """曲を頭から20秒（切らずに流す）。最後の0.8秒で絞る"""
    s = SONG["start"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(s), "-t", str(DUR), "-i", SONG["file"],
                    "-af", f"afade=t=out:st={DUR - 0.8}:d=0.8", out], check=True)

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
        ts = [0.4, 2.9, 4.9, 6.4, 7.6, 9.6, 11.1, 11.8, 12.5, 13.2, 14.4, 16.6]
        tiles = [cv2.resize(frame(s), (640, 360)) for s in ts]
        cv2.imwrite("out/process_sheet.png", cv2.cvtColor(np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]), cv2.COLOR_RGB2BGR))
        print("out/process_sheet.png")
    else:
        main(*sys.argv[1:2])
