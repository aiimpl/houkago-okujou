# 生成した曲から、映像に合わせて16秒を切り出して cache/music.wav にする
#   python src/pick_song.py <曲.flac|wav> --info          … 拍の位置と、小節ごとの音量の表
#   python src/pick_song.py <曲> <切り出し開始の秒>        … その秒から16秒（120 BPM の拍に合わせて微調整）
#   映像の山は 10.0秒（DANCE の白フラッシュ）。曲が盛り上がる小節の頭が 10.0秒に来るよう開始を選ぶ
import subprocess, sys, wave
import numpy as np

SR, BPM, DUR = 44100, 120, 16.0
SPB = 60 / BPM

def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).reshape(-1, 2).astype(np.float32) / 32768

def onset_env(x, hop=441):
    """10ms ごとの、音の立ち上がりの強さ"""
    m = x.mean(1); n = len(m) // hop
    e = np.sqrt((m[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-9)
    return np.maximum(0, np.diff(np.log(e), prepend=np.log(e[0]))), hop / SR

def beat_phase(x):
    """120 BPM の拍の位相（秒, 0〜SPB）：立ち上がりを拍の格子に畳んで、いちばん強い位置"""
    o, dt = onset_env(x)
    k = int(round(SPB / dt)); n = len(o) // k
    fold = o[:n * k].reshape(n, k).sum(0)
    return float(np.argmax(np.convolve(np.r_[fold, fold, fold], np.ones(3), "same")[k:2 * k]) * dt)

def info(path):
    x = load(path); ph = beat_phase(x)
    print(f"長さ {len(x) / SR:.1f}秒  拍の位相 {ph:.3f}秒")
    bar = 4 * SPB; t = ph
    while t + bar <= len(x) / SR:
        seg = x[int(t * SR):int((t + bar) * SR)]
        print(f"  小節 {t:6.2f}秒  音量 {20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9):6.1f} dB")
        t += bar

def cut(path, start, out="cache/music.wav"):
    x = load(path); ph = beat_phase(x)
    start = ph + round((float(start) - ph) / SPB) * SPB          # 拍の頭にそろえる
    y = x[int(start * SR):int(start * SR) + int(DUR * SR)].copy()
    if len(y) < DUR * SR: sys.exit(f"曲が短い：{start:.2f}秒から16秒取れない")
    f = int(0.01 * SR); y[:f] *= np.linspace(0, 1, f)[:, None]    # 頭のプチ音だけ消す
    g = int(0.25 * SR); y[-g:] *= np.linspace(1, 0, g)[:, None]   # 終わりは短く絞る（頭の商品カットへ戻る）
    y /= max(1e-6, np.abs(y).max() / 0.89)
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((y * 32767).astype(np.int16).tobytes())
    print(f"{out}：{path} の {start:.3f}秒から16秒")

if __name__ == "__main__":
    if sys.argv[2:3] == ["--info"]: info(sys.argv[1])
    else: cut(sys.argv[1], sys.argv[2])
