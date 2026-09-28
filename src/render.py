# 書き出し：1コマずつ組み立てて ffmpeg にパイプで流し、H.264（yuv420p・テレビ範囲）の mp4 にして曲を合わせる
#   python src/render.py                    … out/okujou.mp4（1920×1080・30fps・16秒）
#   python src/render.py --preview          … 960×540 で速く
#   python src/render.py --still 6.2 10.4   … その秒のコマだけ out/still_6.20.png などに
#   python src/render.py --sheet            … 0.5秒ごとのコマを並べた一覧 out/sheet.png
import argparse, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from comp import W, H, FPS, DUR
from scenes import frame, Cast

ENC = ["-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "16",
       "-vf", "scale=out_color_matrix=bt709:in_range=pc:out_range=tv,format=yuv420p", "-pix_fmt", "yuv420p",
       "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]

def main():
    ap = argparse.ArgumentParser(description="屋上ダンスの動画を書き出す")
    ap.add_argument("-o", "--out", default="out/okujou.mp4")
    ap.add_argument("--preview", action="store_true", help="960×540 で書き出す")
    ap.add_argument("--still", type=float, nargs="+", help="その秒のコマだけ PNG に")
    ap.add_argument("--sheet", action="store_true", help="0.5秒ごとのコマの一覧")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--end", type=float, default=DUR)
    a = ap.parse_args()
    D = Cast(); os.makedirs("out", exist_ok=True)
    size = (W // 2, H // 2) if a.preview else (W, H)
    bgr = lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if a.still:
        for s in a.still:
            out = f"out/still_{s:05.2f}.png"; cv2.imwrite(out, bgr(frame(s, D))); print(out)
        return
    if a.sheet:
        ts = np.arange(0, DUR, 0.5) + 0.25
        tiles = [cv2.resize(frame(s, D), (480, 270), interpolation=cv2.INTER_AREA) for s in ts]
        rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
        cv2.imwrite("out/sheet.png", bgr(np.vstack(rows))); print("out/sheet.png"); return
    tmp = a.out + ".video.mp4"
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                           "-r", str(FPS), "-i", "-", *ENC, tmp], stdin=subprocess.PIPE)
    t0 = time.time(); n0, n1 = int(a.start * FPS), int(round(a.end * FPS))
    for n in range(n0, n1):
        img = frame(n / FPS, D)
        if size != (W, H): img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        ff.stdin.write(img.tobytes())
        if n % 30 == 0: print(f"\r{n}/{n1}", end="", flush=True)
    ff.stdin.close(); ff.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", str(a.start), "-t", str(a.end - a.start), "-i", "cache/music.wav",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], check=True)
    os.remove(tmp)
    print(f"\r{a.out}  {n1 - n0}コマ  {time.time() - t0:.0f}秒")

if __name__ == "__main__":
    main()
