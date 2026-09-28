# 生成の最初のコマ：今の素材から1コマ切り抜いて、余白を広く取った無地のグリーンバックに置く
#   python src/first_frame.py <素材のコマ番号> <出力.png> [幅 高さ 身長の割合]
#   python src/first_frame.py close <生成したクリップ.mp4> <コマ番号> <出力.png>   … 最後の寄り：腰から上を切り出して拡大
#   腕を広げても・跳んでも画面の端に当たらないよう、今の素材（身長が画面の94%）より引いて置く
import json, sys
import numpy as np, cv2

GREEN = (64, 190, 80)   # RGB。素材の緑に近い無地

def main(idx, out, w=832, h=1216, frac=0.70):
    idx, w, h, frac = int(idx), int(w), int(h), float(frac)
    m = json.load(open("cache/rgba.json")); f = m["frames"][idx]
    im = cv2.cvtColor(cv2.imread(f"cache/rgba/{idx:04d}.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(np.float32)
    top = f["y"]; foot = f["foot"]
    k = frac * h / (foot - top)                                   # 頭の上から足の先までを画面の高さの frac に
    im = cv2.resize(im, (int(im.shape[1] * k), int(im.shape[0] * k)), interpolation=cv2.INTER_AREA)
    c = np.zeros((h, w, 3), np.float32); c[:] = GREEN
    x0 = int(w / 2 - (m["w"] / 2 - f["x"]) * k); y0 = int(h * 0.90 - (foot - top) * k)   # 足の先を画面の高さの 90% に
    H_, W_ = im.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(w, x0 + W_), min(h, y0 + H_)
    a = im[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, 3:4] / 255
    c[Y0:Y1, X0:X1] = im[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, :3] * a + c[Y0:Y1, X0:X1] * (1 - a)
    cv2.imwrite(out, cv2.cvtColor(c.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)); print(out, w, h)

def close(src, n, out, w=832, h=1216):
    """生成したクリップの1コマを、頭の上から身長の42%まで切り出して拡大する（画風をそのまま保つため、描き直さない）"""
    cap = cv2.VideoCapture(src)
    for _ in range(int(n) + 1): ok, f = cap.read()
    a = f[..., 1].astype(int) - np.maximum(f[..., 0], f[..., 2]).astype(int)
    ys, xs = np.nonzero(a < 40); top = ys.min(); body = ys.max() - top
    y0 = max(0, top - int(body * 0.05)); y1 = top + int(body * 0.42)
    cx = int(xs[(ys > top) & (ys < y1)].mean()); hh = y1 - y0; ww = int(hh * w / h); x0 = max(0, cx - ww // 2)
    big = cv2.resize(f[y0:y1, x0:x0 + ww], (w, h), interpolation=cv2.INTER_LANCZOS4)
    big = cv2.addWeighted(big, 1.5, cv2.GaussianBlur(big, (0, 0), 1.2), -0.5, 0)   # 拡大のぼけを輪郭だけ戻す
    cv2.imwrite(out, big); print(out, f"{h / hh:.2f}倍")

if __name__ == "__main__":
    if sys.argv[1:2] == ["close"]: close(*sys.argv[2:])
    else: main(*sys.argv[1:])
