# キーイング：グリーンバックの素材を1コマずつ切り抜いて、cache/rgba/ に保存する
#   透明度 … ピクセルごとに「緑が赤・青よりどれだけ強いか」で決める
#   デスピル … 縁に残る緑を、緑の値を赤・青の大きい方までに抑えて消す
#   端の処理 … 素材の端で切れた手足がまっすぐな線に見えないよう、端の近くをなめらかに透明にする
#   python src/key.py [素材.mp4] [出力先]（省略時は assets/dance_green.mp4 → cache/）
import json, os, sys
import numpy as np, cv2

SRC = "assets/dance_green.mp4"
OUT = "cache"
LO, HI = 18, 58     # 緑の強さがこの範囲で、不透明→透明へ
# 素材の端から何ピクセルでなめらかに消すか（左右・上・下）。左右は画面の外へ振った腕を消す。
# 上下は背景の上に全身で立たせるので、髪と足を削らないよう短くする
FEATHER_X, FEATHER_TOP, FEATHER_BOTTOM = 36, 8, 2

def key_frame(bgr):
    """BGR の1コマ → 切り抜いた RGBA"""
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    spill = g - np.maximum(r, b)
    a = 1 - np.clip((spill - LO) / (HI - LO), 0, 1)
    a = cv2.GaussianBlur(cv2.erode(a, np.ones((2, 2), np.uint8)), (3, 3), 0)  # 縁を1ピクセル締めて、少しぼかす
    f[..., 1] = np.minimum(g, np.maximum(r, b))                           # デスピル
    h, w = a.shape
    ramp = lambda n, f: np.clip(np.arange(n) / f, 0, 1)
    ex = np.minimum(ramp(w, FEATHER_X), ramp(w, FEATHER_X)[::-1])
    ey = np.minimum(ramp(h, FEATHER_TOP), ramp(h, FEATHER_BOTTOM)[::-1])
    a = a * np.minimum.outer(ey, ex) ** 0.7
    return np.dstack([f[..., 2], f[..., 1], f[..., 0], a * 255]).clip(0, 255).astype(np.uint8)

def cut_at_edge(bgr, margin=3):
    """キャラ（緑でない所）が素材の画面の端から margin ピクセル以内にかかっているか＝手足などが切れているか"""
    b, g, r = [bgr[..., k].astype(int) for k in range(3)]
    fg = (g - np.maximum(r, b)) < 40
    return bool(fg[:, :margin].any() or fg[:, -margin:].any() or fg[:margin].any() or fg[-margin:].any())

def main(src=SRC, OUT=OUT):
    os.makedirs(f"{OUT}/rgba", exist_ok=True)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): sys.exit(f"素材が開けません：{src}（README の「素材を用意する」を見てください）")
    fps, meta, size = cap.get(cv2.CAP_PROP_FPS), [], None
    while True:
        ok, bgr = cap.read()
        if not ok: break
        size = bgr.shape[:2]
        rgba = key_frame(bgr)
        ys, xs = np.nonzero(rgba[..., 3] > 8)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1      # 外側の透明な所は切り落として保存する
        cv2.imwrite(f"{OUT}/rgba/{len(meta):04d}.png", cv2.cvtColor(rgba[y0:y1, x0:x1], cv2.COLOR_RGBA2BGRA))
        foot = int(np.nonzero((rgba[..., 3] > 128).any(1))[0].max())             # 足の先（不透明ないちばん下の行）
        meta.append({"x": int(x0), "y": int(y0), "foot": foot, "cut": cut_at_edge(bgr)})
    json.dump({"fps": fps, "w": size[1], "h": size[0], "frames": meta}, open(f"{OUT}/rgba.json", "w"))
    print(f"{len(meta)}コマを切り抜き（{size[1]}×{size[0]}・{fps:g}fps）。画面の端で切れているコマ：{sum(m['cut'] for m in meta)}")

if __name__ == "__main__":
    main(*sys.argv[1:3])
