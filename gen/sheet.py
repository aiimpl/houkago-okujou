# 生成したクリップのコマ一覧（コマ番号入り）：python gen/sheet.py <動画> <出力.png> [何コマおき]
import sys
import numpy as np, cv2

def main(src, out, step=2):
    cap = cv2.VideoCapture(src); tiles = []; n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if n % int(step) == 0:
            t = cv2.resize(f[60:1140, 96:736], (160, 270))
            cv2.putText(t, str(n), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3); cv2.putText(t, str(n), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(t)
        n += 1
    cols = 14; tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]))

if __name__ == "__main__":
    main(*sys.argv[1:])
