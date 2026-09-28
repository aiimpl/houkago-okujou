# ダンス・最初のコマ・曲の生成（RunPod・使い捨ての Pod）

コンテ（[../conte/CONTE.md](../conte/CONTE.md)）のダンスを、カットごとに Wan2.2 I2V（14B・lightx2v 4step）で作る。
曲は ACE-Step 1.5 turbo。どちらも Pod の /tmp に ComfyUI v0.34.3 とモデルを置いて動かす（ボリュームは使わない。freeze.txt で Python パッケージの版を固定）。

## 手順
1. 最初のコマ（元のダンス素材を `python src/key.py` で切り抜いてから）
   - `python src/first_frame.py 324 gen/in/cross.png`（腕組み）
   - `python src/first_frame.py 228 gen/in/front.png`（正面）
   - 最後の寄り：`python src/first_frame.py close gen/out/c7_x_s128_00001_.mp4 26 gen/in/close_orig.png`（生成した c7 の目を開けた X のコマを、腰から上で切り出して拡大）
2. ワークフロー
   - ダンス：`cd gen && python build.py`（カット名を渡すとそのカットだけ）→ wf/*.api.json
   - 曲：`python build_stage.py` → wf/b_song_*.api.json
   - 最後の寄り：`python build_stage.py close close_orig.png` → wf3/
3. 署名付き URL（1時間ほどで切れる）：`python hfdl.py --resolve models.json -o direct.json`
4. Pod を立てる：image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 / A100 SXM 80GB / CUDA 13.0 以上 / disk 100GB / 22/tcp / startSsh
5. `bash pod.sh <host> <port> send`（準備2分＋モデル取得2分、そのあと wf/ を順に生成）→ `check` → `fetch`（gen/out/ に持ち帰る）
6. Pod を削除する

## 実績（2026-09-28）
- A100 SXM：Wan2.2 49コマ 約95秒／65コマ 約145秒／81コマ 約200秒／97コマ 約220秒。ACE-Step 32秒の曲 1本 数秒
- 3回の Pod で動画25本（ダンス18・最後の寄り7）と曲4本、最初のコマの画像編集4枚。GPU 代は約3ドル（$1.59/時）
- 選んだクリップと曲は CHOICE.json、選んだ理由はコンテの「選んだシード」と「作り直し」
- コマ番号入りの一覧：`python gen/sheet.py <動画> <出力.png> 3`
