#!/bin/bash
# Pod の上で、SSH から切り離して動かす（setsid）。進み具合は /tmp/okujou/status に1行ずつ書く
#   モデル・入力・出力・設定・DB はすべて /tmp（Pod の使い捨てディスク）。ネットワークボリュームがあっても書き込まない
#   ComfyUI 本体と Python 環境だけ、ボリュームのもの（v0.34.3 / venv-std）を読んで使う
set -u
J=/tmp/okujou; S=$J/status
say() { echo "$(date -u +%H:%M:%S) $*" >> $S; }
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $J/models $J/in $J/out $J/user $J/tmp
cp $J/inputs/* $J/in/
if [ -x /runpod-volume/venv-std/bin/python ]; then
  C=/runpod-volume/ComfyUI; PY=/runpod-volume/venv-std/bin/python; say "0/4 ボリュームの ComfyUI を使う"
else
  # ボリュームなし：手元の Mac と同じ手順で、ComfyUI v0.34.3 と Python 環境を /tmp に作る
  say "0/4 ComfyUI v0.34.3 を用意（ボリュームなし）"
  C=$J/ComfyUI
  git clone -q --depth 1 --branch v0.34.3 https://github.com/comfyanonymous/ComfyUI.git $C > $J/setup.log 2>&1 || { say "失敗: git clone"; exit 1; }
  python3 -m venv $J/venv >> $J/setup.log 2>&1 || { say "失敗: venv"; exit 1; }
  PY=$J/venv/bin/python
  $PY -m pip install -q --upgrade pip >> $J/setup.log 2>&1
  if [ -f $J/freeze.txt ]; then  # 前回の Pod で記録した版の一覧があれば、それで入れる（同じシードで同じ動画にするため）
    $PY -m pip install -q -r $J/freeze.txt >> $J/setup.log 2>&1 || { say "失敗: freeze.txt"; exit 1; }
  else
    $PY -m pip install -q torch==2.14.0 torchvision torchaudio >> $J/setup.log 2>&1 || { say "失敗: torch"; exit 1; }
    $PY -m pip install -q -r $C/requirements.txt "huggingface_hub[hf_xet]" >> $J/setup.log 2>&1 || { say "失敗: requirements"; exit 1; }
  fi
  mkdir -p $J/out/okujou && $PY -m pip freeze > $J/out/okujou/freeze.txt   # 入った版の一覧（fetch で一緒に持ち帰る）
  say "  $($PY -c 'import torch;print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())' 2>&1)"
fi

say "1/4 モデルの取得を開始（署名付き URL）"
t0=$(date +%s)
$PY $J/hfdl.py --fetch $J/direct.json --root $J/models >> $J/dl.log 2>&1
n=$(find $J/models -name "*.safetensors" | wc -l)
say "  取得 $n 本 / $(du -sh $J/models | cut -f1) / $(( $(date +%s) - t0 ))秒"
[ "$n" -ge "$($PY -c 'import json;print(len(json.load(open("/tmp/okujou/direct.json"))))')" ] || { say "失敗: モデルの取得（$J/dl.log）"; tail -5 $J/dl.log >> $S; exit 1; }

cat > $J/extra_model_paths.yaml <<EOF
okujou:
  base_path: $J/models
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  loras: loras
  
EOF

say "2/4 ComfyUI を起動"
cd $C && setsid $PY main.py --listen 127.0.0.1 --port 8288 --extra-model-paths-config $J/extra_model_paths.yaml \
  --input-directory $J/in --output-directory $J/out --temp-directory $J/tmp --user-directory $J/user \
  < /dev/null > $J/comfy.log 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:8288/system_stats > /dev/null && break; sleep 2; done
curl -sf localhost:8288/system_stats > /dev/null || { say "失敗: ComfyUI が起動しない"; exit 1; }
say "  ComfyUI 起動済み"

say "3/4 生成を開始"
$PY - >> $S 2>&1 <<'PY'
import glob, json, os, time, urllib.request
URL = "http://127.0.0.1:8288"
for f in sorted(glob.glob("/tmp/okujou/wf/*.json")):
    wf = json.load(open(f)); t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/prompt", data=json.dumps({"prompt": wf}).encode(), headers={"Content-Type": "application/json"})))
    if r.get("node_errors"): print(time.strftime("%H:%M:%S"), "  入力エラー", os.path.basename(f), json.dumps(r["node_errors"], ensure_ascii=False)[:500], flush=True); continue
    pr = r["prompt_id"]
    while True:
        time.sleep(3)
        h = json.load(urllib.request.urlopen(f"{URL}/history/{pr}"))
        if pr in h:
            st = h[pr]["status"]; err = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            print(time.strftime("%H:%M:%S"), f"  {os.path.basename(f)}: {st.get('status_str')} {time.time()-t0:.0f}秒",
                  (err[0][1].get("exception_message", "")[:400] if err else ""), flush=True)
            break
PY
say "4/4 終了"
