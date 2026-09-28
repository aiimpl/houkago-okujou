#!/bin/bash
# 手元から Pod を操作する。どれも短い SSH で終わる（つなぎっぱなしにしない）
#   bash pod.sh <host> <port> send     … 入力とワークフローを送って、remote_job.sh を切り離して起動
#   bash pod.sh <host> <port> check    … 進み具合・GPU の使用率・ComfyUI のログの最後を見る
#   bash pod.sh <host> <port> fetch    … 出力を gen/out/ に持ち帰る
set -u
HOST=$1; PORT=$2; CMD=$3
cd "$(dirname "$0")"
KEY=${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}   # RunPod に登録した SSH 鍵
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p "$PORT" "root@$HOST")
case $CMD in
  send)
    STAGE=$(mktemp -d); mkdir -p "$STAGE/inputs" "$STAGE/wf"
    cp in/* "$STAGE/inputs/"; cp wf/*.api.json "$STAGE/wf/"; cp remote_job.sh hfdl.py direct.json "$STAGE/"; [ -f freeze.txt ] && cp freeze.txt "$STAGE/"
    COPYFILE_DISABLE=1 tar -C "$STAGE" --no-xattrs -cf - . | "${SSH[@]}" "rm -rf /tmp/okujou && mkdir -p /tmp/okujou && tar -C /tmp/okujou --no-same-owner -xf -"
    rm -rf "$STAGE"
    "${SSH[@]}" "setsid bash /tmp/okujou/remote_job.sh < /dev/null > /tmp/okujou/job.log 2>&1 & sleep 2; cat /tmp/okujou/status 2>/dev/null"
    ;;
  check)
    "${SSH[@]}" 'echo "== 進み具合"; cat /tmp/okujou/status 2>/dev/null; echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader;
                 echo "== ComfyUI のログ（最後）"; tail -c 600 /tmp/okujou/comfy.log 2>/dev/null | tr "\r" "\n" | tail -4; echo "== 出力"; ls -la /tmp/okujou/out/okujou 2>/dev/null'
    ;;
  fetch)
    mkdir -p out
    "${SSH[@]}" 'cd /tmp/okujou/out/okujou && tar -cf - .' | tar -xf - -C out && ls -la out
    ;;
esac
