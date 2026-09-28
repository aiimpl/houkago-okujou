#!/usr/bin/env python3
"""HuggingFace のモデルを、429 を踏まずに Pod へ落とす。

## なぜ要るか

⚠️ 2026-09-10、EUR-IS-1 の Pod から `huggingface.co` に繋ぐと **HTTP 429**（CloudFront の
レート制限）が返り、6 本すべての DL が失敗した。同じ URL を日本のローカルから叩くと
**200** だったので、アカウントの制限ではなく **その経路（`x-amz-cf-pop: LHR3-P3`）固有**。

429 が出るのは `huggingface.co/.../resolve/main/...` の **リダイレクト取得だけ**で、
実体を配る CDN（`us.aws.cdn.hf.co`）は別ホスト。そこで

  1. **繋がる場所（ローカル）で 302 を解決して署名付き URL を得る**（--resolve）
  2. **その URL を Pod に渡して CDN から直接落とす**（--fetch）

に分ければ `huggingface.co` を一度も踏まずに済む。

## 使い方

    # 手元（HF に繋がる場所）で URL を解決する
    python3 tools/hfdl.py --resolve models.json -o direct.json

    # Pod 側で落とす（direct.json を送ってから）
    python3 tools/hfdl.py --fetch direct.json --root /workspace/ComfyUI/models

`models.json` は [{"dst": "vae/foo.safetensors", "url": "https://huggingface.co/..."}] の配列。
`--resolve` はこれに署名付き URL を入れた同じ形を書き出す。

⚠️ 署名付き URL には有効期限がある（HF は概ね 24 時間）。解決したら早めに使う。
⚠️ 期限切れや途中終了は `.part` が残る形にしてあるので、再実行すれば途中から続く。
"""
import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request

UA = {"User-Agent": "curl/8"}


def _token():
    """gated リポジトリ用のトークンを探す。

    ⚠️ **トークンが要るのは --resolve だけ。**署名付き URL には認可が埋まっているので、
    Pod 側の --fetch には渡さなくてよい。gated なモデルを落とすときも
    **トークンは手元に留まる**（Pod に送る必要がない）。

    ⚠️ `Lightricks/LTX-2.5` は gated=auto。匿名だと resolve が 401 になる。
    """
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t.strip()
    for p in ("~/.cache/huggingface/token", "~/.huggingface/token"):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            t = open(p).read().strip()
            if t:
                return t
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, newurl, headers, fp)


def resolve(items):
    """302 を追わずに Location を取り出す。繋がる場所で実行すること。"""
    op = urllib.request.build_opener(_NoRedirect)
    hdr = dict(UA)
    tok = _token()
    if tok:
        hdr["Authorization"] = f"Bearer {tok}"
        print("  （トークンを使う）")
    out = []
    for it in items:
        url = it["url"]
        try:
            op.open(urllib.request.Request(url, method="HEAD", headers=hdr), timeout=30)
            out.append(dict(it))                      # リダイレクトが無ければそのまま使う
            print(f"  = {it['dst']}（直リンク）")
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                real = e.reason if isinstance(e.reason, str) else e.headers.get("Location")
                out.append({"dst": it["dst"], "url": real, "origin": url})
                print(f"  ✓ {it['dst']} -> {real.split('/')[2]}")
            elif e.code == 401:
                # gated リポジトリ。トークンが無い / 権限が足りない
                print(f"  ! 401 {it['dst']}"
                      + ("（トークンはあるが権限が足りない。model card で同意する）"
                         if tok else "（gated。HF_TOKEN か ~/.cache/huggingface/token が要る）"),
                      file=sys.stderr)
            else:
                print(f"  ! HTTP {e.code} {it['dst']}", file=sys.stderr)
        except Exception as e:
            print(f"  ! {it['dst']}: {e}", file=sys.stderr)
    return out


def fetch_one(item, root, results):
    dst = os.path.join(root, item["dst"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    name = os.path.basename(dst)
    try:
        req = urllib.request.Request(item["url"], headers=dict(UA))
        # ⚠️ 途中まで落ちている .part があれば Range で続きから取る
        have = os.path.getsize(dst + ".part") if os.path.exists(dst + ".part") else 0
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            results[name] = ("skip", os.path.getsize(dst))
            print(f"  skip {name} {os.path.getsize(dst)/2**30:.2f} GB", flush=True)
            return
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=120) as r:
            total = int(r.headers.get("Content-Length") or 0) + have
            mode = "ab" if have and r.status == 206 else "wb"
            if mode == "wb":
                have = 0
            got = have
            with open(dst + ".part", mode) as f:
                while True:
                    chunk = r.read(1 << 22)           # 4MB
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
        # ⚠️ Content-Length と一致しなければ完成にしない。途中で切れた形を残す
        if total and got != total:
            results[name] = ("short", got)
            print(f"  ! {name} 途中で切れた {got}/{total}", flush=True)
            return
        os.replace(dst + ".part", dst)
        results[name] = ("ok", got)
        print(f"  ok {name} {got/2**30:.2f} GB", flush=True)
    except Exception as e:
        results[name] = ("error", str(e))
        print(f"  ! {name}: {e}", flush=True)


def fetch(items, root, par=3):
    """⚠️ **バッチ単位で join してはいけない。**

    2026-09-10、`items[i:i+par]` ごとに全部の完了を待つ実装にしていたため、
    バッチ内に 20GB の遅い 1 本が混ざると **他が終わっても次のバッチに進まず**、
    par=8 を指定しても実際には 2 本しか走らなかった（帯域 11MB/s）。
    セマフォで「常に par 本を走らせる」形にする。
    """
    results = {}
    sem = threading.Semaphore(par)

    def worker(it):
        with sem:
            fetch_one(it, root, results)

    ths = [threading.Thread(target=worker, args=(it,)) for it in items]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    ok = sum(1 for v in results.values() if v[0] in ("ok", "skip"))
    print(f"\n  {ok}/{len(items)} 本 完了")
    for n, (st, v) in sorted(results.items()):
        if st not in ("ok", "skip"):
            print(f"  ! {n}: {st} {v}")
    return 0 if ok == len(items) else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--resolve", metavar="JSON", help="302 を解決する（HF に繋がる場所で）")
    g.add_argument("--fetch", metavar="JSON", help="解決済み URL から落とす（Pod 側で）")
    ap.add_argument("-o", "--out", help="--resolve の出力先")
    ap.add_argument("--root", default=".", help="--fetch の展開先（models ディレクトリ）")
    ap.add_argument("--par", type=int, default=3, help="同時に落とす本数")
    a = ap.parse_args()

    if a.resolve:
        items = json.load(open(a.resolve))
        out = resolve(items)
        if not a.out:
            raise SystemExit("--out が必要")
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"  {len(out)}/{len(items)} 本 -> {a.out}")
        return 0 if len(out) == len(items) else 1
    return fetch(json.load(open(a.fetch)), a.root, a.par)


if __name__ == "__main__":
    sys.exit(main())
