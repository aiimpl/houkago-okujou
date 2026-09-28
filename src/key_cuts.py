# 選んだダンスのクリップ（gen/CHOICE.json：カット名 → gen/out の動画）を切り抜いて cache/cuts/<カット名>/ に置く
#   python src/key_cuts.py
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import key

def main(choice="gen/CHOICE.json"):
    for name, v in json.load(open(choice)).items():
        if name.startswith("_"): continue    # _ で始まるキーはメモ
        out = f"cache/cuts/{name}"
        if os.path.exists(f"{out}/rgba.json") and os.path.getmtime(f"{out}/rgba.json") > os.path.getmtime(v["file"]): continue
        print(name, "←", v["file"]); key.main(v["file"], out)

if __name__ == "__main__":
    main()
