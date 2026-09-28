# make … ダンスの切り抜き → 缶と紋章 → 曲 → 動画（out/okujou.mp4）
#   ダンスは gen/ で RunPod に作らせたクリップ（gen/README.md）。gen/CHOICE.json に選んだものを書く
PY ?= python3

all: out/okujou.mp4 out/process.mp4

cache/cuts/.done: gen/CHOICE.json src/key_cuts.py src/key.py
	$(PY) src/key_cuts.py && touch $@

cache/crest.png: assets/can_green.png src/prep.py src/key.py src/typo.py
	$(PY) src/prep.py

cache/music.wav: gen/CHOICE.json src/pick_song.py
	$(PY) -c "import json;c=json.load(open('gen/CHOICE.json'))['_song'];print(c['file'],c['start'])" | xargs $(PY) src/pick_song.py

out/okujou.mp4: src/scenes.py src/comp.py src/typo.py src/render.py assets/bg_*.png cache/cuts/.done cache/crest.png cache/music.wav
	$(PY) src/render.py

out/process.mp4: src/process.py out/okujou.mp4
	$(PY) src/process.py

sheet: cache/cuts/.done cache/crest.png
	$(PY) src/render.py --sheet

clean:
	rm -rf cache out

.PHONY: all sheet clean
