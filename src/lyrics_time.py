# 曲の歌詞が歌われる秒を測る（文字起こし・単語ごとの時刻）→ scenes.py の L_* と process.py の WORDS に使う
#   pip install faster-whisper（初回はモデルを Hugging Face から取ってくる）
#   python src/lyrics_time.py [曲.flac]（省略時は gen/CHOICE.json の _song）
import json, sys
from faster_whisper import WhisperModel

def main(path=None):
    song = json.load(open("gen/CHOICE.json"))["_song"]
    path = path or song["file"]; start = song["start"] if path == song["file"] else 0.0
    segs, _ = WhisperModel("medium", device="cpu", compute_type="int8").transcribe(path, language="ja", word_timestamps=True)
    for s in segs:
        print(f"{s.start - start:6.2f}–{s.end - start:6.2f}  {s.text.strip()}")
        print("        " + "  ".join(f"{w.word.strip()}@{w.start - start:.2f}" for w in s.words))

if __name__ == "__main__":
    main(*sys.argv[1:2])
