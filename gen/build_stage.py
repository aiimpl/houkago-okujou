# 曲（ACE-Step 1.5）と最後の寄り（Wan2.2・腰から上）のワークフロー
#   python build_stage.py                         … 曲 → wf/b_song_*.api.json
import json, os

# ---------- 曲 ----------
# 映像は 120 BPM・16秒（32拍）。歌入りは画面の歌詞をそのまま歌わせる。長めに作って、良い16秒を切り出す
LYRICS = """[Intro]

[Verse]
六時間目の チャイムで
階段を かけ上がる
空にいちばん 近いステージ

[Chorus]
ダンス ダンス 屋上で
Rooftop dance crew
"""
SONGS = {
 "song_vocal":  ("J-pop anime opening, bright summer, female vocal, energetic, electric guitar, slap bass, punchy drums, "
                 "brass stabs, sparkling synth, youthful, after school, catchy chorus", LYRICS, "ja", "A major"),
 "song_inst":   ("instrumental, upbeat anime J-pop, summer rooftop, funky electric guitar cutting, slap bass, four on the floor kick, "
                 "handclaps, brass hits, bright synth lead, festival energy, no vocals", "[Instrumental]", "en", "D major"),
}

def song(name, tags, lyrics, lang, key, seed, seconds=32.0):
    return {
     "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "acestep_v1.5_turbo.safetensors", "weight_dtype": "default"}},
     "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "qwen_0.6b_ace15.safetensors", "clip_name2": "qwen_1.7b_ace15.safetensors", "type": "ace", "device": "default"}},
     "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ace_1.5_vae.safetensors"}},
     "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
     "5": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"clip": ["2", 0], "tags": tags, "lyrics": lyrics, "seed": seed, "bpm": 120,
            "duration": seconds, "timesignature": "4", "language": lang, "keyscale": key, "generate_audio_codes": True,
            "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
     "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
     "7": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": seconds, "batch_size": 1}},
     "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "denoise": 1.0}},
     "9": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
     "10": {"class_type": "SaveAudio", "inputs": {"audio": ["9", 0], "filename_prefix": f"okujou/{name}_s{seed}"}},
    }

if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    for f in os.listdir("wf"): os.remove(f"wf/{f}")
    for name, (tags, lyr, lang, key) in SONGS.items():
        for s in (31, 7): json.dump(song(name, tags, lyr, lang, key, s), open(f"wf/b_{name}_s{s}.api.json", "w"), ensure_ascii=False, indent=1)
    print(sorted(os.listdir("wf")))

# ---------- 2段目：最後の寄りの動画（Wan2.2 I2V・腰から上） ----------
#   python build_stage.py close <最初のコマ.png>
CLOSE_LOOK = ("A waist-up close-up of an anime girl with a short black bob, a pink X-shaped hairpin, a white cropped hoodie with pink and cyan patches "
              "and a black crop top, on a flat, evenly lit solid green chroma key background. ")
CLOSE_TAIL = (" Both of her eyes stay open and clear the whole time, looking at the viewer. Her face stays clear and her hands keep five fingers. "
              "The camera stays still in a waist-up shot. Clean anime line art, vibrant colors, the background stays plain green.")
CLOSE = {   # 最初のコマは gen/in/close_orig.png（src/first_frame.py close …）。採用は c8o_wink の s128
 "c8o_wink": "She keeps her forearms crossed in an X in front of her chest, gives a quick playful wink, then laughs, "
             "and ends looking straight at the viewer with both eyes open and a bright smile.",
}

def close(name, first, motion, seed):
    from build import NEG
    w = json.load(open("wan_template.api.json"))
    w["8"]["inputs"]["text"] = CLOSE_LOOK + motion + CLOSE_TAIL; w["9"]["inputs"]["text"] = NEG.replace("，转身，背影", "，闭眼，眨眼，转身，背影")
    w["11"]["inputs"]["image"] = first
    w["12"]["inputs"].update({"width": 832, "height": 1216, "length": 49})
    w["13"]["inputs"]["noise_seed"] = seed
    w["17"]["inputs"]["filename_prefix"] = f"okujou/{name}_s{seed}"
    return w

if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["close"]:
        os.makedirs("wf3", exist_ok=True)
        for f in os.listdir("wf3"): os.remove(f"wf3/{f}")
        for name, motion in CLOSE.items():
            for s in (128, 7, 42): json.dump(close(name, sys.argv[2], motion, s), open(f"wf3/{name}_s{s}.api.json", "w"), ensure_ascii=False, indent=1)
        print(sorted(os.listdir("wf3")))
