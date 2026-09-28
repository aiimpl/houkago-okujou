# コンテ（../conte/CONTE.md）のダンスを、カットごとの Wan2.2 I2V ワークフローにする → wf/*.api.json
#   python build.py
#   時刻の指定（At 1 second …）には従わないので、動きは「まず… 次に…」の順番で書く。回転は入れない
import json

LOOK = ("An anime girl with a short black bob, a pink X-shaped hairpin, a white cropped hoodie with pink and cyan patches, a black crop top, "
        "navy shorts with white straps and pink sneakers, dancing on a flat, evenly lit solid green chroma key background. ")
GROOVE = "She dances with a wide stance, bending her knees on every beat and shifting her weight from side to side so her hips sway. "
TAIL = (" She always faces the camera. Her face stays clear and her hands keep five fingers. The camera stays still in a full-body shot "
        "with her whole body inside the frame. Clean anime line art, vibrant colors, the background stays plain green with no shadows, props or text.")
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
       "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，转身，背影")

W, H = 832, 1216
SEEDS = (128, 7)

# カット: (最初のコマ, コマ数 4k+1（16fps）, 動き)
CUTS = {
 "c1_open": ("cross.png", 49,
   "She starts with both forearms crossed in an X in front of her chest and her head lowered. Then she lifts her head with a bright smile "
   "and swings both arms open to the sides with open palms, and keeps bouncing her knees."),
 "c2_chime": ("front.png", 81,
   GROOVE + "First she does side steps: she steps to her right and taps her left foot next to it, then steps to her left and taps her right foot, "
   "her shoulders bouncing. Then she raises her right index finger beside her face and taps the air twice as if ringing a bell. "
   "Then she jogs in place, lifting her knees high and swinging her arms."),
 "c2b_chime": ("front.png", 97,   # 歌に合わせた撮り直し：チャイム → 横ステップ → 駆け足（採用は s7）
   "She raises her right index finger beside her face and taps the air twice as if ringing a bell, with a bright smile. "
   "Then she does wide side steps, stepping to her right and to her left, bending her knees on each step and looking left and right playfully. "
   "Then she jogs in place, lifting her knees high and swinging her arms, as if running up the stairs."),
 "c3_sky": ("front.png", 81,
   "She walks two steps toward the camera with a confident groove, her shoulders swaying. Then she reaches her right arm straight up and points at the sky. "
   "Then she swings the arm down and shows the floor in front of her with an open palm. Then she puts one hand on her hip and tilts her head with a smile."),
 "c4_wave": ("front.png", 65,
   GROOVE + "She does a smooth body wave from her chest down to her hips, twice. Then she shimmies her shoulders, alternating forward and back. "
   "Then she does a small hop and lands with her feet apart."),
 "c4b_swing": ("front.png", 65,
   GROOVE + "She swings both arms across in front of her body and out to the sides in big arcs while stepping side to side, "
   "then rolls her shoulders back twice with attitude, then hops up and lands in a wide stance with both arms stretched out to the sides and a big smile."),
 "c5_dance": ("front.png", 65,
   "She starts with both hands in front of her chest. Then she thrusts both arms straight up above her head and crosses her wrists in an X. "
   "Then she brings her arms down and claps once in front of her chest. Then she throws both arms up and out in a wide V and does a small jump."),
 "c6_punch": ("front.png", 65,
   GROOVE + "She punches her right fist up diagonally to the upper right with a small forward kick of her right foot, then her left fist to the upper left "
   "with a kick of her left foot, and keeps alternating in rhythm with a big smile."),
 "c7_x": ("front.png", 49,
   "She points both hands at the camera like finger guns with a grin. Then she pulls her arms in and crosses her forearms in an X in front of her chest, "
   "tilts her head a little, winks and holds the pose."),
}

def wf(name, first, length, motion, seed):
    w = json.load(open("wan_template.api.json"))
    w["8"]["inputs"]["text"] = LOOK + motion + TAIL; w["9"]["inputs"]["text"] = NEG
    w["11"]["inputs"]["image"] = first
    w["12"]["inputs"].update({"width": W, "height": H, "length": length})
    w["13"]["inputs"]["noise_seed"] = seed
    w["17"]["inputs"]["filename_prefix"] = f"okujou/{name}_s{seed}"
    return w

if __name__ == "__main__":
    import sys
    only = sys.argv[1:]   # 名前を渡すとそのカットだけ作る
    for name, (first, length, motion) in CUTS.items():
        if only and name not in only: continue
        for s in SEEDS:
            json.dump(wf(name, first, length, motion, s), open(f"wf/{name}_s{s}.api.json", "w"), ensure_ascii=False, indent=1)
    print("wf/ にワークフローを書き出した")
