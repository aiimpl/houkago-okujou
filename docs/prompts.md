# 背景のプロンプト（GPT Image 2.5・16:9・2K・quality high）

## 正面（bg_front.png / bg_front2.png）
3枚作って2枚を使った。

```
Japanese anime background art, hand-painted feature-film quality, empty background plate with NO people and NO characters. Rooftop of a Tokyo city building on a bright summer afternoon after school. Eye-level camera at about 1.3 m height, horizon line slightly below the vertical center of the frame, straight-on view. Flat concrete rooftop floor with faded painted white court lines and a small basketball hoop at the far left edge, low concrete parapet and a tall green-grey chain-link safety fence running across the middle ground, a few air-conditioner units and pipes. Behind the fence: dense Tokyo skyline and distant sea, huge towering white cumulus clouds, deep saturated blue sky, crisp sunlight from upper right, soft lens glare. Very detailed, vivid, clean line art, cel-painted shading, cinematic wide shot. Center of the frame kept relatively open for compositing a dancer. No text, no logos, no watermark.
```

## 別アングル（bg_front.png を参考画像にして）
どれも頭に `The same rooftop (basketball court) as the reference image, same art style (Japanese anime background art, hand-painted, cel shading), same summer afternoon light.`、最後に `Empty plate, NO people, NO characters, no text, no logos.` を付けた。

| ファイル | カメラの指定 |
|---|---|
| bg_low.png | low angle, camera 40 cm above the floor looking slightly upward. The concrete floor with white court lines fills the lower third, the fence and parapet sit higher in frame, towering cumulus clouds dominate, the sun is visible in the upper right corner with a strong lens flare and light streaks. |
| bg_diag.png | eye level about 1.3 m, turned 40 degrees to the left so the chain-link fence runs diagonally from the left foreground into the distance, city skyline and sea behind it, big cumulus clouds, a concrete stairwell structure at the far right edge. Horizon slightly below center. |
| bg_build.png | eye level about 1.3 m, turned around 180 degrees to face the building side of the rooftop: a weathered concrete stairwell hut with a metal door, a cylindrical water tank on a steel frame on top of it, pipes, air conditioner units, a railing, blue sky with cumulus behind. Horizon slightly below center. |
| bg_top.png | high angle, camera about 3 m high looking down at 40 degrees onto the concrete floor. The sunlit concrete floor with thick painted white court lines and the curve of the center circle fills most of the frame, soft shadow of the fence falling across the floor, a strip of the parapet and fence visible at the very top edge. |
