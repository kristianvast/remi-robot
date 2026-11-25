"""Contact sheets of car-eye views: plain scan | visual layer | randomised variants, one row per pose.

  python sim/isaac/scene/visual_sheet.py <plain_views_dir> <visual_views_dir> <out_prefix> [--variants 3] [--rows 10]

Reads <stem>_rgb.jpg from the plain run and <stem>_after_rgb.jpg / <stem>_varN_rgb.jpg from the visual run
(nav/run.py views), stems from scene/proof/visual_poses.json; writes <out_prefix>_1.jpg, _2.jpg, ...
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

POSES = Path(__file__).resolve().parent / "proof" / "visual_poses.json"
TILE = (320, 240)


def tile(path):
    if path.exists():
        return Image.open(path).convert("RGB").resize(TILE)
    img = Image.new("RGB", TILE, (60, 0, 0))
    ImageDraw.Draw(img).text((10, 110), "missing", fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plain", type=Path)
    ap.add_argument("visual", type=Path)
    ap.add_argument("out_prefix")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--rows", type=int, default=10)
    args = ap.parse_args()
    poses = json.loads(POSES.read_text())["poses"]
    columns = [("plain scan", args.plain, "")] + [("visual layer", args.visual, "_after")] + [
        (f"random {k}", args.visual, f"_var{k}") for k in range(1, args.variants + 1)]
    header = 22
    for sheet, start in enumerate(range(0, len(poses), args.rows), 1):
        chunk = poses[start:start + args.rows]
        out = Image.new("RGB", (TILE[0] * len(columns), header + TILE[1] * len(chunk)), (20, 20, 20))
        draw = ImageDraw.Draw(out)
        for c, (label, _d, _s) in enumerate(columns):
            draw.text((c * TILE[0] + 6, 5), label, fill=(255, 255, 0))
        for r, pose in enumerate(chunk):
            stem = Path(pose["file"]).stem
            for c, (_label, folder, suffix) in enumerate(columns):
                out.paste(tile(folder / f"{stem}{suffix}_rgb.jpg"), (c * TILE[0], header + r * TILE[1]))
            draw.text((6, header + r * TILE[1] + 4), f"{pose['area']} #{stem[-1]} yaw {pose['yaw']:.0f}",
                      fill=(0, 255, 255))
        path = f"{args.out_prefix}_{sheet}.jpg"
        out.save(path, quality=88)
        print("SHEET", path, out.size)


if __name__ == "__main__":
    main()
