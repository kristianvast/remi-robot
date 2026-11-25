"""PNG top-down viewer. Run: python -m sim.macsim.viewer"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .simulator import Simulator
from .world import floor


def save_path(output: Path, truth, estimate=None, *, title: str = "Simulator run") -> None:
    f = floor()
    truth = np.asarray(truth)
    view = Image.fromarray(np.where(f.image >= 245, 247, np.where(f.image == 0, 98, 205))
                           .astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(view)

    def pixels(path):
        r, c = f.cells(*np.asarray(path)[:, :2].T)
        return [(int(x), int(y)) for x, y in zip(c, r, strict=True)]

    for a, b, c, d in f.closures:
        draw.line(pixels([[a, b], [c, d]]), fill="#8a4597", width=2)
    for room in f.rooms:
        p, = pixels([[room["center"]["x"], room["center"]["y"]]])
        draw.text(p, room["id"], fill="#42484f", anchor="mm", stroke_width=1, stroke_fill="#eeeeee")
    if estimate is not None:
        pred = pixels(estimate)
        draw.line(pred, fill="#d2691e", width=4)
    true = pixels(truth)
    draw.line(true, fill="#056891", width=3)
    for points, color in [(true, "#056891")] + ([(pred, "#d2691e")] if estimate is not None else []):
        x, y = points[0]
        draw.ellipse((x-5, y-5, x+5, y+5), outline=color, width=2)
        x, y = points[-1]
        draw.rectangle((x-4, y-4, x+4, y+4), fill=color)
    draw.rounded_rectangle((16, 16, 680, 81), 9, fill="#ffffff", outline="#687581")
    draw.text((27, 24), title, fill="#182533", font_size=19)
    draw.text((27, 53), "Truth", fill="#056891", font_size=16)
    if estimate is not None:
        draw.text((89, 53), "Estimate", fill="#b25512", font_size=16)
    draw.text((180, 53), "Circle: start   Square: finish   Purple: scan closure", fill="#42484f", font_size=16)
    draw.line((32, view.height-26, 82, view.height-26), fill="#202020", width=3)
    draw.text((33, view.height-48), "1 m", fill="#202020", font_size=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    view.save(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("sim/macsim/run.png"))
    args = ap.parse_args()
    car = Simulator()
    path = [car.pose]
    for i in range(95):
        car.step(*( (80, 80) if i < 40 else (-65, 65) if i < 57 else (80, 80)))
        path.append(car.pose)
    save_path(args.out, path)


if __name__ == "__main__":
    main()
