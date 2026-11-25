"""Texture bank index: the clips' rectified crops plus generic procedural indoor textures.

  python sim/rooms/texture_bank.py

Reads sim/rooms/bank/<clip>_bank.json (written by vm/bank_crops.py), writes the generic textures once
(sim/rooms/bank/generic_*.jpg: plaster, striped and dotted wallpaper, wood planks, carpet, tiles, ceiling
board, panel door) and sim/rooms/bank/index.json, which the visual scene builder and the Isaac randomiser read.
Kinds: floor, wood, ceiling, wall, wall_upper, tall. size_m is the physical size one texture repeat covers.
"""

import glob
import json
import os

import cv2
import numpy as np

BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank")
PX = 512


def noise(rng, scale_px, px=PX):
    small = rng.standard_normal((max(2, px // scale_px),) * 2).astype(np.float32)
    big = cv2.resize(np.tile(small, (3, 3)), (3 * px, 3 * px), interpolation=cv2.INTER_CUBIC)
    return big[px:2 * px, px:2 * px]


def colourise(field, base, amount):
    img = np.asarray(base, np.float32)[None, None] * (1 + amount * field[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


def plaster(rng, base):
    return colourise(0.6 * noise(rng, 64) + 0.25 * noise(rng, 8) + 0.1 * noise(rng, 2), base, 0.05)


def stripes(rng, a, b, period_px=64):
    x = np.arange(PX)
    band = ((x // (period_px // 2)) % 2).astype(np.float32)
    img = np.where(band[None, :, None] > 0, np.asarray(a, np.float32), np.asarray(b, np.float32))
    img = np.broadcast_to(img, (PX, PX, 3)).astype(np.float32)
    return np.clip(img * (1 + 0.03 * noise(rng, 16)[..., None]), 0, 255).astype(np.uint8)


def dots(rng, ground, dot, period_px=64):
    img = plaster(rng, ground).astype(np.float32)
    yy, xx = np.mgrid[:PX, :PX]
    for oy in (0, period_px // 2):
        d = np.hypot((xx + oy) % period_px - period_px / 2, (yy + oy) % period_px - period_px / 2)
        img[d < period_px / 8] = dot
    return cv2.GaussianBlur(img, (3, 3), 0).astype(np.uint8)


def planks(rng, base, rows=6):
    img = np.zeros((PX, PX, 3), np.float32)
    h = PX // rows
    for r in range(rows):
        shade = 1 + 0.12 * rng.standard_normal()
        grain = noise(rng, 4)[:h] * 0.08 + np.sin(np.linspace(0, rng.uniform(10, 30), PX))[None] * 0.04
        img[r * h:(r + 1) * h] = np.asarray(base, np.float32) * shade * (1 + grain[..., None])
        img[r * h:r * h + 2] *= 0.55
        cut = int(rng.integers(PX))
        img[r * h:(r + 1) * h, cut:cut + 2] *= 0.55
    return np.clip(img, 0, 255).astype(np.uint8)


def carpet(rng, base):
    return colourise(0.5 * noise(rng, 1) + 0.3 * noise(rng, 3) + 0.2 * noise(rng, 48), base, 0.12)


def tiles(rng, base, n=4):
    img = plaster(rng, base).astype(np.float32)
    step = PX // n
    img[::step] *= 0.6
    img[:, ::step] *= 0.6
    img[1::step] *= 0.75
    img[:, 1::step] *= 0.75
    return np.clip(img, 0, 255).astype(np.uint8)


def panel_door(rng, base, w=256):
    h = int(w * 2.0 / 0.9)
    img = np.broadcast_to(np.asarray(base, np.float32), (h, w, 3)).copy()
    img *= 1 + 0.02 * cv2.resize(noise(rng, 16), (w, h))[..., None]
    for y0, y1 in ((0.06, 0.45), (0.52, 0.94)):
        for x0, x1 in ((0.12, 0.45), (0.55, 0.88)):
            p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
            cv2.rectangle(img, p0, p1, tuple(float(c) * 0.82 for c in base), 3)
    cv2.circle(img, (int(0.9 * w), int(0.5 * h)), 5, (60, 60, 60), -1)
    img[:, :4] *= 0.7
    img[:, -4:] *= 0.7
    return np.clip(img, 0, 255).astype(np.uint8)


def generic(rng):
    out = []
    for name, rgb in (("white", (238, 236, 230)), ("cream", (232, 220, 190)), ("sage", (178, 190, 160)),
                      ("blue", (180, 196, 214)), ("grey", (170, 170, 168)), ("terracotta", (196, 120, 90))):
        out.append((f"plaster_{name}", "wall", (1.0, 1.0), plaster(rng, rgb)))
    out.append(("stripes_green", "wall", (0.5, 0.5), stripes(rng, (150, 170, 130), (175, 190, 155))))
    out.append(("stripes_red", "wall", (0.5, 0.5), stripes(rng, (150, 60, 60), (185, 95, 85))))
    out.append(("dots_beige", "wall", (0.4, 0.4), dots(rng, (215, 200, 170), (160, 130, 90))))
    out.append(("dots_blue", "wall", (0.4, 0.4), dots(rng, (190, 205, 225), (90, 110, 150))))
    for name, rgb in (("oak", (170, 120, 70)), ("walnut", (110, 70, 40)), ("ash", (205, 175, 130))):
        out.append((f"planks_{name}", "floor", (1.2, 1.2), planks(rng, rgb)))
        out.append((f"planks_{name}_top", "wood", (0.6, 0.6), planks(rng, rgb, rows=3)))
    for name, rgb in (("red", (150, 30, 45)), ("grey", (120, 118, 115)), ("blue", (60, 70, 120)),
                      ("beige", (190, 170, 140)), ("green", (50, 100, 70))):
        out.append((f"carpet_{name}", "floor", (0.5, 0.5), carpet(rng, rgb)))
    out.append(("tiles_white", "floor", (0.6, 0.6), tiles(rng, (225, 225, 220))))
    out.append(("tiles_terracotta", "floor", (0.6, 0.6), tiles(rng, (180, 100, 70))))
    out.append(("ceiling_white", "ceiling", (1.0, 1.0), plaster(rng, (242, 241, 236))))
    out.append(("ceiling_tiles", "ceiling", (1.2, 1.2), tiles(rng, (235, 234, 228), n=2)))
    out.append(("ceiling_cream", "ceiling", (1.0, 1.0), plaster(rng, (236, 228, 205))))
    out.append(("door_white", "tall", (0.9, 2.0), panel_door(rng, (232, 232, 228))))
    out.append(("door_wood", "tall", (0.9, 2.0), panel_door(rng, (120, 75, 45))))
    return out


def main():
    os.makedirs(BANK, exist_ok=True)
    entries = []
    for name, kind, size, img in generic(np.random.default_rng(20261124)):
        fname = f"generic_{name}.jpg"
        path = os.path.join(BANK, fname)
        if not os.path.exists(path):
            cv2.imwrite(path, img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        entries.append(dict(file=fname, kind=kind, clip="generic", size_m=list(size),
                            mean_srgb=[int(x) for x in img.reshape(-1, 3).mean(0)]))
    clips = []
    for path in sorted(glob.glob(os.path.join(BANK, "*_bank.json"))):
        with open(path) as fh:
            data = json.load(fh)
        clips.append(data["clip"])
        entries += [e for e in data["entries"] if os.path.exists(os.path.join(BANK, e["file"]))]
    with open(os.path.join(BANK, "index.json"), "w") as fh:
        json.dump(dict(clips=clips, entries=entries), fh, indent=1)
    counts = {}
    for e in entries:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    print("BANK_INDEX", len(entries), "textures from", clips, "+ generic;", counts)


if __name__ == "__main__":
    main()
