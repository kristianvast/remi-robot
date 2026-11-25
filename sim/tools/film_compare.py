"""Pick car-camera poses inside the film's coverage, write render specs, and build comparison sheets.

Usage (sim venv, project root), after film_align.py align + coverage:
  sim/.venv/bin/python -B sim/tools/film_compare.py specs
      -> scans/film/work/render_car.json, render_film.json, render_drive.json
  (render them on the VM with film_render.py into scans/film/compare/ and scans/film/drive/)
  sim/.venv/bin/python -B sim/tools/film_compare.py sheets
      -> compare/compare_sheet.jpg, compare/film_sheet.jpg, drive_splat.mp4

A mesh picture counts as covered when its camera cell and the floor point 1.5 m straight ahead are
both in the covered area (scans/film/work/covered.npy). Car poses reuse poses_rendered.json exactly
(x, y, lens z, yaw, pitch, hfov, 640x480); the camera model is a pinhole with square pixels.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILM = os.path.join(ROOT, "scans", "film")
WORK = os.path.join(FILM, "work")
VIEWS = os.path.join(ROOT, "scans", "merged", "views")
MAGENTA = [0.55, 0.0, 0.55]
sys.path.insert(0, os.path.dirname(__file__))
from film_align import FloorMap  # noqa: E402


def car_cam(p):
    y, pt = math.radians(p["yaw"]), math.radians(p["pitch"])
    fwd = np.array([math.cos(y) * math.cos(pt), math.sin(y) * math.cos(pt), math.sin(pt)])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    down = np.cross(fwd, right)
    f = 320.0 / math.tan(math.radians(p["hfov"]) / 2)
    return dict(C=[p["x"], p["y"], p["z"]], R_c2w=np.stack([right, down, fwd], 1).tolist(),
                fx=f, fy=f, cx=320.0, cy=240.0, w=640, h=480)


def covered_fn():
    fm = FloorMap()
    cov = np.load(os.path.join(WORK, "covered.npy"))

    def at(x, y):
        c, r = fm.to_px(x, y)
        c, r = int(c), int(r)
        return bool(0 <= r < fm.h and 0 <= c < fm.w and cov[r, c])
    return at


def cmd_specs():
    at = covered_fn()
    poses = json.load(open(os.path.join(VIEWS, "poses_rendered.json")))["poses"]

    def covered(p):
        y = math.radians(p["yaw"])
        return at(p["x"], p["y"]) and at(p["x"] + 1.5 * math.cos(y), p["y"] + 1.5 * math.sin(y))
    room = [p for p in poses if p["kind"] == "room" and covered(p)]
    car = [dict(car_cam(p), file=p["file"][:-4] + "_splat.jpg", room=p["room"], src=p["file"]) for p in room]
    json.dump(dict(background=MAGENTA, poses=car), open(os.path.join(WORK, "render_car.json"), "w"), indent=1)
    drive = [p for p in poses if p["kind"] == "drive"]
    flags = [at(p["x"], p["y"]) for p in drive]
    runs, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            runs.append((start, i))
            start = None
    keep = [i for a, b in runs if b - a >= 10 for i in range(a, b)]
    dspec = [dict(car_cam(drive[i]), file="frames/frame_%04d.jpg" % (k + 1), src=drive[i]["file"])
             for k, i in enumerate(keep)]
    json.dump(dict(background=MAGENTA, poses=dspec, runs=runs), open(os.path.join(WORK, "render_drive.json"), "w"),
              indent=1)
    cams = json.load(open(os.path.join(WORK, "film_cams_metric.json")))
    cam = cams["camera"]
    test = [c for i, c in enumerate(cams["cams"]) if i % 8 == 0]
    pick = [test[int(round(k))] for k in np.linspace(0, len(test) - 1, 6)]
    fx, fy, cx, cy = cam["params"][:4]
    film = [dict(C=c["C"], R_c2w=c["R_c2w"], fx=fx / 2, fy=fy / 2, cx=cx / 2, cy=cy / 2, w=cam["w"] // 2,
                 h=cam["h"] // 2, file="film_view_%d.jpg" % (k + 1), src=c["name"]) for k, c in enumerate(pick)]
    json.dump(dict(background=MAGENTA, poses=film), open(os.path.join(WORK, "render_film.json"), "w"), indent=1)
    print("SPECS room views covered %d of %d: %s" % (len(car), sum(p["kind"] == "room" for p in poses),
                                                     [c["src"] for c in car]))
    print("SPECS drive covered runs %s -> %d frames; film views %s" % (runs, len(dspec), [f["src"] for f in film]))


def cmd_sheets():
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    car = json.load(open(os.path.join(WORK, "render_car.json")))["poses"]
    comp = os.path.join(FILM, "compare")
    tw, th, lab = 560, 420, 34
    W = 2 * tw + 10
    cols = 2 if len(car) > 6 else 1
    rows = math.ceil(len(car) / cols)
    sheet = Image.new("RGB", (cols * W + (cols - 1) * 20, 50 + rows * (th + lab)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 12), "left: existing scan-mesh render   right: film Gaussian splat, same car-camera pose "
           "(640x480, 12.6 cm, pitch +20 deg, HFOV 65 deg). Magenta = nothing there.", fill="black", font=font)
    for k, p in enumerate(car):
        c, r = k % cols, k // cols
        x0, y0 = c * (W + 20), 50 + r * (th + lab)
        a = Image.open(os.path.join(VIEWS, p["src"])).resize((tw, th), Image.LANCZOS)
        b = Image.open(os.path.join(comp, p["file"])).resize((tw, th), Image.LANCZOS)
        sheet.paste(a, (x0, y0 + lab))
        sheet.paste(b, (x0 + tw + 10, y0 + lab))
        d.text((x0 + 4, y0 + 6), "%s  (%s)" % (p["src"][:-4], p["room"]), fill="black", font=font)
    sheet.save(os.path.join(comp, "compare_sheet.jpg"), quality=88)
    film = json.load(open(os.path.join(WORK, "render_film.json")))["poses"]
    fw, fh = 1100, 619
    fs = Image.new("RGB", (2 * fw + 10, 50 + len(film) * (fh + lab)), "white")
    d = ImageDraw.Draw(fs)
    d.text((10, 12), "left: real film frame (held-out, not trained on)   right: splat render from the same pose",
           fill="black", font=font)
    for k, p in enumerate(film):
        y0 = 50 + k * (fh + lab)
        real = Image.open(os.path.join(FILM, "frames_tm", p["src"])).resize((fw, fh), Image.LANCZOS)
        sp = Image.open(os.path.join(comp, p["file"])).resize((fw, fh), Image.LANCZOS)
        real.save(os.path.join(comp, p["file"].replace(".jpg", "_real.jpg")), quality=90)
        fs.paste(real, (0, y0 + lab))
        fs.paste(sp, (fw + 10, y0 + lab))
        d.text((4, y0 + 6), "%s  %s" % (p["file"][:-4], p["src"]), fill="black", font=font)
    fs.thumbnail((2400, 100000))
    fs.save(os.path.join(comp, "film_sheet.jpg"), quality=88)
    drive_dir = os.path.join(FILM, "drive", "frames")
    if os.path.isdir(drive_dir) and os.listdir(drive_dir):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i",
                        os.path.join(drive_dir, "frame_%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", os.path.join(FILM, "drive", "drive_splat.mp4")], check=True)
    print("SHEETS compare_sheet %s, film_sheet %s" % (sheet.size, fs.size))


if __name__ == "__main__":
    {"specs": cmd_specs, "sheets": cmd_sheets}[sys.argv[1]]()
