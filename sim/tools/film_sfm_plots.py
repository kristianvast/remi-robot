"""Write scans/film/sfm_summary.txt and scans/film/sfm_paths.png from the per-model SfM summary.

Usage (sim venv): sim/.venv/bin/python -B sim/tools/film_sfm_plots.py scans/film/work/logs/sfm_models.json
Each model's camera path is drawn top-down in its own arbitrary SfM frame (projected onto the plane
of its two largest PCA axes); a straight segment between consecutive frames longer than 4x the
median step is drawn dashed red as a jump. The timeline row shows which film frames each model holds.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILM = os.path.join(ROOT, "scans", "film")
N_FRAMES = 392
ATTEMPTS = [
    ("a1_global", "Attempt 1 GLOMAP global mapper, original flat frames, SIFT default"),
    ("a1_incremental", "Attempt 1 incremental mapper, original flat frames, SIFT default"),
    ("a2_global", "Attempt 2 GLOMAP global mapper, tone-mapped frames, SIFT peak 0.002"),
    ("a2_incremental", "Attempt 2 incremental mapper, tone-mapped frames, SIFT peak 0.002"),
    ("a3_relaxed", "Attempt 3 incremental mapper, relaxed registration thresholds"),
]
PALETTE = [(220, 40, 40), (30, 110, 220), (40, 160, 60), (230, 140, 0), (150, 60, 190), (0, 170, 170),
           (120, 120, 120), (190, 90, 40), (90, 90, 200)]


def ranges(frames):
    out, s, p = [], None, None
    for f in frames:
        if s is None:
            s = p = f
        elif f == p + 1:
            p = f
        else:
            out.append((s, p))
            s = p = f
    if s is not None:
        out.append((s, p))
    return out


def fmt(rs):
    return ", ".join("%d-%d" % r if r[0] != r[1] else str(r[0]) for r in rs)


def main(path):
    data = json.load(open(path))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except OSError:
        font = small = ImageFont.load_default()
    lines = ["Structure-from-motion on the 392 film frames (scans/film/frames, 1920x1080, ordered).",
             "COLMAP 4.2.1 (CUDA): SIFT, sequential matching overlap 20 + vocab-tree loop detection,",
             "one shared OPENCV camera. A model is usable for a splat only if its camera path is planar",
             "(level walk) and has no jumps; planarity = smallest/largest PCA sigma of camera centres.", ""]
    pw, ph, tl = 470, 330, 26
    sheet = Image.new("RGB", (40 + 4 * pw, len(ATTEMPTS) * (ph + tl + 60) + 20), "white")
    d = ImageDraw.Draw(sheet)
    for ai, (key, title) in enumerate(ATTEMPTS):
        models = sorted([(k, v) for k, v in data.items() if k.startswith(key + "/")], key=lambda kv: -kv[1]["n"])
        reg = sorted(set(f for _, v in models for f in v["frames"]))
        lost = sorted(set(range(N_FRAMES)) - set(reg))
        big = models[0][1]
        lines.append("%s" % title)
        lines.append("  registered in any model: %d / %d in %d models; largest connected model: %d (%.0f%%)"
                     % (len(reg), N_FRAMES, len(models), big["n"], 100.0 * big["n"] / N_FRAMES))
        for k, v in models:
            lines.append("  %-18s %3d frames, %6d points, planarity %.3f, max step/median %s, frames %s"
                         % (k, v["n"], v["points"], v["planarity"], v["max_step_over_median"], fmt(ranges(v["frames"]))))
        lines.append("  never registered: %d frames: %s" % (len(lost), fmt(ranges(lost))))
        lines.append("  not in the largest model: %s" % fmt(ranges(sorted(set(range(N_FRAMES)) - set(big["frames"])))))
        lines.append("")
        y0 = 20 + ai * (ph + tl + 60)
        d.text((20, y0), "%s   (largest %d/392, all models %d/392)" % (title, big["n"], len(reg)), fill="black", font=font)
        ty = y0 + 24
        d.rectangle([20, ty, 20 + 4 * pw, ty + tl - 6], outline=(180, 180, 180))
        for mi, (k, v) in enumerate(models):
            col = PALETTE[mi % len(PALETTE)]
            for f in v["frames"]:
                x = 20 + f * 4 * pw / N_FRAMES
                d.rectangle([x, ty + 1, x + 4 * pw / N_FRAMES, ty + tl - 7], fill=col)
        for f in range(0, N_FRAMES, 50):
            d.text((20 + f * 4 * pw / N_FRAMES, ty + tl - 6), str(f), fill="black", font=small)
        for mi, (k, v) in enumerate(models[:4]):
            col = PALETTE[mi % len(PALETTE)]
            x0, py0 = 20 + mi * pw, ty + tl + 14
            C = np.array(v["centres"])
            fr = v["frames"]
            mu = C.mean(0)
            _, _, vt = np.linalg.svd(C - mu)
            xy = (C - mu) @ vt[:2].T
            span = np.ptp(xy, axis=0).max() or 1.0
            sc = (min(pw, ph) - 40) / span
            pts = [(x0 + pw / 2 + p[0] * sc, py0 + ph / 2 - p[1] * sc) for p in xy]
            steps = [np.linalg.norm(xy[i + 1] - xy[i]) for i in range(len(xy) - 1) if fr[i + 1] - fr[i] <= 2]
            med = np.median(steps) if steps else 1.0
            for i in range(len(pts) - 1):
                if fr[i + 1] - fr[i] > 5:
                    continue
                jump = np.linalg.norm(xy[i + 1] - xy[i]) > 4 * med
                if jump:
                    for t in np.linspace(0, 1, 12)[::2]:
                        a = (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t)
                        b = (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * (t + 0.08),
                             pts[i][1] + (pts[i + 1][1] - pts[i][1]) * (t + 0.08))
                        d.line([a, b], fill=(255, 0, 0), width=2)
                else:
                    d.line([pts[i], pts[i + 1]], fill=col, width=2)
            for i in range(0, len(pts), max(1, len(pts) // 6)):
                d.text((pts[i][0] + 3, pts[i][1] - 12), str(fr[i]), fill="black", font=small)
            d.rectangle([x0 + 4, py0, x0 + pw - 4, py0 + ph], outline=(200, 200, 200))
            d.text((x0 + 10, py0 + 4), "%s: %d frames, planarity %.3f" % (k.split("/")[1], v["n"], v["planarity"]),
                   fill=col, font=small)
    sheet.save(os.path.join(FILM, "sfm_paths.png"))
    with open(os.path.join(FILM, "sfm_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
