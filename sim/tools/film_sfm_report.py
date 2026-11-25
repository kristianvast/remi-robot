"""Summarise every COLMAP model of every SfM attempt into one small JSON (runs on the VM).

Usage: python film_sfm_report.py out.json name=model_root [name=model_root ...]
model_root holds numbered COLMAP binary models (0, 1, ...). Per model: registered frame numbers,
camera centres, path planarity (smallest / largest PCA sigma of camera centres) and the largest
step between consecutive registered frames relative to the median step.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

COLMAP = os.path.expanduser("~/colmap-env/bin/colmap")


def qvec2rot(w, x, y, z):
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def model_summary(path):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([COLMAP, "model_converter", "--input_path", path, "--output_path", tmp,
                        "--output_type", "TXT"], check=True, capture_output=True)
        lines = [l for l in open(os.path.join(tmp, "images.txt")) if not l.startswith("#")]
        n_points = sum(1 for l in open(os.path.join(tmp, "points3D.txt")) if not l.startswith("#"))
    rows = []
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        R = qvec2rot(*[float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        rows.append((int(p[9][6:10]), (-R.T @ t).tolist(), (-R[1]).tolist()))
    rows.sort()
    frames = [r[0] for r in rows]
    C = np.array([r[1] for r in rows])
    sig = np.linalg.svd(C - C.mean(0), compute_uv=False) if len(C) >= 3 else np.array([1.0, 1.0, 1.0])
    steps = [float(np.linalg.norm(C[k + 1] - C[k])) for k in range(len(C) - 1) if frames[k + 1] - frames[k] <= 2]
    med = float(np.median(steps)) if steps else 0.0
    return dict(n=len(frames), points=n_points, frames=frames, centres=C.round(4).tolist(),
                ups=[r[2] for r in rows], planarity=round(float(sig[2] / sig[0]), 4),
                max_step_over_median=round(max(steps) / med, 1) if med > 0 else None)


def main(out, specs):
    res = {}
    for spec in specs:
        name, root = spec.split("=", 1)
        for d in sorted(os.listdir(root), key=lambda s: (len(s), s)):
            if d.isdigit():
                res["%s/%s" % (name, d)] = model_summary(os.path.join(root, d))
                print(name, d, res["%s/%s" % (name, d)]["n"])
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
