"""Metric scale check from objects of known height, with no external pose.

For each film640 frame (fx 707.8 px, cx 320, cy 240) a floor plane is fitted to
the metric model's back-projected depth in hand-picked floor patches; then the
height above that plane of hand-picked points on an object of known height is
compared with the nominal value. The camera's own height above the plane is
reported too (phone held at roughly chest height, ~1.0-1.5 m).
usage: scale_check.py  -> results/scale_check.json
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FX, CX, CY = 707.8, 320.0, 240.0
MODELS = ["da2met_518_coreml_all", "da2met_336_coreml_all", "dp2s_640_mps_bf16", "dp2s_320_mps_bf16"]
CASES = [
    {"frame": "frame_0300", "object": "office desk top (left desk)", "nominal_m": 0.73,
     "floor": [(0, 400, 120, 470), (360, 410, 520, 470)], "points": [(60, 97), (150, 100), (230, 103)]},
    {"frame": "frame_0378", "object": "wooden table top", "nominal_m": 0.75,
     "floor": [(0, 370, 390, 470)], "points": [(150, 178), (300, 182), (430, 188)]},
    {"frame": "frame_0042", "object": "dining chair seat front", "nominal_m": 0.45,
     "floor": [(0, 400, 360, 470)], "points": [(420, 322), (460, 320), (520, 322)]},
]


def backproject(depth, us, vs):
    z = depth[vs, us]
    return np.stack([(us - CX) * z / FX, (vs - CY) * z / FX, z], 1)


def fit_plane(p):
    for _ in range(3):
        c = p.mean(0)
        n = np.linalg.svd(p - c)[2][2]
        r = np.abs((p - c) @ n)
        p = p[r < max(0.02, 2.5 * np.median(r))]
    return n, -n @ c


def main():
    out = []
    for case in CASES:
        for m in MODELS:
            d = np.load(os.path.join(HERE, "results", "depth", m, case["frame"] + ".npy")).astype(np.float32)
            d = cv2.resize(d, (640, 480), interpolation=cv2.INTER_LINEAR)
            d = cv2.medianBlur(d, 5)
            fl = np.concatenate([np.stack(np.meshgrid(np.arange(x0, x1, 4), np.arange(y0, y1, 4)), -1).reshape(-1, 2)
                                 for x0, y0, x1, y1 in case["floor"]])
            n, off = fit_plane(backproject(d, fl[:, 0], fl[:, 1]))
            pts = np.array(case["points"])
            h = float(np.median(np.abs(backproject(d, pts[:, 0], pts[:, 1]) @ n + off)))
            out.append({"frame": case["frame"], "object": case["object"], "model": m, "nominal_m": case["nominal_m"],
                        "measured_m": round(h, 3), "error_pct": round(100 * (h / case["nominal_m"] - 1), 1),
                        "camera_height_m": round(float(abs(off)), 2)})
    json.dump(out, open(os.path.join(HERE, "results", "scale_check.json"), "w"), indent=1)
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
