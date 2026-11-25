"""Car camera frame -> depth observation (sim/realdepth/observation_spec.md).

    from realdepth import observe
    obs = observe(rgb)   # rgb: (480, 640, 3) uint8 RGB -> (120, 160) float32 in [0, 1]

    .venv/bin/python realdepth.py --benchmark       # latency over real frames, JPEG decode included
    .venv/bin/python realdepth.py --image a.jpg     # writes a_obs.png (bright = near)

Model: Apple Depth Pro 2-S (apple/DepthPro2-S, Apple ML Research licence:
non-commercial research only) at 320x240 on the Apple GPU (PyTorch MPS, bfloat16).
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "models", "hf"))

import torch  # noqa: E402
from depth_pro_2 import DepthPro2  # noqa: E402
from depth_pro_2.model import to_normalized_tensor  # noqa: E402

CAR_FX_640 = 502.3
IN_W, IN_H = 320, 240
OUT_W, OUT_H = 160, 120
Z_NEAR, Z_FAR = 0.15, 5.0

_device = torch.device("mps")
_model = None


def _load():
    global _model
    if _model is None:
        _model = DepthPro2.from_pretrained("apple/DepthPro2-S", precision=torch.bfloat16, fov=False)
        _model = _model.to(_device).eval()
    return _model


def encode(z):
    """Metric optical-axis depth (m, inf = no hit) -> 1 - log(z/0.15)/log(5/0.15), in [0, 1]."""
    z = np.clip(z, Z_NEAR, Z_FAR)
    return (1.0 - np.log(z / Z_NEAR) / np.log(Z_FAR / Z_NEAR)).astype(np.float32)


@torch.no_grad()
def observe(rgb):
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"expected (480, 640, 3) uint8 RGB, got {rgb.dtype} {rgb.shape}")
    x = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    z = _load().infer(to_normalized_tensor(x, _device), f_px=CAR_FX_640 * IN_W / 640)["depth"]
    z = z.float().cpu().numpy()
    inv = 1.0 / np.clip(z, Z_NEAR, np.inf)
    inv = inv.reshape(OUT_H, IN_H // OUT_H, OUT_W, IN_W // OUT_W).mean(axis=(1, 3))
    return encode(1.0 / np.maximum(inv, 1e-6))


def _real_jpegs():
    car = sorted(glob.glob(os.path.join(HERE, "..", "..", "patrol-photos", "bench", "*.jpg")))
    film = sorted(glob.glob(os.path.join(HERE, "data", "film640", "*.jpg")))[::20]
    return [open(p, "rb").read() for p in car + film]


def benchmark(n=200, warm=20):
    jpegs = _real_jpegs()
    decode = lambda b: cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    _load()
    print(f"model load {time.perf_counter() - t0:.1f} s; {len(jpegs)} real frames (car bench photos + film-derived)")
    for i in range(warm):
        observe(decode(jpegs[i % len(jpegs)]))
    ts = []
    for i in range(n):
        t = time.perf_counter()
        obs = observe(decode(jpegs[i % len(jpegs)]))
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    print(f"frames {n}  JPEG 640x480 -> obs {obs.shape} {obs.dtype}")
    print(f"median {np.median(ts):.2f} ms  p95 {np.percentile(ts, 95):.2f} ms  max {ts.max():.2f} ms")
    print(f"pictures per second: {1000 / np.median(ts):.1f} (median), {1000 / np.percentile(ts, 95):.1f} (p95)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--image")
    a = ap.parse_args()
    if a.benchmark:
        benchmark()
    elif a.image:
        rgb = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
        out = os.path.splitext(a.image)[0] + "_obs.png"
        cv2.imwrite(out, (observe(rgb) * 255).astype(np.uint8))
        print(out)
    else:
        ap.print_help()
