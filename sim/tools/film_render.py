"""Render a 3DGUT/3DGS PLY (scan frame, metres, +Z up) from given camera poses with gsplat.

Runs on the GPU VM inside the 3dgrut venv:
  python film_render.py splat.ply poses.json out_dir
poses.json: {"poses": [{"file": "x.jpg", "C": [x,y,z], "R_c2w": 3x3 OpenCV camera-to-world
(columns = right, down, forward), "fx","fy","cx","cy","w","h"}], "background": [r,g,b] 0-1}.
Uses gsplat's unscented-transform rasterizer (with_ut, with_eval3d), the 3DGUT render model.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from plyfile import PlyData


def load_ply(path, dev):
    v = PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    dc = np.stack([v["f_dc_%d" % i] for i in range(3)], 1)[:, None, :]
    rest_names = sorted([n for n in names if n.startswith("f_rest_")], key=lambda n: int(n[7:]))
    if rest_names:
        rest = np.stack([v[n] for n in rest_names], 1).reshape(len(xyz), 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest], 1)
    else:
        sh = dc
    deg = int(round(np.sqrt(sh.shape[1]))) - 1
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    return dict(
        means=t(xyz),
        quats=torch.nn.functional.normalize(t(np.stack([v["rot_%d" % i] for i in range(4)], 1)), dim=1),
        scales=torch.exp(t(np.stack([v["scale_%d" % i] for i in range(3)], 1))),
        opacities=torch.sigmoid(t(v["opacity"])),
        colors=t(sh), sh_degree=deg)


def main(ply, poses_path, out_dir):
    dev = "cuda"
    g = load_ply(ply, dev)
    spec = json.load(open(poses_path))
    bg = torch.tensor(spec.get("background", [0, 0, 0]), dtype=torch.float32, device=dev)[None]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for p in spec["poses"]:
        Rc = np.array(p["R_c2w"], float)
        C = np.array(p["C"], float)
        w2c = np.eye(4)
        w2c[:3, :3] = Rc.T
        w2c[:3, 3] = -Rc.T @ C
        K = np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]])
        img, alpha, _ = rasterization(
            g["means"], g["quats"], g["scales"], g["opacities"], g["colors"],
            torch.tensor(w2c, dtype=torch.float32, device=dev)[None],
            torch.tensor(K, dtype=torch.float32, device=dev)[None],
            int(p["w"]), int(p["h"]), sh_degree=g["sh_degree"], near_plane=0.02, far_plane=100.0,
            backgrounds=bg, with_ut=True, with_eval3d=True, packed=False)
        arr = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        out = os.path.join(out_dir, p["file"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(arr).save(out, quality=92)
    print("RENDER %d views from %d gaussians in %.1fs" % (len(spec["poses"]), len(g["means"]), time.time() - t0))


if __name__ == "__main__":
    main(*sys.argv[1:4])
