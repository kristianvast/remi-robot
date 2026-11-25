"""Re-extract the 392 selected film frames with HDR->SDR tone mapping.

Runs on the cloud VM:  python film_extract.py IMG_4474.MOV frames/ frames_tm/
The original frames were picked as the sharpest frame per third of a second, but their source
timestamps were not recorded. Every film frame is decoded with zscale+tonemap (Hable), then each
original frame is matched, in order, to the tone-mapped frame whose 160x90 zero-mean normalised
gradient image correlates best, so frames_tm/frame_NNNN.jpg is the same instant as frames/frame_NNNN.jpg.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
           "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=1920:-2:flags=lanczos")


def signature(path):
    a = np.asarray(Image.open(path).convert("L").resize((160, 90), Image.BILINEAR), np.float32)
    g = np.hypot(np.diff(a, axis=1)[:-1], np.diff(a, axis=0)[:, :-1])
    g -= g.mean()
    return (g / (np.linalg.norm(g) + 1e-6)).ravel()


def main(mov, orig_dir, out_dir):
    all_dir = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "all_tm")
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    if not glob.glob(all_dir + "/t_*.jpg"):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", mov, "-vf", TONEMAP,
                        "-q:v", "2", os.path.join(all_dir, "t_%05d.jpg")], check=True)
    cand = sorted(glob.glob(all_dir + "/t_*.jpg"))
    orig = sorted(glob.glob(orig_dir + "/frame_*.jpg"))
    cs = np.stack([signature(f) for f in cand])
    os_ = np.stack([signature(f) for f in orig])
    sim = os_ @ cs.T
    n, m = sim.shape
    # monotonic assignment (dynamic programming): orig i -> cand j_i with j strictly increasing
    best = np.full((n, m), -np.inf)
    arg = np.zeros((n, m), int)
    best[0] = sim[0]
    for i in range(1, n):
        shifted = np.concatenate([[-np.inf], best[i - 1][:-1]])
        prev = np.maximum.accumulate(shifted)
        is_new_max = np.concatenate([[True], shifted[1:] > prev[:-1]])
        prev_arg = np.maximum.accumulate(np.where(is_new_max, np.arange(m), 0)) - 1
        best[i] = sim[i] + prev
        arg[i] = np.maximum(prev_arg, 0)
    j = int(np.argmax(best[-1]))
    match = [0] * n
    for i in range(n - 1, -1, -1):
        match[i] = int(j)
        j = int(arg[i][j])
    scores = [float(sim[i, match[i]]) for i in range(n)]
    for i, f in enumerate(orig):
        Image.open(cand[match[i]]).save(os.path.join(out_dir, os.path.basename(f)), quality=95)
    with open(os.path.join(out_dir, "..", "frames_tm_match.json"), "w") as f:
        json.dump({"source_frames": len(cand), "match_index": match, "score": scores}, f)
    print("matched %d frames to %d film frames, score min %.3f median %.3f" % (
        n, len(cand), min(scores), float(np.median(scores))))


if __name__ == "__main__":
    main(*sys.argv[1:4])
