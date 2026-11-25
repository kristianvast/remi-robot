"""Is batched Depth Pro (harness.DepthPro.batch) the same as realdepth.observe() frame by frame? Run in the image:
    python sim/cmdtest/depth_check.py <frame.jpg> ...
Prints repeatability of observe() itself, batch-of-1 vs observe, and batch-of-N vs observe (encoded units).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sim" / "isaac")]

from sim.cmdtest.harness import DepthPro  # noqa: E402


def main():
    frames = [cv2.cvtColor(cv2.resize(cv2.imread(p), (640, 480)), cv2.COLOR_BGR2RGB) for p in sys.argv[1:]]
    dp = DepthPro(torch)
    obs = dp.rd.observe
    single = [obs(f) for f in frames]
    again = [obs(f) for f in frames]
    one = [dp.batch([f])[0] for f in frames]
    many = dp.batch(frames)
    stat = lambda a, b: dict(max=round(max(float(np.abs(x - y).max()) for x, y in zip(a, b)), 5),
                             mean=round(float(np.mean([np.abs(x - y).mean() for x, y in zip(a, b)])), 5))
    print("DEPTH_CHECK", json.dumps(dict(frames=len(frames), observe_vs_observe=stat(single, again),
                                         batch1_vs_observe=stat(one, single), batchN_vs_observe=stat(many, single))))


if __name__ == "__main__":
    main()
