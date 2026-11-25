"""12 noisy re-encodes of each car photo (Gaussian sd 4 + JPEG q 12-30) -> data/noise/, for analyze.py."""
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "noise")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    for p in sorted(glob.glob(os.path.join(HERE, "..", "..", "patrol-photos", "bench", "*.jpg"))):
        im = cv2.imread(p).astype(np.float32)
        stem = os.path.splitext(os.path.basename(p))[0]
        for k in range(12):
            noisy = np.clip(im + rng.normal(0, 4, im.shape), 0, 255).astype(np.uint8)
            q = int(rng.integers(12, 31))
            cv2.imwrite(os.path.join(OUT, f"{stem}_n{k:02d}.jpg"), noisy, [cv2.IMWRITE_JPEG_QUALITY, q])
    print("wrote", len(os.listdir(OUT)))
