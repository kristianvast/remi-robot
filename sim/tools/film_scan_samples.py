import sys, time
import numpy as np
import trimesh

R = sys.argv[1]
t = time.time()
m = trimesh.load(f"{R}/scans/merged/building.glb", force="mesh")
print("mesh", m.faces.shape, m.bounds.tolist(), round(time.time() - t, 1), "s")
pts, fi = trimesh.sample.sample_surface(m, 2_000_000)
nrm = m.face_normals[fi]
np.savez_compressed(f"{R}/scans/film/work/scan_samples.npz", pts=pts.astype(np.float32), nrm=nrm.astype(np.float32))
print("z pct", np.percentile(pts[:, 2], [1, 25, 50, 75, 99]).round(2).tolist(),
      "vertical frac", round(float((np.abs(nrm[:, 2]) < 0.3).mean()), 3))
for lo, hi in [(0.3, 0.8), (0.8, 1.3), (1.3, 1.8), (1.8, 2.3)]:
    print("band", lo, hi, int(((pts[:, 2] > lo) & (pts[:, 2] < hi)).sum()))
