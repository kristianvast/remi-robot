"""Truth rasters for the place and goal-direction heads (run on elegoo-sim-3, numpy + scipy + scikit-image).

  python labels.py <project_root> <out.npz>

Writes, on the floor map's 2 cm grid (pixel (r, c) centre = (X0 + c RES, Y0 - r RES)):
  area   uint8 [H,W]: index into common.AREAS of the rooms_next.json polygon holding the cell; cells outside every
         polygon take the nearest polygon's index (so every pose has a room label)
  angle  float16 [10,H,W]: world direction (rad, CCW from +x) of steepest descent of the geodesic distance to each
         area's default named goal (places.yaml), i.e. the way to drive toward that place
  dist   float16 [10,H,W]: that geodesic distance in metres (inf = unreachable)
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from skimage.graph import MCP_Geometric
from skimage.measure import points_in_poly

RES, X0, Y0 = 0.02, -11.49, 11.17
CAR_CLEARANCE_M = 0.16


def main(root: Path, out: Path) -> None:
    img = np.array(Image.open(root / "scans/merged/floor_map.png"))
    free = img == 254
    h, w = free.shape
    clearance = ndimage.distance_transform_edt(free) * RES
    cost = np.where(clearance > CAR_CLEARANCE_M, 1 + 3 * np.maximum(0, 0.45 - clearance) / 0.45, np.inf)
    rows, cols = np.mgrid[0:h, 0:w]
    xy = np.column_stack((X0 + cols.ravel() * RES, Y0 - rows.ravel() * RES))
    rooms = {r["id"]: np.array(r["polygon"]) for r in json.loads((root / "scans/merged/rooms_next.json").read_text())}
    areas = sorted(rooms)
    assert areas == [f"area_{i:02d}" for i in range(1, 11)], areas
    label = np.full(h * w, 255, np.uint8)
    for i, a in enumerate(areas):
        label[points_in_poly(xy, rooms[a]) & (label == 255)] = i
    label = label.reshape(h, w)
    _, (ir, ic) = ndimage.distance_transform_edt(label == 255, return_indices=True)
    label = label[ir, ic]
    places = yaml.safe_load((root / "sim/navigate/places.yaml").read_text())["areas"]
    angle = np.zeros((len(areas), h, w), np.float16)
    dist = np.full((len(areas), h, w), np.inf, np.float16)
    for i, a in enumerate(areas):
        g = places[a]["goals"][places[a]["default"]]
        cell = (int(round((Y0 - g["y"]) / RES)), int(round((g["x"] - X0) / RES)))
        if not np.isfinite(cost[cell]):
            near = np.argwhere(np.isfinite(cost))
            cell = tuple(near[np.argmin(np.abs(near - cell).sum(1))])
        d, _ = MCP_Geometric(cost).find_costs([cell])
        d = d * RES
        finite = np.isfinite(d)
        _, (fr, fc) = ndimage.distance_transform_edt(~finite, return_indices=True)
        filled = np.where(finite, d, d[fr, fc] + 2 * np.hypot(rows - fr, cols - fc) * RES)
        smooth = ndimage.gaussian_filter(filled, 2.0)
        gy, gx = np.gradient(smooth)
        # Row index grows toward -y, so the world gradient is (gx, -gy); drive down it.
        angle[i] = np.arctan2(gy, -gx).astype(np.float16)
        dist[i] = np.where(finite, d, np.inf).astype(np.float16)
        print(a, "goal", cell, "reachable cells", int(finite.sum()), flush=True)
    np.savez_compressed(out, area=label, angle=angle, dist=dist, x0=X0, y0=Y0, res=RES, areas=np.array(areas))
    print("LABELS_OK", out, label.shape, np.bincount(label.ravel(), minlength=10).tolist())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
