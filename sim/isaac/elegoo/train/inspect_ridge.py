"""Look for a ridge in the collider where every controller tilted near its area_07 start (driver.md).

  sim/cloud/run_headless.sh drv/elegoo/train/inspect_ridge.py --output /workspace/elegoo/drv/ridge --viz none
Rasterises the collision mesh (the one PhysX uses) into 2 cm cells around the failures, keeps the highest
point below 15 cm per cell, and prints the clusters of cells that the floor map calls free but that rise
1-8 cm above the floor, with the old evaluation's area_07 failure points. Writes heights.npz and heights.png.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
REGION = (-1.5, -8.8, 2.0, -5.2)   # x0, y0, x1, y1 around the failures at x 0.2-0.4, y -6.8 to -7.3
CELL = 0.02
LOW, HIGH = 0.01, 0.08


def analyse(output: Path) -> None:
    from pxr import Usd, UsdGeom
    from scipy import ndimage

    from elegoo.nav.env import BUILDING_USD, COLLIDER, HERE as NAV

    stage = Usd.Stage.Open(str(BUILDING_USD))
    prim = stage.GetPrimAtPath(COLLIDER.replace("/World/Building", "/World", 1))
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    index = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    points = np.c_[points, np.ones(len(points))] @ matrix
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    tris = np.concatenate([np.stack((index[s], index[s + k], index[s + k + 1]), 1)
                           for k in range(1, counts.max() - 1)
                           for s in [starts[counts > k + 1]]])
    corners = points[tris, :3]
    x0, y0, x1, y1 = REGION
    centre = corners.mean(1)
    keep = (centre[:, 0] > x0) & (centre[:, 0] < x1) & (centre[:, 1] > y0) & (centre[:, 1] < y1) & (centre[:, 2] < 0.15)
    corners = corners[keep]
    a, b = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9))
    a, b = a[a + b <= 1], b[a + b <= 1]
    samples = (corners[:, None, 0] * (1 - a - b)[None, :, None] + corners[:, None, 1] * a[None, :, None]
               + corners[:, None, 2] * b[None, :, None]).reshape(-1, 3)
    cols, rows = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    ci = ((samples[:, 0] - x0) / CELL).astype(int).clip(0, cols - 1)
    ri = ((y1 - samples[:, 1]) / CELL).astype(int).clip(0, rows - 1)
    top = np.full((rows, cols), np.nan)
    np.fmax.at(top, (ri, ci), samples[:, 2])
    floor = np.nanmedian(top[top < 0.01]) if np.any(top < 0.01) else 0.0
    rise = top - floor
    free = np.load(NAV / "free_map.npz")
    shape = tuple(free["shape"])
    allowed = np.unpackbits(free["allowed"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    fx0, fy0, res = float(free["x0"]), float(free["y0"]), float(free["res"])
    xs = x0 + (np.arange(cols) + 0.5) * CELL
    ys = y1 - (np.arange(rows) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    on_map = allowed[np.clip(np.round((fy0 - gy) / res).astype(int), 0, shape[0] - 1),
                     np.clip(np.round((gx - fx0) / res).astype(int), 0, shape[1] - 1)]
    bump = on_map & (rise >= LOW) & (rise <= HIGH)
    labels, n = ndimage.label(bump)
    clusters = []
    for k in range(1, n + 1):
        cells = labels == k
        if cells.sum() < 3:
            continue
        clusters.append((int(cells.sum()), float(gx[cells].min()), float(gx[cells].max()), float(gy[cells].min()),
                         float(gy[cells].max()), float(np.nanmax(rise[cells]))))
    clusters.sort(reverse=True)
    print(f"RIDGE floor_z={floor:.4f} triangles={int(keep.sum())} free_cells={int(on_map.sum())} "
          f"raised_free_cells={int(bump.sum())}", flush=True)
    for size, ax, bx, ay, by, peak in clusters[:15]:
        print(f"RIDGE_CLUSTER cells={size} x=({ax:.2f},{bx:.2f}) y=({ay:.2f},{by:.2f}) max_rise_cm={peak * 100:.1f}",
              flush=True)
    failures = []
    with (HERE / "results/episodes.csv").open() as f:
        for e in csv.DictReader(f):
            if e["area"] == "area_07" and e["condition"] == "nominal" and e["success"] == "0":
                failures.append((float(e["end_x"]), float(e["end_y"]), int(e["route"]), e["controller"]))
    with np.load(HERE / "routes.npz") as data:
        way = data["test_xy"]
    for x, y, route, controller in failures:
        start = way[route, 0]
        c = int(np.clip((x - x0) / CELL, 0, cols - 1))
        r = int(np.clip((y1 - y) / CELL, 0, rows - 1))
        near = rise[max(r - 5, 0):r + 6, max(c - 5, 0):c + 6]
        print(f"RIDGE_FAIL route={route} {controller} start=({start[0]:.2f},{start[1]:.2f}) end=({x:.2f},{y:.2f}) "
              f"max_rise_within_10cm_cm={np.nanmax(near) * 100:.1f}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "heights.npz", rise=rise, on_map=on_map, region=np.array(REGION), cell=CELL)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    shown = ax.imshow(np.where(on_map, rise * 100, np.nan), extent=(x0, x1, y0, y1), cmap="magma", vmin=0, vmax=8)
    ax.contour(xs, ys, on_map, levels=[0.5], colors="cyan", linewidths=0.6)
    ax.scatter([f[0] for f in failures], [f[1] for f in failures], c="lime", s=12, label="area_07 failure ends")
    ax.scatter([way[f[2], 0, 0] for f in failures], [way[f[2], 0, 1] for f in failures], c="white", marker="x", s=20,
               label="their starts")
    fig.colorbar(shown, label="collider rise above floor (cm), free-map cells only")
    ax.legend(loc="upper right")
    ax.set_title("area_07: highest collider point below 15 cm per 2 cm cell")
    fig.savefig(output / "heights.png", dpi=110)
    print("RIDGE_DONE", output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    sim_cfg = SimulationCfg(dt=0.01, device=args.device)
    with launch_simulation(sim_cfg, args):
        analyse(args.output)


if __name__ == "__main__":
    main()
