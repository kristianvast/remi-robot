"""Build fixed route pools: sim/.venv/bin/python sim/isaac/elegoo/train/routes.py.

  routes.py                    train / dev / test pools from the spawn poses (rewrites routes.npz)
  routes.py --add-home-short   keeps every existing array and adds home_{train,test} (start beside the office middle
                               desk, HOME_XY, any heading) and short_{train,test} (straight 0.3-1.5 m moves)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.graph import MCP_Geometric

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nav"))
from plan import CAR_CLEARANCE_M, META, Map, RES, X0, Y0, to_px

HOME_XY = (6.23, -1.23)  # = nav/env.py OFFICE_DESK_XY
HOME_RADIUS_M = 0.15
SPACING_M = 0.22


def cost_map() -> np.ndarray:
    floor = Map()
    cost = np.where(
        floor.clearance > CAR_CLEARANCE_M,
        1 + 3 * np.maximum(0, 0.45 - floor.clearance) / 0.45,
        np.inf,
    )
    closure = json.loads((HERE.parent / "nav/building_plan.json").read_text())["closure"]
    for ax, ay, bx, by in closure:
        t = np.linspace(0, 1, int(np.hypot(bx - ax, by - ay) / RES * 2) + 2)
        rows = np.rint((Y0 - ay - t * (by - ay)) / RES).astype(int)
        cols = np.rint((ax + t * (bx - ax) - X0) / RES).astype(int)
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                if dx * dx + dy * dy <= 8**2:
                    cost[np.clip(rows + dy, 0, cost.shape[0] - 1),
                         np.clip(cols + dx, 0, cost.shape[1] - 1)] = np.inf
    return cost


def resample(path_px: np.ndarray) -> tuple[np.ndarray, float]:
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path_px, axis=0), axis=1) * RES)]
    xy = np.column_stack((X0 + path_px[:, 1] * RES, Y0 - path_px[:, 0] * RES))
    way = np.column_stack([np.interp(np.r_[np.arange(0, s[-1], SPACING_M), s[-1]], s, xy[:, j]) for j in range(2)])
    return way, float(s[-1])


def pack(split: str, routes: list) -> dict:
    xy, yaw, length, area = zip(*routes, strict=True)
    count = np.array([len(p) for p in xy])
    print(split, len(routes), dict(zip(*np.unique(area, return_counts=True), strict=True)))
    return {
        f"{split}_xy": np.stack([np.pad(p, ((0, count.max() - len(p)), (0, 0)), mode="edge") for p in xy]).astype(np.float32),
        f"{split}_yaw": np.asarray(yaw, dtype=np.float32),
        f"{split}_length": np.asarray(length, dtype=np.float32),
        f"{split}_area": np.asarray(area),
        f"{split}_count": count,
    }


def add_home_short() -> None:
    """Home routes start on free cells within HOME_RADIUS_M of HOME_XY and go to the split's own route goals; short
    routes leave the split's own start cells straight along a clear heading. Existing arrays are copied unchanged."""
    rng = np.random.default_rng(911)
    cost = cost_map()
    with np.load(HERE / "routes.npz") as data:
        packed = {k: data[k] for k in data.files}
    hx, hy = to_px(*HOME_XY)
    r = int(HOME_RADIUS_M / RES)
    home_cells = [(hx + dy, hy + dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                  if dx * dx + dy * dy <= r * r and np.isfinite(cost[hx + dy, hy + dx])]
    for split, n_home, n_short, max_len in (("train", 480, 480, 14), ("test", 80, 80, 9)):
        way, count = packed[f"{split}_xy"], packed[f"{split}_count"]
        goals = list(dict.fromkeys(to_px(*way[i, count[i] - 1]) for i in range(len(count))))
        home = []
        while len(home) < n_home:
            cell = home_cells[rng.integers(len(home_cells))]
            solver = MCP_Geometric(cost)
            totals, _ = solver.find_costs([cell], goals, find_all_ends=True)
            for goal in rng.permutation(len(goals))[:12]:
                goal = goals[goal]
                if not np.isfinite(totals[goal]):
                    continue
                path, length = resample(np.asarray(solver.traceback(goal)))
                if 0.9 <= length <= max_len and len(home) < n_home:
                    home.append((path, rng.uniform(-np.pi, np.pi), length, "home"))
        short = []
        starts = [(to_px(*way[i, 0]), packed[f"{split}_area"][i]) for i in range(len(count))]
        while len(short) < n_short:
            (row, col), area = starts[rng.integers(len(starts))]
            heading, length = rng.uniform(-np.pi, np.pi), rng.uniform(0.3, 1.5)
            t = np.linspace(0, length / RES, int(length / RES) + 1)
            line = np.column_stack((row - t * np.sin(heading), col + t * np.cos(heading)))
            cells = np.rint(line).astype(int)
            if (cells < 0).any() or (cells[:, 0] >= cost.shape[0]).any() or (cells[:, 1] >= cost.shape[1]).any():
                continue
            if np.isfinite(cost[cells[:, 0], cells[:, 1]]).all():
                path, length = resample(line)
                short.append((path, heading + rng.uniform(-0.15, 0.15), length, f"short_{area}"))
        packed.update(pack(f"home_{split}", home))
        packed.update(pack(f"short_{split}", short))
    np.savez_compressed(HERE / "routes.npz", **packed)


def main() -> None:
    """Select start cells by area before planning; never recycle a test start in training."""
    rng = np.random.default_rng(723)
    cost = cost_map()
    starts = defaultdict(list)
    for p in META["spawn_poses"]:
        cell = to_px(p["x"], p["y"])
        if np.isfinite(cost[cell]):
            starts[p["area"]].append(cell)
    named = [to_px(p["x"], p["y"]) for p in META["goals"]]
    named = [cell for cell in named if np.isfinite(cost[cell])]
    splits = defaultdict(list)
    for area, cells in sorted(starts.items()):
        cells = list(dict.fromkeys(cells))
        rng.shuffle(cells)
        held = min(max(2, len(cells) // 5), len(cells) - 2)
        for i, cell in enumerate(cells):
            split = "test" if i < held else "dev" if i == held else "train"
            routes = []
            # Local pseudo-goals supplement fixed named goals, especially in small areas.
            local = rng.integers(-180, 181, size=(160, 2)) + np.array(cell)
            candidates = named + [
                (int(y), int(x)) for y, x in local
                if 0 <= y < cost.shape[0] and 0 <= x < cost.shape[1] and np.isfinite(cost[y, x])
            ]
            solver = MCP_Geometric(cost)
            totals, _ = solver.find_costs([cell], list(dict.fromkeys(candidates)), find_all_ends=True)
            for goal in dict.fromkeys(candidates):
                if not np.isfinite(totals[goal]) or not 0.8 < np.linalg.norm(np.subtract(cell, goal)) * RES < 18:
                    continue
                way, length = resample(np.asarray(solver.traceback(goal)))
                if not 0.9 <= length <= (9 if split != "train" else 14):
                    continue
                d = way[2] - way[0]
                yaw = np.arctan2(d[1], d[0]) + rng.uniform(-0.6, 0.6)
                routes.append((way, yaw, length, area))
            rng.shuffle(routes)
            splits[split].extend(routes[: (max(6, (24 + held - 1) // held) if split == "test" else 16)])
    packed = {}
    for split, routes in splits.items():
        packed.update(pack(split, routes))
    assert len(splits["test"]) >= 200 and len(set(p[3] for p in splits["test"])) == len(starts)
    np.savez_compressed(HERE / "routes.npz", **packed)


if __name__ == "__main__":
    add_home_short() if "--add-home-short" in sys.argv else main()
