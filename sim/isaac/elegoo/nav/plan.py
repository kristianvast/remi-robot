"""Host-side planning on the floor map for the building tests (no Isaac needed).

Writes next to this file:
  building_plan.json
    drive     - a clearance-weighted grid path through several areas (stage frame, metres)
    collide   - straight-line test runs: wall, furniture, lifted patch, each checked against
                the 3D collider with horizontal rays at bumper height
    closure   - merged bottom edges of /World/Shell/ClosureWalls, where no scan wall exists
  free_map.npz - floor-map free cells dilated by 6 cm, for the off-map episode flag

Run on the Mac: sim/.venv/bin/python sim/isaac/elegoo/nav/plan.py
"""

import json
from pathlib import Path

import numpy as np
import trimesh
import yaml
from PIL import Image
from pxr import Usd, UsdGeom
from scipy import ndimage
from skimage.graph import route_through_array
from skimage.measure import points_in_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MERGED = ROOT / "scans" / "merged"
SCENE = HERE.parents[1] / "scene"
META = yaml.safe_load((SCENE / "scene_meta.yaml").read_text())
FREE_MARGIN_PX = 3
RES = 0.02
X0, Y0 = -11.49, 11.17  # centre of pixel (0, 0); y decreases with row
CAR_CLEARANCE_M = 0.16  # body half-diagonal is about 0.135 m
LIFT_CENTRE = (1.28, -6.62)


def to_px(x, y):
    return int(round((Y0 - y) / RES)), int(round((x - X0) / RES))


def to_xy(row, col):
    return X0 + col * RES, Y0 - row * RES


class Map:
    def __init__(self) -> None:
        img = np.array(Image.open(MERGED / "floor_map.png"))
        self.free = img == 254
        self.occupied = img == 0
        self.clearance = ndimage.distance_transform_edt(self.free) * RES
        labels, n = ndimage.label(self.occupied, structure=np.ones((3, 3)))
        sizes = ndimage.sum(self.occupied, labels, range(1, n + 1)) * RES * RES
        self.labels, self.sizes = labels, sizes
        rooms = json.loads((MERGED / "rooms_next.json").read_text())
        self.rooms = [(r["id"], np.array(r["polygon"])) for r in rooms]

    def area_of(self, x: float, y: float) -> str | None:
        for name, poly in self.rooms:
            if points_in_poly([[x, y]], poly)[0]:
                return name
        return None

    def march(self, x, y, yaw, max_m=3.0):
        """Distance until the car's swept corridor leaves free space, and the blocking pixel."""
        d = np.array([np.cos(yaw), np.sin(yaw)])
        n = np.array([-d[1], d[0]])
        for s in np.arange(0.0, max_m, RES):
            for w in np.linspace(-0.07, 0.07, 5):
                px, py = np.array([x, y]) + d * s + n * w
                r, c = to_px(px, py)
                if not self.free[r, c]:
                    return s, (r, c)
        return max_m, None


def horizontal_hits(mesh, x, y, yaw, max_m):
    """Collider hit distance along the heading at three bumper heights (front edge offset 0.115 m)."""
    d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    n = np.array([-d[1], d[0], 0.0])
    origins, dirs = [], []
    for z in (0.03, 0.08, 0.13):
        for w in (-0.06, 0.0, 0.06):
            origins.append(np.array([x, y, z]) + n * w)
            dirs.append(d)
    loc, idx, _ = mesh.ray.intersects_location(np.array(origins), np.array(dirs), multiple_hits=False)
    dist = np.full(len(origins), np.inf)
    dist[idx] = np.linalg.norm(loc - np.array(origins)[idx], axis=1)
    return float(dist.min()) if np.isfinite(dist).any() else None


def column_top(mesh, x, y):
    loc, _, _ = mesh.ray.intersects_location([[x, y, 3.0]], [[0, 0, -1.0]], multiple_hits=False)
    return float(loc[0, 2]) if len(loc) else None


def plan_drive(m: Map) -> dict:
    cost = np.where(m.clearance > CAR_CLEARANCE_M, 1.0 + 3.0 * np.clip(0.45 - m.clearance, 0, None) / 0.45, np.inf)
    cost = np.where(np.isfinite(cost), cost, 1e6)
    goals = META["goals"]
    best = None
    for a in goals:
        for b in goals:
            if a["area"] >= b["area"]:
                continue
            pa, pb = to_px(a["x"], a["y"]), to_px(b["x"], b["y"])
            if m.clearance[pa] <= CAR_CLEARANCE_M or m.clearance[pb] <= CAR_CLEARANCE_M:
                continue
            path, total = route_through_array(cost, pa, pb, fully_connected=True, geometric=True)
            path = np.array(path)
            if cost[path[:, 0], path[:, 1]].max() >= 1e6:
                continue
            xy = np.array([to_xy(r, c) for r, c in path])
            length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            areas = []
            for x, y in xy[::10]:
                name = m.area_of(x, y)
                if name and (not areas or areas[-1] != name):
                    areas.append(name)
            distinct = len(set(areas))
            if length < 15 or distinct < 3 or length > 30:
                continue
            score = distinct * 10 + length
            if best is None or score > best["score"]:
                best = dict(score=score, start=a, goal=b, length_m=length, areas=areas, xy=xy)
    xy = best["xy"]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    keep = np.searchsorted(s, np.arange(0, s[-1], 0.25))
    way = np.vstack([xy[keep], xy[-1]])
    first = way[min(4, len(way) - 1)] - way[0]
    return {
        "start": best["start"]["name"], "goal": best["goal"]["name"],
        "length_m": round(best["length_m"], 2), "areas": best["areas"],
        "start_yaw": float(np.arctan2(first[1], first[0])),
        "min_clearance_m": float(min(m.clearance[to_px(x, y)] for x, y in way)),
        "waypoints": np.round(way, 3).tolist(),
    }


def plan_collisions(m: Map, mesh) -> dict:
    wall_label = np.argmax(m.sizes) + 1  # the connected outline is the largest occupied component
    candidates = {"wall": [], "furniture": []}
    for pose in META["spawn_poses"]:
        for yaw in np.linspace(-np.pi, np.pi, 36, endpoint=False):
            dist, hit = m.march(pose["x"], pose["y"], yaw)
            if hit is None or not 0.9 <= dist <= 1.6:
                continue
            label = m.labels[hit]
            hx, hy = to_xy(*hit)
            top = column_top(mesh, hx, hy)
            solid = horizontal_hits(mesh, pose["x"], pose["y"], yaw, dist + 0.3)
            if solid is None or abs(solid - dist) > 0.12 or top is None:
                continue
            item = dict(area=pose["area"], x=pose["x"], y=pose["y"], yaw=float(yaw), map_hit_m=float(dist),
                        collider_hit_m=solid, hit_xy=[hx, hy], obstacle_top_m=top,
                        component_m2=float(m.sizes[label - 1]))
            if label == wall_label and top > 0.8:
                candidates["wall"].append(item)
            elif label != wall_label and 0.05 <= m.sizes[label - 1] <= 2.0 and 0.2 <= top <= 1.2:
                candidates["furniture"].append(item)
    runs = {}
    for kind, items in candidates.items():
        # Head-on approach: prefer the hit closest to the straight-line prediction from the map.
        items.sort(key=lambda it: abs(it["collider_hit_m"] - it["map_hit_m"]) + abs(it["map_hit_m"] - 1.2))
        runs[kind] = items[0]
        runs[kind]["candidates"] = len(items)
    lx, ly = LIFT_CENTRE
    lifted = []
    for yaw in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        start = (lx - 1.0 * np.cos(yaw), ly - 1.0 * np.sin(yaw))
        if not m.free[to_px(*start)]:
            continue
        dist, _ = m.march(*start, yaw, max_m=2.2)
        solid = horizontal_hits(mesh, *start, yaw, 2.5)
        lifted.append(dict(area=m.area_of(lx, ly), x=float(start[0]), y=float(start[1]), yaw=float(yaw),
                           free_ahead_m=float(dist), collider_hit_m=solid))
    runs["lifted"] = max(lifted, key=lambda it: min(it["free_ahead_m"], it["collider_hit_m"] or 9.0))
    return runs


def plan_closure_run(m: Map, closure: list[list[float]]) -> dict:
    """Approach the middle of the longest closure segment head-on from the free side, 1.0 m away."""
    x0, y0, x1, y1 = max(closure, key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]))
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    normal = np.array([-(y1 - y0), x1 - x0]) / np.hypot(x1 - x0, y1 - y0)
    for sign in (1, -1):
        start = mid + sign * normal * 1.0
        if m.free[to_px(*start)]:
            yaw = float(np.arctan2(-sign * normal[1], -sign * normal[0]))
            dist, _ = m.march(*start, yaw, max_m=2.0)
            return dict(area=m.area_of(*start), x=float(start[0]), y=float(start[1]), yaw=yaw,
                        segment=[x0, y0, x1, y1], wall_distance_m=1.0, map_free_ahead_m=float(dist))
    raise RuntimeError("no free side next to the longest closure segment")


def crossings(path: list[list[float]], closure: list[list[float]]) -> int:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    count = 0
    for p, q in zip(path, path[1:]):
        for x0, y0, x1, y1 in closure:
            a, b = (x0, y0), (x1, y1)
            if ccw(p, a, b) != ccw(q, a, b) and ccw(p, q, a) != ccw(p, q, b):
                count += 1
    return count


def closure_segments() -> list[list[float]]:
    """Bottom edges of the closure-wall quads, merged where consecutive pieces are collinear."""
    stage = Usd.Stage.Open(str(SCENE / "layers" / "shell.usda"))
    segments = []
    for prim in stage.Traverse():
        if "/ClosureWalls/" not in str(prim.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get())
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
        run = []
        for face in faces:
            low = points[face][points[face][:, 2] < 0.5][:, :2]
            a, b = low[0], low[1]
            if run and np.allclose(run[-1][1], a, atol=0.01):
                d0 = run[-1][1] - run[-1][0]
                d1 = b - a
                cos = d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1))
                if cos > np.cos(np.radians(5)):
                    run[-1][1] = b
                    continue
            run.append([a, b])
        segments += [[*map(float, a), *map(float, b)] for a, b in run]
    return [[round(v, 3) for v in s] for s in segments]


def main() -> None:
    m = Map()
    mesh = trimesh.load(MERGED / "collision.obj", force="mesh")
    closure = closure_segments()
    plan = {"drive": plan_drive(m), "collide": plan_collisions(m, mesh), "closure": closure}
    plan["collide"]["closure"] = plan_closure_run(m, closure)
    plan["drive"]["closure_crossings"] = crossings(plan["drive"]["waypoints"], closure)
    (HERE / "building_plan.json").write_text(json.dumps(plan, indent=1))
    allowed = ndimage.binary_dilation(m.free, iterations=FREE_MARGIN_PX)
    np.savez_compressed(HERE / "free_map.npz", allowed=np.packbits(allowed), shape=allowed.shape,
                        x0=X0, y0=Y0, res=RES)
    length = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in closure)
    print(f"CLOSURE segments={len(closure)} length={length:.2f} m; free_map allowed={allowed.mean():.3f} "
          f"drive_crossings={plan['drive']['closure_crossings']}")
    d = plan["drive"]
    print(f"DRIVE {d['start']} -> {d['goal']} length={d['length_m']} m areas={d['areas']} "
          f"min_clearance={d['min_clearance_m']:.3f} waypoints={len(d['waypoints'])}")
    for k, v in plan["collide"].items():
        print("COLLIDE", k, {key: (round(val, 3) if isinstance(val, float) else val) for key, val in v.items()})


if __name__ == "__main__":
    main()
