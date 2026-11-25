"""Read-only geometry shared by simulation, planning and plotting."""

from functools import lru_cache
import json
from pathlib import Path
import re

import cv2
import numpy as np
from embreex.mesh_construction import TriangleMesh
from embreex.rtcore_scene import EmbreeScene
from PIL import Image
from scipy.ndimage import distance_transform_edt
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "scans/merged"
ISAAC = ROOT / "sim/isaac"
CAR_RADIUS = 0.135  # circumscribes the 23 × 14 cm footprint
Pose = tuple[float, float, float]


@lru_cache(maxsize=1)
def scene_meta():
    return yaml.safe_load((ISAAC / "scene/scene_meta.yaml").read_text())


class Floor:
    """Map coordinates and conservative clearance from unknown/occupied/closure."""

    def __init__(self, image: Path | None = None):
        """image: an occupancy PNG with the scan map's yaml (same origin and resolution); default the scan's."""
        info = yaml.safe_load((SCAN / "floor_map.yaml").read_text())
        self.res = info["resolution"]
        self.x0, y0, _ = info["origin"]
        with Image.open(image or SCAN / "floor_map.png") as img:
            self.image = np.asarray(img.convert("L"))
        self.ytop = y0 + self.image.shape[0] * self.res
        self.closures = np.array(json.loads((ISAAC / "elegoo/nav/building_plan.json").read_text())["closure"])
        free = ((1. - self.image / 255.) < info["free_thresh"]).astype(np.uint8)
        for ax, ay, bx, by in self.closures:
            a = tuple(int(v) for v in self.cells(ax, ay)[::-1])
            b = tuple(int(v) for v in self.cells(bx, by)[::-1])
            cv2.line(free, a, b, 0, 3)
        self.free = free.astype(bool)
        self.clearance = (distance_transform_edt(self.free) * self.res - self.res / 2).astype(np.float32)
        self.rooms = json.loads((SCAN / "rooms_next.json").read_text())
        self.labels = np.zeros_like(free)
        for i, room in enumerate(self.rooms, 1):
            poly = np.asarray(room["polygon"])
            r, c = self.cells(*poly.T)
            mask = np.zeros_like(free)
            cv2.fillPoly(mask, [np.column_stack([c, r]).astype(np.int32)], 1)
            for ring in room.get("holes", []):
                r, c = self.cells(*np.asarray(ring).T)
                cv2.fillPoly(mask, [np.column_stack([c, r]).astype(np.int32)], 0)
            self.labels[mask != 0] = i

    def cells(self, x, y):
        return (np.floor((self.ytop - y) / self.res).astype(int),
                np.floor((x - self.x0) / self.res).astype(int))

    def xy(self, r, c):
        return self.x0 + (c + 0.5) * self.res, self.ytop - (r + 0.5) * self.res

    def valid(self, x, y, radius: float = CAR_RADIUS):
        r, c = self.cells(x, y)
        h, w = self.free.shape
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        return inside & (self.clearance[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)] >= radius)

    def valid_point(self, x: float, y: float, radius: float = CAR_RADIUS) -> bool:
        r = int((self.ytop-y)/self.res)
        c = int((x-self.x0)/self.res)
        h, w = self.free.shape
        return 0 <= r < h and 0 <= c < w and self.clearance[r, c] >= radius

    def area(self, x: float, y: float) -> str:
        r, c = self.cells(x, y)
        if 0 <= r < self.labels.shape[0] and 0 <= c < self.labels.shape[1] and self.labels[r, c]:
            return self.rooms[self.labels[r, c] - 1]["id"]
        return ""


@lru_cache(maxsize=1)
def floor() -> Floor:
    return Floor()


def _scan_triangles() -> np.ndarray:
    vertices, faces = [], []
    with (SCAN / "collision.obj").open() as src:
        for line in src:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                faces.extend((indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1))
    return np.asarray(vertices, dtype=np.float32)[np.asarray(faces)]


def _shell_triangles() -> np.ndarray:
    """Read the polygon arrays from the generated, untransformed ASCII layer."""
    layer = (ISAAC / "scene/layers/shell.usda").read_text()
    triangles = []
    for block in layer.split('def Mesh "')[1:]:
        points = np.fromstring(re.search(r"\bpoints = \[([^\]]+)", block)[1]
                               .replace("(", "").replace(")", ""), sep=",").reshape(-1, 3)
        counts = np.fromstring(re.search(r"faceVertexCounts = \[([^\]]+)", block)[1], sep=",", dtype=int)
        indices = np.fromstring(re.search(r"faceVertexIndices = \[([^\]]+)", block)[1], sep=",", dtype=int)
        start = 0
        for n in counts:
            face = indices[start:start+n]
            triangles.extend(points[[face[0], face[j], face[j+1]]] for j in range(1, n-1))
            start += n
    return np.asarray(triangles, dtype=np.float32)


@lru_cache(maxsize=2)
def geometry(optical: bool = True) -> EmbreeScene:
    scan = _scan_triangles()
    scene = EmbreeScene()
    if optical:
        triangles = np.concatenate((scan, _shell_triangles()))
    else:
        # Reject slab / sub-1.5 cm floor-only triangles without masking a
        # wall behind the floor. Closure segments get physical echo faces.
        a, b, c, d = floor().closures.T
        z = np.full_like(a, 0.015)
        h = np.full_like(a, 0.5)
        p, q = np.column_stack((a, b, z)), np.column_stack((c, d, z))
        r, s = np.column_stack((c, d, h)), np.column_stack((a, b, h))
        triangles = np.concatenate((scan[scan[:, :, 2].max(axis=1) >= .015],
                                    np.stack((p, q, r), axis=1), np.stack((p, r, s), axis=1)))
    TriangleMesh(scene, triangles.astype(np.float32, copy=False))
    return scene


def cast(scene: EmbreeScene, origin, rays: np.ndarray) -> np.ndarray:
    dirs = np.ascontiguousarray(rays, dtype=np.float32)
    origins = np.ascontiguousarray(np.broadcast_to(origin, dirs.shape), dtype=np.float32)
    hit = scene.run(origins, dirs, query="DISTANCE")
    hit[hit >= 1e36] = np.inf
    return hit
