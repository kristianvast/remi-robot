"""Scan analysis for the Isaac Sim scene build: surface samples and plan rasters.

All rasters share the floor-map grid (floor_map.yaml: 0.02 m/px, lower-left
origin). Row 0 is the top image row, i.e. the largest y.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MERGED = os.path.join(ROOT, "scans", "merged")
CACHE = os.path.join(HERE, "cache")
GLB = os.path.join(MERGED, "building.glb")
COLLISION_OBJ = os.path.join(MERGED, "collision.obj")

SAMPLES_PER_M2 = 6000
FREE, OCC, UNKNOWN = 254, 0, 205


class Grid:
    """Floor-map pixel grid <-> stage XY (metres)."""

    def __init__(self):
        from PIL import Image

        meta = {}
        with open(os.path.join(MERGED, "floor_map.yaml")) as f:
            for line in f:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
        self.res = float(meta["resolution"])
        self.ox, self.oy = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
        self.img = np.array(Image.open(os.path.join(MERGED, meta["image"])))
        self.h, self.w = self.img.shape
        self.free = self.img >= 250
        self.occ = self.img <= 5
        self.known = self.img != UNKNOWN

    def to_px(self, x, y):
        """Continuous pixel coordinates (col, row) of stage points; pixel centres at +0.5."""
        return (np.asarray(x) - self.ox) / self.res, self.h - (np.asarray(y) - self.oy) / self.res

    def to_world(self, col, row):
        """Stage XY of pixel coordinates (continuous; pass idx + 0.5 for centres)."""
        return self.ox + np.asarray(col) * self.res, self.oy + (self.h - np.asarray(row)) * self.res

    def cell(self, x, y):
        c, r = self.to_px(x, y)
        c, r = np.floor(c).astype(int), np.floor(r).astype(int)
        ok = (c >= 0) & (c < self.w) & (r >= 0) & (r < self.h)
        return c, r, ok

    def rooms(self):
        with open(os.path.join(MERGED, "rooms_next.json")) as f:
            return json.load(f)

    def room_labels(self, rooms):
        """int raster: 0 = none, i+1 = rooms[i] (free pixels only)."""
        from PIL import Image, ImageDraw

        lab = np.zeros((self.h, self.w), np.int32)
        for i, room in enumerate(rooms):
            im = Image.new("L", (self.w, self.h), 0)
            d = ImageDraw.Draw(im)
            d.polygon([tuple(self.to_px(x, y)) for x, y in room["polygon"]], fill=1)
            for hole in room.get("holes", []):
                d.polygon([tuple(self.to_px(x, y)) for x, y in hole], fill=0)
            lab[(np.array(im) > 0) & self.free & (lab == 0)] = i + 1
        return lab


def load_glb_parts():
    import trimesh

    scene = trimesh.load(GLB, process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        mesh = scene.geometry[gname].copy()
        if not np.allclose(tf, np.eye(4)):
            mesh.apply_transform(tf)
        parts.append((node, mesh))
    return parts


def texel_colors(mesh, fidx, pts):
    """Base-colour texels at surface points; trimesh UVs have a lower-left origin."""
    import trimesh

    tri = mesh.triangles[fidx]
    bary = trimesh.triangles.points_to_barycentric(tri, pts)
    uv = np.einsum("ij,ijk->ik", bary, mesh.visual.uv[mesh.faces[fidx]])
    img = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
    h, w = img.shape[:2]
    col = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
    row = np.clip((1.0 - uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
    return img[row, col]


def surface_samples(force=False):
    import trimesh

    path = os.path.join(CACHE, "samples.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    xyz, nrm, rgb = [], [], []
    for _name, mesh in load_glb_parts():
        n = int(mesh.area * SAMPLES_PER_M2)
        pts, fidx = trimesh.sample.sample_surface(mesh, n, seed=7)
        xyz.append(pts.astype(np.float32))
        nrm.append(mesh.face_normals[fidx].astype(np.float32))
        rgb.append(texel_colors(mesh, fidx, pts).astype(np.uint8))
    out = dict(xyz=np.concatenate(xyz), nrm=np.concatenate(nrm), rgb=np.concatenate(rgb))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out


def plan_rasters(grid, s):
    """Per-pixel scan evidence on the floor-map grid.

    floor:   near-horizontal surface samples at -0.15..0.20 m (scanned floor, incl. drift patch)
    vert_n:  near-vertical samples (|nz| < 0.35) per pixel
    vert_top: highest near-vertical sample per pixel (m), 0 where none
    tall:    pixels with near-vertical samples above 0.9 m
    """
    xyz, nz = s["xyz"], np.abs(s["nrm"][:, 2])
    c, r, ok = grid.cell(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2]
    floor = np.zeros((grid.h, grid.w), bool)
    m = ok & (nz > 0.8) & (z > -0.15) & (z < 0.20)
    floor[r[m], c[m]] = True
    vm = ok & (nz < 0.35)
    vert_n = np.zeros((grid.h, grid.w), np.int32)
    np.add.at(vert_n, (r[vm], c[vm]), 1)
    vert_top = np.zeros((grid.h, grid.w), np.float32)
    np.maximum.at(vert_top, (r[vm], c[vm]), z[vm])
    tall = np.zeros((grid.h, grid.w), np.int32)
    tm = vm & (z > 0.9)
    np.add.at(tall, (r[tm], c[tm]), 1)
    return dict(floor=floor, vert_n=vert_n, vert_top=vert_top, tall=tall >= 2)
