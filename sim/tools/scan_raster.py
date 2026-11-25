"""Metric projection of Scaniverse triangles into north-up (positive Y) rasters."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import trimesh


@dataclass(frozen=True, slots=True)
class Grid:
    """Lower-left outer corner and north-up image dimensions."""

    origin: np.ndarray
    size: tuple[int, int]
    resolution: float = .02

    @classmethod
    def around(cls, bounds: np.ndarray) -> "Grid":
        res = .02
        low = np.floor(bounds[0, :2] / res).astype(int) - 12
        high = np.ceil(bounds[1, :2] / res).astype(int) + 12
        return cls(low * res, (int(high[0] - low[0]), int(high[1] - low[1])), res)

    def continuous(self, xy: np.ndarray) -> np.ndarray:
        pixels = (xy - self.origin) / self.resolution
        return np.stack((pixels[..., 0], self.size[1] - pixels[..., 1]), axis=-1)

    def index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, r = np.floor(self.continuous(xy)).astype(int).T
        return r.clip(0, self.size[1] - 1), c.clip(0, self.size[0] - 1)

    def metric(self, rc: np.ndarray) -> np.ndarray:
        r, c = rc.T
        return self.origin + self.resolution * np.column_stack((c, self.size[1] - r))


def clip_height(tri: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip a triangle against the two horizontal slab planes."""
    vertices = tri
    for level, sign in ((low, 1), (high, -1)):
        clipped = []
        for a, b in zip(np.roll(vertices, 1, axis=0), vertices, strict=True):
            inside_a = sign * (a[2] - level) >= 0
            inside_b = sign * (b[2] - level) >= 0
            if inside_a != inside_b:
                clipped.append(a + (b - a) * ((level - a[2]) / (b[2] - a[2])))
            if inside_b:
                clipped.append(b)
        vertices = np.array(clipped)
    return vertices


def raster(grid: Grid, triangles: np.ndarray,
           slab: tuple[float, float] | None = None) -> np.ndarray:
    """Rasterize projected faces; only geometry inside an optional slab counts."""
    image = Image.new("1", grid.size)
    draw = ImageDraw.Draw(image)
    if slab is not None:
        low, high = slab
        mn, mx = triangles[:, :, 2].min(axis=1), triangles[:, :, 2].max(axis=1)
        crossing = triangles[(mn <= high) & (mx >= low) & ((mn < low) | (mx > high))]
        triangles = triangles[(mn >= low) & (mx <= high)]
        for tri in crossing:
            poly = clip_height(tri, low, high)
            if len(poly) >= 3:
                xy = np.floor(grid.continuous(poly[:, :2])).astype(int).ravel()
                draw.polygon(xy.tolist(), fill=1)
    projected = np.floor(grid.continuous(triangles[:, :, :2])).astype(int).reshape(-1, 6)
    for poly in projected.tolist():
        draw.polygon(poly, fill=1)
    return np.array(image, dtype=bool)


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def small_holes(mask: np.ndarray, max_pixels: int) -> np.ndarray:
    """Fill enclosed islands no larger than a specified pixel count."""
    holes = ndimage.binary_fill_holes(mask) & ~mask
    labels, _ = ndimage.label(holes)
    counts = np.bincount(labels.ravel())
    counts[0] = max_pixels + 1
    return mask | (counts[labels] <= max_pixels)


# Near-horizontal faces this close to the datum are floor. The top is 12 cm, not 3 cm, because scan
# drift lifts whole rooms by up to 9 cm with no step at the edge; a real low object still shows
# through its vertical sides.
FLOOR_BAND = (-.13, .12)


def floor_faces(mesh: trimesh.Trimesh, limits: tuple[float, float]) -> np.ndarray:
    """Mask of nearly horizontal near-ground faces (in both winding directions)."""
    z = mesh.triangles_center[:, 2]
    low, high = limits
    return (abs(mesh.face_normals[:, 2]) > .91) & (z >= low) & (z <= high)


def floor_triangles(mesh: trimesh.Trimesh, limits: tuple[float, float]) -> np.ndarray:
    return mesh.triangles[floor_faces(mesh, limits)]


def floor_mask(grid: Grid, mesh: trimesh.Trimesh, limits: tuple[float, float]) -> np.ndarray:
    return raster(grid, floor_triangles(mesh, limits))


def footprint(grid: Grid, mesh: trimesh.Trimesh) -> np.ndarray:
    """Plan ownership from below 65 cm, not ceilings or projecting high furniture."""
    low = raster(grid, mesh.triangles, (-.32, .65))
    return small_holes(ndimage.binary_closing(low, structure=disk(6)), 1000)
