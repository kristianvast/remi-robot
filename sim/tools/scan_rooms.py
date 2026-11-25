"""Extract walkable room and corridor areas from narrow-neck floor regions."""

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely import box, union_all
from shapely.geometry import Polygon
import trimesh

from scan_diagnostics import texture_samples
from scan_raster import Grid, disk, small_holes


def area_polygon(grid: Grid, mask: np.ndarray) -> Polygon:
    """Union horizontal pixel runs and simplify the largest connected polygon."""
    rr, cc = np.where(np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1))
    starts, ends, rows = cc[::2], cc[1::2], rr[::2]
    base = grid.metric(np.column_stack((rows + 1, starts)))
    spans = (ends - starts) * grid.resolution
    # Integer grid coordinates avoid gaps from platform-specific rounding.
    merged = union_all(box(base[:, 0], base[:, 1], base[:, 0] + spans,
                           base[:, 1] + grid.resolution), grid_size=.001)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda poly: poly.area)
    return merged.simplify(.055, preserve_topology=True)


def room_seeds(free: np.ndarray, occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close measurement pinholes for segmentation, then cut narrow doorways."""
    walk = small_holes(ndimage.binary_closing(free, structure=disk(4)), 420) & (
        ~occupied | ~ndimage.binary_opening(occupied, structure=disk(2)))
    seeds = np.zeros(free.shape, np.int32)
    # Wide cores identify rooms; a smaller clearance finds remaining corridors.
    distance = ndimage.distance_transform_edt(walk)
    index = 0
    for threshold, minimum, exclusion in ((27, 550, 0), (11, 425, 57)):
        available = distance >= threshold
        if index:
            available &= ndimage.distance_transform_edt(seeds == 0) > exclusion
        found, _ = ndimage.label(available)
        sizes = np.bincount(found.ravel())
        for label in np.flatnonzero(sizes[1:] >= minimum) + 1:
            index += 1
            seeds[found == label] = index
    return walk, seeds


def write_rooms(out: Path, grid: Grid, free: np.ndarray, occupied: np.ndarray,
                meshes: tuple[trimesh.Trimesh, trimesh.Trimesh]) -> None:
    """Segment room cores, grow across free areas, and infer conservative colour labels."""
    walk, seeds = room_seeds(free, occupied)
    _, nearest = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    labels = seeds[tuple(nearest)] * walk
    colors = []
    for mesh in meshes:
        pts = mesh.triangles_center
        ids = np.flatnonzero((abs(mesh.face_normals[:, 2]) > .95) &
                             (pts[:, 2] > -.09) & (pts[:, 2] < .12))[::5]
        loc = grid.index(pts[ids, :2])
        colors.append((labels[loc], texture_samples(mesh, ids).astype(float)))
    records = []
    for label in range(1, seeds.max() + 1):
        mask = labels == label
        groups, _ = ndimage.label(mask)
        connected = np.bincount(groups[seeds == label]).argmax()
        mask = groups == connected
        poly = area_polygon(grid, mask)
        if poly.area < 1.0:
            continue
        rgb = np.concatenate([c[ids == label] for ids, c in colors])
        color = np.median(rgb, axis=0) if len(rgb) else np.array([0., 0., 0.])
        red, green, blue = color
        name = "Floor area"
        if red > 1.55 * green and red > 1.2 * blue and blue > .35 * red:
            name = "Red-carpet area"
        if green > 1.12 * red and green > .93 * blue:
            name = "Green-floor area"
        if red > 1.1 * green and green > 1.15 * blue and name == "Floor area":
            name = "Warm-brown floor area"
        center = poly.centroid if poly.covers(poly.centroid) else poly.representative_point()
        records.append({
            "id": f"area_{len(records) + 1:02}", "name": name,
            "center": {"x": round(center.x, 3), "y": round(center.y, 3)},
            "area_m2": round(poly.area, 3),
            "polygon": np.asarray(poly.exterior.coords).round(3).tolist(),
            "holes": [np.asarray(ring.coords).round(3).tolist() for ring in poly.interiors],
        })
    (out / "rooms.json").write_text(json.dumps(records, indent=2) + "\n", encoding="ascii")
    print("Segmented area ids, names, area m2:", [(r["id"], r["name"], r["area_m2"])
                                                 for r in records], flush=True)
