"""Independent registration measurements and texture-ground inspection."""

import numpy as np
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import trimesh

from scan_raster import Grid, disk, floor_mask, footprint
from scan_register import wall_samples


def level_pair(a: trimesh.Trimesh, b: trimesh.Trimesh) -> tuple[float, np.ndarray]:
    """Compare matching flat ground cells; fit a robust delta-height plane."""
    planes = []
    for mesh in (a, b):
        tri, n = mesh.triangles_center, mesh.face_normals
        keep = (abs(n[:, 2]) > .97) & (tri[:, 2] > -.01) & (tri[:, 2] < .095)
        pts = tri[keep]
        index = np.floor(pts[:, :2] / .12).astype(int) + 200
        keys = index[:, 0] * 400 + index[:, 1]
        count = np.bincount(keys, minlength=160000)
        accum = np.bincount(keys, weights=pts[:, 2], minlength=160000)
        planes.append((count, accum))
    (ca, sa), (cb, sb) = planes
    ids = np.flatnonzero((ca >= 3) & (cb >= 3))
    xy = (np.column_stack(np.divmod(ids, 400)) - 199.5) * .12
    dz = sa[ids] / ca[ids] - sb[ids] / cb[ids]
    design = np.column_stack((xy, np.ones(len(xy))))
    plane = optimize.least_squares(lambda h: design @ h - dz, np.zeros(3),
                                  loss="soft_l1", f_scale=.015).x
    height = float(np.median(sa[ids] / ca[ids]))
    height_only = np.median(dz)
    print("Matching ground cells, reference median, dz percentiles, dz slopes/offset:",
          len(ids), height, np.percentile(dz, [5, 50, 95]), plane, flush=True)
    print("Ground delta MAD (offset only, tilt+offset):",
          np.median(abs(dz - height_only)), np.median(abs(dz - design @ plane)),
          flush=True)
    # Tiny fitted tilt with no material change to median absolute residual.
    if np.linalg.norm(plane[:2]) < np.tan(np.deg2rad(.10)):
        plane[:] = [0, 0, height_only]
    return height, plane


def wall_residual(grid: Grid, a: trimesh.Trimesh, b: trimesh.Trimesh) -> None:
    """Measure held-out symmetric nearest parallel wall samples in overlap."""
    pa, na = wall_samples(a, 450)
    pb, nb = wall_samples(b, 920)
    fpa, fpb = footprint(grid, a), footprint(grid, b)
    roi = ndimage.binary_dilation(fpa, structure=disk(15)) & ndimage.binary_dilation(
        fpb, structure=disk(15))
    results = []
    for src, sn, dst, dn in ((pa, na, pb, nb), (pb, nb, pa, na)):
        valid = roi[grid.index(src[:, :2])]
        src, sn = src[valid][::3], sn[valid][::3]
        distance, ids = cKDTree(dst).query(src, k=32, workers=6)
        parallel = abs(np.einsum("ijk,ik->ij", dn[ids], sn)) >= .85
        normal = abs(np.einsum("ijk,ijk->ij", src[:, None] - dst[ids], dn[ids]))
        normal[~parallel] = np.inf
        # First orientation-compatible nearest sample, then point-to-plane distance.
        col = np.argmax(parallel, axis=1)
        row = np.arange(len(col))
        euclidean, perpendicular = distance[row, col], normal[row, col]
        matched = (euclidean <= .25) & np.isfinite(perpendicular)
        print("Overlap wall samples eligible/matched; matched Euclidean and perpendicular cm:",
              len(src), matched.sum(), np.percentile(euclidean[matched] * 100, [50, 95]),
              np.percentile(perpendicular[matched] * 100, [50, 95]), flush=True)
        results.append(np.column_stack((euclidean[matched], perpendicular[matched])))
    print("Pooled wall residual cm (rows median/p95; cols Euclidean/perpendicular):",
          np.percentile(np.concatenate(results) * 100, [50, 95], axis=0), flush=True)


def texture_samples(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    """Sample base-color texels at triangle UV centroids."""
    texture = mesh.visual.material.baseColorTexture
    uv = mesh.visual.uv[mesh.faces[faces]].mean(axis=1) % 1
    dims = np.asarray(texture.size)
    px = np.minimum((uv * dims).astype(int), dims - 1)
    bitmap = np.asarray(texture.convert("RGB"))
    return bitmap[dims[1] - 1 - px[:, 1], px[:, 0]]


def texture_ground(grid: Grid, mesh: trimesh.Trimesh) -> Image.Image:
    """Draw near-horizontal ground by texture, including suspicious elevated patches."""
    z = mesh.triangles_center[:, 2]
    selected = np.flatnonzero((abs(mesh.face_normals[:, 2]) > .88) & (z > -.10) & (z < .23))
    selected = selected[np.argsort(z[selected])]
    rgb = texture_samples(mesh, selected)
    polygons = np.floor(grid.continuous(mesh.triangles[selected, :, :2])).astype(int)
    pic = Image.new("RGB", grid.size, (244, 244, 244))
    draw = ImageDraw.Draw(pic)
    for polygon, color in zip(polygons.reshape(-1, 6).tolist(), rgb.tolist(), strict=True):
        draw.polygon(polygon, tuple(color))
    return pic


def inspect(grid: Grid, a: trimesh.Trimesh, b: trimesh.Trimesh) -> Image.Image:
    """Overlay the full scans' floor and outline evidence in the registered frame."""
    masks = [floor_mask(grid, m, (-.09, .13)) for m in (a, b)]
    image = np.full((*masks[0].shape, 3), 248, dtype=np.uint8)
    image[masks[0]] = [151, 200, 190]
    image[masks[1]] = [218, 173, 193]
    image[masks[0] & masks[1]] = [115, 148, 199]
    for mesh, rgb in ((a, (6, 107, 111)), (b, (169, 48, 89))):
        outline = footprint(grid, mesh)
        image[outline ^ ndimage.binary_erosion(outline, iterations=2)] = rgb
    return Image.fromarray(image)
