"""Partial-overlap registration for the November 24 Scaniverse pair."""

from pathlib import Path
from typing import Final

import numpy as np
from scipy import ndimage, optimize, signal
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

ROOT: Final = Path(__file__).resolve().parents[2]
INPUT: Final = ROOT / "scans/2026-11-24"
OUTPUT: Final = ROOT / "scans/merged"
Z_UP: Final = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                        [0., 1., 0., 0.], [0., 0., 0., 1.]])


def load_pair() -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Apply the GLB scene transforms and convert to the simulator axes."""
    meshes = []
    for stamp in ("102526", "103223"):
        scene = trimesh.load_scene(INPUT / f"Scaniverse 2026-11-24 {stamp}.glb",
                                   process=False)
        mesh, = scene.dump()
        mesh.apply_transform(Z_UP)
        meshes.append(mesh)
    return meshes[0], meshes[1]


def wall_samples(mesh: trimesh.Trimesh, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample nearly vertical surfaces by area in the common wall height band."""
    center = mesh.triangles_center
    mask = (abs(mesh.face_normals[:, 2]) < .20) & (center[:, 2] > .24) & (center[:, 2] < 1.11)
    candidates = np.flatnonzero(mask)
    weights = mesh.area_faces[candidates]
    rng = np.random.default_rng(seed)
    ids = rng.choice(candidates, 90000, p=weights / weights.sum())
    uv = rng.random((len(ids), 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    points = np.einsum("ijk,ij->ik", mesh.triangles[ids],
                       np.column_stack((1 - uv.sum(axis=1), uv)))
    return points, mesh.face_normals[ids]


def coarse(target: tuple[np.ndarray, np.ndarray], source: tuple[np.ndarray, np.ndarray],
           red: tuple[np.ndarray, np.ndarray]) -> list[np.ndarray]:
    """Cross-correlate directed wall rasters and red-floor evidence over all yaw."""
    step = .10
    primary, pa = target
    secondary, pb = source
    target = np.rint(primary[:, :2] / step).astype(int)
    offset = target.min(axis=0)
    target -= offset
    shape = target.max(axis=0) + 1
    channels = (np.ones(len(pa)), np.cos(2 * pa), np.sin(2 * pa),
                np.cos(4 * pa), np.sin(4 * pa))
    grids = []
    for channel in channels:
        grid = np.zeros(shape, np.float32)
        np.add.at(grid, tuple(target.T), channel)
        np.divide(grid, np.maximum(1, np.bincount(np.ravel_multi_index(target.T, shape),
                  minlength=np.prod(shape)).reshape(shape)), out=grid)
        grids.append(ndimage.gaussian_filter(grid, 1.1))
    red_map = np.zeros(shape, np.float32)
    red_idx = np.rint(red[0] / step).astype(int) - offset
    red_idx = red_idx[((red_idx >= 0) & (red_idx < shape)).all(axis=1)]
    red_map[tuple(red_idx.T)] = 1
    red_map = ndimage.gaussian_filter(red_map, 1.4)
    best = []
    for yaw in np.deg2rad(np.arange(-180, 180, 1)):
        rotation = Rotation.from_rotvec([0, 0, yaw])
        xy = rotation.apply(secondary)[:, :2]
        red_xy = rotation.apply(np.column_stack((red[1], np.zeros(len(red[1])))))[:, :2]
        pix = np.rint(np.vstack((xy, red_xy)) / step).astype(int)
        start = pix.min(axis=0)
        pix -= start
        patch_shape = pix.max(axis=0) + 1
        wall_idx, floor_idx = np.split(pix, [len(xy)])
        count = np.bincount(np.ravel_multi_index(wall_idx.T, patch_shape),
                            minlength=np.prod(patch_shape)).reshape(patch_shape)
        angles = pb + yaw
        channels = (np.ones(len(pb)), np.cos(2 * angles), np.sin(2 * angles),
                    np.cos(4 * angles), np.sin(4 * angles))
        corr = 0.
        for grid, channel in zip(grids, channels, strict=True):
            patch = np.zeros(patch_shape, np.float32)
            np.add.at(patch, tuple(wall_idx.T), channel)
            patch /= np.maximum(1, count)
            corr = corr + signal.fftconvolve(grid, patch[::-1, ::-1])
        corr /= 3 * np.count_nonzero(count)
        floor = np.zeros(patch_shape, np.float32)
        floor[tuple(floor_idx.T)] = 1
        corr += .18 * signal.fftconvolve(red_map, floor[::-1, ::-1]) / floor.sum()
        for _ in range(4):
            row = np.array(np.unravel_index(corr.argmax(), corr.shape))
            trans = (row - patch_shape + 1 + offset - start) * step
            best.append((float(corr[tuple(row)]), yaw, *trans))
            x, y = row
            corr[max(0, x - 8):x + 9, max(0, y - 8):y + 9] = 0
    best.sort(reverse=True)
    selected = []
    for score, yaw, x, y in best:
        if all(abs(np.angle(np.exp(1j * (yaw - old[1])))) > .11
               or np.hypot(x - old[2], y - old[3]) > .8 for old in selected):
            selected.append((score, yaw, x, y))
        if len(selected) == 12:
            break
    print("Top wall correlations (score, yaw deg, tx, ty):",
          [(round(s, 4), round(np.rad2deg(y), 1), x, z) for s, y, x, z in selected],
          flush=True)
    return [np.array(item[1:]) for item in selected]


def refine(target: tuple[np.ndarray, np.ndarray], source: tuple[np.ndarray, np.ndarray],
           initial: np.ndarray) -> np.ndarray:
    """Robust yaw/XY point-to-plane ICP, rejecting mismatched surface orientations."""
    a, na = target
    b, nb = source
    b, nb = b[::5], nb[::5]
    tree = cKDTree(a)
    pose = initial
    for radius in [.30] * 5 + [.19] * 5 + [.13] * 8:
        yaw, x, y = pose
        rotation = Rotation.from_rotvec([0, 0, yaw])
        moved = rotation.apply(b) + [x, y, 0]
        dist, ids = tree.query(moved, workers=6)
        good = (dist < radius) & (abs(np.sum(rotation.apply(nb) * na[ids], axis=1)) > .82)
        ref, norm = a[ids[good]], na[ids[good]]

        def error(param: np.ndarray) -> np.ndarray:
            angle, tx, ty = param
            pos = Rotation.from_rotvec([0, 0, angle]).apply(b[good]) + [tx, ty, 0]
            return np.sum((pos - ref) * norm, axis=1)

        pose = optimize.least_squares(error, pose, loss="soft_l1", f_scale=.025).x
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec([0, 0, pose[0]]).as_matrix()
    matrix[:2, 3] = pose[1:]
    print("Refined yaw (deg), x, y:", np.r_[np.rad2deg(pose[0]), pose[1:]], flush=True)
    return matrix


def register(primary: trimesh.Trimesh, secondary: trimesh.Trimesh) -> np.ndarray:
    """Register the scan pair horizontally; floor diagnostics handle vertical alignment."""
    from scan_diagnostics import texture_samples

    a, b = wall_samples(primary, 16), wall_samples(secondary, 24)
    red = []
    for mesh in (primary, secondary):
        points = mesh.triangles_center
        indices = np.flatnonzero((abs(mesh.face_normals[:, 2]) > .93)
                                 & (points[:, 2] > -.02) & (points[:, 2] < .20))
        color = texture_samples(mesh, indices).astype(float)
        good = (color[:, 0] > 70) & (color[:, 0] > 1.65 * color[:, 1]) & (
            color[:, 2] > .38 * color[:, 0])
        red.append(points[indices[good], :2])
    candidates = coarse((a[0][::3], np.arctan2(a[1][::3, 1], a[1][::3, 0])),
                        (b[0][::3], np.arctan2(b[1][::3, 1], b[1][::3, 0])),
                        (red[0], red[1]))
    tree = cKDTree(a[0])
    ranked = []
    for initial in candidates[:8]:
        transform = refine(a, b, initial)
        moved = trimesh.transform_points(b[0][1::4], transform)
        nd = b[1][1::4] @ transform[:3, :3].T
        d, ii = tree.query(moved, workers=6)
        aligned = abs(np.sum(nd * a[1][ii], axis=1)) > .84
        score = np.mean(aligned * np.exp(-(d / .09) ** 2))
        print("Candidate held-aside wall score:", score, flush=True)
        ranked.append((score, transform))
    return max(ranked, key=lambda item: item[0])[1]


if __name__ == "__main__":
    a, b = load_pair()
    print(register(a, b))
