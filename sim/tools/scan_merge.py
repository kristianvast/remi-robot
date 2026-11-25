"""Generate the scan merge and simulator exports. Execute with sim/.venv/bin/python."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Final

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
import trimesh

from scan_diagnostics import level_pair, texture_ground, wall_residual
from scan_maps import write_maps
from scan_raster import FLOOR_BAND, Grid, floor_faces, floor_mask, footprint
from scan_register import OUTPUT, Z_UP, load_pair, register

BLENDER: Final = Path("/opt/homebrew/bin/blender")


def keep_additions(grid: Grid, a: trimesh.Trimesh, b: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep whole secondary faces outside primary low-height plan ownership."""
    primary = footprint(grid, a)
    clearance = ndimage.distance_transform_edt(~primary) * grid.resolution
    triangles = b.triangles
    # Also check the centroid and edge midpoints to catch overhanging triangles.
    positions = np.concatenate((triangles, (triangles + np.roll(triangles, 1, axis=1)) / 2,
                                b.triangles_center[:, None]), axis=1)
    ij = grid.index(positions[:, :, :2].reshape(-1, 2))
    keep = (clearance[ij].reshape(len(triangles), -1) > .025).all(axis=1)
    added = b.copy()
    added.update_faces(keep)
    added.remove_unreferenced_vertices()
    print("Source and retained secondary faces:", len(b.faces), len(added.faces), flush=True)
    a_floor, b_floor, kept = (floor_mask(grid, m, (-.11, .16)) for m in (a, b, added))
    pixel_area = grid.resolution ** 2
    print("Ground area m2: primary, secondary, overlap, secondary-only, retained new:",
          [round(float(mask.sum() * pixel_area), 3) for mask in
           (a_floor, b_floor, a_floor & b_floor, b_floor & ~a_floor, kept & ~a_floor)],
          flush=True)
    print("Retained floor overlap pixels:", (kept & a_floor).sum(), flush=True)
    return added


def write_collision(meshes: tuple[trimesh.Trimesh, trimesh.Trimesh], output: Path) -> None:
    """Weld UV splits, replace near-datum ground, and add a continuous floor box."""
    mesh = trimesh.util.concatenate([
        trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False) for m in meshes])
    mesh.merge_vertices(digits_vertex=5)
    lo, hi = mesh.bounds[:, :2]
    level = floor_faces(mesh, FLOOR_BAND)
    mesh.update_faces(~level & mesh.nondegenerate_faces() & mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    with TemporaryDirectory(dir=output, prefix="scan_collision_") as tmp:
        source, dest = Path(tmp) / "source.obj", Path(tmp) / "reduced.obj"
        mesh.export(source, include_color=False, include_texture=False,
                    include_normals=False)
        proc = subprocess.run([str(BLENDER), "--background", "--factory-startup",
                               "--disable-autoexec", "--python-exit-code", "2", "--python",
                               str(Path(__file__).with_name("scan_collision.py")), "--",
                               str(source), str(dest)], cwd=tmp, check=True, capture_output=True,
                              text=True)
        print(proc.stdout, flush=True)
        reduced = trimesh.load_mesh(dest, process=False)
    assert np.max(np.abs(reduced.bounds - mesh.bounds)) < .08, (mesh.bounds, reduced.bounds)
    floor = trimesh.creation.box(extents=(*((hi - lo) + 1), .04))
    floor.apply_translation([*((lo + hi) / 2), -.02])
    collision = trimesh.util.concatenate([reduced, floor])
    assert len(collision.faces) < 150000, len(collision.faces)
    collision.export(output / "collision.obj", include_color=False, include_texture=False,
                     include_normals=False)
    print("Collision triangles including 12-face floor slab:", len(collision.faces), flush=True)


def main() -> None:
    """Register, measure, merge, and verify the serialized model."""
    a, b = load_pair()
    pose = register(a, b)
    b.apply_transform(pose)
    datum, plane = level_pair(a, b)
    height_pose = np.eye(4)
    height_pose[:3, :3] = Rotation.from_rotvec([plane[1], -plane[0], 0]).as_matrix()
    height_pose[2, 3] = plane[2] - datum
    a.apply_translation([0, 0, -datum])
    b.apply_transform(height_pose)
    print("Secondary original Y-up -> world (row-major):\n", height_pose @ pose @ Z_UP, flush=True)
    print("Primary original Y-up -> world (row-major):\n",
          trimesh.transformations.translation_matrix([0, 0, -datum]) @ Z_UP, flush=True)
    grid = Grid.around(np.array([np.minimum(a.bounds[0], b.bounds[0]),
                                 np.maximum(a.bounds[1], b.bounds[1])]))
    wall_residual(grid, a, b)
    OUTPUT.mkdir(exist_ok=True)
    additions = keep_additions(grid, a, b)
    scene = trimesh.Scene()
    for mesh, label in ((a, "primary_102526"), (additions, "addition_103223")):
        scene.add_geometry(mesh, node_name=label, geom_name=label)
    model = OUTPUT / "building.glb"
    scene.export(model)
    loaded = trimesh.load_scene(model, process=False)
    print("Reload building.glb bounds, face count:", loaded.bounds.tolist(),
          sum(len(m.faces) for m in loaded.geometry.values()), flush=True)
    assert len(loaded.geometry) == 2
    for mesh in loaded.geometry.values():
        assert mesh.visual.material.baseColorTexture.size == (8192, 8192)
        assert mesh.visual.uv.shape == (len(mesh.vertices), 2)
    write_maps(OUTPUT, grid, a, b, additions)
    write_collision((a, additions), OUTPUT)
    print("SCAN_MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
