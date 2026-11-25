"""Blender batch worker: quadric decimation of geometry-only OBJ."""

from pathlib import Path
import sys

import bpy


def decimate(source: Path, dest: Path) -> None:
    """Preserve the input Z-up coordinates on both sides of the OBJ exchange."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=str(source), forward_axis="NEGATIVE_Y", up_axis="Z")
    mesh, = bpy.context.selected_objects
    bpy.context.view_layer.objects.active = mesh
    modifier = mesh.modifiers.new(name="Scan physics QEM", type="DECIMATE")
    modifier.ratio = min(1.0, 132000 / len(mesh.data.polygons))
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.ops.wm.obj_export(filepath=str(dest), forward_axis="NEGATIVE_Y", up_axis="Z",
                          export_materials=False, export_uv=False, export_normals=False,
                          export_triangulated_mesh=True, export_selected_objects=True)


if __name__ == "__main__":
    src, dst = sys.argv[sys.argv.index("--") + 1:]
    decimate(Path(src), Path(dst))
