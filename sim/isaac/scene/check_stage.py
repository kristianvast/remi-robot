"""Load building.usda with usd-core (no Isaac Sim) and report what Isaac Sim will see.

  sim/.venv/bin/python sim/isaac/scene/check_stage.py
"""

import collections
import os
import sys

from pxr import Usd, UsdGeom, UsdPhysics

STAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "building.usda")


def summary(stage, label):
    counts = collections.Counter(p.GetTypeName() or "(untyped)" for p in stage.Traverse())
    faces = sum(len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or [])
                for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath("/World")).ComputeAlignedRange()
    print("[%s] prims by type: %s" % (label, dict(sorted(counts.items()))))
    print("[%s] mesh faces total: %d" % (label, faces))
    print("[%s] world bounds min %s max %s" % (label, tuple(round(v, 3) for v in rng.GetMin()),
                                               tuple(round(v, 3) for v in rng.GetMax())))


def main():
    stage = Usd.Stage.Open(STAGE)
    errors = []
    print("stage", STAGE)
    print("layers:", [os.path.relpath(l.realPath, os.path.dirname(STAGE)) for l in stage.GetUsedLayers() if l.realPath])
    print("metersPerUnit", UsdGeom.GetStageMetersPerUnit(stage), "upAxis", UsdGeom.GetStageUpAxis(stage),
          "kilogramsPerUnit", UsdPhysics.GetStageKilogramsPerUnit(stage), "defaultPrim", stage.GetDefaultPrim().GetPath())
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        errors.append("stage is not metres / Z-up")
    for part in ("/World/Scan", "/World/Collision", "/World/Shell"):
        prim = stage.GetPrimAtPath(part)
        kids = [c.GetName() for c in prim.GetChildren()] if prim else None
        print("part", part, "valid" if prim else "MISSING", "children:", kids)
        if not prim:
            errors.append("missing " + part)

    collider_ok = False
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        path = str(prim.GetPath())
        if path.startswith("/World/Collision"):
            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            vis = UsdGeom.Imageable(prim).ComputeVisibility()
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            rigid = any(a.HasAPI(UsdPhysics.RigidBodyAPI) for a in [prim, prim.GetParent()])
            print("collider %s: CollisionAPI=%s enabled=%s approximation=%s visibility=%s purpose=%s rigid_body=%s" % (
                path, has_col, enabled, approx, vis, UsdGeom.Imageable(prim).ComputePurpose(), rigid))
            collider_ok = has_col and enabled and vis == UsdGeom.Tokens.invisible and not rigid
        elif has_col:
            errors.append("unexpected CollisionAPI on " + path)
    scan_meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith("/World/Scan")]
    for p in scan_meshes:
        print("visual %s: CollisionAPI=%s faces=%d material=%s" % (
            p.GetPath(), p.HasAPI(UsdPhysics.CollisionAPI), len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get()),
            [str(t) for t in p.GetRelationship("material:binding").GetTargets()]))
    if not collider_ok:
        errors.append("collider is missing CollisionAPI, is visible, or is a rigid body")
    shell = stage.GetPrimAtPath("/World/Shell")
    print("shell variant:", shell.GetVariantSets().GetVariantSet("shell").GetVariantSelection(),
          "ceilingHeight:", shell.GetAttribute("elegoo:ceilingHeight").Get())
    for scope in ("Walls", "ClosureWalls", "Ceilings"):
        kids = stage.GetPrimAtPath("/World/Shell/" + scope).GetChildren()
        print("  /World/Shell/%s: %d meshes" % (scope, len(kids)))
    summary(stage, "shell on")
    shell.GetVariantSets().GetVariantSet("shell").SetVariantSelection("off")
    summary(stage, "shell off")
    print("shell off -> /World/Shell children:", [c.GetName() for c in shell.GetChildren()])
    print("RESULT", "OK" if not errors else "FAIL " + "; ".join(errors))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
