"""Build the Isaac Sim USD scene for the merged building scan.

One command, from the project root:

  sim/.venv/bin/python sim/isaac/scene/build_scene.py --ceiling 3.0

Writes, next to this file:
  building.usda             root stage (metres, +Z up), references the three layers below
  layers/scan_visual.usdc   textured scan, visual only        -> /World/Scan
  layers/collision.usdc     collision.obj, invisible collider -> /World/Collision
  layers/shell.usda         generated walls + ceilings        -> /World/Shell (variant set "shell": on|off)
  textures/*.png            the GLB's base-colour images, byte-identical
  scene_meta.yaml           ceiling height, transforms, bounds, spawn poses, goals, wall statistics
  cache/                    surface samples and a plan-view debug image of the shell

The scan and collision layers are reused when present (--force rebuilds them);
the shell layer, root stage and metadata are always regenerated.
"""

import argparse
import json
import math
import os
import struct
import sys

import cv2
import numpy as np
import shapely
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy import ndimage
from shapely.geometry import Polygon

import scene_analysis as sa

LAYERS = os.path.join(sa.HERE, "layers")
TEXTURES = os.path.join(sa.HERE, "textures")
ROOT_USD = os.path.join(sa.HERE, "building.usda")
SCAN_USD = os.path.join(LAYERS, "scan_visual.usdc")
COLLISION_USD = os.path.join(LAYERS, "collision.usdc")
SHELL_USD = os.path.join(LAYERS, "shell.usda")
META = os.path.join(sa.HERE, "scene_meta.yaml")

OFF_WHITE_SRGB = (236, 233, 226)
CEILING_SRGB = (238, 236, 231)
WALL_OFFSET_PX = 5
OUTLINE_SIMPLIFY_PX = 4.0
BACKED_RADIUS_PX = 10
BACKED_MIN_TOP_M = 0.30
PIECE_M = 0.25
MIN_ISLAND_PX = 5000
MAX_FILLED_HOLE_PX = 2500
INTERIOR_MIN_LEN_M = 0.8
INTERIOR_MAX_THICK_M = 0.30
INTERIOR_CLEAR_OF_OUTLINE_PX = 6
UPPER_WALL_MIN_Z = 0.9
UPPER_WALL_BAND_PX = 15
MIN_COLOUR_SAMPLES = 300
SPAWN_CLEARANCE_M = 0.20
N_SPAWNS = 240


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def srgb_to_linear(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def new_stage(path):
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage


def define_mesh(stage, path, points, counts, indices, st=None, normals=None, double_sided=True):
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = np.asarray(points, np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(indices, np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(double_sided)
    lo, hi = points.min(0), points.max(0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    if st is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(st, np.float32)))
    if normals is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    return mesh


def preview_material(stage, path, linear_rgb=None, texture=None, roughness=0.9):
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    diffuse = surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if texture is None:
        diffuse.Set(Gf.Vec3f(*map(float, linear_rgb)))
    else:
        reader = UsdShade.Shader.Define(stage, path + "/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/diffuseTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        diffuse.ConnectToSource(tex.ConnectableAPI(), tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3).GetBaseName())
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def glb_base_color_images():
    with open(sa.GLB, "rb") as f:
        data = f.read()
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_len])
    bin_off = 20 + json_len
    bin_len = struct.unpack_from("<I", data, bin_off)[0]
    blob = data[bin_off + 8:bin_off + 8 + bin_len]
    out = {}
    for node in gltf["nodes"]:
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        mat = gltf["materials"][prim["material"]]
        image = gltf["images"][gltf["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        out[node["name"]] = (blob[start:start + view["byteLength"]], image["mimeType"])
    return out


def write_scan_layer():
    os.makedirs(TEXTURES, exist_ok=True)
    images = glb_base_color_images()
    stage = new_stage(SCAN_USD)
    root = UsdGeom.Xform.Define(stage, "/Scan")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Scan/Looks")
    for name, mesh in sa.load_glb_parts():
        data, mime = images[name]
        tex_file = os.path.join(TEXTURES, name + (".png" if mime == "image/png" else ".jpg"))
        with open(tex_file, "wb") as f:
            f.write(data)
        usd_mesh = define_mesh(stage, "/Scan/" + name, mesh.vertices, np.full(len(mesh.faces), 3), mesh.faces,
                               st=mesh.visual.uv)
        mat = preview_material(stage, "/Scan/Looks/" + name, texture="../textures/" + os.path.basename(tex_file))
        bind(usd_mesh.GetPrim(), mat)
        print("SCAN %s: %d vertices, %d faces, texture %s (%.1f MB)" % (
            name, len(mesh.vertices), len(mesh.faces), os.path.basename(tex_file), len(data) / 1e6))
    stage.GetRootLayer().Save()


def write_collision_layer():
    import trimesh

    mesh = trimesh.load(sa.COLLISION_OBJ, process=False, force="mesh")
    stage = new_stage(COLLISION_USD)
    root = UsdGeom.Xform.Define(stage, "/Collision")
    stage.SetDefaultPrim(root.GetPrim())
    usd_mesh = define_mesh(stage, "/Collision/collision_mesh", mesh.vertices, np.full(len(mesh.faces), 3), mesh.faces)
    usd_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    usd_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    prim = usd_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
    stage.GetRootLayer().Save()
    print("COLLISION %d vertices, %d triangles, static triangle-mesh collider" % (len(mesh.vertices), len(mesh.faces)))


def building_region(grid, ras):
    """Scanned building footprint: known map cells plus scanned floor, specks closed,
    stray islands dropped, enclosed holes under ~1 m2 (furniture shadows) filled."""
    region = ndimage.binary_closing(grid.known | ras["floor"], structure=disk(3))
    lab, _ = ndimage.label(region)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= MIN_ISLAND_PX
    keep[0] = False
    region = keep[lab]
    holes = ndimage.binary_fill_holes(region) & ~region
    hl, _ = ndimage.label(holes)
    hs = np.bincount(hl.ravel())
    small = hs < MAX_FILLED_HOLE_PX
    small[0] = False
    return region | small[hl]


def nearest_room_labels(room_lab):
    _d, (ri, ci) = ndimage.distance_transform_edt(room_lab == 0, return_indices=True)
    return room_lab[ri, ci]


def outline_pieces(grid, region, ras, labels_all):
    """Wall pieces (<= PIECE_M) along the region outline, offset WALL_OFFSET_PX outward."""
    walls = ndimage.binary_dilation(region, structure=disk(WALL_OFFSET_PX))
    contours, _h = cv2.findContours(walls.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    top = np.where(ras["vert_n"] >= 2, ras["vert_top"], 0.0)
    top_near = ndimage.maximum_filter(top, footprint=disk(BACKED_RADIUS_PX))
    pieces = []
    for cnt in contours:
        poly = cv2.approxPolyDP(cnt, OUTLINE_SIMPLIFY_PX, True)[:, 0, :].astype(float) + 0.5
        if len(poly) < 3:
            continue
        xs, ys = grid.to_world(poly[:, 0], poly[:, 1])
        pts = np.stack([xs, ys], 1)
        for a, b in zip(pts, np.roll(pts, -1, 0)):
            length = float(np.linalg.norm(b - a))
            n = max(1, int(math.ceil(length / PIECE_M)))
            for k in range(n):
                p0, p1 = a + (b - a) * k / n, a + (b - a) * (k + 1) / n
                mid = (p0 + p1) / 2
                c, r, ok = grid.cell(mid[0], mid[1])
                if not ok:
                    continue
                pieces.append(dict(p0=p0, p1=p1, length=length / n, scan_top=float(top_near[r, c]),
                                   room=int(labels_all[r, c]), kind="outline"))
    for p in pieces:
        p["backed"] = p["scan_top"] >= BACKED_MIN_TOP_M
    return pieces


def interior_pieces(grid, region, ras, labels_all):
    """Straight walls along long, thin, tall (>0.9 m) scan surfaces inside the region,
    cut wherever they cross free floor so doorways stay open. They start just below the
    local scan wall top so the scanned lower wall stays visible in front of them."""
    inside = ndimage.distance_transform_edt(region) > INTERIOR_CLEAR_OF_OUTLINE_PX
    tall = ndimage.binary_closing(ras["tall"], structure=disk(2)) & inside
    top = np.where(ras["vert_n"] >= 2, ras["vert_top"], 0.0)
    top_near = ndimage.maximum_filter(top, footprint=disk(4))
    lab, n = ndimage.label(tall, structure=np.ones((3, 3)))
    pieces = []
    for i in range(1, n + 1):
        rr, cc = np.nonzero(lab == i)
        if len(rr) < 20:
            continue
        xs, ys = grid.to_world(cc + 0.5, rr + 0.5)
        pts = np.stack([xs, ys], 1)
        mu = pts.mean(0)
        _u, _s, vt = np.linalg.svd(pts - mu, full_matrices=False)
        along, across = (pts - mu) @ vt[0], (pts - mu) @ vt[1]
        lo, hi = np.percentile(along, [1, 99])
        thick = np.percentile(across, 95) - np.percentile(across, 5)
        if hi - lo < INTERIOR_MIN_LEN_M or thick > INTERIOR_MAX_THICK_M:
            continue
        ts = np.arange(lo, hi, grid.res)
        line = mu + ts[:, None] * vt[0]
        c, r, ok = grid.cell(line[:, 0], line[:, 1])
        keep = ok.copy()
        keep[ok] = ~grid.free[r[ok], c[ok]]
        top_along = top_near[r[ok], c[ok]]
        z0 = max(0.0, float(np.percentile(top_along, 20)) - 0.15)
        runs = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(int), [0]])))
        for s, e in zip(runs[::2], runs[1::2]):
            if (e - s) * grid.res < 0.3:
                continue
            p0, p1 = line[s], line[e - 1]
            m = (p0 + p1) / 2
            cm, rm, _ok = grid.cell(m[0], m[1])
            pieces.append(dict(p0=p0, p1=p1, length=float(np.linalg.norm(p1 - p0)), scan_top=z0 + 0.15,
                               z0=z0, room=int(labels_all[rm, cm]), kind="interior", backed=True))
    return pieces


def wall_quads(pieces, region, grid, ceiling):
    pts, counts, idx, normals = [], [], [], []
    for p in pieces:
        p0, p1 = p["p0"], p["p1"]
        d = p1 - p0
        nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        probe = (p0 + p1) / 2 + nrm * 0.1
        c, r, ok = grid.cell(probe[0], probe[1])
        if p["kind"] == "outline" and not (ok and region[r, c]):
            nrm = -nrm
        base = len(pts)
        z0 = p.get("z0", -0.02)
        pts += [(p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], ceiling + 0.02),
                (p0[0], p0[1], ceiling + 0.02)]
        counts.append(4)
        idx += [base, base + 1, base + 2, base + 3]
        normals.append((nrm[0], nrm[1], 0.0))
    return np.array(pts), counts, idx, normals


def room_ceiling_polygons(grid, region, labels_all, n_rooms):
    cover = ndimage.binary_dilation(region, structure=disk(WALL_OFFSET_PX + 2))
    out = {}
    for rid in range(1, n_rooms + 1):
        mask = (cover & (labels_all == rid)).astype(np.uint8)
        if mask.sum() == 0:
            continue
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for i, cnt in enumerate(contours):
            if hier[0][i][3] != -1 or len(cnt) < 3:
                continue
            holes = [contours[j] for j in range(len(contours)) if hier[0][j][3] == i and len(contours[j]) >= 3]

            def world(c):
                xs, ys = grid.to_world(c[:, 0, 0] + 0.5, c[:, 0, 1] + 0.5)
                return np.stack([xs, ys], 1)

            poly = Polygon(world(cnt), [world(h) for h in holes]).buffer(0)
            polys.append(poly)
        geom = shapely.union_all(polys).simplify(0.03).buffer(0.03, join_style="mitre")
        out[rid] = geom
    return out


def triangulate_ceiling(geom, z):
    tris = shapely.constrained_delaunay_triangles(geom)
    pts, idx = [], []
    for tri in shapely.get_parts(tris):
        c = np.asarray(tri.exterior.coords)[:3]
        area = (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - (c[2, 0] - c[0, 0]) * (c[1, 1] - c[0, 1])
        if abs(area) < 1e-10:
            continue
        if area > 0:
            c = c[::-1]
        base = len(pts)
        pts += [(x, y, z) for x, y in c]
        idx += [base, base + 1, base + 2]
    return np.array(pts), idx


def room_wall_colours(grid, region, labels_all, samples, rooms):
    xyz, nz, rgb = samples["xyz"], np.abs(samples["nrm"][:, 2]), samples["rgb"]
    near_outline = (ndimage.distance_transform_edt(region) <= UPPER_WALL_BAND_PX) | ~region
    c, r, ok = grid.cell(xyz[:, 0], xyz[:, 1])
    sel = ok & (nz < 0.35) & (xyz[:, 2] >= UPPER_WALL_MIN_Z)
    sel[sel] = near_outline[r[sel], c[sel]]
    lab = np.zeros(len(xyz), np.int32)
    lab[sel] = labels_all[r[sel], c[sel]]
    out = {}
    for i, room in enumerate(rooms):
        m = lab == i + 1
        n = int(m.sum())
        if n >= MIN_COLOUR_SAMPLES:
            col = tuple(int(v) for v in np.median(rgb[m].astype(float), axis=0))
            out[room["id"]] = dict(srgb=col, samples=n, source="scan upper-wall texels")
        else:
            out[room["id"]] = dict(srgb=OFF_WHITE_SRGB, samples=n, source="off-white default")
    return out


def free_pixels_under_walls(grid, pieces):
    canvas = np.zeros((grid.h, grid.w), np.uint8)
    for p in pieces:
        (c0, c1), (r0, r1) = grid.to_px([p["p0"][0], p["p1"][0]], [p["p0"][1], p["p1"][1]])
        cv2.line(canvas, (int(c0), int(r0)), (int(c1), int(r1)), 1, 1)
    return int((canvas.astype(bool) & grid.free).sum())


def write_shell_layer(ceiling, grid, rooms, region, labels_all, pieces, colours):
    stage = new_stage(SHELL_USD)
    shell = UsdGeom.Xform.Define(stage, "/Shell")
    stage.SetDefaultPrim(shell.GetPrim())
    shell.GetPrim().CreateAttribute("elegoo:ceilingHeight", Sdf.ValueTypeNames.Float).Set(float(ceiling))
    for scope in ("Looks", "Walls", "ClosureWalls", "Ceilings"):
        UsdGeom.Scope.Define(stage, "/Shell/" + scope)
    ceiling_mat = preview_material(stage, "/Shell/Looks/ceiling", srgb_to_linear(CEILING_SRGB), roughness=0.95)
    wall_mats = {}
    for room in rooms:
        wall_mats[room["id"]] = preview_material(stage, "/Shell/Looks/wall_" + room["id"],
                                                 srgb_to_linear(colours[room["id"]]["srgb"]))
    stats = dict(ceiling_m2={}, wall_faces=0, ceiling_triangles=0)
    for i, room in enumerate(rooms):
        rid = room["id"]
        for scope, prefix, pick in (("Walls", "shell_wall_", lambda p: p["backed"]),
                                    ("ClosureWalls", "shell_closure_", lambda p: not p["backed"])):
            sel = [p for p in pieces if p["room"] == i + 1 and pick(p)]
            if not sel:
                continue
            pts, counts, idx, nrm = wall_quads(sel, region, grid, ceiling)
            mesh = define_mesh(stage, "/Shell/%s/%s%s" % (scope, prefix, rid), pts, counts, idx, normals=nrm)
            bind(mesh.GetPrim(), wall_mats[rid])
            stats["wall_faces"] += len(counts)
    polys = room_ceiling_polygons(grid, region, labels_all, len(rooms))
    for i, room in enumerate(rooms):
        if i + 1 not in polys:
            continue
        pts, idx = triangulate_ceiling(polys[i + 1], ceiling)
        nrm = np.tile([0.0, 0.0, -1.0], (len(idx) // 3, 1))
        mesh = define_mesh(stage, "/Shell/Ceilings/shell_ceiling_" + room["id"], pts, np.full(len(idx) // 3, 3), idx,
                           normals=nrm)
        bind(mesh.GetPrim(), ceiling_mat)
        stats["ceiling_m2"][room["id"]] = round(float(polys[i + 1].area), 2)
        stats["ceiling_triangles"] += len(idx) // 3
    stage.GetRootLayer().Save()
    return stats


def write_root():
    stage = new_stage(ROOT_USD)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().documentation = "Merged building scan for Isaac Sim; generated by build_scene.py"
    UsdGeom.Xform.Define(stage, "/World/Scan").GetPrim().GetReferences().AddReference("./layers/scan_visual.usdc")
    UsdGeom.Xform.Define(stage, "/World/Collision").GetPrim().GetReferences().AddReference("./layers/collision.usdc")
    shell = UsdGeom.Xform.Define(stage, "/World/Shell").GetPrim()
    vset = shell.GetVariantSets().AddVariantSet("shell")
    vset.AddVariant("on")
    vset.AddVariant("off")
    vset.SetVariantSelection("on")
    with vset.GetVariantEditContext():
        shell.GetReferences().AddReference("./layers/shell.usda")
    vset.SetVariantSelection("on")
    stage.GetRootLayer().Save()


def farthest_points(pts, k, rng):
    if len(pts) <= k:
        return np.arange(len(pts))
    chosen = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return np.array(chosen)


def spawn_and_goals(grid, rooms, room_lab):
    rng = np.random.default_rng(20261124)
    clear_occ = ndimage.distance_transform_edt(~grid.occ) * grid.res
    clear_any = ndimage.distance_transform_edt(grid.free) * grid.res
    ok = grid.free & (clear_occ >= SPAWN_CLEARANCE_M) & (clear_any >= SPAWN_CLEARANCE_M)
    eligible = {room["id"]: np.argwhere(ok & (room_lab == i + 1)) for i, room in enumerate(rooms)}
    total = sum(len(v) for v in eligible.values())
    spawns, goals = [], []
    for room in rooms:
        cells = eligible[room["id"]]
        if len(cells) == 0:
            continue
        n_sp = max(6, int(round(N_SPAWNS * len(cells) / total)))
        xy = np.stack(grid.to_world(cells[:, 1] + 0.5, cells[:, 0] + 0.5), 1)
        for j in farthest_points(xy, n_sp, rng):
            r, c = cells[j]
            spawns.append(dict(area=room["id"], x=round(float(xy[j, 0]), 3), y=round(float(xy[j, 1]), 3),
                               yaw=round(float(rng.uniform(-math.pi, math.pi)), 4),
                               clearance_m=round(float(min(clear_occ[r, c], clear_any[r, c])), 3)))
        n_goal = int(np.clip(round(room["area_m2"] / 8.0), 3, 6))
        deep = cells[clear_any[cells[:, 0], cells[:, 1]] >= 0.30]
        src = deep if len(deep) >= n_goal else cells
        gxy = np.stack(grid.to_world(src[:, 1] + 0.5, src[:, 0] + 0.5), 1)
        for k, j in enumerate(farthest_points(gxy, n_goal, rng)):
            goals.append(dict(name="%s_goal_%d" % (room["id"], k + 1), area=room["id"],
                              x=round(float(gxy[j, 0]), 3), y=round(float(gxy[j, 1]), 3)))
    return spawns, goals


def stage_bounds(variant):
    stage = Usd.Stage.Open(ROOT_USD)
    stage.GetPrimAtPath("/World/Shell").GetVariantSets().GetVariantSet("shell").SetVariantSelection(variant)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath("/World")).ComputeAlignedRange()
    return [round(float(v), 3) for v in box.GetMin()], [round(float(v), 3) for v in box.GetMax()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ceiling", type=float, default=3.0, help="ceiling height above the floor datum, metres")
    ap.add_argument("--force", action="store_true", help="rebuild the scan and collision layers and the sample cache")
    args = ap.parse_args()
    if args.ceiling <= 2.46:
        sys.exit("--ceiling must be above 2.46 m: the scan's own walls reach 2.45 m")
    os.makedirs(LAYERS, exist_ok=True)
    if args.force or not os.path.exists(SCAN_USD):
        write_scan_layer()
    if args.force or not os.path.exists(COLLISION_USD):
        write_collision_layer()

    grid = sa.Grid()
    rooms = grid.rooms()
    samples = sa.surface_samples(force=args.force)
    ras = sa.plan_rasters(grid, samples)
    region = building_region(grid, ras)
    room_lab = grid.room_labels(rooms)
    labels_all = nearest_room_labels(room_lab)
    pieces = outline_pieces(grid, region, ras, labels_all) + interior_pieces(grid, region, ras, labels_all)
    colours = room_wall_colours(grid, region, labels_all, samples, rooms)
    stats = write_shell_layer(args.ceiling, grid, rooms, region, labels_all, pieces, colours)
    write_root()

    outline = [p for p in pieces if p["kind"] == "outline"]
    backed = [p for p in outline if p["backed"]]
    interior = [p for p in pieces if p["kind"] == "interior"]
    closure = [p for p in outline if not p["backed"]]
    scan_area = sum(p["length"] * min(p["scan_top"], args.ceiling) for p in backed)
    wall_stats = dict(
        outline_wall_m=round(sum(p["length"] for p in outline), 1),
        scan_backed_outline_m=round(sum(p["length"] for p in backed), 1),
        closure_wall_m=round(sum(p["length"] for p in closure), 1),
        interior_wall_m=round(sum(p["length"] for p in interior), 1),
        interior_wall_segments=len(interior),
        median_scan_wall_top_m=round(float(np.median([p["scan_top"] for p in backed])), 2) if backed else None,
        shell_wall_area_m2=round(sum(p["length"] for p in pieces) * args.ceiling, 1),
        scan_wall_area_behind_shell_m2=round(scan_area, 1),
        free_pixels_under_shell_walls=free_pixels_under_walls(grid, pieces),
        building_region_m2=round(float(region.sum()) * grid.res ** 2, 1),
        **stats,
    )
    spawns, goals = spawn_and_goals(grid, rooms, room_lab)
    bmin, bmax = stage_bounds("on")
    bmin_off, bmax_off = stage_bounds("off")
    meta = dict(
        stage="building.usda",
        units=dict(meters_per_unit=1.0, up_axis="Z"),
        ceiling_height_m=float(args.ceiling),
        layers=dict(scan_visual="layers/scan_visual.usdc", collision="layers/collision.usdc", shell="layers/shell.usda"),
        prims=dict(scan="/World/Scan", collider="/World/Collision/collision_mesh", shell="/World/Shell",
                   shell_walls="/World/Shell/Walls", shell_closure_walls="/World/Shell/ClosureWalls",
                   shell_ceilings="/World/Shell/Ceilings"),
        shell_switch="variant set 'shell' on /World/Shell: 'on' (default) references layers/shell.usda, 'off' has none",
        floor_map=dict(
            image="scans/merged/floor_map.png", yaml="scans/merged/floor_map.yaml", resolution_m=grid.res,
            size_px=[grid.w, grid.h], origin_xy=[grid.ox, grid.oy],
            map_to_stage_4x4=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            pixel_to_stage_xy="x = origin_x + (col + 0.5) * res ; y = origin_y + (rows - row - 0.5) * res "
                              "(row 0 = top image row)",
            pixel_to_stage_3x3=[[grid.res, 0, grid.ox + 0.5 * grid.res],
                                [0, -grid.res, grid.oy + (grid.h - 0.5) * grid.res], [0, 0, 1]]),
        bounding_box_m=dict(shell_on=dict(min=bmin, max=bmax), shell_off=dict(min=bmin_off, max=bmax_off)),
        wall_colours={k: dict(v, srgb=list(v["srgb"])) for k, v in colours.items()},
        shell_stats=wall_stats,
        spawn_rule="free map cell, >= %.2f m from every occupied cell and from every non-free cell; "
                   "yaw radians from +X toward +Y; z = 0 floor" % SPAWN_CLEARANCE_M,
        spawn_poses=spawns,
        goals=goals,
    )
    with open(META, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=None, width=120)
    debug_image(grid, region, pieces)
    print("SHELL", json.dumps(wall_stats))
    print("COLOURS", json.dumps({k: (v["srgb"], v["samples"]) for k, v in colours.items()}))
    print("META spawns %d over %d areas, goals %d; bounds on %s %s" % (
        len(spawns), len({s["area"] for s in spawns}), len(goals), bmin, bmax))


def debug_image(grid, region, pieces):
    img = np.full((grid.h, grid.w, 3), 205, np.uint8)
    img[region] = (235, 235, 215)
    img[grid.free] = (255, 255, 255)
    img[grid.occ] = (70, 70, 70)
    colour = dict(outline=(0, 160, 0), closure=(255, 0, 255), interior=(0, 90, 255))
    for p in pieces:
        (c0, c1), (r0, r1) = grid.to_px([p["p0"][0], p["p1"][0]], [p["p0"][1], p["p1"][1]])
        key = "interior" if p["kind"] == "interior" else ("outline" if p["backed"] else "closure")
        cv2.line(img, (int(c0), int(r0)), (int(c1), int(r1)), colour[key], 3)
    from PIL import Image

    os.makedirs(sa.CACHE, exist_ok=True)
    Image.fromarray(img).save(os.path.join(sa.CACHE, "shell_plan.png"))


if __name__ == "__main__":
    main()
