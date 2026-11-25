"""Visual scene for the car's upward camera: the filmed DA3 rooms wherever the clips reached, the scan's own photo
look only where no clip did (and in DA3 holes). building.usda is not touched; collision stays the scan's everywhere.

  <python with pxr, trimesh, shapely> sim/isaac/scene/build_visual.py       (project root; CPU, about a minute)

Writes next to this file:
  building_visual.usda       /World/Scan and /World/Shell exactly as in building.usda but invisible (the ray-cast
                             depth camera and sonar still hit them), /World/Collision, /World/Visual (visible)
  layers/visual/visual.usdc  /Visual: Looks (one plain OmniPBR slot per area x wall/floor/ceiling, scan atlas looks),
                             Scan (scan faces with the scan atlas, minus those a filmed room replaces),
                             Rooms (one cropped DA3 mesh per clip with displayColor; no GeomSubsets: RTX in Isaac
                             Sim 6.2 drew spurious triangles from subset meshes and logged CUDA copy errors),
                             Shell (generated walls/ceilings with UVs and a floor underlay 1.2 cm below the
                             floor datum that fills scan floor holes), Lights (ceiling sphere lights + dome)
  visual_meta.json           slots and their colours, scan looks, filmed rooms, lights for the randomiser
  proof/visual_poses.json    two car poses per area for contact sheets (run.py views format)
  cache/visual_plan.png      which clip owns which floor cells

Filmed area of a clip: floor cells under its DA3 vertices (below 2.5 m), closed and hole-filled, inside the areas
where it saw at least 1 m2 of floor and within 4.5 m of its camera path; overlapping clips go to the one with more
vertices there. In that area every scan wall and ceiling face is hidden (DA3 shows the room's walls; the generated
shell behind them fills DA3 wall holes), and any other scan face is hidden if the clip's mesh comes within 15 cm
of it, so only DA3 holes (under tables, low clutter) keep the scan. DA3 faces are kept 4 cells past the owned area
(seam overlap) and below 2.8 m. Classes: floor = level below 0.15 m, ceiling = level above 1.9 m, wall = steep and
on a structural wall cell (tall scan surfaces or the building outline) or above 1.9 m. Visible scan faces keep the
scan atlas; the generated shell (walls, flat ceilings, floor underlay) takes plain per-area colours: the scan's
median wall and floor colour of that area, a fixed off-white ceiling. Shell walls and the underlay are cut away
where DA3 has a surface nearby (cut_near_da3), so in filmed rooms they only fill DA3 holes. DA3 vertex colours
are sRGB and are written to displayColor in linear space. UVs are planar in metres along the building's dominant wall axes.
"""

import glob
import json
import math
import os

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
from scipy import ndimage
from scipy.spatial import cKDTree

import build_scene as bs
import scene_analysis as sa

ROOMS_DIR = os.path.join(sa.ROOT, "sim", "rooms")
VISUAL_DIR = os.path.join(sa.HERE, "layers", "visual")
VISUAL_USD = os.path.join(VISUAL_DIR, "visual.usdc")
ROOT_VISUAL = os.path.join(sa.HERE, "building_visual.usda")
VISUAL_META = os.path.join(sa.HERE, "visual_meta.json")
POSES = os.path.join(sa.HERE, "proof", "visual_poses.json")
TEXTURES_FROM_LAYER = "../../textures/"

WALL, FLOOR, CEILING, OTHER = 0, 1, 2, 3
CLASS_NAMES = ("wall", "floor", "ceiling")
FLOOR_MAX_Z, CEILING_MIN_Z, LEVEL_NZ, STEEP_NZ = 0.15, 1.9, 0.8, 0.4
HIDE_NEAR_M = 0.15
ROOM_MAX_Z = 2.8
MIN_AREA_FLOOR_M2 = 1.0
CAMERA_REACH_M = 4.5
SEAM_PX = 4
FILL_TILE_M = 0.1
FILL_HIDE_M = {"wall": 0.25, "ceiling": 0.0, "floor": 0.15}
WALL_REACH_M = 0.3
LIGHT_Z = 2.7
FLOOR_UNDERLAY_Z = -0.012
LIGHT_RADIUS_M = 0.12
LIGHT_INTENSITY = 20000.0
DOME_INTENSITY = 150.0
LIGHT_TEMPERATURE_K = 4000.0
CEILING_SRGB = (240, 238, 232)


def write_mesh(stage, path, points, faces, st=None, colors=None):
    """Triangle mesh; st is per face corner [F,3,2] (faceVarying), colors per vertex (displayColor)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = np.asarray(points, np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, points.min(0))), Gf.Vec3f(*map(float, points.max(0)))]))
    if st is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                      UsdGeom.Tokens.faceVarying)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(st, np.float32).reshape(-1, 2)))
    if colors is not None:
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.asarray(colors, np.float32)))
    return mesh


def compact(tri):
    flat = tri.reshape(-1, 3)
    points, inverse = np.unique(flat, axis=0, return_inverse=True)
    return points, inverse.reshape(-1, 3)


def omnipbr(stage, path, texture=None, srgb=None, roughness=0.85):
    """OmniPBR with either a 1 m texture (scan atlas looks) or a plain sRGB colour (shell slots)."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if texture is not None:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0, 1.0))
    else:
        linear = [(c / 255.0) ** 2.2 for c in srgb]
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*linear))
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
        output.ConnectToSource(out)
    return mat


def cut_near_da3(pts, faces, da3_cells, tree, reach, grid):
    """Shell pieces are large polygons that sit within centimetres of the DA3 surfaces: tile them to FILL_TILE_M and
    drop the tiles inside DA3-covered cells that have DA3 geometry within reach, so they only fill DA3 holes. Tiles
    above ROOM_MAX_Z stay: the DA3 rooms are cropped there."""
    if reach <= 0:
        return pts, faces
    pts, faces = trimesh.remesh.subdivide_to_size(pts, faces, max_edge=FILL_TILE_M)
    cen = pts[faces].mean(1)
    drop = da3_cells[cells_of(grid, cen[:, :2])] & (cen[:, 2] < ROOM_MAX_Z)
    d, _ = tree.query(cen[drop], distance_upper_bound=reach)
    drop[np.flatnonzero(drop)[~np.isfinite(d)]] = False
    faces = faces[~drop]
    used, inverse = np.unique(faces, return_inverse=True)
    return pts[used], inverse.reshape(-1, 3)


def classify(centroids, normals, wall_cells, grid):
    nz, z = np.abs(normals[:, 2]), centroids[:, 2]
    c, r, ok = grid.cell(centroids[:, 0], centroids[:, 1])
    on_wall = np.zeros(len(z), bool)
    on_wall[ok] = wall_cells[r[ok], c[ok]]
    cls = np.full(len(z), OTHER, np.int8)
    cls[(nz > LEVEL_NZ) & (z < FLOOR_MAX_Z)] = FLOOR
    cls[(nz > LEVEL_NZ) & (z > CEILING_MIN_Z)] = CEILING
    cls[(nz < STEEP_NZ) & (on_wall | (z > CEILING_MIN_Z)) & (z > 0.03)] = WALL
    return cls


def planar_st(tri, normals, yaw):
    """Per-corner UVs in metres: level faces use plan XY, steep faces the horizontal axis they face along, and Z,
    both in the frame rotated by the building's dominant wall direction."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    x = tri[..., 0] * c - tri[..., 1] * s
    y = tri[..., 0] * s + tri[..., 1] * c
    nx = normals[:, 0] * c - normals[:, 1] * s
    ny = normals[:, 0] * s + normals[:, 1] * c
    level = (np.abs(normals[:, 2]) > 0.7)[:, None]
    u = np.where(level, x, np.where((np.abs(nx) > np.abs(ny))[:, None], y, x))
    v = np.where(level, y, tri[..., 2])
    return np.stack((u, v), -1)


def dominant_yaw(normals, areas):
    steep = np.abs(normals[:, 2]) < 0.2
    angle = np.mod(np.arctan2(normals[steep, 1], normals[steep, 0]), math.pi / 2)
    hist, edges = np.histogram(angle, bins=90, range=(0, math.pi / 2), weights=areas[steep])
    hist = ndimage.uniform_filter1d(hist, 3, mode="wrap")
    return float(edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2)


def cells_of(grid, xy):
    c, r = grid.to_px(xy[:, 0], xy[:, 1])
    return (np.clip(np.floor(r).astype(int), 0, grid.h - 1), np.clip(np.floor(c).astype(int), 0, grid.w - 1))


def load_clips(grid, rooms, labels_all):
    clips = []
    for ready in sorted(glob.glob(os.path.join(ROOMS_DIR, "IMG_*", "sim_ready.json"))):
        folder = os.path.dirname(ready)
        with open(ready) as fh:
            if not json.load(fh).get("sim_ready"):
                continue
        with open(os.path.join(folder, "transform.json")) as fh:
            transform = json.load(fh)
        areas = [b["area"] for b in transform["coverage"]["by_area"] if b["da3_floor_in_area_m2"] >= MIN_AREA_FLOOR_M2]
        usd = Usd.Stage.Open(os.path.join(folder, "room_aligned.usd"))
        mesh = UsdGeom.Mesh(usd.GetPrimAtPath("/Room/mesh"))
        points = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
        colors = np.asarray(mesh.GetDisplayColorPrimvar().Get(), np.float32)
        cams = np.load(os.path.join(folder, "cam2world_aligned.npy"))[:, :2, 3]
        area_ids = [i + 1 for i, room in enumerate(rooms) if room["id"] in areas]
        allowed = np.isin(labels_all, area_ids)
        band = points[:, 2] < 2.5
        rr, cc = cells_of(grid, points[band, :2])
        count = np.zeros((grid.h, grid.w), np.float32)
        np.add.at(count, (rr, cc), 1)
        cover = ndimage.binary_fill_holes(ndimage.binary_closing(count > 0, structure=bs.disk(3)))
        cam_mask = np.zeros((grid.h, grid.w), bool)
        cam_mask[cells_of(grid, cams)] = True
        near = ndimage.distance_transform_edt(~cam_mask) * grid.res <= CAMERA_REACH_M
        score = np.where(cover & allowed & near, ndimage.gaussian_filter(count, 3) + 1e-3, 0)
        clips.append(dict(name=os.path.basename(folder), areas=areas, points=points, faces=faces, colors=colors,
                          score=score))
        print(f"CLIP {clips[-1]['name']}: areas {areas}, {len(faces)} faces, covered cells {(score > 0).sum()}")
    return clips


def assign_owner(grid, clips):
    if not clips:
        return np.zeros((grid.h, grid.w), np.int32)
    stack = np.stack([c["score"] for c in clips])
    return np.where(stack.max(0) > 0, stack.argmax(0) + 1, 0).astype(np.int32)


def room_index(grid, labels_all, xy):
    return labels_all[cells_of(grid, xy)]


def floor_colours(grid, labels_all, samples, rooms):
    xyz, nz, rgb = samples["xyz"], np.abs(samples["nrm"][:, 2]), samples["rgb"]
    sel = (nz > LEVEL_NZ) & (xyz[:, 2] < FLOOR_MAX_Z)
    lab = room_index(grid, labels_all, xyz[sel, :2])
    out = {}
    for i, room in enumerate(rooms):
        m = lab == i + 1
        out[room["id"]] = tuple(int(v) for v in np.median(rgb[sel][m], 0)) if m.sum() > 100 else (150, 120, 90)
    return out


def poses(grid, rooms, room_lab, meta):
    out = []
    for i, room in enumerate(rooms):
        spawns = np.array([[p["x"], p["y"]] for p in meta["spawn_poses"] if p["area"] == room["id"]])
        if len(spawns) == 0:
            continue
        pick = bs.farthest_points(spawns, 2, np.random.default_rng(i))
        cells = np.argwhere(room_lab == i + 1)
        xy = np.stack(grid.to_world(cells[:, 1] + 0.5, cells[:, 0] + 0.5), 1)
        for k, j in enumerate(pick):
            far = xy[np.argmax(np.linalg.norm(xy - spawns[j], axis=1))]
            yaw = math.degrees(math.atan2(far[1] - spawns[j, 1], far[0] - spawns[j, 0]))
            out.append(dict(kind="visual", area=room["id"], file=f"{room['id']}_{k}.jpg", x=round(float(spawns[j, 0]), 3),
                            y=round(float(spawns[j, 1]), 3), yaw=round(yaw, 1), z=0.0))
    return out


def light_spots(grid, rooms, room_lab):
    spots = []
    for i, room in enumerate(rooms):
        cells = np.argwhere(room_lab == i + 1)
        if len(cells) == 0:
            continue
        n = int(np.clip(round(room["area_m2"] / 15.0), 1, 4))
        xy = np.stack(grid.to_world(cells[:, 1] + 0.5, cells[:, 0] + 0.5), 1)
        for j in bs.farthest_points(xy, n, np.random.default_rng(100 + i)):
            spots.append((room["id"], float(xy[j, 0]), float(xy[j, 1])))
    return spots


def debug_plan(grid, owner, clips):
    import cv2

    img = np.full((grid.h, grid.w, 3), 205, np.uint8)
    img[grid.free] = 255
    img[grid.occ] = 70
    palette = [(230, 80, 80), (80, 170, 80), (80, 110, 230), (220, 170, 40), (170, 80, 200)]
    for k in range(len(clips)):
        img[owner == k + 1] = (0.5 * img[owner == k + 1] + 0.5 * np.array(palette[k % 5])).astype(np.uint8)
    cv2.imwrite(os.path.join(sa.CACHE, "visual_plan.png"), img[..., ::-1])


def main():
    grid = sa.Grid()
    rooms = grid.rooms()
    samples = sa.surface_samples()
    ras = sa.plan_rasters(grid, samples)
    region = bs.building_region(grid, ras)
    room_lab = grid.room_labels(rooms)
    labels_all = bs.nearest_room_labels(room_lab)
    wall_cells = ndimage.binary_dilation(ndimage.binary_closing(ras["tall"], structure=bs.disk(2)), structure=bs.disk(4))
    wall_cells |= ndimage.distance_transform_edt(region) <= 8
    with open(bs.META) as fh:
        import yaml

        scene_meta = yaml.safe_load(fh)

    clips = load_clips(grid, rooms, labels_all)
    owner = assign_owner(grid, clips)
    trees = [cKDTree(c["points"]) for c in clips]
    # Room walls stand at the edge of the floor a clip owns, often just outside it: walls, ceilings and shell tiles
    # count as that clip's within WALL_REACH_M of its floor.
    gap, (near_r, near_c) = ndimage.distance_transform_edt(owner == 0, return_indices=True)
    owner_wide = np.where(gap * grid.res <= WALL_REACH_M, owner[near_r, near_c], 0)
    parts = sa.load_glb_parts()
    all_normals = np.concatenate([m.face_normals for _n, m in parts])
    all_areas = np.concatenate([m.area_faces for _n, m in parts])
    yaw = dominant_yaw(all_normals, all_areas)
    print(f"DOMINANT_WALL_YAW {math.degrees(yaw):.1f} deg")

    os.makedirs(VISUAL_DIR, exist_ok=True)
    stage = bs.new_stage(VISUAL_USD)
    root = UsdGeom.Xform.Define(stage, "/Visual")
    stage.SetDefaultPrim(root.GetPrim())
    for scope in ("Looks", "Scan", "Rooms", "Shell", "Lights"):
        UsdGeom.Scope.Define(stage, "/Visual/" + scope)

    wall_rgb = bs.room_wall_colours(grid, region, labels_all, samples, rooms)
    floor_rgb = floor_colours(grid, labels_all, samples, rooms)
    room_owner = {}
    for i, room in enumerate(rooms):
        own = owner[room_lab == i + 1]
        counts = np.bincount(own, minlength=len(clips) + 1)[1:] if own.size else np.zeros(len(clips))
        best = int(np.argmax(counts)) if len(clips) and counts.max() > 0.3 * max(own.size, 1) else -1
        room_owner[room["id"]] = clips[best]["name"] if best >= 0 else None
    slots, slot_mats = [], {}
    for room in rooms:
        rid, clip = room["id"], room_owner[room["id"]]
        targets = {"wall": wall_rgb[rid]["srgb"], "floor": floor_rgb[rid], "ceiling": CEILING_SRGB}
        for kind in CLASS_NAMES:
            path = f"/Visual/Looks/slot_{rid}_{kind}"
            srgb = [int(c) for c in targets[kind]]
            slot_mats[(rid, kind)] = omnipbr(stage, path, srgb=srgb)
            slots.append(dict(area=rid, kind=kind, material="/World" + path, shader="/World" + path + "/Shader",
                              srgb=srgb, filmed_by=clip))

    textures = {os.path.splitext(os.path.basename(p))[0]: os.path.basename(p)
                for p in glob.glob(os.path.join(sa.HERE, "textures", "*"))}
    stats = dict(scan_faces=0, scan_hidden={}, scan_kept_in_filmed=0, scan_by_class=[0, 0, 0, 0])
    scan_looks = []
    for name, mesh in parts:
        tri = mesh.triangles
        cen, nrm = tri.mean(1), mesh.face_normals
        cls = classify(cen, nrm, wall_cells, grid)
        cells = cells_of(grid, cen[:, :2])
        own = np.where((cls == WALL) | (cls == CEILING), owner_wide[cells], owner[cells])
        hidden = np.zeros(len(cen), bool)
        for k, tree in enumerate(trees):
            m = own == k + 1
            if m.any():
                d, _ = tree.query(cen[m], distance_upper_bound=HIDE_NEAR_M)
                replaced = np.isfinite(d) | (cls[m] == WALL) | (cls[m] == CEILING)
                hidden[np.flatnonzero(m)[replaced]] = True
                stats["scan_hidden"][clips[k]["name"]] = stats["scan_hidden"].get(clips[k]["name"], 0) + int(
                    replaced.sum())
        stats["scan_faces"] += len(cen)
        stats["scan_kept_in_filmed"] += int(((own > 0) & ~hidden).sum())
        for c in range(4):
            stats["scan_by_class"][c] += int(((cls == c) & ~hidden).sum())
        sel = ~hidden
        look = omnipbr(stage, f"/Visual/Looks/scan_{name}", TEXTURES_FROM_LAYER + textures[name], roughness=0.9)
        scan_looks.append(dict(part=name, shader=f"/World/Visual/Looks/scan_{name}/Shader"))
        points, faces = compact(tri[sel])
        uv = mesh.visual.uv[mesh.faces[sel]]
        piece = write_mesh(stage, f"/Visual/Scan/{name}", points, faces, st=uv)
        UsdShade.MaterialBindingAPI.Apply(piece.GetPrim()).Bind(look)

    filmed = []
    for k, clip in enumerate(clips):
        keep_cells = ndimage.binary_dilation(owner == k + 1, structure=bs.disk(SEAM_PX))
        pts, faces = clip["points"], clip["faces"]
        tri = pts[faces]
        cen = tri.mean(1)
        keep = keep_cells[cells_of(grid, cen[:, :2])] & (cen[:, 2] < ROOM_MAX_Z)
        faces = faces[keep]
        used, inverse = np.unique(faces, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        pts, colors = pts[used], clip["colors"][used]
        path = f"/Visual/Rooms/{clip['name']}"
        UsdGeom.Xform.Define(stage, path)
        write_mesh(stage, f"{path}/photo", pts, faces, colors=np.power(colors, 2.2))
        filmed.append(dict(clip=clip["name"], faces=len(faces), areas=clip["areas"],
                           owned_m2=round(float((owner == k + 1).sum()) * grid.res ** 2, 1)))
        print(f"ROOM {clip['name']}: kept {len(faces)} of {len(clip['faces'])} faces, owned "
              f"{filmed[-1]['owned_m2']} m2, hides {stats['scan_hidden'].get(clip['name'], 0)} scan faces")

    da3_cells = owner_wide > 0
    da3_tree = cKDTree(np.concatenate([c["points"] for c in clips]))
    shell = Usd.Stage.Open(bs.SHELL_USD)
    for prim in shell.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = prim.GetName()
        kind = "ceiling" if name.startswith("shell_ceiling_") else "wall"
        rid = name.split("_", 2)[2]
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        tris, start = [], 0
        for n in counts:
            f = idx[start:start + n]
            tris += [(f[0], f[j], f[j + 1]) for j in range(1, n - 1)]
            start += n
        faces = np.array(tris)
        pieces = [(name, kind, pts, faces)]
        if kind == "ceiling":
            under = pts.copy()
            under[:, 2] = FLOOR_UNDERLAY_Z
            pieces.append((f"floor_underlay_{rid}", "floor", under, faces))
        for piece, look, p, f in pieces:
            p, f = cut_near_da3(p, f, da3_cells, da3_tree, FILL_HIDE_M[look], grid)
            if not len(f):
                continue
            tri = p[f]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            out = write_mesh(stage, f"/Visual/Shell/{piece}", p, f, st=planar_st(tri, nrm, yaw))
            UsdShade.MaterialBindingAPI.Apply(out.GetPrim()).Bind(slot_mats[(rid, look)])

    lights = []
    for j, (rid, x, y) in enumerate(light_spots(grid, rooms, room_lab)):
        light = UsdLux.SphereLight.Define(stage, f"/Visual/Lights/lamp_{rid}_{j}")
        light.CreateRadiusAttr(LIGHT_RADIUS_M)
        light.CreateIntensityAttr(LIGHT_INTENSITY)
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(LIGHT_TEMPERATURE_K)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x, y, LIGHT_Z))
        lights.append(dict(path=f"/World/Visual/Lights/lamp_{rid}_{j}", area=rid, intensity=LIGHT_INTENSITY))
    dome = UsdLux.DomeLight.Define(stage, "/Visual/Lights/dome")
    dome.CreateIntensityAttr(DOME_INTENSITY)
    dome.CreateEnableColorTemperatureAttr(True)
    dome.CreateColorTemperatureAttr(LIGHT_TEMPERATURE_K)
    stage.GetRootLayer().Save()

    root_stage = bs.new_stage(ROOT_VISUAL)
    world = UsdGeom.Xform.Define(root_stage, "/World")
    root_stage.SetDefaultPrim(world.GetPrim())
    root_stage.GetRootLayer().documentation = ("Visual scene (DA3 rooms, scan photo look elsewhere); generated by build_visual.py. "
                                               "Scan and Shell are invisible ray-cast targets, Collision as building.usda")
    for path, ref, visible in (("/World/Scan", "./layers/scan_visual.usdc", False),
                               ("/World/Collision", "./layers/collision.usdc", True),
                               ("/World/Shell", "./layers/shell.usda", False),
                               ("/World/Visual", "./layers/visual/visual.usdc", True)):
        prim = UsdGeom.Xform.Define(root_stage, path)
        prim.GetPrim().GetReferences().AddReference(ref)
        if not visible:
            prim.MakeInvisible()
    root_stage.GetRootLayer().Save()

    view_poses = poses(grid, rooms, room_lab, scene_meta)
    os.makedirs(os.path.dirname(POSES), exist_ok=True)
    with open(POSES, "w") as fh:
        json.dump(dict(stage="building_visual.usda", poses=view_poses), fh, indent=1)
    meta = dict(stage="building_visual.usda", visual_layer="layers/visual/visual.usdc",
                dominant_wall_yaw_deg=round(math.degrees(yaw), 2), slots=slots, scan_looks=scan_looks,
                filmed=filmed, lights=lights, dome=dict(path="/World/Visual/Lights/dome", intensity=DOME_INTENSITY),
                light_temperature_k=LIGHT_TEMPERATURE_K, stats=stats)
    with open(VISUAL_META, "w") as fh:
        json.dump(meta, fh, indent=1)
    debug_plan(grid, owner, clips)
    print("VISUAL_OK", json.dumps(dict(slots=len(slots), filmed=[f["clip"] for f in filmed], lights=len(lights),
                                       poses=len(view_poses), scan_faces=stats["scan_faces"],
                                       kept_by_class=stats["scan_by_class"], hidden=stats["scan_hidden"],
                                       kept_in_filmed=stats["scan_kept_in_filmed"])))


if __name__ == "__main__":
    main()
