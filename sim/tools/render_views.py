"""Render the merged building scan as the robot car's camera sees it.

One file, three stages:

  plan     (sim venv python)  pick free-floor camera poses per room and a drive path
                              from floor_map.png / rooms.json -> views/poses.json
  render   (Blender, -b -P)   import building.glb, flat texture-colour lighting
                              (Workbench FLAT + TEXTURE), render every pose
  gallery  (sim venv python)  drive.mp4, map with camera markers, index.html,
                              contact_sheet.jpg

Usage from the project root:

  sim/.venv/bin/python sim/tools/render_views.py all --pitch 20 --hfov 65
  # or stage by stage:
  sim/.venv/bin/python sim/tools/render_views.py plan --pitch 20 --hfov 65
  /opt/homebrew/bin/blender -b -P sim/tools/render_views.py -- scans/merged/views/poses.json
  sim/.venv/bin/python sim/tools/render_views.py gallery

Frame: merged scan frame, metres, +Z up. Yaw is measured from +X toward +Y.
Pitch is positive upward. The camera height is above the floor directly below
the camera, found by a downward ray cast in Blender.
"""

import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MERGED = os.path.join(ROOT, "scans", "merged")
VIEWS = os.path.join(MERGED, "views")
BLENDER = "/opt/homebrew/bin/blender"

CAM_HEIGHT_M = 0.126
WIDTH, HEIGHT = 640, 480
HOLE_MAGENTA = (0.55, 0.0, 0.55)


def blender_main(poses_path):
    import bpy
    from mathutils import Matrix, Vector

    with open(poses_path) as f:
        plan = json.load(f)
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=plan["glb"])
    t_import = time.time() - t0

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

    def bounds():
        lo = Vector((1e9, 1e9, 1e9))
        hi = Vector((-1e9, -1e9, -1e9))
        bpy.context.view_layer.update()
        for o in meshes:
            for c in o.bound_box:
                w = o.matrix_world @ Vector(c)
                lo = Vector(map(min, lo, w))
                hi = Vector(map(max, hi, w))
        return lo, hi

    lo, hi = bounds()
    # building.glb stores +Z up data; the glTF importer converts Y-up to Z-up,
    # which turns the building on its side. Undo that if the height span is large.
    if hi.z - lo.z > 5.0:
        fix = Matrix.Rotation(math.radians(-90.0), 4, "X")
        for o in bpy.context.scene.objects:
            if o.parent is None:
                o.matrix_world = fix @ o.matrix_world
        lo, hi = bounds()
    print("RENDER bounds min", tuple(round(v, 3) for v in lo), "max", tuple(round(v, 3) for v in hi))

    # Workbench uses each material's active image node as texture colour.
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        for link in mat.node_tree.links:
            if link.to_socket.name == "Base Color" and link.from_node.type == "TEX_IMAGE":
                nodes.active = link.from_node

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "TEXTURE"
    shading.show_backface_culling = False
    shading.show_shadows = False
    shading.show_cavity = False
    scene.display.render_aa = "8"
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("bg")
    world.color = HOLE_MAGENTA
    scene.world = world
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90

    cam_data = bpy.data.cameras.new("car_cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.01
    cam_data.clip_end = 60.0
    cam = bpy.data.objects.new("car_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    depsgraph = bpy.context.evaluated_depsgraph_get()

    def floor_z(x, y):
        hit, loc, _n, _i, _o, _m = scene.ray_cast(depsgraph, Vector((x, y, 0.35)), Vector((0, 0, -1)), distance=0.8)
        if hit and -0.15 < loc.z < 0.25:
            return loc.z
        return 0.0

    done = []
    t1 = time.time()
    for p in plan["poses"]:
        z = floor_z(p["x"], p["y"]) + CAM_HEIGHT_M
        cam_data.angle = math.radians(p["hfov"])
        cam.location = (p["x"], p["y"], z)
        cam.rotation_euler = (math.radians(90.0 + p["pitch"]), 0.0, math.radians(p["yaw"] - 90.0))
        out = os.path.join(plan["views"], p["file"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        done.append(dict(p, z=round(z, 4)))
    t_render = time.time() - t1
    with open(os.path.join(plan["views"], "poses_rendered.json"), "w") as f:
        json.dump({"import_s": round(t_import, 1), "render_s": round(t_render, 1),
                   "n": len(done), "poses": done}, f, indent=1)
    print("RENDER done %d images, import %.1fs, render %.1fs" % (len(done), t_import, t_render))


class FloorMap:
    def __init__(self):
        import numpy as np
        from PIL import Image
        from scipy import ndimage

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
        self.clear = ndimage.distance_transform_edt(self.free) * self.res

    def to_px(self, x, y):
        col, row = (x - self.ox) / self.res, self.h - (y - self.oy) / self.res
        return col, row

    def to_world(self, col, row):
        return self.ox + (col + 0.5) * self.res, self.oy + (self.h - row - 0.5) * self.res

    def room_mask(self, room):
        import numpy as np
        from PIL import Image, ImageDraw

        im = Image.new("L", (self.w, self.h), 0)
        d = ImageDraw.Draw(im)
        d.polygon([self.to_px(x, y) for x, y in room["polygon"]], fill=1)
        for hole in room.get("holes", []):
            d.polygon([self.to_px(x, y) for x, y in hole], fill=0)
        return np.array(im, dtype=bool)

    def free_distance_along(self, col, row, yaw_deg, max_m=5.0):
        c, s = math.cos(math.radians(yaw_deg)), -math.sin(math.radians(yaw_deg))
        n = int(max_m / self.res)
        for i in range(1, n):
            cc, rr = int(col + c * i), int(row + s * i)
            if not (0 <= cc < self.w and 0 <= rr < self.h) or not self.free[rr, cc]:
                return i * self.res
        return max_m


def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def plan_room_poses(fm, room, n_views, pitch, hfov, min_spread_m=0.5):
    import numpy as np

    mask = fm.room_mask(room) & fm.free
    best = None
    for thr in (0.40, 0.30, 0.22, 0.15, 0.10):
        rows, cols = np.nonzero(mask & (fm.clear >= thr))
        if len(rows) < n_views:
            continue
        pts = np.stack([cols, rows], 1).astype(float)
        first = int(np.argmax(fm.clear[rows, cols]))
        chosen = [first]
        dmin = np.linalg.norm(pts - pts[first], axis=1)
        while len(chosen) < n_views:
            nxt = int(np.argmax(dmin))
            spread_m = dmin[nxt] * fm.res
            chosen.append(nxt)
            dmin = np.minimum(dmin, np.linalg.norm(pts - pts[nxt], axis=1))
        if best is None or spread_m > best[0]:
            best = (spread_m, thr, pts, chosen)
        if spread_m >= min_spread_m:
            break
    if best is None:
        return [], None
    _spread, used_thr, pts, chosen = best
    yaws = np.arange(0, 360, 5.0)
    poses = []
    base = None
    for k, idx in enumerate(chosen):
        col, row = pts[idx]
        rays = np.array([fm.free_distance_along(col, row, y) for y in yaws])
        if base is None:
            base = float(yaws[int(np.argmax(rays))])
        want = (base + k * 360.0 / n_views) % 360.0
        score = np.minimum(rays, 2.0) - 3.0 * np.array([angdiff(y, want) for y in yaws]) / 180.0
        j = int(np.argmax(score))
        x, y = fm.to_world(col, row)
        poses.append(dict(file="%s_%d.jpg" % (room["id"], k + 1), room=room["id"], kind="room",
                          x=round(x, 3), y=round(y, 3), yaw=float(yaws[j]), pitch=pitch, hfov=hfov,
                          clearance_m=round(float(fm.clear[int(row), int(col)]), 3),
                          free_ahead_m=round(float(rays[j]), 2)))
    return poses, used_thr


def plan_drive(fm, rooms, room_poses, n_rooms, pitch, hfov, max_frames):
    import numpy as np
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    step = round(0.1 / fm.res)
    gh, gw = fm.h // step, fm.w // step
    clr = fm.clear[step // 2::step, step // 2::step][:gh, :gw]
    ok = clr >= 0.18
    lab, _ = ndimage.label(ok, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp = lab == int(np.argmax(sizes))
    idx = -np.ones((gh, gw), int)
    rr, cc = np.nonzero(comp)
    idx[rr, cc] = np.arange(len(rr))
    src, dst, wts = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc
        good = (r2 >= 0) & (r2 < gh) & (c2 >= 0) & (c2 < gw)
        good[good] &= comp[r2[good], c2[good]]
        a, b = idx[rr[good], cc[good]], idx[r2[good], c2[good]]
        pen = 1.0 + 3.0 * np.clip(0.45 - np.minimum(clr[rr[good], cc[good]], clr[r2[good], c2[good]]), 0, None) / 0.45
        src += [a, b]
        dst += [b, a]
        wts += [math.hypot(dr, dc) * pen] * 2
    g = coo_matrix((np.concatenate(wts), (np.concatenate(src), np.concatenate(dst))), shape=(len(rr),) * 2).tocsr()

    way = {}
    for room in rooms:
        m = fm.room_mask(room)[step // 2::step, step // 2::step][:gh, :gw] & comp
        if m.sum() < 4:
            continue
        r_, c_ = np.nonzero(m)
        j = int(np.argmax(clr[r_, c_]))
        way[room["id"]] = idx[r_[j], c_[j]]
    ids = list(way)
    dist, pred = dijkstra(g, indices=[way[i] for i in ids], return_predecessors=True)
    best = None
    for s in range(len(ids)):
        order, cur = [s], s
        while len(order) < n_rooms:
            rest = [k for k in range(len(ids)) if k not in order and np.isfinite(dist[cur, way[ids[k]]])]
            if not rest:
                break
            cur = min(rest, key=lambda k: dist[cur, way[ids[k]]])
            order.append(cur)
        length = sum(dist[a, way[ids[b]]] for a, b in zip(order, order[1:]))
        key = (len(order), -length)
        if best is None or key > best[0]:
            best = (key, order)
    order = best[1]
    cells = []
    for a, b in zip(order, order[1:]):
        seg, node = [], way[ids[b]]
        while node != way[ids[a]] and node >= 0:
            seg.append(node)
            node = pred[a, node]
        seg.append(way[ids[a]])
        cells += seg[::-1][1 if cells else 0:]
    pts = np.array([fm.to_world(cc[i] * step + step // 2, rr[i] * step + step // 2) for i in cells])
    k = 9
    pad = np.vstack([np.repeat(pts[:1], k, 0), pts, np.repeat(pts[-1:], k, 0)])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.stack([np.convolve(pad[:, d], ker, "valid") for d in (0, 1)], 1)
    sm[0], sm[-1] = pts[0], pts[-1]
    seglen = np.linalg.norm(np.diff(sm, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seglen)])
    total = s[-1]
    n = int(min(max_frames, max(40, total / 0.08)))
    si = np.linspace(0, total, n)
    xs, ys = np.interp(si, s, sm[:, 0]), np.interp(si, s, sm[:, 1])
    look = 0.35
    xa, ya = np.interp(np.minimum(si + look, total), s, sm[:, 0]), np.interp(np.minimum(si + look, total), s, sm[:, 1])
    xb, yb = np.interp(np.maximum(si - look, 0), s, sm[:, 0]), np.interp(np.maximum(si - look, 0), s, sm[:, 1])
    yaw = np.degrees(np.unwrap(np.arctan2(ya - yb, xa - xb)))
    frames, min_clear = [], 9.0
    for i in range(n):
        col, row = fm.to_px(xs[i], ys[i])
        c = float(fm.clear[int(row), int(col)])
        min_clear = min(min_clear, c)
        frames.append(dict(file="drive/frame_%04d.jpg" % (i + 1), kind="drive", x=round(float(xs[i]), 3),
                           y=round(float(ys[i]), 3), yaw=round(float(yaw[i]) % 360.0, 2), pitch=pitch,
                           hfov=hfov, clearance_m=round(c, 3)))
    info = dict(rooms=[ids[i] for i in order], length_m=round(float(total), 2), frames=n,
                min_clearance_m=round(min_clear, 3))
    return frames, info


def cmd_plan(args):
    fm = FloorMap()
    with open(os.path.join(MERGED, "rooms.json")) as f:
        rooms = json.load(f)
    poses, room_info = [], {}
    for room in rooms:
        ps, thr = plan_room_poses(fm, room, args.views, args.pitch, args.hfov)
        room_info[room["id"]] = dict(name=room["name"], area_m2=room["area_m2"], n=len(ps), clearance_thr_m=thr)
        poses += ps
    big = sorted(rooms, key=lambda r: -r["area_m2"])[:8]
    for room in big:
        p = next(q for q in poses if q["room"] == room["id"])
        poses.append(dict(p, file="level/%s_level.jpg" % p["file"][:-4], kind="level", pitch=0.0))
    drive, drive_info = plan_drive(fm, rooms, poses, args.drive_rooms, args.pitch, args.hfov, args.max_frames)
    poses += drive
    os.makedirs(VIEWS, exist_ok=True)
    out = os.path.join(VIEWS, "poses.json")
    with open(out, "w") as f:
        json.dump(dict(glb=os.path.join(MERGED, "building.glb"), views=VIEWS, pitch=args.pitch, hfov=args.hfov,
                       cam_height=CAM_HEIGHT_M, rooms=room_info, drive=drive_info, poses=poses), f, indent=1)
    for rid, ri in room_info.items():
        print("PLAN %s n=%d clearance>=%s" % (rid, ri["n"], ri["clearance_thr_m"]))
    print("PLAN drive", drive_info)
    print("PLAN total poses", len(poses), "->", out)
    return out


def cmd_gallery(args):
    import base64
    import io
    import subprocess

    from PIL import Image, ImageDraw, ImageFont

    with open(os.path.join(VIEWS, "poses.json")) as f:
        plan = json.load(f)
    with open(os.path.join(VIEWS, "poses_rendered.json")) as f:
        rendered = json.load(f)
    with open(os.path.join(MERGED, "rooms.json")) as f:
        rooms = json.load(f)
    poses = rendered["poses"]

    mp4 = os.path.join(VIEWS, "drive.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i",
                    os.path.join(VIEWS, "drive", "frame_%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", mp4], check=True)

    fm = FloorMap()
    base = Image.fromarray(fm.img).convert("RGB")
    known = fm.img != 205
    import numpy as np
    rr, cc = np.nonzero(known)
    pad = 40
    box = (max(cc.min() - pad, 0), max(rr.min() - pad, 0), min(cc.max() + pad, fm.w), min(rr.max() + pad, fm.h))
    scale = 1.0
    img = base.copy()
    d = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except OSError:
        font = small = ImageFont.load_default()
    palette = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 200, 200),
               (240, 50, 230), (150, 150, 0), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128),
               (100, 100, 100)]
    colour = {r["id"]: palette[i % len(palette)] for i, r in enumerate(rooms)}
    for r in rooms:
        d.polygon([fm.to_px(x, y) for x, y in r["polygon"]], outline=colour[r["id"]] + (255,), width=2)
        cx, cy = fm.to_px(r["center"]["x"], r["center"]["y"])
        d.text((cx + 8, cy - 22), r["id"].replace("area_", "A"), fill=colour[r["id"]] + (255,), font=font)
    drive = [p for p in poses if p["kind"] == "drive"]
    d.line([fm.to_px(p["x"], p["y"]) for p in drive], fill=(255, 200, 0, 255), width=4)
    for p in poses:
        if p["kind"] != "room":
            continue
        col, row = fm.to_px(p["x"], p["y"])
        c = colour[p["room"]] + (255,)
        L = 22
        ex = col + L * math.cos(math.radians(p["yaw"]))
        ey = row - L * math.sin(math.radians(p["yaw"]))
        half = p["hfov"] / 2
        for a in (p["yaw"] - half, p["yaw"] + half):
            d.line([(col, row), (col + 16 * math.cos(math.radians(a)), row - 16 * math.sin(math.radians(a)))],
                   fill=c[:3] + (120,), width=1)
        d.line([(col, row), (ex, ey)], fill=c, width=3)
        d.ellipse([col - 5, row - 5, col + 5, row + 5], fill=c, outline=(0, 0, 0, 255))
        d.text((col + 6, row + 3), p["file"][5:-4], fill=(0, 0, 0, 255), font=small)
    img = img.crop(box)
    img.save(os.path.join(VIEWS, "map_marked.png"))

    def b64(path, fmt="jpeg"):
        with open(path, "rb") as f:
            return "data:image/%s;base64,%s" % (fmt, base64.b64encode(f.read()).decode())

    def tag(p):
        return ('<figure><img src="%s" alt="%s"><figcaption>%s<br>x=%.2f y=%.2f yaw=%.0f pitch=%.0f '
                'hfov=%.0f clear=%.2fm</figcaption></figure>' % (
                    b64(os.path.join(VIEWS, p["file"])), p["file"], p["file"], p["x"], p["y"], p["yaw"],
                    p["pitch"], p["hfov"], p["clearance_m"]))

    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Car camera views</title><style>",
            "body{font-family:-apple-system,Helvetica,sans-serif;margin:20px;background:#f4f4f4;color:#222}",
            "figure{display:inline-block;margin:4px;background:#fff;padding:4px;vertical-align:top}",
            "figure img{width:320px;display:block}figcaption{font-size:11px}",
            "h2{margin-top:28px;border-bottom:3px solid}.map{max-width:100%;border:1px solid #999}",
            "video{max-width:640px}</style></head><body>",
            "<h1>Merged building scan, seen from the car camera</h1>",
            "<p>Camera %.3f m above the floor, %dx%d, pitched up %.0f deg, horizontal FOV %.0f deg "
            "(pitch and FOV are estimates). Flat lighting shows the scan texture colours unchanged. "
            "<b>Magenta</b> = no scan surface in that direction (hole or open edge). Map: black occupied, white "
            "free, grey unknown; dots and lines are camera positions and directions, the yellow line is the "
            "drive path.</p>" % (CAM_HEIGHT_M, WIDTH, HEIGHT, plan["pitch"], plan["hfov"]),
            "<img class='map' src='%s'>" % b64(os.path.join(VIEWS, "map_marked.png"), "png")]
    for r in rooms:
        ps = [p for p in poses if p["kind"] == "room" and p["room"] == r["id"]]
        html.append("<h2 style='border-color:rgb%s'>%s - %s (%.1f m2)</h2>" % (
            str(colour[r["id"]]), r["id"], r["name"], r["area_m2"]))
        html += [tag(p) for p in ps]
    di = plan["drive"]
    html.append("<h2>Drive-through: %s, %.1f m, %d frames</h2>" % (" &rarr; ".join(di["rooms"]), di["length_m"],
                                                                     di["frames"]))
    if os.path.getsize(mp4) < 20e6:
        with open(mp4, "rb") as f:
            html.append("<video controls loop src='data:video/mp4;base64,%s'></video><br>"
                        % base64.b64encode(f.read()).decode())
    html.append("<p>drive.mp4 next to this file; frames in drive/.</p>")
    html += [tag(p) for p in drive[:: max(1, len(drive) // 8)]]
    html.append("<h2>Level camera (pitch 0) for comparison</h2>")
    for p in poses:
        if p["kind"] == "level":
            tilted = next(q for q in poses if q["kind"] == "room" and q["file"] == p["file"][6:-10] + ".jpg")
            html += [tag(tilted), tag(p), "<br>"]
    html.append("</body></html>")
    with open(os.path.join(VIEWS, "index.html"), "w") as f:
        f.write("\n".join(html))

    ncol = max(sum(1 for p in poses if p["kind"] == "room" and p["room"] == r["id"]) for r in rooms)
    label_w = 150
    tw = min(400, (2400 - label_w) // ncol)
    th = tw * 3 // 4
    sheet = Image.new("RGB", (label_w + ncol * tw, len(rooms) * th), (255, 255, 255))
    sd = ImageDraw.Draw(sheet)
    for i, r in enumerate(rooms):
        sd.text((6, i * th + 6), r["id"], fill=(0, 0, 0), font=font)
        sd.text((6, i * th + 26), "%.1f m2" % r["area_m2"], fill=(80, 80, 80), font=small)
        ps = [p for p in poses if p["kind"] == "room" and p["room"] == r["id"]]
        for j, p in enumerate(ps):
            im = Image.open(os.path.join(VIEWS, p["file"])).resize((tw, th), Image.LANCZOS)
            sheet.paste(im, (label_w + j * tw, i * th))
    sheet.save(os.path.join(VIEWS, "contact_sheet.jpg"), quality=88)
    print("GALLERY index.html, contact_sheet.jpg %dx%d, map_marked.png, drive.mp4" % sheet.size)


def main():
    import argparse
    import subprocess

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "gallery", "all"])
    ap.add_argument("--pitch", type=float, default=20.0, help="camera pitch up, degrees")
    ap.add_argument("--hfov", type=float, default=65.0, help="horizontal field of view, degrees")
    ap.add_argument("--views", type=int, default=5, help="pictures per room")
    ap.add_argument("--drive-rooms", type=int, default=5, help="rooms visited by the drive path")
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    if args.stage in ("plan", "all"):
        out = cmd_plan(args)
    if args.stage == "all":
        t0 = time.time()
        subprocess.run([BLENDER, "-b", "-P", os.path.abspath(__file__), "--", out], check=True)
        print("BLENDER wall time %.1fs" % (time.time() - t0))
    if args.stage in ("gallery", "all"):
        cmd_gallery(args)


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        main()
    else:
        blender_main(sys.argv[sys.argv.index("--") + 1])
