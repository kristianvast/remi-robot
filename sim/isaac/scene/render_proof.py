"""Proof renders of building.usda from robot-camera poses, shell off and shell on.

  sim/.venv/bin/python sim/isaac/scene/render_proof.py

Plans 12 poses (sim venv), renders each twice in Blender (shell objects hidden,
then shown) and writes proof/NN_<area>_<kind>_shell_{off,on}.jpg, proof/poses.json
and proof/pairs.jpg. Camera: 0.126 m above the floor under it, pitched 20 deg up,
65 deg horizontal FOV, 640x480. Magenta background = nothing rendered there.
Yaw is measured from +X toward +Y.
"""

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.join(HERE, "proof")
STAGE = os.path.join(HERE, "building.usda")
BLENDER = "/opt/homebrew/bin/blender"
CAM_HEIGHT_M, PITCH_DEG, HFOV_DEG, WIDTH, HEIGHT = 0.126, 20.0, 65.0, 640, 480
HOLE_MAGENTA = (0.55, 0.0, 0.55)
N_DOORWAY, N_TOTAL = 6, 12


def blender_main(poses_path):
    import bpy
    from mathutils import Vector

    with open(poses_path) as f:
        plan = json.load(f)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=plan["stage"])
    scene = bpy.context.scene
    shell = [o for o in scene.objects if o.type == "MESH" and o.name.startswith("shell_")]
    scan = [o for o in scene.objects if o.type == "MESH" and not o.name.startswith("shell_")]
    print("PROOF scan objects", [o.name for o in scan], "shell objects", len(shell))
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                mat.node_tree.nodes.active = node
            if node.type == "BSDF_PRINCIPLED":
                mat.diffuse_color = tuple(node.inputs["Base Color"].default_value)

    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "TEXTURE"
    shading.show_backface_culling = False
    shading.show_shadows = False
    shading.show_cavity = True
    shading.cavity_type = "WORLD"
    shading.cavity_ridge_factor = 0.0
    shading.cavity_valley_factor = 1.0
    scene.display.render_aa = "8"
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("bg")
    world.color = HOLE_MAGENTA
    scene.world = world
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90
    cam_data = bpy.data.cameras.new("car_cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(HFOV_DEG)
    cam_data.clip_start, cam_data.clip_end = 0.01, 60.0
    cam = bpy.data.objects.new("car_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    def set_shell(visible):
        for o in shell:
            o.hide_render = not visible
            o.hide_viewport = not visible

    set_shell(False)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for p in plan["poses"]:
        hit, loc, *_rest = scene.ray_cast(depsgraph, Vector((p["x"], p["y"], 0.35)), Vector((0, 0, -1)), distance=0.8)
        p["floor_z"] = round(loc.z, 4) if hit and -0.15 < loc.z < 0.25 else 0.0
    for p in plan["poses"]:
        cam.location = (p["x"], p["y"], p["floor_z"] + CAM_HEIGHT_M)
        cam.rotation_euler = (math.radians(90.0 + PITCH_DEG), 0.0, math.radians(p["yaw_deg"] - 90.0))
        for visible, suffix in ((False, "off"), (True, "on")):
            set_shell(visible)
            scene.render.filepath = os.path.join(plan["out"], "%s_shell_%s.jpg" % (p["stem"], suffix))
            bpy.ops.render.render(write_still=True)
    with open(poses_path, "w") as f:
        json.dump(plan, f, indent=1)
    print("PROOF rendered %d pairs" % len(plan["poses"]))


def line_of_sight(free, c0, r0, c1, r1):
    import numpy as np

    n = int(max(abs(c1 - c0), abs(r1 - r0))) + 1
    cs = np.linspace(c0, c1, n).astype(int)
    rs = np.linspace(r0, r1, n).astype(int)
    return bool(free[rs, cs].all())


def plan_poses():
    import numpy as np
    from scipy import ndimage

    sys.path.insert(0, HERE)
    import scene_analysis as sa

    grid = sa.Grid()
    rooms = grid.rooms()
    lab = grid.room_labels(rooms)
    clear = ndimage.distance_transform_edt(grid.free) * grid.res
    ids = [r["id"] for r in rooms]

    doors = []
    for a in range(1, len(rooms) + 1):
        ma = lab == a
        for b in range(1, len(rooms) + 1):
            if a == b:
                continue
            touch = ma & ndimage.binary_dilation(lab == b, structure=np.ones((3, 3)))
            if touch.sum() < 10:
                continue
            tl, n = ndimage.label(ndimage.binary_dilation(touch, iterations=2))
            sizes = np.bincount(tl.ravel())[1:]
            big = tl == (int(np.argmax(sizes)) + 1)
            rr, cc = np.nonzero(big & touch)
            doors.append(dict(a=a, b=b, col=float(cc.mean()), row=float(rr.mean()), width_px=int(len(rr))))

    poses, used_areas = [], set()

    def best_view(room_idx, target, dmin=1.5, dmax=3.5, thr=0.25):
        rr, cc = np.nonzero((lab == room_idx) & (clear >= thr))
        if len(rr) == 0:
            return None
        d = np.hypot(cc - target[0], rr - target[1]) * grid.res
        cand = np.flatnonzero((d >= dmin) & (d <= dmax))
        order = cand[np.argsort(-clear[rr[cand], cc[cand]])]
        for j in order[:400]:
            if line_of_sight(grid.free, cc[j], rr[j], int(target[0]), int(target[1])):
                return int(cc[j]), int(rr[j])
        return None

    for door in sorted(doors, key=lambda d: -d["width_px"]):
        if len([p for p in poses if p["kind"] == "doorway"]) >= N_DOORWAY:
            break
        if ids[door["a"] - 1] in used_areas:
            continue
        spot = best_view(door["a"], (door["col"], door["row"]))
        if spot is None:
            continue
        c, r = spot
        x, y = grid.to_world(c + 0.5, r + 0.5)
        tx, ty = grid.to_world(door["col"] + 0.5, door["row"] + 0.5)
        poses.append(dict(kind="doorway", area=ids[door["a"] - 1], toward=ids[door["b"] - 1], x=round(float(x), 3),
                          y=round(float(y), 3), yaw_deg=round(math.degrees(math.atan2(ty - y, tx - x)), 1),
                          target_xy=[round(float(tx), 3), round(float(ty), 3)]))
        used_areas.add(ids[door["a"] - 1])

    def free_ahead(c, r, yaw):
        for i in range(1, 400):
            cc, rr = int(c + math.cos(yaw) * i), int(r - math.sin(yaw) * i)
            if not (0 <= cc < grid.w and 0 <= rr < grid.h) or not grid.free[rr, cc]:
                return i * grid.res
        return 400 * grid.res

    remaining = [i for i, rid in enumerate(ids) if rid not in used_areas] + list(range(len(ids)))
    for i in remaining:
        if len(poses) >= N_TOTAL:
            break
        rr, cc = np.nonzero(lab == i + 1)
        if len(rr) == 0:
            continue
        j = int(np.argmax(clear[rr, cc]))
        c, r = int(cc[j]), int(rr[j])
        taken = [p for p in poses if p["area"] == ids[i]]
        yaws = np.radians(np.arange(0, 360, 10))
        score = [free_ahead(c, r, a) - 5.0 * any(abs((math.degrees(a) - p["yaw_deg"] + 180) % 360 - 180) < 60
                                                  for p in taken) for a in yaws]
        a = float(yaws[int(np.argmax(score))])
        x, y = grid.to_world(c + 0.5, r + 0.5)
        poses.append(dict(kind="room", area=ids[i], x=round(float(x), 3), y=round(float(y), 3),
                          yaw_deg=round(math.degrees(a), 1)))
    for k, p in enumerate(poses):
        p["stem"] = "%02d_%s_%s" % (k + 1, p["area"], p["kind"])
    os.makedirs(PROOF, exist_ok=True)
    out = os.path.join(PROOF, "poses.json")
    with open(out, "w") as f:
        json.dump(dict(stage=STAGE, out=PROOF, cam_height_m=CAM_HEIGHT_M, pitch_deg=PITCH_DEG, hfov_deg=HFOV_DEG,
                       size=[WIDTH, HEIGHT], poses=poses), f, indent=1)
    for p in poses:
        print("PLAN", p["stem"], p.get("toward", ""), p["x"], p["y"], p["yaw_deg"])
    print("PLAN areas", sorted({p["area"] for p in poses}))
    return out


def contact_sheet(poses_path):
    from PIL import Image, ImageDraw, ImageFont

    with open(poses_path) as f:
        plan = json.load(f)
    tw, th = 480, 360
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    poses = plan["poses"]
    sheet = Image.new("RGB", (2 * tw, len(poses) * (th + 22)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(poses):
        y0 = i * (th + 22)
        label = "%s  x=%.2f y=%.2f yaw=%.0f%s" % (p["stem"], p["x"], p["y"], p["yaw_deg"],
                                                   ("  toward " + p["toward"]) if "toward" in p else "")
        d.text((6, y0 + 3), label + "   | left: shell off, right: shell on", fill=(0, 0, 0), font=font)
        for j, suffix in enumerate(("off", "on")):
            im = Image.open(os.path.join(PROOF, "%s_shell_%s.jpg" % (p["stem"], suffix))).resize((tw, th))
            sheet.paste(im, (j * tw, y0 + 22))
    sheet.save(os.path.join(PROOF, "pairs.jpg"), quality=85)


def main():
    out = plan_poses()
    subprocess.run([BLENDER, "-b", "-P", os.path.abspath(__file__), "--", out], check=True)
    contact_sheet(out)
    print("PROOF done ->", PROOF)


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        main()
    else:
        blender_main(sys.argv[sys.argv.index("--") + 1])
