"""Build the ELEGOO Smart Robot Car V4 URDF from the GrabCAD V3 part meshes.

Usage: python build_urdf.py <parts_dir> <out_dir>

<parts_dir> is the output of step_to_parts.py. Writes <out_dir>/elegoo_v4.urdf
and <out_dir>/meshes/*.stl. V3-only parts (Bluetooth module, L298N driver,
18650 holder) are dropped; the V4 camera, battery box and GY-521 IMU are added
as primitives, so their dimensions are estimates, not CAD.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

# CAD frame: millimetres, +Y up, +Z forward, +X left. ROS frame: metres, +X forward, +Y left, +Z up.
CAD_TO_ROS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
BASE_ORIGIN_CAD = np.array([0.0, 11.25, 0.5])  # chassis centre at wheel-axle height

V3_ONLY = {"Bluetooth Module", "Motor Driver", "Battery", "Battery Holder", "Battery Cover"}
PAN_PARTS = {"Sensor Mount Bracket", "Ultrasonic Sensor"}
PAN_HARDWARE = ("Hex Nut_M1.6", "Pan Screw_M1.6")
PAN_AXIS_CAD = np.array([0.0, 57.0, 85.8])  # servo output shaft, top of servo body
# The CAD has the sensor bracket yawed ~18 deg on the servo; undo it so pan = 0 faces forward.
PAN_STRAIGHTEN_RAD = math.atan2(-12.6, 38.7)

MATERIALS = {
    "acrylic": (0.08, 0.08, 0.08, 1),
    "motor": (0.95, 0.80, 0.15, 1),
    "wheel": (0.95, 0.85, 0.25, 1),
    "pcb": (0.10, 0.35, 0.55, 1),
    "blue": (0.15, 0.35, 0.85, 1),
    "metal": (0.70, 0.70, 0.72, 1),
}

WHEELS = {
    "front_left": (58.0, 41.4),
    "front_right": (-58.0, 41.4),
    "rear_left": (58.0, -40.4),
    "rear_right": (-58.0, -40.4),
}
WHEEL_RADIUS = 0.0333
WHEEL_WIDTH = 0.0257
WHEEL_AXLE_Y = 11.25

MASS = {"base": 0.62, "wheel": 0.03, "pan": 0.03}


def to_ros(points_mm, origin_mm):
    return (np.asarray(points_mm) - origin_mm) @ CAD_TO_ROS.T * 0.001


def straighten_pan(points_mm):
    c, s = math.cos(PAN_STRAIGHTEN_RAD), math.sin(PAN_STRAIGHTEN_RAD)
    rot_about_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (np.asarray(points_mm) - PAN_AXIS_CAD) @ rot_about_y.T + PAN_AXIS_CAD


def material_of(name):
    if name in ("Base Plate", "Top Plate", "Dead Plate"):
        return "acrylic"
    if name.startswith("Motor_"):
        return "motor"
    if name.startswith("Wheel_"):
        return "wheel"
    if name in ("Microcontroller", "Expansion Board", "Line Tracking Module"):
        return "pcb"
    if name in ("Servo", "Ultrasonic Sensor"):
        return "blue"
    return "metal"


def bbox_center(part):
    lo, hi = np.array(part["bbox_mm"])
    return (lo + hi) / 2


def inside(point, part, margin=3.0):
    lo, hi = np.array(part["bbox_mm"])
    return bool(np.all(point >= lo - margin) and np.all(point <= hi + margin))


def assign_links(parts):
    removed = [p for p in parts if p["name"] in V3_ONLY]
    groups = {}
    for part in parts:
        name, centre = part["name"], bbox_center(part)
        if name in V3_ONLY:
            continue
        is_fastener = material_of(name) == "metal" and name not in ("Motor Mount Block",)
        if is_fastener and any(inside(centre, r) for r in removed):
            continue
        if name.startswith("Wheel_") or name.startswith("Pan Screw_M2_25mm"):
            link = min(WHEELS, key=lambda w: math.dist((centre[0], centre[2]), WHEELS[w]))
            link = f"{link}_wheel"
        elif name in PAN_PARTS or name.startswith(PAN_HARDWARE):
            link = "sensor_pan"
        else:
            link = "base_link"
        groups.setdefault((link, material_of(name)), []).append(part)
    return groups


def link_origin_cad(link):
    if link.endswith("_wheel"):
        x, z = WHEELS[link.removesuffix("_wheel")]
        return np.array([x, WHEEL_AXLE_Y, z])
    if link == "sensor_pan":
        return PAN_AXIS_CAD
    return BASE_ORIGIN_CAD


def write_meshes(parts_dir, groups, mesh_dir):
    mesh_dir.mkdir(parents=True, exist_ok=True)
    visuals = {}
    for (link, material), parts in sorted(groups.items()):
        meshes = [trimesh.load(parts_dir / p["stl"], force="mesh") for p in parts]
        merged = trimesh.util.concatenate(meshes)
        vertices = straighten_pan(merged.vertices) if link == "sensor_pan" else merged.vertices
        merged.vertices = to_ros(vertices, link_origin_cad(link))
        fname = f"{link}_{material}.stl"
        merged.export(mesh_dir / fname)
        visuals.setdefault(link, []).append((fname, material))
    return visuals


def xyz(v):
    return " ".join(f"{c:.5f}" for c in v)


def box_inertia(m, sx, sy, sz):
    return (m * (sy**2 + sz**2) / 12, m * (sx**2 + sz**2) / 12, m * (sx**2 + sy**2) / 12)


def inertial(mass, com, ixx, iyy, izz):
    return (
        f'    <inertial>\n      <origin xyz="{xyz(com)}"/>\n      <mass value="{mass}"/>\n'
        f'      <inertia ixx="{ixx:.3e}" ixy="0" ixz="0" iyy="{iyy:.3e}" iyz="0" izz="{izz:.3e}"/>\n'
        "    </inertial>\n"
    )


def mesh_visuals(visuals):
    return "".join(
        f'    <visual>\n      <geometry><mesh filename="meshes/{f}"/></geometry>\n'
        f'      <material name="{m}"/>\n    </visual>\n'
        for f, m in visuals
    )


def box_element(tag, centre_cad, size_cad, origin_cad, material=None):
    centre = to_ros(centre_cad, origin_cad)
    size = np.abs(CAD_TO_ROS @ np.asarray(size_cad, dtype=float)) * 0.001
    mat = f'\n      <material name="{material}"/>' if material else ""
    return (
        f'    <{tag}>\n      <origin xyz="{xyz(centre)}"/>\n'
        f'      <geometry><box size="{xyz(size)}"/></geometry>{mat}\n    </{tag}>\n'
    )


def build(parts_dir, out_dir):
    parts = json.loads((parts_dir / "parts.json").read_text())
    visuals = write_meshes(parts_dir, assign_links(parts), out_dir / "meshes")
    base, pan = BASE_ORIGIN_CAD, PAN_AXIS_CAD

    out = ['<?xml version="1.0"?>\n<robot name="elegoo_smart_robot_car_v4">\n']
    for name, rgba in MATERIALS.items():
        out.append(f'  <material name="{name}"><color rgba="{xyz(rgba)}"/></material>\n')
    out.append('  <material name="camera"><color rgba="0.05 0.05 0.05 1"/></material>\n')

    out.append('  <link name="base_footprint"/>\n')
    out.append('  <joint name="base_footprint_joint" type="fixed">\n'
               '    <parent link="base_footprint"/>\n    <child link="base_link"/>\n'
               f'    <origin xyz="0 0 {WHEEL_RADIUS:.5f}"/>\n  </joint>\n')

    base_size = (0.232, 0.132, 0.075)
    out.append('  <link name="base_link">\n')
    out.append(mesh_visuals(visuals["base_link"]))
    out.append(box_element("visual", (1.0, 56.9, -75.0), (89.6, 25.0, 42.0), base, "acrylic"))
    out.append(box_element("collision", (0, 20.0, 0), (132.2, 49.2, 231.9), base))
    out.append(box_element("collision", (0, -12.6, 92.0), (45.0, 11.5, 35.0), base))
    out.append(box_element("collision", (-2.0, 57.0, -30.0), (90.0, 25.0, 150.0), base))
    out.append(inertial(MASS["base"], to_ros((0, 25.0, -5.0), base), *box_inertia(MASS["base"], *base_size)))
    out.append("  </link>\n")

    for wheel, (x, z) in WHEELS.items():
        link = f"{wheel}_wheel"
        m = MASS["wheel"]
        i_axle = m * WHEEL_RADIUS**2 / 2
        i_diam = m * (3 * WHEEL_RADIUS**2 + WHEEL_WIDTH**2) / 12
        out.append(f'  <link name="{link}">\n')
        out.append(mesh_visuals(visuals[link]))
        out.append('    <collision>\n      <origin rpy="1.5708 0 0"/>\n'
                   f'      <geometry><cylinder radius="{WHEEL_RADIUS}" length="{WHEEL_WIDTH}"/></geometry>\n'
                   "    </collision>\n")
        out.append(inertial(m, (0, 0, 0), i_diam, i_axle, i_diam))
        out.append("  </link>\n")
        out.append(f'  <joint name="{wheel}_wheel_joint" type="continuous">\n'
                   f'    <parent link="base_link"/>\n    <child link="{link}"/>\n'
                   f'    <origin xyz="{xyz(to_ros((x, WHEEL_AXLE_Y, z), base))}"/>\n'
                   '    <axis xyz="0 1 0"/>\n    <limit effort="0.1" velocity="21"/>\n  </joint>\n')

    out.append('  <link name="sensor_pan">\n')
    out.append(mesh_visuals(visuals["sensor_pan"]))
    out.append(box_element("visual", (0.0, 98.0, 78.0), (28.0, 40.0, 8.0), pan, "camera"))
    out.append(box_element("visual", (0.0, 96.0, 74.5), (34.0, 45.0, 3.0), pan, "acrylic"))
    out.append(box_element("collision", (0.0, 85.0, 92.0), (46.0, 50.0, 38.0), pan))
    out.append(inertial(MASS["pan"], to_ros((0.0, 85.0, 90.0), pan), *box_inertia(MASS["pan"], 0.04, 0.046, 0.05)))
    out.append("  </link>\n")
    out.append('  <joint name="sensor_pan_joint" type="revolute">\n'
               '    <parent link="base_link"/>\n    <child link="sensor_pan"/>\n'
               f'    <origin xyz="{xyz(to_ros(pan, base))}"/>\n    <axis xyz="0 0 1"/>\n'
               '    <limit lower="-1.5708" upper="1.5708" effort="0.18" velocity="5.2"/>\n  </joint>\n')

    for name, parent, pos_cad, origin in (
        ("ultrasonic_link", "sensor_pan", (0.0, 79.5, 110.5), pan),
        ("camera_link", "sensor_pan", (0.0, 104.0, 83.0), pan),
        ("imu_link", "base_link", (-20.0, 80.0, 20.0), base),
        ("line_tracker_link", "base_link", (0.0, -18.4, 92.0), base),
    ):
        out.append(f'  <link name="{name}"/>\n')
        out.append(f'  <joint name="{name}_joint" type="fixed">\n'
                   f'    <parent link="{parent}"/>\n    <child link="{name}"/>\n'
                   f'    <origin xyz="{xyz(to_ros(pos_cad, origin))}"/>\n  </joint>\n')
    out.append('  <link name="camera_optical_link"/>\n'
               '  <joint name="camera_optical_joint" type="fixed">\n'
               '    <parent link="camera_link"/>\n    <child link="camera_optical_link"/>\n'
               '    <origin rpy="-1.5708 0 -1.5708"/>\n  </joint>\n')

    out.append("</robot>\n")
    (out_dir / "elegoo_v4.urdf").write_text("".join(out))
    print(f"wrote {out_dir / 'elegoo_v4.urdf'}")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
