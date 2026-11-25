"""Building experiments on the L4: smoke, views, drive, collide, bench.

Push a folder holding elegoo/ (with nav/) and scene/ side by side, then for example
  sim/cloud/run_headless.sh integ/elegoo/nav/run.py views --poses /workspace/elegoo/integ/poses_rendered.json \
      --output /workspace/elegoo/integ/out --viz none
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxCfg

wp.config.enable_backward = False
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def save_rgb(image, path: Path) -> None:
    from PIL import Image

    Image.fromarray(image).save(path, quality=92)


def save_depth_mm(depth, path: Path) -> None:
    import numpy as np
    from PIL import Image

    mm = np.where(np.isfinite(depth), np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)
    Image.fromarray(mm).save(path)


def frame(obs: dict, i: int = 0) -> tuple:
    rgb = obs["rgb"][i].cpu().numpy() if "rgb" in obs else None
    depth = obs["depth"][i, ..., 0].cpu().numpy() if "depth" in obs else None
    return rgb, depth


def hold(env, steps: int):
    import torch

    zero = torch.zeros((env.spec.count, 2), device=env.device)
    for _ in range(steps):
        obs = env.step(zero)
    return obs


def lens_position(env):
    import torch
    from isaaclab.utils.math import quat_apply

    from elegoo.world import PAN_PIVOT

    camera = env.spec.camera
    pan = env.robot.data.body_link_pose_w.torch[:, env.robot.find_bodies("sensor_pan")[0][0]]
    offset = torch.tensor([camera.forward_m - PAN_PIVOT[0], 0.0, camera.height_m - PAN_PIVOT[2]], device=env.device)
    return pan[:, :3] + quat_apply(pan[:, 3:], offset.expand(len(pan), 3))


def smoke(env, out: Path) -> None:
    """Cars settle on the collider; car 1 parked 1 m ahead of car 0 must not appear in car 0's camera."""
    import torch

    hold(env, 20)
    p = env.privileged()
    print("SMOKE settled_z", [round(v, 4) for v in p["pos_w"][:, 2].tolist()], "off_map", p["off_map"].tolist(),
          flush=True)
    ids = torch.arange(env.spec.count, device=env.device)
    xy = env.spawns[:env.spec.count, :2].clone()
    yaw = env.spawns[:env.spec.count, 2].clone()
    xy[0] = torch.tensor([-5.99, 5.17], device=env.device)
    yaw[0] = math.radians(5.0)
    xy[1] = xy[0] + 1.0 * torch.stack((yaw[0].cos(), yaw[0].sin()))
    yaw[1] = yaw[0]
    env.place(ids, xy, yaw)
    obs = hold(env, 12)
    closures = [p for p in env.sim.stage.Traverse() if str(p.GetPath()).startswith("/World/ClosureColliders/")]
    print("SMOKE closure_colliders", len(closures), flush=True)
    for i in range(min(2, env.spec.count)):
        rgb, depth = frame(obs, i)
        save_rgb(rgb, out / f"smoke_{i}_rgb.jpg")
        save_depth_mm(depth, out / f"smoke_{i}_depth_mm.png")
        print(f"SMOKE car{i} pos={env.privileged()['pos_w'][i].tolist()} lens={lens_position(env)[i].tolist()} "
              f"range_cm={obs['range_cm'][i].item()} depth_min={float(depth[depth > 0].min()):.3f} "
              f"rgb_mean={rgb.mean():.1f}", flush=True)


def views(env, out: Path, poses: list[dict], variants: int = 0) -> None:
    """Each pose once; in the visual scene first the canonical look (suffix _after), then `variants` randomised
    building looks (_var1..) with fresh per-car lens jitter and image augmentation at every pose."""
    import torch

    ids = torch.arange(1, device=env.device)
    hold(env, 30)
    app = env.appearance
    draws = [None] if app is None else [0] + list(range(1, variants + 1))
    log = []
    for draw in draws:
        suffix = "" if draw is None else ("_after" if draw == 0 else f"_var{draw}")
        if app is not None:
            record = app.draw_scene(canonical=draw == 0)
            print("DRAW", suffix, json.dumps({k: v for k, v in record.items() if k != "slots"}), flush=True)
            hold(env, 20)
        for k, pose in enumerate(poses):
            stem = Path(pose["file"]).with_suffix("")
            target = out / "views" / stem.parent
            target.mkdir(parents=True, exist_ok=True)
            env.place(ids, torch.tensor([[pose["x"], pose["y"]]], device=env.device),
                      torch.tensor([math.radians(pose["yaw"])], device=env.device))
            if app is not None:
                (app.neutral if draw == 0 else app.reset)(ids)
            obs = hold(env, 4)
            rgb, depth = frame(obs)
            save_rgb(rgb, target / f"{stem.name}{suffix}_rgb.jpg")
            if depth is not None:
                save_depth_mm(depth, target / f"{stem.name}{suffix}_depth_mm.png")
            log.append({"file": pose["file"], "variant": suffix,
                        "camera_xyz": [round(c, 4) for c in lens_position(env)[0].tolist()],
                        "scan_camera_z": pose["z"], "range_cm": obs["range_cm"][0].item(),
                        "rgb_mean": round(float(rgb.mean()), 1),
                        "depth_valid": None if depth is None else round(float((depth > 0).mean()), 3)})
            if k % 25 == 0:
                print("VIEW", k, log[-1], flush=True)
    (out / "views" / f"views_log_{len(poses)}.json").write_text(json.dumps(log, indent=1))
    print("VIEWS_DONE", len(log), flush=True)


def drive(env, out: Path) -> None:
    """Pure pursuit on the planned waypoints, steering from the true pose; wheel commands only."""
    import torch

    plan = json.loads((HERE / "building_plan.json").read_text())["drive"]
    way = torch.tensor(plan["waypoints"], device=env.device)
    ids = torch.arange(1, device=env.device)
    env.place(ids, way[:1], torch.tensor([plan["start_yaw"]], device=env.device))
    hold(env, 10)
    frames = out / "drive_frames"
    frames.mkdir(parents=True, exist_ok=True)
    cumulative = torch.cat((torch.zeros(1, device=env.device), (way[1:] - way[:-1]).norm(dim=1).cumsum(0)))
    rows, progress, best, best_t, stuck, recover_until = [], 0, 0.0, 0.0, [], -1.0
    t, dt, limit_s = 0.0, env.spec.decimation * env.dt, 260.0
    while t < limit_s:
        p = env.privileged()
        xy, heading = p["pos_w"][0, :2], p["heading_rad"][0]
        window = slice(progress, min(progress + 12, len(way)))
        progress += int((way[window] - xy).norm(dim=1).argmin())
        along = cumulative[progress].item()
        if along > best + 0.05:
            best, best_t = along, t
        if (way[-1] - xy).norm() < 0.2:
            break
        target = way[min(progress + 2, len(way) - 1)]
        error = torch.atan2(target[1] - xy[1], target[0] - xy[0]) - heading
        error = math.atan2(math.sin(error), math.cos(error))
        if t < recover_until:
            left = right = -90.0
        elif t - best_t > 3.0:
            stuck.append({"t": round(t, 2), "x": round(xy[0].item(), 3), "y": round(xy[1].item(), 3),
                          "along_m": round(along, 2)})
            if len(stuck) >= 6:
                break
            recover_until, best_t = t + 0.7, t + 0.7
            left = right = -90.0
        elif abs(error) > math.radians(40):
            spin = 85.0 if error > 0 else -85.0
            left, right = -spin, spin
        else:
            turn = max(-40.0, min(40.0, 70.0 * error))
            left, right = 100.0 - turn, 100.0 + turn
        obs = env.step(torch.tensor([[left, right]], device=env.device) / 255)
        save_rgb(frame(obs)[0], frames / f"{len(rows):05d}.jpg")
        p = env.privileged()
        rows.append((round(t, 3), *[round(v, 4) for v in p["pos_w"][0].tolist()], round(heading.item(), 4), left,
                     right, obs["range_cm"][0].item(), int(obs["on_ground"][0].item()),
                     int(p["off_map"][0].item()), progress, round(along, 3)))
        t += dt
    with (out / "drive_log.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("t", "x", "y", "z", "heading", "pwm_left", "pwm_right", "range_cm", "on_ground",
                         "off_map", "waypoint", "along_m"))
        writer.writerows(rows)
    final = env.privileged()["pos_w"][0]
    result = {"reached": bool((way[-1] - final[:2]).norm() < 0.2), "sim_seconds": round(t, 2), "frames": len(rows),
              "planned_m": plan["length_m"], "progress_m": round(best, 2), "stuck": stuck,
              "off_map_steps": sum(r[9] for r in rows), "final_xy": [round(v, 3) for v in final[:2].tolist()]}
    (out / "drive_result.json").write_text(json.dumps(result, indent=1))
    print("DRIVE", json.dumps(result), flush=True)


def collide(env, out: Path) -> None:
    """Four cars at once: into a wall, into furniture, into a closure wall, and across the lifted patch."""
    import torch

    runs = json.loads((HERE / "building_plan.json").read_text())["collide"]
    names = ["wall", "furniture", "closure", "lifted"]
    ids = torch.arange(len(names), device=env.device)
    env.place(ids, torch.tensor([[runs[n]["x"], runs[n]["y"]] for n in names], device=env.device),
              torch.tensor([runs[n]["yaw"] for n in names], device=env.device))
    obs = hold(env, 10)
    for i, name in enumerate(names):
        save_rgb(frame(obs, i)[0], out / f"collide_{name}_start_rgb.jpg")
    start = env.privileged()["pos_w"][:, :2].clone()
    yaw0 = env.sensors.yaw_deg.clone()
    heading0 = env.privileged()["heading_rad"].clone()
    stop_after = torch.tensor([9.0, 9.0, 9.0, 1.9 / 0.39], device=env.device)
    trace = {n: [] for n in names}
    t, dt = 0.0, env.spec.decimation * env.dt
    while t < 7.0:
        correction = (-0.9 * (env.sensors.yaw_deg - yaw0)).clamp(-25, 25)
        pwm = torch.stack((110 - correction, 110 + correction), dim=1)
        pwm[t >= stop_after] = 0
        obs = env.step(pwm / 255)
        p = env.privileged()
        x, y, z, w = env.robot.data.root_link_pose_w.torch[:, 3:].unbind(-1)
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        speed = env.robot.data.root_com_vel_w.torch[:, :2].norm(dim=1)
        forward = torch.stack((heading0.cos(), heading0.sin()), dim=1)
        travel = ((p["pos_w"][:, :2] - start) * forward).sum(1)
        for i, name in enumerate(names):
            trace[name].append([round(t, 3), round(travel[i].item(), 4), round(p["pos_w"][i, 2].item(), 4),
                                round(speed[i].item(), 4), round(math.degrees(pitch[i].item()), 2),
                                round(math.degrees(roll[i].item()), 2), obs["range_cm"][i].item(),
                                int(obs["on_ground"][i].item()), int(p["off_map"][i].item())])
        t += dt
    for i, name in enumerate(names):
        save_rgb(frame(obs, i)[0], out / f"collide_{name}_end_rgb.jpg")
    summary = {}
    for name in names:
        rows = trace[name]
        travel = [r[1] for r in rows]
        summary[name] = {
            "obstacle_at_m": runs[name].get("collider_hit_m", runs[name].get("wall_distance_m")),
            "peak_travel_m": max(travel), "final_travel_m": travel[-1], "final_speed_m_s": rows[-1][3],
            "z_range_m": [min(r[2] for r in rows), max(r[2] for r in rows)],
            "max_abs_pitch_deg": max(abs(r[4]) for r in rows), "max_abs_roll_deg": max(abs(r[5]) for r in rows),
            "always_on_ground": all(r[7] for r in rows), "ever_off_map": any(r[8] for r in rows),
            "final_range_cm": rows[-1][6],
        }
        print("COLLIDE", name, json.dumps(summary[name]), flush=True)
    (out / "collide_result.json").write_text(json.dumps({"summary": summary, "trace_columns": [
        "t", "travel_m", "z", "speed", "pitch_deg", "roll_deg", "range_cm", "on_ground", "off_map"],
        "trace": trace}, indent=1))


def bench(env, args: argparse.Namespace, out: Path) -> None:
    import torch

    def used_mib() -> float:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2**20

    action = torch.full((args.num_envs, 2), 80 / 255, device=env.device)
    for _ in range(10):
        env.step(action)
    torch.cuda.synchronize()
    peak = used_mib()
    start = time.perf_counter()
    for i in range(args.steps):
        env.step(action)
        if i % 10 == 0:
            peak = max(peak, used_mib())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    row = dict(envs=args.num_envs, camera=f"{args.width}x{args.height}",
               outputs="+".join(env.spec.camera.outputs) or "none", steps=args.steps, seconds=round(elapsed, 3),
               vector_steps_s=round(args.steps / elapsed, 2), env_steps_s=round(args.num_envs * args.steps / elapsed, 1),
               physics_ticks_s=round(args.num_envs * args.steps * env.spec.decimation / elapsed, 1),
               device_used_mib=round(peak), torch_max_alloc_mib=round(torch.cuda.max_memory_allocated() / 2**20),
               scene="visual" if env.appearance is not None else "scan", scene_draw_s=None)
    if env.appearance is not None:
        torch.cuda.synchronize()
        begin = time.perf_counter()
        env.appearance.draw_scene()
        env.step(action)
        torch.cuda.synchronize()
        row["scene_draw_s"] = round(time.perf_counter() - begin, 3)
    path = out / "bench.csv"
    new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print("BENCH", json.dumps(row), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "views", "drive", "collide", "bench"))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--outputs", default="rgb,depth")
    parser.add_argument("--pitch", type=float, default=20.0)
    parser.add_argument("--hfov", type=float, default=65.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ambient", type=float, default=4.0)
    parser.add_argument("--poses", type=Path, default=None, help="poses_rendered.json for views")
    parser.add_argument("--kinds", default="room,drive", help="pose kinds to render in views mode")
    parser.add_argument("--visual", action="store_true", help="textured visual scene with appearance randomisation")
    parser.add_argument("--variants", type=int, default=3, help="randomised looks per pose in visual views")
    parser.add_argument("--visual-period", type=int, default=250, help="steps between look changes (not in views)")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    outputs = tuple(o for o in args.outputs.split(",") if o)
    args.enable_cameras = bool(outputs)
    cfg = SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg())
    with launch_simulation(cfg, args):
        from elegoo.nav.env import BuildingNavEnv, NavSpec
        from elegoo.world import CameraSpec

        args.output.mkdir(parents=True, exist_ok=True)
        count = {"views": 1, "drive": 1, "collide": 4}.get(args.mode, args.num_envs)
        camera = CameraSpec(outputs=outputs, width=args.width, height=args.height,
                            pitch_deg=args.pitch, hfov_deg=args.hfov, far_clip_m=20.0)
        started = time.perf_counter()
        period = 0 if args.mode == "views" else args.visual_period
        env = BuildingNavEnv(SimulationContext(cfg), NavSpec(count=count, camera=camera, ambient_light=args.ambient,
                                                             visual=args.visual, visual_period=period))
        print(f"SCENE_READY envs={count} seconds={time.perf_counter() - started:.1f}", flush=True)
        if args.mode == "smoke":
            smoke(env, args.output)
        elif args.mode == "views":
            kinds = set(args.kinds.split(","))
            views(env, args.output, [p for p in json.loads(args.poses.read_text())["poses"] if p["kind"] in kinds],
                  args.variants)
        elif args.mode == "drive":
            drive(env, args.output)
        elif args.mode == "collide":
            collide(env, args.output)
        else:
            bench(env, args, args.output)


if __name__ == "__main__":
    main()
