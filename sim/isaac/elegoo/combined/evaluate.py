"""Held-out evaluation of the combined model next to driver_v2 and the privileged teacher, 100 ms steps.

Groups (condition/controller): nominal|stress x <student>|v2, nominal/teacher. A student controller is any name
other than v2 / teacher, matched to `--heads name=path` (a bare path is the controller `student`); several students
in one run share routes, seeds and the camera link draws (p7: p4 vs combined2 side by side). Student "stress" = the camera link at its
worst (0.25 s latency, 14 % drops in bursts, JPEG q60, long exposure, sonar missing 75 %); v2 "stress" = the depth
spec's stress mode (train/README.md). Students need the RTX camera (visual scene); v2 / teacher run without it.

  sim/cloud/run_headless.sh elegoo-sim-9 p4c/sim/isaac/elegoo/combined/evaluate.py --groups nominal/student,stress/student \
      --heads /workspace/elegoo/p4c/run/heads.pt --output /workspace/elegoo/p4c/eval_student --viz none
Outputs: episodes.csv, place.npz (per student step: head outputs + truth), record.npz (localiser inputs + frames
for --record-per-area student nominal cars).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[4] / "pylib"))
from elegoo.combined.common import Encoder
from elegoo.train.evaluate import select_routes


def run(args, sim) -> None:
    from elegoo.combined.env import V2, CombinedEnv, Link, act, encode, load_heads, student_inputs
    from elegoo.train.driver_env import RunSpec
    from elegoo.world import CameraSpec

    torch.manual_seed(args.seed)
    started = time.monotonic()
    groups = [tuple(g.split("/")) for g in args.groups.split(",")]
    with np.load(Path(__file__).resolve().parents[1] / "train/routes.npz") as data:
        areas = data[f"{args.split}_area"]
    routes = select_routes(areas, args.per_area)
    count = len(groups) * len(routes)
    head_paths = dict(h.split("=", 1) if "=" in h else ("student", h) for h in args.heads)
    students = any(k not in ("v2", "teacher") for _, k in groups)
    camera = CameraSpec(outputs=("rgb",), width=args.rgb_size[0], height=args.rgb_size[1]) if students \
        else CameraSpec(outputs=())
    env = CombinedEnv(sim, RunSpec(count=count, randomize=False, seed=args.seed, split=args.split, camera=camera,
                                   visual=students, step_ticks=(10,)))
    device = env.device
    group_of = torch.arange(count, device=device) // len(routes)
    route_of = torch.as_tensor(np.tile(routes, len(groups)), device=device)
    kind = [groups[g][1] for g in group_of.tolist()]
    cond = [groups[g][0] for g in group_of.tolist()]
    is_student = torch.tensor([k not in ("v2", "teacher") for k in kind], device=device)
    is_v2 = torch.tensor([k == "v2" for k in kind], device=device)
    stress = torch.tensor([c == "stress" for c in cond], device=device)
    env.set_conditions(torch.ones(count, device=device), stress & is_v2)
    env.reset(env.ids, route_of)
    link = Link(count, device, stress=stress & is_student, keep_rgb=True, real_path=args.real_path)
    encoder = Encoder.pretrained(args.da2).to(device).half().eval().requires_grad_(False) if students else None
    heads = {name: load_heads(Path(head_paths[name]), device).eval()
             for name in sorted({k for k in kind if k not in ("v2", "teacher")})}
    head_of = {name: torch.tensor([k == name for k in kind], device=device) for name in heads}
    v2 = V2(args.teacher, count, device)
    v2_ids, teacher_ids = is_v2.nonzero().flatten(), torch.tensor(
        [i for i, k in enumerate(kind) if k == "teacher"], dtype=torch.long, device=device)
    record = torch.tensor([i for i in range(count) if is_student[i] and cond[i] == "nominal"
                           and int(np.sum(areas[routes][: i % len(routes)] == areas[routes][i % len(routes)]))
                           < args.record_per_area], dtype=torch.long, device=device)
    print(f"EVAL_READY cars={count} routes={len(routes)} groups={groups} record={len(record)} "
          f"startup_s={time.monotonic() - started:.1f}", flush=True)
    env.advance(torch.zeros((count, 2), device=device))
    done = torch.zeros(count, dtype=torch.bool, device=device)
    keys = ("success", "collision", "kind", "timeout", "travel_m", "time_s", "peak_m_s", "contacts", "near_s",
            "min_clearance")
    outcome = {k: torch.zeros(count, device=device) for k in keys}
    place_rows, rec = [], {k: [] for k in ("delivered", "depth", "rgb", "range_cm", "gyro_heading", "pwm", "pose",
                                           "room_p", "pose_hat", "done")}
    step = 0
    while not done.all() and step < args.max_steps:
        depth, aux = env.policy_obs()
        truth = env.truth()
        action = torch.zeros((count, 2), device=device)
        if len(v2_ids):
            action[v2_ids] = v2(depth, aux, v2_ids)[v2_ids]
        if len(teacher_ids):
            aux_t = env.teacher_aux(aux, truth["xy"], truth["yaw"])
            action[teacher_ids] = v2(encode(env.true_depth())[:, None], aux_t, teacher_ids)[teacher_ids]
        if students:
            link.capture(encoder, env.frame_rgb, torch.deg2rad(aux[:, 1] * 250))
            link.deliver(env.control_dt)
            aux_s = student_inputs(env, link, aux, truth["xy"], truth["yaw"])
            for m, (name, model) in enumerate(heads.items()):
                pwm_m, (_, room_m, pose_m, goal_m) = act(model, link, aux_s)
                if m == 0:
                    pwm_s, room, pose, goal = pwm_m, room_m, pose_m, goal_m
                else:
                    mine = head_of[name]
                    pwm_s = torch.where(mine[:, None], pwm_m, pwm_s)
                    room = torch.where(mine[:, None], room_m, room)
                    pose = torch.where(mine[:, None], pose_m, pose)
                    goal = torch.where(mine[:, None, None], goal_m, goal)
            action = torch.where(is_student[:, None], pwm_s, action)
            prob = room.softmax(1)
            rel = (goal * truth["goal"]).sum(-1).clamp(-1, 1).acos()
            sid = is_student & ~done
            place_rows.append(torch.cat((
                sid[:, None].float(), truth["room"][:, None].float(), prob.argmax(1, keepdim=True).float(),
                prob.amax(1, keepdim=True), truth["xy"], pose[:, :2] * 10, truth["yaw"][:, None],
                torch.atan2(pose[:, 2], pose[:, 3])[:, None], pose[:, 4:6].exp(),
                torch.where(truth["goal_ok"] > 0, rel, torch.nan)), 1).half().cpu())
            if len(record):
                rec["delivered"].append(link.delivered[record].cpu())
                rec["depth"].append((link.delivered_depth[record] * 255).round().to(torch.uint8).cpu())
                rec["rgb"].append(link.delivered_rgb[record].cpu())
                _, ok = link.sonar(aux[:, 0] * 150)
                rec["range_cm"].append(torch.where(ok[record] > 0, aux[record, 0] * 150, -1.0).cpu())
                rec["gyro_heading"].append(torch.deg2rad(env.sensors.yaw_deg[record]).cpu())
                rec["pose"].append(torch.cat((truth["xy"][record], truth["yaw"][record, None]), 1).cpu())
                rec["room_p"].append(prob[record].half().cpu())
                rec["pose_hat"].append(torch.cat((pose[record, :2] * 10, torch.atan2(pose[record, 2], pose[record, 3])[:, None],
                                                  pose[record, 4:6].exp()), 1).cpu())
                rec["done"].append(done[record].cpu())
        action = torch.where(done[:, None], 0.0, action)
        if students and len(record):
            rec["pwm"].append((action[record] * 255).round().to(torch.int16).cpu())
        _, finished = env.episode_step(action)
        new = finished & ~done
        if new.any():
            for key, value in (("success", env.success), ("collision", env.collision), ("kind", env.collision_kind),
                               ("timeout", env.timeout), ("travel_m", env.travel),
                               ("time_s", env.elapsed * env.control_dt), ("peak_m_s", env.peak),
                               ("contacts", env.contacts), ("near_s", env.near_steps * env.control_dt),
                               ("min_clearance", env.min_clearance)):
                outcome[key] = torch.where(new, value.float(), outcome[key])
            done |= new
        step += 1
        if step % 50 == 0:
            print(f"EVAL_STEP {step} done={int(done.sum())}/{count} t={time.monotonic() - started:.0f}s", flush=True)
    for key, value in (("travel_m", env.travel), ("time_s", env.elapsed * env.control_dt), ("peak_m_s", env.peak),
                       ("contacts", env.contacts), ("near_s", env.near_steps * env.control_dt),
                       ("min_clearance", env.min_clearance)):
        outcome[key] = torch.where(~done, value.float(), outcome[key])
    args.output.mkdir(parents=True, exist_ok=True)
    out = {k: v.cpu().numpy() for k, v in outcome.items()}
    lengths = env.length[route_of].cpu().numpy()
    with (args.output / "episodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("car", "condition", "controller", "route", "area", "planned_m", "success", "collision",
                    "collision_kind", "timeout", "unfinished", "travel_m", "time_s", "peak_m_s", "contacts", "near_s",
                    "min_clearance_m"))
        for i in range(count):
            r = int(route_of[i])
            w.writerow((i, cond[i], kind[i], r, areas[r], round(float(lengths[i]), 3), int(out["success"][i]),
                        int(out["collision"][i]), int(out["kind"][i]), int(out["timeout"][i]), int(not done[i]),
                        round(float(out["travel_m"][i]), 3), round(float(out["time_s"][i]), 2),
                        round(float(out["peak_m_s"][i]), 3), int(out["contacts"][i]), round(float(out["near_s"][i]), 2),
                        round(float(out["min_clearance"][i]), 3)))
    if students:
        np.savez_compressed(args.output / "place.npz", rows=torch.stack(place_rows, 1).numpy(),
                            columns=np.array(["active", "room", "room_hat", "room_conf", "x", "y", "x_hat", "y_hat",
                                              "yaw", "yaw_hat", "sd_xy", "sd_yaw"] + [f"goal_err_{i}" for i in range(10)]),
                            condition=np.array(cond), controller=np.array(kind), route=route_of.cpu().numpy())
        if len(record):
            np.savez_compressed(args.output / "record.npz", cars=record.cpu().numpy(),
                                route=route_of[record].cpu().numpy(), area=areas[route_of[record].cpu().numpy()],
                                success=out["success"][record.cpu().numpy()], kind=out["kind"][record.cpu().numpy()],
                                way=env.way[route_of[record]].cpu().numpy(), start_area=areas[route_of[record].cpu().numpy()],
                                **{k: torch.stack(v, 1).numpy() for k, v in rec.items()})
    if args.real_path and link.path_stats:
        stats = torch.cat(link.path_stats).numpy()
        np.save(args.output / "path_stats.npy", stats)
        print("PATH_MATCH", json.dumps({"frames": len(stats), "feature_cos_median": round(float(np.median(stats[:, 0])), 4),
              "feature_cos_p5": round(float(np.percentile(stats[:, 0], 5)), 4),
              "depth_abs_diff_mean": round(float(stats[:, 1].mean()), 4),
              "pixel_abs_diff_mean": round(float(stats[:, 2].mean()), 3)}), flush=True)
    summary = {"split": args.split, "groups": args.groups, "heads": head_paths, "seed": args.seed,
               "cars": count, "steps": step, "seconds": round(time.monotonic() - started, 1),
               "unfinished": int((~done).sum())}
    (args.output / "eval_run.json").write_text(json.dumps(summary))
    print("EVAL_DONE", json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="nominal/student,stress/student")
    parser.add_argument("--heads", nargs="*", default=[], help="name=heads.pt per student controller, or one bare path")
    parser.add_argument("--per-area", type=int, default=24)
    parser.add_argument("--split", choices=("test", "dev", "home_test", "short_test"), default="test")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--record-per-area", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rgb-size", type=int, nargs=2, default=[224, 168])
    parser.add_argument("--teacher", type=Path, default=Path("/workspace/elegoo/p4c/driver_v2.onnx"))
    parser.add_argument("--da2", default="/workspace/elegoo/p4c/da2")
    parser.add_argument("--real-path", action="store_true",
                        help="driver.py's input path: render 640x480, libjpeg (cv2) round trip, cv2.INTER_AREA; also logs"
                             " how far the training path's features are on the same frames")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = any(g.split("/")[1] not in ("v2", "teacher") for g in args.groups.split(","))
    wp.config.enable_backward = False
    torch.backends.cudnn.benchmark = True
    sim_cfg = SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        run(args, SimulationContext(sim_cfg))


if __name__ == "__main__":
    main()
