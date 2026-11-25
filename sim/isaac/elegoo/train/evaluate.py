"""Held-out evaluation: every (condition, controller, test route) is one car, all in one stage.

Conditions: nominal car (randomize off) with the spec's depth corruptions sampled per episode;
slow / fast = speed and turn response x0.75 / x1.25; stress = every depth corruption at its upper
magnitude. Controllers: the learned policy, nav/run.py's pure pursuit driven by the heading command
instead of the true pose, and a hand-written reactive controller. All three get the same depth and
aux arrays; none sees the true pose.

  sim/cloud/run_headless.sh drv/elegoo/train/evaluate.py --checkpoint /workspace/elegoo/drv/run/ppo.pt \
      --output /workspace/elegoo/drv/eval --viz none
"""

import argparse
import csv
import json
import math
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
from elegoo.train.depth import decode
from elegoo.train.model import FEATURES, HISTORY, driver_model, load_actor

CONDITIONS = {"nominal": (1.0, False), "slow": (0.75, False), "fast": (1.25, False), "stress": (1.0, True)}
CONTROLLERS = ("policy", "pursuit", "reactive")


def pursuit(aux: torch.Tensor, recover: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """nav/run.py drive(): forward 100 +/- 70 PWM/rad (cap 40), spin 85 beyond 40 deg, reverse 90 for
    0.7 s to recover. Its stuck test needs the pose, so here the trigger is the range sensor < 12 cm."""
    error = torch.atan2(aux[:, 4], aux[:, 5])
    turn = (70.0 * error).clamp(-40, 40)
    left, right = 100.0 - turn, 100.0 + turn
    spin = 85.0 * error.sign()
    wide = error.abs() > math.radians(40)
    left, right = torch.where(wide, -spin, left), torch.where(wide, spin, right)
    recover = torch.where((aux[:, 0] * 150 < 12) & (recover <= 0), 0.7, recover)
    backing = recover > 0
    left, right = torch.where(backing, -90.0, left), torch.where(backing, -90.0, right)
    return torch.stack((left, right), 1) / 255 * aux[:, 7:8], recover - dt


def reactive(depth: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    """Steer to the commanded heading; slow down as the nearest obstacle ahead (range sensor or depth
    rows above the floor) closes in; below 0.30 m spin toward the side with more free depth; back off
    below 12 cm."""
    z = decode(depth[:, 0, 20:100])
    front = torch.nn.functional.avg_pool2d(z[:, None, :, 56:104], 4)[:, 0].flatten(1).amin(1)
    left_free, right_free = z[..., :53].mean((-1, -2)), z[..., 107:].mean((-1, -2))
    near = torch.minimum(aux[:, 0] * 1.5, front)
    error = torch.atan2(aux[:, 4], aux[:, 5])
    speed = 110.0 * ((near - 0.30) / 0.70).clamp(0.35, 1.0)
    side = ((left_free - right_free) / (left_free + right_free)).clamp(-1, 1)
    turn = (80.0 * error).clamp(-45, 45) + 25.0 * side * (near < 1.0)
    left, right = speed - turn, speed + turn
    spin_dir = torch.where(left_free >= right_free, 1.0, -1.0)
    blocked = near < 0.30
    left, right = torch.where(blocked, -90.0 * spin_dir, left), torch.where(blocked, 90.0 * spin_dir, right)
    wide = (error.abs() > math.radians(45)) & ~blocked
    left, right = torch.where(wide, -85.0 * error.sign(), left), torch.where(wide, 85.0 * error.sign(), right)
    back = aux[:, 0] * 150 < 12
    left, right = torch.where(back, -80.0, left), torch.where(back, -80.0, right)
    return torch.stack((left, right), 1) / 255 * aux[:, 7:8]


def select_routes(areas: np.ndarray, per_area: int) -> np.ndarray:
    picked = [np.flatnonzero(areas == a)[:per_area] for a in sorted(set(areas.tolist()))]
    return np.concatenate(picked)


def run(args, sim) -> None:
    from elegoo.train.driver_env import DriverEnv, RunSpec
    from elegoo.world import CameraSpec

    torch.manual_seed(args.seed)
    started = time.monotonic()
    conditions = args.conditions.split(",")
    policies = ["policy"] if len(args.checkpoint) == 1 else [f"policy{k}" for k in range(len(args.checkpoint))]
    controllers = [p for k in args.controllers.split(",") for p in (policies if k == "policy" else [k])]
    with np.load(Path(__file__).with_name("routes.npz")) as data:
        test_areas = data[f"{args.split}_area"]
    routes = select_routes(test_areas, args.per_area)
    groups = [(c, k) for c in conditions for k in controllers]
    count = len(groups) * len(routes)
    env = DriverEnv(sim, RunSpec(count=count, randomize=False, seed=args.seed, split=args.split,
                                 camera=CameraSpec(outputs=()), step_ticks=(round(args.step_ms / 10),)))
    device = env.device
    group_of = torch.arange(count, device=device) // len(routes)
    route_of = torch.as_tensor(np.tile(routes, len(groups)), device=device)
    response = torch.tensor([CONDITIONS[c][0] for c, _ in groups], device=device)[group_of]
    stress = torch.tensor([CONDITIONS[c][1] for c, _ in groups], device=device)[group_of]
    env.set_conditions(response, stress)
    env.reset(env.ids, route_of)
    controller_of = [groups[g][1] for g in group_of.tolist()]
    policy_of = torch.tensor([policies.index(k) if k in policies else -1 for k in controller_of], device=device)
    is_policy = policy_of >= 0
    is_pursuit = torch.tensor([k == "pursuit" for k in controller_of], device=device)
    models = [driver_model(load_actor(path, device)) for path in args.checkpoint] \
        if any(k in policies for k in controllers) else []
    print(f"EVAL_READY cars={count} routes={len(routes)} groups={groups} startup_s={time.monotonic() - started:.1f}",
          flush=True)

    env.advance(torch.zeros((count, 2), device=device))
    histories = [torch.zeros((count, HISTORY, FEATURES), device=device) for _ in models]
    recover = torch.zeros(count, device=device)
    done = torch.zeros(count, dtype=torch.bool, device=device)
    outcome = {k: torch.zeros(count, device=device) for k in
               ("success", "collision", "kind", "timeout", "travel_m", "time_s", "peak_m_s", "x", "y",
                "contacts", "near_s", "min_clearance")}
    record_ids = (group_of == groups.index(("nominal", policies[0]))).nonzero().flatten() \
        if ("nominal", policies[0]) in groups else torch.zeros(0, dtype=torch.long, device=device)
    replay_ids = record_ids[: args.replay_cars]
    frames, trace = [], []
    replay = {"depth": [], "aux": [], "action": []}
    step = 0
    while not done.all() and step < args.max_steps:
        depth, aux = env.policy_obs()
        action = torch.zeros((count, 2), device=device)
        for k, model in enumerate(models):
            with torch.inference_mode():
                output, histories[k] = model(depth, aux, histories[k])
            action = torch.where((policy_of == k)[:, None], output, action)
            if k == 0:
                learned = output
        piloted, recover = pursuit(aux, recover, env.control_dt)
        action = torch.where(is_pursuit[:, None], piloted, action)
        action = torch.where((~is_policy & ~is_pursuit)[:, None], reactive(depth, aux), action)
        action = torch.where(done[:, None], 0.0, action)
        if len(record_ids):
            frames.append((depth[record_ids, 0] * 255).round().to(torch.uint8).cpu())
        if step < args.replay_steps and len(replay_ids):
            replay["depth"].append(depth[replay_ids, 0].cpu())
            replay["aux"].append(aux[replay_ids].cpu())
            replay["action"].append(learned[replay_ids].cpu())
        p = env.privileged()
        trace.append(torch.cat((p["pos_w"][:, :2], p["heading_rad"][:, None]), 1).half().cpu())
        _, finished = env.episode_step(action)
        new = finished & ~done
        if new.any():
            for key, value in (("success", env.success), ("collision", env.collision), ("kind", env.collision_kind),
                               ("timeout", env.timeout), ("travel_m", env.travel),
                               ("time_s", env.elapsed * env.control_dt), ("peak_m_s", env.peak),
                               ("x", env.xy[:, 0]), ("y", env.xy[:, 1]), ("contacts", env.contacts),
                               ("near_s", env.near_steps * env.control_dt), ("min_clearance", env.min_clearance)):
                outcome[key] = torch.where(new, value.float(), outcome[key])
            done |= new
        step += 1
        if step % 50 == 0:
            print(f"EVAL_STEP {step} done={int(done.sum())}/{count} t={time.monotonic() - started:.0f}s", flush=True)
    open_ids = ~done
    for key, value in (("travel_m", env.travel), ("time_s", env.elapsed * env.control_dt), ("peak_m_s", env.peak),
                       ("x", env.xy[:, 0]), ("y", env.xy[:, 1]), ("contacts", env.contacts),
                       ("near_s", env.near_steps * env.control_dt), ("min_clearance", env.min_clearance)):
        outcome[key] = torch.where(open_ids, value.float(), outcome[key])
    args.output.mkdir(parents=True, exist_ok=True)
    out = {k: v.cpu().numpy() for k, v in outcome.items()}
    lengths = env.length[route_of].cpu().numpy()
    with (args.output / "episodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("car", "condition", "controller", "route", "area", "planned_m", "success", "collision",
                    "collision_kind", "timeout", "unfinished", "travel_m", "time_s", "peak_m_s", "end_x", "end_y",
                    "contacts", "near_s", "min_clearance_m", "step_ms"))
        for i in range(count):
            c, k = groups[group_of[i]]
            w.writerow((i, c, k, int(route_of[i]), test_areas[int(route_of[i])], round(float(lengths[i]), 3),
                        int(out["success"][i]), int(out["collision"][i]), int(out["kind"][i]), int(out["timeout"][i]),
                        int(not done[i]), round(float(out["travel_m"][i]), 3), round(float(out["time_s"][i]), 2),
                        round(float(out["peak_m_s"][i]), 3), round(float(out["x"][i]), 3), round(float(out["y"][i]), 3),
                        int(out["contacts"][i]), round(float(out["near_s"][i]), 2),
                        round(float(out["min_clearance"][i]), 3), args.step_ms))
    np.savez_compressed(args.output / "trace.npz", pose=torch.stack(trace, 1).numpy(),
                        route=route_of.cpu().numpy(), group=group_of.cpu().numpy(),
                        groups=np.array([f"{c}/{k}" for c, k in groups]))
    if len(record_ids):
        success = out["success"][record_ids.cpu().numpy()] > 0
        ids = record_ids.cpu().numpy()
        chosen = list(ids[success][:: max(1, success.sum() // args.videos)][: args.videos]) \
            + list(ids[~success][: args.videos])
        stack = torch.stack(frames, 1)
        position = {int(c): j for j, c in enumerate(ids)}
        np.savez_compressed(
            args.output / "video.npz", cars=np.array(chosen),
            depth=stack[[position[int(c)] for c in chosen]].numpy(),
            steps=np.array([int(round(out["time_s"][c] / float(env.control_dt[c]))) for c in chosen]),
            success=out["success"][chosen], kind=out["kind"][chosen], route=route_of[chosen].cpu().numpy(),
            way=env.way[route_of[chosen]].cpu().numpy(), pose=torch.stack(trace, 1)[chosen].float().numpy(),
            area=test_areas[route_of[chosen].cpu().numpy()])
    if len(replay_ids):
        depth = torch.stack(replay["depth"], 1).flatten(0, 1).numpy()
        aux = torch.stack(replay["aux"], 1).flatten(0, 1).numpy()
        action = torch.stack(replay["action"], 1).flatten(0, 1).numpy()
        reset = np.zeros(len(depth), dtype=bool)
        reset[:: len(replay["depth"])] = True
        np.savez_compressed(args.output / "replay.npz", depth=depth, aux=aux, reset=reset, action=action,
                            pwm=np.rint(np.clip(action, -1, 1) * 255).astype(np.int16))
    summary = {"cars": count, "steps": step, "seconds": round(time.monotonic() - started, 1),
               "unfinished": int((~done).sum())}
    (args.output / "eval_run.json").write_text(json.dumps(summary))
    print("EVAL_DONE", json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--conditions", default="nominal,slow,fast,stress")
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    parser.add_argument("--per-area", type=int, default=24)
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--replay-cars", type=int, default=2)
    parser.add_argument("--replay-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step-ms", type=int, choices=(70, 100), default=70, help="control step of every car")
    parser.add_argument("--videos", type=int, default=3, help="successes and failures recorded for videos")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    wp.config.enable_backward = False
    sim_cfg = SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        run(args, SimulationContext(sim_cfg))


if __name__ == "__main__":
    main()
