"""Held-out scoring of a NavDP checkpoint behind the shared policy code, with real-car inputs (env.py).

Every (condition, controller, test route) is one car, all in one stage. Controllers: `navdp` (the checkpoint)
and `expert` (the same pure-pursuit tracker on the privileged route path: a check of the tracker and the
harness, not a driver). Conditions: nominal (car response and lens randomised per car, every depth
corruption sampled), stress (every depth corruption at its upper magnitude), dpscale (nominal, but the global
depth scale, near/far compression and smooth error follow Depth Pro on the renders: --dp-scale (default its
p10-p90), --dp-gamma, --dp-smooth-sd, measured by depth_measure.py), depthpro (depth = sim/realdepth observe(), Depth Pro 2-S, on each delivered 640x480 JPEG frame;
needs --rgb-size 640 480 and the cmdtest image). Same --seed, split and route set => the same per-car draws for
every checkpoint, so runs pair up car by car.

  sim/cloud/run_headless.sh elegoo-sim-8 navdp/elegoo/navdp/evaluate.py --checkpoint /workspace/elegoo/ckpt/x.ckpt \
      --output /workspace/elegoo/navdp/eval_zeroshot --viz none
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

CONDITIONS = {"nominal": False, "stress": True, "dpscale": False, "depthpro": False}


def select_routes(areas: np.ndarray, per_area: int) -> np.ndarray:
    return np.concatenate([np.flatnonzero(areas == a)[:per_area] for a in sorted(set(areas.tolist()))])


def run(args, sim) -> None:
    from elegoo.navdp.env import NavDPEnv, NavDPSpec
    from elegoo.world import CameraSpec
    from policy import NavDPPolicy, Tracker
    from xnavdp.net import load

    torch.manual_seed(args.seed)
    started = time.monotonic()
    conditions = args.conditions.split(",")
    controllers = args.controllers.split(",")
    with np.load(Path(__file__).resolve().parents[1] / "train" / "routes.npz") as data:
        areas = data[f"{args.split}_area"]
    routes = np.array(args.routes) if args.routes else select_routes(areas, args.per_area)
    groups = [(c, k) for c in conditions for k in controllers]
    count = len(groups) * len(routes)
    env = NavDPEnv(sim, NavDPSpec(count=count, randomize=True, random_camera=True, seed=args.seed, split=args.split,
                                  visual=True, camera=CameraSpec(outputs=("rgb",), width=args.rgb_size[0], height=args.rgb_size[1]),
                                  step_ticks=(10,)))
    device = env.device
    group_of = torch.arange(count, device=device) // len(routes)
    route_of = torch.as_tensor(np.tile(routes, len(groups)), device=device)
    stress = torch.tensor([CONDITIONS[c] for c, _ in groups], device=device)[group_of]
    env.set_conditions(torch.ones(count, device=device), stress)
    cond_of = [groups[g][0] for g in group_of.tolist()]
    scaled = torch.tensor([c == "dpscale" for c in cond_of], device=device).nonzero().squeeze(1)
    env.set_depth_domain(scaled, tuple(args.dp_scale), args.dp_smooth_sd, args.dp_gamma)
    real = torch.tensor([c == "depthpro" for c in cond_of], device=device).nonzero().squeeze(1)
    observe = None
    if len(real):
        if tuple(args.rgb_size) != (640, 480):
            raise SystemExit("depthpro needs --rgb-size 640 480 (realdepth's camera model)")
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from sim.realdepth import realdepth

        realdepth._device = torch.device("cuda")
        observe = realdepth.observe
    env.reset(env.ids, route_of)
    kind_of = [groups[g][1] for g in group_of.tolist()]
    pol = torch.tensor([k == "navdp" for k in kind_of], device=device).nonzero().squeeze(1)
    exp = torch.tensor([k == "expert" for k in kind_of], device=device).nonzero().squeeze(1)
    net = load(args.checkpoint, device, torch.bfloat16) if len(pol) else None
    generator = torch.Generator(device=device).manual_seed(args.seed)
    policy = NavDPPolicy(net, len(pol), samples=args.samples, plan_every=args.plan_every, generator=generator) \
        if net is not None else None
    expert = Tracker(len(exp), device)
    print(f"EVAL_READY cars={count} routes={len(routes)} groups={groups} startup_s={time.monotonic() - started:.1f}",
          flush=True)

    env.advance(torch.zeros((count, 2), device=device))
    done = torch.zeros(count, dtype=torch.bool, device=device)
    keys = ("success", "collision", "kind", "timeout", "travel_m", "time_s", "peak_m_s", "x", "y", "contacts",
            "near_s", "min_clearance")
    outcome = {k: torch.zeros(count, device=device) for k in keys}
    video = pol[: args.video_cars] if ("nominal", "navdp") in groups else pol[:0]
    frames, depths, plans, trace = [], [], [], []
    step_s, spacing, reach = [], [], []
    split_s = np.zeros(3)
    step = 0
    while not done.all() and step < args.max_steps:
        t0 = time.monotonic()
        inp = env.car_inputs()
        if observe is not None:
            fresh = real[inp["rgb_valid"][real] & ~done[real]]
            if len(fresh):
                frames = inp["rgb"][fresh].cpu().numpy()
                inp["depth"][fresh] = torch.as_tensor(np.stack([observe(f) for f in frames]), device=device)
        torch.cuda.synchronize()
        t1 = time.monotonic()
        pwm = torch.zeros((count, 2), device=device)
        if len(pol):
            sl = {k: v[pol] for k, v in inp.items()}
            pwm[pol] = policy.step(sl["rgb"], sl["rgb_valid"], sl["depth"], sl["depth_valid"], sl["target_xy"],
                                   sl["speed_limit"], sl["heading"], sl["range_cm"], sl["last_pwm"],
                                   rgb_age=sl["rgb_age"], gyro_rate=sl["gyro_rate"])
            if step % 10 == 5:
                best = policy.tracker.path
                spacing.append(torch.diff(best, dim=1, prepend=torch.zeros_like(best[:, :1])).norm(dim=-1).median().item())
                reach.append(best[:, -1].norm(dim=-1).median().item())
        if len(exp):
            local = torch.arange(len(exp), device=device)
            expert.advance(inp["heading"][exp], inp["last_pwm"][exp])
            expert.set_path(local, env.expert_path()[exp], inp["heading"][exp], torch.zeros((len(exp), 3), device=device))
            pwm[exp] = expert.act(inp["speed_limit"][exp], inp["range_cm"][exp])
        pwm = torch.where(done[:, None], 0.0, pwm)
        if len(video):
            v = torch.searchsorted(pol, video)
            frames.append(torch.nn.functional.interpolate(inp["rgb"][video].permute(0, 3, 1, 2).float(), size=(84, 112),
                                                          mode="area").round().to(torch.uint8).permute(0, 2, 3, 1).cpu())
            depths.append((inp["depth"][video, ::2, ::2] * 255).round().to(torch.uint8).cpu())
            plans.append(policy.tracker.path[v].half().cpu())
        torch.cuda.synchronize()
        t2 = time.monotonic()
        p = env.privileged()
        trace.append(torch.cat((p["pos_w"][:, :2], p["heading_rad"][:, None]), 1).half().cpu())
        _, finished = env.episode_step(pwm / 255)
        new = finished & ~done
        if new.any():
            for key, value in (("success", env.success), ("collision", env.collision), ("kind", env.collision_kind),
                               ("timeout", env.timeout), ("travel_m", env.travel),
                               ("time_s", env.elapsed * env.control_dt), ("peak_m_s", env.peak),
                               ("x", env.xy[:, 0]), ("y", env.xy[:, 1]), ("contacts", env.contacts),
                               ("near_s", env.near_steps * env.control_dt), ("min_clearance", env.min_clearance)):
                outcome[key] = torch.where(new, value.float(), outcome[key])
            done |= new
        torch.cuda.synchronize()
        step_s.append(time.monotonic() - t0)
        split_s += (t1 - t0, t2 - t1, time.monotonic() - t2)
        step += 1
        if step % 25 == 0:
            print(f"EVAL_STEP {step} done={int(done.sum())}/{count} step_s={np.mean(step_s[-25:]):.2f} "
                  f"inputs/policy/physics_s={np.round(split_s / step, 2).tolist()} t={time.monotonic() - started:.0f}s",
                  flush=True)
    for key, value in (("travel_m", env.travel), ("time_s", env.elapsed * env.control_dt), ("peak_m_s", env.peak),
                       ("x", env.xy[:, 0]), ("y", env.xy[:, 1]), ("contacts", env.contacts),
                       ("near_s", env.near_steps * env.control_dt), ("min_clearance", env.min_clearance)):
        outcome[key] = torch.where(~done, value.float(), outcome[key])
    args.output.mkdir(parents=True, exist_ok=True)
    out = {k: v.cpu().numpy() for k, v in outcome.items()}
    lengths = env.length[route_of].cpu().numpy()
    with (args.output / "episodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("car", "condition", "controller", "checkpoint", "split", "route", "area", "planned_m", "success",
                    "collision", "collision_kind", "timeout", "unfinished", "travel_m", "time_s", "peak_m_s",
                    "end_x", "end_y", "contacts", "near_s", "min_clearance_m"))
        for i in range(count):
            c, k = groups[group_of[i]]
            w.writerow((i, c, k, args.label, args.split, int(route_of[i]), areas[int(route_of[i])], round(float(lengths[i]), 3),
                        int(out["success"][i]), int(out["collision"][i]), int(out["kind"][i]), int(out["timeout"][i]),
                        int(not done[i]), round(float(out["travel_m"][i]), 3), round(float(out["time_s"][i]), 2),
                        round(float(out["peak_m_s"][i]), 3), round(float(out["x"][i]), 3), round(float(out["y"][i]), 3),
                        int(out["contacts"][i]), round(float(out["near_s"][i]), 2),
                        round(float(out["min_clearance"][i]), 3)))
    pose = torch.stack(trace, 1)
    np.savez_compressed(args.output / "trace.npz", pose=pose.numpy(), route=route_of.cpu().numpy(),
                        group=group_of.cpu().numpy(), groups=np.array([f"{c}/{k}" for c, k in groups]))
    if len(video):
        ids = video.cpu().numpy()
        np.savez_compressed(args.output / "video.npz", cars=ids, rgb=torch.stack(frames, 1).numpy(),
                            depth=torch.stack(depths, 1).numpy(), plan=torch.stack(plans, 1).float().numpy(),
                            pose=pose[ids].float().numpy(), success=out["success"][ids], kind=out["kind"][ids],
                            steps=np.round(out["time_s"][ids] / 0.1).astype(int), route=route_of[video].cpu().numpy(),
                            way=env.way[route_of[video]].cpu().numpy(), count=env.counts[route_of[video]].cpu().numpy(),
                            area=areas[route_of[video].cpu().numpy()])
    summary = {"cars": count, "steps": step, "seconds": round(time.monotonic() - started, 1),
               "unfinished": int((~done).sum()), "mean_step_s": round(float(np.mean(step_s)), 3),
               "policy_cars": int(len(pol)), "samples": args.samples, "plan_every": args.plan_every,
               "rgb_size": args.rgb_size,
               "path_step_m_median": round(float(np.median(spacing)), 4) if spacing else None,
               "path_end_m_median": round(float(np.median(reach)), 3) if reach else None}
    (args.output / "eval_run.json").write_text(json.dumps(summary))
    print("EVAL_DONE", json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", default="navdp")
    parser.add_argument("--conditions", default="nominal,stress")
    parser.add_argument("--controllers", default="navdp,expert")
    parser.add_argument("--per-area", type=int, default=24)
    parser.add_argument("--split", choices=("test", "dev", "home_test", "short_test"), default="test")
    parser.add_argument("--dp-scale", type=float, nargs=2, default=[0.83, 2.05], metavar=("LO", "HI"),
                        help="dpscale condition: global depth scale range, log-uniform")
    parser.add_argument("--dp-smooth-sd", type=float, default=0.25, help="dpscale condition: mean smooth error log sd")
    parser.add_argument("--dp-gamma", type=float, default=0.35, help="dpscale condition: max compression exponent")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--plan-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--rgb-size", type=int, nargs=2, default=[224, 168], metavar=("W", "H"),
                        help="render size; 640 480 = the real car's frame, resized by the driver's own code")
    parser.add_argument("--video-cars", type=int, default=60)
    parser.add_argument("--routes", type=int, nargs="*", help="explicit test route ids instead of --per-area")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    wp.config.enable_backward = False
    sim_cfg = SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        run(args, SimulationContext(sim_cfg))


if __name__ == "__main__":
    main()
