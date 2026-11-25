"""How does Depth Pro 2-S read the rebuilt DA3 visual scene? Measures what the training depth must cover (p7).

Cars drive training routes (base / home / short mix, expert tracker) in the visual scene with nominal lens (RTX RGB
and ray-cast depth share the mount). At random steps each car's 640x480 frame goes through a JPEG round trip
(q60-90) and sim/realdepth's exact observe() steps on CUDA; truth = DriverEnv.true_depth(). Per frame, on the
160x120 grid where truth is 0.3-4.5 m:
- global: exp(median log(est / true))                       -> per-episode global scale
- smooth: sd of the residual log error low-passed with a 20 px Gaussian (residual = log ratio - global)
- fine:   sd of what is left                                 (the spec's existing pixel noise / edges)
Also: global scale vs the frame's median true depth, the same stats for the training corruption (spec and p7
domain) on the same frames, and Depth Pro's ms per frame (one frame at a time, as observe(); and a batch of 8).

  bash vm_run.sh measure sim/isaac/elegoo/navdp/depth_measure.py --output /workspace/elegoo/p7navdp/out/depth --viz none
"""

import argparse
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

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(ROOT)]


def gaussian(sigma: float, device) -> torch.Tensor:
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = (-0.5 * (x / sigma) ** 2).exp()
    return k / k.sum()


def lowpass(x: torch.Tensor, w: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Masked separable blur of x [N,H,W] with weights w (normalised convolution)."""
    from torch.nn import functional as F

    def blur(v):
        r = k.numel() // 2
        v = F.conv2d(F.pad(v[:, None], (r, r, 0, 0), mode="reflect"), k[None, None, None, :])
        return F.conv2d(F.pad(v, (0, 0, r, r), mode="reflect"), k[None, None, :, None])[:, 0]

    return blur(x * w) / blur(w).clamp_min(1e-3)


def frame_stats(est: torch.Tensor, true: torch.Tensor, k: torch.Tensor) -> dict[str, torch.Tensor]:
    ok = torch.isfinite(true) & (true > 0.3) & (true < 4.5) & torch.isfinite(est) & (est > 0)
    log_ratio = torch.where(ok, (est.clamp_min(1e-3) / true.nan_to_num(posinf=1.0).clamp_min(1e-3)).log(), 0.0)
    n = ok.sum((1, 2))
    big = torch.where(ok, log_ratio, torch.nan)
    g = big.flatten(1).nanmedian(1).values
    res = torch.where(ok, log_ratio - g[:, None, None], 0.0)
    smooth = lowpass(res, ok.float(), k)
    fine = torch.where(ok, res - smooth, 0.0)
    sd = lambda v: ((v.square() * ok).sum((1, 2)) / n.clamp_min(1)).sqrt()  # noqa: E731
    med_true = torch.where(ok, true, torch.nan).flatten(1).nanmedian(1).values
    return {"valid_px": n, "global": g.exp(), "smooth_sd": sd(torch.where(ok, smooth, 0.0)), "fine_sd": sd(fine),
            "median_true_m": med_true}


def run(args, sim) -> None:
    import cv2

    from elegoo.navdp.env import NavDPEnv, NavDPSpec
    from elegoo.train.depth import decode
    from elegoo.world import CameraSpec
    from policy import Tracker
    from sim.realdepth import realdepth

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = NavDPEnv(sim, NavDPSpec(count=args.cars, randomize=True, random_camera=False, seed=args.seed, visual=True,
                                  camera=CameraSpec(outputs=("rgb",), width=640, height=480), step_ticks=(10,),
                                  home_fraction=0.7, short_fraction=0.15,
                                  exclude=((-0.05, -7.35, 0.6, -6.7),)))
    env.max_len = 14.0
    device = env.device
    nominal = env.camera_range[:, 0].tolist()
    print("MEASURE_LENS calibration nominal pitch/hfov/height/forward", nominal, "rgb CameraSpec 20/65/0.126/0.0825",
          flush=True)
    realdepth._device = torch.device("cuda")
    model = realdepth._load()
    k = gaussian(20.0, device)
    expert = Tracker(args.cars, device)
    n = args.cars
    env.reset()
    env.advance(torch.zeros((n, 2), device=device))
    expert.reset(env.ids)
    # Two corruption arms on the same frames: the spec's (p4 training) and the p7 domain.
    p7 = torch.zeros(n, dtype=torch.bool, device=device)
    p7[n // 2:] = True
    env.set_depth_domain(env.ids[p7], tuple(args.p7_scale), args.p7_smooth_sd, args.p7_gamma)
    env.reset(env.ids)
    expert.reset(env.ids)
    take_at = torch.as_tensor(rng.integers(3, 60, n), device=device)
    rows = {k2: [] for k2 in ("dp", "corr_spec", "corr_p7")}
    ms_single, ms_batch = [], []
    step = 0
    while len(rows["dp"]) < args.frames and step < args.max_steps:
        inp = env.car_inputs()
        expert.advance(inp["heading"], inp["last_pwm"])
        expert.set_path(env.ids, env.expert_path(), inp["heading"], torch.zeros((n, 3), device=device))
        pwm = expert.act(inp["speed_limit"], inp["range_cm"])
        due = ((env.age == take_at) & (env.age > 2)).nonzero().squeeze(1)
        if len(due):
            rgb = env.camera.data.output["rgb"].torch[due, ..., :3].cpu().numpy()
            true = env.true_depth()[due]
            est = []
            for i, frame in enumerate(rgb):
                q = int(rng.choice((60, 70, 80, 90)))
                ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
                frame = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                est.append(realdepth.observe(frame))
                torch.cuda.synchronize()
                ms_single.append((time.perf_counter() - t0) * 1000)
            est_z = decode(torch.as_tensor(np.stack(est), device=device))
            est_z = torch.where(torch.as_tensor(np.stack(est), device=device) > 1e-4, est_z, torch.inf)
            corr = decode(inp["depth"][due])
            corr = torch.where(inp["depth"][due] > 1e-4, corr, torch.inf)
            # car_inputs' depth is one step old (latency); compare corruption against this step's truth anyway:
            # at 100 ms and <= 0.4 m/s the view moves <= 4 cm, small next to the scale terms measured here.
            s_dp = frame_stats(est_z, true, k)
            s_corr = frame_stats(corr, true, k)
            for i in range(len(due)):
                rows["dp"].append({key: float(v[i]) for key, v in s_dp.items()})
                arm = "corr_p7" if bool(p7[due[i]]) else "corr_spec"
                rows[arm].append({key: float(v[i]) for key, v in s_corr.items()})
            if len(ms_batch) < 5 and len(rgb) >= 8:
                x = torch.stack([realdepth.to_normalized_tensor(
                    cv2.resize(f, (realdepth.IN_W, realdepth.IN_H), interpolation=cv2.INTER_AREA), realdepth._device)
                    for f in rgb[:8]])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.infer(x, f_px=realdepth.CAR_FX_640 * realdepth.IN_W / 640)
                torch.cuda.synchronize()
                ms_batch.append((time.perf_counter() - t0) * 1000 / 8)
            take_at[due] = env.age[due] + torch.as_tensor(rng.integers(15, 45, len(due)), device=device)
            print(f"MEASURE frames={len(rows['dp'])} step={step}", flush=True)
        _, done = env.episode_step(pwm / 255)
        ended = done.nonzero().squeeze(1)
        if len(ended):
            env.reset(ended)
            expert.reset(ended)
            take_at[ended] = torch.as_tensor(rng.integers(3, 60, len(ended)), device=device)
        if len(env.restarted):
            expert.reset(env.restarted)
        step += 1

    def summary(rs):
        out = {"frames": len(rs)}
        rs = [r for r in rs if r["valid_px"] >= 500]
        out["frames_used"] = len(rs)
        for key in ("global", "smooth_sd", "fine_sd"):
            v = np.array([r[key] for r in rs])
            out[key] = {p: round(float(np.percentile(v, q)), 3) for p, q in
                        (("p05", 5), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("p95", 95))}
        g = np.log([r["global"] for r in rs])
        d = np.array([r["median_true_m"] for r in rs])
        bands = {}
        for lo, hi in ((0.3, 1.0), (1.0, 2.0), (2.0, 4.5)):
            m = (d >= lo) & (d < hi)
            if m.sum() >= 5:
                bands[f"{lo}-{hi}m"] = {"frames": int(m.sum()), "median_global": round(float(np.exp(np.median(g[m]))), 3)}
        out["global_by_median_true_depth"] = bands
        out["log_global_sd"] = round(float(g.std()), 3)
        return out

    result = {"machine": args.machine, "scene": "sim/isaac/scene/building_visual.usda (DA3 rebuild)",
              "depth_pro": summary(rows["dp"]), "corruption_spec_p4": summary(rows["corr_spec"]),
              "corruption_p7": summary(rows["corr_p7"]), "p7_scale": args.p7_scale, "p7_smooth_sd": args.p7_smooth_sd, "p7_gamma": args.p7_gamma,
              "depth_pro_ms_per_frame_single": round(float(np.median(ms_single)), 1) if ms_single else None,
              "depth_pro_ms_per_frame_batch8": round(float(np.median(ms_batch)), 1) if ms_batch else None,
              "steps": step}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "depth_measure.json").write_text(json.dumps(result, indent=1))
    with (args.output / "depth_measure_frames.csv").open("w") as f:
        f.write("source,valid_px,global,smooth_sd,fine_sd,median_true_m\n")
        for src, rs in rows.items():
            for r in rs:
                f.write(f"{src},{int(r['valid_px'])},{r['global']:.4f},{r['smooth_sd']:.4f},{r['fine_sd']:.4f},"
                        f"{r['median_true_m']:.3f}\n")
    print("MEASURE_DONE " + json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cars", type=int, default=32)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--p7-scale", type=float, nargs=2, default=[0.8, 2.1])
    parser.add_argument("--p7-smooth-sd", type=float, default=0.25)
    parser.add_argument("--p7-gamma", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--machine", default="elegoo-sim")
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
