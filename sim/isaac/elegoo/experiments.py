"""Evidence from actual simulated rigid-body trajectories and renderer buffers."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
import yaml

from .drive import CALIBRATION
from .environment import ElegooEnv
from .world import BOXES


def match(env: ElegooEnv, output: Path) -> None:
    """Parallel trials: two straight, three spin/coast, watchdog, and heading hold."""
    real = yaml.safe_load(CALIBRATION.read_text())["measured"]
    action = torch.tensor(
        [[80, 80], [110, 110], [-80, 80], [-95, 95], [-110, 110], [110, 110], [110, 110]],
        device=env.device, dtype=torch.float32,
    )
    env.advance(torch.zeros_like(action), 25)
    env.reset()
    # Preload zero packets briefly to settle contacts, then align the first test packet.
    env.advance(torch.zeros_like(action), 25)
    env.drive.next_send.fill_(env.drive.time)
    pos0 = env.robot.data.root_link_pose_w.torch[:, :2].clone()
    yaw0 = env.sensors.yaw_deg.clone()
    start = env.drive.time
    headings: list[torch.Tensor] = []
    spin_samples: list[torch.Tensor] = []
    # 2 s pulse; actual real test duration was not recorded.
    for i in range(200):
        # Cross the second integer PWM threshold before the 1.5 degree limit.
        correction = (-1.2 * (env.sensors.yaw_deg[6] - yaw0[6])).clamp(-25, 25)
        action[6, 0], action[6, 1] = 110 - correction, 110 + correction
        env.advance(action / 255, 1)
        headings.append(env.sensors.yaw_deg[6] - yaw0[6])
        if 150 <= i:
            spin_samples.append(env.sensors.rate_deg_s[2:5].clone())
    spin = torch.stack(spin_samples).mean(dim=0).cpu().tolist()
    max_heading = torch.stack(headings).abs().max().item()
    # Align the host stop send to a transport slot, as in the observed stop test.
    for _ in range(9):
        if env.drive.time + 1e-6 >= env.drive.next_send[2].item():
            break
        env.advance(action / 255, 1)
    send_time = env.drive.time
    yaw_at_stop = env.sensors.yaw_deg[2:5].clone()
    # Watchdog uses a separate trial, because None stops packets to every row.
    env.advance(torch.zeros_like(action), 110)
    travel = (env.robot.data.root_link_pose_w.torch[:2, :2] - pos0[:2]).norm(dim=1).cpu().tolist()
    coast = (env.sensors.yaw_deg[2:5] - yaw_at_stop).abs().cpu().tolist()
    run_s = send_time - start
    env.reset()
    env.advance(action / 255, 80)
    motor_off = float("nan")
    for _ in range(65):
        env.advance(None, 1)
        last_arrival = env.drive.last_receive[5].item()
        if env.drive.pwm[5].abs().sum().item() == 0:
            motor_off = env.drive.time - env.dt - last_arrival
            break
    rows: list[tuple[str, str, str, float, float, float, str]] = []

    def row(metric: str, pwm: int | str, unit: str, value: float, ref: float, tol: float, op: str = "delta") -> None:
        rows.append((metric, str(pwm), unit, value, ref, tol, op))

    for i, pwm in enumerate((80, 110)):
        row("straight_short_run", pwm, "cm/s", travel[i] * 100 / run_s, real["straight_short_run_cm_s"][pwm], .10)
    for i, pwm in enumerate((80, 95, 110)):
        row("spin", pwm, "deg/s", spin[i], real["spin_deg_s"][pwm], .10)
        row("coast_after_host_stop", pwm, "deg", coast[i], real["coast_deg"][pwm], .30)
    row("watchdog_motor_cutoff", 110, "s", motor_off, real["watchdog_s"], .10)
    row("heading_hold_max_abs", 110, "deg", max_heading, real["heading_hold_abs_max_deg"], .10, "upper_bound")
    with (output / "car-match.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "pwm", "units", "simulated", "measured", "error_pct", "tol_pct", "comparison", "pass"))
        for metric, pwm, unit, value, ref, tol, op in rows:
            error = abs(value - ref) / ref
            passed = value <= ref * (1 + tol) if op == "upper_bound" else error <= tol
            record = (metric, pwm, unit, f"{value:.6f}", ref, f"{100*error:.3f}", 100*tol, op, passed)
            writer.writerow(record)
            print("MATCH", *record, flush=True)


def proof(env: ElegooEnv, output: Path) -> None:
    """Save three pan views with an RGB/depth pair each."""
    proof_dir = output / "proof"
    proof_dir.mkdir(exist_ok=True)
    action = torch.zeros((1, 2), device=env.device)
    for label, degrees in (("center", 0), ("left", 21), ("right", -21)):
        env.set_pan(torch.deg2rad(torch.tensor([degrees], device=env.device)))
        for _ in range(10):
            obs = env.step(action)
        rgb = obs["rgb"][0].cpu().numpy()
        depth = obs["depth"][0, ..., 0].cpu().numpy()
        assert rgb.max() > rgb.min() and np.isfinite(depth).any(), "Blank sensor capture"
        Image.fromarray(rgb).save(proof_dir / f"{label}_rgb.png")
        # Log depth reserves black for misses while making the distant floor legible.
        preview = np.where(np.isfinite(depth), np.clip(1 - np.log1p(depth) / np.log(30), 0.12, 1), 0)
        Image.fromarray(np.uint8(preview * 255)).save(proof_dir / f"{label}_depth.png")
        print(f"PROOF {label} {rgb.shape} finite_depth={np.isfinite(depth).mean():.3f} "
              f"range_cm={obs['range_cm'][0].item()} pan_rad={env.robot.data.joint_pos.torch[0, env.pan_ids[0]].item():.4f}",
              flush=True)
    intrinsics = env.camera.data.intrinsic_matrices.torch[0].cpu().tolist()
    print(f"PROOF_INTRINSICS {intrinsics}", flush=True)


def benchmark(env: ElegooEnv, args: argparse.Namespace) -> None:
    """Measure real vector step throughput including both requested image outputs."""
    action = torch.full((args.num_envs, 2), 80 / 255, device=env.device)
    for _ in range(5):
        obs = env.step(action)
    assert obs["rgb"].shape == (args.num_envs, args.height, args.width, 3)
    assert obs["depth"].shape == (args.num_envs, args.height, args.width, 1)
    assert (obs["rgb"].flatten(1).amax(1) > obs["rgb"].flatten(1).amin(1)).all(), "Blank RGB view"
    assert ((obs["depth"] > 0) & torch.isfinite(obs["depth"])).flatten(1).any(1).all(), "Blank depth view"
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(args.steps):
        env.step(action)
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    result = (args.num_envs, args.width, args.height, args.steps, elapsed, args.steps / elapsed,
              args.num_envs * args.steps / elapsed)
    with (args.output / f"bench-{args.num_envs}-{args.width}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("envs", "width", "height", "timed_calls", "seconds", "vector_steps_s", "env_steps_s"))
        writer.writerow(result)
    print("BENCH", *result, flush=True)


def wall(env: ElegooEnv) -> None:
    """Drive at an obstacle long enough to pass it if contacts are not active."""
    action = torch.full((1, 2), 110 / 255, device=env.device)
    peak_x = 0.0
    for _ in range(70):
        obs = env.step(action)
        x = env.robot.data.root_link_pose_w.torch[0, 0].item()
        peak_x = max(peak_x, x)
    near_face = BOXES[0].center[0] - BOXES[0].size[0] / 2
    speed_at_wall = env.robot.data.root_com_vel_w.torch[0, :2].norm().item()
    print(f"WALL peak_x={peak_x:.4f} near_face={near_face:.4f} final_speed={speed_at_wall:.4f} "
          f"final_range_cm={obs['range_cm'][0].item()} on_ground={obs['on_ground'][0].item()}", flush=True)
    assert 0.5 < peak_x < near_face and speed_at_wall < 0.06
