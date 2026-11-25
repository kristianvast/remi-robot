"""Stage 1: DAgger imitation of the privileged v2 teacher, with place + goal-direction supervision, in one Isaac process.

Every step all cars are labelled by the teacher (driver_v2 on clean depth + true-pose command); per episode a car
is driven by the teacher with probability beta, else by the current student (on-policy correction). The DA2-S
encoder is frozen; frame features are cached in a CPU replay buffer and the heads train between collection chunks.

p7: home / short route mixing (RunSpec.home_fraction / short_fraction), and against the p4 place-head overfit
(room 0.97 train vs 0.72 held-out): dropout in the heads, random blanking of history slots, stronger weight decay,
and a start-cell holdout - routes whose start falls in a held-out 1 m cell (home starts excepted) never enter the
training buffer but fill a validation buffer, whose room accuracy / xy error is logged every iteration (v_*).

  sim/cloud/run_headless.sh elegoo-sim-9 p4c/sim/isaac/elegoo/combined/train.py --num-envs 192 \
      --budget-sec 5400 --output /workspace/elegoo/p4c/run --viz none
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[4] / "pylib"))
from elegoo.combined.common import AUX, FRAME, HISTORY, Encoder, losses


class Buffer:
    def __init__(self, capacity: int):
        self.capacity, self.size, self.head = capacity, 0, 0
        pin = torch.cuda.is_available()
        z = lambda *shape, dtype=torch.float16: torch.zeros((capacity, *shape), dtype=dtype, pin_memory=pin)
        self.data = {"history": z(HISTORY, FRAME), "ages": z(HISTORY), "aux": z(AUX, dtype=torch.float32),
                     "teacher": z(2, dtype=torch.float32), "room": z(dtype=torch.long), "pose": z(4),
                     "goal": z(10, 2), "goal_ok": z(10)}

    def add(self, mask: torch.Tensor | None = None, **batch) -> None:
        if mask is not None:
            batch = {k: v[mask] for k, v in batch.items()}
        n = len(batch["aux"])
        if n == 0:
            return
        idx = (torch.arange(n) + self.head) % self.capacity
        for k, v in batch.items():
            self.data[k][idx] = v.detach().to("cpu", self.data[k].dtype)
        self.head = (self.head + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, n: int, device) -> dict[str, torch.Tensor]:
        idx = torch.randint(self.size, (n,))
        return {k: v[idx].to(device, non_blocking=True).float() if v.dtype != torch.long else v[idx].to(device)
                for k, v in self.data.items()}


def beta_at(step: int, args) -> float:
    if step < args.teacher_steps:
        return 1.0
    frac = min(1.0, (step - args.teacher_steps) / args.anneal_steps)
    return 1.0 + frac * (args.beta_min - 1.0)


def held_out_routes(way: torch.Tensor, pool: torch.Tensor, fraction: float, cell_m: float) -> torch.Tensor:
    """Per route: does its start lie in a held-out cell (deterministic hash of the 1 m start cell)? Home routes
    all start beside the office desk and are never held out."""
    cell = torch.floor(way[:, 0] / cell_m).long()
    h = ((cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663)) % 1000
    return (h < int(fraction * 1000)) & (pool != 1)


def blank_slots(b: dict[str, torch.Tensor], p: float) -> None:
    """Training batches only: blank history slots at random (as a dropped frame would), so the heads cannot lean on
    one exact frame sequence."""
    if p <= 0:
        return
    from elegoo.combined.common import AGE_NONE

    gone = torch.rand(b["ages"].shape, device=b["ages"].device) < p
    b["ages"] = torch.where(gone, AGE_NONE, b["ages"])
    b["history"] = torch.where(gone[..., None], 0.0, b["history"])


def run(args, sim) -> None:
    from elegoo.combined.env import AREA07_RIDGE, V2, CombinedEnv, Link, act, encode, load_heads, student_inputs
    from elegoo.train.driver_env import RunSpec
    from elegoo.world import CameraSpec

    started = time.monotonic()
    torch.manual_seed(args.seed)
    env = CombinedEnv(sim, RunSpec(count=args.num_envs, randomize=True, seed=args.seed, visual=True,
                                   camera=CameraSpec(outputs=("rgb",), width=args.rgb_size[0], height=args.rgb_size[1]),
                                   step_ticks=(10,), random_camera=True, facing_fraction=args.facing_fraction,
                                   exclude=(AREA07_RIDGE,), home_fraction=args.home_fraction,
                                   short_fraction=args.short_fraction))
    env.max_len = args.max_len
    device = env.device
    encoder = Encoder.pretrained(args.da2).to(device).half().eval().requires_grad_(False)
    heads = load_heads(args.resume, device, dropout=args.dropout)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    holdout = held_out_routes(env.way, env.pool, args.holdout, args.cell_m)
    print(f"COMBINED_HOLDOUT routes={int(holdout.sum())}/{len(holdout)} per pool="
          f"{[int((holdout & (env.pool == k)).sum()) for k in range(int(env.pool.max()) + 1)]}", flush=True)
    val_buffer = Buffer(args.val_buffer)
    best_val = -1.0
    teacher = V2(args.teacher, args.num_envs, device)
    link = Link(args.num_envs, device)
    buffer = Buffer(args.buffer)
    n = args.num_envs
    teacher_drives = torch.ones(n, dtype=torch.bool, device=device)
    print(f"COMBINED_READY envs={n} startup_s={time.monotonic() - started:.1f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    env.reset()
    env.advance(torch.zeros((n, 2), device=device))
    teacher.reset(env.ids)
    # per driver (0 teacher, 1 student): episodes, successes, collisions, metres, seconds, contact onsets, tilts
    tally = torch.zeros((2, 7), device=device)
    step = iteration = 0
    profile = {}
    clock = [time.perf_counter()]

    def lap(name: str) -> None:
        if args.profile:
            torch.cuda.synchronize()
            now = time.perf_counter()
            profile[name] = profile.get(name, 0.0) + now - clock[0]
            clock[0] = now

    def timed(name, fn):
        def wrapper(*a, **k):
            lap(f"{name}_before")
            out = fn(*a, **k)
            lap(name)
            return out
        return wrapper

    if args.profile:
        from elegoo.nav.env import BuildingNavEnv

        env.appearance.augment = timed("obs_augment", env.appearance.augment)
        class TimedCall:
            def __init__(self, inner, name):
                self.inner, self.call = inner, timed(name, inner.__call__)

            def __call__(self, *a, **k):
                return self.call(*a, **k)

            def __getattr__(self, key):
                return getattr(self.inner, key)

        env.depth_sim = TimedCall(env.depth_sim, "obs_depth_corrupt")
        env.command = timed("obs_command", env.command)
        env.truth = timed("obs_truth", env.truth)
        env.privileged = timed("obs_privileged", env.privileged)
        env.appearance.tick = timed("obs_appearance_tick", env.appearance.tick)
        env.true_depth = timed("obs_true_depth", env.true_depth)
        base_observe = BuildingNavEnv.observe
        BuildingNavEnv.observe = timed("obs_observe_incl_readback", base_observe)
    with (args.output / "learning.csv").open("a", newline="", buffering=1) as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(("iteration", "step", "elapsed_s", "beta", "buffer", "steps_per_s", "drive", "room_ce", "room_acc",
                        "xy_err_m", "nll_yaw", "goal", "t_episodes", "t_success", "t_contacts_per_m", "s_episodes",
                        "s_success", "s_collisions_per_m", "s_contacts_per_m", "s_tilts", "s_m_s", "v_buffer",
                        "v_drive", "v_room_acc", "v_xy_err_m", "v_goal"))
        while True:
            chunk_start = time.monotonic()
            profile.clear()
            for _ in range(args.steps_per_iter):
                lap("other")
                depth, aux = env.policy_obs()
                truth = env.truth()
                lap("obs")
                aux_t = env.teacher_aux(aux, truth["xy"], truth["yaw"])
                clean = encode(env.true_depth())[:, None]
                pwm_t = teacher(clean, aux_t)
                lap("teacher")
                link.capture(encoder, env.frame_rgb, torch.deg2rad(aux[:, 1] * 250), lap)
                lap("encoder")
                link.deliver(env.control_dt)
                aux_s = student_inputs(env, link, aux, truth["xy"], truth["yaw"])
                pwm_s, _ = act(heads, link, aux_s)
                lap("heads")
                sample = dict(history=link.history, ages=link.ages, aux=aux_s, teacher=pwm_t, room=truth["room"],
                              pose=truth["pose"], goal=truth["goal"], goal_ok=truth["goal_ok"])
                val = holdout[env.route]
                buffer.add(~val, **sample)
                val_buffer.add(val, **sample)
                lap("buffer")
                action = torch.where(teacher_drives[:, None], pwm_t, pwm_s)
                _, done = env.episode_step(action)
                lap("physics_render")
                step += 1
                ended = done.nonzero().flatten()
                if len(ended):
                    who = (~teacher_drives[ended]).long()
                    stats = torch.stack((torch.ones_like(env.travel[ended]), env.success[ended].float(),
                                         env.collision[ended].float(), env.travel[ended],
                                         env.elapsed[ended] * env.control_dt[ended], env.contacts[ended],
                                         (env.collision_kind[ended] & 1).float()), 1)
                    tally.index_add_(0, who, stats)
                    env.reset(ended)
                    beta = beta_at(step, args)
                    teacher_drives[ended] = torch.rand(len(ended), device=device) < beta
                restart = torch.cat((ended, env.restarted))
                if len(restart):
                    link.reset(restart)
                    teacher.reset(restart)
                    env.reset_estimate(restart)
            rate = args.steps_per_iter * n / (time.monotonic() - chunk_start)
            if args.profile:
                free, total = torch.cuda.mem_get_info()
                print("COMBINED_PROFILE", n, f"{rate:.0f} car-steps/s", f"GPU_MiB={(total - free) / 2**20:.0f}",
                      json.dumps({k: round(1000 * v / args.steps_per_iter, 1) for k, v in profile.items()}),
                      "ms/step", flush=True)
            heads.train()
            parts = {}
            for _ in range(args.updates_per_iter):
                b = buffer.sample(args.batch, device)
                blank_slots(b, args.slot_drop)
                out = heads(b["history"], b["ages"], b["aux"])
                loss, p = losses(out, b["aux"], b["teacher"], b["room"], b["pose"], b["goal"], b["goal_ok"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                opt.step()
                for k, v in p.items():
                    parts[k] = parts.get(k, 0.0) + v / args.updates_per_iter
            heads.eval()
            v = {}
            if val_buffer.size >= 2048:
                with torch.no_grad():
                    for _ in range(4):
                        b = val_buffer.sample(2048, device)
                        out = heads(b["history"], b["ages"], b["aux"])
                        _, p = losses(out, b["aux"], b["teacher"], b["room"], b["pose"], b["goal"], b["goal_ok"])
                        for k, x in p.items():
                            v[k] = v.get(k, 0.0) + x / 4
            elapsed = time.monotonic() - started
            t, s = tally.tolist()
            per = lambda a, b: round(a / b, 4) if b else ""
            w.writerow((iteration, step, round(elapsed, 1), round(beta_at(step, args), 3), buffer.size, round(rate, 1),
                        *(round(parts[k], 4) for k in ("drive", "room_ce", "room_acc", "xy_err_m", "nll_yaw", "goal")),
                        int(t[0]), per(t[1], t[0]), per(t[5], t[3]), int(s[0]), per(s[1], s[0]), per(s[2], s[3]),
                        per(s[5], s[3]), int(s[6]), per(s[3], s[4]), val_buffer.size,
                        *(round(v[k], 4) if v else "" for k in ("drive", "room_acc", "xy_err_m", "goal"))))
            if iteration % 5 == 0:
                print("COMBINED_LEARN", iteration, step, f"{elapsed:.0f}s", f"{rate:.0f} car-steps/s",
                      json.dumps({k: round(v, 3) for k, v in parts.items()}),
                      f"student eps={int(s[0])} success={per(s[1], s[0])} teacher eps={int(t[0])} success={per(t[1], t[0])}",
                      "val", json.dumps({k: round(x, 3) for k, x in v.items() if k in ("drive", "room_acc", "xy_err_m", "goal")}),
                      flush=True)
                tally.zero_()
            last = elapsed >= args.budget_sec
            if v and step > args.teacher_steps + args.anneal_steps and v["room_acc"] > best_val:
                best_val = v["room_acc"]
                torch.save(heads.state_dict(), args.output / "heads_best_val_room.pt")
            if iteration % 20 == 0 or last:
                torch.save(heads.state_dict(), args.output / f"heads_{iteration:04d}.pt")
                torch.save(heads.state_dict(), args.output / "heads.pt")
            iteration += 1
            if last:
                print(f"COMBINED_TRAIN_DONE iteration={iteration} step={step} seconds={elapsed:.0f}", flush=True)
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=192)
    parser.add_argument("--budget-sec", type=float, default=5400)
    parser.add_argument("--seed", type=int, default=511)
    parser.add_argument("--rgb-size", type=int, nargs=2, default=[224, 168])
    parser.add_argument("--max-len", type=float, default=14.0)
    parser.add_argument("--facing-fraction", type=float, default=0.15)
    parser.add_argument("--steps-per-iter", type=int, default=32)
    parser.add_argument("--updates-per-iter", type=int, default=48)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer", type=int, default=180_000)
    parser.add_argument("--teacher-steps", type=int, default=600, help="control steps with every episode teacher-driven")
    parser.add_argument("--anneal-steps", type=int, default=3000)
    parser.add_argument("--beta-min", type=float, default=0.15)
    parser.add_argument("--home-fraction", type=float, default=0.0, help="episodes starting beside the office desk")
    parser.add_argument("--short-fraction", type=float, default=0.0, help="episodes on straight 0.3-1.5 m routes")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--slot-drop", type=float, default=0.0, help="chance a history slot is blanked in a training batch")
    parser.add_argument("--holdout", type=float, default=0.0, help="fraction of 1 m start cells kept for validation")
    parser.add_argument("--cell-m", type=float, default=1.0)
    parser.add_argument("--val-buffer", type=int, default=40_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--profile", action="store_true", help="CUDA-synced per-stage timings each iteration")
    parser.add_argument("--teacher", type=Path, default=Path("/workspace/elegoo/p4c/driver_v2.onnx"))
    parser.add_argument("--da2", default="/workspace/elegoo/p4c/da2")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    wp.config.enable_backward = False
    torch.backends.cudnn.benchmark = True
    sim_cfg = SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        run(args, SimulationContext(sim_cfg))


if __name__ == "__main__":
    main()
