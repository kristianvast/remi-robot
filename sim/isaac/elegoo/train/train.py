"""Train the building driver with the Isaac Lab 3.0 container's RSL-RL (rsl-rl-lib 5.5.1) PPO.

  sim/cloud/run_headless.sh drv/elegoo/train/train.py --num-envs 1024 --budget-sec 3000 \
      --output /workspace/elegoo/drv/run --viz none
  fine-tune the shipped policy (v2):  ... --finetune drv/elegoo/train/results/driver_ppo.pt --init-std 0.2
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
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from elegoo.train.depth import NAMES
from elegoo.train.model import FEATURES, HISTORY, PPO_CFG, pack, wheels

STAGES = (3.0, 5.0, 8.0, 14.0)
TICK_S = 0.01


class DriverVec(VecEnv):
    """RSL-RL view of DriverEnv; the policy's raw output is a residual mapped to PWM by model.wheels."""

    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.spec.count
        self.num_actions = 2
        self.max_episode_length = 1200
        self.episode_length_buf = env.elapsed
        self.cfg = {}
        self.history = torch.zeros((self.num_envs, HISTORY, FEATURES), device=self.device)
        # episodes, successes, collisions, metres, seconds, reward sum, contact onsets, seconds within the margin
        self.metrics = torch.zeros(8, device=self.device)
        self.env.reset()
        self.env.advance(torch.zeros((self.num_envs, 2), device=self.device))
        self._observe()

    def _observe(self) -> None:
        depth, self.aux = self.env.policy_obs()
        vector, self.history = pack(depth, self.aux, self.history)
        self.obs = TensorDict({"driver": vector}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self.obs

    def step(self, actions: torch.Tensor):
        reward, done = self.env.episode_step(wheels(actions, self.aux))
        env = self.env
        timeouts = env.timeout.clone()
        duration = env.elapsed * env.control_dt
        self.metrics += torch.stack((done.sum(), env.success.sum(), env.collision.sum(),
                                     (env.travel * done).sum(), (duration * done).sum(), reward.sum(),
                                     (env.contacts * done).sum(), (env.near_steps * env.control_dt * done).sum()))
        ended = done.nonzero().flatten()
        if len(ended):
            env.reset(ended)
            self.history[ended] = 0
        self.history[env.restarted] = 0
        self._observe()
        return self.obs, reward, done, {"time_outs": timeouts}


def probe(env, output: Path) -> None:
    """Geometry check on the VM: lens draws, true depth images, clearances and the bumper origin height."""
    p = env.privileged()
    bumper, nearest = env.clearance(p["heading_rad"])
    cars = torch.arange(min(8, env.spec.count), device=env.device)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "probe.npz", true_depth=env.true_depth()[cars].cpu().numpy(),
                        view=env.view[cars].cpu().numpy(), ticks=env.ticks[cars].cpu().numpy(),
                        bumper=bumper[cars].cpu().numpy(), nearest=nearest[cars].cpu().numpy(),
                        range_cm=env.sonar_cm()[cars].cpu().numpy(), pos=p["pos_w"][cars].cpu().numpy(),
                        bumper_origin=env.bumper.data.pos_w.torch[cars].cpu().numpy(),
                        camera_pos=env.depth_camera.data.pos_w.torch[cars].cpu().numpy())
    print("DRIVER_PROBE", json.dumps({
        "view": env.view[cars].round(decimals=3).tolist(), "ticks": env.ticks[cars].tolist(),
        "bumper_m": bumper[cars].round(decimals=3).tolist(), "nearest_m": nearest[cars].round(decimals=3).tolist(),
        "range_cm": env.sonar_cm()[cars].tolist(),
        "bumper_origin_z": env.bumper.data.pos_w.torch[cars, 2].round(decimals=4).tolist(),
        "camera_z": env.depth_camera.data.pos_w.torch[cars, 2].round(decimals=4).tolist(),
        "centre_depth_m": env.true_depth()[cars, 60, 80].round(decimals=3).tolist()}), flush=True)


def run(args, sim) -> None:
    from elegoo.train.driver_env import DriverEnv, RunSpec
    from elegoo.world import CameraSpec

    started = time.monotonic()
    torch.manual_seed(args.seed)
    rgb = args.rgb or args.visual
    camera = CameraSpec(outputs=("rgb",), width=args.rgb_size[0], height=args.rgb_size[1]) if rgb else CameraSpec(outputs=())
    env = DriverEnv(sim, RunSpec(count=args.num_envs, randomize=True, seed=args.seed, camera=camera, visual=args.visual,
                                 depth_corruptions=NAMES - frozenset(args.ablate),
                                 collision_penalty=args.collision_penalty,
                                 step_ticks=tuple(round(ms / 1000 / TICK_S) for ms in args.step_ms),
                                 random_camera=not args.fixed_camera, margin_m=args.margin,
                                 proximity_weight=args.proximity_weight, contact_weight=args.contact_weight,
                                 facing_fraction=args.facing_fraction,
                                 exclude=tuple(tuple(float(v) for v in box.split(",")) for box in args.exclude)))
    print(f"DRIVER_READY envs={args.num_envs} startup_s={time.monotonic() - started:.1f} "
          f"usable_train_routes={int(env.usable.sum())}/{len(env.usable)}", flush=True)
    stage = args.stage
    if args.resume is not None:
        stage = json.loads((args.resume.parent / "curriculum.json").read_text())["stage"]
    env.max_len = STAGES[stage]
    vec = DriverVec(env)
    if args.bench > 0:
        zero = torch.zeros((args.num_envs, 2), device=env.device)
        with torch.inference_mode():
            for _ in range(4):
                vec.step(zero)
            torch.cuda.synchronize()
            begin = time.monotonic()
            for _ in range(args.bench):
                vec.step(zero)
            torch.cuda.synchronize()
            duration = time.monotonic() - begin
            probe(env, args.output)
        free, total = torch.cuda.mem_get_info()
        print(f"DRIVER_BENCH envs={args.num_envs} vector_steps_s={args.bench / duration:.3f} "
              f"car_steps_s={args.num_envs * args.bench / duration:.1f} GPU_MiB={(total - free) / 2**20:.0f}",
              flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(vec, json.loads(json.dumps(PPO_CFG)), log_dir=None, device=env.device)
    checkpoint = args.resume or args.finetune
    if checkpoint is not None:
        runner.load(str(checkpoint))
    else:
        # Start at the heading-following prior: a near-zero residual.
        with torch.no_grad():
            last = [m for m in runner.alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
            last.weight.mul_(0.05)
            last.bias.zero_()
    if args.init_std is not None:
        std = [(n, p) for n, p in runner.alg.actor.named_parameters() if n.endswith(("std", "std_param"))]
        if not std:
            raise RuntimeError(f"No std parameter in {[n for n, _ in runner.alg.actor.named_parameters()]}")
        with torch.no_grad():
            for _, p in std:
                p.fill_(args.init_std)
        print(f"DRIVER_STD {[n for n, _ in std]} = {args.init_std}", flush=True)
    first = runner.current_learning_iteration + (1 if checkpoint is not None else 0)
    with (args.output / "learning.csv").open("a", newline="", buffering=1) as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(("iteration", "segment_seed", "collision_penalty", "stage_max_m", "elapsed_s", "episodes",
                        "success_rate", "collision_rate", "collisions_per_m", "mean_speed_m_s", "reward_per_step",
                        "action_std", "contacts_per_m", "near_fraction"))
        for iteration in range(first, first + args.iters):
            runner.current_learning_iteration = iteration
            vec.metrics.zero_()
            runner.learn(1)
            n, success, collision, distance, seconds, reward, contacts, near = vec.metrics.tolist()
            elapsed = time.monotonic() - started
            rate = success / n if n else 0.0
            row = (iteration, args.seed, args.collision_penalty, env.max_len, round(elapsed, 1), int(n), round(rate, 4),
                   round(collision / n, 4) if n else "", round(collision / distance, 4) if distance else "",
                   round(distance / seconds, 4) if seconds else "",
                   round(reward / (env.spec.count * PPO_CFG["num_steps_per_env"]), 4),
                   round(runner.alg.get_policy().output_std.mean().item(), 4),
                   round(contacts / distance, 4) if distance else "", round(near / seconds, 4) if seconds else "")
            w.writerow(row)
            if rate >= 0.7 and stage + 1 < len(STAGES) and n >= 100:
                stage += 1
                env.max_len = STAGES[stage]
                print(f"DRIVER_STAGE iteration={iteration} max_len={env.max_len}", flush=True)
            if iteration % 5 == 0:
                print("DRIVER_LEARN", row, flush=True)
            last_run = elapsed >= args.budget_sec or iteration + 1 == first + args.iters
            if iteration % 25 == 0 or last_run:
                runner.save(str(args.output / f"ppo_{iteration:04d}.pt"))
                runner.save(str(args.output / "ppo.pt"))
                (args.output / "curriculum.json").write_text(json.dumps({"stage": stage, "iteration": iteration}))
            if last_run:
                print(f"DRIVER_TRAIN_DONE iteration={iteration} seconds={elapsed:.1f}", flush=True)
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--budget-sec", type=float, default=3000)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--bench", type=int, default=0)
    parser.add_argument("--ablate", nargs="*", choices=sorted(NAMES), default=[])
    parser.add_argument("--collision-penalty", type=float, default=15.0)
    parser.add_argument("--step-ms", type=int, nargs="+", default=[70, 100], help="control steps drawn per episode")
    parser.add_argument("--fixed-camera", action="store_true", help="nominal lens for every car")
    parser.add_argument("--margin", type=float, default=0.10, help="proximity cost below this clearance (m)")
    parser.add_argument("--proximity-weight", type=float, default=0.3)
    parser.add_argument("--contact-weight", type=float, default=1.0)
    parser.add_argument("--facing-fraction", type=float, default=0.15)
    parser.add_argument("--exclude", nargs="*", default=[], help="x0,y0,x1,y1 boxes; routes touching them are not drawn")
    parser.add_argument("--stage", type=int, default=0, help="curriculum stage to start at (not with --resume)")
    parser.add_argument("--output", type=Path, default=Path("driver-run"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path, help="continue a run (its curriculum.json sets the stage)")
    group.add_argument("--finetune", type=Path, help="start from another run's checkpoint in a new output")
    parser.add_argument("--init-std", type=float, help="reset the action std (exploration) after loading")
    parser.add_argument("--rgb", action="store_true", help="also render each car's RTX RGB camera every step")
    parser.add_argument("--visual", action="store_true",
                        help="RGB camera in the textured visual scene with appearance randomisation (implies --rgb)")
    parser.add_argument("--rgb-size", type=int, nargs=2, default=[160, 120], metavar=("W", "H"))
    add_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = args.rgb or args.visual
    wp.config.enable_backward = False
    sim_cfg = SimulationCfg(dt=TICK_S, device=args.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        run(args, SimulationContext(sim_cfg))


if __name__ == "__main__":
    main()
