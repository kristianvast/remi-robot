"""Fine-tune X-NavDP in our building: online imitation (NavOL style) from a privileged expert, with the
critic regressed to a privileged clearance score.

Each 100 ms step every car gets the real-car inputs of env.py. A car's episode is driven either by the
current policy (sampled trajectories, critic choice, tracker) or by the expert (the same tracker on the
planned route from the true pose), chosen per episode with probability beta, which decays from
--beta-start to --beta-end (DAgger). Every step a random batch of cars contributes:
- the diffusion loss toward the expert label (route from the true pose, resampled every --spacing m), at
  the observations the mixed policy actually visited;
- for cars that planned this step, a critic loss: the sampled trajectories and the expert one regressed to
  env.path_score (progress - 2 x obstacle hit - proximity on the floor-map clearance).
Frozen: the RGB DINOv2 encoder (so RGB tokens are computed once per frame). Trained: depth encoder, memory
transformer, goal encoder, both diffusion decoders, the critic.

  sim/cloud/run_headless.sh elegoo-sim-8 navdp/elegoo/navdp/train.py --checkpoint /workspace/elegoo/ckpt/x.ckpt \
      --output /workspace/elegoo/navdp/run --budget-sec 5400 --viz none
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

COLUMNS = ("step", "elapsed_s", "beta", "episodes_policy", "success_policy", "collision_policy", "contacts_per_m_policy",
           "speed_policy", "episodes_expert", "success_expert", "loss_imitation", "loss_critic", "car_steps_s")


def run(args, sim) -> None:
    from elegoo.navdp.env import RGB_SIZE, NavDPEnv, NavDPSpec
    from elegoo.train.depth import NAMES
    from elegoo.world import CameraSpec
    from policy import NavDPPolicy, Tracker
    from xnavdp.net import load

    torch.manual_seed(args.seed)
    started = time.monotonic()
    env = NavDPEnv(sim, NavDPSpec(count=args.num_envs, randomize=True, random_camera=True, seed=args.seed,
                                  visual=True, camera=CameraSpec(outputs=("rgb",), width=RGB_SIZE[0], height=RGB_SIZE[1]),
                                  depth_corruptions=NAMES, step_ticks=(10,), facing_fraction=args.facing,
                                  label_spacing_m=args.spacing,
                                  home_fraction=args.home_fraction, short_fraction=args.short_fraction,
                                  depth_scale=tuple(args.depth_scale) if args.depth_scale else None,
                                  smooth_sd=args.smooth_sd, depth_gamma=args.depth_gamma,
                                  exclude=((-0.05, -7.35, 0.6, -6.7),)))
    env.max_len = args.max_route_m
    device = env.device
    net = load(args.checkpoint, device, torch.float32)
    for p in net.rgbd_encoder.rgb_model.parameters():
        p.requires_grad_(False)
    for head in (net.q1_target_heads, net.q2_target_heads):
        for p in head.parameters():
            p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    print(f"TRAIN_READY envs={args.num_envs} trainable_M={sum(p.numel() for p in params) / 1e6:.1f} "
          f"startup_s={time.monotonic() - started:.1f}", flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    policy = NavDPPolicy(net, args.num_envs, samples=args.samples, plan_every=args.plan_every, generator=generator)
    expert = Tracker(args.num_envs, device)
    n = args.num_envs
    by_expert = torch.zeros(n, dtype=torch.bool, device=device)
    env.reset()
    env.advance(torch.zeros((n, 2), device=device))
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / "learning.csv").open("a", newline="", buffering=1)
    writer = csv.writer(log)
    if log.tell() == 0:
        writer.writerow(COLUMNS)
    window = {k: [] for k in ("li", "lc")}
    stats = torch.zeros((2, 5), device=device)
    step, last_save, window_start = 0, time.monotonic(), time.monotonic()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    def beta_now() -> float:
        f = min(1.0, (time.monotonic() - started) / (0.7 * args.budget_sec))
        return args.beta_start + (args.beta_end - args.beta_start) * f

    def restart(ids: torch.Tensor) -> None:
        policy.reset(ids)
        expert.reset(ids)
        by_expert[ids] = torch.rand(len(ids), device=device) < beta_now()

    restart(env.ids)
    while time.monotonic() - started < args.budget_sec:
        inp = env.car_inputs()
        label = env.expert_path()
        net.eval()
        with autocast:
            pwm_policy = policy.step(inp["rgb"], inp["rgb_valid"], inp["depth"], inp["depth_valid"], inp["target_xy"],
                                     inp["speed_limit"], inp["heading"], inp["range_cm"], inp["last_pwm"],
                                     rgb_age=inp["rgb_age"], gyro_rate=inp["gyro_rate"])
        expert.advance(inp["heading"], inp["last_pwm"])
        expert.set_path(env.ids, label, inp["heading"], torch.zeros((n, 3), device=device))
        pwm = torch.where(by_expert[:, None], expert.act(inp["speed_limit"], inp["range_cm"]), pwm_policy)

        net.train()
        net.rgbd_encoder.rgb_model.eval()
        ready = (env.age > 2).nonzero().squeeze(1)
        sub = ready[torch.randperm(len(ready), device=device)[: args.batch]]
        if len(sub) >= 4:
            with autocast:
                depth_tokens = net.rgbd_encoder.depth_tokens(policy.depth[sub])
                memory = net.rgbd_encoder.memory(policy.last_memory[sub], depth_tokens)
                loss_im = net.imitation_loss(memory, inp["target_xy"][sub], label[sub])
                loss = loss_im
                planned = policy.last_due
                pos = torch.full((n,), -1, dtype=torch.long, device=device)
                pos[planned] = torch.arange(len(planned), device=device)
                keep = (pos[sub] >= 0).nonzero().squeeze(1)
                if len(keep) and args.critic_weight > 0:
                    cars = sub[keep]
                    paths = policy.last_paths[pos[cars]]
                    ref = torch.cat((label[cars], torch.zeros_like(label[cars][..., :1])), -1)[:, None]
                    paths = torch.cat((paths, ref), 1)
                    m = paths.shape[1]
                    score = env.path_score(cars, paths[..., :2], label)
                    loss_q = net.critic_loss(memory[keep].repeat_interleave(m, 0),
                                             inp["target_xy"][cars].repeat_interleave(m, 0),
                                             paths.flatten(0, 1), score.flatten())
                    loss = loss + args.critic_weight * loss_q
                    window["lc"].append(loss_q.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            window["li"].append(loss_im.item())

        _, done = env.episode_step(pwm / 255)
        ended = done.nonzero().squeeze(1)
        if len(ended):
            mode = by_expert[ended].long()
            values = torch.stack((torch.ones_like(env.travel[ended]), env.success[ended].float(),
                                  env.collision[ended].float(), env.contacts[ended], env.travel[ended]), 1)
            stats.index_add_(0, mode, values)
            env.reset(ended)
            restart(ended)
        if len(env.restarted):
            policy.reset(env.restarted)
            expert.reset(env.restarted)
        step += 1
        if step % args.log_every == 0:
            elapsed = time.monotonic() - started
            rate = n * args.log_every / (time.monotonic() - window_start)
            p_n, p_s, p_c, p_k, p_m = stats[0].tolist()
            e_n, e_s = stats[1, 0].item(), stats[1, 1].item()
            row = (step, round(elapsed, 1), round(beta_now(), 3), int(p_n), round(p_s / p_n, 4) if p_n else "",
                   round(p_c / p_n, 4) if p_n else "", round(p_k / p_m, 4) if p_m else "", "",
                   int(e_n), round(e_s / e_n, 4) if e_n else "",
                   round(float(np.mean(window["li"])), 4) if window["li"] else "",
                   round(float(np.mean(window["lc"])), 4) if window["lc"] else "", round(rate, 1))
            writer.writerow(row)
            print("TRAIN_LOG", row, flush=True)
            stats.zero_()
            window = {k: [] for k in window}
            window_start = time.monotonic()
        if time.monotonic() - last_save > args.save_every_sec:
            save(net, args.output / f"ckpt_{step:06d}.pt", step)
            last_save = time.monotonic()
    save(net, args.output / f"ckpt_{step:06d}.pt", step)
    save(net, args.output / "final.pt", step)
    print(f"TRAIN_DONE steps={step} seconds={time.monotonic() - started:.1f}", flush=True)


def save(net, path: Path, step: int) -> None:
    state = {k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}
    torch.save({"model": state, "step": step}, path)
    print(f"TRAIN_SAVED {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=160)
    parser.add_argument("--budget-sec", type=float, default=5400)
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--plan-every", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=0.05,
                        help="expert label waypoint spacing (m); zero-shot NavDP paths have a median of 0.046 m")
    parser.add_argument("--critic-weight", type=float, default=0.5)
    parser.add_argument("--beta-start", type=float, default=0.7)
    parser.add_argument("--beta-end", type=float, default=0.1)
    parser.add_argument("--facing", type=float, default=0.15)
    parser.add_argument("--max-route-m", type=float, default=8.0)
    parser.add_argument("--home-fraction", type=float, default=0.0, help="episodes on home_train routes")
    parser.add_argument("--short-fraction", type=float, default=0.0, help="episodes on short_train routes")
    parser.add_argument("--depth-scale", type=float, nargs=2, metavar=("LO", "HI"),
                        help="per-episode global depth scale range, log-uniform (default: the spec's 0.75-1.33)")
    parser.add_argument("--smooth-sd", type=float, default=0.0, help="mean log sd of the smooth depth error field")
    parser.add_argument("--depth-gamma", type=float, default=0.0, help="max depth-range compression exponent")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every-sec", type=float, default=900)
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
