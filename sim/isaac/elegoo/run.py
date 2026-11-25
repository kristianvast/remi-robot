"""Run match experiments, camera proof, or a timed vector rollout."""

import argparse
from pathlib import Path
import sys

import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxCfg

wp.config.enable_backward = False
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("match", "proof", "bench", "wall"))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--minimal-rtx", action="store_true", help="Use RTX Minimal diffuse RGB in benchmark mode.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    add_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = args.mode in ("proof", "bench")
    cfg = SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg())
    with launch_simulation(cfg, args):
        from elegoo.environment import ElegooEnv, EnvSpec
        from elegoo.experiments import benchmark, match, proof, wall
        from elegoo.world import CameraSpec

        if args.mode != "wall":
            args.output.mkdir(parents=True, exist_ok=True)
        outputs = ("rgb", "depth")
        if args.minimal_rtx:
            if args.mode != "bench":
                parser.error("--minimal-rtx is only supported for bench")
            outputs = ("simple_shading_diffuse_mdl", "depth")
        camera = CameraSpec(
            outputs=outputs if args.enable_cameras else (),
            width=args.width, height=args.height,
            far_clip_m=25.0 if args.mode == "proof" else 6.0,
        )
        count = 7 if args.mode == "match" else 1 if args.mode in ("wall", "proof") else args.num_envs
        env = ElegooEnv(SimulationContext(cfg), EnvSpec(
            count=count, camera=camera, obstacles=args.mode != "match", randomize=args.randomize,
        ))
        if args.mode == "match":
            match(env, args.output)
        elif args.mode == "proof":
            proof(env, args.output)
        elif args.mode == "bench":
            benchmark(env, args)
        else:
            wall(env)


if __name__ == "__main__":
    main()
