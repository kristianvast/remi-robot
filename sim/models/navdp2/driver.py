"""NavDP camera driver, p7 retrain (navdp2), behind sim/cmdtest/DRIVER_API.md: CarObs + DriveCommand in, signed PWM out.

    from sim.models.navdp2.driver import make_driver   # or sim/cmdtest/api.load_driver("navdp2")
    driver = make_driver()            # fine-tuned weights, MPS on the Mac, CUDA on a VM
    driver.reset(seed=0)
    left, right = driver.act(obs, cmd)

CarObs and DriveCommand are read by attribute name only (the fields in DRIVER_API.md), so any object with those
fields works. See README.md for the input / output mapping. mode "back" is not learned: a forward camera cannot
see behind the car, so it reverses straight on the gyro heading held from the start of the command.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from policy import PWM_MAX, NavDPPolicy, autocast_dtype  # noqa: E402
from xnavdp.net import load  # noqa: E402

WEIGHTS = HERE / "weights" / "navdp2_elegoo.pt"
BACK_PWM, STEER_PWM_PER_DEG, STEER_MAX = 110, 4.0, 60   # sim/cmdtest/baselines.py drive.py constants
NEW_COMMAND_M = 0.3                                    # goal jump that marks a new back command


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class NavDPDriver:
    name = "navdp2"

    def __init__(self, weights=WEIGHTS, device: str | None = None, samples: int = 8, plan_every: int = 2,
                 v_max: float = 0.40, seed: int = 0) -> None:
        self.device = device or default_device()
        self.net = load(weights, self.device, autocast_dtype(self.device))
        self.generator = torch.Generator(device=self.device)
        self.policy = NavDPPolicy(self.net, 1, samples=samples, plan_every=plan_every, generator=self.generator,
                                  v_max=v_max)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.policy.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        self.hold_heading = None
        self.back_goal_m = None

    def back(self, obs, cmd) -> tuple[int, int]:
        """Straight reverse at BACK_PWM, steering back onto the heading held since the command began."""
        goal_m = math.hypot(*cmd.goal_xy)
        if self.hold_heading is None or goal_m > self.back_goal_m + NEW_COMMAND_M:
            self.hold_heading = float(obs.gyro_heading)
        self.back_goal_m = goal_m
        drift_deg = math.degrees(wrap(float(obs.gyro_heading) - self.hold_heading))
        steer = max(-STEER_MAX, min(STEER_MAX, STEER_PWM_PER_DEG * drift_deg))
        clip = lambda v: int(round(max(-PWM_MAX, min(PWM_MAX, v))))  # noqa: E731
        return clip(-BACK_PWM + steer), clip(-BACK_PWM - steer)

    def act(self, obs, cmd) -> tuple[int, int]:
        if cmd.mode == "back":
            return self.back(obs, cmd)
        if self.hold_heading is not None:
            self.reset()
        dev = self.device
        rgb = None if obs.rgb is None else torch.from_numpy(np.ascontiguousarray(obs.rgb))[None].to(dev)
        depth = None if obs.depth is None else torch.from_numpy(np.asarray(obs.depth, dtype=np.float32))[None].to(dev)
        scalar = lambda v: torch.tensor([float(v)], device=dev)  # noqa: E731
        pwm = self.policy.step(
            rgb, torch.tensor([rgb is not None], device=dev),
            depth, torch.tensor([depth is not None], device=dev),
            torch.tensor([list(cmd.target_xy)], dtype=torch.float32, device=dev),
            scalar(cmd.speed_limit), scalar(obs.gyro_heading),
            scalar(np.inf if obs.range_cm is None else obs.range_cm),
            torch.tensor([list(obs.last_action)], dtype=torch.float32, device=dev),
            rgb_age=scalar(obs.rgb_age_s), gyro_rate=scalar(obs.gyro_rate),
        )
        left, right = pwm[0].tolist()
        return int(left), int(right)


def make_driver(**kw) -> NavDPDriver:
    return NavDPDriver(**kw)
