"""NavDP camera driver (option A) behind sim/cmdtest/DRIVER_API.md: CarObs + DriveCommand in, signed PWM out.

    from sim.models.navdp.driver import make_driver   # or load this file by path
    driver = make_driver()            # fine-tuned weights, MPS on the Mac, CUDA on a VM
    driver.reset(seed=0)
    left, right = driver.act(obs, cmd)

CarObs and DriveCommand are read by attribute name only (the fields in DRIVER_API.md), so any object with those
fields works. See README.md for the input / output mapping.
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from policy import NavDPPolicy, autocast_dtype  # noqa: E402
from xnavdp.net import load  # noqa: E402

WEIGHTS = HERE / "weights" / "navdp_elegoo.pt"


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class NavDPDriver:
    name = "navdp"

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

    def act(self, obs, cmd) -> tuple[int, int]:
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
