"""The driver contract of DRIVER_API.md as code, and the loader every harness run uses.

A driver sees exactly these two dataclasses per 100 ms step. Load one with load_driver(name):
"hand" and "policy_v2_veto" are today's drivers (sim/cmdtest/baselines.py); any other name is
sim/models/<name>/driver.py's make_driver().
"""

import importlib.util
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PWM_MAX = 150


@dataclass(frozen=True, slots=True)
class CarObs:
    rgb: np.ndarray | None          # uint8 (480, 640, 3), None on a dropped frame
    rgb_age_s: float                # time since that frame was captured
    depth: np.ndarray | None        # float32 (120, 160), realdepth observe(rgb); None when rgb is None
    range_cm: float | None          # ultrasonic, cap 150; None when not read this step
    gyro_rate: float                # rad/s, CCW +
    gyro_heading: float             # rad, integrated gyro (drifts)
    last_action: tuple[int, int]    # last PWM sent (left, right)
    t: float                        # s


@dataclass(frozen=True, slots=True)
class DriveCommand:
    kind: Literal["goal"]
    goal_xy: tuple[float, float]    # (forward m, left m), car frame, from the real estimate
    target_xy: tuple[float, float]  # lookahead waypoint, same frame; = goal_xy without a route
    heading_error: float            # rad, direction of target_xy, CCW +
    how_far: int                    # 0 NEAR < 0.5 m, 1 MEDIUM, 2 FAR > 2 m (to target_xy)
    speed_limit: float              # 0..1
    mode: Literal["go_to", "forward", "back", "explore", "patrol"]


class VisionDriver(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, obs: CarObs, cmd: DriveCommand) -> tuple[int, int]: ...


CONTRACT_FIELDS = {"CarObs": [f.name for f in fields(CarObs)], "DriveCommand": [f.name for f in fields(DriveCommand)]}


def load_driver(name: str, **kw) -> VisionDriver:
    if name in ("hand", "policy_v2_veto"):
        from .baselines import make_baseline

        return make_baseline(name)
    path = ROOT / "sim" / "models" / name / "driver.py"
    if not path.exists():
        raise FileNotFoundError(f"no driver {name!r}: expected {path} with make_driver()")
    spec = importlib.util.spec_from_file_location(f"sim.models.{name}.driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_driver(**kw)


def clip_pwm(action) -> tuple[int, int]:
    left, right = action
    return int(np.clip(round(float(left)), -PWM_MAX, PWM_MAX)), int(np.clip(round(float(right)), -PWM_MAX, PWM_MAX))
