"""The navigator's driver slot for a camera driver of sim/cmdtest/DRIVER_API.md (sim/models/<name>/driver.py).

The navigator keeps localising, planning, marking obstacles and recovering; the camera driver only steers. Its
act(depth, range, rate, last, err, how_far, limit) becomes driver.act(CarObs, DriveCommand) with the goal and the
line-of-sight target in the car frame of the localiser's estimate, as sim/cmdtest/tools.py's DriverShim does in
the command test harness. The model loads on first use and once per process.
"""

import importlib.util
from dataclasses import dataclass
from functools import cache
from math import cos, isnan, sin
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sim.macsim.simulator import Observation

MODELS = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True, slots=True)
class CameraObservation(Observation):
    """An observation with the camera frame its depth was estimated from."""
    rgb: np.ndarray | None = None     # uint8 (480, 640, 3), RGB, un-mirrored like the renders
    rgb_age_s: float = 0.             # time since that frame was captured


@cache
def _model(name: str):
    path = MODELS / name / "driver.py"
    spec = importlib.util.spec_from_file_location(f"sim.models.{name}.driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_driver()


def _car_frame(x: float, y: float, yaw: float, gx: float, gy: float) -> tuple[float, float]:
    dx, dy = gx - x, gy - y
    return cos(yaw) * dx + sin(yaw) * dy, -sin(yaw) * dx + cos(yaw) * dy


class VisionDriver:
    def __init__(self, name: str):
        self.name = name
        self.nav = None
        self.target: tuple[float, float] | None = None   # map frame, from the last plan

    def bind(self, nav, command):
        """Follow `nav` and return its command function, wrapped to remember the line-of-sight target."""
        self.nav = nav

        def planned(pose):
            err, dist, path = command(pose)
            if not isnan(err):
                x, y, yaw = pose
                self.target = (x + dist * cos(yaw + err), y + dist * sin(yaw + err))
            return err, dist, path

        return planned

    def reset(self) -> None:
        _model(self.name).reset()

    def act(self, depth, range_cm, gyro_rate, last_action, heading_error, how_far, speed_limit) -> tuple[int, int]:
        nav = self.nav
        x, y, yaw = nav.est.pose
        goal = _car_frame(x, y, yaw, *nav.goal)
        target = _car_frame(x, y, yaw, *self.target) if self.target is not None else goal
        seen = nav.obs
        obs = SimpleNamespace(
            rgb=getattr(seen, "rgb", None) if depth is not None else None,  # None while the head looks aside
            rgb_age_s=getattr(seen, "rgb_age_s", 0.), depth=depth, range_cm=range_cm, gyro_rate=gyro_rate,
            gyro_heading=seen.gyro, last_action=last_action, t=nav.t)
        cmd = SimpleNamespace(kind="goal", goal_xy=goal, target_xy=target, heading_error=float(heading_error),
                              how_far=int(how_far), speed_limit=float(speed_limit), mode="go_to")
        return _model(self.name).act(obs, cmd)
