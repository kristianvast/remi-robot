"""Planar robot dynamics; library pose is metres and CCW radians."""

from collections import deque
from dataclasses import dataclass
from math import cos, isfinite, radians, remainder, sin, tau

import numpy as np

from .depth import Corruptions, encode, pinhole, sonar
from .drive import Drive, TICK
from .world import Pose, floor, scene_meta


@dataclass(frozen=True, slots=True)
class Observation:
    gyro: float
    range_cm: int
    depth: np.ndarray | None
    head_angle: float
    on_ground: bool


class Simulator:
    """Advance explicitly, then observe. No physics or hardware I/O in observe."""

    def __init__(self, *, seed: int = 0, randomize: bool = False, corrupt: bool = False,
                 scale: float | None = None, failure: float = .03):
        self.rng = np.random.default_rng(seed)
        self.randomize, self.corrupt = randomize, corrupt
        self.scale, self.failure = scale, failure
        self.floor = floor()
        self.reset()

    def reset(self, pose: Pose | None = None) -> None:
        """Place the chassis and zero integrated gyro; redraw episode parameters."""
        if pose is None:
            first = scene_meta()["spawn_poses"][0]
            pose = first["x"], first["y"], first["yaw"]
        if not all(map(isfinite, pose)) or not self.floor.valid(*pose[:2]):
            raise ValueError(f"Cannot reset in collision: {pose}")
        self.pose = tuple(map(float, pose))
        self.drive = Drive(self.rng if self.randomize else None)
        self.image_error = Corruptions(self.rng, self.scale, self.failure) if self.corrupt else None
        self.head = self.target = self.gyro = 0.
        self.bumped = False
        self._frames: deque[tuple[int, np.ndarray]] = deque()
        self._last_frame = self._last_range = -1
        self._range = 150
        self._depth = None

    @property
    def time(self) -> float:
        return self.drive.ticks * TICK

    def set_head(self, angle: float) -> None:
        """Set pan target in radians, positive left. Full sweep takes ~0.45 s."""
        self.target = float(np.clip(angle, -radians(80), radians(80)))

    def step(self, left_pwm: int | None = None, right_pwm: int | None = None, *,
             dt: float = .07) -> None:
        """Advance a multiple of 10 ms. Both None drops packets, unlike (0, 0)."""
        count = round(dt / TICK)
        if count <= 0 or abs(count*TICK-dt) > 1e-8:
            raise ValueError(f"Step must be a positive multiple of 10 ms; got {dt}")
        if left_pwm is None and right_pwm is None:
            action = None
        elif left_pwm is not None and right_pwm is not None:
            action = (max(-255, min(255, round(left_pwm))), max(-255, min(255, round(right_pwm))))
        else:
            raise ValueError("Both wheel PWM values or neither are required")
        p = self.drive.p
        limit, bias, noise = (radians(p[k]) for k in
                              ("gyro_rate_limit_deg_s", "gyro_bias_deg_s", "gyro_noise_deg_s"))
        self.bumped = False
        x, y, yaw = self.pose
        turn = radians(360)*TICK
        for _ in range(count):
            v, w = self.drive.tick(action)
            mid = yaw + w*TICK/2
            a, b = x + v*TICK*cos(mid), y + v*TICK*sin(mid)
            if self.floor.valid_point(a, b):
                x, y = a, b
            elif abs(v) > .001:
                self.bumped = True
            yaw = remainder(yaw + w*TICK, tau)
            self.gyro += (max(-limit, min(limit, w+bias+self.rng.normal(0, noise)))) * TICK
            self.head += max(-turn, min(turn, self.target-self.head))
        self.pose = x, y, yaw

    def metric_depth(self) -> np.ndarray:
        """Current clean optical-axis metres, inf for a miss (diagnostics/rendering)."""
        return pinhole(self.pose, self.head, self.drive.p)[0]

    def observe(self, *, camera: bool = True) -> Observation:
        """Return the current sensors; corrupted camera output has 100 ms lag.

        Pass current gyro/head with each frame; a consumer can buffer timestamps
        to align the delayed camera. Until the first frame matures, depth is None.
        """
        tick = self.drive.ticks
        if camera and self._last_frame != tick:
            z, h = pinhole(self.pose, self.head, self.drive.p)
            if self.image_error is None:
                self._depth = encode(z)
            else:
                self._frames.append((tick, self.image_error.apply(z, h)))
                # Do not replay the same frame as new evidence.
                self._depth = None
                while self._frames and tick - self._frames[0][0] >= 10:
                    _, self._depth = self._frames.popleft()
            self._last_frame = tick
        if self._last_range != tick:
            self._range = sonar(self.pose, self.head, self.drive.p, self.rng)
            self._last_range = tick
        return Observation(self.gyro, self._range, self._depth if camera else None, self.head, True)
