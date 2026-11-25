"""Scalar CPU port of Isaac Drive: 10 ms ticks, signed PWM, metres and radians."""

from collections import deque
from math import ceil, copysign, exp, inf, radians
from functools import lru_cache

import numpy as np
import yaml

from .world import ISAAC

PWM = tuple[int, int]
TICK = .01


@lru_cache(maxsize=1)
def calibration():
    return yaml.safe_load((ISAAC / "elegoo/calibration.yaml").read_text())


class Drive:
    def __init__(self, rng: np.random.Generator | None = None):
        data = calibration()
        self.p = {k: float(n if rng is None else rng.uniform(lo, hi))
                  for k, (n, lo, hi) in data["parameters"].items()}
        self.curves = data["curves"]
        self.ticks = 0
        self.pending: deque[tuple[int, PWM]] = deque()
        self.pwm: PWM = (0, 0)
        self.last_receive = -inf
        self.next_send = 0.
        self.speed = self.omega = 0.
        self.coast = self.p["yaw_accel_lag_s"]
        self.delay = ceil(self.p["command_delay_s"] / TICK - 1e-5)
        self.speed_alpha = exp(-TICK / self.p["speed_lag_s"])
        self.turn_alpha = exp(-TICK / self.p["yaw_accel_lag_s"])

    def tick(self, action: PWM | None) -> tuple[float, float]:
        p, curves = self.p, self.curves
        t = self.ticks * TICK
        period = p["command_period_s"]
        if action is not None and t + 1e-6 >= self.next_send:
            deadline = t if t - self.next_send >= period else self.next_send
            self.next_send = deadline + period
            self.pending.append((self.ticks + self.delay, action))
        if self.pending and self.pending[0][0] <= self.ticks:
            _, self.pwm = self.pending.popleft()
            self.last_receive = t
        if t - self.last_receive + 1e-6 >= p["watchdog_s"]:
            self.pwm = (0, 0)
        left, right = self.pwm
        mean, diff = (left + right) / 2, (right - left) / 2
        v = copysign(max(0., abs(mean) - p["deadband_pwm"]), mean) * p["speed_m_s_per_pwm"] * p["speed_scale"]
        a = abs(diff)
        if left * right > 0 and abs(mean) > p["deadband_pwm"] and a < 80:
            turn = a * (130 / 80)
        else:
            turn = np.interp(a, curves["spin_pwm"], curves["spin_deg_s"]) if a else 0.
        w = radians(copysign(turn, diff) * p["turn_scale"] +
                    p["straight_yaw_bias_deg_s"] * max(-1., min(1., v / .39)))
        if left or right:
            self.coast = np.interp(a, curves["coast_pwm"], curves["coast_lag_s"]) * p["coast_lag_scale"]
            alpha = self.turn_alpha
        else:
            alpha = exp(-TICK / self.coast)
        self.speed = v + (self.speed - v) * self.speed_alpha
        self.omega = w + (self.omega - w) * alpha
        self.ticks += 1
        return self.speed, self.omega

    def stop(self) -> None:
        """Explicit emergency packet: flush transport; retain calibrated coast."""
        self.pending.clear()
        self.pwm = (0, 0)
