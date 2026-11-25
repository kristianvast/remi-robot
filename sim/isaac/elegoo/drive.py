"""Vectorized packet arrival, watchdog, and measured PWM response."""

from math import ceil
from pathlib import Path

import torch
import yaml

CALIBRATION = Path(__file__).with_name("calibration.yaml")


class Drive:
    """Command transport and first-order drive state, independent of contacts."""

    def __init__(self, count: int, dt: float, device: str, *, randomize: bool = False) -> None:
        data = yaml.safe_load(CALIBRATION.read_text())
        self.dt = dt
        self.limit = data["measured"]["pwm_limit"]
        self.p = {
            name: (
                torch.rand(count, device=device) * (high - low) + low
                if randomize else torch.full((count,), float(nominal), device=device)
            )
            for name, (nominal, low, high) in data["parameters"].items()
        }
        self.curves = {
            name: torch.tensor(values, dtype=torch.float32, device=device)
            for name, values in data["curves"].items()
        }
        self.time = 0.0
        self.index = 0
        self.slots = ceil(data["parameters"]["command_delay_s"][2] / dt) + 2
        self.rows = torch.arange(count, device=device)
        self.delay_ticks = torch.ceil(self.p["command_delay_s"] / dt - 1e-5).long()
        self.queue = torch.zeros((self.slots, count, 2), device=device)
        self.queued = torch.zeros((self.slots, count), dtype=torch.bool, device=device)
        self.pwm = torch.zeros((count, 2), device=device)
        self.last_receive = torch.full((count,), -torch.inf, device=device)
        self.next_send = torch.zeros_like(self.last_receive)
        self.speed = torch.zeros_like(self.last_receive)
        self.omega = torch.zeros_like(self.last_receive)
        self.coast_tau = self.p["yaw_accel_lag_s"].clone()

    def reset(self, ids: torch.Tensor) -> None:
        self.queued[:, ids] = False
        self.pwm[ids] = 0
        self.last_receive[ids] = -torch.inf
        self.next_send[ids] = self.time
        self.speed[ids] = 0
        self.omega[ids] = 0
        self.coast_tau[ids] = self.p["yaw_accel_lag_s"][ids]

    def tick(self, action: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one physics tick; None means no packet, not zero PWM."""
        i, t = self.index, self.time
        if action is not None:
            send = t + 1e-6 >= self.next_send
            period = self.p["command_period_s"]
            deadline = torch.where(t - self.next_send >= period, t, self.next_send)
            self.next_send = torch.where(
                send, deadline + period, self.next_send,
            )
            slots = (i + self.delay_ticks) % self.slots
            self.queued[slots, self.rows] |= send
            previous = self.queue[slots, self.rows]
            command = (action.clamp(-1, 1) * self.limit).round()
            self.queue[slots, self.rows] = torch.where(send[:, None], command, previous)

        received = self.queued[i]
        self.pwm = torch.where(received[:, None], self.queue[i], self.pwm)
        self.last_receive = torch.where(received, t, self.last_receive)
        self.queued[i].zero_()
        self.pwm = torch.where((t - self.last_receive + 1e-6 >= self.p["watchdog_s"])[:, None], 0, self.pwm)

        left, right = self.pwm.unbind(dim=1)
        mean, diff = (left + right) / 2, (right - left) / 2
        target_speed = (
            mean.sign() * (mean.abs() - self.p["deadband_pwm"]).clamp_min(0)
            * self.p["speed_m_s_per_pwm"] * self.p["speed_scale"]
        )
        # Do not apply the spin deadband to steering corrections while already rolling.
        rolling = (left * right > 0) & (mean.abs() > self.p["deadband_pwm"])
        turn = self._interpolate(diff.abs(), "spin_pwm", "spin_deg_s")
        first_spin_pwm = self.curves["spin_pwm"][2]
        steering_slope = self.curves["spin_deg_s"][2] / first_spin_pwm
        turn = torch.where(rolling & (diff.abs() < first_spin_pwm), diff.abs() * steering_slope, turn)
        bias = self.p["straight_yaw_bias_deg_s"] * (target_speed / 0.39).clamp(-1, 1)
        target_omega = torch.deg2rad(turn * diff.sign() * self.p["turn_scale"] + bias)
        stopping = (left == 0) & (right == 0)
        fitted = self._interpolate(diff.abs(), "coast_pwm", "coast_lag_s")
        self.coast_tau = torch.where(stopping, self.coast_tau, fitted * self.p["coast_lag_scale"])
        tau = torch.where(stopping, self.coast_tau, self.p["yaw_accel_lag_s"])
        self.speed = target_speed + (self.speed - target_speed) * torch.exp(-self.dt / self.p["speed_lag_s"])
        self.omega = target_omega + (self.omega - target_omega) * torch.exp(-self.dt / tau)
        self.time = t + self.dt
        self.index = (i + 1) % self.slots
        return self.speed, self.omega

    def _interpolate(self, x: torch.Tensor, knots: str, values: str) -> torch.Tensor:
        xs, ys = self.curves[knots], self.curves[values]
        hi = torch.searchsorted(xs, x).clamp(1, len(xs) - 1)
        lo = hi - 1
        return torch.lerp(ys[lo], ys[hi], (x - xs[lo]) / (xs[hi] - xs[lo]))
