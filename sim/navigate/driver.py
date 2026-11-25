"""Local driver behind the interface the learned policy (brief-driver.md) will implement.

act(depth, range_cm, gyro_rate, last_action, heading_error, how_far, speed_limit) -> (left, right)
  depth          encoded (120, 160) float32 per observation_spec.md, or None (no frame / head turned)
  range_cm       ultrasonic reading in cm (cap 150), or None
  gyro_rate      yaw rate rad/s, CCW positive
  last_action    (left, right) PWM sent last step
  heading_error  rad, direction of the next waypoint relative to the car's heading, CCW positive
  how_far        NEAR (< 0.5 m), MEDIUM, FAR (> 2 m) to the goal
  speed_limit    0..1
"""

from math import cos, degrees, radians, sin
from typing import Protocol

import numpy as np

from sim.macsim.depth import FX, PAN_X, Z_FAR, Z_NEAR

NEAR, MEDIUM, FAR = 0, 1, 2
PITCH = radians(20)
CAM_H, CAM_FWD = .126, .08
LOW, HIGH = .045, .30              # obstacle heights that matter to a 14 cm car
_ROWS = np.arange(70, 120)
_V = (60 - (_ROWS + .5))[:, None] / FX
_U = (np.arange(160) + .5 - 80)[None, :] / FX
_FWD = np.cos(PITCH) - np.sin(PITCH) * _V
_UP = np.sin(PITCH) + np.cos(PITCH) * _V


def depth_points(depth: np.ndarray, reach: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Obstacle points (forward, left) in metres relative to the base centre, from the lower rows."""
    e = depth[_ROWS]
    z = Z_NEAR * np.exp((1 - e) * np.log(Z_FAR / Z_NEAR))
    height = CAM_H + z * _UP
    ok = (e > .02) & (height > LOW) & (height < HIGH)
    forward = (z * _FWD + CAM_FWD)[ok]
    left = -(z * _U)[ok]
    near = forward < reach
    return forward[near], left[near]


class Driver(Protocol):
    def reset(self) -> None: ...

    def act(self, depth: np.ndarray | None, range_cm: float | None, gyro_rate: float,
            last_action: tuple[int, int], heading_error: float, how_far: int,
            speed_limit: float) -> tuple[int, int]: ...


def motion(left: float, right: float) -> tuple[float, float]:
    """Nominal (v m/s, w rad/s) for a PWM pair, from calibration.yaml's measured response."""
    mean, diff = (left + right) / 2, (right - left) / 2
    v = np.sign(mean) * max(0., abs(mean) - 12.5) * .004
    if left * right > 0 and abs(mean) > 12.5 and abs(diff) < 80:
        w = abs(diff) * 130 / 80
    else:
        w = float(np.interp(abs(diff), [0, 12.5, 80, 95, 110, 145, 255], [0, 0, 130, 150, 200, 265, 350]))
    return float(v), radians(np.sign(diff) * w)


class HandDriver:
    """Steer toward the waypoint heading on arcs that stay clear of recently seen obstacles.

    Obstacle points from depth (heights 4.5-30 cm) and the range sensor are kept for 3 s in the
    car's frame, moved by dead reckoning from last_action and gyro_rate, because the camera sees
    neither the car's sides nor low things within ~0.3 m. Candidate (speed, turn) commands are
    rolled out for 1 s and rejected if the car's 0.135 m circle comes within 2 cm of a point.
    """

    CRUISE = {NEAR: 55, MEDIUM: 85, FAR: 105}
    MEMORY_S = 3.
    BODY = .155

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.pts = np.zeros((0, 2))
        self.born = np.zeros(0)
        self.t = 0.
        self.front = np.inf

    def _remember(self, depth, range_cm, gyro_rate, last_action) -> None:
        dt = .1
        self.t += dt
        v, _ = motion(*last_action)
        a = -gyro_rate * dt
        p = self.pts - (v * dt, 0.)
        self.pts = np.column_stack((p[:, 0] * cos(a) - p[:, 1] * sin(a), p[:, 0] * sin(a) + p[:, 1] * cos(a)))
        keep = (self.t - self.born < self.MEMORY_S) & (np.hypot(*self.pts.T) < 1.5)
        self.pts, self.born = self.pts[keep], self.born[keep]
        sonar = None if range_cm is None else range_cm / 100
        new = []
        if depth is not None:
            forward, left = depth_points(depth, reach=1.2)
            columns = len(np.unique(np.round(left / np.maximum(forward, .1) * FX)))
            centre = forward[np.abs(left) < .1]
            # A failed frame ("all near" or a random field) covers the whole view while the range
            # sensor still sees free space; real walls agree with the range sensor.
            fake = (columns > 120 and sonar is not None and len(centre)
                    and np.median(centre) + .35 < min(sonar, 1.5))
            if len(forward) and not fake:
                cells = np.unique(np.round(np.column_stack((forward, left)) / .03), axis=0) * .03
                new.append(cells)
        if sonar is not None and sonar < 1.0:
            cone = np.radians([-10, 0, 10])
            new.append(np.column_stack((PAN_X + (sonar + .022) * np.cos(cone), (sonar + .022) * np.sin(cone))))
        if new:
            add = np.vstack(new)
            self.pts = np.vstack((self.pts, add))
            self.born = np.concatenate((self.born, np.full(len(add), self.t)))

    def _clear(self, left: float, right: float, horizon: float = 1.) -> float:
        """Closest approach (m) of the car centre to remembered points along the commanded arc."""
        pts = self.pts
        if not len(pts):
            return np.inf
        v, w = motion(left, right)
        s = np.arange(1, int(horizon / .05) + 1) * .05
        if abs(w) < 1e-3:
            xy = np.column_stack((v * s, np.zeros_like(s)))
        else:
            xy = np.column_stack((v / w * np.sin(w * s), v / w * (1 - np.cos(w * s))))
        d = np.hypot(pts[None, :, 0] - xy[:, None, 0], pts[None, :, 1] - xy[:, None, 1])
        return float(d.min())

    def _need(self) -> float:
        # Already closer than the body allows (noise or a missed obstacle): accept only arcs that
        # do not get any closer, so the car can leave instead of deadlocking or pushing in.
        now = float(np.hypot(*self.pts.T).min()) if len(self.pts) else np.inf
        return min(self.BODY, now + .01)

    def allows(self, left: float, right: float) -> bool:
        """Whether a command's 1 s arc keeps the body 2 cm from remembered points (call after act)."""
        return self._clear(left, right) >= self._need()

    def act(self, depth, range_cm, gyro_rate, last_action, heading_error, how_far, speed_limit):
        self._remember(depth, range_cm, gyro_rate, last_action)
        lane = self.pts[(np.abs(self.pts[:, 1]) < self.BODY) & (self.pts[:, 0] > 0)]
        self.front = float(lane[:, 0].min()) if len(lane) else np.inf
        deg = degrees(heading_error)
        spin = 1 if deg > 0 else -1
        if abs(deg) > 50:
            diff = int(np.interp(abs(deg), [50, 120], [50, 72]))
            return -diff * spin, diff * spin
        want_base = max(45., self.CRUISE[how_far] * speed_limit * float(np.clip((self.front - .2) / .4, .5, 1)))
        want_turn = float(np.clip(1.15 * deg - .1 * degrees(gyro_rate), -48, 48))
        best, best_cost = None, np.inf
        need = self._need()
        for base in (want_base, .75 * want_base, 45.):
            for dturn in (0, 12, -12, 24, -24, 36, -36, 48, -48, 64, -64):
                turn = float(np.clip(want_turn + dturn, -64, 64))
                gap = self._clear(base - turn, base + turn)
                if gap < need:
                    continue
                cost = abs(turn - want_turn) / 48 + (want_base - base) / want_base + max(0., .3 - gap) / .15
                if cost < best_cost:
                    best, best_cost = (base, turn), cost
            if best is not None and best[0] == base:
                break
        if best is None:
            # No clear arc: spin in place (safe for a round body) toward the waypoint or the freer side.
            if abs(deg) < 20 and len(self.pts):
                near = self.pts[np.hypot(*self.pts.T) < .6]
                spin = 1 if np.sum(near[:, 1] > 0) < np.sum(near[:, 1] < 0) else -1
            return -62 * spin, 62 * spin
        base, turn = best
        return int(round(base - turn)), int(round(base + turn))
