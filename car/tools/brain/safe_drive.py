"""Guard car commands with cancellation, floor checks, and stall tracking."""

import math
import time
from collections import deque
from collections.abc import Callable
from statistics import median

import navigation
from drive import NO_FLOOR, speed
from stall import StallWatch, distance_at
from telemetry import Recorder
from vision import Vision


class EndTask(Exception):
    """The job cannot continue safely."""


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


class SafeDrive:
    """Wrap the existing recorder without changing Driver or the wire protocol."""

    def __init__(self, robot: Recorder, reason: Callable[[], str | None], vision: Vision | None = None,
                 where: navigation.Whereabouts | None = None):
        self.robot, self.reason = robot, reason
        self.where = where
        self.navigating = False  # go_to tracks its own position; any other wheel command makes it stale
        self.head_angle = 90
        self.last_floor = 0.0
        self._pwm = (0.0, 0.0)
        self._wheel_at = time.monotonic()
        self.distance_driven = 0.0
        self._odometer: deque[tuple[float, float]] = deque(maxlen=200)
        self.vote_while_driving = False
        self.stall = StallWatch(vision)  # every wheel command goes through here, so this feeds it

    def check(self) -> None:
        if reason := self.reason():
            raise EndTask(reason)
        if time.monotonic() - self.last_floor >= 0.24:
            self.heading()

    def wait(self, seconds: float) -> None:
        until = time.monotonic() + seconds
        while (remaining := until - time.monotonic()) > 0:
            self.check()
            time.sleep(min(remaining, 0.04))
        self.check()

    def _check_floor(self, grounded: bool) -> None:
        self.last_floor = time.monotonic()
        if not grounded:
            raise EndTask(NO_FLOOR)

    def heading(self) -> tuple[float, bool]:
        if reason := self.reason():
            raise EndTask(reason)
        status = self.robot.heading()
        self._check_floor(status[1])
        return status

    def wheels(self, left: float, right: float) -> tuple[float, bool]:
        if left or right:
            self.check()
        now = time.monotonic()
        self.distance_driven += abs(sum(math.copysign(speed(v), v) for v in self._pwm) / 2) * min(
            0.4, now - self._wheel_at
        )
        self._pwm, self._wheel_at = (left, right), now
        self._odometer.append((now, self.distance_driven))
        status = self.robot.wheels(left, right)
        if left or right:
            self._check_floor(status[1])
            if self.where and not self.navigating:
                self.where.moved()
        if self.stall.wheels(left, right, status[0], self.head_angle):
            self.distance_driven = distance_at(self._odometer, self.stall.still_since)
        return status

    def head(self, angle: int) -> None:
        self.check()
        self.robot.head(angle)
        self.head_angle = angle

    def distance(self) -> int:
        if self.vote_while_driving:
            return int(self.range_vote())
        return self._range_sample()

    def _range_sample(self) -> int:
        self.check()
        cm = self.robot.distance()
        self.stall.ranged(cm, self.head_angle)
        return cm

    def range_vote(self) -> float:
        return median(self._range_sample() for _ in range(3))
