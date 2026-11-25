"""Serialize the human and agent wheel commands and manage the car link."""

import threading
import time

from livekit.agents import ToolError

import behaviors
from drive import CM_PER_S, Driver
from robot import Robot
from telemetry import Recorder

STOP_CM = 15  # crash protocol: forward motion stops this close to a wall
MANUAL_SPEED = 150
HUMAN_HOLD_S = 1.5  # after the human touches the controls, the agent keeps its wheels still this long
RECONNECT_S = 3.0


class Car:
    def __init__(self, host, port, telemetry):
        self.host = host
        self.port = port
        self.telemetry = telemetry
        self._robot = None
        self._lock = threading.Lock()
        self._retry_at = 0.0
        self._human_at = 0.0
        self.human_epoch = 0

    def human_input(self):
        self._human_at = time.monotonic()
        self.human_epoch += 1

    def human_active(self):
        return time.monotonic() - self._human_at < HUMAN_HOLD_S

    def _call(self, fn, by):
        try:
            if self._robot is None:
                self._robot = Robot.wifi(self.host, self.port)
            return fn(Recorder(self._robot, self.telemetry, by))
        except OSError as e:
            self._drop()
            self._retry_at = time.monotonic() + RECONNECT_S
            self.telemetry.link.update(state="offline", error=str(e))
            self.telemetry.sensors["on_ground"] = None
            self.telemetry.moved(None, None)
            self.telemetry.touch()
            raise

    def run(self, fn, by="agent"):
        with self._lock:
            try:
                return self._call(fn, by)
            except OSError as e:
                raise ToolError(f"I can't reach my wheels right now ({e}).") from e

    def run_quiet(self, fn, by="human"):
        with self._lock:
            return self._quiet(fn, by)

    def try_run(self, fn):
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._quiet(fn, None)
        finally:
            self._lock.release()

    def _quiet(self, fn, by):
        if self._robot is None and time.monotonic() < self._retry_at:
            return None
        try:
            return self._call(fn, by)
        except OSError:
            return None

    def _drop(self):
        robot, self._robot = self._robot, None
        if robot is not None:
            try:
                robot.close()
            except OSError:
                pass

    def close(self):
        with self._lock:
            self._drop()


def _forward(robot, cm):
    gone, why = Driver(robot, stall=robot.stall).forward(cm, STOP_CM, max_s=3.0 + 1.5 * cm / CM_PER_S)
    if why:
        return f"Stopped after about {gone:.0f} cm: {why}."
    return f"Drove forward about {gone:.0f} cm. Clear distance ahead is {robot.distance()} cm."


def _back(robot, cm):
    gone, why = Driver(robot, stall=robot.stall).back(cm)
    return f"Backed up about {gone:.0f} cm" + (f", then stopped: {why}." if why else ".")


def _turn(robot, degrees):
    turned, why = Driver(robot).turn(degrees)
    side = "left" if turned > 0 else "right"
    return f"Turned {abs(turned):.0f} degrees {side}" + (f", then stopped: {why}." if why else ".")


def _explore(robot, seconds):
    driven, turns, why = Driver(robot, stall=robot.stall).explore(seconds, STOP_CM)
    return f"Explored: drove about {driven:.0f} cm and changed direction {turns} times. " + (f"Ended early: {why}." if why else "Time is up.")


def _look(robot, degrees):
    robot.head(behaviors.CENTER + degrees)
    if not degrees:
        return "Head facing forward."
    side = "left" if degrees > 0 else "right"
    return f"Head turned {abs(degrees)} degrees {side}."


def _scan(robot):
    readings = []
    for angle in range(10, 171, 20):
        robot.head(angle)
        robot.wait(0.3)
        readings.append((angle, robot.distance()))
    robot.head(behaviors.CENTER)
    parts = [f"{angle - behaviors.CENTER:+d} deg: {cm} cm" for angle, cm in readings]
    return "Free distance by head angle (+ is left): " + ", ".join(parts)
