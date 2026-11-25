"""Closed-loop driving, run on the Mac at the pace of the car's link (about 10 to 15 commands a second).

Every tick sends signed wheel speeds and gets the gyro heading and floor state back, so the car holds
its heading on straights, turns to a measured angle, and ramps its speed instead of stop-starting.
Distance is dead reckoning from a calibrated speed: the car has no wheel encoders, so a StallWatch
checks that the camera picture really changes while the wheels drive straight.
"""

import json
import statistics
import time
from collections import deque
from pathlib import Path

import behaviors
import stall

__all__ = [
    "BACK", "CALIBRATION", "CM_PER_S", "CREEP", "CRUISE", "DEAD", "MOVE_LIMIT", "NO_FLOOR",
    "RAMP", "SLOW_CM", "SONAR_EVERY", "STALLED", "STEER", "STEER_MAX", "TURN_DONE", "TURN_GAIN",
    "TURN_LEAD_S", "TURN_MAX", "TURN_MIN", "TURN_RETRY", "Driver", "calibrate", "speed",
]


def speed(pwm: float) -> float:
    return CM_PER_S * max(0.0, abs(pwm) - DEAD) / (CRUISE - DEAD)


CALIBRATION = Path(__file__).with_name("calibration.json")
CRUISE = 150  # wheel PWM on a clear straight
CREEP = 80  # slowest PWM that still moves the car
DEAD = 12  # PWM where the measured speed line crosses zero
BACK = 110
RAMP = 300  # PWM per second
STEER = 4.0  # PWM per degree of heading error
STEER_MAX = 60
SLOW_CM = 45  # the car eases off over this much remaining room
SONAR_EVERY = 3  # ticks between range readings while driving forward
TURN_MIN, TURN_MAX = 65, 110  # measured: 80 spins about 130 deg/s, and the gyro saturates at 250 deg/s
TURN_GAIN = 0.5  # PWM per degree left to turn
TURN_DONE = 3.0
TURN_RETRY = 6.0  # a turn that settles further off than this gets one correcting pass
TURN_LEAD_S = 0.15  # the car keeps spinning about this long after the stop command
NO_FLOOR = "no floor under my front sensors (an edge, or I am lifted)"
MOVE_LIMIT = "that is as far as I go in one move"
STALLED = "I was pushing against something: my wheels turned but my camera view stood still"


def _calibrated_speed():
    try:
        return float(json.loads(CALIBRATION.read_text())["cm_per_s"])
    except FileNotFoundError:
        return 55.0  # from 27 cm/s at PWM 80 and 39 cm/s at 110, measured on carpet


CM_PER_S = _calibrated_speed()  # at CRUISE


class Driver:
    def __init__(self, robot, should_stop=lambda: None, stall=None):
        self.robot = robot
        self.should_stop = should_stop  # returns a reason to stop now, or None
        self.stall = stall  # the StallWatch the robot wrapper feeds, or None

    def _stop_reason(self):
        return self.should_stop() or (self.stall.stalled() if self.stall else None)

    def _roll(self, pwm, target):
        heading, _ = self._status
        steer = max(-STEER_MAX, min(STEER_MAX, STEER * (target - heading)))
        self._status = self.robot.wheels(pwm - steer, pwm + steer)

    def _halt(self):
        self._status = self.robot.wheels(0, 0)

    def forward(self, cm, stop_cm, max_s=7.0):
        """Drive straight ahead. Returns (cm driven, reason it stopped early or None)."""
        r = self.robot
        r.head(behaviors.CENTER)
        time.sleep(0.3)
        self._status = r.heading()
        target = self._status[0]
        ahead = r.distance()
        gone, pwm, tick, why = 0.0, 0.0, 0, None
        start = last = time.monotonic()
        marks = deque(maxlen=200)
        try:
            while gone < cm:
                why = self._stop_reason()
                if why == STALLED:
                    gone = stall.distance_at(marks, self.stall.still_since)
                if why:
                    break
                if not self._status[1]:
                    why = NO_FLOOR
                    break
                tick += 1
                if tick % SONAR_EVERY == 0:
                    ahead = r.distance()
                if ahead < stop_cm:
                    why = f"obstacle {ahead} cm ahead"
                    break
                now = time.monotonic()
                if now - start > max_s:
                    why = MOVE_LIMIT
                    break
                gone += speed(pwm) * (now - last)
                marks.append((now, gone))
                room = min(cm - gone, ahead - stop_cm)
                limit = min(CRUISE, CREEP + (CRUISE - CREEP) * room / SLOW_CM)
                pwm = max(CREEP, min(limit, pwm + RAMP * (now - last)))
                last = now
                self._roll(pwm, target)
        finally:
            self._halt()
        return gone, why

    def back(self, cm):
        """Back up straight; there is no sensor behind. Returns (cm driven, reason it stopped early or None)."""
        self._status = self.robot.heading()
        target = self._status[0]
        gone, why = 0.0, None
        last = time.monotonic()
        marks = deque(maxlen=200)
        try:
            while gone < cm:
                why = self._stop_reason()
                if why == STALLED:
                    gone = stall.distance_at(marks, self.stall.still_since)
                if why:
                    break
                self._roll(-BACK, target)
                now = time.monotonic()
                gone += speed(BACK) * (now - last)
                marks.append((now, gone))
                last = now
        finally:
            self._halt()
        return gone, why

    def turn(self, degrees, max_s=6.0):
        """Spin in place, positive to the left. Returns (degrees actually turned, reason it stopped early or None)."""
        r = self.robot
        start, _ = r.heading()
        heading, why = start, None
        began = time.monotonic()
        for _ in range(2):
            want = start + degrees - heading
            rate, last = 0.0, time.monotonic()
            try:
                while True:
                    left = start + degrees - heading - rate * TURN_LEAD_S
                    if abs(left) < TURN_DONE or left * want < 0:
                        break
                    why = self.should_stop()
                    if why:
                        break
                    if time.monotonic() - began > max_s:
                        why = "the turn took too long; my wheels may be stuck"
                        break
                    pwm = min(TURN_MAX, TURN_MIN + TURN_GAIN * abs(left))
                    side = 1 if left > 0 else -1
                    new, _ = r.wheels(-side * pwm, side * pwm)
                    now = time.monotonic()
                    rate = (new - heading) / (now - last)
                    heading, last = new, now
            finally:
                r.wheels(0, 0)
            time.sleep(0.2)
            heading, _ = r.heading()
            if why or abs(start + degrees - heading) < TURN_RETRY:
                break
        return heading - start, why

    def explore(self, seconds, stop_cm):
        """Wander: drive until something is in the way, then turn toward the most open direction."""
        deadline = time.monotonic() + seconds
        driven, turns, why = 0.0, 0, None
        while time.monotonic() < deadline:
            gone, why = self.forward(300, stop_cm, max_s=deadline - time.monotonic())
            driven += gone
            if why in (None, MOVE_LIMIT):
                why = None
                continue
            if why == NO_FLOOR:
                self.back(15)
                _, why = self.turn(120)
            elif why.startswith("obstacle"):
                angle, best = max(behaviors.scan(self.robot), key=lambda reading: reading[1])
                _, why = self.turn(180 if best <= stop_cm + 10 else angle - behaviors.CENTER)
            elif why == STALLED:
                # The range sensor missed whatever is ahead, so back off it and pick an open side.
                self.back(stall.STALL_BACK_CM)
                readings = [reading for reading in behaviors.scan(self.robot)
                            if abs(reading[0] - behaviors.CENTER) >= stall.STALL_AVOID_DEG]
                angle, best = max(readings, key=lambda reading: reading[1])
                _, why = self.turn(180 if best <= stop_cm + 10 else angle - behaviors.CENTER)
            else:
                break
            turns += 1
            if why:
                break
        return driven, turns, why


def calibrate(robot, log=print):
    """Check the turn direction against the gyro and measure cruise speed against a wall."""
    start, grounded = robot.heading()
    if not grounded:
        raise SystemExit("put the car on the floor first")
    until = time.monotonic() + 0.3
    while time.monotonic() < until:
        robot.wheels(-80, 80)
    robot.wheels(0, 0)
    time.sleep(0.5)
    turned = robot.heading()[0] - start
    log(f"short spin left: gyro says {turned:+.0f} degrees (the car should have spun counter-clockwise seen from above)")
    if turned < 10:
        raise SystemExit("the gyro did not see a left turn: wheel sides or gyro sign are swapped, fix before driving")
    Driver(robot).turn(-turned)

    robot.head(behaviors.CENTER)
    time.sleep(0.4)
    before = statistics.median(robot.distance() for _ in range(5))
    if not 100 <= before < 150:
        raise SystemExit(f"wall is {before} cm ahead; face a wall 110 to 140 cm away to measure speed")
    driver = Driver(robot)
    driver._status = robot.heading()
    target = driver._status[0]
    began = time.monotonic()
    while time.monotonic() - began < 1.2:
        driver._roll(CRUISE, target)
    took = time.monotonic() - began
    driver._halt()
    time.sleep(0.5)
    after = statistics.median(robot.distance() for _ in range(5))
    speed = (before - after) / took
    log(f"cruise: {before} cm -> {after} cm in {took:.2f} s = {speed:.0f} cm/s")
    if speed <= 5:
        raise SystemExit("the car barely moved; not saving")
    CALIBRATION.write_text(json.dumps({"cm_per_s": round(speed, 1)}) + "\n")
    log(f"saved {CALIBRATION.name}")
