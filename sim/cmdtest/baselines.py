"""Today's drivers behind the DRIVER_API contract.

go_to runs the navigator's driver (sim/navigate: HandDriver or VetoedPolicyDriver(V2)) exactly as the navigator
calls it, with the yaw rate taken from gyro_heading differences as navigator.step does. drive_forward / drive_back /
explore / patrol do not use those drivers on today's car: they run drive.py's straight controller (heading hold at
STEER PWM/deg, ramp RAMP PWM/s, CREEP..CRUISE, easing off over SLOW_CM; back at BACK PWM). That controller is
reproduced here from CarObs + DriveCommand only: heading to hold = the goal's bearing, room = distance to the goal
and the last range reading.
"""

from math import atan2, degrees, hypot, pi

from sim.navigate.driver import HandDriver
from sim.navigate.navigator import DT, wrap
from sim.navigate.policy_driver import V2, VetoedPolicyDriver

from .api import CarObs, DriveCommand, clip_pwm

CRUISE, CREEP, BACK, RAMP, STEER, STEER_MAX, SLOW_CM = 150, 80, 110, 300, 4.0, 60, 45
STOP_CM = {"forward": 25, "explore": 25, "patrol": 35}


class Baseline:
    def __init__(self, name: str, nav_driver):
        self.name = name
        self.nav = nav_driver
        self.command_style = "waypoint" if name == "hand" else "line_of_sight"
        self.reset()

    @property
    def front(self) -> float:
        return float(getattr(self.nav, "front", float("inf")))

    def reset(self, seed: int | None = None) -> None:
        self.nav.reset()
        self.prev_heading = None
        self.pwm = 0.0
        self.ahead_cm = 150.0
        self.mode = None
        self.last_goal_m = None

    def act(self, obs: CarObs, cmd: DriveCommand) -> tuple[int, int]:
        rate = 0.0 if self.prev_heading is None else wrap(obs.gyro_heading - self.prev_heading) / DT
        self.prev_heading = obs.gyro_heading
        if obs.range_cm is not None:
            self.ahead_cm = float(obs.range_cm)
        goal_m = hypot(*cmd.goal_xy)
        if cmd.mode != self.mode or (self.last_goal_m is not None and goal_m > self.last_goal_m + .3):
            self.pwm, self.mode = 0.0, cmd.mode
            if cmd.mode == "go_to":
                self.nav.reset()
        self.last_goal_m = goal_m
        if cmd.mode == "go_to":
            return clip_pwm(self.nav.act(obs.depth, obs.range_cm, rate, obs.last_action,
                                         cmd.heading_error, cmd.how_far, cmd.speed_limit))
        if cmd.mode == "back":
            behind = wrap(atan2(cmd.goal_xy[1], cmd.goal_xy[0]) - pi)
            steer = max(-STEER_MAX, min(STEER_MAX, STEER * degrees(behind)))
            return clip_pwm((-BACK - steer, -BACK + steer))
        room = min(goal_m * 100, self.ahead_cm - STOP_CM[cmd.mode])
        limit = min(CRUISE, CREEP + (CRUISE - CREEP) * room / SLOW_CM) * cmd.speed_limit
        self.pwm = max(CREEP, min(limit, self.pwm + RAMP * DT))
        bearing = atan2(cmd.goal_xy[1], cmd.goal_xy[0])
        steer = max(-STEER_MAX, min(STEER_MAX, STEER * degrees(bearing)))
        return clip_pwm((self.pwm - steer, self.pwm + steer))


def make_baseline(name: str) -> Baseline:
    match name:
        case "hand":
            return Baseline(name, HandDriver())
        case "policy_v2_veto":
            return Baseline(name, VetoedPolicyDriver(V2))
    raise ValueError(name)
