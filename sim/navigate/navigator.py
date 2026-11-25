"""Go to a named place, closed loop on the localiser's estimate (never the true pose).

    nav = Navigator("kitchen", start_room="area_03")     # or (x, y); start_room=None = cold start
    nav = Navigator("kitchen", start_pose=(x, y, yaw))  # a known start (e.g. home)
    every 100 ms: left, right, head, status = nav.step(observation)
    nav.blocked_ahead()   # the car is pushing against something its sensors miss (a stall)

observation is what macsim's Simulator.observe() returns and the real car can supply:
gyro (integrated yaw, rad), range_cm, depth (encoded 120x160 or None), head_angle (rad).
Send (left, right) to the wheels and head to the pan servo until status.state is
"arrived" or "failed" (status.reason: lost, blocked or timeout).
"""

from collections import defaultdict
from dataclasses import dataclass
from math import cos, degrees, pi, radians, sin, tau

import numpy as np

from sim.localize.localise import Localizer

from .driver import FAR, MEDIUM, NEAR, Driver, HandDriver, depth_points, motion
from .places import resolve
from .planner import Planner
from .policy_driver import V2, PolicyDriver, VetoedPolicyDriver
from .vision_driver import VisionDriver

DT = .1
TRUST = .5
ARRIVE_M = .3
STUCK_S, PROGRESS_M = 8., .25
MAX_RECOVERIES = 8
POOR_FIT = .5          # map_fit under this: drive slower (the map differs from what the car sees)
MAP_WRONG = .25        # map_fit under this while certain: remember the spot as "the map is wrong here"


DRIVERS = ("hand", "policy", "policy_veto", "policy_v2", "policy_v2_veto", "navdp")


def wrap(a: float) -> float:
    return (a + pi) % tau - pi


def make_driver(name: str) -> Driver:
    match name:
        case "hand":
            return HandDriver()
        case "policy":
            return PolicyDriver()
        case "policy_veto":
            return VetoedPolicyDriver()
        case "policy_v2":
            return PolicyDriver(V2)
        case "policy_v2_veto":
            return VetoedPolicyDriver(V2)
        case "navdp":
            return VisionDriver("navdp")
    raise ValueError(f"Unknown driver {name!r}; expected one of {DRIVERS}")


@dataclass(frozen=True, slots=True)
class Status:
    state: str            # localising | driving | recovering | arrived | failed
    reason: str           # failure reason (lost | blocked | timeout) or ""
    t: float
    pose: tuple[float, float, float]
    confidence: float     # position certainty (Estimate.confidence)
    to_goal_m: float      # straight-line, from the estimate
    map_fit: float = 1.   # Estimate.map_fit

    def line(self) -> str:
        x, y, a = self.pose
        extra = f" ({self.reason})" if self.reason else ""
        return (f"t={self.t:6.1f}s {self.state:<11}{extra} est=({x:6.2f},{y:6.2f},{degrees(a):5.0f}deg) "
                f"conf={self.confidence:.2f} fit={self.map_fit:.2f} to_goal={self.to_goal_m:5.2f}m")


class Navigator:
    def __init__(self, goal: str | tuple[float, float], start_room: str | None = None, *,
                 start_pose: tuple[float, float, float] | None = None,
                 driver: str = "hand", car_turn: bool = True, drop_furniture: bool = False,
                 map_image=None, time_limit: float = 300., seed: int = 0):
        """Start prior: start_pose (known spot), else start_room, else none (cold start).
        map_image: the occupancy PNG the car believes (default the scan's floor map)."""
        x, y, _ = resolve(goal)
        self.goal = (x, y)
        self.loc = Localizer(seed=seed, drop_furniture=drop_furniture, map_image=map_image)
        self.loc.reset(pose=start_pose, room=None if start_pose is not None else start_room)
        self.planner = Planner(drop_furniture, map_image)
        if not self.planner.plan(self.goal):
            raise ValueError(f"Goal {goal!r} is not reachable on the planning map")
        self.driver = make_driver(driver)
        # The learned driver was trained on a line-of-sight target 0.44-2.4 m ahead with the distance
        # level measured to it; the hand driver aims 0.45 m ahead and is told the distance to the goal.
        self.command = self.planner.waypoint if driver == "hand" else self.planner.line_of_sight
        if isinstance(self.driver, VisionDriver):
            self.command = self.driver.bind(self, self.command)
        self.obs = None
        self.car_turn = car_turn
        self.time_limit = time_limit
        self.t = 0.
        self.cmd = (0, 0)
        self.head = 0.
        self.prev_gyro = self.prev_head = None
        self.est = None
        self.good = self.bad = 0
        self.state, self.reason = "localising", ""
        self._enter_localise()
        self.recoveries = 0
        self.hits: dict[int, int] = defaultdict(int)
        self.last_plan = -1.
        self.best_cost, self.best_t = np.inf, 0.
        self.lost_log: list[bool] = []
        self.first_confident = None
        self.localise_s = 0.
        self.replans = 0
        self.planned_path = None
        self.steady = 0
        self.odo = 0.
        self.turn_odo = None
        self.at_goal_s = 0.
        self.map_wrong: set[tuple[int, int]] = set()   # 0.5 m cells where a certain estimate fit badly

    def _enter_localise(self) -> None:
        self.state = "localising"
        self.phase, self.phase_t = "pan", self.t
        self.turned = 0.
        self.pulse_start = None

    def _after_pan(self) -> str:
        # A full circle shows the whole room; repeating it on the same spot adds nothing.
        moved = self.turn_odo is None or self.odo - self.turn_odo >= 1.
        return "turn" if self.car_turn and moved else "creep"

    def _status(self) -> Status:
        pose = self.est.pose if self.est else (np.nan, np.nan, np.nan)
        conf = self.est.confidence if self.est else 0.
        return Status(self.state, self.reason, round(self.t, 2), pose, conf,
                      float(np.hypot(pose[0]-self.goal[0], pose[1]-self.goal[1])),
                      self.est.map_fit if self.est else 1.)

    def _finish(self, state: str, reason: str = "") -> None:
        self.state, self.reason = state, reason
        self.cmd = (0, 0)

    def step(self, obs) -> tuple[int, int, float, Status]:
        self.t += DT
        self.obs = obs
        rate = 0. if self.prev_gyro is None else wrap(obs.gyro - self.prev_gyro) / DT
        self.prev_gyro = obs.gyro
        self.odo += abs(motion(*self.cmd)[0]) * DT
        previous = self.est
        self.est = est = self.loc.update(self.cmd, obs.gyro, obs.range_cm, obs.depth, obs.head_angle)
        jumped = previous is None or np.hypot(est.pose[0]-previous.pose[0], est.pose[1]-previous.pose[1]) > .5
        self.steady = 0 if jumped else self.steady + 1
        confident = est.confidence >= TRUST
        self.good = self.good + 1 if confident else 0
        self.bad = 0 if confident else self.bad + 1
        if confident and self.first_confident is None:
            self.first_confident = self.t
        self.lost_log.append(not confident)
        # The camera frame is 100 ms old: use it for driving only if the head was forward then too.
        forward = abs(obs.head_angle) < .1 and self.prev_head is not None and abs(self.prev_head) < .1
        self.prev_head = obs.head_angle
        depth = obs.depth if forward else None
        range_cm = obs.range_cm if abs(obs.head_angle) < .1 else None
        if self.state in ("arrived", "failed"):
            return 0, 0, self.head, self._status()
        if self.t > self.time_limit:
            recent = self.lost_log[-int(60/DT):]
            if self.first_confident is None or self.state == "localising" or np.mean(recent) > .5:
                self._finish("failed", "lost")
            elif self.recoveries >= 3:
                self._finish("failed", "blocked")
            else:
                self._finish("failed", "timeout")
            return 0, 0, self.head, self._status()
        match self.state:
            case "localising":
                self.localise_s += DT
                self.cmd = self._localise(est, depth, range_cm, rate, obs.gyro)
            case "driving":
                self.cmd = self._drive(est, depth, range_cm, rate, obs.head_angle)
            case "recovering":
                self.cmd = self._recover(obs.gyro, depth)
        return self.cmd[0], self.cmd[1], self.head, self._status()

    def _localise(self, est, depth, range_cm, rate, gyro) -> tuple[int, int]:
        settled = self.phase == "creep" and est.confidence >= .3 and self.steady >= 50
        if self.good >= 10 or settled:
            self.state = "driving"
            self.best_cost, self.best_t = self.planner.cost_to_go(*est.pose[:2]), self.t
            self.driver.reset()
            if self.planned_path is None:
                self.planned_path = self.planner.path(*est.pose[:2], max_m=80.)
            return 0, 0
        elapsed = self.t - self.phase_t
        if self.phase == "pan":
            self.head = est.look_around or 0.
            if elapsed >= 2.5:
                self.phase, self.phase_t = self._after_pan(), self.t
            return 0, 0
        if self.phase == "turn":
            self.head = est.look_around or 0.
            if self.pulse_start is None:
                if elapsed >= .8:                          # hold still between pulses
                    self.pulse_start, self.pulse_t = gyro, self.t
                return 0, 0
            swept = abs(wrap(gyro - self.pulse_start))
            if swept >= radians(40) or self.t - self.pulse_t >= .8:
                self.turned += swept
                self.pulse_start, self.phase_t = None, self.t
                if self.turned >= tau:
                    self.phase, self.turn_odo = "creep", self.odo
                return 0, 0
            return -70, 70
        # creep: move slowly for 8 s so the localiser sees new views (standing still near furniture
        # the map lacks keeps its fit score bad); follow the plan if the estimate is steady.
        self.head = 0.
        if elapsed >= 8.:
            self.phase, self.phase_t, self.turned = "pan", self.t, 0.
            return 0, 0
        if depth is None and range_cm is None:
            return 0, 0
        err, dist, _ = self.command(est.pose) if self.steady >= 20 else (np.nan, np.inf, None)
        if np.isnan(err):
            return self.driver.act(depth, range_cm, rate, self.cmd, 0., MEDIUM, .5)
        return self.driver.act(depth, range_cm, rate, self.cmd, err, MEDIUM if dist > .5 else NEAR, .5)

    def _drive(self, est, depth, range_cm, rate, head_angle) -> tuple[int, int]:
        self.head = 0.
        # Relocalise when position certainty is near zero or the estimate jumps for 1 s, or after 15 s
        # under TRUST: where the map lacks furniture a converged estimate may never earn certainty
        # (localise.py's EARN_FIT) while staying right, and can still drive on carefully.
        if self.bad >= 10 and (est.confidence < .25 or self.steady < 20) or self.bad >= 150:
            self._enter_localise()
            return 0, 0
        x, y, yaw = est.pose
        if est.confidence >= TRUST and est.map_fit < MAP_WRONG:
            self.map_wrong.add((int(np.floor(x/.5)), int(np.floor(y/.5))))
        if np.hypot(x-self.goal[0], y-self.goal[1]) < ARRIVE_M:
            if est.confidence >= TRUST:
                self._finish("arrived")
                return 0, 0
            # At the goal but unsure: stand still and look around rather than circle the point.
            self.head = est.look_around or 0.
            self.best_t = self.t
            self.at_goal_s += DT
            if self.at_goal_s > 6.:
                self.at_goal_s = 0.
                self._enter_localise()
            return 0, 0
        self.at_goal_s = 0.
        if depth is not None and est.confidence >= .6:
            self._mark(est.pose, depth)
        err, dist, path = self.command(est.pose)
        if np.isnan(err):
            self.planner.clear_marks()
            self._replan()
            err, dist, path = self.command(est.pose)
            if np.isnan(err):           # the estimate sits where no route exists: it is wrong
                self._enter_localise()
                return 0, 0
        cost = self.planner.cost_to_go(x, y)
        if cost < self.best_cost - PROGRESS_M:
            self.best_cost, self.best_t = cost, self.t
        elif self.t - self.best_t > STUCK_S:
            return self._start_recovery()
        if abs(head_angle) >= .1:
            return 0, 0                 # wait for the head to face forward again
        how_far = NEAR if dist < .5 else FAR if dist > 2 else MEDIUM
        limit = 1. if est.confidence >= .72 else .75 if est.confidence >= TRUST else .5
        if est.map_fit < POOR_FIT:
            limit = min(limit, .75)
        return self.driver.act(depth, range_cm, rate, self.cmd, err, how_far, limit)

    def _mark(self, pose, depth) -> None:
        """Remember obstacles the map lacks; cells seen in 3 frames close the plan around them."""
        forward, left = depth_points(depth, reach=.8)
        if not len(forward):
            return
        x, y, yaw = pose
        # Only obstacles the map cannot explain: depth's +-30 % scale error puts walls up to 30 %
        # nearer, and marking those would close doorways. Keep points well short of the map's wall.
        bearing = yaw + np.arctan2(left, forward)
        expected = self.loc.map.ranges(np.full(len(forward), x), np.full(len(forward), y), bearing[:, None])[:, 0]
        unexplained = np.hypot(forward, left) < .65 * expected - .1
        forward, left = forward[unexplained], left[unexplained]
        px = x + forward*cos(yaw) - left*sin(yaw)
        py = y + forward*sin(yaw) + left*cos(yaw)
        d, idx = self.planner.tree.query(np.column_stack((px, py)), distance_upper_bound=.05)
        new = []
        for i in np.unique(idx[np.isfinite(d)]):
            self.hits[i] += 1
            if self.hits[i] == 3:
                new.append(self.planner.xy[i])
        if new and self.t - self.last_plan >= 1.:
            self.planner.mark(np.array(new), self.t)
            self._replan()

    def _replan(self) -> None:
        self.last_plan = self.t
        self.replans += 1
        if not self.planner.plan(self.goal, self.t):
            self.planner.clear_marks()
            self.planner.plan(self.goal, self.t)
        self.best_cost = self.planner.cost_to_go(*self.est.pose[:2])

    def blocked_ahead(self) -> None:
        """Something the sensors miss blocks the car (e.g. a stall detector fired): back up, turn,
        block the spot ahead and replan now instead of after STUCK_S without progress. Only while driving."""
        if self.state == "driving":
            self.cmd = self._start_recovery()

    def _start_recovery(self) -> tuple[int, int]:
        self.recoveries += 1
        if self.recoveries > MAX_RECOVERIES:
            self._finish("failed", "blocked")
            return 0, 0
        x, y, yaw = self.est.pose
        if self.driver_front() < .5:
            self.planner.mark(np.array([[x + .3*cos(yaw), y + .3*sin(yaw)]]), self.t)
        self.state, self.phase, self.phase_t = "recovering", "back", self.t
        self.turn_sign = 1 if self.recoveries % 2 else -1
        return 0, 0

    def driver_front(self) -> float:
        return float(getattr(self.driver, "front", np.inf))

    def _recover(self, gyro, depth) -> tuple[int, int]:
        elapsed = self.t - self.phase_t
        if self.phase == "back":
            if elapsed < .8:
                return -70, -70
            self.phase, self.phase_t, self.turn_start = "turn", self.t, gyro
            return 0, 0
        if abs(wrap(gyro - self.turn_start)) < radians(70) and elapsed < 1.5:
            s = self.turn_sign
            return -70*s, 70*s
        self._replan()
        self.state = "driving"
        self.best_t = self.t
        self.driver.reset()
        return 0, 0
