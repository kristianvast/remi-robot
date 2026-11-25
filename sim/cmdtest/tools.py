"""The car agent's tool layer, replayed in simulation for one car (runs in its own thread, lockstep with the env).

Mirrors car/tools/brain:
  go_to          navigation.plan_trip (start = last confident arrival, else home; start_room -> room prior) and
                 trips.go_to's loop on sim/navigate's Navigator; its driver slot is replaced by DriverShim, which
                 hands the driver CarObs + DriveCommand built from the localiser's estimate. StallWatch -> blocked_ahead.
  drive_forward  dead reckoning (gyro heading + navigate.driver.motion of the PWM sent) from the call's start pose,
  drive_back     goal N cm ahead / behind; stops like drive.py: obstacle < STOP_CM (forward), stall, move time limit.
  turn           drive.py Driver.turn itself, on a FakeRobot.
  look_direction robot.head(CENTER + degrees), then SETTLE_S, as car_agent.py.
  explore/patrol goal 1.5 m (patrol 1.4 m) ahead along a random free bearing chosen from depth; blocked -> drive.py's
                 explore recovery (behaviors.scan + Driver.turn; stall: back 15 cm first).
  stop           cancels the running job; its finally sends (0, 0).
Only CarObs, the commands it sends and the camera JPEG (StallWatch, as on the real car) reach this layer;
simulator truth stays in the harness.
"""

import sys
import threading
from dataclasses import replace
from math import atan2, cos, degrees, hypot, inf, isnan, radians, sin

import numpy as np

from sim.navigate.driver import FAR, MEDIUM, NEAR, depth_points, motion
from sim.navigate.navigator import DT
from sim.navigate.vision_driver import CameraObservation

from .api import ROOT, CarObs, DriveCommand, clip_pwm

BRAIN = ROOT / "car" / "tools" / "brain"
if str(BRAIN) not in sys.path:
    sys.path.append(str(BRAIN))

import behaviors  # noqa: E402
import drive as brain_drive  # noqa: E402
import depth_feed  # noqa: E402
import navigation  # noqa: E402
import places  # noqa: E402
import robot  # noqa: E402
import stall  # noqa: E402

STOP_CM = 25
SETTLE_S = 0.4
REPEAT_S = 3.0
WATCHDOG_S = 0.4
MOVE_MIN_SPEED = 0.15          # forward/back time limit = max(drive.py's 7 s, distance / this)
LEG = {"explore": (1.5, 25), "patrol": (1.4, 35)}
FREE_M = 1.0


class Preempted(Exception):
    """The replay's next event (a later call, or a spoken stop) is due."""


class Aborted(Exception):
    """The harness ended the trial (time cap)."""


class SimClock:
    """Stands in for the time module of drive.py, behaviors.py and navigation.py: sim time, sleeps step the car."""

    def __init__(self):
        self.now = 0.0
        self.local = threading.local()

    def monotonic(self):
        return self.now

    time = perf_counter = monotonic

    def sleep(self, seconds):
        self.local.link.wait(seconds)


CLOCK = SimClock()
brain_drive.time = behaviors.time = navigation.time = CLOCK

_places_home = places.home


OFFICE_DESK_HOME = dict(x=6.23, y=-1.23, yaw=0.3089, area="area_05")  # = nav/env.py OFFICE_DESK_XY, its spawn heading


def owner_set_home():
    """Home as if the owner had run set_home beside the office middle desk, where the real car usually starts (owner
    confirmed 2026-11-25): navigation.plan_trip cold-starts and refuses "home" while home is a placeholder; every trial
    spawns here or in this room."""
    return replace(_places_home(), placeholder=False, **OFFICE_DESK_HOME)


places.home = owner_set_home


class Link:
    """One car's command channel to the lockstep stepper: step() sends a command and returns the next CarObs."""

    def __init__(self):
        self.cmd = (0, 0)
        self.head_servo = 90
        self.obs: CarObs | None = None
        self.frame = None           # (capture time, jpeg) of the newest delivered frame
        self.ready = threading.Event()
        self.go = threading.Event()
        self.preempt_at = inf
        self.abort = False
        self.finished = False
        self.sent_at = 0.0
        self.head_moved = False

    def _tick(self, check: bool) -> CarObs:
        self.ready.set()
        self.go.wait()
        self.go.clear()
        if self.abort:
            raise Aborted
        if check and CLOCK.now >= self.preempt_at - 1e-9:
            raise Preempted
        return self.obs

    def step(self, left, right) -> CarObs:
        self.cmd = clip_pwm((left, right))
        self.sent_at = CLOCK.now
        return self._tick(check=self.cmd != (0, 0))

    def head(self, servo: int) -> None:
        servo = int(max(0, min(180, servo)))
        self.head_moved = abs(servo - self.head_servo) >= 2
        self.head_servo = servo

    def wait(self, seconds: float) -> None:
        end = CLOCK.now + seconds
        while CLOCK.now < end - 1e-9:
            if CLOCK.now - self.sent_at >= WATCHDOG_S:
                self.cmd = (0, 0)
            self._tick(check=True)


class CameraFeed:
    """vision.Vision's newest(max_age) for StallWatch, from the frames the car really delivered."""

    def __init__(self, link):
        self.link = link

    def newest(self, max_age):
        frame = self.link.frame
        if frame is None or CLOCK.now - frame[0] > max_age:
            return None
        return type("Frame", (), {"timestamp": frame[0], "jpeg": frame[1], "people": ()})()


class FakeRobot:
    """The Robot / SafeDrive surface drive.py and behaviors.py use, on the Link."""

    def __init__(self, tools):
        self.tools = tools
        self.link = tools.link
        self.stall = stall.StallWatch(CameraFeed(self.link))
        self.head_angle = 90

    def heading(self):
        return degrees(self.link.obs.gyro_heading), True

    def wheels(self, left, right):
        obs = self.link.step(left, right)
        if (left or right) and not self.tools.navigating:
            self.tools.where.moved()
        self.stall.wheels(left, right, degrees(obs.gyro_heading), self.head_angle)
        return degrees(obs.gyro_heading), True

    def head(self, angle):
        self.link.head(angle)
        self.head_angle = int(angle)

    def distance(self):
        for _ in range(3):
            cm = self.link.obs.range_cm
            if cm is not None and not self.link.head_moved:
                self.stall.ranged(cm, self.head_angle)
                return int(cm)
            self.link.wait(DT)
        return robot.MAX_RANGE_CM

    def wait(self, seconds):
        self.link.wait(seconds)


def car_frame(x, y, h, gx, gy):
    dx, dy = gx - x, gy - y
    return (cos(h) * dx + sin(h) * dy, -sin(h) * dx + cos(h) * dy)


def level(m):
    return NEAR if m < .5 else FAR if m > 2 else MEDIUM


def command(goal_xy, target_xy, mode, speed_limit=1.0):
    return DriveCommand("goal", tuple(map(float, goal_xy)), tuple(map(float, target_xy)),
                        float(atan2(target_xy[1], target_xy[0])), level(hypot(*target_xy)), float(speed_limit), mode)


class DriverShim:
    """The navigator's Driver slot: its act(depth, range, rate, last, err, how_far, limit) becomes driver.act(CarObs,
    DriveCommand) with the goal and lookahead target in the car frame of the localiser's estimate."""

    def __init__(self, tools, nav):
        self.tools, self.nav = tools, nav
        self.target = None
        style = getattr(tools.driver, "command_style", "line_of_sight")
        base = nav.planner.waypoint if style == "waypoint" else nav.planner.line_of_sight

        def planned(pose):
            err, dist, path = base(pose)
            if not isnan(err):
                x, y, yaw = pose
                if style == "waypoint":
                    d = np.hypot(*(path - (x, y)).T)
                    ahead = np.flatnonzero(d >= .45)
                    self.target = tuple(path[ahead[0]] if len(ahead) else path[-1])
                else:
                    self.target = (x + dist * cos(yaw + err), y + dist * sin(yaw + err))
            return err, dist, path

        nav.command = planned

    @property
    def front(self):
        return float(getattr(self.tools.driver, "front", inf))

    def reset(self):
        self.tools.driver.reset()

    def act(self, depth, range_cm, rate, last_action, heading_error, how_far, speed_limit):
        x, y, yaw = self.nav.est.pose
        goal = car_frame(x, y, yaw, *self.nav.goal)
        target = car_frame(x, y, yaw, *self.target) if self.target is not None else goal
        obs = self.tools.link.obs
        obs = replace(obs, depth=depth, range_cm=range_cm, rgb=obs.rgb if depth is not None else None)
        cmd = DriveCommand("goal", goal, target, float(heading_error), int(how_far), float(speed_limit), "go_to")
        return self.tools.driver.act(obs, cmd)


class DeadReckoning:
    def __init__(self, heading):
        self.x = self.y = 0.0
        self.h = heading

    def update(self, obs):
        v, _ = motion(*obs.last_action)
        self.h = obs.gyro_heading
        self.x += v * DT * cos(self.h)
        self.y += v * DT * sin(self.h)


class CarTools:
    """One trial's tool layer. call(name, args) runs a tool like agent.py and returns (result text, info dict)."""

    def __init__(self, link, driver, seed):
        self.link = link
        self.driver = driver
        self.rng = np.random.default_rng(seed)
        self.where = navigation.Whereabouts(lambda: 0)
        self.navigating = False
        self.robot = FakeRobot(self)

    @property
    def obs(self) -> CarObs:
        return self.link.obs

    def halt(self):
        self.link.step(0, 0)

    def call(self, name, args):
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return "unsupported in simulation", {"status": "unsupported"}
        return fn(**args)

    def tool_drive_forward(self, centimeters):
        return self._straight(max(5, min(300, int(centimeters))), "forward")

    def tool_drive_back(self, centimeters):
        return self._straight(max(5, min(40, int(centimeters))), "back")

    def _straight(self, cm, mode):
        r = self.robot
        if mode == "forward":
            r.head(behaviors.CENTER)
            r.wait(0.3)
        dr = DeadReckoning(self.obs.gyro_heading)
        sign = 1 if mode == "forward" else -1
        goal = (sign * cm / 100 * cos(dr.h), sign * cm / 100 * sin(dr.h))
        heading0 = dr.h
        start, why, gone = CLOCK.now, None, 0.0
        limit_s = max(7.0, cm / 100 / MOVE_MIN_SPEED)
        marks = []
        try:
            while True:
                gone = sign * ((dr.x * cos(heading0)) + (dr.y * sin(heading0)))
                marks.append((CLOCK.now, gone * 100))
                if r.stall.stalled():
                    why = brain_drive.STALLED
                    gone = stall.distance_at(marks, r.stall.still_since) / 100
                    break
                if gone >= cm / 100:
                    break
                if mode == "forward" and self.obs.range_cm is not None and self.obs.range_cm < STOP_CM:
                    why = f"obstacle {self.obs.range_cm:.0f} cm ahead"
                    break
                if CLOCK.now - start > limit_s:
                    why = brain_drive.MOVE_LIMIT
                    break
                rel = car_frame(dr.x, dr.y, dr.h, *goal)
                left, right = self.driver.act(self.obs, command(rel, rel, mode))
                r.wheels(left, right)
                dr.update(self.obs)
        finally:
            self.halt()
        if mode == "forward":
            text = (f"Stopped after about {gone * 100:.0f} cm: {why}." if why
                    else f"Drove forward about {gone * 100:.0f} cm.")
        else:
            text = f"Backed up about {gone * 100:.0f} cm" + (f", then stopped: {why}." if why else ".")
        return text, {"status": "done", "why": why or "", "dr_cm": gone * 100}

    def tool_turn(self, direction, degrees):
        deg = max(5, min(180, abs(int(degrees)))) * (1 if direction == "left" else -1)
        turned, why = brain_drive.Driver(self.robot).turn(deg)
        side = "left" if turned > 0 else "right"
        return (f"Turned {abs(turned):.0f} degrees {side}" + (f", then stopped: {why}." if why else "."),
                {"status": "done", "why": why or "", "gyro_deg": turned, "asked_deg": deg})

    def tool_look_direction(self, direction, degrees=45):
        deg = 0 if direction == "forward" else max(10, min(80, abs(int(degrees)))) * (1 if direction == "left" else -1)
        self.robot.head(behaviors.CENTER + deg)
        self.link.wait(SETTLE_S)
        return f"Head turned {abs(deg)} degrees.", {"status": "done", "asked_deg": deg}

    def tool_stop(self):
        self.halt()
        return "Stopped: I was told to stop.", {"status": "done"}

    def tool_where_am_i(self):
        return self.where.describe(), {"status": "no_motion"}

    def tool_task_status(self):
        return "Nothing is running.", {"status": "no_motion"}

    def tool_scan_distances(self):
        readings = behaviors.scan(self.robot)
        return f"Scanned {len(readings)} directions.", {"status": "done"}

    def tool_explore(self, seconds):
        return self._roam(max(10, min(120, int(seconds))), "explore")

    def tool_patrol(self, seconds=30):
        return self._roam(max(10, min(120, int(seconds))), "patrol")

    def _free_bearing(self, depth):
        forward, left = depth_points(depth, reach=1.5)
        bearings = np.radians(np.linspace(-28, 28, 15))
        free = []
        for b in bearings:
            along = forward * cos(b) + left * sin(b)
            across = np.abs(-forward * sin(b) + left * cos(b))
            hit = along[(across < .12) & (along > 0)]
            free.append(min(1.5, float(hit.min())) if len(hit) else 1.5)
        free = np.array(free)
        ok = np.flatnonzero(free >= FREE_M)
        if not len(ok):
            return None
        return float(bearings[self.rng.choice(ok, p=free[ok] / free[ok].sum())])

    def _recover(self, why, stop_cm):
        driver = brain_drive.Driver(self.robot)
        if why == brain_drive.STALLED:
            self._straight(stall.STALL_BACK_CM, "back")
            readings = [x for x in behaviors.scan(self.robot)
                        if abs(x[0] - behaviors.CENTER) >= stall.STALL_AVOID_DEG]
        else:
            readings = behaviors.scan(self.robot)
        angle, best = max(readings, key=lambda reading: reading[1])
        driver.turn(180 if best <= stop_cm + 10 else angle - behaviors.CENTER)

    def _roam(self, seconds, mode):
        leg_m, stop_cm = LEG[mode]
        r = self.robot
        deadline = CLOCK.now + seconds
        legs = turns = 0
        dr = DeadReckoning(self.obs.gyro_heading)
        try:
            r.head(behaviors.CENTER)
            while CLOCK.now < deadline:
                if self.obs.depth is None:
                    r.wheels(0, 0)
                    dr.update(self.obs)
                    continue
                bearing = self._free_bearing(self.obs.depth)
                if bearing is None:
                    self._recover("obstacle", stop_cm)
                    turns += 1
                    dr = DeadReckoning(self.obs.gyro_heading)
                    continue
                goal = (dr.x + leg_m * cos(dr.h + bearing), dr.y + leg_m * sin(dr.h + bearing))
                legs += 1
                began, why = CLOCK.now, None
                while CLOCK.now < deadline:
                    rel = car_frame(dr.x, dr.y, dr.h, *goal)
                    if hypot(*rel) < .25:
                        break
                    if r.stall.stalled():
                        why = brain_drive.STALLED
                        break
                    if self.obs.range_cm is not None and self.obs.range_cm < stop_cm:
                        why = "obstacle"
                        break
                    if CLOCK.now - began > leg_m / MOVE_MIN_SPEED:
                        break
                    r.wheels(*self.driver.act(self.obs, command(rel, rel, mode)))
                    dr.update(self.obs)
                if why and CLOCK.now < deadline:
                    r.wheels(0, 0)
                    self._recover(why, stop_cm)
                    turns += 1
                    dr = DeadReckoning(self.obs.gyro_heading)
        finally:
            self.halt()
        return f"{mode}: {legs} legs, {turns} direction changes. Time is up.", {"status": "done", "legs": legs}

    def tool_go_to(self, place, start_room=""):
        try:
            trip = navigation.plan_trip(place, start_room or None, self.where)
        except ValueError as e:
            return (str(e) if isinstance(e, places.UnknownPlace) else f"I can't go there: {e}."), {
                "status": "refused"}
        nav = trip.navigator
        nav.driver = DriverShim(self, nav)
        info = {"status": "started", "goal": trip.goal, "start": trip.start, "label": trip.label}
        status, stalls, seen = None, 0, len(self.robot.stall.events)
        self.navigating = True
        head_servo = 90
        try:
            while True:
                obs = self.obs
                fresh = obs.depth if obs.rgb_age_s < depth_feed.DEPTH_MAX_AGE_S else None
                head = radians(head_servo - 90)
                ob = CameraObservation(obs.gyro_heading, obs.range_cm, fresh, head, True, obs.rgb, obs.rgb_age_s)
                left, right, head_rad, status = nav.step(ob)
                self.where.update(status.pose, status.confidence, trip.doing)
                info["est"] = status.pose
                if status.state in ("arrived", "failed"):
                    break
                head_servo = round(90 + degrees(head_rad))
                self.robot.head(head_servo)
                self.robot.wheels(left, right)
                if len(self.robot.stall.events) > seen:
                    seen = len(self.robot.stall.events)
                    stalls += 1
                    nav.blocked_ahead()
        finally:
            self.navigating = False
            self.where.settle(status is not None and status.state == "arrived")
            self.robot.head(90)
            self.halt()
        info.update(status="arrived" if status.state == "arrived" else "failed", reason=status.reason,
                    nav_s=status.t, confidence=status.confidence, stalls=stalls, recoveries=nav.recoveries)
        if status.state == "arrived":
            return f"I think I'm at {trip.there} now, after {status.t:.0f} seconds.", info
        return f"I stopped without reaching {trip.there}: {status.reason}.", info


def label(name, args):
    match name:
        case "drive_forward":
            return f"Drive forward {max(5, min(300, int(args.get('centimeters', 0))))} cm"
        case "drive_back":
            return f"Back up {max(5, min(40, int(args.get('centimeters', 0))))} cm"
        case "turn":
            return f"Turn {args.get('direction')} {max(5, min(180, abs(int(args.get('degrees', 0)))))}"
    return None


MOTION = {"drive_forward", "drive_back", "turn", "explore", "patrol", "go_to", "look_direction", "stop",
          "scan_distances", "find_object", "follow_person", "set_home"}
BACKGROUND = {"explore", "patrol", "go_to", "find_object", "follow_person"}
MODEL_GAP_S = 1.0
NEXT_TURN_S = 2.0


def replay(tools, request, events):
    """Run a recorded request's calls in order like the agent; append one dict per call to events."""
    turns = request["turns"]
    spoken_at = None
    for k, turn in enumerate(turns):
        last_move = None
        nxt = turns[k + 1] if k + 1 < len(turns) else None
        stop_at = None
        calls = turn["calls"]
        for j, call in enumerate(calls):
            name, args = call["name"], call["args"]
            ev = {"turn": k, "index": j, "name": name, "args": args,
                  "t0": spoken_at if j == 0 and spoken_at is not None else CLOCK.now}
            events.append(ev)
            lab = label(name, args)
            if lab and last_move and last_move[0] == lab and CLOCK.now - last_move[1] < REPEAT_S:
                ev.update(t1=CLOCK.now, status="skipped_repeat", result="Skipped: that exact move just finished.")
                continue
            later = next((c for c in calls[j + 1:] if c["name"] in MOTION), None)
            preempt = inf
            if name in BACKGROUND and later is not None:
                same_trip = name == "go_to" and later["name"] == "go_to" and later["args"] == args
                preempt = inf if same_trip else CLOCK.now + MODEL_GAP_S
            if nxt is not None and nxt["after_s"] is not None and name in MOTION:
                stop_at = CLOCK.now + nxt["after_s"] if stop_at is None else stop_at
                preempt = min(preempt, stop_at)
            tools.link.preempt_at = preempt
            try:
                result, info = tools.call(name, args)
                ev.update(info)
            except Preempted:
                result = "interrupted"
                ev.update(status="interrupted")
                tools.link.preempt_at = inf
                tools.navigating = False
                tools.halt()
            tools.link.preempt_at = inf
            ev.update(t1=CLOCK.now, result=result)
            if lab:
                last_move = (lab, CLOCK.now)
            if name not in BACKGROUND and j + 1 < len(calls):
                tools.link.wait(MODEL_GAP_S)
        spoken_at = None
        if nxt is not None:
            if nxt["after_s"] is not None and stop_at is not None:
                if CLOCK.now < stop_at:
                    tools.link.wait(stop_at - CLOCK.now)
                spoken_at = stop_at
            elif nxt["after_s"] is None:
                tools.link.wait(NEXT_TURN_S)
    tools.link.wait(1.0)
