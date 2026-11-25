"""Go to a named place: the navigator of the project's sim/ folder, run live on the car's sensors.

This module and places.py own the brain's simulator imports, apart from depth_feed's lazy sim/realdepth
import. places.py appends the project root (four folders up, the folder holding sim/) to sys.path, so it
is imported first here: sim/navigate, sim/localize and sim/realdepth import each other as `sim.<package>`.
Their dependencies are in requirements.txt: scipy, pyyaml, opencv and embreex everywhere, torch and
depth_pro_2 (Apple Depth Pro 2-S) on the Mac only.

Observation, built each 100 ms control step (Sensors): depth from the newest Vision frame through
sim/realdepth/realdepth.py's observe(), on its own thread (DepthFeed), with that frame and its age for camera
drivers (sim/navigate/vision_driver.py); range from the head's sensor;
gyro = the heading in the wheel replies, unwrapped; head angle as commanded; wheel commands as sent.
Depth Pro 2 is under Apple's research-only licence; observation_spec.md names Depth Anything V2
(Apache-2.0, Core ML) as the fallback if that is unacceptable. Not wired in: realdepth.py has no DA2 path yet.

CAR_NAV_DRIVER: navdp (default) | hand | policy_v2_veto | policy_v2 | policy | policy_veto. navdp is the camera
driver of sim/models/navdp (torch, MPS; loads at the start of the first trip). The owner made it the default on
2026-11-25. In Isaac (.omo/waves/p4/report.md, real-car inputs) it led held-out route driving (93.3 %, 0.0028
contacts/m) and go_to on Gemini's commands 9/65 vs hand 6/65 (localiser-limited, thin), with more contacts
(0.82 vs 0.56 per m on go_to). It only drives go_to; drive_forward and the other moves stay on drive.py.
Through this go_to path against the building stand-in (43 trips from home, same seeds, 2026-11-24;
.omo/waves/p3/goto/trips-{hand2,v2veto}.jsonl) hand arrived 35 with 0.42 contacts per trip in a median 66 s,
policy_v2_veto 34 with 0.98 in 85 s, no false arrivals either way.
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import degrees, radians

import numpy as np

import depth_feed
import places as named_places  # puts the folder holding sim/ on sys.path for the imports below

# isort: split
from sim.localize.localise import Localizer
from sim.navigate.navigator import DRIVERS, DT, Navigator
from sim.navigate.vision_driver import CameraObservation

DRIVER = os.getenv("CAR_NAV_DRIVER", "navdp")
if DRIVER not in DRIVERS:
    raise SystemExit(f"CAR_NAV_DRIVER must be one of {', '.join(DRIVERS)}, not {DRIVER!r}")

TRUST = 0.5  # the localiser's confidence from which its pose is trusted (localize.md)
# Saving home needs more: a wrong home spoils every later trip. localize.md: confidence > 0.7 while > 1 m off
# in at most 0.2 % of steps. A head pan alone locked the lounge 1.5 m off at 0.93, so locate also turns a circle.
HOME_TRUST = 0.8
PULSE_DEG, PULSE_STEPS, REST_STEPS, PAN_STEPS = 40, 8, 8, 25  # the navigator's relocalisation turn


@dataclass(frozen=True, slots=True)
class Fix:
    pose: tuple[float, float, float]
    confidence: float
    at: float
    doing: str
    live: bool


class Whereabouts:
    """The localiser's latest guess, live while a go_to job runs, else the last one.

    It is trusted as the next trip's start only after a confident arrival (or a set home), with no wheel
    command from another job and no hand driving since.
    """

    def __init__(self, human_epoch: Callable[[], int]):
        self.human_epoch = human_epoch
        self.fix: Fix | None = None
        self._trusted_epoch: int | None = None
        self.last_trip: dict = {}
        self.on_fix: Callable[[tuple[float, float, float]], None] | None = None  # confident poses go to the map pane

    def update(self, pose, confidence: float, doing: str, live: bool = True) -> None:
        self.fix = Fix(tuple(map(float, pose)), float(confidence), time.monotonic(), doing, live)
        if self.on_fix and confidence >= TRUST and np.all(np.isfinite(self.fix.pose)):
            self.on_fix(self.fix.pose)

    def settle(self, trusted: bool) -> None:
        if self.fix:
            self.fix = Fix(self.fix.pose, self.fix.confidence, self.fix.at, self.fix.doing, False)
        self._trusted_epoch = self.human_epoch() if trusted and self.fix else None

    def moved(self) -> None:
        self._trusted_epoch = None

    def start_pose(self) -> tuple[float, float, float] | None:
        if self.fix and self._trusted_epoch is not None and self._trusted_epoch == self.human_epoch():
            return self.fix.pose
        return None

    def describe(self) -> str:
        fix = self.fix
        if fix is None or not np.all(np.isfinite(fix.pose)):
            return ("I don't know where I am yet. I work it out while going to a place, starting from home "
                    "or from a room you tell me.")
        x, y, _ = fix.pose
        guess = (f"I think I'm in {named_places.room_name(fix.pose)}" if fix.confidence >= TRUST
                 else f"My best guess is {named_places.room_name(fix.pose)}, but I'm not sure")
        detail = f"(about {x:.1f}, {y:.1f} on my map, confidence {fix.confidence:.0%})"
        if fix.live:
            return f"While {fix.doing}: {guess} {detail}."
        stale = "" if self.start_pose() else " I have moved or been moved since, so it may be out of date."
        return f"{guess} {detail}, as of {time.monotonic() - fix.at:.0f} seconds ago, when I was {fix.doing}.{stale}"


@dataclass(slots=True)
class Trip:
    label: str
    goal: tuple[float, float]
    navigator: Navigator
    start: str

    @property
    def there(self) -> str:
        return "home" if self.label == "home" else f"the {self.label}"

    @property
    def doing(self) -> str:
        return "going home" if self.label == "home" else f"going to the {self.label}"


def plan_trip(place: str, start_room: str | None, where: Whereabouts,
              dead_reckoned: tuple[float, float, float] | None = None) -> Trip:
    """Pick the goal and the start prior: a named room, else the dead-reckoned map pose (the owner clicked the
    start on the pane's map), else where the last trip arrived, else home once the owner has set it, else none
    (a cold start: the car finds itself first). Assuming the placeholder home made
    the real car "arrive" home without moving and drive from a wrong, certain pose (2026-11-24).

    Raises UnknownPlace for names not in places.yaml, ValueError for goals the map cannot reach.
    """
    label, goal, _ = named_places.find_place(place)
    if label == "home" and named_places.home().placeholder:
        raise named_places.UnknownPlace("Home isn't set yet. Put me on my home spot and say 'this is home', then I can go back there.")
    if start_room:
        room, _, area = named_places.find_place(start_room)
        if room != "home":
            return Trip(label, goal, Navigator(goal, area, driver=DRIVER), f"somewhere in the {room}")
    if dead_reckoned:
        start, how = dead_reckoned, "where my map tracking puts me"
    elif pose := where.start_pose():
        start, how = pose, "where my last trip ended"
    elif (h := named_places.home()).placeholder:
        return Trip(label, goal, Navigator(goal, driver=DRIVER), "an unknown spot, so I find myself first (home is not set)")
    else:
        start, how = (h.x, h.y, h.yaw), "home"
    return Trip(label, goal, Navigator(goal, start_pose=start, driver=DRIVER), how)


class Sensors:
    """Sends each step's command and reads the sensors for the next step's observation.

    The link answers one request at a time, about 57 ms each on the stand-in (45 ms reply, robot.py's 20 ms
    poll), and the navigator computes in about 7 ms. Two requests a step do not fit in 100 ms, so every step
    sends the wheel command (its reply carries the heading) and every second step also reads the range or
    moves the head; the other steps pass range None rather than an old reading.
    """

    def __init__(self, r, feed: depth_feed.DepthFeed):
        self.r, self.feed = r, feed
        r.head(90)
        self.prev_yaw, _ = r.heading()
        self.gyro = 0.0
        self.head = self.want_head = 0.0
        self.range: int | None = r.distance()
        self.sent = (0, 0)
        self.steps = 0

    def observe(self) -> CameraObservation:
        taken = self.feed.take()
        if taken is None:
            return CameraObservation(self.gyro, self.range, None, self.head, True)
        captured, depth, rgb = taken
        return CameraObservation(self.gyro, self.range, depth, self.head, True, rgb, time.monotonic() - captured)

    def send(self, left: int, right: int, head: float) -> None:
        self.steps += 1
        self.range = None
        if self.steps % 2 == 0:
            servo = max(0, min(180, round(90 + degrees(self.want_head))))
            if abs(servo - self.r.head_angle) >= 2:
                self.r.head(servo)
                self.head = radians(servo - 90)
            else:
                self.range = self.r.distance()
        command = (int(left), int(right))
        if command == (0, 0) and self.sent == (0, 0):
            yaw, _ = self.r.heading()  # the wheels already stand: only the heading is needed
        else:
            yaw, _ = self.r.wheels(*command)
        self.sent = command
        self.want_head = head
        self.gyro += radians((yaw - self.prev_yaw + 180) % 360 - 180)
        self.prev_yaw = yaw


def locate(r, feed: depth_feed.DepthFeed, area: str, where: Whereabouts, seconds: float = 50.0):
    """Find the car's pose in `area` on the spot: pan the head, turn one full circle in pulses, then hold still
    until the localiser has HOME_TRUST for 2 s. None if it never does within `seconds`."""
    loc = Localizer()
    loc.reset(room=area)
    sensors = Sensors(r, feed)
    sure, turned, step, pulse = 0, 0.0, 0, None  # pulse = (gyro at its start, steps so far) while turning
    due = time.monotonic()
    for _ in range(int(seconds / DT)):
        obs = sensors.observe()
        est = loc.update(sensors.sent, obs.gyro, obs.range_cm, obs.depth, obs.head_angle)
        where.update(est.pose, est.confidence, "finding my spot")
        step += 1
        command = (0, 0)
        if step > PAN_STEPS and turned < 2 * np.pi:
            if pulse is None:
                if step % REST_STEPS == 0:
                    pulse = (obs.gyro, 0)
            else:
                swept = abs(obs.gyro - pulse[0])
                if swept >= radians(PULSE_DEG) or pulse[1] >= PULSE_STEPS:
                    turned += swept
                    pulse = None
                else:
                    command, pulse = (-70, 70), (pulse[0], pulse[1] + 1)
        elif turned >= 2 * np.pi:
            sure = sure + 1 if est.confidence >= HOME_TRUST else 0
            if sure >= 20:
                return est
        sensors.send(*command, est.look_around or 0.0)
        due += DT
        r.wait(max(0.0, due - time.monotonic()))
    return None

