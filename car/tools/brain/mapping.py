"""The building map and where the car is on it, by dead reckoning, for the control pane and the voice agent.

Map: the merged scan's occupancy image (scans/merged/floor_map.png + .yaml), the room outlines
(rooms_next.json) and the room names and goal points of sim/navigate/places.yaml. Map frame: metres,
+x east, +y north (up on the image), yaw in radians counter-clockwise from +x.

Pose: the owner clicks the start spot on the pane's map; from then on every wheel command moves the
estimate by the calibrated speed model (drive.speed) and every gyro reading turns it (the firmware's
heading grows to the left, the same sense as yaw). With no wheel encoders the estimate drifts, so
drift_m grows with distance and turning, and a confident go_to localiser fix snaps it back.
"""

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import yaml
from PIL import Image

import drive

ROOT = Path(__file__).resolve().parents[3]  # the folder holding scans/ and sim/
SCAN = ROOT / "scans/merged"
PLACES = ROOT / "sim/navigate/places.yaml"
MAP_IMAGE = SCAN / "floor_map.png"
CAR_MODEL = ROOT / "sim/preview/elegoo_v4.glb"

WATCHDOG_S = 0.4  # the firmware stops the wheels this long after the last N=4 command
TRAIL_STEP_M = 0.05
TRAIL_MAX = 400
DRIFT_START_M = 0.15  # a click on the map is good to about this
DRIFT_PER_M = 0.08  # dead-reckoned distance error per metre driven (speed model vs carpet)
DRIFT_PER_DEG = 0.002  # sideways error per degree turned (gyro scale, wheel slip)
FIX_DRIFT_M = 0.3


def _inside(x, y, ring):
    hit = False
    for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1], strict=True):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            hit = not hit
    return hit


class BuildingMap:
    def __init__(self):
        info = yaml.safe_load((SCAN / "floor_map.yaml").read_text())
        self.resolution = float(info["resolution"])
        self.origin = [float(info["origin"][0]), float(info["origin"][1])]
        with Image.open(MAP_IMAGE) as image:
            self.width_px, self.height_px = image.size
        areas = yaml.safe_load(PLACES.read_text())["areas"]
        self.rooms = []
        for room in json.loads((SCAN / "rooms_next.json").read_text()):
            spec = areas.get(room["id"], {})
            goal = spec.get("goals", {}).get(spec.get("default"))
            label = [goal["x"], goal["y"]] if goal else [room["center"]["x"], room["center"]["y"]]
            self.rooms.append({
                "id": room["id"],
                "name": spec.get("name", room["name"]),
                "polygon": room["polygon"],
                "holes": room.get("holes", []),
                "label": label,
            })
        self.places = [{"name": r["name"], "x": r["label"][0], "y": r["label"][1]} for r in self.rooms]

    def room_at(self, x, y):
        for room in self.rooms:
            if _inside(x, y, room["polygon"]) and not any(_inside(x, y, hole) for hole in room["holes"]):
                return room["name"]
        return None

    def place(self, name):
        key = " ".join(name.casefold().split()).removeprefix("the ")
        return next((p for p in self.places if p["name"] == key), None)

    def home(self):
        """The saved home spot, or None while places.yaml still holds the placeholder."""
        h = yaml.safe_load(PLACES.read_text())["home"]
        return None if h.get("placeholder") else {"x": float(h["x"]), "y": float(h["y"]), "yaw": float(h["yaw"])}

    def hello(self):
        return {
            "image": "/map.png",
            "resolution": self.resolution,
            "origin": self.origin,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "rooms": [{k: r[k] for k in ("id", "name", "polygon", "label")} for r in self.rooms],
            "places": self.places,
        }


def compass(yaw):
    return ("E", "NE", "N", "NW", "W", "SW", "S", "SE")[round(math.degrees(yaw) / 45) % 8]


class Odometry:
    """Dead-reckoned pose, fed by Recorder with every wheel command and gyro reading. Thread-safe."""

    def __init__(self, building: BuildingMap):
        self.map = building
        self._lock = threading.Lock()
        self.placed = False
        self.x = self.y = 0.0
        self.source = "dead-reckoning"
        self.drift = 0.0
        self.travelled = 0.0
        self.trail: deque[tuple[float, float]] = deque(maxlen=TRAIL_MAX)
        self._gyro = None  # last raw firmware heading, degrees
        self._turned = 0.0  # unwrapped heading change, radians
        self._offset = 0.0  # yaw = _turned + _offset
        self._v = 0.0  # m/s
        self._until = 0.0  # monotonic time the command holds until
        self._at = time.monotonic()
        self.home = building.home()

    @property
    def yaw(self):
        return math.atan2(math.sin(self._turned + self._offset), math.cos(self._turned + self._offset))

    def _advance(self, now):
        dt = max(0.0, min(now, self._until) - self._at)
        self._at = now
        if not dt or not self._v:
            return
        step = self._v * dt
        self.travelled += abs(step)
        if not self.placed:
            return
        yaw = self._turned + self._offset
        self.x += step * math.cos(yaw)
        self.y += step * math.sin(yaw)
        self.drift += DRIFT_PER_M * abs(step)
        if not self.trail or math.dist(self.trail[-1], (self.x, self.y)) >= TRAIL_STEP_M:
            self.trail.append((round(self.x, 3), round(self.y, 3)))

    def command(self, cm_per_s, hold_s=math.inf):
        """The wheels were told to move at cm_per_s (signed, forward +) for hold_s seconds."""
        with self._lock:
            now = time.monotonic()
            self._advance(now)
            self._v = cm_per_s / 100.0
            self._until = now + hold_s

    def wheels(self, left, right):
        self.command((math.copysign(drive.speed(left), left) + math.copysign(drive.speed(right), right)) / 2, WATCHDOG_S)

    def drive(self, direction, speed, hold_s=math.inf):
        v = {"forward": drive.speed(speed), "back": -drive.speed(speed)}.get(direction, 0.0)
        self.command(v, hold_s)

    def gyro(self, degrees):
        with self._lock:
            self._advance(time.monotonic())
            if self._gyro is not None:
                delta = (degrees - self._gyro + 180) % 360 - 180
                self._turned += math.radians(delta)
                if self.placed:
                    self.drift += DRIFT_PER_DEG * abs(delta) / 10
            self._gyro = degrees

    def place(self, x, y, yaw):
        with self._lock:
            self._advance(time.monotonic())
            self.x, self.y = float(x), float(y)
            self._offset = float(yaw) - self._turned
            self.placed = True
            self.drift = DRIFT_START_M
            self.travelled = 0.0
            self.source = "dead-reckoning"
            self.trail.clear()
            self.trail.append((round(self.x, 3), round(self.y, 3)))

    def fix(self, pose):
        """A confident localiser pose (go_to's particle filter) replaces the dead-reckoned one."""
        x, y, yaw = pose
        with self._lock:
            self._advance(time.monotonic())
            self.x, self.y = float(x), float(y)
            self._offset = float(yaw) - self._turned
            self.placed = True
            self.drift = FIX_DRIFT_M
            self.source = "localiser"
            self.trail.append((round(self.x, 3), round(self.y, 3)))

    def pose(self):
        with self._lock:
            self._advance(time.monotonic())
            return (self.x, self.y, self.yaw) if self.placed else None

    def snapshot(self):
        with self._lock:
            self._advance(time.monotonic())
            moving = time.monotonic() < self._until
            return {
                "placed": self.placed,
                "x": round(self.x, 3),
                "y": round(self.y, 3),
                "yaw": round(self.yaw, 4),
                "v_cms": round(self._v * 100, 1) if moving else 0.0,
                "room": self.map.room_at(self.x, self.y) if self.placed else None,
                "drift_m": round(self.drift, 2),
                "travelled_m": round(self.travelled, 2),
                "source": self.source,
                "trail": list(self.trail),
                "home": self.home,
            }

    def describe(self):
        pose = self.pose()
        if pose is None:
            return ("Nobody has marked my start spot on the map yet, so I have no map position. "
                    "Ask the owner to click where I am on the dashboard map.")
        x, y, yaw = pose
        room = self.map.room_at(x, y)
        where = f"in the {room}" if room else "between rooms (outside every room outline)"
        text = (f"By dead reckoning I'm {where}, at ({x:.1f}, {y:.1f}) m on the map, facing {compass(yaw)} "
                f"({math.degrees(yaw) % 360:.0f} deg), give or take {self.drift:.1f} m. "
                f"I've driven {self.travelled:.1f} m since my start was set.")
        if self.home:
            text += " " + self.directions(self.home["x"], self.home["y"], "home")
        return text

    def directions(self, gx, gy, name):
        """Straight-line distance and the turn to face (gx, gy) from the current pose."""
        pose = self.pose()
        if pose is None:
            return f"I can't point to {name}: my start spot isn't set on the map."
        x, y, yaw = pose
        dist = math.hypot(gx - x, gy - y)
        if dist < 0.3:
            return f"I'm basically at {name} (within {dist * 100:.0f} cm)."
        turn = math.degrees(math.atan2(gy - y, gx - x) - yaw)
        turn = (turn + 180) % 360 - 180
        side = "left" if turn > 0 else "right"
        aim = "straight ahead" if abs(turn) < 8 else f"{abs(turn):.0f} deg to my {side}"
        return (f"{name.capitalize()} is {dist:.1f} m away as the crow flies, {aim}. Walls may be in between; "
                "go_to plans a real route.")
