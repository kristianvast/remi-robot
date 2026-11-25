"""Shared dashboard state and recording of robot commands and sensor reads."""

import collections
import time

import mapping
from pilot import HEAD_CENTER

SONAR_KEEP_S = 8.0


class Telemetry:
    def __init__(self, stop_cm):
        self.version = 0
        self.link = {"state": "connecting", "rtt_ms": None, "error": None}
        self.sensors = {"distance": None, "distance_t": None, "head": HEAD_CENTER, "line": None, "on_ground": None}
        self.sonar = collections.deque(maxlen=120)
        self.motion = {"by": None, "dir": None}
        self.frame = {"id": 0, "t": None, "jpeg": None}
        self.agent = {"state": "initializing", "user_state": "listening", "autonomy": True, "initiative": False, "mic": True}
        self.transcript = collections.deque(maxlen=40)
        self.tools = collections.deque(maxlen=24)
        self.notices = collections.deque(maxlen=10)
        self.stop_cm = stop_cm
        self.map = mapping.BuildingMap()
        self.odometry = mapping.Odometry(self.map)
        self.memory = {"people": 0, "events": 0}
        self._ids = 0

    def touch(self):
        self.version += 1

    def next_id(self):
        self._ids += 1
        return self._ids

    def set_frame(self, jpeg):
        self.frame = {"id": self.frame["id"] + 1, "t": time.time(), "jpeg": jpeg}
        self.touch()

    def add_distance(self, head, cm):
        now = time.time()
        self.sensors.update(distance=cm, distance_t=now)
        self.sonar.append((head, cm, now))
        self.touch()

    def moved(self, by, direction):
        if direction is not None:
            self.sonar.clear()  # readings are relative to the car, so they go stale once it moves
        self.motion = {"by": by, "dir": direction}
        self.touch()

    def say(self, role, text, final=True):
        last = self.transcript[-1] if self.transcript else None
        if role == "user" and last and last["role"] == "user" and (not last["final"] or last["text"] == text):
            last.update(text=text, final=final, t=time.time())
        else:
            self.transcript.append({"id": self.next_id(), "role": role, "text": text, "final": final, "t": time.time()})
        self.touch()

    def tool_start(self, label):
        entry = {"id": self.next_id(), "label": label, "status": "running", "result": None, "t0": time.time(), "t1": None}
        self.tools.append(entry)
        self.touch()
        return entry

    def tool_end(self, entry, result, ok=True):
        entry.update(status="done" if ok else "error", result=result, t1=time.time())
        self.touch()

    def notice(self, text, level="info"):
        self.notices.append({"id": self.next_id(), "text": text, "level": level, "t": time.time()})
        self.touch()

    def snapshot(self):
        now = time.time()
        return {
            "t": "state",
            "now": now,
            "link": self.link,
            "sensors": self.sensors,
            "sonar": [[h, cm, round(now - t, 2)] for h, cm, t in self.sonar if now - t < SONAR_KEEP_S],
            "motion": self.motion,
            "frame": {"id": self.frame["id"], "t": self.frame["t"]},
            "agent": self.agent,
            "transcript": list(self.transcript),
            "tools": list(self.tools),
            "notices": list(self.notices),
            "stop_cm": self.stop_cm,
            "pose": self.odometry.snapshot(),
            "memory": self.memory,
        }


class Recorder:
    def __init__(self, robot, telemetry, by):
        self._robot = robot
        self._t = telemetry
        self._by = by

    def _timed(self, fn, *args):
        start = time.monotonic()
        result = fn(*args)
        self._t.link.update(state="online", rtt_ms=round((time.monotonic() - start) * 1000), error=None)
        return result

    def head(self, angle):
        angle = max(0, min(180, int(angle)))
        self._timed(self._robot.head, angle)
        self._t.sensors["head"] = angle
        self._t.touch()

    def distance(self):
        cm = self._timed(self._robot.distance)
        self._t.add_distance(self._t.sensors["head"], cm)
        return cm

    def line(self):
        values = self._timed(self._robot.line)
        self._t.sensors["line"] = list(values)
        self._t.touch()
        return values

    def on_ground(self):
        grounded = self._timed(self._robot.on_ground)
        self._t.sensors["on_ground"] = grounded
        self._t.touch()
        return grounded

    def heading(self):
        status = self._timed(self._robot.heading)
        self._t.odometry.gyro(status[0])
        self._t.touch()
        return status

    def drive(self, direction, speed):
        self._timed(self._robot.drive, direction, speed)
        self._t.odometry.drive(direction, speed)
        self._t.moved(self._by, direction)

    def move(self, direction, speed, ms):
        self._t.moved(self._by, direction)
        self._t.odometry.drive(direction, speed, ms / 1000)
        try:
            self._timed(self._robot.move, direction, speed, ms)
        finally:
            self._t.moved(None, None)

    def wheels(self, left, right):
        if not (left or right):
            direction = None
        elif (left > 0) == (right > 0):
            direction = "forward" if left > 0 else "back"
        else:
            direction = "left" if right > left else "right"
        if direction != self._t.motion["dir"]:
            self._t.moved(self._by if direction else None, direction)
        status = self._timed(self._robot.wheels, left, right)
        self._t.odometry.wheels(left, right)
        self._t.odometry.gyro(status[0])
        self._t.sensors["on_ground"] = status[1]
        return status

    def stop(self):
        self._timed(self._robot.stop)
        self._t.odometry.command(0.0)
        self._t.moved(None, None)

    def __getattr__(self, name):
        return getattr(self._robot, name)
