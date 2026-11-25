"""Room geometry and simulated car state for the loopback stand-in."""

import math
import threading
import time

ROOM_W, ROOM_H = 600.0, 400.0
WALL_CM = 250.0
DOOR_Y = (160.0, 250.0)  # span along the east wall
DOOR_CM = 200.0
BOX = (420.0, 280.0, 480.0, 340.0)  # x0, y0, x1, y1; 40 cm tall
BOX_TALL = 40.0
CAR_RADIUS = 12.0
START = (150.0, 200.0, 0.0)  # x, y, heading
PERSON_RADIUS = 17

SELF_STOP_S = 0.4
CM_PER_S, CRUISE, DEAD = 55.0, 150, 12  # same speed line as drive.py's default calibration
TRACK_CM = 24.0  # effective, so PWM 80 spins about 130 deg/s like the real car
CENTER = 90

def log(text):
    print(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} {text}", flush=True)


def wheel_speed(pwm):
    return math.copysign(CM_PER_S * max(0.0, abs(pwm) - DEAD) / (CRUISE - DEAD), pwm)


def ray_walls(x, y, ang):
    """Distance along a ray from inside the room to the first wall, and that wall's name."""
    dx, dy = math.cos(ang), math.sin(ang)
    hits = []
    if dx > 1e-9:
        hits.append(((ROOM_W - x) / dx, "east"))
    if dx < -1e-9:
        hits.append((-x / dx, "west"))
    if dy > 1e-9:
        hits.append(((ROOM_H - y) / dy, "north"))
    if dy < -1e-9:
        hits.append((-y / dy, "south"))
    return min(hits)


def ray_box(x, y, ang):
    """Distance along a ray to the box, or None."""
    dx, dy = math.cos(ang), math.sin(ang)
    near, far = -math.inf, math.inf
    for o, d, lo, hi in ((x, dx, BOX[0], BOX[2]), (y, dy, BOX[1], BOX[3])):
        if abs(d) < 1e-9:
            if not lo <= o <= hi:
                return None
            continue
        t0, t1 = sorted(((lo - o) / d, (hi - o) / d))
        near, far = max(near, t0), min(far, t1)
    return near if near <= far and near > 0 else None


def blocked(x, y):
    r = CAR_RADIUS
    if not (r <= x <= ROOM_W - r and r <= y <= ROOM_H - r):
        return True
    return BOX[0] - r < x < BOX[2] + r and BOX[1] - r < y < BOX[3] + r


class Car:
    def __init__(self):
        self.lock = threading.Lock()
        self.x, self.y, self.heading = START
        self.left = self.right = 0
        self.head = CENTER
        self.last_wheels = 0.0
        self.t = time.monotonic()
        self.person = None
        self.lifted = False
        self.drop_until = 0.0
        self.zero_at = 0.0
        self.barrier = None  # x0, y0, x1, y1: stops the car, but neither the range sensor nor the camera sees it
        self.view = None  # "blank" (a plain surface fills the view) or "dark" replaces the rendered picture

    def _set_wheels(self, left, right, source):
        if (self.left or self.right) and not (left or right):
            self.zero_at = time.monotonic()
            log(f"WHEELS ZERO L=0 R=0 source={source} monotonic={self.zero_at:.6f}")
        self.left, self.right = left, right

    def _advance(self):
        now = time.monotonic()
        while self.t < now:
            dt = min(0.01, now - self.t)
            if self.t >= self.last_wheels + SELF_STOP_S:
                self._set_wheels(0, 0, "watchdog")
            vl, vr = wheel_speed(self.left), wheel_speed(self.right)
            if vl or vr:
                a = math.radians(self.heading)
                v = (vl + vr) / 2
                nx, ny = self.x + v * math.cos(a) * dt, self.y + v * math.sin(a) * dt
                if not (
                    self.lifted
                    or blocked(nx, ny)
                    or (self.person and math.hypot(nx - self.person[0], ny - self.person[1]) < PERSON_RADIUS + CAR_RADIUS)
                    or (self.barrier and self.barrier[0] - CAR_RADIUS < nx < self.barrier[2] + CAR_RADIUS
                        and self.barrier[1] - CAR_RADIUS < ny < self.barrier[3] + CAR_RADIUS)
                ):
                    self.x, self.y = nx, ny
                self.heading += math.degrees((vr - vl) / TRACK_CM * dt)
            self.t += dt

    def pose(self):
        with self.lock:
            self._advance()
            return self.x, self.y, self.heading, self.head

    def status(self):
        return f"{round(-self.heading * 10)},{int(not self.lifted)}"  # like the firmware: yaw grows clockwise

    def test_state(self):
        with self.lock:
            self._advance()
            return {"x": self.x, "y": self.y, "heading": self.heading, "head": self.head, "left": self.left,
                    "right": self.right, "person": self.person, "lifted": self.lifted, "barrier": self.barrier,
                    "view": self.view, "zero_at": self.zero_at, "now": time.monotonic()}

    def handle(self, msg):
        """Apply one command; returns the reply body."""
        n = msg.get("N")
        with self.lock:
            self._advance()
            if n == 4:
                # like the firmware: D1 = left wheels, D2 = right wheels
                left = max(-255, min(255, int(msg.get("D1", 0))))
                right = max(-255, min(255, int(msg.get("D2", 0))))
                self._set_wheels(left, right, "N=4")
                self.last_wheels = time.monotonic()
                return self.status()
            if n == 3:
                speed = max(0, min(255, int(msg.get("D2", 0))))
                # Like the real firmware: code 1 spins left, 2 spins right.
                left, right = {1: (-speed, speed), 2: (speed, -speed), 3: (speed, speed),
                               4: (-speed, -speed)}.get(msg.get("D1"), (0, 0))
                self._set_wheels(left, right, "manual")
                self.last_wheels = math.inf  # N=3 stays active until an explicit stop
                return "ok"
            if n == 24:
                return self.status()
            if n == 21:
                return str(round(self._ahead()))
            if n == 22:
                return "1000" if self.lifted else "120"
            if n == 23:
                return "true" if self.lifted else "false"
            if n == 5:
                self.head = max(0, min(180, int(msg.get("D2", CENTER))))
                return "ok"
            if n in (110, 100):
                self._set_wheels(0, 0, f"N={n}")
                return "ok"
            return "ok"

    def _ahead(self):
        ang = math.radians(self.heading + self.head - CENTER)
        d, _ = ray_walls(self.x, self.y, ang)
        b = ray_box(self.x, self.y, ang)
        if b is not None:
            d = min(d, b)
        if self.person:
            dx, dy = self.person[0] - self.x, self.person[1] - self.y
            offset = (math.atan2(dy, dx) - ang + math.pi) % math.tau - math.pi
            if abs(offset) < math.radians(13):
                d = min(d, math.hypot(dx, dy) - PERSON_RADIUS)
        return max(2.0, d - 5.0)  # the sensor sits about 5 cm behind the bumper line it sees from
