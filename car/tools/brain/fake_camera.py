"""Synthetic camera rendering and HTTP endpoints for the loopback stand-in."""

import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw, ImageFont

from fake_world import (
    BOX_TALL,
    CENTER,
    DOOR_CM,
    DOOR_Y,
    PERSON_RADIUS,
    ROOM_H,
    ROOM_W,
    WALL_CM,
    blocked,
    log,
    ray_box,
    ray_walls,
)

CAM_H = 15.0
CAM_PITCH = math.radians(20)
W, H = 640, 480
FX = (W / 2) / math.tan(math.radians(30))  # 60 degree horizontal view
FY = (H / 2) / math.tan(math.radians(24))  # 48 degree vertical view
HORIZON = H / 2 + FY * math.tan(CAM_PITCH)  # looking up moves the horizon down the picture
COLUMNS = 160
WALLS = {"north": (200, 40, 40), "east": (40, 160, 60), "south": (50, 80, 200), "west": (220, 190, 40)}
BOX_COLOR = (240, 130, 20)
DOOR_COLOR = (90, 55, 25)
CEILING_COLOR = (205, 205, 210)
FLOOR_COLOR = (120, 90, 60)


def render(car):
    scene = car.test_state()
    if scene["view"]:
        level = 8 if scene["view"] == "dark" else 150
        noise = Image.effect_noise((W, H), 3).point(lambda v: max(0, min(255, level + (v - 128))))
        out = io.BytesIO()
        noise.convert("RGB").save(out, "JPEG", quality=80)
        return out.getvalue(), ""
    x, y, heading, head = (scene[k] for k in ("x", "y", "heading", "head"))
    view = math.radians(heading + head - CENTER)
    img = Image.new("RGB", (W, H), CEILING_COLOR)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(HORIZON), W, H], fill=FLOOR_COLOR)
    seen = {}
    step = W // COLUMNS
    for c in range(COLUMNS):
        left, right = c * step, (c + 1) * step - 1
        off = math.atan(((c + 0.5) * step - W / 2) / FX)
        ang = view - off  # picture x grows to the right, angles grow to the left
        d, wall = ray_walls(x, y, ang)
        z = max(1.0, d * math.cos(off))
        shade = 0.55 + 0.45 * min(1.0, 120 / z)
        color = tuple(int(v * shade) for v in WALLS[wall])
        top, bottom = HORIZON - FY * (WALL_CM - CAM_H) / z, HORIZON + FY * CAM_H / z
        draw.rectangle([left, int(top), right, int(bottom)], fill=color)
        seen[wall] = min(seen.get(wall, math.inf), d)
        if wall == "east" and DOOR_Y[0] <= y + d * math.sin(ang) <= DOOR_Y[1]:
            draw.rectangle([left, int(HORIZON - FY * (DOOR_CM - CAM_H) / z), right, int(bottom)], fill=DOOR_COLOR)
            seen["door"] = min(seen.get("door", math.inf), d)
        b = ray_box(x, y, ang)
        if b is not None and b < d:
            zb = max(1.0, b * math.cos(off))
            draw.rectangle([left, int(HORIZON - FY * (BOX_TALL - CAM_H) / zb), right, int(HORIZON + FY * CAM_H / zb)], fill=BOX_COLOR)
            seen["orange box"] = min(seen.get("orange box", math.inf), b)
    font = ImageFont.load_default(size=18)
    lines = ["FAKE CAR CAMERA (synthetic room)"] + [f"{name}{' wall' if name in WALLS else ''}: {d:.0f} cm" for name, d in sorted(seen.items(), key=lambda kv: kv[1])]
    lines.append(f"pose x={x:.0f} y={y:.0f} heading={heading:.0f} head={head}")
    draw.rectangle([0, 0, 360, 8 + 22 * len(lines)], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((8, 4 + 22 * i), line, fill=(255, 255, 255), font=font)
    fixture = ""
    if scene["person"]:
        px, py = scene["person"]
        dx, dy = px - x, py - y
        z = dx * math.cos(view) + dy * math.sin(view)
        if z > 10:
            cx = W / 2 + FX * (dx * math.sin(view) - dy * math.cos(view)) / z
            half = FX * PERSON_RADIUS / z
            top = HORIZON - FY * (165 - CAM_H) / z
            base = HORIZON + FY * CAM_H / z
            x0, x1 = max(0, cx - half), min(W, cx + half)
            y0, y1 = max(0, top), min(H, base)
            if x1 - x0 > 5 and y1 - y0 > 12:
                height = base - top
                draw.ellipse((cx - half / 2, top, cx + half / 2, top + 0.18 * height), fill="#d8a581")
                draw.rounded_rectangle((cx - half, top + 0.17 * height, cx + half, top + 0.60 * height),
                                       radius=int(half / 4), fill="#355cc3")
                for side in (-1, 1):
                    mid = cx + side * 0.53 * half
                    draw.rectangle((mid - 0.32 * half, top + 0.58 * height, mid + 0.32 * half, base),
                                   fill="#202030")
                fixture = ",".join(f"{v:.5f}" for v in (x0 / W, y0 / H, x1 / W, y1 / H))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue(), fixture


def camera_server(car, port, test_controls):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _state(self):
            data = json.dumps(car.test_state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not test_controls or self.path != "/test":
                self.send_error(404)
                return
            try:
                size = int(self.headers["Content-Length"])
                if not 0 < size <= 1024:
                    raise ValueError("wrong request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or not data or data.keys() - {
                    "person", "lifted", "drop_link", "pose", "barrier", "view"
                }:
                    raise ValueError("unknown test command")
                barrier = data.get("barrier")
                if barrier is not None:
                    x0, y0, x1, y1 = map(float, barrier)
                    if not (x0 < x1 and y0 < y1):
                        raise ValueError("barrier needs x0 < x1 and y0 < y1")
                    barrier = (x0, y0, x1, y1)
                if data.get("view") not in (None, "blank", "dark"):
                    raise ValueError("view is blank, dark or null")
                pose = None
                if "pose" in data:
                    values = data["pose"]
                    if not isinstance(values, list) or len(values) != 3 or any(
                        type(value) not in (int, float) or not math.isfinite(value) for value in values
                    ):
                        raise ValueError("pose needs three finite numbers")
                    x, y, heading = values
                    if blocked(x, y):
                        raise ValueError("pose intersects a wall or the box")
                    pose = (x, y, heading)
                with car.lock:
                    car._advance()
                    if pose is not None:
                        car._set_wheels(0, 0, "pose reset")
                        car.x, car.y, car.heading = pose
                        car.head = CENTER
                    if "person" in data:
                        person = data["person"]
                        if person is not None:
                            px, py = map(float, person)
                            if not (0 <= px <= ROOM_W and 0 <= py <= ROOM_H):
                                raise ValueError("person outside room")
                            person = (px, py)
                        car.person = person
                    if "lifted" in data:
                        car.lifted = data["lifted"] is True
                    if data.get("drop_link") is True:
                        car.drop_until = time.monotonic() + 1
                    if "barrier" in data:
                        car.barrier = barrier
                    if "view" in data:
                        car.view = data["view"]
                    log(f"FIXTURE {data} monotonic={time.monotonic():.6f}")
                self._state()
            except (KeyError, OverflowError, TypeError, ValueError):
                self.send_error(400)

        def do_GET(self):
            if self.path == "/test" and test_controls:
                self._state()
            elif self.path.startswith("/capture"):
                jpeg, fixture = render(car)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                if test_controls and fixture:
                    self.send_header("X-Fake-Person", fixture)
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpeg, _ = render(car)
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.2)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
