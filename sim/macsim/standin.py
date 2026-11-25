"""Network stand-in for the car, driven by macsim in the scanned building. From the project root:

    sim/macsim/.venv/bin/python -m sim.macsim.standin --port 18100 --camera-port 18101 [--spawn 0]

Same JSON-over-TCP protocol and camera endpoints as tools/brain/fake_car.py ({"H":id,"N":cmd,...}
answered by {id_reply} after 45 ms, {Heartbeat} sent every second). The simulator advances in real
time in 10 ms ticks; N=4 packets go through macsim's drive model (delay, rate limit, 0.4 s self-stop).
/capture and /stream serve a depth-shaded render (turbo colour map of the clean 160x120 camera depth,
upscaled to 640x480), not a textured picture. GET /test (with --test-controls) returns the true pose,
the contact count and the true distance driven.

Test only (--test-depth, needs --test-controls): GET /depth returns the simulator's own depth observation
(observation_spec.md: 120x160 float32 .npy, all corruptions, 100 ms lag; 204 before the first frame matures),
so the brain's go_to tests can run without the Mac's depth model. The real car has no /depth: there the brain
turns camera pictures into depth with sim/realdepth.
"""

import argparse
import asyncio
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import degrees, radians

import cv2
import numpy as np
from PIL import Image

from .depth import encode
from .simulator import Simulator
from .world import scene_meta

REPLY_DELAY_S = 0.045
HEARTBEAT_S = 1.0
CENTER = 90


class Car:
    def __init__(self, sim: Simulator):
        self.sim = sim
        self.lock = threading.Lock()
        self.packet: tuple[int, int] | None = None
        self.manual: tuple[int, int] | None = None
        self.started = time.monotonic()
        self.contacts = 0
        self.driven = 0.

    def _tick(self, action):
        x, y, _ = self.sim.pose
        bumped = self.sim.bumped
        self.sim.step(*action, dt=.01)
        self.contacts += self.sim.bumped and not bumped
        self.driven += float(np.hypot(self.sim.pose[0] - x, self.sim.pose[1] - y))

    def advance(self):
        with self.lock:
            due = round((time.monotonic() - self.started) / .01)
            while self.sim.drive.ticks < due:
                action = self.packet or self.manual
                queued = len(self.sim.drive.pending)
                self._tick(action or (None, None))
                if len(self.sim.drive.pending) > queued:
                    self.packet = None

    def status(self):
        return f"{round(-degrees(self.sim.gyro) * 10)},1"  # like the firmware: yaw grows clockwise

    def handle(self, msg):
        self.advance()
        n = msg.get("N")
        with self.lock:
            if n == 4:
                # like the firmware: D1 = left wheels, D2 = right wheels
                left = max(-255, min(255, int(msg.get("D1", 0))))
                right = max(-255, min(255, int(msg.get("D2", 0))))
                self.packet, self.manual = (left, right), None
                return self.status()
            if n == 3:
                speed = max(0, min(255, int(msg.get("D2", 0))))
                self.manual = {1: (-speed, speed), 2: (speed, -speed), 3: (speed, speed),  # 1 spins left
                               4: (-speed, -speed)}.get(msg.get("D1"), (0, 0))
                return "ok"
            if n == 24:
                return self.status()
            if n == 21:
                return str(self.sim.observe(camera=False).range_cm)
            if n == 22:
                return "120"
            if n == 23:
                return "false"
            if n == 5:
                self.sim.set_head(radians(max(0, min(180, int(msg.get("D2", CENTER)))) - CENTER))
                return "ok"
            if n in (110, 100):
                self.manual = self.packet = None
                self.sim.drive.stop()
                return "ok"
            return "ok"

    def state(self):
        self.advance()
        with self.lock:
            x, y, yaw = self.sim.pose
            return dict(x=x, y=y, yaw=yaw, head=self.sim.head, gyro=self.sim.gyro, t=self.sim.time,
                        contacts=self.contacts, driven=self.driven)

    def depth(self) -> bytes | None:
        self.advance()
        with self.lock:
            obs = self.sim.observe()
        if obs.depth is None:
            return None
        out = io.BytesIO()
        np.save(out, obs.depth.astype(np.float32))
        return out.getvalue()

    def render(self) -> bytes:
        self.advance()
        with self.lock:
            z = self.sim.metric_depth()
        shade = (encode(z) * 255).astype(np.uint8)
        rgb = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)[:, :, ::-1]
        rgb[~np.isfinite(z)] = 0
        img = Image.fromarray(rgb).resize((640, 480), Image.BILINEAR)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=85)
        return out.getvalue()


def camera_server(car: Car, port: int, test_controls: bool, test_depth: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/test" and test_controls:
                self._send(json.dumps(car.state()).encode(), "application/json")
            elif self.path == "/depth" and test_controls and test_depth:
                if (body := car.depth()) is None:
                    self.send_response(204)
                    self.end_headers()
                else:
                    self._send(body, "application/octet-stream")
            elif self.path.startswith("/capture"):
                self._send(car.render(), "image/jpeg")
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpeg = car.render()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def split_messages(buf):
    out = []
    while True:
        start = buf.find(b"{")
        end = buf.find(b"}", start)
        if start < 0 or end < 0:
            return out, buf[start:] if start >= 0 else b""
        out.append(buf[start:end + 1])
        buf = buf[end + 1:]


def log(text):
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


async def serve_client(car: Car, reader, writer):
    peer = writer.get_extra_info("peername")
    log(f"client {peer} connected")

    async def heartbeat():
        while True:
            writer.write(b"{Heartbeat}")
            await writer.drain()
            await asyncio.sleep(HEARTBEAT_S)

    pulse = asyncio.create_task(heartbeat())
    buf = b""
    try:
        while chunk := await reader.read(1024):
            msgs, buf = split_messages(buf + chunk)
            for raw in msgs:
                if raw == b"{Heartbeat}":
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    log(f"bad message {raw!r}")
                    continue
                reply = car.handle(msg)
                await asyncio.sleep(REPLY_DELAY_S)
                writer.write(b"{" + f"{msg.get('H')}_{reply}".encode() + b"}")
                await writer.drain()
                s = car.state()
                log(f"N={msg.get('N')} {msg} -> {reply}  true x={s['x']:.2f} y={s['y']:.2f} yaw={degrees(s['yaw']):.1f}")
    except (ConnectionError, OSError) as e:
        log(f"client {peer} error: {e}")
    finally:
        pulse.cancel()
        writer.close()
        log(f"client {peer} gone")


async def physics(car: Car):
    while True:
        car.advance()
        await asyncio.sleep(0.01)


async def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--camera-port", type=int, default=18101)
    ap.add_argument("--spawn", type=int, default=0, help="index into scene_meta.yaml spawn_poses")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="draw the response from calibration.yaml ranges")
    ap.add_argument("--test-controls", action="store_true", help="GET /test returns the true pose")
    ap.add_argument("--test-depth", action="store_true",
                    help="test only, with --test-controls: GET /depth serves the corrupted simulated depth observation")
    ap.add_argument("--pose", help="start pose x,y,yaw (metres, radians) instead of --spawn")
    args = ap.parse_args()
    if args.test_depth and not args.test_controls:
        ap.error("--test-depth needs --test-controls")
    sim = Simulator(seed=args.seed, randomize=args.randomize, corrupt=args.test_depth)
    if args.pose:
        sim.reset(tuple(float(v) for v in args.pose.split(",")))
    else:
        spawn = scene_meta()["spawn_poses"][args.spawn]
        sim.reset((spawn["x"], spawn["y"], spawn["yaw"]))
    car = Car(sim)
    camera_server(car, args.camera_port, args.test_controls, args.test_depth)
    server = await asyncio.start_server(lambda r, w: serve_client(car, r, w), "127.0.0.1", args.port)
    log(f"macsim stand-in on 127.0.0.1:{args.port}, camera http://127.0.0.1:{args.camera_port}/capture")
    async with server, asyncio.TaskGroup() as group:
        group.create_task(physics(car))
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
