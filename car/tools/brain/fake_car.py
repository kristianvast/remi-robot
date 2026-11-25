"""Stand-in for the car, for working on the Mac without hardware.

    .venv/bin/python fake_car.py --port 18100 --camera-port 18101

Speaks the car's JSON-over-TCP protocol ({"H":id,"N":cmd,...} answered by {id_reply}) with a
45 ms reply delay, sends {Heartbeat} and expects it echoed like the firmware, and moves a simulated
car in a rectangular room: N=4 wheel speeds turn into heading and position, and the wheels stop by
themselves 0.4 s after the last N=4. N=21 measures the distance to whatever is ahead of the head.
The camera port serves /capture (one JPEG rendered from the car's pose, camera upright and tilted
up 20 degrees like the real one) and /stream (MJPEG). Every command is logged to stdout.

Room: 600 x 400 cm, x to the east, y to the north, heading 0 = facing east, positive = left (counter-
clockwise seen from above), as the real gyro reports it. Walls: north red, east green, south blue,
west yellow, with a brown door in the east wall; an orange box stands in the room.
"""

import argparse
import asyncio
import json
import time

from fake_camera import camera_server
from fake_world import Car, log

REPLY_DELAY_S = 0.045
HEARTBEAT_S = 1.0


def split_messages(buf):
    """Pull complete {...} messages out of the buffer; returns (messages, rest)."""
    out = []
    while True:
        start = buf.find(b"{")
        end = buf.find(b"}", start)
        if start < 0 or end < 0:
            return out, buf[start:] if start >= 0 else b""
        out.append(buf[start:end + 1])
        buf = buf[end + 1:]


async def serve_client(car, reader, writer):
    peer = writer.get_extra_info("peername")
    log(f"client {peer} connected")
    beats = {"sent": 0, "echoed": 0}

    async def heartbeat():
        while True:
            writer.write(b"{Heartbeat}")
            beats["sent"] += 1
            await writer.drain()
            await asyncio.sleep(HEARTBEAT_S)

    pulse = asyncio.create_task(heartbeat())
    buf = b""
    try:
        while chunk := await reader.read(1024):
            if car.drop_until > time.monotonic():
                break
            msgs, buf = split_messages(buf + chunk)
            for raw in msgs:
                if raw == b"{Heartbeat}":
                    beats["echoed"] += 1
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
                x, y, heading, head = car.pose()
                args = " ".join(f"{k}={v}" for k, v in msg.items() if k not in ("H", "N"))
                wheels = f" L={msg.get('D1')} R={msg.get('D2')}" if msg.get("N") == 4 else ""
                log(f"N={msg.get('N')} {args}{wheels} -> {reply}  pose x={x:.1f} y={y:.1f} heading={heading:.1f} head={head}")
    except OSError as e:
        log(f"client {peer} error: {e}")
    finally:
        pulse.cancel()
        writer.close()
        log(f"client {peer} gone (heartbeats sent {beats['sent']}, echoed {beats['echoed']})")


async def physics(car, clients):
    """Advance the watchdog independently of incoming packets and camera requests."""
    while True:
        car.pose()
        if car.drop_until > time.monotonic():
            for writer in tuple(clients):
                writer.transport.abort()
        await asyncio.sleep(0.01)


async def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--camera-port", type=int, default=18101)
    ap.add_argument("--test-controls", action="store_true", help="enable loopback fixture controls at /test")
    args = ap.parse_args()
    car = Car()
    camera_server(car, args.camera_port, args.test_controls)
    clients = set()

    async def connected(reader, writer):
        clients.add(writer)
        try:
            await serve_client(car, reader, writer)
        finally:
            clients.discard(writer)

    server = await asyncio.start_server(connected, "127.0.0.1", args.port)
    log(f"fake car on 127.0.0.1:{args.port}, camera http://127.0.0.1:{args.camera_port}/capture and /stream")
    async with server, asyncio.TaskGroup() as group:
        group.create_task(physics(car, clients))
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
