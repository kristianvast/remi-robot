import json
import re
import socket
import threading
import time

import serial

DEFAULT_PORT = "/dev/cu.usbserial-11210"
DEFAULT_HOST = "elegoo-car.local"
REPLY = re.compile(rb"\{([^{}]*)\}")
HEARTBEAT = b"{Heartbeat}"
# N=2/N=3 direction codes: code 1 spins the car to its left (checked on the real car 2026-11-24).
DIRECTIONS = {"left": 1, "right": 2, "forward": 3, "back": 4}
MAX_RANGE_CM = 150
RANGE_HOLD_CM, RANGE_HOLD_S = 35, 1.0  # see Robot.distance


class WifiLink:
    def __init__(self, host=DEFAULT_HOST, port=100, connect_timeout=5):
        self.sock = socket.create_connection((host, port), timeout=connect_timeout)
        self.sock.settimeout(0.2)
        self._buf = bytearray()
        self._buf_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._alive = True
        self.error = None
        self._pump = threading.Thread(target=self._run, daemon=True)
        self._pump.start()

    def _run(self):
        while self._alive:
            try:
                chunk = self.sock.recv(1024)
            except socket.timeout:
                continue
            except OSError as e:
                self.error = e
                break
            if not chunk:
                self.error = ConnectionError("car closed the connection")
                break
            with self._buf_lock:
                self._buf += chunk
                beats = self._buf.count(HEARTBEAT)
                if beats:
                    self._buf = self._buf.replace(HEARTBEAT, b"")
            for _ in range(beats):
                self.write(HEARTBEAT)

    def _check(self):
        if self.error:
            raise ConnectionError(f"Wi-Fi link lost: {self.error}")

    def write(self, data):
        with self._send_lock:
            self.sock.sendall(data)

    def read(self, _size):
        self._check()
        with self._buf_lock:
            end = self._buf.rfind(b"}") + 1
            out = bytes(self._buf[:end])
            del self._buf[:end]
        if not out:
            time.sleep(0.02)
        return out

    def reset_input_buffer(self):
        with self._buf_lock:
            end = self._buf.rfind(b"}") + 1
            del self._buf[:end]

    def close(self):
        self._alive = False
        self._pump.join(timeout=1)
        self.sock.close()


class Robot:
    def __init__(self, link, reply_timeout=0.6):
        self.link = link
        self.reply_timeout = reply_timeout
        self._id = 0
        # Gyro drift, see _status: the firmware zeroes the gyro once at power-on and a residual of 1-2 deg/s
        # was measured on 2026-11-24, so the drift is re-estimated whenever the car rests.
        self._drift = self._corr = 0.0  # deg/s now; degrees removed so far
        self._corr_t = time.monotonic()
        self._moved_at, self._held, self._rest = 0.0, False, None
        self._head, self._close = 90, None  # last head angle sent; last close range (time, cm, head)

    @classmethod
    def usb(cls, port=DEFAULT_PORT, baud=9600, boot_wait=2.5):
        ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(boot_wait)
        ser.reset_input_buffer()
        return cls(ser)

    @classmethod
    def wifi(cls, host=DEFAULT_HOST, port=100):
        return cls(WifiLink(host, port), reply_timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.stop()
        finally:
            self.link.close()

    def _request(self, payload):
        self._id = self._id % 9000 + 1
        hid = str(self._id)
        msg = json.dumps({"H": hid, **payload}, separators=(",", ":")).encode()
        self.link.reset_input_buffer()
        self.link.write(msg)
        buf = b""
        deadline = time.monotonic() + self.reply_timeout
        while time.monotonic() < deadline:
            buf += self.link.read(64)
            for m in REPLY.finditer(buf):
                body = m.group(1).decode(errors="replace")
                if body.startswith(hid + "_"):
                    return body[len(hid) + 1:]
        raise TimeoutError(f"no reply to {msg.decode()}; got {buf!r}")

    def _moving(self, until=0.0, held=False):
        self._moved_at, self._held, self._rest = max(time.monotonic(), until), held, None

    def stop(self):
        self._request({"N": 110})  # clears every running function and, unlike N=100, is acknowledged
        self._held = False

    def drive(self, direction, speed):
        self._moving(held=True)  # N=3 runs until stop()
        self._request({"N": 3, "D1": DIRECTIONS[direction], "D2": max(0, min(255, int(speed)))})

    def move(self, direction, speed, ms):
        self._moving(until=time.monotonic() + ms / 1000)
        self._request({"N": 2, "D1": DIRECTIONS[direction], "D2": max(0, min(255, int(speed))), "T": int(ms)})

    def _status(self, reply):
        yaw, grounded = reply.split(",")
        # The firmware's yaw grows clockwise; the brain's heading grows to the left (measured 2026-11-24).
        raw, now = -int(yaw) / 10.0, time.monotonic()
        self._corr += self._drift * (now - self._corr_t)
        self._corr_t = now
        # At rest (no wheel command for 1 s, none held), every 2 s of heading change is drift.
        if self._held or now - self._moved_at < 1.0:
            self._rest = None
        elif self._rest is None:
            self._rest = (now, raw)
        elif now - self._rest[0] >= 2.0:
            rate = (raw - self._rest[1]) / (now - self._rest[0])
            if abs(rate) < 10:  # larger means the car was carried or bumped
                self._drift = rate
            self._rest = (now, raw)
        return raw - self._corr, grounded == "1"

    def wheels(self, left, right):
        """Signed wheel speeds, -255..255. The firmware stops them 0.4 s after the last command.

        Returns (heading in degrees, on ground).
        """
        if left or right:
            self._moving()
        else:
            self._held = False
        # D1 drives the left wheels: wheels(-90, 90) spins the car counter-clockwise (measured 2026-11-24).
        return self._status(self._request({"N": 4, "D1": int(left), "D2": int(right)}))

    def heading(self):
        return self._status(self._request({"N": 24}))

    def head(self, angle):
        self._head = max(0, min(180, int(angle)))
        self._request({"N": 5, "D1": 1, "D2": self._head})

    def distance(self):
        """Range ahead of the head in cm, 1..MAX_RANGE_CM.

        The firmware answers 0 for an echo under 1 cm (touching) and MAX_RANGE_CM when no echo returns, which
        also happens right at a wall or at a slant (seen on the car 2026-11-24). So a MAX right after a close
        reading at the same head angle keeps the close reading for up to RANGE_HOLD_S.
        """
        cm = max(1, min(MAX_RANGE_CM, int(self._request({"N": 21, "D1": 2}))))
        now = time.monotonic()
        held = self._close
        if cm >= MAX_RANGE_CM and held and held[2] == self._head and now - held[0] < RANGE_HOLD_S:
            return held[1]
        self._close = (now, cm, self._head) if cm <= RANGE_HOLD_CM else None
        return cm

    def line(self):
        return tuple(int(self._request({"N": 22, "D1": i})) for i in range(3))

    def on_ground(self):
        return self._request({"N": 23}) == "false"  # N=23 answers "has the car left the ground?"
