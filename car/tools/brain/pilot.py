"""Manual driving and sensor polling for the browser control pane."""

import asyncio
import logging
import time

log = logging.getLogger("car-dashboard")

HEAD_CENTER = 90
HEAD_MIN, HEAD_MAX = 10, 170
HEAD_STEP = 15  # each servo command takes ~0.5 s on the car, so small steps crawl
POLL_S = 0.12
WATCHDOG_S = 0.4  # manual drive stops when the browser stops confirming the key is held
SAFETY_S = 0.12


class Pilot:
    def __init__(self, car, telemetry, speed, scan):
        self.car = car
        self.t = telemetry
        self.speed = speed
        self._scan = scan
        self.drive_dir = None
        self.drive_seen = 0.0
        self.pan_dir = 0
        self.pan_seen = 0.0
        self._driving = None
        self._last_safety = 0.0
        self._last_pan = 0.0
        self._poll_i = 0

    async def _car(self, fn, by="human"):
        return await asyncio.to_thread(self.car.run_quiet, fn, by)

    def set_drive(self, direction):
        if direction is not None:
            self.car.human_input()
        self.drive_dir = direction
        self.drive_seen = time.monotonic()

    def set_pan(self, direction):
        if direction:
            self.car.human_input()
        self.pan_dir = direction
        self.pan_seen = time.monotonic()

    def heartbeat(self):
        now = time.monotonic()
        if self.drive_dir:
            self.drive_seen = now
            self.car.human_input()
        if self.pan_dir:
            self.pan_seen = now

    def release_all(self):
        self.drive_dir = None
        self.pan_dir = 0

    async def estop(self):
        self.release_all()
        self.car.human_input()
        await self._car(lambda r: r.stop())
        self._driving = None
        self.t.notice("Emergency stop", "warn")

    async def center(self):
        self.car.human_input()
        await self._car(lambda r: r.head(HEAD_CENTER))

    async def scan(self):
        self.car.human_input()
        self.t.notice("Scanning surroundings")
        await self._car(self._scan)

    def _blocked(self, r):
        if not r.on_ground():
            return "No floor under the front sensors"
        if abs(self.t.sensors["head"] - HEAD_CENTER) > 3:
            r.head(HEAD_CENTER)
        cm = r.distance()
        if cm < self.t.stop_cm:
            return f"Obstacle {cm} cm ahead"
        return None

    async def _drive_tick(self, now):
        want = self.drive_dir
        if want and now - self.drive_seen > WATCHDOG_S:
            self.drive_dir = want = None
            self.t.notice("Control signal lost, stopped", "warn")
        if want != self._driving:
            if want is None:
                await self._car(lambda r: r.stop())
                self._driving = None
                return
            if want == "forward":
                reason = await self._car(self._blocked)
                if reason:
                    self.drive_dir = None
                    self.t.notice(f"{reason}, not driving forward", "warn")
                    if self._driving:
                        await self._car(lambda r: r.stop())
                        self._driving = None
                    return
            def move(r):
                if self.drive_dir == want and time.monotonic() - self.drive_seen <= WATCHDOG_S:
                    r.drive(want, self.speed)
                    return True
                return False

            if not await self._car(move):
                return
            self._driving = want
            self._last_safety = now
        elif self._driving == "forward" and now - self._last_safety > SAFETY_S:
            self._last_safety = now
            reason = await self._car(self._blocked)
            if reason:
                await self._car(lambda r: r.stop())
                self._driving = self.drive_dir = None
                self.t.notice(f"{reason}, stopped", "warn")
        elif self._driving:
            await self._car(lambda r: r.heading())  # the map pose turns with the gyro while the human drives

    async def _pan_tick(self, now):
        if self.pan_dir and now - self.pan_seen > WATCHDOG_S:
            self.pan_dir = 0
        if self.pan_dir and now - self._last_pan > 0.08:
            self._last_pan = now
            target = max(HEAD_MIN, min(HEAD_MAX, self.t.sensors["head"] + HEAD_STEP * self.pan_dir))
            if target != self.t.sensors["head"]:
                await self._car(lambda r: r.head(target))

    def _poll(self, r):
        self._poll_i += 1
        slot = self._poll_i % 6
        if slot == 0:
            r.line()
        elif slot == 3:
            r.on_ground()
        else:
            r.distance()

    async def run(self):
        while True:
            now = time.monotonic()
            try:
                await self._drive_tick(now)
                await self._pan_tick(now)
                if self._driving is None and not self.pan_dir:
                    await asyncio.to_thread(self.car.try_run, self._poll)
            except ValueError as e:
                log.warning("pilot: unreadable reply from the car: %s", e)
            await asyncio.sleep(POLL_S if self._driving is None and not self.pan_dir else 0.04)
