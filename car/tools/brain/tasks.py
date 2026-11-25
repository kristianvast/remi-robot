"""Car job arbitration and opt-in initiative rules."""

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event

from livekit.agents import ToolError

import navigation
import patrol
import safe_drive
import vision as car_vision
from telemetry import Recorder

__all__ = ["Job", "Speak", "TaskFn", "TaskManager"]

log = logging.getLogger("car-agent")
TaskFn = Callable[["safe_drive.SafeDrive"], str]
Speak = Callable[[str], Awaitable[bool]]


@dataclass(slots=True)
class Job:
    label: str
    cancel: Event = field(default_factory=Event)
    reason: str = "I was told to stop"
    future: asyncio.Task[str] | None = None
    initiative: bool = False


class TaskManager:
    """One admission gate for voice work, with cooperative cancellation at the wire boundary."""

    def __init__(self, car, vision: car_vision.Vision, speak: Speak, greet: Callable[[], None]):
        self.car, self.vision, self.speak, self.greet = car, vision, speak, greet
        self.t = car.telemetry
        self.active: Job | None = None
        self.where = navigation.Whereabouts(lambda: car.human_epoch)
        self.where.on_fix = self.t.odometry.fix
        self.last = "No task has run yet."
        self._gate = asyncio.Lock()
        self._initiative_task: asyncio.Task[None] | None = None
        self._closed = False
        self._stop_epoch = 0

    def status(self) -> str:
        return f"Running: {self.active.label}." if self.active else self.last

    def cancel(self, reason: str, *, initiative_only: bool = False) -> None:
        job = self.active
        if job and not job.cancel.is_set() and (not initiative_only or job.initiative):
            job.reason = reason
            job.cancel.set()

    async def _join(self) -> None:
        if self.active and self.active.future:
            await asyncio.shield(self.active.future)

    def _blocked(self) -> str | None:
        if self._closed or not self.t.agent["autonomy"]:
            return "the human has paused my driving"
        if self.car.human_active() or self.t.motion["by"] == "human":
            return "the human has the controls"
        return None

    async def dispatch(self, label: str, fn: TaskFn, *, background: bool = True) -> str:
        stop_epoch = self._stop_epoch
        async with self._gate:
            if reason := self._blocked():
                return f"Not started: {reason}."
            self.cancel("replaced by a new request")
            await self._join()
            if stop_epoch != self._stop_epoch:
                return "Not started: a stop was requested."
            if reason := self._blocked():
                return f"Not started: {reason}."
            future = self._launch(label, fn, background)
        if background:
            return f"Started {label}. I’ll report when it ends."
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if self.active and self.active.future is future:
                self.cancel("the request was interrupted")
            raise

    async def stop(self, reason: str) -> str:
        self._stop_epoch += 1
        self.cancel(reason)
        async with self._gate:
            self.cancel(reason)
            await self._join()
            try:
                await asyncio.to_thread(self.car.run, lambda robot: robot.stop())
            except ToolError:
                return "The link is down; I cannot confirm a stop command. The wheels rely on their watchdog."
        return "Stopped."

    def _launch(self, label: str, fn: TaskFn, background: bool, announcement: str | None = None) -> asyncio.Task[str]:
        job = Job(label, initiative=announcement is not None)
        self.active = job
        job.future = asyncio.create_task(self._run(job, fn, background, announcement))
        return job.future

    async def _run(self, job: Job, fn: TaskFn, background: bool, announcement: str | None) -> str:
        entry = self.t.tool_start(job.label)
        human_epoch = self.car.human_epoch

        def reason() -> str | None:
            if job.cancel.is_set():
                return job.reason
            if self.car.human_epoch != human_epoch:
                return "the human took over the controls"
            return self._blocked()

        def run(robot: Recorder) -> str:
            r = safe_drive.SafeDrive(robot, reason, self.vision, self.where)
            try:
                r.check()
                return fn(r)
            finally:
                robot.stop()

        ok = False
        result = f"{job.label} ended unexpectedly."
        try:
            if announcement:
                with (patrol.PATROLS / "initiative.log").open("a") as history:
                    history.write(f"{datetime.now().astimezone().isoformat()} announcing {job.label}\n")
                # Poll cancellation while speech is queued; never move after interrupted speech.
                voice = asyncio.create_task(self.speak(announcement))
                try:
                    while not voice.done():
                        if why := reason():
                            raise safe_drive.EndTask(why)
                        await asyncio.wait({voice}, timeout=0.10)
                    if not voice.result():
                        raise safe_drive.EndTask("I could not finish announcing it")
                finally:
                    if not voice.done():
                        voice.cancel()
                        await asyncio.gather(voice, return_exceptions=True)
            result = await asyncio.to_thread(self.car.run, run)
            ok = True
        except safe_drive.EndTask as e:
            result = f"{job.label} stopped: {e}."
        except ToolError:
            result = f"{job.label} stopped because I lost the car link. The wheels rely on their watchdog."
        except (ValueError, OSError):
            log.exception("Car job failed: %s", job.label)
        finally:
            self.last = result
            self.t.tool_end(entry, result, ok)
            self.active = None
        if background and not self._closed:
            self.t.notice(result)
            # Do not hold the motion owner while speech plays.
            self._report(result)
        return result

    def _report(self, result: str) -> None:
        async def report() -> None:
            try:
                await self.speak(result)
            except (RuntimeError, OSError):
                log.exception("Could not speak task report")

        asyncio.create_task(report())

    def start(self) -> None:
        self.vision.start()
        self._initiative_task = asyncio.create_task(self._initiative())

    async def close(self) -> None:
        self._closed = True
        self.cancel("agent is shutting down")
        if self._initiative_task:
            self._initiative_task.cancel()
            await asyncio.gather(self._initiative_task, return_exceptions=True)
        await self._join()
        await asyncio.to_thread(self.vision.close)

    async def _initiative(self) -> None:
        schedule = set()
        for value in os.getenv("CAR_PATROL_TIMES", "").split(","):
            time_of_day = value.strip()
            if not time_of_day:
                continue
            try:
                schedule.add(datetime.strptime(time_of_day, "%H:%M").strftime("%H:%M"))
            except ValueError:
                log.warning("Ignoring invalid CAR_PATROL_TIMES value %r", value)
        minute_done = None
        greeted = False
        visible_at = absent_at = last_greeting = 0.0
        while True:
            await asyncio.sleep(0.18)
            now, wall = time.monotonic(), datetime.now()
            frame = self.vision.latest()
            if frame and frame.people:
                visible_at = visible_at or now
                absent_at = 0
            elif frame:
                visible_at = 0
                absent_at = absent_at or now
                if now - absent_at >= 1.5:
                    greeted = False
            else:
                visible_at = 0
                absent_at = 0
            if (
                not self.t.agent["initiative"] or self.active or self._gate.locked() or self._blocked()
                or self.t.motion["dir"] or self.t.link["state"] != "online"
                or self.t.sensors["on_ground"] is not True
                or self.t.agent["state"] != "listening" or self.t.agent["user_state"] == "speaking"
            ):
                continue
            slot = wall.strftime("%Y-%m-%d %H:%M")
            greet = visible_at and now - visible_at > 0.35 and not greeted and now - last_greeting > 45
            scheduled = wall.strftime("%H:%M") in schedule and slot != minute_done
            if not (greet or scheduled):
                continue
            try:
                patrol.PATROLS.mkdir(exist_ok=True)
                if greet:
                    with (patrol.PATROLS / "initiative.log").open("a") as history:
                        history.write(f"{wall.astimezone().isoformat()} announcing greeting\n")
                    greeted, last_greeting = True, now
                    self.greet()
                elif frame:
                    minute_done = slot
                    self._launch("scheduled patrol", lambda r: patrol.patrol(r, self.vision, 30), True,
                                 "I'm starting my scheduled thirty second patrol.")
            except OSError:
                log.exception("Initiative suppressed: cannot write its log")
            except RuntimeError:  # greet() needs a running voice session; without one, keep watching
                log.exception("Initiative greeting failed; initiative keeps running")
