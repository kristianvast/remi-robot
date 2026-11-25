"""Task admission and cancellation at the public manager and wheel boundaries."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace

import anyio
import pytest
from conftest import LOOPBACK, FakeCar

from car import Car
from car_agent import CarAgent
from car_camera import CarCamera
from memory import Memory
from safe_drive import SafeDrive
from tasks import TaskManager
from telemetry import Telemetry
from vision import Frame, Person


@pytest.mark.asyncio
async def test_task_dispatch_reports_active_job_and_stop_cancels_with_zero_wheels(
    fake_car: FakeCar, tmp_path: Path,
) -> None:
    telemetry = Telemetry(25)
    car = Car(LOOPBACK, fake_car.port, telemetry)
    camera = CarCamera(f"{fake_car.camera_url}/capture", f"{fake_car.camera_url}/stream", telemetry)
    jobs = CarAgent(car, camera, Memory(tmp_path)).jobs

    async def speak(_text: str) -> bool:
        return True

    jobs.speak = speak
    label = "wheel regression job"
    reason = "regression-test cancellation"
    moving = Event()

    def roll(drive: SafeDrive) -> str:
        # Keep refreshing motion until cancelled, with a hard limit even if cancel regresses.
        deadline = monotonic() + 5
        while monotonic() < deadline:
            drive.wheels(100, 100)
            moving.set()
            drive.wait(0.1)
        return "completed without cancellation"

    try:
        assert (await jobs.dispatch(label, roll)).startswith("Started")
        assert await anyio.to_thread.run_sync(moving.wait, 2), jobs.status()
        assert (await anyio.to_thread.run_sync(fake_car.state)).wheels == (100, 100)
        assert label in jobs.status() and jobs.status().startswith("Running:")

        with anyio.fail_after(7):
            result = await jobs.stop(reason)

        assert result.startswith("Stopped")
        assert reason in jobs.status()
        assert (await anyio.to_thread.run_sync(fake_car.state)).wheels == (0, 0)
    finally:
        await jobs.close()
        await anyio.to_thread.run_sync(car.close)


@pytest.mark.asyncio
async def test_failed_greeting_leaves_initiative_running(
    fake_car: FakeCar, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Current and next minute, so a minute rollover during the test still matches the schedule.
    now = datetime.now()
    monkeypatch.setenv("CAR_PATROL_TIMES", ",".join((now + timedelta(minutes=i)).strftime("%H:%M") for i in (0, 1)))
    telemetry = Telemetry(25)
    telemetry.agent.update(initiative=False, autonomy=True, state="listening")
    telemetry.link["state"] = "online"
    telemetry.sensors["on_ground"] = True
    car = Car(LOOPBACK, fake_car.port, telemetry)
    person = Frame(monotonic(), b"", (Person(0.0, 0.3, 0.6, False),))
    greetings = ticks = 0
    watched = asyncio.Event()

    def latest() -> Frame:
        # The loop polls once per 0.18 s tick; three ticks put the person in view past the 0.35 s greeting delay.
        nonlocal ticks
        ticks += 1
        if ticks >= 3:
            watched.set()
        return person

    def greet() -> None:
        nonlocal greetings
        greetings += 1
        raise RuntimeError("no running voice session")

    async def speak(_text: str) -> bool:
        return True

    # A camera that always sees one person; start/close are the only other calls TaskManager makes on it.
    vision = SimpleNamespace(latest=latest, start=lambda: None, close=lambda: None)
    jobs = TaskManager(car, vision, speak, greet)
    launched = asyncio.Event()
    labels: list[str] = []

    def launch(label: str, *_args: object) -> None:
        labels.append(label)
        launched.set()

    monkeypatch.setattr(jobs, "_launch", launch)
    jobs.start()
    try:
        await asyncio.wait_for(watched.wait(), 5)
        telemetry.agent["initiative"] = True  # the greeting is due first, the scheduled patrol right after it
        await asyncio.wait_for(launched.wait(), 5)
        assert greetings == 1
        assert labels == ["scheduled patrol"]
    finally:
        await jobs.close()
        await anyio.to_thread.run_sync(car.close)
