"""Loopback-only fixtures for the brain's public interfaces."""

import socket
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from random import SystemRandom
from threading import Event
from time import monotonic
from typing import Final, TypedDict

import httpx2
import pytest
from pydantic import BaseModel

BRAIN: Final = Path(__file__).resolve().parents[1]
LOOPBACK: Final = "127.0.0.1"
sys.path.insert(0, str(BRAIN))

from robot import Robot


class CarState(BaseModel, frozen=True):
    x: float
    y: float
    heading: float
    left: int
    right: int

    @property
    def wheels(self) -> tuple[int, int]:
        return self.left, self.right


class Scene(TypedDict, total=False):
    pose: tuple[float, float, float]
    lifted: bool


@dataclass(frozen=True, slots=True)
class FakeCar:
    port: int
    camera_port: int
    http: httpx2.Client

    @property
    def camera_url(self) -> str:
        return f"http://{LOOPBACK}:{self.camera_port}"

    def state(self, scene: Scene | None = None) -> CarState:
        url = f"{self.camera_url}/test"
        response = self.http.get(url) if scene is None else self.http.post(url, json=scene)
        response.raise_for_status()
        return CarState.model_validate_json(response.content)


@pytest.fixture(autouse=True)
def fake_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAR_FAKE_VISION", "1")


@pytest.fixture
def local_http() -> Iterator[httpx2.Client]:
    # No environment proxies and no redirects: neither may escape loopback.
    with httpx2.Client(timeout=2.0, trust_env=False, follow_redirects=False) as client:
        yield client


def _ports(count: int) -> tuple[int, ...]:
    candidates = list(range(18400, 18500))
    SystemRandom().shuffle(candidates)
    ports: list[int] = []
    for port in candidates:
        with socket.socket() as probe:
            try:
                probe.bind((LOOPBACK, port))
            except OSError:
                continue
        ports.append(port)
        if len(ports) == count:
            return tuple(ports)
    pytest.fail(f"Could not allocate {count} loopback ports in 18400–18499")


@pytest.fixture
def dashboard_port() -> int:
    port, = _ports(1)
    return port


@pytest.fixture
def fake_car(local_http: httpx2.Client, tmp_path: Path) -> Iterator[FakeCar]:
    port, camera_port = _ports(2)
    car = FakeCar(port, camera_port, local_http)
    log = tmp_path / "fake_car.log"
    with log.open("wb") as output, subprocess.Popen(
        [
            sys.executable, "-B", "-u", str(BRAIN / "fake_car.py"),
            "--port", str(port), "--camera-port", str(camera_port), "--test-controls",
        ],
        cwd=BRAIN,
        stdout=output,
        stderr=subprocess.STDOUT,
    ) as process:
        try:
            deadline = monotonic() + 5
            poll = Event()
            while monotonic() < deadline and process.poll() is None:
                try:
                    with socket.create_connection((LOOPBACK, port), timeout=0.2):
                        car.state()
                except (OSError, httpx2.TransportError):
                    poll.wait(0.025)
                else:
                    # Check the child announced these listeners, not another local process.
                    if f"fake car on {LOOPBACK}:{port}," in log.read_text():
                        break
                    poll.wait(0.025)
            else:
                pytest.fail(f"Fake car did not start on {port}/{camera_port}:\n{log.read_text()}")
            yield car
        finally:
            process.kill()
            process.wait(timeout=3)


@pytest.fixture
def robot(fake_car: FakeCar) -> Iterator[Robot]:
    with Robot.wifi(LOOPBACK, fake_car.port) as connection:
        yield connection
