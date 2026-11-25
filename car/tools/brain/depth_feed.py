"""Fetch fresh depth observations off the navigation control thread.

The real feed imports sim/realdepth lazily after navigation sets up the simulator import path.
Test only: CAR_TEST_DEPTH_URL, a 127.0.0.1 URL of the stand-in's /depth (sim/macsim/standin.py
--test-depth), replaces realdepth with the simulator's own corrupted depth observation.
"""

import io
import logging
import os
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np

log = logging.getLogger("car-agent")

TEST_DEPTH_URL = os.getenv("CAR_TEST_DEPTH_URL") or None
if TEST_DEPTH_URL and urlsplit(TEST_DEPTH_URL).hostname not in ("127.0.0.1", "::1"):
    raise SystemExit("CAR_TEST_DEPTH_URL is for the local stand-in only (127.0.0.1)")
DEPTH_MAX_AGE_S = 0.5  # an older depth frame is not passed to the navigator
FIRST_DEPTH_S = 30.0  # go_to waits this long for the first depth frame (the model loads on the first trip)


def _decode(jpeg: bytes) -> np.ndarray:
    import cv2

    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("camera frame is not a JPEG")
    if bgr.shape[:2] != (480, 640):  # realdepth's camera model is the car's 640x480 frame
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DepthFeed:
    """The newest depth observation, made on its own thread so the control loop never waits for it."""

    def __init__(self, vision):
        self.vision = vision
        self.problem: str | None = None
        self.ms: list[float] = []  # time per observation (decode + model), real path only
        self._latest: tuple[float, np.ndarray, np.ndarray | None] | None = None  # (captured, depth, rgb)
        self._taken = 0.0
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="nav-depth")
        self._thread.start()

    def _run(self) -> None:
        try:
            from navigation import DT

            if TEST_DEPTH_URL:
                self._test(TEST_DEPTH_URL, DT)
            else:
                self._real()
        except Exception as e:  # any failure (model, import, link) means no depth; go_to reports it
            log.exception("Depth feed failed")
            self.problem = f"my depth camera pipeline failed ({e})"

    def _real(self) -> None:
        from sim.realdepth import (
            realdepth,  # torch + Depth Pro 2: Mac only, loaded on the first trip
        )

        last = 0.0
        while not self._closed.is_set():
            frame = self.vision.newest(DEPTH_MAX_AGE_S)
            if frame is None or frame.timestamp <= last:
                self._closed.wait(0.01)
                continue
            last = frame.timestamp
            began = time.perf_counter()
            rgb = _decode(frame.jpeg)
            obs = realdepth.observe(rgb)
            self.ms.append((time.perf_counter() - began) * 1000)
            self._latest = (frame.timestamp, obs, rgb)

    def _test(self, url: str, dt: float) -> None:
        while not self._closed.is_set():
            began = time.monotonic()
            with urlopen(url, timeout=2) as response:
                body = response.read()
            if body:
                self._latest = (began, np.load(io.BytesIO(body)), None)
            self._closed.wait(max(0.0, dt - (time.monotonic() - began)))

    def ready(self) -> bool:
        return self._latest is not None

    def take(self) -> tuple[float, np.ndarray, np.ndarray | None] | None:
        """(captured, depth, rgb) not handed out before and younger than DEPTH_MAX_AGE_S, else None (never a replay).
        rgb is the frame the depth came from, None on the stand-in's test depth."""
        latest = self._latest
        if latest and latest[0] > self._taken and time.monotonic() - latest[0] < DEPTH_MAX_AGE_S:
            self._taken = latest[0]
            return latest
        return None

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=1.0)
