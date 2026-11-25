"""Share one camera reader between the model and the control pane."""

import asyncio
import io
import logging
import time

from livekit import rtc
from livekit.agents.voice.io import VideoInput
from PIL import Image

from vision import Vision

LIVE_FPS = 1.0  # Gemini Live takes about one picture a second
CAMERA_FPS = 5.0  # refresh rate of the control pane's still frame
SNAPSHOT_MAX_AGE_S = 3.0  # the model never gets a picture older than this; the real camera needs up to 2 s per picture
SNAPSHOT_WAIT_S = 6.0  # how long a model turn waits for the camera before going on blind
SETTLE_S = 0.4  # let the car and the head come to rest before the model's next picture

log = logging.getLogger("car-agent")


class CarCamera:
    """Hands the newest camera still to the model and the control pane.

    The car's camera serves one request at a time, so Vision is its only reader and everything else
    takes frames from it.
    """

    def __init__(self, url, stream_url, telemetry):
        self.vision = Vision(url, stream_url)
        self._telemetry = telemetry
        self._failing = False
        self._task = None
        self._moved_at = 0.0
        self.live = False  # a LiveVideo streams frames to the realtime model
        self._live_kick = asyncio.Event()
        self._live_due = 0.0
        self._live_sent = 0.0  # capture time of the newest frame handed to the realtime model

    def moved(self):
        """The view just changed: only pictures requested from now on may be shown to the model."""
        self._moved_at = time.monotonic()

    async def after_move(self):
        """With live video, hand the model a picture taken after the move before it hears the move's result."""
        if not self.live:
            return
        self._live_kick.set()
        deadline = time.monotonic() + SNAPSHOT_WAIT_S
        while self._live_sent < self._moved_at and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def live_frame(self):
        """Next picture for the realtime model: one a second, or at once after a move."""
        try:
            await asyncio.wait_for(self._live_kick.wait(), max(0.0, self._live_due - time.monotonic()))
        except TimeoutError:
            pass
        self._live_kick.clear()
        self._live_due = time.monotonic() + 1.0 / LIVE_FPS
        frame = await self._fresh()
        if frame:
            self._live_sent = frame.timestamp
        return frame

    async def snapshot(self):
        frame = await self._fresh()
        return frame.jpeg if frame else None

    async def _fresh(self):
        self.vision.start()
        deadline = time.monotonic() + SNAPSHOT_WAIT_S
        while not ((frame := self.vision.newest(SNAPSHOT_MAX_AGE_S)) and frame.timestamp >= self._moved_at):
            if time.monotonic() > deadline:
                if not self._failing:
                    log.warning("no camera frame for %.0f s", SNAPSHOT_WAIT_S)
                self._failing = True
                return None
            await asyncio.sleep(0.05)
        if self._failing:
            log.info("camera back")
            self._failing = False
        self._telemetry.set_frame(frame.jpeg)
        return frame

    def start(self):
        self._task = asyncio.create_task(self._refresh())

    async def _refresh(self):
        while True:
            await self.snapshot()
            await asyncio.sleep(1.0 / CAMERA_FPS)

    def close(self):
        if self._task:
            self._task.cancel()


def _rgba(jpeg):
    with Image.open(io.BytesIO(jpeg)) as image:
        rgba = image.convert("RGBA")
    return rtc.VideoFrame(rgba.width, rgba.height, rtc.VideoBufferType.RGBA, rgba.tobytes())


class LiveVideo(VideoInput):
    """The realtime model's video input, fed from the camera's single reader."""

    def __init__(self, camera):
        super().__init__(label="car-camera")
        self.camera = camera
        camera.live = True

    async def __anext__(self) -> rtc.VideoFrame:
        while not (frame := await self.camera.live_frame()):
            pass  # live_frame already waited for the camera and logged the outage
        return await asyncio.to_thread(_rgba, frame.jpeg)
