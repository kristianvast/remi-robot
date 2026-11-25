"""Independent local JPEG/person loop; YOLOX-Nano COCO ONNX, CPU, Apache-2.0."""

import logging
import io
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from PIL import Image

MODEL = Path(sys.prefix) / "share/elegoo/yolox_nano.onnx"
SIZE = 416
MAX_JPEG = 2_000_000
STREAM_RETRY_S = 5.0  # after a stream failure, read stills this long before trying the stream again
log = logging.getLogger("car-agent")


@dataclass(frozen=True, slots=True)
class Person:
    bearing: float  # camera-relative degrees, positive left
    width: float  # fraction of image width
    height: float
    clipped: bool


@dataclass(frozen=True, slots=True)
class Frame:
    timestamp: float  # capture request start, monotonic
    jpeg: bytes
    people: tuple[Person, ...]


def person_box(box: tuple[float, float, float, float], hfov: float) -> Person:
    x0, y0, x1, y1 = box
    bearing = math.degrees(math.atan((1 - x0 - x1) * math.tan(math.radians(hfov / 2))))
    return Person(bearing, x1 - x0, y1 - y0, x0 < 0.015 or x1 > 0.985)


class Detector:
    def __init__(self, model: Path = MODEL):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        grids, strides = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[: SIZE // stride, : SIZE // stride]
            grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
            strides.append(np.full((x.size, 1), stride))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def detect(self, jpeg: bytes) -> tuple[Person, ...]:
        """Return visible person boxes projected into camera bearings."""
        np = self.np
        with Image.open(io.BytesIO(jpeg)) as image:
            w, h = image.size
            ratio = SIZE / max(h, w)
            resized = np.asarray(image.convert("RGB").resize((int(w * ratio), int(h * ratio)), Image.Resampling.BILINEAR))
        padded = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized[:, :, ::-1]  # YOLOX expects BGR
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        prediction = self.session.run(None, {self.input: blob})[0][0]
        scores = prediction[:, 4] * prediction[:, 5]  # first COCO class is person
        keep = scores >= 0.30
        predictions = prediction[keep]
        centers = (predictions[:, :2] + self.grid[keep]) * self.stride[keep]
        sizes = np.exp(predictions[:, 2:4]) * self.stride[keep]
        low, high = centers - sizes / 2, centers + sizes / 2
        areas = sizes.prod(axis=1)
        order = np.argsort(-scores[keep])
        people = []
        while order.size:
            i, rest = order[0], order[1:]
            overlap = np.maximum(0, np.minimum(high[i], high[rest]) - np.maximum(low[i], low[rest])).prod(axis=1)
            order = rest[overlap / (areas[i] + areas[rest] - overlap) <= 0.45]
            x, y = low[i]
            right, bottom = high[i]
            x0, y0, x1, y1 = np.clip(
                (x / (w * ratio), y / (h * ratio), right / (w * ratio), bottom / (h * ratio)), 0, 1
            )
            if x1 > x0 and y1 > y0:
                people.append(person_box((x0, y0, x1, y1), 52))
        return tuple(people)


def camera_mirrored(url: str | None) -> bool:
    """The real camera's picture is mirrored; the local stand-in camera's is not. CAR_CAMERA_MIRROR=0/1 overrides."""
    loopback = bool(url) and urlsplit(url).hostname in ("127.0.0.1", "::1")
    return os.getenv("CAR_CAMERA_MIRROR", "0" if loopback else "1") == "1"


def mirror_jpeg(jpeg: bytes) -> bytes:
    """Flip a JPEG left-right; the car's ESP32 camera delivers a mirrored image and has no hmirror control."""
    with Image.open(io.BytesIO(jpeg)) as image:
        out = io.BytesIO()
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(out, format="JPEG", quality=90)
    return out.getvalue()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Vision:
    """Publish only the latest frame; never queue images or read the control link."""

    def __init__(self, camera_url: str, stream_url: str | None = None):
        self.url = camera_url
        self.stream_url = stream_url
        self.fake = (
            os.getenv("CAR_FAKE_VISION") == "1"
            and urlsplit(camera_url).hostname in ("127.0.0.1", "::1")
        )
        self.mirror = camera_mirrored(camera_url)
        self._frame: Frame | None = None
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="car-vision")
            self._thread.start()

    def latest(self) -> Frame | None:
        return self.newest(0.55)

    def newest(self, max_age: float) -> Frame | None:
        frame = self._frame
        return frame if frame and time.monotonic() - frame.timestamp < max_age else None

    def close(self) -> None:
        self._closed.set()
        if self._thread:
            self._thread.join(timeout=1.2)

    def _run(self) -> None:
        installed = MODEL.is_file()
        if not self.fake and not installed:
            log.error("Person detector not installed: %s (the download command is at the top of agent.py)", MODEL)
        detector = Detector() if installed and not self.fake else None
        opener = build_opener(_NoRedirect)
        failing = False
        stream_after = 0.0
        while not self._closed.is_set():
            # The stand-in's person fixture rides on the still's headers, so fake vision reads stills.
            streaming = bool(self.stream_url) and not self.fake and time.monotonic() >= stream_after
            try:
                for started, jpeg, fixture in (self._stream if streaming else self._still)(opener):
                    if not jpeg or len(jpeg) > MAX_JPEG:
                        raise ValueError("unexpected JPEG size")
                    if self.mirror:
                        jpeg = mirror_jpeg(jpeg)
                    if detector:
                        people = detector.detect(jpeg)
                    elif fixture:
                        x0, y0, x1, y1 = map(float, fixture.split(","))
                        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                            raise ValueError("invalid stand-in box")
                        people = (person_box((x0, y0, x1, y1), 60),)
                    else:
                        people = ()
                    self._frame = Frame(started, jpeg, people)
                    failing = False
            except (ValueError, OSError) as e:
                self._frame = None
                if not failing:
                    log.warning("Person camera unavailable: %s", e)
                failing = True
                if streaming:
                    stream_after = time.monotonic() + STREAM_RETRY_S
                self._closed.wait(0.5)

    def _still(self, opener):
        started = time.monotonic()
        with opener.open(self.url, timeout=3) as response:
            jpeg = response.read(MAX_JPEG + 1)
            fixture = response.headers.get("X-Fake-Person", "")
        yield started, jpeg, fixture
        self._closed.wait(max(0, 1 / 12 - (time.monotonic() - started)))

    def _stream(self, opener):
        """Yield JPEGs from the MJPEG stream: one connection, so no per-picture setup to stall on."""
        with opener.open(self.stream_url, timeout=3) as response:
            buffer, started = b"", time.monotonic()
            while not self._closed.is_set():
                chunk = response.read1(65536)
                if not chunk:
                    raise OSError("camera stream ended")
                buffer += chunk
                if len(buffer) > 2 * MAX_JPEG:
                    raise ValueError("no JPEG in the camera stream")
                while (start := buffer.find(b"\xff\xd8")) >= 0 and (end := buffer.find(b"\xff\xd9", start + 2)) >= 0:
                    yield started, buffer[start:end + 2], ""
                    # The camera grabs the next picture only after this one is sent.
                    buffer, started = buffer[end + 2:], time.monotonic()
