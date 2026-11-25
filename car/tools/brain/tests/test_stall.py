"""Stall detection with fresh, textured synthetic camera frames."""

from io import BytesIO
from types import SimpleNamespace
from typing import Final

import pytest
from PIL import Image, ImageDraw

import drive
from drive import STALLED
from stall import StallWatch
from vision import Frame


def _striped_jpeg(*, vertical: bool) -> bytes:
    with Image.new("L", (640, 480), 40) as image, BytesIO() as output:
        draw = ImageDraw.Draw(image)
        for offset in range(0, 480 if not vertical else 640, 128):
            box = (offset, 0, offset + 63, 479) if vertical else (0, offset, 639, offset + 63)
            draw.rectangle(box, fill=220)
        image.save(output, format="JPEG")
        return output.getvalue()


def test_stall_watch_distinguishes_unchanged_and_changing_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    still = _striped_jpeg(vertical=True)
    changed = _striped_jpeg(vertical=False)
    step: Final = 0.25  # Five steps span the 1.2 s window, with opposite frame parity.
    now = 100.0
    monkeypatch.setattr(drive, "time", SimpleNamespace(monotonic=lambda: now))

    for frames, expected in (((still,), STALLED), ((still, changed), None)):
        # A fresh frame on every tick; range is steady but gyro supplies no stall vote.
        shown: list[Frame] = []  # the camera always offers the frame of the current tick
        watch = StallWatch(SimpleNamespace(newest=lambda _age, shown=shown: shown[-1]))
        for sample in range(17):
            now += step
            shown.append(Frame(now, frames[sample % len(frames)], ()))
            watch.wheels(110, 110, 0.0, 90)
            watch.ranged(90, 90)

        assert watch.stalled() == expected
