"""Person following using camera tracks and sonar clearance."""

import logging
import math
import time
from collections import deque
from statistics import median

from safe_drive import SafeDrive, clamp
from vision import Person, Vision

log = logging.getLogger("car-agent")


def follow_person(r: SafeDrive, vision: Vision, target_cm: int) -> str:
    """Track a nearby box by position/size continuity; don't reverse blindly."""
    if not vision.latest():
        return "I could not follow anyone: person vision is unavailable."
    r.head(90)
    start = seen_at = time.monotonic()
    # Do not interpret a frame taken under the old pan angle using the new angle.
    fresh_after = start + 0.10
    last_frame = pan_at = 0.0
    target: Person | None = None
    expected = 0.0
    pan = 90
    speed = 0.0
    tick = 0
    ranges: deque[int] = deque(maxlen=5)
    range_at = 0.0
    while (now := time.monotonic()) - start < 300:
        r.check()
        if why := r.stall.stalled():
            r.wheels(0, 0)
            return f"I stopped following: {why}."
        frame = vision.latest()
        if not frame or frame.timestamp <= max(fresh_after, last_frame):
            if now - seen_at > 0.35:
                r.wheels(0, 0)
                speed = 0
            if now - seen_at > 7:
                return "I lost the person or camera and stopped after a brief search."
            r.wait(0.025)
            continue
        last_frame = frame.timestamp
        candidates = frame.people
        if target is not None:
            candidates = tuple(
                p for p in candidates if abs(p.bearing + pan - 90 - expected) < 33
                and 0.45 <= p.width / target.width <= 2.2
            )
        if not candidates:
            r.wheels(0, 0)
            speed = 0
            ranges.clear()
            lost = now - seen_at
            if lost >= 7:
                return "I lost the person and stopped after a brief search."
            if lost > 0.45 and now - pan_at > 0.62:
                offsets = (0, 22, -22, 42, -42)
                pan = round(clamp(90 + expected + offsets[int(lost / 0.65) % 5], 25, 155))
                r.head(pan)
                pan_at = time.monotonic()
                fresh_after = pan_at + 0.10
            continue
        target = (
            min(candidates, key=lambda p: abs(p.bearing + pan - 90 - expected)
                + 9 * abs(math.log(p.width / target.width)))
            if target else max(candidates, key=lambda p: p.width)
        )
        seen_at = now
        expected = target.bearing + pan - 90
        tick += 1
        # Every range request costs a link round trip, so sample every second decision; the real car makes
        # about 5 a second, so keep 1.5 s of readings.
        if tick % 2 == 1 and abs(target.bearing) <= 12:
            ranges.append(r.distance())
            range_at = time.monotonic()
        if now - range_at > 1.5:
            ranges.clear()
        sonar = median(ranges) if len(ranges) >= 2 else None
        # 38 cm apparent width at 52/60° HFOV; a rough cue, not an absolute safety sensor.
        size_cm = (33 if not vision.fake else 29) / max(0.02, target.width)
        agrees = sonar is not None and sonar < 150 and abs(sonar - size_cm) < max(28, 0.5 * size_cm)
        # The low, level range sensor often sees past a person's legs: then nothing is nearer than the
        # person and the camera's size estimate leads, more slowly.
        clear_past = sonar is None or sonar > size_cm - 28
        desired = 0.0
        if (agrees or clear_past) and not target.clipped and abs(expected) < 14 and abs(pan - 90) <= 14:
            gap = min(sonar, 0.80 * sonar + 0.20 * size_cm) if agrees else size_cm
            error = gap - target_cm
            desired = clamp(52 + 0.85 * error, 0, 102 if agrees else 70) if error > 9 else clamp(6 * error, 0, 58)
        speed += clamp(desired - speed, -29, 17)
        if (sonar is not None and sonar < 35) or (ranges and ranges[-1] < 25) or abs(expected) > 19:
            speed = 0
        if tick % 5 == 0:
            log.info("follow: bearing %.0f width %.2f clipped %s size %.0f cm sonar %s agrees %s pan %d speed %.0f",
                     expected, target.width, target.clipped, size_cm, sonar, agrees, pan, speed)
        steer = math.copysign(clamp(39 + 1.45 * abs(expected), 0, 85), expected) if abs(expected) > 4 else 0
        if ranges and ranges[-1] < 25:
            steer = 0
        r.wheels(speed - steer, speed + steer)
        if now - pan_at > 0.28 and (abs(target.bearing) > 5 or abs(pan - 90) > 5):
            # Keep the head ahead of the wheels, but bleed pan as the chassis catches up.
            new_pan = round(clamp(90 + 0.72 * expected, 30, 150))
            if abs(new_pan - pan) >= 5:
                r.head(new_pan)
                pan = new_pan
                pan_at = time.monotonic()
                fresh_after = pan_at + 0.09
                ranges.clear()
        # Compensate the association prediction for one wheel-response interval.
        expected -= steer * 0.07
    return "I reached the five minute follow limit and stopped."
