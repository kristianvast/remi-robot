"""Autonomous patrol routes with clearance checks and photo capture."""

import logging
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Thread

from drive import MOVE_LIMIT, STALLED, Driver
from safe_drive import EndTask, SafeDrive
from stall import STALL_BACK_CM
from vision import Vision

log = logging.getLogger("car-agent")
PATROLS = Path(__file__).with_name("patrols")


def open_heading(r: SafeDrive, angles: tuple[int, ...] = (90, 40, 140, 15, 165)) -> tuple[int, float]:
    readings = []
    for angle in angles:
        r.head(angle)
        r.wait(0.18)
        readings.append((angle - 90, r.range_vote()))
    r.head(90)
    return max(readings, key=lambda item: item[1])


def patrol(r: SafeDrive, vision: Vision, seconds: int) -> str:
    """Roam like Driver.explore, with an independent sampler; it backs up only off something it pushed against."""
    folder = PATROLS / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True)
    begin = time.monotonic()
    turns, saved, stalls = 0, 0, 0
    sightings: list[str] = []
    seen = Event()

    def capture() -> None:
        nonlocal saved
        last_frame = last_person = 0.0
        next_picture = begin
        present = False
        while not seen.wait(0.07):
            frame = vision.latest()
            if not frame or frame.timestamp <= last_frame:
                continue
            last_frame = frame.timestamp
            now = time.monotonic()
            if frame.people:
                if not present:
                    sightings.append(f"{len(frame.people)} at {datetime.now():%H:%M:%S}")
                last_person = now
            arrival = bool(frame.people) and not present
            present = bool(frame.people) or (present and now - last_person < 1)
            if arrival or now >= next_picture:
                try:
                    (folder / f"{datetime.now():%H%M%S-%f}.jpg").write_bytes(frame.jpeg)
                    saved += 1
                except OSError:
                    log.exception("Could not save patrol photo")
                    return
                if now >= next_picture:
                    next_picture = now + 5

    sampler = Thread(target=capture, name="patrol-camera", daemon=True)
    sampler.start()
    deadline = begin + seconds
    original_reason = r.reason
    r.vote_while_driving = True
    r.reason = lambda: original_reason() or ("patrol time is up" if time.monotonic() >= deadline else None)
    driver = Driver(r, stall=r.stall)
    try:
        while True:
            r.check()
            # A shallow wall can intersect the chassis while the forward beam still reads
            # clear. Stop between short legs and vote on both forward flanks before continuing.
            _, why = driver.forward(140, 35, max_s=min(2.8, deadline - time.monotonic()))
            if why == STALLED:
                # The beam missed what the car is pushing against, so straight ahead is not an option.
                stalls += 1
                driver.back(STALL_BACK_CM)
                angle, room = open_heading(r, (40, 140, 15, 165))
                turn = angle if room > 50 else 160
            elif why and why != MOVE_LIMIT:
                angle, room = open_heading(r)
                turn = angle if room > 50 and abs(angle) > 10 else 160
            else:
                flanks = []
                for pan in (25, 155):
                    r.head(pan)
                    r.wait(0.18)
                    flanks.append(r.range_vote())
                r.head(90)
                if min(flanks) >= 28:
                    continue
                turn = 65 if flanks[1] > flanks[0] else -65
                if max(flanks) < 40:
                    turn = 160 if turn > 0 else -160
            turned, why = driver.turn(turn)
            turns += abs(turned) > 10
            if why:
                break
    except EndTask as e:
        why = str(e)
    finally:
        r.vote_while_driving = False
        seen.set()
        sampler.join()
        r.reason = original_reason
    people = f"People seen: {', '.join(sightings)}." if sightings else "No people detected."
    ending = f" Ended early: {why}." if why and why != "patrol time is up" else ""
    pushed = f" {stalls} times I was pushing against something and backed off." if stalls else ""
    return (
        f"Patrol: {time.monotonic() - begin:.0f} seconds, about {r.distance_driven:.0f} cm, "
        f"{turns} direction changes. {people} {saved} pictures in patrols/{folder.name}/.{pushed}{ending}"
    )
