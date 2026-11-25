"""Depth-guided trips to named places and home calibration."""

import logging
import time

import numpy as np

import depth_feed
import navigation
import places
from safe_drive import SafeDrive
from vision import Vision

log = logging.getLogger("car-agent")


def _first_depth(r: SafeDrive, feed: depth_feed.DepthFeed) -> str | None:
    """Wait for the first depth picture (the Mac loads the depth model on the first trip); the problem, or None."""
    until = time.monotonic() + depth_feed.FIRST_DEPTH_S
    while not feed.ready():
        if feed.problem:
            return feed.problem
        if time.monotonic() > until:
            return "no depth picture from my camera"
        r.wait(0.05)
    return None


def go_to(r: SafeDrive, vision: Vision, trip: navigation.Trip, where: navigation.Whereabouts) -> str:
    """Drive to a named place on the navigator, 10 steps a second, and say honestly how it ended."""
    nav = trip.navigator
    feed = depth_feed.DepthFeed(vision)
    stalls, seen = 0, len(r.stall.events)
    stamps: list[float] = []
    compute_ms: list[float] = []
    send_ms: list[float] = []
    late = 0
    status = None
    r.navigating = True
    try:
        nav.driver.reset()  # a camera driver loads its model here, while the depth model loads on its thread
        if problem := _first_depth(r, feed):
            return f"I did not start {trip.doing}: {problem}."
        sensors = navigation.Sensors(r, feed)
        due = time.monotonic()
        while True:
            stamps.append(time.monotonic())
            obs = sensors.observe()
            left, right, head, status = nav.step(obs)
            if len(stamps) % 10 == 1:  # once a second: what the navigator believes and what it got
                log.info("nav: %s | depth %s range %s map_fit %.2f cmd %d,%d stalls %d votes %s", status.line(),
                         "yes" if obs.depth is not None else "NO", obs.range_cm, status.map_fit, left, right, stalls,
                         r.stall.signal.get("votes"))
            computed = time.monotonic()
            compute_ms.append((computed - stamps[-1]) * 1000)
            where.update(status.pose, status.confidence, trip.doing)
            if status.state in ("arrived", "failed"):
                break
            sensors.send(left, right, head)
            send_ms.append((time.monotonic() - computed) * 1000)
            if len(r.stall.events) > seen:
                seen = len(r.stall.events)
                stalls += 1
                # The sensors miss what the car pushes against: recover now instead of after 8 s without progress.
                nav.blocked_ahead()
            due += navigation.DT
            if (pause := due - time.monotonic()) > 0:
                r.wait(pause)
            else:
                late += 1
                if pause < -2 * navigation.DT:  # far behind: start counting afresh rather than rush
                    due = time.monotonic()
    finally:
        feed.close()
        r.navigating = False
        where.settle(status is not None and status.state == "arrived")
        periods = np.diff(stamps) if len(stamps) > 1 else np.zeros(1)
        where.last_trip = {
            "state": status.state if status else "not started", "reason": status.reason if status else "",
            "nav_s": status.t if status else 0.0, "stalls": stalls, "recoveries": nav.recoveries, "late_steps": late,
            "steps": len(stamps), "period_median_ms": float(np.median(periods) * 1000),
            "period_mean_ms": float(np.mean(periods) * 1000), "period_p95_ms": float(np.percentile(periods, 95) * 1000),
            "compute_median_ms": float(np.median(compute_ms or [0])), "send_median_ms": float(np.median(send_ms or [0])),
            "depth_ms": list(feed.ms), "start": trip.start,
        }
    pushed = f" {stalls} times I was pushing against something and backed off." if stalls else ""
    if status.state == "arrived":
        return (f"I think I'm at {trip.there} now: my position estimate is within 30 cm of it, at "
                f"{status.confidence:.0%} confidence, after {status.t:.0f} seconds.{pushed}")
    why = {"lost": "I lost track of where I am", "blocked": "I couldn't find a way through",
           "timeout": f"I ran out of time after {status.t:.0f} seconds"}[status.reason]
    room = places.room_name(status.pose)
    guess = (f"I think I'm in {room}" if status.confidence >= navigation.TRUST
             else f"My best guess is {room}, but I'm not sure")
    return f"I stopped without reaching {trip.there}: {why}. {guess}.{pushed}"


def set_home(r: SafeDrive, vision: Vision, room: str, area: str, where: navigation.Whereabouts) -> str:
    """Stand and look around until the localiser knows the spot in `area`, then save it as home."""
    feed = depth_feed.DepthFeed(vision)
    found = None
    try:
        if problem := _first_depth(r, feed):
            return f"I did not save home: {problem}."
        found = navigation.locate(r, feed, area, where)
    finally:
        feed.close()
        where.settle(found is not None)
    if found is None:
        return f"I couldn't work out exactly where I am in the {room}, so home is unchanged."
    h = places.save_home(found.pose)
    return (f"Saved: this spot is home now, in {places.room_name(found.pose)} "
            f"(about {h.x:.1f}, {h.y:.1f} on my map, confidence {found.confidence:.0%}).")
