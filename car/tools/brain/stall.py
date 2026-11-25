"""Detect and account for stalls during straight drives."""

import io
import logging
import statistics
from collections import deque

import numpy as np
from PIL import Image

import behaviors
from robot import MAX_RANGE_CM

log = logging.getLogger("car-agent")

# Gyro creep or heading hold at its limit can reveal a stall. Range readings that shrink as
# expected indicate motion; readings that jitter without a trend abstain.
# If only one sign can tell (the gyro mostly cannot while heading hold works), it can signal a
# stall after ONE_SIGN_S, unless the other readable sign says the car moves.
# Values marked GUESS come from the building stand-in (a depth-shaded render, not a real camera) and
# fake_car.py; set them from stall_probe.py recordings on the real car. Measurements: .omo/waves/p3/stall.md.
STALL_MIN_PWM = 80  # straight drives slower than this are not judged (CREEP is 80)
STALL_WINDOW_S = 1.2  # the votes must hold this long
STALL_PAUSE_S = 3.0  # a stop shorter than this (patrol's flank check) continues the same straight drive
ONE_SIGN_S = 2.5  # GUESS: a still picture, or a steady range reading, this long with no sign of motion
STALL_MIN_FRAMES = 5  # fresh frames needed inside the window (the real camera gives about 11 a second)
STALL_FRAME_AGE_S = 0.4  # an older frame cannot supply a picture vote
STILL_SCORE = 1.0  # GUESS: picture change / texture; stand-in stuck 0.03-0.54, moving 3.9-13.8 (5-95 %)
TEXTURE_MIN = 1.0  # GUESS: mean grey step between pixels; under it the view is blank (stand-in moving: 5 % at 1.25)
DARK_MIN = 20  # GUESS: mean grey level; under it the room is too dark to judge
GYRO_CREEP_DEG = 1.5  # GUESS: heading change in the window; under heading hold the stand-in stays within 0.4
RANGE_STILL_CM = 3  # GUESS: spread of the range readings; stand-in noise 0.3 cm, real sensors about 1 cm
RANGE_MIN_READINGS = 3  # range readings needed in the window (forward() reads one every third tick)
STALL_BACK_CM = 15  # recovery: back off this far, then turn away from straight ahead
STALL_AVOID_DEG = 30  # recovery ignores scan directions this close to straight ahead
PICTURE = (80, 60)  # the JPEG is decoded at 1/8 scale, grey
PX_PER_DEG = 80 / 52  # at PICTURE width, 52 degree camera
SHIFT_X, SHIFT_Y = 4, 2  # pixels searched around the gyro-predicted shift


def picture(jpeg):
    """The JPEG as a small grey array, decoded at 1/8 scale so it costs about a millisecond."""
    with Image.open(io.BytesIO(jpeg)) as image:
        image.draft("L", PICTURE)
        grey = image.convert("L")
        if grey.size != PICTURE:
            grey = grey.resize(PICTURE, Image.Resampling.BILINEAR)
        return np.asarray(grey, dtype=np.float32)


def texture(grey):
    """Mean grey-level step between neighbouring pixels: how much a picture could show motion."""
    return float(np.abs(np.diff(grey, axis=1)).mean() + np.abs(np.diff(grey, axis=0)).mean()) / 2


def change(before, after, shift=0.0):
    """Mean grey difference between two pictures, at the best alignment near `shift` pixels to the right.

    The search absorbs turning (the view slides sideways) and bobbing; driving makes things grow and
    spread outwards, which no shift explains. Mean brightness is removed, so exposure changes do not count.
    """
    # The tuning script sets this search radius for its offline score grid.
    shift_x = SHIFT_X
    h, w = before.shape
    centre = round(max(-20.0, min(20.0, shift)))
    mx = abs(centre) + shift_x
    a = before[SHIFT_Y:h - SHIFT_Y, mx:w - mx]
    a = a - a.mean()
    best = float("inf")
    for dy in range(-SHIFT_Y, SHIFT_Y + 1):
        for dx in range(centre - shift_x, centre + shift_x + 1):
            b = after[SHIFT_Y + dy:h - SHIFT_Y + dy, mx + dx:w - mx + dx]
            best = min(best, float(np.abs(a - (b - b.mean())).mean()))
    return best


def distance_at(marks, since):
    """From (time, distance so far) marks, the distance reached by `since`, when the car came to rest.

    The latest mark at or before `since`, not the largest: a running count drops at every earlier stall.
    """
    return next((gone for at, gone in reversed(marks) if at <= since), marks[0][1] if marks else 0.0)


class StallWatch:
    """Notices a straight drive that goes nowhere: two of picture, gyro and range say the car is not moving.

    The robot wrapper that sends the wheel commands feeds it (wheels() after each command, ranged() after
    each range reading); drives read stalled(). While the picture feed has no fresh frame, gyro and range
    still vote. Its votes are logged once a second during straight drives.
    """

    def __init__(self, camera=None):
        self.camera = camera  # anything with newest(max_age) -> Frame | None, such as vision.Vision
        self.events = []  # (found at, still since) per stall, monotonic seconds
        self.signal = {}  # the latest raw values and votes, for logs
        self._logged_at = 0.0
        self._clear()

    def _clear(self):
        self._sign, self._began, self._newest = 0, 0.0, 0.0
        self._pictures = deque()  # (frame timestamp, grey picture, view direction in degrees)
        self._commands = deque()  # (time, heading, steering = (right - left) / 2)
        self._ranges = deque()  # (time, cm) with the head facing ahead
        self._picture_vote, self._picture_since, self._range_since = None, None, None
        self.still_since = None
        self._paused_at = None

    def stalled(self):
        if self.still_since is None:
            return None
        from drive import STALLED

        return STALLED

    def ranged(self, cm, head):
        from drive import time

        if self._sign and abs(head - behaviors.CENTER) <= 15:
            self._ranges.append((time.monotonic(), cm))

    def wheels(self, left, right, heading, head):
        """Judge after a wheel command; True when a stall is newly found. Only straight drives count."""
        # The simulator and tests replace drive.time; timestamps must still come from the same clock.
        from drive import time

        sign = (left > 0 and right > 0) - (left < 0 and right < 0)
        now = time.monotonic()
        if not sign or abs(left + right) / 2 < STALL_MIN_PWM:
            # Patrol stops between short legs to look at both flanks; a short pause keeps the evidence, so
            # pushing that goes on across legs adds up instead of starting from nothing each leg. A reported
            # stall is cleared at once, so the back-off that follows is not stopped by it.
            if self.still_since is not None:
                self._clear()
            elif self._paused_at is None:
                self._paused_at = now
            return False
        if sign != self._sign or (self._paused_at is not None and now - self._paused_at > STALL_PAUSE_S):
            self._clear()
            self._sign, self._began = sign, now
        self._paused_at = None
        if self.still_since is not None or self.camera is None:
            return False
        frame = self.camera.newest(STALL_FRAME_AGE_S)
        if frame is None:  # the picture feed has stalled (it does on the real car): gyro and range still vote
            self._pictures.clear()
            self._picture_vote, self._picture_since = None, None
        self._commands.append((now, heading, (right - left) / 2))
        while len(self._commands) > 1 and self._commands[1][0] <= now - STALL_WINDOW_S:
            self._commands.popleft()
        while self._ranges and self._ranges[0][0] < now - STALL_WINDOW_S:
            self._ranges.popleft()
        if frame is not None and frame.timestamp > self._newest and frame.timestamp >= self._began:
            self._newest = frame.timestamp
            self._look(picture(frame.jpeg), frame.timestamp, heading + head)
        if now - self._began < STALL_WINDOW_S:
            return False
        votes = (self._picture_vote, self._gyro(heading), self._range())
        self.signal["votes"] = votes
        if now - self._logged_at >= 1.0:
            self._logged_at = now
            log.info("stall votes picture/gyro/range %s, range spread %s closed %s, picture %s", votes,
                     self.signal.get("spread"), self.signal.get("closed"), "fresh" if frame else "none")
        if not votes[2]:
            self._range_since = None
        elif self._range_since is None:
            self._range_since = now - STALL_WINDOW_S
        lone = [since for since, other in ((self._picture_since, votes[2]), (self._range_since, votes[0]))
                if since is not None and other is not False and now - since >= ONE_SIGN_S]
        if sum(vote is True for vote in votes) >= 2:
            self.still_since = now - STALL_WINDOW_S
        elif lone:
            self.still_since = min(lone)
        else:
            return False
        self.events.append((now, self.still_since))
        return True

    def _look(self, grey, at, view):
        self._pictures.append((at, grey, view))
        while len(self._pictures) > 1 and self._pictures[1][0] <= at - STALL_WINDOW_S:
            self._pictures.popleft()
        since, before, then = self._pictures[0]
        detail, light = texture(before), float(before.mean())
        score = change(before, grey, (view - then) * PX_PER_DEG) / max(detail, 1e-6)
        readable = (at - since >= STALL_WINDOW_S and len(self._pictures) >= STALL_MIN_FRAMES
                    and light >= DARK_MIN and detail >= TEXTURE_MIN)
        self._picture_vote = score < STILL_SCORE if readable else None
        if not self._picture_vote:
            self._picture_since = None
        elif self._picture_since is None:
            self._picture_since = since
        self.signal = {
            "t": at, "span": at - since, "frames": len(self._pictures), "score": score,
            "texture": detail, "light": light, "turned": view - then,
        }

    def _gyro(self, heading):
        from drive import STEER_MAX

        steering = [steer for _, _, steer in self._commands]
        if any(abs(steer) > STEER_MAX + 1 for steer in steering):
            return None  # deliberate steering: the gyro cannot tell creep from the turn asked for
        crept = abs(heading - self._commands[0][1])
        held = sum(abs(steer) >= STEER_MAX - 1 for steer in steering) / len(steering)
        self.signal.update(crept=crept, held=held)
        return crept >= GYRO_CREEP_DEG or held >= 0.5

    def _range(self):
        readings = [cm for _, cm in self._ranges]
        if len(readings) < RANGE_MIN_READINGS or max(readings) >= MAX_RANGE_CM:
            return None
        half = len(readings) // 2
        closed = self._sign * (statistics.median(readings[:half]) - statistics.median(readings[half:]))
        spread = max(readings) - min(readings)
        self.signal.update(spread=spread, closed=closed)
        if spread <= RANGE_STILL_CM:
            return True
        return False if closed >= RANGE_STILL_CM else None
