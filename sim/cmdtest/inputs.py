"""Rendered sensors -> what the real car's Mac side receives (one instance per trial).

Per episode: JPEG quality 60-90, frame drop probability, exposure time (motion blur), sonar dropout, which
step parity reads the range sensor. Per step:
  rgb    the frame captured `lag` control steps ago (lag 1 or 2 = 0.1 / 0.2 s, drawn from the 0.10-0.25 s latency
         range: P(lag 2) = 2/3, sticky with p 0.9 per step), motion-blurred by the true yaw rate over the exposure,
         JPEG encoded and decoded; None when dropped or when that frame was already delivered.
  range  read on every second step (navigation.Sensors' schedule), not on the step after a head move, and dropped
         with the episode's probability: None on > 50 % of steps.
  gyro   rate and integrated heading from the env (bias, noise, drift and clip are applied there).
Depth is filled in by the caller from the delivered rgb with the real estimator. The frame delivered at step s was
captured at s-1 or s-2, so observe() at step s-1 already prepares it (`upcoming`), which lets the caller run the
estimator on it while the simulator computes step s.
"""

from collections import deque
from math import radians, tan

import cv2
import numpy as np

from .api import CarObs

DT = 0.1


class RealInputs:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.quality = int(rng.integers(60, 91))
        self.drop_p = float(rng.uniform(.07, .10))
        self.exposure_s = float(rng.uniform(.008, .03))
        self.sonar_drop_p = float(rng.uniform(0., .15))
        self.parity = int(rng.integers(2))
        self.lag = 2 if rng.random() < 2 / 3 else 1
        self.frames: deque = deque(maxlen=4)
        self.step = 0
        self.delivered = -1
        self.upcoming = None
        self.counts = dict(steps=0, rgb_none=0, range_none=0, age_sum=0.0, blur_px_max=0.0)

    def draws(self) -> dict:
        return dict(jpeg_quality=self.quality, drop_p=round(self.drop_p, 4), exposure_s=round(self.exposure_s, 4),
                    sonar_drop_p=round(self.sonar_drop_p, 4))

    def _deliver(self, step, hfov_deg):
        if self.rng.random() < .1:
            self.lag = 2 if self.rng.random() < 2 / 3 else 1
        wanted = step - self.lag
        frame = next((f for f in self.frames if f[0] == wanted), None)
        if frame is None or wanted <= self.delivered:
            return None
        self.delivered = wanted
        if self.rng.random() < self.drop_p:
            return None
        _, captured, image, yaw_rate = frame
        fx = 320 / tan(radians(hfov_deg) / 2)
        blur = abs(yaw_rate) * self.exposure_s * fx
        self.counts["blur_px_max"] = max(self.counts["blur_px_max"], blur)
        if blur >= 1.5:
            image = cv2.blur(image, (int(round(blur)), 1))
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        rgb = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return captured, rgb, jpeg.tobytes()

    def observe(self, now, frame, true_yaw_rate, hfov_deg, range_cm, gyro_rate, gyro_heading, last_action,
                head_moved):
        """Returns (CarObs without depth, jpeg or None, capture time or None); self.upcoming is then next step's."""
        delivered = self.upcoming
        if frame is not None:
            self.frames.append((self.step, now, frame, true_yaw_rate))
        self.upcoming = self._deliver(self.step + 1, hfov_deg)
        read = (self.step + self.parity) % 2 == 0 and not head_moved and self.rng.random() >= self.sonar_drop_p
        self.step += 1
        c = self.counts
        c["steps"] += 1
        c["rgb_none"] += delivered is None
        c["range_none"] += not read
        if delivered is None:
            rgb, jpeg, captured, age = None, None, None, 0.0
        else:
            captured, rgb, jpeg = delivered
            age = now - captured
            c["age_sum"] += age
        obs = CarObs(rgb=rgb, rgb_age_s=float(age), depth=None, range_cm=float(range_cm) if read else None,
                     gyro_rate=float(gyro_rate), gyro_heading=float(gyro_heading),
                     last_action=tuple(int(v) for v in last_action), t=float(now))
        return obs, jpeg, captured
