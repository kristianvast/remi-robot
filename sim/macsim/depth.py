"""Camera projection and realdepth v1 sensor corruption."""

from math import cos, exp, radians, sin, sqrt, tan

import cv2
import numpy as np

from .world import Pose, cast, geometry

Z_NEAR, Z_FAR = .15, 5.
FX = 125.58
PAN_X, PAN_Z = .0853, .07905
COLS, ROWS = np.meshgrid(np.arange(160, dtype=np.float32) + .5 - 80,
                         60 - (np.arange(120, dtype=np.float32) + .5))


def encode(z):
    """Encode finite or infinite optical metres identically to realdepth."""
    z = np.clip(z, Z_NEAR, Z_FAR)
    return (1.0 - np.log(z / Z_NEAR) / np.log(Z_FAR / Z_NEAR)).astype(np.float32)


def pinhole(pose: Pose, head: float, p: dict[str, float], *, origin=None):
    """Render optical metres and the world height of each finite hit."""
    x, y, yaw = pose
    az = yaw + head
    pitch = radians(p["camera_pitch_deg"])
    fx = FX if p["camera_hfov_deg"] == 65 else 80 / tan(radians(p["camera_hfov_deg"] / 2))
    if origin is None:
        f = p["camera_forward_m"] - PAN_X
        origin = (x + PAN_X*cos(yaw) + f*cos(az),
                  y + PAN_X*sin(yaw) + f*sin(az), p["camera_height_m"])
    right, up = COLS / fx, ROWS / fx
    level = cos(pitch) - sin(pitch) * up
    vertical = sin(pitch) + cos(pitch) * up
    rays = np.stack((level*cos(az) + right*sin(az),
                     level*sin(az) - right*cos(az), vertical), axis=-1)
    z = cast(geometry(), origin, rays.reshape(-1, 3)).reshape(120, 160)
    height = origin[2] + np.where(np.isfinite(z), z, 0) * vertical
    return z, height


def _field(rng: np.random.Generator, sigma: float, sd: float) -> np.ndarray:
    image = rng.standard_normal((120, 160), dtype=np.float32)
    image = cv2.GaussianBlur(image, (0, 0), sigma)
    return (image - image.mean()) * (sd / max(float(image.std()), 1e-6))


class Corruptions:
    def __init__(self, rng: np.random.Generator, scale: float | None, failure: float):
        self.rng = rng
        self.scale = scale if scale is not None else exp(rng.uniform(np.log(.75), np.log(1.33)))
        self.upper = rng.uniform(.7, 1.4)
        self.jitter = 0.
        self.rects = []
        if rng.random() < .3:
            for _ in range(int(rng.integers(1, 3))):
                w, h = rng.integers(20, 61), rng.integers(10, 36)
                x, y = rng.integers(0, 161-w), rng.integers(0, 61-h)
                self.rects.append((slice(y, y+h), slice(x, x+w)))
        # Convert target *frame fraction* into stationary burst-start hazard.
        self.hazard = failure / (2 - failure)
        self.remaining = 0
        self.fail_kind = 0
        self.failed = False

    def apply(self, perfect: np.ndarray, height: np.ndarray) -> np.ndarray:
        rng = self.rng
        self.jitter = .5*self.jitter + rng.normal(0, .087)
        z = perfect * (self.scale * exp(self.jitter))
        shape = np.exp(_field(rng, 20, .10))
        z *= shape
        if np.mean(z < .6) > .60 and rng.random() < .5:
            z = np.median(z) * exp(rng.normal(0, .30)) * shape
        inv = 1 / np.maximum(z, 1e-4)
        inv = cv2.GaussianBlur(inv, (0, 0), float(rng.uniform(.3, 1)))
        if rng.random() < .5:
            inv = cv2.dilate(inv, np.ones((3, 3), np.uint8))
        if rng.random() < .7:
            k = np.ones((2, 2), np.uint8)
            # Complementary anchors avoid translating the entire image.
            inv = cv2.dilate(cv2.erode(inv, k, anchor=(0, 0)), k, anchor=(1, 1))
        z = np.full_like(inv, np.inf)
        np.divide(np.exp(_field(rng, 1.5, .06)), inv, out=z, where=inv > 0)
        finite = np.isfinite(z)
        if self.rects and finite.any():
            far = 1.8 * np.median(z[finite])
            for rect in self.rects:
                region = z[rect]
                applicable = finite[rect] & (height[rect] > .35) & (height[rect] < 2.96)
                region[applicable] = far
        z[height > 1.2] *= self.upper
        if self.remaining == 0 and rng.random() < self.hazard:
            self.remaining = int(rng.integers(1, 4))
            self.fail_kind = int(rng.integers(3))
        self.failed = self.remaining > 0
        if self.failed:
            self.remaining -= 1
            match self.fail_kind:
                case 0:
                    z.fill(np.inf)
                case 1:
                    z.fill(rng.uniform(.3, .8))
                case 2:
                    z = exp(rng.uniform(np.log(.45), np.log(4))) * np.exp(_field(rng, 22, .45))
        return encode(np.nan_to_num(z, nan=Z_FAR, posinf=Z_FAR))


def sonar(pose: Pose, head: float, p: dict[str, float], rng: np.random.Generator) -> int:
    """Nearest unit ray in a 3-D cone, plus calibrated centimetre noise."""
    x, y, yaw = pose
    a = yaw + head
    ca, sa = cos(a), sin(a)
    origin = (x + PAN_X*cos(yaw) + .022*ca, y + PAN_X*sin(yaw) + .022*sa, .116)
    # Equal-area rings with one central ray: 1 + 8 + 16 + 24 + 32 + 48.
    rays = [(ca, sa, 0.)]
    half = radians(p["ultrasonic_full_cone_deg"] / 2)
    for i, n in enumerate((8, 16, 24, 32, 48), 1):
        tilt = half * sqrt(i / 5)
        theta = np.arange(n) * (2*np.pi/n) + i*.2
        sideways = sin(tilt) * np.cos(theta)
        rays.extend(zip(cos(tilt)*ca - sideways*sa, cos(tilt)*sa + sideways*ca,
                        sin(tilt)*np.sin(theta), strict=True))
    dist = float(cast(geometry(False), origin, np.asarray(rays)).min())
    cm = min(dist * 100, p["ultrasonic_cap_cm"])
    return round(np.clip(cm + rng.normal(0, p["ultrasonic_range_noise_cm"]), 2, p["ultrasonic_cap_cm"]))
