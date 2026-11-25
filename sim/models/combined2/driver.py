"""Combined camera model "C", p7 retrain (combined2), behind sim/cmdtest/DRIVER_API.md: DA2-S encoder + driving /
place / goal-direction heads.

    from sim.models.combined2.driver import make_driver
    d = make_driver()                       # VisionDriver: name, reset(seed), act(obs, cmd) -> (left, right)
    d.act(obs, cmd)                          # obs: CarObs, cmd: DriveCommand (attributes or dict keys)
    d.where_am_i()                           # (room id, x m, y m, yaw rad, confidence 0..1) after the last act
    d.goal_direction("kitchen")              # (car-frame direction rad to that place's area, confidence)

Uses only CarObs.rgb / rgb_age_s / range_cm / gyro_rate / gyro_heading / last_action / t and DriveCommand.mode /
heading_error / how_far / speed_limit / target_xy / goal_xy. mode "back" is not learned (a forward camera cannot see
behind it): straight reverse at the hand baseline's BACK PWM, holding the gyro heading the back move started with.
CarObs.depth is ignored: the model computes its own metric depth
from the frame (exposed as `depth`, observation_spec encoding). Runs the two ONNX graphs in weights/ with
onnxruntime; I/O and training are described in README.md and sim/isaac/elegoo/combined/.
"""

from collections.abc import Mapping
from math import atan2, cos, degrees, exp, pi, radians, sin
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name("weights")
AREAS = tuple(f"area_{i:02d}" for i in range(1, 11))
HISTORY, AGE_NONE, IMG_W, IMG_H = 4, 2.0, 224, 168
PWM_UNIT = 150 / 255
STEP_S = 0.1
BACK, STEER, STEER_MAX = 110, 4.0, 60   # sim/cmdtest/baselines.py (drive.py): reverse PWM, PWM per degree, cap


def _field(x, name, default=None):
    return x.get(name, default) if isinstance(x, Mapping) else getattr(x, name, default)


def _resize(rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(rgb).resize((IMG_W, IMG_H), Image.BOX))


def _area_names() -> dict[str, int]:
    names = {a: i for i, a in enumerate(AREAS)}
    places = Path(__file__).resolve().parents[2] / "navigate" / "places.yaml"
    if places.exists():
        import yaml

        for area, spec in yaml.safe_load(places.read_text())["areas"].items():
            names[spec["name"].casefold()] = AREAS.index(area)
    return names


def wheels(residual: np.ndarray, sin_e: float, cos_e: float, limit: float) -> tuple[int, int]:
    """Heading-following prior + learned residual (sim/isaac/elegoo/train/model.py), signed PWM, |PWM| <= 150 limit."""
    v = float(np.clip(0.6 * max(cos_e, 0.0) ** 2 + residual[0], -0.4, 1.0))
    turn = float(np.clip((0.6 * sin_e if cos_e > 0 else 0.6 * np.sign(sin_e)) + residual[1], -0.8, 0.8))
    side = np.array([v - turn, v + turn])
    scale = min(1.0, limit / max(float(np.abs(side).max()), 1e-6))
    left, right = np.round(np.clip(side * scale * PWM_UNIT, -1, 1) * 255).astype(int)
    return int(left), int(right)


def _wrap(a: float) -> float:
    return (a + pi) % (2 * pi) - pi


class CombinedDriver:
    name = "combined2"

    def __init__(self, weights: Path = WEIGHTS, providers: list[str] | None = None, threads: int = 4):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = providers or ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(Path(weights) / "encoder.onnx"), options, providers=providers)
        self.heads = ort.InferenceSession(str(Path(weights) / "heads.onnx"), options, providers=["CPUExecutionProvider"])
        self.frame_dim = self.heads.get_inputs()[0].shape[2]
        self.names = _area_names()
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.history = np.zeros((1, HISTORY, self.frame_dim), np.float32)
        self.ages = np.full((1, HISTORY), AGE_NONE, np.float32)
        self.last_t = None
        self.depth = None
        self.outputs = None
        self.mode = None
        self.last_goal_m = None
        self.hold = None

    def _back(self, obs, cmd) -> tuple[int, int]:
        """Straight reverse with gyro heading hold: the heading to keep is gyro_heading when the back move started
        (a new move = mode change or the goal jumping > 0.3 m further away, as the hand baseline resets)."""
        heading = float(_field(obs, "gyro_heading", 0.0))
        goal_m = float(np.hypot(*_field(cmd, "goal_xy", (0.0, 0.0))))
        if self.hold is None or (self.last_goal_m is not None and goal_m > self.last_goal_m + 0.3):
            self.hold = heading
        self.last_goal_m = goal_m
        drift = _wrap(heading - self.hold)            # CCW + since the start; turn back clockwise
        steer = max(-STEER_MAX, min(STEER_MAX, -STEER * degrees(drift)))
        pwm = BACK * float(_field(cmd, "speed_limit", 1.0))
        return int(round(-pwm - steer)), int(round(-pwm + steer))

    def _see(self, obs) -> None:
        t = _field(obs, "t")
        dt = STEP_S if self.last_t is None or t is None else max(0.0, float(t) - self.last_t)
        self.last_t = None if t is None else float(t)
        present = self.ages < AGE_NONE
        self.ages = np.where(present, np.minimum(self.ages + dt, AGE_NONE - 1e-3), self.ages).astype(np.float32)
        rgb = _field(obs, "rgb")
        if rgb is None:
            return
        image = _resize(np.asarray(rgb, np.uint8)).astype(np.float32).transpose(2, 0, 1)[None]
        depth, frame = self.encoder.run(None, {"image": image})
        self.depth = depth[0]
        self.history = np.concatenate((frame[:, None], self.history[:, :-1]), 1)
        age = float(_field(obs, "rgb_age_s", 0.15) or 0.0)
        self.ages = np.concatenate(([[age]], self.ages[:, :-1]), 1).astype(np.float32)

    def _aux(self, obs, cmd) -> np.ndarray:
        range_cm = _field(obs, "range_cm")
        last = _field(obs, "last_action", (0, 0))
        he = float(_field(cmd, "heading_error", 0.0))
        tx, ty = (float(v) for v in _field(cmd, "target_xy", (0.0, 0.0)))
        gx, gy = (float(v) for v in _field(cmd, "goal_xy", (tx, ty)))
        return np.array([[
            0.0 if range_cm is None else min(max(float(range_cm), 0.0), 150.0) / 150, float(range_cm is not None),
            float(_field(obs, "gyro_rate", 0.0)) / radians(250), last[0] / 255, last[1] / 255, sin(he), cos(he),
            float(_field(cmd, "how_far", 1)) / 2, float(_field(cmd, "speed_limit", 1.0)),
            np.clip(tx, -3, 3) / 3, np.clip(ty, -3, 3) / 3, np.log1p(np.hypot(gx, gy)) / 2,
            sin(atan2(gy, gx)), cos(atan2(gy, gx)), min(float(self.ages[0, 0]), AGE_NONE) / AGE_NONE,
        ]], np.float32)

    def act(self, obs, cmd) -> tuple[int, int]:
        self._see(obs)
        mode = _field(cmd, "mode", "go_to")
        if mode != self.mode:
            self.mode, self.hold, self.last_goal_m = mode, None, None
        aux = self._aux(obs, cmd)
        self.outputs = self.heads.run(None, {"history": self.history, "ages": self.ages, "aux": aux})
        if mode == "back":   # heads still run so where_am_i / goal_direction stay current
            return self._back(obs, cmd)
        return wheels(self.outputs[0][0], float(aux[0, 5]), float(aux[0, 6]), float(aux[0, 8]))

    def where_am_i(self, obs=None) -> tuple[str, float, float, float, float]:
        """Place head after the last act() (or after seeing `obs` with a neutral command if act was not called).

        confidence = room probability x exp(-max(0, sd_xy - 0.25 m) / 0.5 m), sd_xy the head's own position sd."""
        if obs is not None and self.outputs is None:
            self._see(obs)
            self.outputs = self.heads.run(None, {"history": self.history, "ages": self.ages,
                                                 "aux": self._aux(obs, {})})
        if self.outputs is None:
            return AREAS[0], 0.0, 0.0, 0.0, 0.0
        room, pose = self.outputs[1][0], self.outputs[2][0]
        p = np.exp(room - room.max())
        p /= p.sum()
        k = int(p.argmax())
        conf = float(p[k]) * exp(-max(0.0, exp(float(pose[4])) - 0.25) / 0.5)
        return AREAS[k], float(pose[0] * 10), float(pose[1] * 10), atan2(float(pose[2]), float(pose[3])), conf

    def goal_direction(self, place: str) -> tuple[float, float]:
        """Car-frame direction (rad, CCW +) of the drivable way toward a place's area (area id or places.yaml name),
        with the place head's confidence; trained on each area's default goal."""
        if self.outputs is None:
            return 0.0, 0.0
        s, c = self.outputs[3][0, self.names[place.strip().casefold()]]
        return atan2(float(s), float(c)), self.where_am_i()[4]


def make_driver(**kw) -> CombinedDriver:
    return CombinedDriver(**kw)


def benchmark(seconds: float, providers: list[str] | None) -> None:
    """Decisions per second on real 640 x 480 JPEGs (car bench photos + film frames), JPEG decode included."""
    import glob
    import time

    import cv2

    root = Path(__file__).resolve().parents[3]
    paths = sorted(glob.glob(str(root / "patrol-photos/bench/*.jpg"))) + \
        sorted(glob.glob(str(root / "sim/realdepth/data/film640/*.jpg")))[::10]
    jpegs = [Path(p).read_bytes() for p in paths]
    d = make_driver(providers=providers)
    cmd = {"heading_error": 0.2, "how_far": 2, "speed_limit": 1.0, "target_xy": (1.5, 0.3), "goal_xy": (4.0, 1.0)}
    times, i = [], 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t = time.perf_counter()
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpegs[i % len(jpegs)], np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        obs = {"rgb": rgb, "rgb_age_s": 0.15, "range_cm": None if i % 2 else 80.0, "gyro_rate": 0.1,
               "last_action": (60, 70), "t": 0.1 * i}
        d.act(obs, cmd)
        times.append(time.perf_counter() - t)
        i += 1
    ms = np.array(times[5:]) * 1000
    print(f"providers={providers or ['CPUExecutionProvider']} frames={len(ms)} from {len(jpegs)} real JPEGs: "
          f"median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms -> {1000 / np.median(ms):.1f} decisions/s "
          f"(p95 {1000 / np.percentile(ms, 95):.1f}); where_am_i {d.where_am_i()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=float, default=5.0, help="seconds of decisions to time")
    parser.add_argument("--coreml", action="store_true", help="run the encoder with onnxruntime's CoreML provider")
    a = parser.parse_args()
    benchmark(a.bench, ["CoreMLExecutionProvider", "CPUExecutionProvider"] if a.coreml else None)
