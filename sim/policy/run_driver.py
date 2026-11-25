"""CPU-only ELEGOO driver (onnxruntime). No network or robot access.

  sim/policy/.venv/bin/python sim/policy/run_driver.py            # replay check + timing
Inputs per call: depth (120, 160) float32 encoded per sim/realdepth/observation_spec.md and
aux (8,) = (range_cm/150, gyro_rate_deg_s/250, last_left_pwm/255, last_right_pwm/255,
sin(heading error), cos(heading error), distance level 0 / 0.5 / 1 for <0.5 m / medium / >2 m,
speed limit 0..1). Heading error is target heading minus gyro yaw, positive = target to the left.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent


class Driver:
    """Keeps the depth-summary history (the policy's recurrent state); call reset() per new run."""

    def __init__(self, model: Path = HERE / "driver.onnx") -> None:
        config = ort.SessionOptions()
        config.intra_op_num_threads = 1
        config.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), config, providers=["CPUExecutionProvider"])
        self.history_shape = tuple(self.session.get_inputs()[2].shape)
        self.reset()

    def reset(self) -> None:
        self.history = np.zeros(self.history_shape, dtype=np.float32)
        self.last_ms = 0.0

    def __call__(self, depth: NDArray[np.float32], aux: NDArray[np.float32]) -> tuple[int, int]:
        """Return (left PWM, right PWM), signed, |PWM| <= 150."""
        started = perf_counter()
        if depth.shape not in ((120, 160), (1, 120, 160)) or aux.shape != (8,):
            raise ValueError((depth.shape, aux.shape))
        aux = np.asarray(aux, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        if not (np.isfinite(depth).all() and np.isfinite(aux).all() and 0 <= aux[7] <= 1):
            raise ValueError("Expected finite encoded observations and a speed limit in [0, 1]")
        action, self.history = self.session.run(
            ["action", "next_history"],
            {"depth": depth.reshape(1, 1, 120, 160), "aux": aux[None], "history": self.history},
        )
        pwm = np.rint(np.clip(action[0], -1, 1) * 255).astype(np.int16)
        self.last_ms = (perf_counter() - started) * 1000
        return int(pwm[0]), int(pwm[1])


def replay(model: Path, recording: Path) -> tuple[int, float, float, float, int]:
    """Feed recorded observations through ONNX; compare with the PyTorch PWM recorded on the L4."""
    driver = Driver(model)
    with np.load(recording) as data:
        depth, aux, reset, pwm = (data[k] for k in ("depth", "aux", "reset", "pwm"))
    n = len(depth)
    if n == 0 or not reset[0] or aux.shape != (n, 8) or reset.shape != (n,) or pwm.shape != (n, 2):
        raise ValueError("Replay must contain a reset at frame zero and matching depth/aux/reset/pwm rows")
    found, measured = [], []
    started = perf_counter()
    for image, features, episode_reset in zip(depth, aux, reset, strict=True):
        if episode_reset:
            driver.reset()
        found.append(driver(image, features))
        measured.append(driver.last_ms)
    hz = n / (perf_counter() - started)
    error = int(np.abs(np.asarray(found, dtype=np.int16) - pwm).max())
    return n, hz, float(np.median(measured)), float(np.percentile(measured, 95)), error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=HERE / "driver.onnx")
    parser.add_argument("--replay", type=Path, default=HERE / "replay.npz")
    args = parser.parse_args()
    n, hz, median, p95, error = replay(args.model, args.replay)
    print(f"{n} CPU inferences: {hz:.1f}/s; median {median:.3f} ms, p95 {p95:.3f} ms; "
          f"max PyTorch(L4)/ONNX(Mac) difference {error} PWM")
    return 0 if error <= 1 and hz >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
