"""The learned driver (sim/policy/driver.onnx, trained in sim/isaac/elegoo/train/) behind the Driver interface.

Observation mapping, checked against train/driver_env.py and train/model.py:
  depth      encoded (120, 160) float32 per observation_spec.md: identical encoding and corruptions in macsim.
             Missing frames (None: head turned, first 100 ms) never occur in training; the last frame is
             reused for up to 0.3 s, then an all-far frame is shown, as training does right after a reset.
  aux[0]     range_cm / 150; a missing reading (head turned) reuses the last one, else 150 cm.
  aux[1]     gyro yaw rate deg/s / 250 (the navigator gives rad/s), clipped to +-1 like the saturating gyro.
  aux[2:4]   last sent PWM / 255 per side.
  aux[4:6]   sin, cos of the heading error (CCW positive, same sign as training's target minus gyro yaw).
  aux[6]     distance level 0 / 0.5 / 1 for the distance to the LOOKAHEAD target (< 0.5 m / medium / > 2 m),
             not to the goal: the navigator supplies a training-style target (Planner.line_of_sight).
  aux[7]     speed limit 0..1 (trained on U(0.5, 1)).
  history    the policy's recurrent state (previous 3 depth summaries), kept across steps; reset() clears it.
Model: driver.onnx was trained at 70 ms steps only, so at the navigator's 100 ms each command is held 43 %
longer and the 3-frame history spans 300 ms instead of 210 ms. driver_v2.onnx (V2) was trained at 70 and
100 ms, with a proximity / contact cost and backing-out starts (.omo/waves/p3/retrain.md).
"""

from pathlib import Path

import numpy as np

from sim.policy.run_driver import HERE as POLICY_DIR
from sim.policy.run_driver import Driver as OnnxDriver

from .driver import FAR, MEDIUM, NEAR, HandDriver

LEVEL = {NEAR: 0., MEDIUM: .5, FAR: 1.}
STALE_FRAMES = 3                    # reuse a depth frame for at most 0.3 s
V1, V2 = POLICY_DIR / "driver.onnx", POLICY_DIR / "driver_v2.onnx"


class PolicyDriver:
    def __init__(self, model: Path = V1):
        self.net = OnnxDriver(model)
        self.reset()

    def reset(self) -> None:
        self.net.reset()
        self.depth = np.zeros((120, 160), np.float32)
        self.depth_age = STALE_FRAMES
        self.range_cm = 150.
        self.front = np.inf

    def act(self, depth, range_cm, gyro_rate, last_action, heading_error, how_far, speed_limit):
        if depth is not None:
            self.depth, self.depth_age = depth, 0
        else:
            self.depth_age += 1
            if self.depth_age > STALE_FRAMES:
                self.depth = np.zeros((120, 160), np.float32)
        if range_cm is not None:
            self.range_cm = float(min(range_cm, 150))
        # Read by the navigator's recovery to block the spot ahead when something is close.
        self.front = self.range_cm / 100
        aux = np.array([self.range_cm / 150, np.clip(np.degrees(gyro_rate) / 250, -1, 1),
                        last_action[0] / 255, last_action[1] / 255,
                        np.sin(heading_error), np.cos(heading_error), LEVEL[how_far],
                        float(np.clip(speed_limit, 0, 1))], np.float32)
        return self.net(self.depth, aux)


class VetoedPolicyDriver(PolicyDriver):
    """The policy, overruled by the hand driver when its 1 s arc would come within 2 cm of an obstacle
    in the hand driver's 3 s memory (depth points and range readings, dead-reckoned)."""

    def __init__(self, model: Path = V1):
        self.guard = HandDriver()
        self.steps = self.vetoes = 0
        super().__init__(model)

    def reset(self) -> None:
        super().reset()
        self.guard.reset()

    def act(self, depth, range_cm, gyro_rate, last_action, heading_error, how_far, speed_limit):
        wanted = super().act(depth, range_cm, gyro_rate, last_action, heading_error, how_far, speed_limit)
        fallback = self.guard.act(depth, range_cm, gyro_rate, last_action, heading_error, how_far, speed_limit)
        self.front = self.guard.front
        self.steps += 1
        if self.guard.allows(*wanted):
            return wanted
        self.vetoes += 1
        return fallback
