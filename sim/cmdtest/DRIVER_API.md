# Vision driver contract (p4: NavDP fine-tune "A" and combined model "C")

Both new models plug into the car through this one interface, and the command test harness
(`sim/cmdtest/`) drives every controller through it. The point of the contract: a driver gets
EXACTLY what the real Mac side has every 100 ms step on the real car, nothing more. No true pose,
no true velocity, no clean depth, no map distances, no wheel odometry (the car has no encoders).

## Python interface (runs on the Mac CPU / MPS and inside the GPU VM harness)

```python
class VisionDriver(Protocol):
    name: str
    def reset(self, seed: int | None = None) -> None: ...
    def act(self, obs: CarObs, cmd: DriveCommand) -> tuple[int, int]: ...   # signed PWM (left, right), |pwm| <= 150
```

Each driver lives in its own package and exposes `make_driver(**kw) -> VisionDriver`:
`sim/models/navdp/driver.py` (A), `sim/models/combined/driver.py` (C). The existing drivers are
wrapped by the harness: `hand`, `policy_v2_veto` (sim/navigate/driver.py, policy_driver.py).

## CarObs: one control step, as the real car delivers it

| Field | Type | Real-car source | What the harness must simulate |
|---|---|---|---|
| `rgb` | uint8 (480, 640, 3) or None | ESP32 JPEG stream, ~11 fps, decoded | render from the car camera (randomised mount: pitch 17-23 deg, HFOV 60-70, height 0.121-0.131 m, forward 0.078-0.086 m), JPEG round trip (quality 60-90), motion blur, exposure/white-balance/noise from `appearance.py`; None on a dropped frame (real: depth/frames missing on 11-14 % of steps) |
| `rgb_age_s` | float | time since the frame was captured | 0.10-0.25 s latency per frame (Wi-Fi + decode) |
| `depth` | float32 (120, 160) or None | `sim/realdepth/realdepth.py` `observe(rgb)` = Depth Pro 2-S on the frame, encoded per `observation_spec.md` | run the SAME estimator on the rendered, JPEG-degraded frame (Depth Pro 2-S, CUDA on the VM); never the simulator's clean depth. None when `rgb` is None |
| `range_cm` | float or None | ultrasonic, cone 25-35 deg, cap 150 | ray-cast cone with noise; None on > 50 % of steps (measured on the real car) |
| `gyro_rate` | float rad/s, CCW + | MPU6050 | true yaw rate + bias + noise + +-250 deg/s clip (calibration.yaml) |
| `gyro_heading` | float rad | integrated gyro | integral of the noisy rate (drifts) |
| `last_action` | (int, int) | last PWM sent | as sent |
| `t` | float s | wall clock | sim time |

A driver may ignore `depth` and estimate its own from `rgb`, but then the estimator runs inside the
driver and counts toward its Mac speed budget (>= 10 decisions/s including vision on an M5 Pro).

## DriveCommand: what the tool layer hands the driver each step

| Field | Type | Meaning |
|---|---|---|
| `kind` | "goal" | always a point goal; spins (`turn`), `look_direction` and `stop` stay in the existing gyro code for every driver |
| `goal_xy` | (forward m, left m) | the final goal of this command in the car frame, from the best REAL estimate: the localiser estimate during `go_to`, dead reckoning (gyro + commanded-speed model, `sim/navigate/driver.py: motion`) from the tool call's start pose for `drive_forward` / `drive_back` |
| `target_xy` | (forward m, left m) | the planner's lookahead waypoint (0.44-2.4 m ahead on the planned route, line of sight), same estimate; equals `goal_xy` when there is no route |
| `heading_error` | rad | direction of `target_xy`, CCW + (what policy_v2 consumes) |
| `how_far` | 0 / 1 / 2 | NEAR < 0.5 m, MEDIUM, FAR > 2 m to `target_xy` |
| `speed_limit` | 0..1 | 1 unless the tool asks for slow |
| `mode` | "go_to" / "forward" / "back" / "explore" / "patrol" | which tool is running; explore / patrol give a goal 1.5 m ahead along a random free heading the tool layer chooses from `depth`, like today |

Only the harness's tool layer may read simulator truth, and only for scoring.

## Units and frames
Car frame: x forward, y left, metres; angles CCW positive. PWM per side, left first,
positive = forward, |pwm| <= 150, deadband ~9-17 (calibration.yaml). One control step = 100 ms.
