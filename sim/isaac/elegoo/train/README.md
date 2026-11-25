# Depth + heading-command driver (Isaac Lab 3.0, rsl-rl-lib 5.5.1 PPO)

The car gets a heading-error command from outside (a navigator, or a person follower) and must turn it
into wheel PWM that makes fast progress without hitting things, using only what the real car has.

## Files
| File | Runs on | What |
|---|---|---|
| `depth.py` | L4 | Spec v1 depth encoding and the 11 corruptions (`sim/realdepth/observation_spec.md`), each switchable (`--ablate`), per-car `stress` |
| `driver_env.py` | L4 | `DriverEnv(BuildingNavEnv)`: routes, navigator command, reward, terminations, metrics |
| `model.py` | anywhere | Input packing, action prior (`wheels`), `DriverModel` export graph, PPO config, `load_actor` |
| `train.py` | L4 | PPO training (`OnPolicyRunner`, one iteration per call, curriculum, `learning.csv`, checkpoints); `--bench N` for throughput |
| `evaluate.py` | L4 | Held-out routes x {nominal, slow 0.75x, fast 1.25x, stress} x {policy, pursuit, reactive}, all cars in one stage |
| `export.py` | L4 (container python, no Kit) | ONNX export + PyTorch CPU replay of the recorded observations |
| `summarize.py` | Mac | `episodes.csv` -> `.omo/waves/p3/driver-eval.csv` |
| `routes.py` / `routes.npz` | Mac | Planned route pools (kept from the previous worker): train 2880, dev 160, test 366 routes; test start cells are never train start cells |
| `inspect_ridge.py` | GPU VM (Kit for pxr) | Height map of the collider around the area_07 start tilts |

## Observation (what the policy may use)
- Depth: 160 x 120 optical-axis z from a Warp ray-cast pinhole camera (Isaac Lab `MultiMeshRayCasterCamera`) on
  the visual scan + generated shell (/World/Scan, /World/Shell: what the RTX camera rendered for v1; ray casting
  needs no RT cores, so it also runs on an A100), corrupted (spec section 4, in order), encoded
  `1 - log(z/0.15)/log(5/0.15)`. Summarised in the graph as max (nearest) and mean over a 4 x 10 grid of
  30 x 16 px cells (80 numbers), plus the previous 3 summaries (frame stack = the policy's recurrent state).
- aux (8): range_cm/150 (cone sonar, now also hits the closure colliders), gyro yaw rate/250 (drifting,
  clipped), last sent PWM/255 per side, sin and cos of the heading error, distance level (0 near < 0.5 m,
  0.5 medium, 1 far > 2 m) and the speed limit 0..1.
- Never used: true pose, true velocity, distance travelled, wheel speeds.

## Command (navigator model)
From the planned route (waypoints every 0.22 m): every 0.5-1.0 s (1-2 Hz) the navigator takes the nearest
waypoint (true position) and aims at the farthest lookahead waypoint 0.44-2.4 m ahead that stays within 10 cm
of a straight line, so it sometimes cuts corners through furniture. The target heading is stored in the gyro
frame, so between updates the heading error follows the drifting gyro. Corruptions: per-episode bias +-3 deg
drifting up to +-5 deg/min (capped 12.6 deg); 3.5 % of updates off by 20-40 deg; in training 6 % of updates
are deliberate shortcuts (7 waypoints ahead regardless of line of sight). Speed limit: U(0.5, 1) in training,
1 in evaluation.

## Action
The network outputs a 2-d residual `(dv, dturn)` on a heading-following prior (`model.wheels`):
`v = clip(0.6 cos+(err)^2 + dv, -0.4, 1)`, `turn = clip(0.6 sin(err) (or 0.6 sign(sin) when behind) + dturn, +-0.8)`,
sides `v -+ turn`, scaled so the larger side is at most `limit`, in units of 150 PWM. So |PWM| <= 150 per side
(the top of the measured / extrapolated drive calibration; 145 PWM spin = 265 deg/s already saturates the gyro).
Reversing was always reachable: v down to -0.4 = -60 PWM per side (~0.19 m/s back, above the 9-17 PWM deadband);
v2 keeps that range and teaches it with facing starts (below) and a reset exploration std (`--init-std`).
Sent PWM is rounded to integers and goes through car-env's delay queue, 14 Hz send rate and 0.4 s watchdog.
One control step = 7 or 10 PhysX ticks of 10 ms (70 or 100 ms), drawn per episode (`--step-ms 70 100`). Cars
share the physics clock: a 70 ms car is held still for the last 3 ticks of a 10-tick step, with its drive state,
packet queue (shifted one slot per held tick) and gyro paused, so it sees exactly a 70 ms step. Latency stays
100 ms: at 100 ms steps the policy always gets the previous frame.

## Car response randomisation (training)
Every `parameters` entry of `calibration.yaml` is drawn per car per process (speed_scale 0.85-1.15,
turn_scale 0.8-1.2, deadband 9-17, lags, command period 60-90 ms, delay 35-90 ms, gyro bias / noise, sonar
cone 25-35 deg and noise...). Camera pitch 17-23 deg, HFOV 60-70 deg, height 0.121-0.131 m and forward offset
0.078-0.086 m are drawn per car per episode (`DriverEnv.aim`: per-env intrinsics + lens offsets of the ray-cast
camera); v1 had one draw per process because the tiled RTX camera shares one projection.

## Reward (per step; `s` = step / 70 ms, so per-time terms cost the same at 70 and 100 ms)
- `+10 x (decrease of the remaining planned-path length)`; remaining = arc length from the waypoint after the
  nearest one (searched forward only) plus the straight distance to it. Potential-based, so it pays for
  progress along the route, not for cutting through furniture.
- `-0.02 s` per step (time), `-0.01` per 30 PWM of change in the sent command (smoothness).
- Proximity (v2): `-w_p s p (0.25 + v / 0.4 m/s)` with `p = clip((0.10 m - gap) / 0.10 m, 0, 1)`, v = speed over
  the step and gap = the smallest of: the bumper gap (72 horizontal rays at 2.5 / 7 / 12 cm from the car origin
  on the collider + closure walls, minus the distance to the car outline from the asset bounds), the cone sonar
  minus the emitter's setback, and the lowest 40 rows of the true depth image minus the lens setback.
- Contact (v2): `-w_c s (1 + v / 0.2 m/s)` while the bumper gap is <= 1 cm. Defaults w_p 0.3, w_c 1.0.
  For scale: progress at 0.3 m/s pays +0.21 per 70 ms step; touching at that speed costs 0.375 + 2.5.
- `+10` on success (within 0.30 m of the goal), `-15` (v1 segment 2 and v2: `--collision-penalty 25`) on collision.
- Metrics: contact onsets (gap crossing to <= 1 cm) per metre, fraction of time inside the 10 cm margin.
- Timeouts (8 s + 5 s per planned metre) are truncations: PPO bootstraps the value (`time_outs`).

## Terminations
- Success: within 0.30 m of the last waypoint and no collision flag.
- Collision (`nav/env.py` privileged flags): `tilted` = |pitch| or |roll| > 6 deg (furniture bases are ragged
  scan slopes; the car climbed to 11 / 10 deg), `stalled` = mean |PWM| >= 55 with < 2.5 cm/s progress for
  0.8 s (pushing against something), `off_map` = centre outside the floor map's free cells (dilated 6 cm).
- Timeout as above.

## Facing starts (v2, `--facing-fraction 0.15`)
After the first step of a train episode, 15 % of cars are moved (<= 0.4 m) and turned to face an obstacle seen
by the bumper rays, with 10-20 cm between it and the front of the car, every hit point >= 5 cm outside the
outline and the centre on the floor map; the route and command stay the same, so the target is usually behind
or beside the car and it must back out or turn without touching (driver.md video F05).

## Excluded routes (`--exclude x0,y0,x1,y1`)
Training routes with any waypoint inside a box are not drawn (used for the area_07 collider ridge,
`inspect_ridge.py`). Test routes are unchanged; summarize.py reports all of them and the subset without them.

## Curriculum
Routes are drawn from the training pool up to a maximum planned length: 3, 5, 8 and then 14 m. The stage
advances when an iteration's success rate is >= 0.7 over >= 100 finished episodes. Stage and iteration are
saved in `curriculum.json` next to the checkpoints, so a resumed segment continues at the same stage.

## Depth corruption stress mode
Per car, `stress` uses the spec's upper magnitudes and forces every random event on: scale 0.75 or 1.33,
upper-wall factor 0.7 or 1.4, two 60 px windows per episode, edge blur sigma 1 px with dilation every frame,
thin-structure loss every frame keeping only 50 % contrast, flattening whenever > 60 % of pixels are nearer than
0.6 m, whole-frame failures always 3 steps long (start probability stays 3 % per step).
Jitter (sd 0.10), shape (0.10), pixel noise (0.06) and latency (100 ms) have one value in the spec and are on.

## Reproduce
```
sim/cloud/push.sh-equivalent: tar sim/isaac/elegoo (without source/ proof/ benchmark/) -> ~/elegoo-sim/work/drv/elegoo,
  and ln -s ../integ/scene ~/elegoo-sim/work/drv/scene (identical md5 to sim/isaac/scene)
sim/cloud/run_headless.sh drv/elegoo/train/train.py --num-envs 1024 --budget-sec 2400 --seed 391 --output /workspace/elegoo/drv/run --viz none
sim/cloud/run_headless.sh drv/elegoo/train/train.py --num-envs 1024 --budget-sec 2400 --seed 392 --resume /workspace/elegoo/drv/run/ppo.pt --output /workspace/elegoo/drv/run --viz none
sim/cloud/run_headless.sh drv/elegoo/train/evaluate.py --checkpoint /workspace/elegoo/drv/run/ppo.pt --output /workspace/elegoo/drv/eval --viz none
container: /opt/isaaclab-venv/bin/python drv/elegoo/train/export.py --checkpoint drv/run/ppo.pt --replay drv/eval/replay.npz --output drv/export
Mac: python3 sim/isaac/elegoo/train/summarize.py episodes.csv .omo/waves/p3/driver-eval.csv
Mac: sim/policy/.venv/bin/python sim/policy/make_videos.py video.npz sim/policy/videos
Mac: sim/policy/.venv/bin/python sim/policy/test_driver.py
```
Run containers with `PYTHONDONTWRITEBYTECODE=1`: root-owned `__pycache__` files otherwise block the next push.
