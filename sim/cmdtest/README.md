# cmdtest: Gemini's real tool calls, replayed in Isaac Sim through a driver that gets only real-car inputs

## One command
```bash
sim/cmdtest/run.sh hand policy_v2_veto                 # today's drivers (baselines)
sim/cmdtest/run.sh hand policy_v2_veto navdp combined  # the final comparison, same script and seeds
CMDTEST_ARGS="--prefix final --record goto_kitchen:101 fwd_1m:101" sim/cmdtest/run.sh navdp combined
sim/cmdtest/run.sh smoke --drivers hand --requests fwd_1m --seeds 101   # plumbing test on elegoo-sim-3, no GPU
```
`run.sh` runs 24 cars at once with each car's localiser in its own process (`--slots 24 --loc-workers`; later
`CMDTEST_ARGS` override). It starts elegoo-sim-6 (`CMDTEST_VM`), builds the image `elegoo-cmdtest:1` once (`remote/Dockerfile`:
Isaac Lab 3.0.0 + depth_pro_2 @ aa389cb, timm, embreex, onnxruntime, pyserial; torch/numpy/warp pinned to the
container's), copies the Depth Pro 2-S weights once (`sim/realdepth/models/hf` -> `~/elegoo-sim/hf`), pushes the
code in the project layout to `~/elegoo-sim/work/p4cmd`, runs `harness.py` in the container, polls every minute,
pulls `results/` and the run log, and stops the machine (`KEEP_VM=1` keeps it). Any driver other than `hand` /
`policy_v2_veto` is loaded with `make_driver()` from `sim/models/<name>/driver.py` (pushed with the code).
Outputs per driver: `results/<prefix>_<driver>.csv` (one row per tool call, the scoring rules and the seed list in
the `#` header), `summary_<prefix>_<driver>.md`, `audit_<prefix>_<driver>.json`, `trials_<prefix>_<driver>.jsonl`,
`run_<prefix>.log`; with `--record request:seed`, `<driver>_<request>_<seed>_car_eye.mp4` and `_topdown.png`.

## Files
| File | What |
|---|---|
| `DRIVER_API.md` | the contract (lead's; not changed here) |
| `api.py` | `CarObs`, `DriveCommand`, `VisionDriver`, `load_driver(name)` |
| `baselines.py` | hand / policy_v2_veto behind the contract (go_to: navigator's driver; forward/back/explore/patrol: drive.py's straight controller, as today's car) |
| `record_gemini.py` | recorded `gemini_script.json` once on the Mac with agent.py's exact Live config (`--dry-run` prints the declarations) |
| `gemini_script.json`, `gemini_recording.log` | the fixed script (43 requests, 51 calls) and the recording run's output |
| `tools.py` | agent.py's tool layer on a sim clock (go_to on sim/navigate + sim/localize, drive.py turn/StallWatch, dead reckoning) and the replay |
| `inputs.py` | rendered frame -> latency, drops, motion blur, JPEG; sonar schedule and dropout |
| `env.py` | `CmdEnv(DriverEnv)`: visual scene + appearance, per-episode car response / camera mount / HFOV / sonar cone / gyro drift, truth for scoring |
| `score.py` | the scoring rules (also in every CSV header), CSV and summary writers |
| `harness.py` | lockstep scheduler: `--slots` cars at once, trials, audit, video/plot, batched overlapped depth |
| `locpool.py` | each car's Localizer in its own process (`--loc-workers`), same code and seed |
| `depth_check.py` | batched vs per-frame Depth Pro comparison |
| `run.sh`, `remote/Dockerfile` | the one command |

## What a driver gets (enforced)
Only `AuditedDriver` holds the driver; every `act()` checks `type(obs) is CarObs and type(cmd) is DriveCommand` and
the audit JSON records the field names, the types seen, and per-trial input statistics (frame drop rate, range
None rate, frame age, blur, JPEG quality, camera mount, car response). Truth (pose, bumper ring, tilt) is read only
by `score.py` from the per-step log.

## Real-like inputs, per step / per episode
- RGB: RTX render at 720x540 / 70 deg, centre-cropped to the episode's HFOV (60-70) and resized to 640x480; lens
  pitch 17-23 deg, height 0.121-0.131 m, forward 0.078-0.086 m plus appearance.py's jitter; visual scene with
  building-wide looks every 250 steps and per-car exposure / white balance / saturation / gamma / blur / noise;
  motion blur from the true yaw rate over a 8-30 ms exposure; JPEG quality 60-90; delivered 1-2 steps late
  (0.10-0.25 s latency, 0.1 s steps), dropped or not new on 11-14 % of steps (measured per run in the audit).
- Depth: `sim/realdepth/realdepth.py`'s steps and constants (Depth Pro 2-S, 320x240 input, bfloat16) on the
  delivered JPEG-degraded frame, on CUDA, batched over all cars with the model's cross-frame attention run once per
  frame (each frame a one-frame clip, as observe() runs it; a plain batch would be one video clip). The run logs
  `DEPTH_BATCH_CHECK` against observe() (batch of 1: bit-exact; larger batches: bf16 numerics, mean 0.002 encoded).
  The delivered frame is known one step ahead (it was captured 1-2 steps earlier), so its estimate runs on a side CUDA
  stream while the simulator computes the next step. `depth_check.py` repeats the comparison on JPEGs.
- Range: cone sonar (25-35 deg) on the collider, read every second step (navigation.Sensors' schedule), not right
  after a head move, plus per-episode dropout: None on > 50 % of steps.
- Gyro: calibration.yaml bias / noise / clip per episode plus a bias random walk; heading = its integral.
- Car response: every calibration.yaml parameter redrawn per episode (speed/turn scale, deadband, lags, delay...).

## Trials
Each request x seed: go_to requests start at the home spot of places.yaml with +-8 cm / +-8 deg placement noise
(as the real car and goto.md); others start on a random free cell (>= 0.35 m clearance) of home's room with a
random heading. Calls run in order like the agent: blocking moves wait to finish, background jobs (go_to, explore,
patrol) are replaced by the next motion call after 1 s, a spoken stop arrives `after_s` into the move, the next
user turn 2 s after the last call; agent.py's 3 s repeat guard and go_to's "already going" guard apply.
Per-episode draws are seeded by the trial seed, so every driver gets the same spawn, car, camera and input draws;
the building-wide look sequence and per-step sensor noise are not seed-locked.

## Not modelled
Floor drops, people / objects (find_object, follow_person are "unsupported"), Gemini's own latency beyond the fixed
1 s, camera lens distortion, Wi-Fi bursts longer than 0.2 s.
