# NavDP camera driver, p7 retrain (navdp2)

Same code as `sim/models/navdp` (p4) with new weights (`weights/navdp2_elegoo.pt`, retrained in the rebuilt DA3
scene with home-desk starts, short straight moves and Depth Pro-like depth; report `.omo/waves/p7/navdp2.md`) and a
scripted `mode == "back"`. Load with `sim/cmdtest/api.load_driver("navdp2")`.

`driver.py: make_driver(**kw) -> NavDPDriver`, the `VisionDriver` of `sim/cmdtest/DRIVER_API.md`:
`reset(seed)`, `act(obs: CarObs, cmd: DriveCommand) -> (left, right)` signed PWM, |pwm| <= 150, one call per
100 ms step. Fields are read by name, so any object with DRIVER_API's fields works.

Model: X-NavDP (InternRobotics/NavDP, `baselines/x-navdp`; arXiv 2607.28560), a point-goal navigation diffusion
policy (RGB-D in, 24-waypoint trajectory out) post-trained with RL in Isaac Sim, fine-tuned here on our car in
our building (`sim/isaac/elegoo/navdp/`, reports `.omo/waves/p4/navdp.md`, `.omo/waves/p7/navdp2.md`).

## mode "back" (not learned)
A forward camera cannot see behind the car, so `back` reverses straight at PWM 110 and holds the gyro heading taken
when the back command starts (4 PWM per degree of drift, at most 60, each side clipped to 150; a goal jump of
> 0.3 m starts a new command). The tool layer ends the move by dead reckoning, as for the hand baseline. Any other
mode after `back` resets the network memory and tracker.

## Camera orientation
The real ESP32 picture is mirrored; `brain/vision.py` (`Vision`, CAR_CAMERA_MIRROR defaults to 1 for the car's
URL) flips every JPEG before publishing it, and `brain/depth_feed.py` runs Depth Pro on those flipped frames. The
Isaac camera renders unmirrored. Feed this driver `Vision.newest()` frames, never the raw ESP32 stream.

## Input mapping (CarObs / DriveCommand -> network)
| Network input | From | How |
|---|---|---|
| RGB memory: 8 frames x 256 DINOv2 tokens | `obs.rgb` (640x480 uint8) | scaled to 224x168, zero-padded to 224x224, /255, ImageNet-normalised, frozen ViT-S/14 tokens per frame; memory = 7 frames sampled evenly from the last 16 steps + the current one (upstream). `rgb None` (dropped frame): the last good frame is reused |
| depth image | `obs.depth` (120x160, observation_spec.md encoding) | decoded to metres (`0.15 * (5/0.15)^(1-obs)`), encoded 0 (far / no return) -> 0, letterboxed like RGB, values outside 0.1-5 m -> 0 (upstream convention). `depth None`: last good depth reused |
| point goal (x forward, y left, 0) m | `cmd.target_xy` | the planner's lookahead waypoint (line of sight); clipped to 25 m. `goal_xy` is not used (target_xy equals it when there is no route) |
| (tracker) heading | `obs.gyro_heading` | dead reckoning between plans |
| (tracker) speed | `obs.last_action` | speed of the last sent PWM from calibration.yaml's nominal model |
| (tracker) latency | `obs.rgb_age_s`, `obs.gyro_rate` | the plan is shifted by the motion since the frame was taken |
| (tracker) reflex | `obs.range_cm` | no forward motion below 8 cm (turning and reversing stay allowed); None = no reading |
| (tracker) speed cap | `cmd.speed_limit` | scales the 0.40 m/s cruise speed |
`mode` selects the scripted reverse (`back`, which also reads `goal_xy` to spot a new command); every other mode runs
the network. Not used: `heading_error`, `how_far`, `kind`, `t`. Nothing else is read.

## Output mapping (trajectory -> PWM)
Each step the network samples 8 trajectories toward the target (10 DDPM steps; base decoder for t >= 6, wheeled
embodiment decoder below), the critic picks the best, and a pure-pursuit tracker follows it:
lookahead 0.30 m on the path after the closest point; target behind (> 115 deg) and nearer than 0.8 m -> reverse
at 0.15 m/s on the arc; target more than 52 deg off -> spin at 1.8 rad/s; otherwise forward at up to 0.40 m/s
(slower near the path end) on the pure-pursuit arc; stop when the path end is within 6 cm.
(v, omega) -> PWM inverts `sim/isaac/elegoo/drive.py`: mean = v / 0.004 + 12.5 deadband; turn from the spin
curve (linear 130/80 deg/s per PWM while rolling); both wheels keep one sign while rolling; scaled to |pwm| <= 150.

## Options (`make_driver(**kw)`)
`weights` (default `weights/navdp_elegoo.pt`; the upstream file also loads), `device` (mps > cuda > cpu),
`samples` (8), `plan_every` (2: the network plans every 2nd step, the tracker drives the step in between; 1: every
step), `v_max` (0.40), `seed`.

## Measured
- Mac M5 Pro, MPS fp16, `act()` with 640x480 RGB: 13.5 decisions/s at plan_every=2, 7.1/s at plan_every=1.
- Isaac, 240 held-out routes, real-car inputs: 93.3 % success, 0.0028 contacts/m, 0.38 m/s (plan_every=2); every step
  93.3 %, 0.0041 contacts/m. Details: `.omo/waves/p4/navdp.md`.

## Files
- `xnavdp/net.py`: the network, parameter names equal the released checkpoint; `xnavdp/dinov2*`: DINOv2 /
  Depth Anything V2 ViT code, copied unchanged (Apache-2.0).
- `policy.py`: batched preprocessing, RGB memory, planning, tracker; shared with the Isaac harness.
- `weights/`: fine-tuned weights, `WEIGHTS.md` (source, hashes, licences), licence texts.
