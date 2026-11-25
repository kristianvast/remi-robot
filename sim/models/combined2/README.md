# Combined camera model "C" (p4): drive + where am I + which way to the goal

One Depth Anything V2 Metric-Indoor-Small encoder (frozen, Apache-2.0) shared by three small heads, trained in
Isaac Lab in our building (sim/isaac/elegoo/combined/). Report: .omo/waves/p4/combined.md.

## Use
```python
from sim.models.combined.driver import make_driver
d = make_driver()                  # optional: providers=["CoreMLExecutionProvider", "CPUExecutionProvider"], threads=4
d.reset()
left, right = d.act(obs, cmd)      # CarObs, DriveCommand per sim/cmdtest/DRIVER_API.md (objects or dicts)
room, x, y, yaw, conf = d.where_am_i()
heading, conf = d.goal_direction("kitchen")   # area id or places.yaml area name
d.depth                            # (120, 160) observation_spec depth of the newest frame (the model's own)
```
Needs numpy, onnxruntime, and cv2 (or Pillow) for the resize; sim/realdepth/.venv has all three.

Speed on this Mac (M5 Pro, `python -m sim.models.combined.driver --bench 5`, 640x480 JPEG decode + encoder + heads):
56 decisions/s on the CPU provider (median 17.8 ms, p95 18.4 ms); the CoreML provider is slower (24/s), so CPU is
the default.

Input path: train renders 224x168, upsamples to 640x480, blurs, JPEG q60-90 at 640x480, area-downsamples to
224x168; this driver area-resizes the real 640x480 decoded frame (cv2.INTER_AREA). Checked on 8692 held-out
frames rendered at 640x480: feature cosine (driver path vs training path) median 0.998, p5 0.993; same held-out
success (91.7 % on 60 routes). Details and all scores: .omo/waves/p4/combined.md.

## I/O mapping (what each CarObs / DriveCommand field becomes)
| Input | Used as |
|---|---|
| `rgb` (480, 640, 3) uint8 | resized to 224 x 168 (area) -> encoder -> frame feature (CLS + 2 x 4 pooled patch tokens + 80-number depth summary) pushed into a 4-frame history; None = no new frame, history kept |
| `rgb_age_s` | age of the newest history slot (all slots then age by the step time from `t`) |
| `depth` | ignored (the model estimates its own; exposed as `driver.depth`) |
| `range_cm` | range/150 plus a valid flag (None -> 0, flag 0) |
| `gyro_rate` | rad/s / radians(250) |
| `gyro_heading` | not used |
| `last_action` | PWM/255 per side |
| `t` | step time for the frame ages (default 0.1 s) |
| `heading_error` | sin, cos; also drives the heading-following prior |
| `how_far` | /2 |
| `speed_limit` | 0..1, also caps the output |
| `target_xy` | (forward, left) clipped to +-3 m, /3 |
| `goal_xy` | log1p(distance)/2, sin and cos of its bearing |
| `kind`, `mode` | not used (trained on go_to-style route following only) |

Output: signed PWM (left, right), |PWM| <= 150 x speed_limit, from `wheels()` = sim/isaac/elegoo/train/model.py's
heading-following prior plus the driving head's residual.
`where_am_i()`: room id (area_01..area_10 of scans/merged/rooms_next.json), x, y (m, floor-map / stage frame),
yaw (rad, map frame), confidence = room probability x exp(-max(0, sd_xy - 0.25 m) / 0.5 m) with sd_xy the head's
own predicted position sd. `goal_direction(place)`: car-frame direction of steepest descent of the drivable
(clearance-weighted geodesic) distance to that area's default goal (sim/navigate/places.yaml).

## Files
| File | What |
|---|---|
| `driver.py` | `make_driver()`, `CombinedDriver` (VisionDriver + `where_am_i` + `goal_direction`) |
| `weights/encoder.onnx` | DA2 Metric-Indoor-Small at 224 x 168 -> depth (1,120,160), frame (1,3536) |
| `weights/heads.onnx` | history (1,4,3536), ages (1,4), aux (1,15) -> residual, room logits, pose, goal |
| `weights/heads.pt` | the heads' PyTorch state dict (sim/isaac/elegoo/combined/common.py `Heads`) |
| `weights/SHA256SUMS`, `weights/LICENSE.md` | hashes, encoder source and licence |
| `weights/export_check.json` | ONNX vs PyTorch max differences on recorded frames |
| `videos/` | 3 success + 3 failure videos (mp4) and contact sheets from the held-out nominal run |
