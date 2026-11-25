# ELEGOO V4, Isaac Lab 3.0

`environment.ElegooEnv` is a standalone batched environment for Isaac Lab
3.0 / Isaac Sim 6.2. Construct it inside a PhysX `launch_simulation` context
with a 10 ms `SimulationCfg.dt`, passing `EnvSpec` to choose batch size,
camera outputs and resolution, obstacles, and randomization. See `run.py`.

`step(action)` accepts a device tensor `[N, 2]` (left, right), clipped to
`[-1, 1]` and rounded to signed PWM ticks. `step(None)` drops packets,
unlike a zero tensor, which explicitly sends stop. By default each step
advances seven physics ticks and returns a dict of batched device tensors:

| Key | Meaning |
|---|---|
| `last_action` | Most recent applied normalized PWM; zero after watchdog |
| `yaw_deg`, `gyro_rate_deg_s` | Integrated, unwrapped yaw and clipped rate |
| `range_cm` | Nearest sampled cone hit in integer cm, capped at 150 |
| `floor_bits`, `on_ground` | Three near-floor flags and their conjunction |
| `rgb`, `depth` | Optional NHWC RGB bytes and optical-axis depth in metres |

No pose, wheel state, or perfect heading is exposed as an observation.
Camera data lives in renderer buffers; clone frames before storing them.
`set_pan` takes a batch of angles in radians independently of PWM.
`reset` accepts optional environment indices. No reward or termination
policy is imposed, and resets are explicit.

`calibration.yaml` records real values and proposed randomization ranges.
Turn and forward-speed targets use independent lagged curves. The code
applies velocity to a dynamic articulation rather than teleporting poses.
Wheel spin is visual; tire friction is not the calibrated drive mechanism.
With `randomize=True` on construction, dynamics and low-dimensional sensor
parameters are sampled independently per car; camera mounting and FOV use
one sample for the whole RTX batch. To set a specific camera, supply
`CameraSpec` and leave randomization off.

`convert.py` imports the staged `source/` URDF and flattens the PhysX
variant to `asset/elegoo_v4.usdc`. That file is self-contained and is what
`world.py` loads. Run `run.py match`, `run.py wall`, `run.py proof`, or
`run.py bench --num-envs 16 --width 160 --height 120` through the project's
Isaac Lab container helper using `--viz none`. Set `--output` to a mounted
directory writable by the container UID for modes that emit artifacts.
On the L4, the 256-car 640x480 benchmark requires `--minimal-rtx` for
diffuse RGB plus depth; the default photoreal RGB mode exhausted VRAM.
