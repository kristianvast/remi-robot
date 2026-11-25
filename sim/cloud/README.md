# elegoo-sim cloud host (GCP, Isaac Sim + Isaac Lab, headless)

Scripts to create and drive a Google Cloud GPU VM that runs NVIDIA Isaac Sim / Isaac Lab
headless in the official NGC container. Project `your-gcp-project`. Every script refuses
machine names that do not start with `elegoo-sim-`, so the unrelated production VMs in the
project are never touched. All scripts take the machine name as an optional first argument
(default `elegoo-sim-1`) and find its zone themselves.

## What is installed (measured on elegoo-sim-1, 2026-11-24)

| Item | Version |
|---|---|
| Zone / machine type | us-central1-c, g2-standard-12 (1x NVIDIA L4 24 GB, 12 vCPU, 48 GB RAM) |
| Boot disk | 200 GB pd-balanced (52 GB used after the image pulls) |
| OS image | ubuntu-accelerator-2404-amd64-with-nvidia-595-v20261118 (Ubuntu 24.04.5 LTS, kernel 7.0.0-1017-gcp) |
| NVIDIA driver | 595.91.07 (595-server, open kernel module), CUDA 13.2; docs require >= 595.58.03 |
| Extra driver package | libnvidia-gl-595-server 595.91.07 (Vulkan ICD + GL, needed by RTX; see Gotchas) |
| Docker | 29.10.0 |
| NVIDIA Container Toolkit | 1.20.2 |
| Isaac Sim | nvcr.io/nvidia/isaac-sim:6.2.0 (31.9 GB), build 6.2.0-rc.14+release.52805.02189a84.gl |
| Isaac Lab | nvcr.io/nvidia/isaac-lab:3.0.0 (51.4 GB), Isaac Lab 3.0.0 on Isaac Sim 6.2.0, Python 3.12 |

Why the container: the Isaac Sim docs recommend the container install "for deployment on
remote headless servers or the Cloud", and Isaac Lab 3.0.0 publishes a prebuilt release image
on top of it (`nvcr.io/nvidia/isaac-lab:3.0.0`); Isaac Sim 6.2.0 <-> Isaac Lab v3.0.0-GA is the
pairing in the Isaac Sim compatibility matrix.

Why g2-standard-12 and not g2-standard-8: on 2026-11-24 every g2-standard-8 + L4 request in
us-central1-a, -b and -c failed with `STOCKOUT` ("A g2-standard-8 VM instance with 1
nvidia-l4 accelerator(s) is currently unavailable"). g2-standard-12 (same single L4, more
CPU/RAM) was available in us-central1-c. `create.sh` tries g2-standard-8 first in every zone,
then g2-standard-12, and falls back from us-central1 to other US L4 zones
(override with `MACHINE_TYPES=...` / `ELEGOO_SIM_ZONES=...`). g2-standard-4 is not offered:
its 16 GB RAM is below Isaac Sim's 32 GB minimum.

## Price (looked up 2026-11-24)

From https://cloud.google.com/products/compute/pricing/accelerator-optimized (G2 table, on-demand,
GPU included) and https://cloud.google.com/compute/disks-image-pricing:

| Resource | Price |
|---|---|
| g2-standard-12 (this VM) | $1.000416348 / hour while RUNNING |
| g2-standard-8 (preferred size) | $0.853624312 / hour while RUNNING |
| 200 GB pd-balanced disk | $0.000136986 / GiB-hour = ~$0.027 / hour = ~$20 / month, billed also while TERMINATED |

So: about $1.03/hour running, about $0.66/day stopped. The pricing page did not let me confirm
which region its default table shows; treat these as us-central1 list prices to re-check.
Delete the VM (and its disk) when it is not needed for weeks.

## Scripts

| Script | What it does |
|---|---|
| `create.sh [name]` | Creates the VM (first zone/shape with capacity), then copies `remote/` and runs `remote/setup_vm.sh` (idempotent; re-running on an existing VM just re-provisions). |
| `start.sh [name]` / `stop.sh [name]` | Start / stop; prints the resulting status. |
| `ssh.sh [name] [-- cmd ...]` | Interactive shell, or run one command. |
| `push.sh [name] <folder>` | Copies a local folder to `~/elegoo-sim/work/<folder>` on the VM = `/workspace/elegoo/<folder>` in the container. |
| `pull.sh [name] <path> [local_dir]` | Copies back; relative paths are under `~/elegoo-sim/work`. |
| `run_headless.sh [name] <script.py> [args]` | Runs an Isaac Lab script headless in the isaac-lab container with the GPU, Isaac Sim caches mounted, working dir `/workspace/elegoo`. Relative script paths are looked up in `/workspace/elegoo` (pushed folders) first, then `/workspace/isaaclab` (stock scripts). |

`remote/` holds what runs on the VM: `setup_vm.sh` (driver GL libs, Docker, container toolkit,
CDI refresh, image pulls, idle watchdog) and the idle-shutdown script and systemd units.

## Proof run (reproduce)

```bash
cd sim/cloud
./start.sh
./push.sh examples/step_counter
./run_headless.sh step_counter/run_counted.py --task Isaac-Cartpole-Camera-Direct \
  --num_envs 4 --max_steps 300 --viz none physics=isaacsim_physx renderer=isaacsim_rtx \
  env.write_image_to_file=True env.scene.tiled_camera.width=480 env.scene.tiled_camera.height=270
./pull.sh cartpole_rgb.png proof/
./stop.sh
```

`run_counted.py` is the stock Isaac Lab random agent plus a step counter. Measured on
elegoo-sim-1 (warm caches): `PROOF_STEPS=300 STEP_SECONDS=20.6` (4 envs, 480x270 RTX tiled
camera), 1 min 7 s for the whole command including Kit start-up. The first run on a fresh VM
took 5 min 27 s (shader compilation). During the run `nvidia-smi` showed the
`/opt/isaaclab-venv/bin/python` process as type C+G using 4216 MiB on the L4. Kit reported
`Graphics API: Vulkan`, `NVIDIA L4`, renderer `IsaacRtxRenderer`, env device `cuda:0`.
Result: `proof/cartpole_rgb_rtx_300steps.png` (966x546 tile of the 4 camera views).

## Idle cost guard: automatic shutdown after 45 minutes without a GPU process

`elegoo-idle-shutdown.timer` runs `/usr/local/bin/elegoo-idle-shutdown` every minute.
The script checks `nvidia-smi -q -d PIDS` (lists compute AND graphics processes; Isaac Sim's
RTX renderer is a C+G process) and stores the last time a GPU process was seen in
`/run/elegoo-idle-last-busy` (reset at every boot, so a fresh start gets the full 45 minutes).
When no GPU process has been seen for 45 minutes it runs `shutdown -h now`; GCE then shows the
VM as TERMINATED and vCPU/GPU billing stops (the disk is still billed). An idle SSH session does
not keep the VM alive - only GPU work does. If `nvidia-smi` itself fails, the minute counts as
busy (no shutdown on a broken driver; check `journalctl -t elegoo-idle-shutdown`).

- Change the limit: edit `Environment=IDLE_MINUTES=45` in `remote/elegoo-idle-shutdown.service`
  and re-run `./create.sh`, or on the VM `sudo systemctl edit elegoo-idle-shutdown.service`.
- Disable for a long interactive session: `./ssh.sh -- sudo systemctl stop elegoo-idle-shutdown.timer`
  (it comes back on the next boot).
- Tested end to end on 2026-11-24 with a runtime-only override `IDLE_MINUTES=2` (in /run, gone
  after reboot): the VM powered itself off and gcloud reported TERMINATED (see the report in
  `.omo/waves/p1/gcp-sim-host.md`).

## Gotchas found while building this

1. The ubuntu-accelerator image ships the compute-only driver. Without `libnvidia-gl-595-server`
   Isaac Sim logs `vkCreateInstance failed ... ERROR_INCOMPATIBLE_DRIVER`, still exits 0, and
   writes black camera images. `setup_vm.sh` installs the GL package pinned to the driver version.
2. Docker uses the CDI spec in `/run/cdi/nvidia.yaml`, generated by `nvidia-cdi-refresh` at
   toolkit install time; it must be regenerated after adding the GL libs (setup does this, and
   `/run` is rebuilt on every boot).
3. Isaac Lab 3 has no `--headless` flag; headless is the default without a display. The random
   agent defaults to the `newton_gl` visualizer, which crashes without a display
   (`pyglet ... NoSuchConfigException`) - pass `--viz none`.
4. Isaac Lab 3 defaults to Newton physics and the Newton Warp renderer; to actually use Isaac Sim
   pass `physics=isaacsim_physx renderer=isaacsim_rtx`.
5. `isaaclab.sh` prints a deprecation warning (removed in 3.1, use `uv run isaaclab`).
