#!/usr/bin/env bash
# Usage: run_headless.sh [elegoo-sim-NAME] <script.py> [script args ...]
# Runs an Isaac Lab python script headless in the nvcr.io/nvidia/isaac-lab container.
# <script.py> is resolved inside the container: absolute paths as-is, otherwise first under
# /workspace/elegoo (folders sent with push.sh), then under /workspace/isaaclab (stock
# Isaac Lab scripts, e.g. scripts/environments/random_agent.py).
# The working directory is /workspace/elegoo, so relative outputs land in ~/elegoo-sim/work
# on the VM and can be fetched with pull.sh.
# Isaac Lab 3 has no --headless flag: with no DISPLAY in the container and no Kit visualizer
# requested, every run is headless (rendering for camera sensors still happens offscreen).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
pick_name "$@"
shift "$SHIFT"
[ $# -ge 1 ] || die "usage: run_headless.sh [elegoo-sim-NAME] <script.py> [args ...]"
require_zone
[ "$(status_of)" = "RUNNING" ] || die "$NAME is not RUNNING (run ./start.sh $NAME)"

ARGS=""
for a in "$@"; do ARGS+=" $(printf '%q' "$a")"; done

gssh --command "bash -s" <<EOF
set -euo pipefail
sudo docker run --rm --gpus all --network=host \\
  -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all \\
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \\
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \\
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \\
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \\
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \\
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \\
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \\
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \\
  -v ~/elegoo-sim/work:/workspace/elegoo:rw \\
  -w /workspace/elegoo \\
  --entrypoint bash $ISAACLAB_IMAGE -c '
    set -- $ARGS
    s="\$1"; shift
    case "\$s" in
      /*) ;;
      *) if [ -f "/workspace/elegoo/\$s" ]; then s="/workspace/elegoo/\$s"; else s="/workspace/isaaclab/\$s"; fi ;;
    esac
    exec /workspace/isaaclab/isaaclab.sh -p "\$s" "\$@"
  '
EOF
