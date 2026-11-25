#!/usr/bin/env bash
# Runs ON the elegoo-sim VM: start an Isaac Lab script detached in a container (same mounts as
# sim/cloud/run_headless.sh), log to ~/elegoo-sim/work/$WORK/logs/<name>.log. The default image is the cmdtest
# image (isaac-lab 3.0.0 + depth_pro_2, sim/cmdtest/remote/Dockerfile) so realdepth runs in the same container.
#   WORK=p7navdp bash vm_run.sh <name> <script relative to /workspace/elegoo/$WORK> [args ...]
set -euo pipefail
NAME="$1"; SCRIPT="$2"; shift 2
WORK="${WORK:-p7navdp}"
IMAGE="${IMAGE:-elegoo-cmdtest:1}"
LOGS=~/elegoo-sim/work/$WORK/logs
mkdir -p "$LOGS"
ARGS=""
for a in "$@"; do ARGS+=" $(printf '%q' "$a")"; done
sudo docker rm -f "navdp-$NAME" >/dev/null 2>&1 || true
sudo docker run -d --name "navdp-$NAME" --gpus all --network=host \
  -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONDONTWRITEBYTECODE=1 \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v ~/elegoo-sim/work:/workspace/elegoo:rw \
  -w /workspace/elegoo/$WORK \
  --entrypoint bash "$IMAGE" -c \
  "/workspace/isaaclab/isaaclab.sh -p /workspace/elegoo/$WORK/$SCRIPT $ARGS > /workspace/elegoo/$WORK/logs/$NAME.log 2>&1; echo EXIT \$? >> /workspace/elegoo/$WORK/logs/$NAME.log"
echo "started navdp-$NAME"
