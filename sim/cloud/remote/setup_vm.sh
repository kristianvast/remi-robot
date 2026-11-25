#!/usr/bin/env bash
# Runs ON the VM (copied there by create.sh). Idempotent: safe to re-run.
set -euo pipefail

ISAACLAB_IMAGE="${1:-nvcr.io/nvidia/isaac-lab:3.0.0}"
ISAACSIM_IMAGE="${2:-nvcr.io/nvidia/isaac-sim:6.2.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== driver"
nvidia-smi

echo "== driver graphics libs (Vulkan/GL for RTX)"
# The ubuntu-accelerator image ships the compute-only driver; Isaac Sim RTX needs the Vulkan
# ICD and GL libs from libnvidia-gl, pinned to the exact installed driver version.
compute_pkg=$(dpkg-query -W -f='${Package}\n' 'libnvidia-compute-*' | head -n1)
driver_ver=$(dpkg-query -W -f='${Version}' "$compute_pkg")
gl_pkg=${compute_pkg/compute/gl}
if ! dpkg -s "$gl_pkg" >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$gl_pkg=$driver_ver"
fi
ls /usr/share/vulkan/icd.d/nvidia_icd.json

echo "== docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
fi
sudo usermod -aG docker "$USER"

echo "== nvidia container toolkit"
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -q
  sudo apt-get install -y -q nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
# Docker uses the CDI spec in /run/cdi; regenerate it so it includes the libnvidia-gl files.
sudo rm -f /etc/cdi/nvidia.yaml
sudo systemctl restart nvidia-cdi-refresh.service
grep -q nvidia_icd /var/run/cdi/nvidia.yaml
sudo systemctl restart docker

echo "== gpu inside a container"
sudo docker run --rm --runtime=nvidia --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi -L
sudo docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 \
  ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0

echo "== host folders (uid 1000 is the container user)"
mkdir -p ~/docker/isaac-sim/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}
mkdir -p ~/elegoo-sim/work
sudo chown -R 1000:1000 ~/docker/isaac-sim ~/elegoo-sim/work
sudo chmod -R a+rwX ~/elegoo-sim/work

echo "== images"
sudo docker pull "$ISAACSIM_IMAGE"
sudo docker pull "$ISAACLAB_IMAGE"

echo "== idle auto-shutdown"
sudo install -m 0755 "$HERE/idle_shutdown.sh" /usr/local/bin/elegoo-idle-shutdown
sudo install -m 0644 "$HERE/elegoo-idle-shutdown.service" /etc/systemd/system/
sudo install -m 0644 "$HERE/elegoo-idle-shutdown.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now elegoo-idle-shutdown.timer

echo "== versions"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
sudo docker --version
nvidia-ctk --version | head -n1
sudo docker image ls --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'isaac'
echo "SETUP_OK"
