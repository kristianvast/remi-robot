#!/usr/bin/env bash
# Runs ON elegoo-sim-8 after vm_chain_realpath.sh: re-drive the fine-tuned model on routes where it failed in the
# nominal run (different per-car draws, so a failure may not repeat) and render their videos.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
until [ -f ~/elegoo-sim/work/p4navdp/logs/chain_realpath.done ]; do sleep 30; done
bash "$HERE/vm_run.sh" ft_failvideos sim/isaac/elegoo/navdp/evaluate.py --checkpoint /workspace/elegoo/p4navdp/out/run/final.pt \
  --label finetuned_failroutes --conditions nominal --controllers navdp --routes 35 157 161 290 \
  --plan-every 2 --output /workspace/elegoo/p4navdp/out/ft_failvideos --viz none
sudo docker wait navdp-ft_failvideos
sudo docker run --rm -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip -v ~/elegoo-sim/work:/workspace/elegoo \
  -w /workspace/elegoo/p4navdp --entrypoint bash nvcr.io/nvidia/isaac-lab:3.0.0 -c \
  '/opt/isaaclab-venv/bin/python -m pip install -q opencv-python-headless >/dev/null 2>&1; /opt/isaaclab-venv/bin/python sim/isaac/elegoo/navdp/videos.py out/ft_failvideos/video.npz out/ft_failvideos/videos > logs/vid_fail.log 2>&1'
echo DONE > ~/elegoo-sim/work/p4navdp/logs/chain_videos.done
