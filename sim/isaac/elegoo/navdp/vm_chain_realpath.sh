#!/usr/bin/env bash
# Runs ON elegoo-sim-8 after vm_chain.sh: the real-car image path on 5 test routes per area (50): RTX at 640x480,
# JPEG q60-90, resized by the driver's own code (policy.rgb_input), fine-tuned weights, nominal, navdp only.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
until [ -f ~/elegoo-sim/work/p4navdp/logs/chain.done ]; do sleep 30; done
bash "$HERE/vm_run.sh" ft_nominal_640 sim/isaac/elegoo/navdp/evaluate.py --checkpoint /workspace/elegoo/p4navdp/out/run/final.pt \
  --label finetuned_640 --conditions nominal --controllers navdp --per-area 5 --rgb-size 640 480 --video-cars 0 \
  --plan-every 2 --output /workspace/elegoo/p4navdp/out/ft_nominal_640 --viz none
sudo docker wait navdp-ft_nominal_640
echo DONE > ~/elegoo-sim/work/p4navdp/logs/chain_realpath.done
