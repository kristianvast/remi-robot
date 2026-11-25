#!/usr/bin/env bash
# Runs ON the VM after training (p7): p4 navdp (old) and navdp2 (new) through identical evaluate.py runs (same split,
# routes, seed 7, plan every 2nd step), old then new per run so pairs land together; docker wait between runs.
# dpscale uses evaluate.py's defaults (depth_measure.py); depthpro runs Depth Pro 2-S on 640x480 frames.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE=/workspace/elegoo/p7navdp
OLD=$BASE/sim/models/navdp/weights/navdp_elegoo.pt
NEW=$BASE/out/run/final.pt
OUT=$BASE/out/eval
EVAL=sim/isaac/elegoo/navdp/evaluate.py
sudo docker wait navdp-train >/dev/null 2>&1 || true
run() {
  local name="$1"; shift
  bash "$HERE/vm_run.sh" "$name" "$EVAL" "$@" --controllers navdp --plan-every 2 --video-cars 0 --viz none
  sudo docker wait "navdp-$name"
}
pair() {
  local tag="$1"; shift
  run "old_$tag" --checkpoint "$OLD" --label navdp_p4 --output "$OUT/old_$tag" "$@"
  run "new_$tag" --checkpoint "$NEW" --label navdp2 --output "$OUT/new_$tag" "$@"
}
pair test_ns --split test --conditions nominal,stress
pair short --split short_test --per-area 1000 --conditions nominal,stress,dpscale
pair short_dp --split short_test --per-area 1000 --conditions depthpro --rgb-size 640 480
pair home --split home_test --per-area 1000 --conditions nominal,stress,dpscale
pair home_dp --split home_test --per-area 1000 --conditions depthpro --rgb-size 640 480
pair test_dp --split test --per-area 5 --conditions depthpro --rgb-size 640 480
echo CHAIN_DONE > "$BASE/logs/chain.done"
