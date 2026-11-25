#!/usr/bin/env bash
# Throughput with the building loaded: one fresh Kit process per configuration.
# Usage (from the project root, machine RUNNING, integ/ pushed): sim/isaac/elegoo/nav/bench_sweep.sh [elegoo-sim-NAME]
# Appends rows to integ/out/bench.csv on the VM; logs land in /tmp/elegoo_logs/bench_*.log.
set -uo pipefail
NAME="${1:-elegoo-sim-1}"
CLOUD="$(cd "$(dirname "$0")/../../../cloud" && pwd)"
mkdir -p /tmp/elegoo_logs
run() {
  local tag="$1"; shift
  "$CLOUD/run_headless.sh" "$NAME" integ/elegoo/nav/run.py bench --output /workspace/elegoo/integ/out \
    --viz none "$@" > "/tmp/elegoo_logs/bench_$tag.log" 2>&1
  echo "$tag exit=$? $(grep -h '^BENCH\|Error\|out of memory' "/tmp/elegoo_logs/bench_$tag.log" | tail -2)"
}
for n in 16 64 256; do
  run "${n}_160_depth" --num-envs "$n" --width 160 --height 120 --outputs depth --steps 200
  run "${n}_160_rgbd" --num-envs "$n" --width 160 --height 120 --outputs rgb,depth --steps 200
  run "${n}_640_rgbd" --num-envs "$n" --width 640 --height 480 --outputs rgb,depth --steps 60
done
run "256_640_minimal" --num-envs 256 --width 640 --height 480 --outputs simple_shading_diffuse_mdl,depth --steps 60
run "256_nocam" --num-envs 256 --outputs "" --steps 200
