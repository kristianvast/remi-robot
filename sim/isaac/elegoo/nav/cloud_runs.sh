#!/usr/bin/env bash
# All building runs on the L4, then pull and build sim/isaac/views/.
# Usage (from the project root, machine RUNNING): sim/isaac/elegoo/nav/cloud_runs.sh [elegoo-sim-NAME] [steps...]
# steps default to: push smoke views level collide drive bench pull report
set -uo pipefail
NAME="${1:-elegoo-sim-1}"; shift || true
STEPS="${*:-push smoke views level collide drive bench pull report}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CLOUD="$ROOT/sim/cloud"
STAGE=/tmp/elegoo_integ/integ
LOGS=/tmp/elegoo_logs
PY="$ROOT/sim/.venv/bin/python"
OUT=/workspace/elegoo/integ/out
mkdir -p "$LOGS"
run() {
  local tag="$1"; shift
  "$CLOUD/run_headless.sh" "$NAME" integ/elegoo/nav/run.py "$@" --output "$OUT" --viz none > "$LOGS/$tag.log" 2>&1
  echo "$tag exit=$?"; grep -h '^SMOKE\|^SCENE_READY\|^VIEWS_DONE\|^DRIVE\|^COLLIDE\|Traceback\|Error:' "$LOGS/$tag.log" | tail -12
}
for step in $STEPS; do
  case "$step" in
    push)
      rm -rf "$STAGE" && mkdir -p "$STAGE"
      rsync -a --exclude source --exclude __pycache__ --exclude results --exclude benchmark --exclude proof \
        "$ROOT/sim/isaac/elegoo/" "$STAGE/elegoo/"
      rsync -a --exclude cache --exclude proof --exclude __pycache__ "$ROOT/sim/isaac/scene/" "$STAGE/scene/"
      cp "$ROOT/scans/merged/views/poses_rendered.json" "$STAGE/"
      "$CLOUD/ssh.sh" "$NAME" -- 'sudo rm -rf ~/elegoo-sim/work/integ'
      "$CLOUD/push.sh" "$NAME" "$STAGE" | tail -1 ;;
    smoke) run smoke smoke --num-envs 2 ;;
    views) run views views --poses /workspace/elegoo/integ/poses_rendered.json --kinds room,drive ;;
    level) run level views --poses /workspace/elegoo/integ/poses_rendered.json --kinds level --pitch 0 ;;
    collide) run collide collide ;;
    drive) run drive drive ;;
    bench) "$(dirname "$0")/bench_sweep.sh" "$NAME" ;;
    pull) rm -rf /tmp/elegoo_out && "$CLOUD/pull.sh" "$NAME" integ/out /tmp/elegoo_out | tail -1 ;;
    report) "$PY" "$ROOT/sim/isaac/elegoo/nav/report.py" /tmp/elegoo_out/out ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
