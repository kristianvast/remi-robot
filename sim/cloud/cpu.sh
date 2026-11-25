#!/usr/bin/env bash
# CPU-only cloud machine for the Mac-side simulator work (macsim, localize, navigate, brain stand-in tests).
# Usage:
#   cpu.sh setup                 one-time: uv, idle shutdown, first push, venvs
#   cpu.sh push                  copy the project's simulator code and data to ~/elegoo on the machine
#   cpu.sh run '<command>'       run a shell command in ~/elegoo on the machine (project root layout)
#   cpu.sh pull <path>...        copy files or folders (relative to the project root) back to the Mac
#   cpu.sh start | stop | status
set -euo pipefail
NAME="${ELEGOO_CPU_NAME:-elegoo-sim-3}"
PROJECT=your-gcp-project
ZONE=us-central1-c
P="$(cd "$(dirname "$0")/../.." && pwd)"
case "$NAME" in elegoo-sim-*) ;; *) echo "refusing: $NAME is not an elegoo-sim-* machine" >&2; exit 1 ;; esac
gssh() { gcloud compute ssh "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet "$@"; }

# What the simulator code reads, with the same layout as on the Mac. No venvs, caches or bulky outputs.
PATHS=(sim/macsim sim/localize sim/navigate sim/policy/driver.onnx sim/policy/driver_v2.onnx sim/policy/run_driver.py sim/realdepth/observation_spec.md
       sim/isaac/scene/scene_meta.yaml sim/isaac/scene/layers sim/isaac/elegoo
       scans/merged/collision.obj scans/merged/floor_map.png scans/merged/floor_map.yaml
       scans/merged/rooms.json scans/merged/rooms_next.json scans/merged/views/poses_rendered.json
       sim/isaac/views car/tools/brain .omo/waves/p1 .omo/waves/p3)
EXCLUDES=(--exclude .venv --exclude .uv-cache --exclude __pycache__ --exclude '*.mp4' --exclude patrols
          --exclude 'sim/isaac/views/scan_drive' --exclude '*_rgb.jpg' --exclude '*_depth.png' --exclude .env)

case "${1:-}" in
  push)
    existing=(); for p in "${PATHS[@]}"; do [ -e "$P/$p" ] && existing+=("$p"); done
    COPYFILE_DISABLE=1 tar -C "$P" "${EXCLUDES[@]}" -czf - "${existing[@]}" | gssh --command "mkdir -p ~/elegoo && tar -C ~/elegoo -xzf - 2>/dev/null; echo pushed"
    ;;
  run)
    shift
    gssh --command "cd ~/elegoo && export PATH=\$HOME/.local/bin:\$PATH && $*"
    ;;
  pull)
    shift
    gssh --command "cd ~/elegoo && tar -czf - $*" | tar -C "$P" -xzf -
    ;;
  setup)
    gssh --command "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; sudo apt-get -qq update >/dev/null && sudo apt-get -qq install -y libgl1 libglib2.0-0 >/dev/null; echo tools ok"
    # Power off after 45 minutes with a 1-minute load under 0.5.
    gssh --command "printf '%s\n' '#!/bin/sh' 'f=/run/elegoo-idle; l=\$(cut -d\" \" -f1 /proc/loadavg); if [ \"\$(echo \"\$l < 0.5\" | bc)\" = 1 ]; then n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); else n=0; fi; echo \$n > \$f; [ \$n -ge 9 ] && shutdown -h now' | sudo tee /usr/local/bin/elegoo-idle >/dev/null && sudo chmod +x /usr/local/bin/elegoo-idle && sudo apt-get -qq install -y bc >/dev/null && echo '*/5 * * * * root /usr/local/bin/elegoo-idle' | sudo tee /etc/cron.d/elegoo-idle >/dev/null && echo idle-shutdown ok"
    "$0" push
    "$0" run "cd sim/macsim && uv sync -q && echo macsim venv ok"
    ;;
  start)  gcloud compute instances start "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet ;;
  stop)   gcloud compute instances stop "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet ;;
  status) gcloud compute instances describe "$NAME" --project "$PROJECT" --zone "$ZONE" --format="value(status,machineType.basename())" ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
