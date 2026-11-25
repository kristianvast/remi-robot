#!/usr/bin/env bash
# Called every minute by elegoo-idle-shutdown.timer. Powers the VM off (GCE status becomes
# TERMINATED, so vCPU/GPU billing stops) once no process has used the GPU for IDLE_MINUTES.
# The idle clock starts at boot, so a freshly started VM gets the full grace period.
set -u

IDLE_MINUTES="${IDLE_MINUTES:-45}"
STATE=/run/elegoo-idle-last-busy
LOG_TAG=elegoo-idle-shutdown

now=$(date +%s)
[ -f "$STATE" ] || echo "$now" > "$STATE"

# "nvidia-smi -q -d PIDS" lists compute AND graphics processes; Isaac Sim's RTX renderer is a
# graphics client, which --query-compute-apps alone would miss.
if ! gpu_pids=$(nvidia-smi -q -d PIDS 2>&1); then
  logger -t "$LOG_TAG" "nvidia-smi failed, not counting as idle: $gpu_pids"
  echo "$now" > "$STATE"
  exit 0
fi

if echo "$gpu_pids" | grep -q "Process ID"; then
  echo "$now" > "$STATE"
  exit 0
fi

last_busy=$(cat "$STATE")
idle_min=$(( (now - last_busy) / 60 ))
if [ "$idle_min" -ge "$IDLE_MINUTES" ]; then
  logger -t "$LOG_TAG" "no GPU process for ${idle_min} min, powering off"
  /sbin/shutdown -h now "elegoo idle shutdown: no GPU process for ${idle_min} min"
fi
