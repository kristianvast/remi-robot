#!/usr/bin/env bash
# Usage: create.sh [elegoo-sim-NAME]   (default elegoo-sim-1)
# Creates the VM in the first zone/shape with L4 capacity (see ZONES in _common.sh and
# MACHINE_TYPES below), then provisions it.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
pick_name "$@"

read -r -a MACHINE_TYPES <<< "${MACHINE_TYPES:-g2-standard-8 g2-standard-12}"
DISK_GB="${DISK_GB:-200}"
IMAGE_FAMILY="ubuntu-accelerator-2404-amd64-with-nvidia-595"
IMAGE_PROJECT="ubuntu-os-accelerator-images"

ZONE="$(zone_of "$NAME")"
if [ -n "$ZONE" ]; then
  echo "$NAME already exists in $ZONE; provisioning only"
else
  for z in "${ZONES[@]}"; do
   for mt in "${MACHINE_TYPES[@]}"; do
    echo "trying $z $mt ..."
    if gcloud compute instances create "$NAME" \
        --project "$PROJECT" --zone "$z" \
        --machine-type "$mt" \
        --maintenance-policy TERMINATE --restart-on-failure \
        --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
        --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
        --tags elegoo-sim \
        --labels purpose=elegoo-sim; then
      ZONE="$z"
      break 2
    fi
   done
  done
  [ -n "$ZONE" ] || die "no zone in ${ZONES[*]} had capacity for ${MACHINE_TYPES[*]}"
fi

echo "waiting for ssh on $NAME ($ZONE) ..."
for _ in $(seq 1 30); do
  if gssh --command "true" 2>/dev/null; then break; fi
  sleep 10
done

gssh --command "rm -rf ~/elegoo-sim-setup"
gcloud compute scp --project "$PROJECT" --zone "$ZONE" --quiet --recurse \
  "$CLOUD_DIR/remote" "$NAME":~/elegoo-sim-setup
gssh --command "bash ~/elegoo-sim-setup/setup_vm.sh '$ISAACLAB_IMAGE' '$ISAACSIM_IMAGE'"
echo "$NAME ready in $ZONE"
