# Shared settings and helpers for the elegoo-sim-* GCP scripts. Sourced, not executed.
# Every script refuses machine names that do not start with elegoo-sim- so it can never
# touch the unrelated production VMs in this project.

PROJECT="${ELEGOO_SIM_PROJECT:-your-gcp-project}"
# L4 zones tried in order by create.sh; us-central1 first, other US regions as stockout fallback.
read -r -a ZONES <<< "${ELEGOO_SIM_ZONES:-us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c}"
DEFAULT_NAME="elegoo-sim-1"
CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ISAACLAB_IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0"
ISAACSIM_IMAGE="nvcr.io/nvidia/isaac-sim:6.2.0"

die() { echo "error: $*" >&2; exit 1; }

check_name() {
  case "$1" in
    elegoo-sim-*) ;;
    *) die "refusing machine name '$1': only elegoo-sim-* machines may be managed" ;;
  esac
}

# pick_name "$@": if the first argument looks like a machine name, use it as NAME and
# set SHIFT=1 so the caller can drop it; otherwise NAME defaults to elegoo-sim-1.
pick_name() {
  SHIFT=0
  NAME="$DEFAULT_NAME"
  if [ $# -gt 0 ] && [[ "$1" == elegoo-sim-* ]]; then
    NAME="$1"
    SHIFT=1
  fi
  check_name "$NAME"
}

zone_of() {
  gcloud compute instances list --project "$PROJECT" \
    --filter="name=($1)" --format="value(zone.basename())" 2>/dev/null | head -n1
}

require_zone() {
  ZONE="$(zone_of "$NAME")"
  [ -n "$ZONE" ] || die "instance $NAME not found in project $PROJECT"
}

status_of() {
  gcloud compute instances describe "$NAME" --zone "$ZONE" --project "$PROJECT" --format="value(status)"
}

gssh() {
  gcloud compute ssh "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet "$@"
}
