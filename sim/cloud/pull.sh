#!/usr/bin/env bash
# Usage: pull.sh [elegoo-sim-NAME] <remote_path> [local_folder]
# remote_path is relative to ~/elegoo-sim/work on the VM (= /workspace/elegoo in the
# container) unless it is absolute. local_folder defaults to the current directory.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
pick_name "$@"
shift "$SHIFT"
[ $# -ge 1 ] && [ $# -le 2 ] || die "usage: pull.sh [elegoo-sim-NAME] <remote_path> [local_folder]"
REMOTE="$1"
LOCAL="${2:-.}"
case "$REMOTE" in
  /*) ;;
  *) REMOTE="~/elegoo-sim/work/$REMOTE" ;;
esac
require_zone
mkdir -p "$LOCAL"
gcloud compute scp --project "$PROJECT" --zone "$ZONE" --quiet --recurse \
  "$NAME":"$REMOTE" "$LOCAL"
