#!/usr/bin/env bash
# Usage: push.sh [elegoo-sim-NAME] <local_folder>
# Copies the folder to ~/elegoo-sim/work/<folder_name> on the VM, which run_headless.sh
# mounts into the container as /workspace/elegoo/<folder_name>.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
pick_name "$@"
shift "$SHIFT"
[ $# -eq 1 ] || die "usage: push.sh [elegoo-sim-NAME] <local_folder>"
SRC="${1%/}"
[ -d "$SRC" ] || die "not a folder: $SRC"
require_zone
DEST_NAME="$(basename "$SRC")"
gssh --command "mkdir -p ~/elegoo-sim/work && rm -rf ~/elegoo-sim/work/$DEST_NAME"
gcloud compute scp --project "$PROJECT" --zone "$ZONE" --quiet --recurse \
  "$SRC" "$NAME":~/elegoo-sim/work/"$DEST_NAME"
gssh --command "chmod -R a+rwX ~/elegoo-sim/work/$DEST_NAME"
echo "pushed $SRC -> $NAME:~/elegoo-sim/work/$DEST_NAME (container: /workspace/elegoo/$DEST_NAME)"
