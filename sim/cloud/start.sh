#!/usr/bin/env bash
# Usage: start.sh [elegoo-sim-NAME]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
pick_name "$@"
require_zone
gcloud compute instances start "$NAME" --zone "$ZONE" --project "$PROJECT"
status_of
