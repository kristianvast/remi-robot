#!/usr/bin/env bash
# Usage: ssh.sh [elegoo-sim-NAME] [-- command ...]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
pick_name "$@"
shift "$SHIFT"
require_zone
if [ "${1:-}" = "--" ]; then
  shift
  gssh --command "$*"
else
  gssh
fi
