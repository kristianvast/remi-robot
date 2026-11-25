#!/bin/sh
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SKETCH=$(basename "$REPO")
PORT=${PORT:-/dev/cu.usbserial-11210}
FQBN=arduino:avr:uno
TOOLCHAIN=${AVR_TOOLCHAIN:-$REPO/../toolchain/bin}
BUILD=${BUILD_DIR:-$REPO/../build}

case "$1" in
  build)
    arduino-cli compile --fqbn $FQBN --build-property compiler.path="$TOOLCHAIN/" --output-dir "$BUILD" "$REPO" ;;
  backup)
    avrdude -c arduino -p m328p -P "$PORT" -b 115200 -U flash:r:"backup-$(date +%Y%m%d-%H%M%S).hex":i ;;
  upload)
    avrdude -c arduino -p m328p -P "$PORT" -b 115200 -U flash:w:"$BUILD/$SKETCH.ino.hex":i ;;
  *)
    echo "usage: $0 build|backup|upload  (backup/upload need the shield switch on UPLOAD)" >&2; exit 1 ;;
esac
