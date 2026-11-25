import os
import re
import select
import shutil
import sys
import termios
import time
import tty

KEYS = {"w": "forward", "s": "back", "a": "left", "d": "right"}
HELP = "w/a/s/d drive (hold)  space stop  +/- speed  mouse left/right pans camera, click centres  q quit"
MOUSE_ON = "\x1b[?1003h\x1b[?1006h"
MOUSE_OFF = "\x1b[?1003l\x1b[?1006l"
MOUSE_EVENT = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")
HEAD_STEP = 5
HEAD_INTERVAL = 0.1


def run(robot, speed=120, release_after=0.6):
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    moving = None
    last_key = 0.0
    head = 90
    head_target = 90
    head_sent_at = 0.0
    buf = b""
    print(HELP)
    print(f"speed {speed}")
    try:
        tty.setcbreak(fd)
        sys.stdout.write(MOUSE_ON)
        sys.stdout.flush()
        while True:
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                buf += os.read(fd, 1024)
            while buf:
                m = MOUSE_EVENT.match(buf)
                if m:
                    buf = buf[m.end():]
                    button, col = int(m.group(1)), int(m.group(2))
                    if button == 0 and m.group(4) == b"M":
                        head_target = 90
                    elif button & 32:
                        width = max(2, shutil.get_terminal_size().columns)
                        head_target = round(180 - (col - 1) * 180 / (width - 1))
                    continue
                if buf.startswith(b"\x1b"):
                    if len(buf) < 12 and b"M" not in buf and b"m" not in buf:
                        break
                    buf = buf[1:]
                    continue
                key = chr(buf[0]).lower()
                buf = buf[1:]
                if key == "q":
                    return
                if key in KEYS:
                    last_key = time.monotonic()
                    if moving != KEYS[key]:
                        moving = KEYS[key]
                        robot.drive(moving, speed)
                elif key == " ":
                    robot.stop()
                    moving = None
                elif key in "+=-":
                    speed = max(60, min(255, speed + (20 if key in "+=" else -20)))
                    print(f"speed {speed}")
                    if moving:
                        robot.drive(moving, speed)
            now = time.monotonic()
            if moving and now - last_key > release_after:
                robot.stop()
                moving = None
            if abs(head_target - head) >= HEAD_STEP and now - head_sent_at > HEAD_INTERVAL:
                head = max(0, min(180, head_target))
                robot.head(head)
                head_sent_at = now
    finally:
        sys.stdout.write(MOUSE_OFF)
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        robot.stop()
