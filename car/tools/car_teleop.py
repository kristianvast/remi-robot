#!/usr/bin/env python3
"""Keyboard teleop for the ELEGOO Smart Robot Car V4.0 (firmware 2.1.2) over USB serial.

Keys:  w/s/a/d = forward/back/left/right   q/e/z/c = diagonals   space = stop
       +/- = speed   1 = line tracking   2 = obstacle avoid   3 = follow   x = quit
Usage: car_teleop.py [port]   (default: first /dev/ttyUSB* or /dev/ttyACM*)
"""
import curses
import glob
import sys
import time

import serial

DIRS = {"w": 1, "s": 2, "a": 3, "d": 4, "q": 5, "z": 6, "e": 7, "c": 8}
STOP = 9


def find_port():
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        sys.exit("No /dev/ttyUSB* or /dev/ttyACM* found - is the car plugged in over USB?")
    return ports[0]


def main(scr, ser):
    curses.curs_set(0)
    scr.nodelay(True)
    speed = 150
    status = "ready"
    seq = 0

    def send(n, d1, d2=None):
        nonlocal seq
        seq += 1
        msg = '{"N":%d,"H":"%d","D1":%d' % (n, seq, d1)
        if d2 is not None:
            msg += ',"D2":%d' % d2
        ser.write((msg + "}").encode())

    while True:
        scr.erase()
        scr.addstr(0, 0, __doc__.split("Usage")[0])
        scr.addstr(6, 0, f"port={ser.port}  speed={speed}  last={status}")
        while ser.in_waiting:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                status = f"{status} | car: {line}"[-70:]
        scr.refresh()

        key = scr.getch()
        if key == -1:
            time.sleep(0.03)
            continue
        ch = chr(key) if 0 <= key < 256 else ""
        if ch == "x":
            send(102, STOP, 0)
            return
        if ch in DIRS:
            send(102, DIRS[ch], speed)
            status = f"drive {ch}"
        elif ch == " ":
            send(102, STOP, 0)
            status = "stop"
        elif ch in "+=":
            speed = min(255, speed + 10)
        elif ch == "-":
            speed = max(0, speed - 10)
        elif ch in "123":
            send(101, int(ch))
            status = ["", "line tracking", "obstacle avoidance", "follow"][int(ch)]


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else find_port()
    with serial.Serial(port, 9600, timeout=0.1) as ser:
        time.sleep(2)  # opening the port resets the UNO; wait for boot
        curses.wrapper(main, ser)
