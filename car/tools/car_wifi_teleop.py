#!/usr/bin/env python3
"""Keyboard teleop for the ELEGOO Smart Robot Car V4.0 over WiFi (camera module TCP port 100).

Keys:  w/s/a/d = forward/back/left/right   q/e/z/c = diagonals   space = stop
       +/- = speed   1 = line tracking   2 = obstacle avoid   3 = follow   x = quit
Usage: car_wifi_teleop.py [ip]   (default elegoo-car.local; camera: http://<ip>/)
"""
import curses
import socket
import sys
import threading
import time

DIRS = {"w": 1, "s": 2, "a": 3, "d": 4, "q": 5, "z": 6, "e": 7, "c": 8}
STOP = 9


def main(scr, sock, ip):
    curses.curs_set(0)
    scr.nodelay(True)
    speed = 150
    status = "ready"
    last_drive = None
    lock = threading.Lock()
    alive = True

    def send(text):
        with lock:
            sock.sendall(text.encode())

    def heartbeat():
        while alive:
            try:
                send("{Heartbeat}")
            except OSError:
                return
            time.sleep(1)

    threading.Thread(target=heartbeat, daemon=True).start()
    sock.settimeout(0.01)

    while True:
        scr.erase()
        scr.addstr(0, 0, __doc__.split("Usage")[0])
        scr.addstr(6, 0, f"car={ip}  speed={speed}  last={status}"[:78])
        scr.refresh()
        try:
            data = sock.recv(512).decode(errors="replace").replace("{Heartbeat}", "")
            if data:
                status = f"{status} | car: {data}"[-60:]
        except socket.timeout:
            pass

        key = scr.getch()
        if key == -1:
            time.sleep(0.03)
            continue
        ch = chr(key) if 0 <= key < 256 else ""
        if ch == "x":
            send('{"N":102,"D1":%d,"D2":0}' % STOP)
            alive = False
            return
        if ch in DIRS:
            cmd = '{"N":102,"D1":%d,"D2":%d}' % (DIRS[ch], speed)
            if cmd != last_drive:
                send(cmd)
                last_drive = cmd
            status = f"drive {ch}"
        elif ch == " ":
            send('{"N":102,"D1":%d,"D2":0}' % STOP)
            last_drive = None
            status = "stop"
        elif ch in "+=":
            speed = min(255, speed + 10)
        elif ch == "-":
            speed = max(0, speed - 10)
        elif ch in "123":
            send('{"N":101,"D1":%s}' % ch)
            status = ["", "line tracking", "obstacle avoidance", "follow"][int(ch)]


if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else "elegoo-car.local"
    with socket.create_connection((ip, 100), timeout=5) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        curses.wrapper(main, sock, ip)
