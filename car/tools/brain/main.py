import argparse
import pathlib
import sys
import threading
import time
import urllib.request

import behaviors
import drive
import teleop
from robot import DEFAULT_HOST, DEFAULT_PORT, Robot


def take_photos(host, folder, every, stop):
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    while not stop.wait(every):
        try:
            data = urllib.request.urlopen(f"http://{host}/capture", timeout=5).read()
        except OSError as e:
            print(f"photo failed: {e}")
            continue
        n += 1
        (folder / f"{n:03d}-{time.strftime('%H%M%S')}.jpg").write_bytes(data)


def main():
    p = argparse.ArgumentParser(description="Mac-side brain for the ELEGOO Smart Robot Car V4.0")
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--wifi", action="store_true", help="control over the car's Wi-Fi instead of USB")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--bench", action="store_true", help="allow moving while the car is lifted")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sensors")
    sub.add_parser("scan")
    sub.add_parser("calibrate", help="on the floor, facing a wall 110 to 140 cm away: check turn direction, measure speed")
    sub.add_parser("keys").add_argument("--speed", type=int, default=120)
    for name in ("patrol", "follow"):
        sp = sub.add_parser(name)
        sp.add_argument("--seconds", type=float, default=30)
        sp.add_argument("--photos", type=pathlib.Path, help="save a camera photo here every --every seconds (Wi-Fi only)")
        sp.add_argument("--every", type=float, default=2.0)
    args = p.parse_args()

    with (Robot.wifi(args.host) if args.wifi else Robot.usb(args.port)) as robot:
        if args.cmd == "sensors":
            print(f"distance {robot.distance()} cm, line L/M/R {robot.line()}, on ground {robot.on_ground()}")
            return
        if args.cmd == "scan":
            for angle, cm in behaviors.scan(robot):
                print(f"head {angle:3d} deg  {cm:3d} cm  {'#' * (cm // 5)}")
            return
        if args.cmd == "calibrate":
            drive.calibrate(robot)
            return
        if args.cmd == "keys":
            teleop.run(robot, args.speed)
            return
        if not robot.on_ground() and not args.bench:
            sys.exit("car is lifted; put it on the floor or pass --bench")
        if args.photos and not args.wifi:
            sys.exit("--photos needs --wifi")
        stop = threading.Event()
        if args.photos:
            threading.Thread(target=take_photos, args=(args.host, args.photos, args.every, stop), daemon=True).start()
        try:
            getattr(behaviors, args.cmd)(robot, args.seconds)
        except KeyboardInterrupt:
            print("stopped")
        finally:
            stop.set()


if __name__ == "__main__":
    main()
