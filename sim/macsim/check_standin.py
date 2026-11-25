"""Drive the stand-in through the unmodified tools/brain/robot.py. From the project root:

    sim/macsim/.venv/bin/python -m sim.macsim.check_standin
"""

import json
import subprocess
import sys
import time
import urllib.request
from math import degrees, hypot, remainder, tau

from .simulator import Simulator
from .world import ROOT, scene_meta

sys.path.insert(0, str(ROOT / "car/tools/brain"))
from robot import Robot  # noqa: E402

PORT, CAM = 18190, 18191


def truth():
    with urllib.request.urlopen(f"http://127.0.0.1:{CAM}/test", timeout=2) as r:
        return json.load(r)


def open_spawn() -> int:
    sim = Simulator()
    for i, s in enumerate(scene_meta()["spawn_poses"]):
        sim.reset((s["x"], s["y"], s["yaw"]))
        if sim.observe(camera=False).range_cm >= 150:
            return i
    raise RuntimeError("no spawn with 150 cm free ahead")


def main():
    spawn = open_spawn()
    server = subprocess.Popen([sys.executable, "-m", "sim.macsim.standin", "--port", str(PORT),
                               "--camera-port", str(CAM), "--test-controls", "--spawn", str(spawn)],
                              cwd=ROOT, stdout=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                truth()
                break
            except OSError:
                time.sleep(.1)
        with Robot.wifi("127.0.0.1", PORT) as car:
            report = {"spawn": spawn, "heading_start": car.heading(), "range_cm": car.distance(), "on_ground": car.on_ground()}
            a = truth()
            end = time.monotonic() + 2
            while time.monotonic() < end:
                car.wheels(80, 80)
                time.sleep(1 / 14)
            time.sleep(.8)
            b = truth()
            report["straight_2s_m"] = round(hypot(b["x"] - a["x"], b["y"] - a["y"]), 3)
            end = time.monotonic() + 1
            while time.monotonic() < end:
                car.wheels(-80, 80)
                time.sleep(1 / 14)
            time.sleep(.8)
            c = truth()
            report["spin_1s_true_deg"] = round(degrees(remainder(c["yaw"] - b["yaw"], tau)), 1)
            report["heading_after_spin"] = car.heading()
            car.head(170)
            time.sleep(.6)
            report["head_true_deg"] = round(degrees(truth()["head"]), 1)
            report["range_left_cm"] = car.distance()
            car.head(90)
            with urllib.request.urlopen(f"http://127.0.0.1:{CAM}/capture", timeout=5) as r:
                jpeg = r.read()
            report["capture_bytes"] = len(jpeg)
            report["capture_is_jpeg"] = jpeg[:2] == b"\xff\xd8"
            (ROOT / "sim/macsim/standin_capture.jpg").write_bytes(jpeg)
        print(json.dumps(report, indent=2))
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
