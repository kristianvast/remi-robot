"""Calibrate the car camera from photos of the A4 checkerboard (9 x 6 inner corners, 25 mm squares).

    capture:  sim/realdepth/.venv/bin/python sim/realcar/calibrate.py capture --url http://elegoo-car.local/capture
              Saves a picture each time you press Enter while the board is visible (q + Enter to finish).
    solve:    sim/realdepth/.venv/bin/python sim/realcar/calibrate.py solve
              Fits the lens model and writes sim/realcar/camera.json.
    selftest: sim/realdepth/.venv/bin/python sim/realcar/calibrate.py selftest
              Renders the board through a known camera and checks that solve recovers it.
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "calib_shots"
CORNERS = (9, 6)
SQUARE_M = 0.025


def board_points():
    grid = np.zeros((CORNERS[0] * CORNERS[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:CORNERS[0], 0:CORNERS[1]].T.reshape(-1, 2) * SQUARE_M
    return grid


def find(gray):
    ok, corners = cv2.findChessboardCornersSB(gray, CORNERS, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    return corners if ok else None


def capture(url, out=SHOTS):
    out.mkdir(parents=True, exist_ok=True)
    kept = len(list(out.glob("*.jpg")))
    print("Hold the board in front of the camera, Enter to take a picture, q + Enter to stop.")
    while input(f"[{kept} kept] > ").strip().lower() != "q":
        jpeg = urllib.request.urlopen(url, timeout=5).read()
        gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if find(gray) is None:
            print("  board not found: move it so all of it is in view and well lit")
            continue
        path = out / f"shot_{int(time.time() * 1000)}.jpg"
        path.write_bytes(jpeg)
        kept += 1
        print(f"  kept {path.name}; vary distance, tilt and position in the picture")


def solve(shots=SHOTS, out=HERE / "camera.json"):
    objp, imgp, size = [], [], None
    for path in sorted(shots.glob("*.jpg")):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        corners = find(gray)
        if corners is not None:
            objp.append(board_points())
            imgp.append(corners)
    if len(objp) < 8:
        sys.exit(f"only {len(objp)} usable pictures in {shots}; take at least 8, better 15")
    rms, K, dist, _, _ = cv2.calibrateCamera(objp, imgp, size, None, None)
    hfov = math.degrees(2 * math.atan(size[0] / (2 * K[0, 0])))
    vfov = math.degrees(2 * math.atan(size[1] / (2 * K[1, 1])))
    result = {"width": size[0], "height": size[1], "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
              "dist": dist.ravel().tolist(), "hfov_deg": hfov, "vfov_deg": vfov, "rms_px": rms, "pictures": len(objp)}
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"reprojection error {rms:.2f} px (under 0.5 is good, over 1 means retake); horizontal field of view {hfov:.1f} deg")
    return result


def selftest():
    w, h, fx = 640, 480, 520.0
    K = np.array([[fx, 0, 322], [0, fx, 238], [0, 0, 1]])
    dist = np.array([-0.25, 0.08, 0, 0, 0])
    scale = 2000
    board = np.full(((CORNERS[1] + 3) * 100, (CORNERS[0] + 3) * 100), 255, np.uint8)
    for r in range(CORNERS[1] + 1):
        for c in range(CORNERS[0] + 1):
            if (r + c) % 2 == 0:
                board[100 + r * 100:200 + r * 100, 100 + c * 100:200 + c * 100] = 0
    shots = HERE / "selftest_shots"
    shots.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(15):
        rvec = rng.uniform([-0.5, -0.5, -0.3], [0.5, 0.5, 0.3])
        tvec = np.array([rng.uniform(-0.06, -0.02), rng.uniform(-0.05, -0.02), rng.uniform(0.28, 0.45)])
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        und = cv2.undistortPoints(np.stack([xs.ravel(), ys.ravel()], 1)[:, None], K, dist).reshape(-1, 2)
        R, _ = cv2.Rodrigues(rvec)
        rays = np.column_stack([und, np.ones(len(und))]) @ R
        origin = -R.T @ tvec
        t = -origin[2] / rays[:, 2]
        hit = origin + rays * t[:, None]
        u = (hit[:, 0] / SQUARE_M * 100 + 200).reshape(h, w).astype(np.float32)
        v = (hit[:, 1] / SQUARE_M * 100 + 200).reshape(h, w).astype(np.float32)
        img = cv2.remap(board, u, v, cv2.INTER_LINEAR, borderValue=200)
        cv2.imwrite(str(shots / f"shot_{i:02d}.jpg"), cv2.GaussianBlur(img, (3, 3), 0))
    got = solve(shots, HERE / "selftest_camera.json")
    err = abs(got["fx"] - fx) / fx
    print(f"SELFTEST fx true {fx:.1f} solved {got['fx']:.1f} ({100 * err:.1f}% off); k1 true -0.25 solved {got['dist'][0]:.3f}")
    print("SELFTEST_OK" if err < 0.02 and abs(got["dist"][0] + 0.25) < 0.05 else "SELFTEST_FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["capture", "solve", "selftest"])
    ap.add_argument("--url", default="http://elegoo-car.local/capture")
    args = ap.parse_args()
    {"capture": lambda: capture(args.url), "solve": solve, "selftest": selftest}[args.mode]()
