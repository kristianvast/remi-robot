"""Make car-like 640x480 frames from the walk-through film (NOT car frames).

Film: iPhone 13 Pro Max, 1920x1080, COLMAP OPENCV intrinsics fx=1592.6 fy=1601.9.
Undistort, centre-crop to 4:3 (1440x1080), resize to 640x480. The result has
HFOV 48.6 deg (fx = 708.0 px), not the car's ~65 deg: the film is too narrow.
"""
import glob, json, os, sys
import cv2
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "film640")
cam = json.load(open(R + "/scans/film/work/film_cams_metric.json"))["camera"]["params"]
fx, fy, cx, cy, k1, k2, p1, p2 = cam
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# new camera: square pixels, fx used for both axes
Kn = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
maps = cv2.initUndistortRectifyMap(K, np.array([k1, k2, p1, p2]), None, Kn, (1920, 1080), cv2.CV_32FC1)
FX640 = fx * 640 / 1440


def make(path):
    im = cv2.imread(path)
    u = cv2.remap(im, *maps, cv2.INTER_LINEAR)
    c = u[:, 240:1680]
    return cv2.resize(c, (640, 480), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:]
    for n in names:
        cv2.imwrite(os.path.join(OUT, n), make(R + "/scans/film/frames/" + n), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", len(names), "fx640", round(FX640, 1), "hfov", round(2 * np.degrees(np.arctan(320 / FX640)), 1))
