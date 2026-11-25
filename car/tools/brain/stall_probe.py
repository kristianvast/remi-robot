"""Record the stall detector's picture signals from a camera, to set stall.py's thresholds from real numbers.

    .venv/bin/python stall_probe.py CAPTURE_URL [--stream STREAM_URL] [--out stall_probe.csv]

It drives nothing and opens no car link: it only reads the camera URLs given here, through the same Vision
reader the agent uses, so stop anything else that reads the camera first. Hold the car still, push it by
hand, drive it from the pane; type a word and Enter (e.g. "push", "still", "wall") to put a mark in the log.
Every new frame becomes one CSV row; a running line shows the latest values and the picture verdict.
Range and gyro come over the car link and are not recorded here; the detector also uses them (see stall.py).
"""

import argparse
import csv
import sys
import threading
import time
from collections import deque

from stall import (
    DARK_MIN,
    STALL_FRAME_AGE_S,
    STALL_MIN_FRAMES,
    STALL_WINDOW_S,
    STILL_SCORE,
    TEXTURE_MIN,
    change,
    picture,
    texture,
)
from vision import Vision

FIELDS = ("wall_time", "frame_time", "jpeg_bytes", "fps", "frames_in_window", "span_s", "light", "texture",
          "score", "step_score", "verdict", "mark")


def verdict(frames, span, light, detail, score):
    if frames < STALL_MIN_FRAMES or span < STALL_WINDOW_S:
        return "warming"
    if light < DARK_MIN:
        return "dark"
    if detail < TEXTURE_MIN:
        return "blank"
    return "still" if score < STILL_SCORE else "moving"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture_url")
    ap.add_argument("--stream", help="MJPEG stream URL; faster than stills if the camera serves one")
    ap.add_argument("--out", default="stall_probe.csv")
    args = ap.parse_args()
    vision = Vision(args.capture_url, args.stream)
    vision.start()
    mark = [""]

    def read_marks():
        for line in sys.stdin:
            mark[0] = line.strip()

    threading.Thread(target=read_marks, daemon=True).start()
    window = deque()  # (frame time, grey picture)
    newest = 0.0
    arrivals = deque(maxlen=12)
    with open(args.out, "w", newline="") as handle:
        log = csv.writer(handle)
        log.writerow(FIELDS)
        print(f"logging to {args.out}; Ctrl-C to stop", file=sys.stderr)
        try:
            while True:
                frame = vision.newest(STALL_FRAME_AGE_S)
                if frame is None:
                    print("\rno camera frame                                                     ", end="", flush=True)
                    window.clear()
                    time.sleep(0.05)
                    continue
                if frame.timestamp <= newest:
                    time.sleep(0.01)
                    continue
                newest = frame.timestamp
                arrivals.append(frame.timestamp)
                grey = picture(frame.jpeg)
                step = change(window[-1][1], grey) / max(texture(window[-1][1]), 1e-6) if window else 0.0
                window.append((frame.timestamp, grey))
                while len(window) > 1 and window[1][0] <= frame.timestamp - STALL_WINDOW_S:
                    window.popleft()
                since, before = window[0]
                detail, light = texture(before), float(before.mean())
                score = change(before, grey) / max(detail, 1e-6)
                fps = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.0
                span = frame.timestamp - since
                said = verdict(len(window), span, light, detail, score)
                log.writerow((f"{time.time():.3f}", f"{frame.timestamp:.3f}", len(frame.jpeg), f"{fps:.1f}", len(window),
                              f"{span:.2f}", f"{light:.1f}", f"{detail:.2f}", f"{score:.3f}", f"{step:.3f}", said, mark[0]))
                handle.flush()
                print(f"\r{fps:4.1f} fps  score {score:6.3f} (still < {STILL_SCORE})  step {step:6.3f}  "
                      f"texture {detail:5.2f}  light {light:5.1f}  {said:8s} {mark[0][:12]:12s}", end="", flush=True)
                mark[0] = ""
        except KeyboardInterrupt:
            print()
        finally:
            vision.close()


if __name__ == "__main__":
    main()
