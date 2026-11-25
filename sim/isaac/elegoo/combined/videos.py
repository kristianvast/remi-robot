"""Success / failure videos from evaluate.py's record.npz (container python with cv2, no Kit).

Panels: what the student received (JPEG-degraded RGB, frozen on dropped frames), the model's DA2 depth, and a map
with the planned route, the true track, the place head's (x, y) estimate and room guess.

  /opt/isaaclab-venv/bin/python p4c/sim/isaac/elegoo/combined/videos.py p4c/eval_student/record.npz p4c/videos
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[4] / "pylib"))
from elegoo.combined.common import AREAS

KINDS = {1: "tilt", 2: "stall", 4: "off map"}
PANEL = (360, 480)


def map_background(labels, way, pose, size=PANEL):
    """Room-coloured map crop around the route and track, and its world -> pixel projection (once per video)."""
    x0, y0, res = float(labels["x0"]), float(labels["y0"]), float(labels["res"])
    pts = np.concatenate((way, pose[:, :2]))
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
    span = max(hi[0] - lo[0], (hi[1] - lo[1]) * size[1] / size[0])
    hi = lo + np.array([span, span * size[0] / size[1]])
    c0, c1 = int((lo[0] - x0) / res), int((hi[0] - x0) / res)
    r0, r1 = int((y0 - hi[1]) / res), int((y0 - lo[1]) / res)
    area = labels["area"]
    crop = area[max(r0, 0):r1, max(c0, 0):c1]
    colours = (np.array([[(37 * i) % 255, (91 * i + 60) % 255, (151 * i + 90) % 255] for i in range(10)]) * 0.35 + 40)
    img = cv2.resize(colours[crop].astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    to_px = lambda p: (int((p[0] - lo[0]) / span * size[1]), int((hi[1] - p[1]) / (hi[1] - lo[1]) * size[0]))
    for a, b in zip(way[:-1], way[1:]):
        cv2.line(img, to_px(a), to_px(b), (200, 200, 200), 1)
    return img, to_px, span


def map_panel(background, pose, est, t):
    img, to_px, span = background
    img = img.copy()
    for a, b in zip(pose[: t][:-1, :2], pose[: t][1:, :2]):
        cv2.line(img, to_px(a), to_px(b), (60, 220, 60), 2)
    p = pose[t]
    tip = p[:2] + 0.3 * np.array([np.cos(p[2]), np.sin(p[2])])
    cv2.arrowedLine(img, to_px(p[:2]), to_px(tip), (60, 255, 60), 2, tipLength=0.4)
    cv2.circle(img, to_px(est[:2]), max(3, int(est[3] / span * PANEL[1])), (40, 120, 255), 1)
    cv2.circle(img, to_px(est[:2]), 3, (40, 120, 255), -1)
    return img


def main(record: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with np.load(record) as f:
        r = {k: f[k] for k in f.files}
    with np.load(Path(__file__).with_name("labels.npz")) as f:
        labels = {k: f[k] for k in ("area", "x0", "y0", "res")}
    success = r["success"] > 0
    ok, bad = np.flatnonzero(success), np.flatnonzero(~success)
    pick = [("S", i) for i in ok[:: max(1, len(ok) // 3)][:3]] + [("F", i) for i in bad[:3]]
    for tag, i in pick:
        steps = int(np.argmax(r["done"][i])) if r["done"][i].any() else r["done"].shape[1]
        steps = max(2, steps)
        name = f"{tag}_{r['area'][i]}_route{int(r['route'][i])}"
        if tag == "F":
            name += "_" + KINDS.get(int(r["kind"][i]), "timeout").replace(" ", "")
        writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (PANEL[1] * 3, PANEL[0]))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter cannot write mp4v here ({name})")
        pose = r["pose"][i].astype(np.float32)
        background = map_background(labels, r["way"][i], pose[:steps])
        sheet = []
        for t in range(steps):
            rgb = cv2.resize(r["rgb"][i, t].transpose(1, 2, 0), (PANEL[1], PANEL[0]), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = cv2.applyColorMap(cv2.resize(r["depth"][i, t], (PANEL[1], PANEL[0])), cv2.COLORMAP_TURBO)
            est = r["pose_hat"][i, t]
            room = AREAS[int(np.argmax(r["room_p"][i, t]))]
            m = map_panel(background, pose, est, t)
            frame = np.concatenate((rgb, depth, m), 1)
            truth = AREAS[int(labels["area"][int((float(labels["y0"]) - r["pose"][i, t, 1]) / float(labels["res"])),
                                             int((r["pose"][i, t, 0] - float(labels["x0"])) / float(labels["res"]))])]
            text = (f"{name} t={t / 10:.1f}s pwm={tuple(int(v) for v in r['pwm'][i, t])} "
                    f"{'frame' if r['delivered'][i, t] else 'DROPPED'} room {room} ({float(np.max(r['room_p'][i, t])):.2f}) true {truth}")
            cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            if t % max(1, steps // 8) == 0 and len(sheet) < 8:
                sheet.append(cv2.resize(frame, (PANEL[1] * 3 // 2, PANEL[0] // 2)))
        frame_last = cv2.resize(frame, (PANEL[1] * 3 // 2, PANEL[0] // 2))
        sheet = (sheet + [frame_last])[:9]
        cv2.imwrite(str(out / f"{name}_sheet.jpg"), np.concatenate(sheet, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
        writer.release()
        print("VIDEO", name, steps, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
