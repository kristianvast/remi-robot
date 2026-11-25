"""video.npz from evaluate.py -> mp4 per chosen episode (3 successes, 3 failures) and one contact sheet.

Panels: the RGB the driver got (after JPEG, latency), its encoded depth, and a top-down floor-map crop with the
planned route (grey), the driven trace (blue) and the trajectory the tracker was following (red).
  /opt/isaaclab-venv/bin/python videos.py out/eval_ft/video.npz out/eval_ft/videos
"""

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RES, X0, Y0 = 0.02, -11.49, 11.17
KINDS = {1: "tilt", 2: "stall", 4: "off map"}


def pick(success: np.ndarray, n: int = 3) -> list[int]:
    ok, bad = np.flatnonzero(success > 0), np.flatnonzero(success == 0)
    return list(ok[:: max(1, len(ok) // n)][:n]) + list(bad[:n])


def main(npz: str, out: str) -> None:
    with np.load(npz) as data:
        v = {k: data[k] for k in data.files}
    root = next(p for p in (HERE.parents[3], HERE.parents[1]) if (p / "scans/merged/floor_map.png").exists())
    floor = cv2.cvtColor(cv2.imread(str(root / "scans/merged/floor_map.png"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = []
    for j in pick(v["success"]):
        steps = max(2, min(int(v["steps"][j]) + 1, v["rgb"].shape[1]))
        way = v["way"][j][: int(v["count"][j])]
        pose = v["pose"][j][:steps]
        px = lambda xy: np.stack(((xy[..., 0] - X0) / RES, (Y0 - xy[..., 1]) / RES), -1).astype(np.int32)  # noqa: E731
        allpts = px(np.concatenate((way, pose[:, :2])))
        lo = np.maximum(allpts.min(0) - 40, 0)
        hi = allpts.max(0) + 40
        base = floor[lo[1]:hi[1], lo[0]:hi[0]].copy()
        local = lambda xy: px(xy) - lo  # noqa: E731
        cv2.polylines(base, [local(way)], False, (160, 160, 160), 2)
        status = "success" if v["success"][j] else f"FAIL {KINDS.get(int(v['kind'][j]), 'timeout')}"
        name = f"{'ok' if v['success'][j] else 'fail'}_{v['area'][j]}_route{int(v['route'][j])}"
        writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (336 * 3, 252))
        frames = []
        for t in range(steps):
            rgb = cv2.resize(cv2.cvtColor(v["rgb"][j, t], cv2.COLOR_RGB2BGR), (336, 252), interpolation=cv2.INTER_NEAREST)
            depth = cv2.applyColorMap(cv2.resize(v["depth"][j, t], (336, 252), interpolation=cv2.INTER_NEAREST),
                                      cv2.COLORMAP_TURBO)
            top = base.copy()
            cv2.polylines(top, [local(pose[: t + 1, :2])], False, (255, 80, 0), 2)
            x, y, yaw = pose[t]
            c, s = np.cos(yaw), np.sin(yaw)
            plan = v["plan"][j, t]
            world = np.stack((x + c * plan[:, 0] - s * plan[:, 1], y + s * plan[:, 0] + c * plan[:, 1]), -1)
            cv2.polylines(top, [local(world)], False, (0, 0, 255), 2)
            tip = local(np.array([x + 0.2 * c, y + 0.2 * s]))
            cv2.arrowedLine(top, tuple(int(v) for v in local(np.array([x, y]))), tuple(int(v) for v in tip), (0, 160, 0), 2,
                            tipLength=0.4)
            crop = cv2.resize(top, (336, 252))
            panel = np.concatenate((rgb, depth, crop), 1)
            cv2.putText(panel, f"{name} t={t / 10:.1f}s {status}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(panel)
            frames.append(panel)
        writer.release()
        sheet.append(np.concatenate([frames[int(k)] for k in np.linspace(0, len(frames) - 1, 3)], 1))
        print("VIDEO", name, steps, flush=True)
    cv2.imwrite(str(out_dir / "sheet.jpg"), cv2.resize(np.concatenate(sheet, 0), None, fx=0.5, fy=0.5),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main(*sys.argv[1:3])
