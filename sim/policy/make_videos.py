"""video.npz (evaluate.py) -> MP4s: the policy's depth input (left) and a top-down path (right).

  python make_videos.py <video.npz> <out_dir> [free_map.npz]   (run on elegoo-sim-1; needs ffmpeg, PIL, matplotlib)
Left: encoded corrupted depth exactly as the policy saw it (turbo, bright = near). Right: free floor
(white), planned route (blue), true path so far (red), car (black dot + heading), goal (green).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

FREE_MAP = Path(__file__).resolve().parents[1] / "isaac/elegoo/nav/free_map.npz"
KINDS = {1: "tilt", 2: "stall", 3: "tilt+stall", 4: "off map", 5: "tilt+off map", 6: "stall+off map", 7: "all"}
H = 360


def floor(path: Path) -> tuple[np.ndarray, float, float, float]:
    data = np.load(path)
    shape = tuple(data["shape"])
    allowed = np.unpackbits(data["allowed"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    return allowed, float(data["x0"]), float(data["y0"]), float(data["res"])


def render(path: Path, depth, pose, way, steps: int, title: str, allowed, x0, y0, res) -> None:
    pts = np.concatenate((way, pose[:, :2]))
    lo, hi = pts.min(0) - 0.8, pts.max(0) + 0.8
    span = max(hi - lo)
    lo, hi = (lo + hi) / 2 - span / 2, (lo + hi) / 2 + span / 2
    scale = H / span
    c0, c1 = int((lo[0] - x0) / res), int((hi[0] - x0) / res)
    r0, r1 = int((y0 - hi[1]) / res), int((y0 - lo[1]) / res)
    crop = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    rr, cc = np.clip(np.arange(r0, r1), 0, allowed.shape[0] - 1), np.clip(np.arange(c0, c1), 0, allowed.shape[1] - 1)
    crop[:] = allowed[np.ix_(rr, cc)]
    crop[(np.arange(r0, r1) < 0) | (np.arange(r0, r1) >= allowed.shape[0])] = False
    background = Image.fromarray(np.where(crop, 255, 150).astype(np.uint8)).convert("RGB").resize((H, H))
    to_px = lambda xy: [((x - lo[0]) * scale, (hi[1] - y) * scale) for x, y in xy]
    draw = ImageDraw.Draw(background)
    draw.line(to_px(way), fill=(40, 90, 220), width=3)
    gx, gy = to_px(way[-1:])[0]
    draw.ellipse((gx - 6, gy - 6, gx + 6, gy + 6), outline=(0, 160, 0), width=3)
    turbo = colormaps["turbo"]
    steps = min(steps + 1, len(depth))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{480 + H}x{H + 30}",
         "-r", f"{1 / 0.07:.4f}", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(path)],
        stdin=subprocess.PIPE)
    for t in range(steps):
        left = Image.fromarray((turbo(depth[t] / 255.0)[..., :3] * 255).astype(np.uint8)).resize((480, H), Image.NEAREST)
        right = background.copy()
        d = ImageDraw.Draw(right)
        trail = to_px(pose[: t + 1, :2])
        if len(trail) > 1:
            d.line(trail, fill=(220, 30, 30), width=2)
        x, y = trail[-1]
        heading = float(pose[t, 2])
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 0))
        d.line((x, y, x + 14 * np.cos(heading), y - 14 * np.sin(heading)), fill=(0, 0, 0), width=2)
        frame = Image.new("RGB", (480 + H, H + 30), (255, 255, 255))
        frame.paste(left, (0, 30))
        frame.paste(right, (480, 30))
        ImageDraw.Draw(frame).text((6, 8), f"{title}   t={t * 0.07:5.1f}s", fill=(0, 0, 0))
        ffmpeg.stdin.write(frame.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


def main(source: str, target: str, free_map: str = str(FREE_MAP)) -> None:
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    allowed, x0, y0, res = floor(Path(free_map))
    data = np.load(source)
    for i, car in enumerate(data["cars"]):
        ok = bool(data["success"][i])
        label = "success" if ok else f"fail_{KINDS.get(int(data['kind'][i]), 'timeout').replace(' ', '_').replace('+', '_')}"
        name = f"{'S' if ok else 'F'}{i:02d}_{data['area'][i]}_route{int(data['route'][i])}_{label}.mp4"
        title = f"{data['area'][i]} route {int(data['route'][i])}: {label.replace('_', ' ')}"
        render(out / name, data["depth"][i], data["pose"][i], data["way"][i], int(data["steps"][i]), title,
               allowed, x0, y0, res)
        print("wrote", out / name)


if __name__ == "__main__":
    main(*sys.argv[1:4])
