"""Trip plots from evaluate_nav's runs.pkl, plus one MP4 (and a still strip) of depth beside the map.

    sim/macsim/.venv/bin/python -m sim.navigate.plots
"""

import pickle
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.macsim.world import floor

from .evaluate_nav import RUNS, row
from .places import resolve
from .trip import run_trip

OUT = Path(__file__).parent / "plots"


def background(ax, focus: np.ndarray | None = None):
    f = floor()
    h, w = f.image.shape
    extent = (f.x0, f.x0 + w * f.res, f.ytop - h * f.res, f.ytop)
    ax.imshow(f.image, cmap="gray", extent=extent, vmin=0, vmax=255)
    if focus is not None:
        lo, hi = focus.min(axis=0) - 1.5, focus.max(axis=0) + 1.5
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")


def trip_plot(case, trip, extra, r, title, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    pts = [trip.truth[:, :2], np.array([case.goal_xy])]
    if trip.planned is not None and len(trip.planned):
        pts.append(trip.planned)
    background(ax, np.vstack(pts))
    if trip.planned is not None and len(trip.planned):
        ax.plot(*trip.planned.T, "-", color="tab:green", lw=2, alpha=.7, label="planned (first confident estimate)")
    ax.plot(*case.planned.T, ":", color="tab:olive", lw=1.5, label="shortest path from the true start")
    sc = ax.scatter(*trip.est[:, :2].T, c=trip.conf, cmap="viridis", vmin=0, vmax=1, s=4, label="estimate (colour = confidence)")
    ax.plot(*trip.truth[:, :2].T, "-", color="tab:red", lw=1.5, label="true path")
    ax.plot(*case.start[:2], "o", color="tab:red", ms=9, mfc="none", mew=2, label="start")
    ax.plot(*case.goal_xy, "*", color="gold", mec="k", ms=16, label="goal")
    ax.plot(*trip.truth[-1, :2], "x", color="tab:red", ms=10, mew=2, label="true end")
    if extra.get("kidnap_to"):
        ax.annotate("", xy=extra["kidnap_to"][:2], xytext=extra["kidnap_from"],
                    arrowprops=dict(arrowstyle="->", color="magenta", lw=2))
        ax.plot([], [], color="magenta", label="kidnap (car carried)")
    fig.colorbar(sc, ax=ax, fraction=.03, label="confidence")
    ax.set_title(f"{title}\n{case.experiment} #{case.index}: {case.start_area} -> {case.goal} | {r['status']} "
                 f"{r['reason']} | true error {r['true_err_m']} m | {r['time_s']} s | contacts {r['contacts']}",
                 fontsize=9)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pick(results):
    rows = [row(*res) for res in results]
    chosen = []

    def first(exp, ok, key):
        sel = [(res, r) for res, r in zip(results, rows) if r["experiment"] == exp and bool(r["success"]) == ok]
        sel.sort(key=key)
        return sel[0] if sel else None

    spec = [("good_room_prior", "room_prior", True, lambda x: -x[1]["planned_m"]),
            ("good_cold_start", "cold_start", True, lambda x: -x[1]["planned_m"]),
            ("good_kidnap", "kidnap", True, lambda x: -x[1]["planned_m"]),
            ("good_furniture_mismatch", "furniture_mismatch", True, lambda x: -x[1]["planned_m"]),
            ("bad_room_prior", "room_prior", False, lambda x: -x[1]["true_err_m"]),
            ("bad_cold_start", "cold_start", False, lambda x: -x[1]["true_err_m"]),
            ("bad_kidnap", "kidnap", False, lambda x: -x[1]["true_err_m"]),
            ("bad_furniture_mismatch", "furniture_mismatch", False, lambda x: -x[1]["true_err_m"])]
    for name, exp, ok, key in spec:
        got = first(exp, ok, key)
        if got:
            chosen.append((name, *got))
    return chosen


def video(seed: int = 7):
    """Re-run one room-prior trip keeping every depth frame; write MP4 + an 8-frame strip."""
    from sim.macsim.world import scene_meta
    s = [p for p in scene_meta()["spawn_poses"] if p["area"] == "area_01"][3]
    trip = run_trip((s["x"], s["y"], 1.0), "dining room", "area_01", seed=seed, keep_depth_every=1)
    f = floor()
    h, w = f.image.shape
    xy = trip.truth[:, :2]
    lo, hi = xy.min(axis=0) - 1.2, xy.max(axis=0) + 1.2
    c0, r0 = int((lo[0] - f.x0) / f.res), int((f.ytop - hi[1]) / f.res)
    c1, r1 = int((hi[0] - f.x0) / f.res), int((f.ytop - lo[1]) / f.res)
    crop = cv2.cvtColor(f.image[r0:r1, c0:c1], cv2.COLOR_GRAY2BGR)
    scale = 480 / crop.shape[0]
    base = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def px(x, y):
        return int(((x - f.x0) / f.res - c0) * scale), int(((f.ytop - y) / f.res - r0) * scale)

    frames = []
    for i in range(len(trip.t)):
        m = base.copy()
        for j in range(1, i + 1):
            cv2.line(m, px(*trip.truth[j - 1, :2]), px(*trip.truth[j, :2]), (0, 0, 220), 2)
        gx, gy, _ = resolve("dining room")
        cv2.drawMarker(m, px(gx, gy), (0, 200, 255), cv2.MARKER_STAR, 22, 2)
        tx, ty, ta = trip.truth[i]
        ex, ey, ea = trip.est[i]
        cv2.circle(m, px(tx, ty), 7, (0, 0, 220), 2)
        cv2.line(m, px(tx, ty), px(tx + .4 * np.cos(ta), ty + .4 * np.sin(ta)), (0, 0, 220), 2)
        col = (0, 180, 0) if trip.conf[i] >= .5 else (0, 140, 255)
        cv2.circle(m, px(ex, ey), 5, col, -1)
        d = trip.depth[i] if i < len(trip.depth) and trip.depth[i] is not None else np.zeros((120, 160), np.float32)
        dep = cv2.applyColorMap((np.clip(d, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dep = cv2.resize(dep, (640, 480), interpolation=cv2.INTER_NEAREST)
        frame = np.hstack((dep, m))
        text = f"t={trip.t[i]:5.1f}s {trip.state[i]:<11} conf={trip.conf[i]:.2f}  red=true  dot=estimate (green: conf>=0.5)"
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        frames.append(frame)
    OUT.mkdir(exist_ok=True)
    out = cv2.VideoWriter(str(OUT / "trip_depth_map.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                          (frames[0].shape[1], frames[0].shape[0]))
    for fr in frames:
        out.write(fr)
    out.release()
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    strip = np.vstack([cv2.resize(frames[i], None, fx=.5, fy=.5) for i in idx])
    cv2.imwrite(str(OUT / "trip_depth_map_strip.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 80])
    gx, gy, _ = resolve("dining room")
    print(f"video: {len(frames)} frames, final {trip.final.line()}, true error "
          f"{np.hypot(trip.truth[-1, 0] - gx, trip.truth[-1, 1] - gy):.2f} m, contacts {trip.contacts}")


class _Runs(pickle.Unpickler):
    """evaluate_nav runs as __main__, so its pickled Case classes name that module."""

    def find_class(self, module, name):
        return super().find_class("sim.navigate.evaluate_nav" if module == "__main__" else module, name)


def main():
    with RUNS.open("rb") as f:
        results = _Runs(f).load()
    OUT.mkdir(exist_ok=True)
    for name, (case, trip, extra, _), r in pick(results):
        trip_plot(case, trip, extra, r, name.replace("_", " "), OUT / f"{name}.png")
        print(name, case.experiment, case.index, r["status"], r["reason"], r["true_err_m"])
    video()


if __name__ == "__main__":
    main()
