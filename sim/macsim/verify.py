"""Repeatable calibration, sensor throughput and Isaac comparison checks."""

import csv
import json
from math import hypot, radians
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .depth import pinhole
from .drive import Drive
from .simulator import Simulator
from .world import ISAAC, SCAN

OUT = Path(__file__).parent


def main() -> None:
    values = {}
    for tag, pwm, goal, which in (("pwm80_cm_s", (80, 80), 27, 0),
                                   ("pwm110_cm_s", (110, 110), 39, 0),
                                   ("spin80_deg_s", (-80, 80), 130, 1)):
        d = Drive()
        for _ in range(200):
            speed, turn = d.tick(pwm)
        actual = speed*100 if which == 0 else np.degrees(turn)
        assert abs(actual/goal - 1) <= .05, (tag, actual)
        values[tag] = float(actual)
    d = Drive()
    d.tick((110, 110))
    while d.pwm == (0, 0):
        d.tick(None)
    while d.pwm != (0, 0):
        d.tick(None)
    values["watchdog_s"] = round((d.ticks-1)*.01-d.last_receive, 3)
    assert abs(values["watchdog_s"]/.4-1) <= .05

    car = Simulator(corrupt=True)
    nd = 300
    for _ in range(15):
        car.step(-45, 45)
        car.observe()
    start = perf_counter()
    for _ in range(nd):
        car.step(-45, 45)
        obs = car.observe()
    values["corrupt_camera_steps_s"] = nd/(perf_counter()-start)
    assert values["corrupt_camera_steps_s"] > 1/.07
    assert obs.depth.shape == (120, 160) and obs.depth.dtype == np.float32
    assert np.all((obs.depth >= 0) & (obs.depth <= 1))

    # Compare the sampling locations at the centers of corresponding 4x4 tiles.
    # Isaac renders a textured visual scan; our renderer uses the collision mesh.
    rows, errors = [], []
    p = car.drive.p.copy()
    for v in json.loads((SCAN / "views/poses_rendered.json").read_text())["poses"]:
        ref = ISAAC / "views" / f"{Path(v['file']).stem}_depth_mm.png"
        if not ref.is_file() or v["kind"] != "room":
            continue
        p["camera_pitch_deg"], p["camera_hfov_deg"] = v["pitch"], v["hfov"]
        z, _ = pinhole((v["x"], v["y"], radians(v["yaw"])), 0., p)
        with Image.open(ref) as im:
            depth = np.asarray(im).astype(np.float32) * .001
        truth = depth[1::4, 1::4]
        mask = (truth > 0) & np.isfinite(z) & (z < 20)
        delta = z[mask] - truth[mask]
        errors.append(delta)
        rows.append((ref.stem, int(mask.sum()), mask.mean(), np.median(abs(delta)),
                     np.mean(abs(delta)), np.percentile(abs(delta), 95)))
    assert len(rows) >= 10
    with (OUT / "isaac-depth.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("name", "paired_pixels", "paired_fraction", "median_abs_m", "mae_m", "p95_abs_m"))
        writer.writerows(rows)
    err = np.concatenate(errors)
    values.update(depth_poses=len(rows), paired_fraction=float(np.mean([r[2] for r in rows])),
                  median_abs_m=float(np.median(abs(err))), mae_m=float(np.mean(abs(err))),
                  p95_abs_m=float(np.percentile(abs(err), 95)), signed_mean_m=float(np.mean(err)))
    print(json.dumps(values, indent=2))
    (OUT / "verification.json").write_text(json.dumps(values, indent=2)+"\n")


if __name__ == "__main__":
    main()
