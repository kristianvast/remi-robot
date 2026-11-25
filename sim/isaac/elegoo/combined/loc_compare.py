"""Place head vs sim/localize on the same frames (elegoo-sim-3, project mirror ~/elegoo as cwd, macsim venv).

For every recorded student car of evaluate.py (record.npz) the localiser gets exactly what the model got: the
delivered frame's depth (the model's own DA2 depth, observation_spec encoding; None on dropped frames), sonar
(None where it was missing), the drifting gyro heading and the PWM sent. Two priors: the start room (as go_to
uses today) and a cold start. Output: per-step rows (CSV) and a JSON summary, including a simple complement
(localiser when its confidence >= 0.5, else the place head).

  sim/macsim/.venv/bin/python ~/p4-combined/loc_compare.py ~/p4-combined/record.npz ~/p4-combined/labels.npz ~/p4-combined/loc
"""

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sim.localize.localise import Localizer

AREAS = tuple(f"area_{i:02d}" for i in range(1, 11))
REC = LAB = None


def room_at(xy):
    r = int(np.clip(round((float(LAB["y0"]) - xy[1]) / float(LAB["res"])), 0, LAB["area"].shape[0] - 1))
    c = int(np.clip(round((xy[0] - float(LAB["x0"])) / float(LAB["res"])), 0, LAB["area"].shape[1] - 1))
    return int(LAB["area"][r, c])


def run_car(job):
    i, prior = job
    loc = Localizer(seed=100 + i)
    if prior == "room":
        loc.reset(room=str(REC["start_area"][i]))
    rows = []
    steps = int(np.argmax(REC["done"][i])) if REC["done"][i].any() else REC["done"].shape[1]
    for t in range(steps):
        depth = REC["depth"][i, t].astype(np.float32) / 255 if REC["delivered"][i, t] else None
        cm = float(REC["range_cm"][i, t])
        command = tuple(int(v) for v in REC["pwm"][i, t - 1]) if t else (0, 0)
        est = loc.update(command, float(REC["gyro_heading"][i, t]), None if cm < 0 else int(round(cm)), depth, 0.0)
        x, y, yaw = (float(v) for v in REC["pose"][i, t])
        lx, ly, _ = est.pose
        hx, hy = (float(v) for v in REC["pose_hat"][i, t, :2])
        p = REC["room_p"][i, t].astype(np.float32)
        truth = room_at((x, y))
        rows.append((i, prior, t, truth, room_at((lx, ly)), int(p.argmax()), round(est.confidence, 3),
                     round(float(p.max()), 3), round(float(np.hypot(lx - x, ly - y)), 3),
                     round(float(np.hypot(hx - x, hy - y)), 3)))
    return rows


def init(record, labels):
    global REC, LAB
    REC = {k: v for k, v in np.load(record).items()}
    LAB = {k: v for k, v in np.load(labels).items()}


def stats(rows, sel):
    r = [x for x in rows if sel(x)]
    if not r:
        return {}
    a = np.array([x[3:] for x in r], dtype=float)
    truth, loc_room, head_room, loc_conf = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    loc_err, head_err = a[:, 5], a[:, 6]
    use_loc = loc_conf >= 0.5
    comp_err = np.where(use_loc, loc_err, head_err)
    comp_room = np.where(use_loc, loc_room, head_room)
    q = lambda e: {"median_m": round(float(np.median(e)), 3), "p95_m": round(float(np.percentile(e, 95)), 3)}
    return {"steps": len(r), "localiser": {"room_acc": round(float((loc_room == truth).mean()), 3), **q(loc_err)},
            "head": {"room_acc": round(float((head_room == truth).mean()), 3), **q(head_err)},
            "complement": {"room_acc": round(float((comp_room == truth).mean()), 3), **q(comp_err),
                           "localiser_share": round(float(use_loc.mean()), 3)}}


def main(record: Path, labels: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cars = len(np.load(record)["cars"])
    jobs = [(i, p) for p in ("room", "cold") for i in range(cars)]
    with Pool(min(56, len(jobs)), initializer=init, initargs=(record, labels)) as pool:
        rows = [row for part in pool.map(run_car, jobs) for row in part]
    with (out / "loc_steps.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("car", "prior", "t", "room", "loc_room", "head_room", "loc_conf", "head_conf", "loc_err_m",
                    "head_err_m"))
        w.writerows(rows)
    summary = {p: {"all": stats(rows, lambda x, p=p: x[1] == p),
                   "after_3s": stats(rows, lambda x, p=p: x[1] == p and x[2] >= 30)} for p in ("room", "cold")}
    (out / "loc_summary.json").write_text(json.dumps(summary, indent=1))
    print("LOC_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
