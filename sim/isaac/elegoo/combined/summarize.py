"""Results tables from evaluate.py outputs and the learning curve from train.py's learning.csv (numpy + matplotlib).

  python summarize.py <out_dir> <eval_dir> [<eval_dir> ...] [--learning learning.csv]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def driving(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["controller"])].append(r)
    table = {}
    for (cond, ctrl), rs in sorted(groups.items()):
        f = lambda k: np.array([float(r[k]) for r in rs])
        travel, time, planned = f("travel_m"), f("time_s"), f("planned_m")
        kind = f("collision_kind").astype(int)
        ok = f("success") > 0
        table[f"{cond}/{ctrl}"] = {
            "routes": len(rs), "success": round(100 * ok.mean(), 1),
            "collisions_per_m": round(f("collision").sum() / travel.sum(), 4),
            "contacts_per_m": round(f("contacts").sum() / travel.sum(), 4),
            "tilts": int(((kind & 1) > 0).sum()), "stalls": int(((kind & 2) > 0).sum()),
            "off_map": int(((kind & 4) > 0).sum()), "timeouts": int(f("timeout").sum()),
            "m_s": round(travel.sum() / time.sum(), 3),
            "s_per_planned_m": round(float(time[ok].sum() / planned[ok].sum()), 2) if ok.any() else None,
        }
    return table


def place(npz):
    d = np.load(npz)
    cols = [str(c) for c in d["columns"]]
    rows = d["rows"].astype(np.float32)
    c = {k: rows[..., i] for i, k in enumerate(cols)}
    out = {}
    who = d["controller"] if "controller" in d.files else np.full(len(d["condition"]), "student")
    for cond, ctrl in sorted(set(zip(d["condition"].tolist(), who.tolist()))):
        car = (d["condition"] == cond) & (who == ctrl)
        active = (c["active"] > 0) & car[:, None]
        if not active.any():
            continue
        err = np.hypot(c["x_hat"] - c["x"], c["y_hat"] - c["y"])[active]
        dyaw = np.abs((c["yaw_hat"] - c["yaw"] + np.pi) % (2 * np.pi) - np.pi)[active]
        goal = np.stack([c[f"goal_err_{i}"] for i in range(10)], -1)[active]
        goal = np.degrees(goal[np.isfinite(goal)])
        out[f"{cond}/{ctrl}"] = {"steps": int(active.sum()), "room_acc": round(float((c["room_hat"] == c["room"])[active].mean()), 3),
                     "xy_median_m": round(float(np.median(err)), 3), "xy_p95_m": round(float(np.percentile(err, 95)), 3),
                     "yaw_median_deg": round(float(np.degrees(np.median(dyaw))), 1),
                     "goal_median_deg": round(float(np.median(goal)), 1), "goal_p90_deg": round(float(np.percentile(goal, 90)), 1),
                     "sd_xy_median_m": round(float(np.median(c["sd_xy"][active])), 3)}
    return out


def plot(learning: Path, out: Path, cars: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(learning.open()))
    x = np.array([float(r["step"]) for r in rows]) * cars / 1e6
    get = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for a, (k, title) in zip(ax.flat, (("drive", "imitation loss (PWM / 150)"), ("room_acc", "room accuracy (train batch)"),
                                       ("xy_err_m", "median xy error m (train batch)"), ("goal", "goal loss (1 - cos)"),
                                       ("s_success", "student-driven success"), ("beta", "beta (teacher share)"))):
        y = get(k)
        a.plot(x, y, lw=1)
        if k == "s_success":
            a.plot(x, get("t_success"), lw=1, label="teacher-driven")
            a.legend()
        if k in ("room_acc", "xy_err_m", "drive", "goal") and f"v_{k}" in rows[0]:
            a.plot(x, get(f"v_{k}"), lw=1, label="held-out start cells")
            a.legend()
        a.set_title(title)
        a.set_xlabel("car-steps (M)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("evals", type=Path, nargs="+")
    p.add_argument("--learning", type=Path)
    p.add_argument("--cars", type=int, default=384, help="cars per vector step in the training run")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    result = {"driving": {}, "place": {}}
    for e in a.evals:
        # one eval dir = one route split; tables are keyed split:condition/controller
        split = json.loads((e / "eval_run.json").read_text()).get("split", e.name)
        rows = list(csv.DictReader((e / "episodes.csv").open()))
        result["driving"].update({f"{split}:{k}": v for k, v in driving(rows).items()})
        if (e / "place.npz").exists():
            result["place"].update({f"{split}:{k}": v for k, v in place(e / "place.npz").items()})
    if a.learning:
        plot(a.learning, a.out / "learning_curves.png", a.cars)
    (a.out / "results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
