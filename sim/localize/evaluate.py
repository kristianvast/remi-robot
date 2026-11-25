"""Localiser evaluation in macsim. Run from the project root:

    sim/macsim/.venv/bin/python -m sim.localize.evaluate [--quick] [--only tracking,...]

Corruptions are ON at observation_spec.md magnitudes and the car's response is drawn from
calibration.yaml's ranges per run; the localiser gets only commands, gyro, range, depth and head.
The scripted driver steers on the TRUE pose (it only provides motion; it is not a navigator).
"""

import argparse
import csv
import pickle
from dataclasses import dataclass, field, replace
from math import pi
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import numpy as np

from sim.macsim.simulator import Simulator
from sim.macsim.world import floor

from .localise import Localizer
from .routes import Driver, Planner, Route

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / ".omo/waves/p3/localize-eval.csv"
RUNS = Path(__file__).parent / "plots/runs.pkl"
DT = .1
LOST_M, LOST_S = 1.0, 5.0
CONVERGED_M, HOLD_S = .5, 5.0
TRUST = .5


@dataclass
class Case:
    experiment: str
    route: Route
    seed: int
    prior: str  # "pose", "room" or "none"
    scale: float | None = None  # None = per-episode draw from the spec
    failure: float = .03
    use_depth: bool = True
    use_range: bool = True
    laps: int = 1  # global runs drive the route out and back to give time to converge
    drop_furniture: bool = False


@dataclass
class Result:
    case: Case
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    truth: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    est: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    travelled: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # per step: mode mass, stable steps, sensor age, spread m, bad_fit, wall_bad (certainty's ingredients)
    parts: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    seconds: float = 0.


def run(case: Case) -> Result:
    started = perf_counter()
    sim = Simulator(seed=case.seed, randomize=True, corrupt=True, scale=case.scale, failure=case.failure)
    route = case.route
    start = (*route.points[0], route.yaw)
    sim.reset(start)
    loc = Localizer(seed=case.seed + 1, drop_furniture=case.drop_furniture)
    if case.prior == "pose":
        loc.reset(pose=start)
    elif case.prior == "room":
        loc.reset(room=route.start_area)
    else:
        loc.reset()
    legs = [route.points]
    for _ in range(case.laps - 1):
        legs.append(legs[-1][::-1])
    rows, travelled, last = [], 0., np.array(start[:2])
    t = 0.
    for leg in legs:
        driver = Driver(Route(route.index, "", "", leg, 0., 0.))
        limit = t + 3 * len(leg) * .11 / .25 + 30
        while not driver.done and t < limit:
            cmd = driver.command(sim.pose)
            sim.step(*cmd, dt=DT)
            t += DT
            obs = sim.observe(camera=case.use_depth)
            est = loc.update(cmd, obs.gyro, obs.range_cm if case.use_range else None,
                             obs.depth if case.use_depth else None, obs.head_angle)
            if est.look_around is not None:
                sim.set_head(est.look_around)
            pos = np.array(sim.pose[:2])
            travelled += float(np.linalg.norm(pos - last))
            last = pos
            rows.append((t, *sim.pose, *est.pose, est.confidence, travelled,
                         est.mass, loc.stable, loc.sensor_age, est.spread_m, loc.bad_fit, loc.wall_bad))
    a = np.array(rows)
    return Result(case, a[:, 0], a[:, 1:4], a[:, 4:7], a[:, 7], a[:, 8], a[:, 9:15], perf_counter() - started)


def errors(r: Result):
    pos = np.hypot(*(r.est[:, :2] - r.truth[:, :2]).T)
    head = np.abs((r.est[:, 2] - r.truth[:, 2] + pi) % (2 * pi) - pi)
    return pos, np.degrees(head)


def longest_run(mask: np.ndarray) -> float:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best * DT


def convergence(r: Result, pos: np.ndarray):
    """First time after which error stays <= 0.5 m to the end (and for at least HOLD_S)."""
    bad = np.flatnonzero(pos > CONVERGED_M)
    i = 0 if len(bad) == 0 else bad[-1] + 1
    if i >= len(pos) or r.t[-1] - r.t[i] < HOLD_S:
        return None
    return i


def row(r: Result) -> dict:
    c = r.case
    pos, head = errors(r)
    out = dict(experiment=c.experiment, route=c.route.index, seed=c.seed, prior=c.prior,
               start_area=c.route.start_area, goal_area=c.route.goal_area,
               route_m=round(c.route.length * c.laps, 2), duration_s=round(r.t[-1], 1),
               driven_m=round(r.travelled[-1], 2),
               pos_median_m=round(float(np.median(pos)), 3), pos_p95_m=round(float(np.percentile(pos, 95)), 3),
               pos_final_m=round(float(pos[-1]), 3), head_median_deg=round(float(np.median(head)), 2),
               head_p95_deg=round(float(np.percentile(head, 95)), 2),
               lost_longest_s=round(longest_run(pos > LOST_M), 1),
               conf_final=round(float(r.conf[-1]), 3),
               confident_wrong_frac=round(float(np.mean((r.conf > .7) & (pos > LOST_M))), 4),
               trusted_frac=round(float(np.mean(r.conf >= TRUST)), 3),
               trusted_within_0_5m=round(float(np.mean(pos[r.conf >= TRUST] < CONVERGED_M)), 4)
               if (r.conf >= TRUST).any() else "",
               cpu_s=round(r.seconds, 1))
    out["lost"] = int(out["lost_longest_s"] > LOST_S)
    i = convergence(r, pos)
    out["converged"] = int(i is not None)
    out["converge_s"] = round(float(r.t[i]), 1) if i is not None else ""
    out["converge_m"] = round(float(r.travelled[i]), 2) if i is not None else ""
    wrong = i is None and pos[-1] > LOST_M
    out["wrong_place"] = int(wrong)
    out["final_true_xy"] = f"{r.truth[-1, 0]:.2f} {r.truth[-1, 1]:.2f}"
    out["final_est_xy"] = f"{r.est[-1, 0]:.2f} {r.est[-1, 1]:.2f}"
    out["final_est_area"] = floor().area(*r.est[-1, :2])
    out["final_true_area"] = floor().area(*r.truth[-1, :2])
    return out


def cases(quick: bool, only: set[str], offset: int = 0) -> list[Case]:
    planner = Planner()
    track = planner.routes(global_starts=False)
    starts = planner.routes(global_starts=True)
    if quick:
        track, starts = track[::10], starts[::25]
    out = []
    if "tracking" in only:
        out += [Case("tracking", r, 1000 + r.index, "pose") for r in track]
    if "global" in only:
        out += [Case("global", r, 2000 + r.index, "none", laps=2) for r in starts]
    if "room" in only:
        out += [Case("room", r, 3000 + r.index, "room", laps=2) for r in starts]
    if "sensitivity" in only:
        for name, kw in (("scale_0.70", dict(scale=.70)), ("scale_1.30", dict(scale=1.30)),
                         ("fail_10pct", dict(failure=.10)), ("range_only", dict(use_depth=False)),
                         ("depth_only", dict(use_range=False))):
            out += [Case(name, r, 4000 + r.index, "pose", **kw) for r in track]
    if "furniture" in only:
        out += [Case("furniture_missing", r, 5000 + r.index, "pose", drop_furniture=True) for r in track]
        out += [Case("global_furniture_missing", r, 6000 + r.index, "none", laps=2, drop_furniture=True)
                for r in starts]
    return [replace(c, seed=c.seed + offset) for c in out]


def summary(rows: list[dict], results: list[Result]) -> list[dict]:
    out = []
    for exp in dict.fromkeys(r["experiment"] for r in rows):
        sel = [res for res, r in zip(results, rows) if r["experiment"] == exp]
        rs = [r for r in rows if r["experiment"] == exp]
        pos = np.concatenate([errors(s)[0] for s in sel])
        head = np.concatenate([errors(s)[1] for s in sel])
        conv = [r for r in rs if r["converged"]]
        out.append(dict(experiment=exp, route="ALL", seed=len(rs), prior=rs[0]["prior"],
                        route_m=round(float(np.mean([r["route_m"] for r in rs])), 1),
                        duration_s=round(float(np.sum([r["duration_s"] for r in rs])), 0),
                        pos_median_m=round(float(np.median(pos)), 3),
                        pos_p95_m=round(float(np.percentile(pos, 95)), 3),
                        pos_final_m=round(float(np.median([r["pos_final_m"] for r in rs])), 3),
                        head_median_deg=round(float(np.median(head)), 2),
                        head_p95_deg=round(float(np.percentile(head, 95)), 2),
                        lost=round(float(np.mean([r["lost"] for r in rs])), 3),
                        converged=round(len(conv) / len(rs), 3),
                        converge_s=round(float(np.median([r["converge_s"] for r in conv])), 1) if conv else "",
                        converge_m=round(float(np.median([r["converge_m"] for r in conv])), 2) if conv else "",
                        wrong_place=round(float(np.mean([r["wrong_place"] for r in rs])), 3),
                        confident_wrong_frac=round(float(np.mean(np.concatenate(
                            [(s.conf > .7) & (errors(s)[0] > LOST_M) for s in sel]))), 4),
                        trusted_frac=round(float(np.mean(np.concatenate([s.conf for s in sel]) >= TRUST)), 3),
                        trusted_within_0_5m=round(float(np.mean(pos[np.concatenate([s.conf for s in sel]) >= TRUST]
                                                                < CONVERGED_M)), 4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="tracking,global,room,sensitivity,furniture")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--seed-offset", type=int, default=0, help="a different seed set (e.g. for tuning)")
    ap.add_argument("--runs", type=Path, default=RUNS)
    args = ap.parse_args()
    todo = cases(args.quick, set(args.only.split(",")), args.seed_offset)
    print(f"{len(todo)} runs", flush=True)
    started = perf_counter()
    with Pool(args.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap(run, todo, chunksize=1)):
            results.append(res)
            r = row(res)
            print(i, r["experiment"], r["route"], r["start_area"], "median", r["pos_median_m"], "final",
                  r["pos_final_m"], "lost", r["lost"], "conv", r["converge_s"], f"{res.seconds:.0f}s", flush=True)
    rows = [row(r) for r in results]
    table = rows + summary(rows, results)
    keys = list(dict.fromkeys(k for r in table for k in r))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(table)
    args.runs.parent.mkdir(parents=True, exist_ok=True)
    with args.runs.open("wb") as f:
        pickle.dump(results, f)
    for s in summary(rows, results):
        print(s)
    print(f"wall {perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
