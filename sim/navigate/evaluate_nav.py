"""Navigator evaluation in macsim. Run from the project root:

    sim/macsim/.venv/bin/python -m sim.navigate.evaluate_nav [--quick] [--only room,cold,kidnap,furniture,no_turn]
        [--driver hand|policy|policy_veto] [--csv PATH] [--runs PATH] [--map PNG]

--map: the occupancy PNG the localiser and planner believe (same yaml as the scan's floor map); the
simulated world stays the scan.

Depth corruptions are ON at observation_spec.md's nominal magnitudes and the car's response is drawn
from calibration.yaml's ranges per trip. The navigator gets only observe() output (and the area name
for room-prior starts); arrival is judged by the simulator's TRUE pose.
"""

import argparse
import csv
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import numpy as np

from sim.macsim.world import floor, scene_meta

from .navigator import DRIVERS
from .planner import Planner
from .trip import Kidnap, Trip, run_trip

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / ".omo/waves/p3/navigate-eval.csv"
RUNS = Path(__file__).parent / "plots/runs.pkl"
SUCCESS_M, FALSE_M = .5, 1.


@dataclass
class Case:
    experiment: str
    index: int
    seed: int
    start: tuple[float, float, float]
    start_area: str
    goal: str
    goal_xy: tuple[float, float]
    goal_area: str
    planned_m: float
    planned: np.ndarray
    prior: bool = True
    drop_furniture: bool = False
    car_turn: bool = True
    kidnap: bool = False
    driver: str = "hand"
    map_image: Path | None = None


def pairs(n: int, seed: int, min_m: float = 0.) -> list[tuple]:
    """Random start in a random area, goal in a different area, connected on the TRUE map."""
    rng = np.random.default_rng(seed)
    spawns, goals = scene_meta()["spawn_poses"], scene_meta()["goals"]
    areas = sorted({s["area"] for s in spawns})
    planners = {}
    out = []
    while len(out) < n:
        area = areas[rng.integers(len(areas))]
        s = [p for p in spawns if p["area"] == area][rng.integers(sum(p["area"] == area for p in spawns))]
        options = [g for g in goals if g["area"] != area]
        g = options[rng.integers(len(options))]
        if g["name"] not in planners:
            planners[g["name"]] = Planner(False)
            planners[g["name"]].plan((g["x"], g["y"]))
        pl = planners[g["name"]]
        path = pl.path(s["x"], s["y"], max_m=200.)
        if len(path) < 2 or np.hypot(*(path[-1] - (g["x"], g["y"]))) > 1e-6:
            continue
        length = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
        if length < min_m:
            continue
        yaw = float(rng.uniform(-np.pi, np.pi))
        out.append(((s["x"], s["y"], yaw), area, g["name"], (g["x"], g["y"]), g["area"], length, path))
    return out


def cases(quick: bool, only: set[str], driver: str = "hand", map_image: Path | None = None,
          offset: int = 0) -> list[Case]:
    """offset != 0: a different set of trips (other start/goal pairs and seeds), e.g. for tuning."""
    k = 5 if quick else 1
    room = pairs(100 // k, 1101 + offset)
    cold = pairs(50 // k, 2202 + offset)
    kid = pairs(30 // k, 3303 + offset, min_m=8.)
    out = []

    def add(name, plist, base, **kw):
        out.extend(Case(name, i, base + offset + i, *p, driver=driver, map_image=map_image, **kw)
                   for i, p in enumerate(plist))

    if "room" in only:
        add("room_prior", room, 10000)
    if "cold" in only:
        add("cold_start", cold, 20000, prior=False)
    if "kidnap" in only:
        add("kidnap", kid, 30000, kidnap=True)
    if "furniture" in only:
        add("furniture_mismatch", room, 10000, drop_furniture=True)
    if "no_turn" in only:
        add("no_car_turn", room, 10000, car_turn=False)
    return out


def run(case: Case) -> tuple[Case, Trip, dict, float]:
    started = perf_counter()
    kidnap = None
    if case.kidnap:
        # Carry the car only to spots from which the goal is reachable on the true map.
        truth = Planner(False)
        truth.plan(case.goal_xy)
        spots = np.array([(s["x"], s["y"]) for s in scene_meta()["spawn_poses"]
                          if np.isfinite(truth.cost_to_go(s["x"], s["y"]))])
        kidnap = Kidnap(after_m=.4 * case.planned_m, candidates=spots, seed=case.seed + 7)
    trip = run_trip(case.start, case.goal, case.start_area if case.prior else None, seed=case.seed,
                    driver=case.driver, drop_furniture=case.drop_furniture, car_turn=case.car_turn,
                    kidnap=kidnap, map_image=case.map_image)
    extra = {}
    if kidnap:
        extra = dict(kidnap_t=kidnap.done_t, kidnap_to=kidnap.to,
                     kidnap_from=None if kidnap.done_t is None else
                     tuple(trip.truth[np.searchsorted(trip.t, kidnap.done_t), :2]))
    return case, trip, extra, perf_counter() - started


def row(case: Case, trip: Trip, extra: dict, cpu: float) -> dict:
    f = floor()
    end = trip.truth[-1]
    true_err = float(np.hypot(end[0] - case.goal_xy[0], end[1] - case.goal_xy[1]))
    claimed = trip.final.state == "arrived"
    err = np.hypot(*(trip.est[:, :2] - trip.truth[:, :2]).T)
    trusted = trip.conf >= .5
    duration = float(trip.t[-1])
    out = dict(experiment=case.experiment, driver=case.driver, index=case.index, seed=case.seed,
               start_area=case.start_area, start_xy=f"{case.start[0]:.2f} {case.start[1]:.2f}",
               goal=case.goal, goal_area=case.goal_area, prior="room" if case.prior else "none",
               planned_m=round(case.planned_m, 2), status=trip.final.state, reason=trip.final.reason,
               claimed_arrival=int(claimed), true_err_m=round(true_err, 2),
               success=int(claimed and true_err <= SUCCESS_M), false_arrival=int(claimed and true_err > FALSE_M),
               time_s=round(duration, 1), driven_m=round(trip.driven_m, 2),
               path_ratio=round(trip.driven_m / case.planned_m, 2),
               mean_speed_m_s=round(trip.driven_m / duration, 3) if duration else 0.,
               contacts=trip.contacts, contacts_per_m=round(trip.contacts / max(trip.driven_m, 1e-6), 3),
               first_confident_s=trip.nav_stats["first_confident_s"] or "",
               localise_s=trip.nav_stats["localise_s"], recoveries=trip.nav_stats["recoveries"],
               poor_fit_frac=round(float(np.mean(trip.map_fit < .5)), 3),
               map_wrong_cells=trip.nav_stats["map_wrong_cells"],
               replans=trip.nav_stats["replans"], veto_frac=trip.nav_stats.get("veto_frac", ""),
               est_err_trusted_median_m=round(float(np.median(err[trusted])), 3) if trusted.any() else "",
               confident_wrong_frac=round(float(np.mean(trusted & (err > 1))), 4),
               final_true_xy=f"{end[0]:.2f} {end[1]:.2f}", final_true_area=f.area(*end[:2]),
               final_est_xy=f"{trip.est[-1, 0]:.2f} {trip.est[-1, 1]:.2f}", cpu_s=round(cpu, 1))
    if case.kidnap:
        t0 = extra["kidnap_t"]
        out["kidnapped"] = int(t0 is not None and extra["kidnap_to"] is not None)
        if out["kidnapped"]:
            after = trip.t > t0
            out["kidnap_m"] = round(float(np.hypot(*(np.array(extra["kidnap_to"][:2]) - extra["kidnap_from"]))), 2)
            window = after & (trip.t <= t0 + 20)
            out["noticed"] = int(bool(np.any(trip.conf[window] < .5)))
            good = after & (err < .5) & trusted
            # Relocalised: first time after the kidnap from which the estimate stays right for 2 s.
            idx = np.flatnonzero(good)
            rel = ""
            for i in idx:
                j = np.searchsorted(trip.t, trip.t[i] + 2)
                if j <= len(trip.t) and good[i:j].all():
                    rel = round(float(trip.t[i] - t0), 1)
                    break
            out["relocalised_s"] = rel
            wrong = after & trusted & (err > 1)
            out["confident_wrong_after_kidnap_s"] = round(float(wrong.sum() * .1), 1)
    return out


def summary(rows: list[dict]) -> list[dict]:
    out = []
    for exp in dict.fromkeys(r["experiment"] for r in rows):
        rs = [r for r in rows if r["experiment"] == exp]
        n = len(rs)
        driven = sum(r["driven_m"] for r in rs)
        reasons = {k: sum(r["reason"] == k for r in rs) for k in ("lost", "blocked", "timeout")}
        ok = [r for r in rs if r["success"]]
        first = [r["first_confident_s"] for r in rs if r["first_confident_s"] != ""]
        s = dict(experiment=exp, driver=rs[0]["driver"], index="ALL", seed=n, prior=rs[0]["prior"],
                 planned_m=round(float(np.mean([r["planned_m"] for r in rs])), 2),
                 success=round(len(ok) / n, 3),
                 claimed_arrival=round(float(np.mean([r["claimed_arrival"] for r in rs])), 3),
                 false_arrival=round(float(np.mean([r["false_arrival"] for r in rs])), 3),
                 reason=" ".join(f"{k}={v}" for k, v in reasons.items()),
                 contacts=round(float(np.mean([r["contacts"] for r in rs])), 2),
                 contacts_per_m=round(sum(r["contacts"] for r in rs) / driven, 4),
                 time_s=round(float(np.median([r["time_s"] for r in ok])), 1) if ok else "",
                 path_ratio=round(float(np.median([r["path_ratio"] for r in ok])), 2) if ok else "",
                 mean_speed_m_s=round(driven / sum(r["time_s"] for r in rs), 3),
                 first_confident_s=round(float(np.median(first)), 1) if first else "",
                 localise_s=round(float(np.median([r["localise_s"] for r in rs])), 1),
                 localise_frac=round(sum(r["localise_s"] for r in rs) / sum(r["time_s"] for r in rs), 3),
                 poor_fit_frac=round(float(np.mean([r["poor_fit_frac"] for r in rs])), 3),
                 recoveries=round(float(np.mean([r["recoveries"] for r in rs])), 2),
                 confident_wrong_frac=round(float(np.mean([r["confident_wrong_frac"] for r in rs])), 4))
        s["time_per_planned_m"] = round(float(np.median([r["time_s"] / r["planned_m"] for r in ok])), 2) if ok else ""
        s["first_confident_p90_s"] = round(float(np.percentile(first, 90)), 1) if first else ""
        if exp == "kidnap":
            kd = [r for r in rs if r.get("kidnapped")]
            rel = [r["relocalised_s"] for r in kd if r["relocalised_s"] != ""]
            s.update(kidnapped=len(kd), noticed=round(float(np.mean([r["noticed"] for r in kd])), 3) if kd else "",
                     relocalised_s=round(float(np.median(rel)), 1) if rel else "",
                     relocalised_frac=round(len(rel) / len(kd), 3) if kd else "")
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="room,cold,kidnap,furniture,no_turn")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--driver", choices=DRIVERS, default="hand")
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--map", type=Path, default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()
    todo = cases(args.quick, set(args.only.split(",")), args.driver, args.map, args.seed_offset)
    print(f"{len(todo)} trips", flush=True)
    started = perf_counter()
    results, rows = [], []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap(run, todo, chunksize=1)):
            results.append(res)
            r = row(*res)
            rows.append(r)
            print(i, r["experiment"], r["index"], r["start_area"], "->", r["goal"], r["status"], r["reason"],
                  "err", r["true_err_m"], "t", r["time_s"], "contacts", r["contacts"], f"{res[3]:.0f}s", flush=True)
    table = rows + summary(rows)
    keys = list(dict.fromkeys(k for r in table for k in r))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(table)
    args.runs.parent.mkdir(parents=True, exist_ok=True)
    with args.runs.open("wb") as f:
        pickle.dump(results, f)
    for s in summary(rows):
        print(s)
    print(f"wall {perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
