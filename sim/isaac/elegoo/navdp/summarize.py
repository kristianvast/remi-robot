"""episodes.csv files from evaluate.py -> one table (markdown on stdout, CSV with --csv).

  python3 sim/isaac/elegoo/navdp/summarize.py out/eval_zeroshot/episodes.csv out/eval_ft/episodes.csv --csv x.csv
Contacts = onsets of bumper gap <= 1 cm per metre driven; tilt / stall / off-map are the collision kinds;
success_excl = without the test routes crossing the area_07 collider ridge (test split only; p4 CSVs have no split
column and are test).
"""

import argparse
import csv
from collections import defaultdict

TILT, STALL, OFF_MAP = 1, 2, 4
# Test routes crossing the area_07 collider ridge box (-0.05..0.6, -7.35..-6.7), as in .omo/waves/p3/retrain.md.
AREA07_BOX_ROUTES = (246, 247, 248, 249, 250, 251, 264, 265, 266, 268, 269)
HEADER = ("checkpoint", "split", "condition", "controller", "routes", "success_%", "success_excl_%", "collisions_per_m", "contacts_per_m",
          "tilt", "stall", "off_map", "timeout", "m_per_s", "s_per_planned_m_ok")


def summarize(paths: list[str]) -> list[tuple]:
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                groups[(r["checkpoint"], r.get("split", "test"), r["condition"], r["controller"])].append(r)
    rows = []
    for (ckpt, split, cond, ctrl), rs in groups.items():
        travel = sum(float(r["travel_m"]) for r in rs)
        time_s = sum(float(r["time_s"]) for r in rs)
        kinds = [int(r["collision_kind"]) for r in rs]
        ok = [r for r in rs if r["success"] == "1"]
        kept = [r for r in rs if split != "test" or int(r["route"]) not in AREA07_BOX_ROUTES]
        rows.append((ckpt, split, cond, ctrl, len(rs), round(100 * len(ok) / len(rs), 1),
                     round(100 * sum(r["success"] == "1" for r in kept) / max(len(kept), 1), 1),
                     round(sum(int(r["collision"]) for r in rs) / max(travel, 1e-6), 4),
                     round(sum(int(r["contacts"]) for r in rs) / max(travel, 1e-6), 4),
                     sum(bool(k & TILT) for k in kinds), sum(bool(k & STALL) for k in kinds),
                     sum(bool(k & OFF_MAP) for k in kinds), sum(int(r["timeout"]) + int(r["unfinished"]) for r in rs),
                     round(travel / max(time_s, 1e-6), 3),
                     round(sum(float(r["time_s"]) for r in ok) / max(sum(float(r["planned_m"]) for r in ok), 1e-6), 2)
                     if ok else ""))
    return sorted(rows, key=lambda r: (r[1], r[2], r[3], r[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--csv")
    args = parser.parse_args()
    rows = summarize(args.episodes)
    print("| " + " | ".join(HEADER) + " |")
    print("|" + "---|" * len(HEADER))
    for r in rows:
        print("| " + " | ".join(str(v) for v in r) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)


if __name__ == "__main__":
    main()
