"""Final comparison table, computed only from the columns of results/<prefix>_<driver>.csv.

    python sim/cmdtest/report/compare.py [--prefix final] [--dir sim/cmdtest/results] [drivers ...]

Per command type (the request's `kind`) and driver:
  requests passed = requests (request, seed) whose scored call rows (passed True/False) all passed, at least one scored;
  contacts/m      = sum(contacts) / sum(path_m) over all call rows of that kind;
  false arrivals  = call rows with false_arrival True;
  tilts           = sum(tilts) over all call rows;
  median call s   = median dur_s over call rows (stop and ERROR rows excluded).
The recomputed request pass is checked against the CSV's request_pass column; mismatches are printed.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

RESULTS = Path(__file__).resolve().parents[1] / "results"
DRIVERS = ("hand", "policy_v2_veto", "navdp", "combined")


def load(path):
    lines = path.read_text().splitlines()
    header = [line[2:].strip() for line in lines if line.startswith("# ")]
    return header, list(csv.DictReader(line for line in lines if not line.startswith("#")))


def summarise(rows):
    calls = defaultdict(list)
    for r in rows:
        calls[(r["request"], r["seed"])].append(r)
    out = defaultdict(lambda: dict(req=0, ok=0, path=0., contacts=0, fa=0, tilts=0, dur=[]))
    mismatches = 0
    for group in calls.values():
        kind = group[0]["kind"]
        scored = [r["passed"] == "True" for r in group if r["passed"] in ("True", "False")]
        ok = bool(scored) and all(scored)
        mismatches += ok != (group[0]["request_pass"] == "True")
        for k in (kind, "all"):
            s = out[k]
            s["req"] += 1
            s["ok"] += ok
            for r in group:
                s["path"] += float(r["path_m"] or 0)
                s["contacts"] += int(r["contacts"] or 0)
                s["fa"] += r["false_arrival"] == "True"
                s["tilts"] += int(r["tilts"] or 0)
                if r["dur_s"] and r["tool"] not in ("stop", "ERROR"):
                    s["dur"].append(float(r["dur_s"]))
    return out, mismatches


def cell(s, what):
    if what == "pass":
        return f"{s['ok']}/{s['req']} ({100 * s['ok'] / s['req']:.0f} %)" if s["req"] else "-"
    if what == "cpm":
        return f"{s['contacts'] / s['path']:.3f}" if s["path"] else "-"
    if what in ("fa", "tilts"):
        return str(s[what])
    return f"{median(s['dur']):.1f}" if s["dur"] else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--dir", type=Path, default=RESULTS)
    ap.add_argument("drivers", nargs="*", default=list(DRIVERS))
    a = ap.parse_args()
    found = {d: a.dir / f"{a.prefix}_{d}.csv" for d in a.drivers if (a.dir / f"{a.prefix}_{d}.csv").exists()}
    if not found:
        raise SystemExit(f"no {a.prefix}_*.csv in {a.dir}")
    data = {}
    for d, p in found.items():
        header, rows = load(p)
        data[d], mismatches = summarise(rows)
        tags = [h for h in header if h.startswith(("harness_hash", "driver "))]
        print(f"{d}: {len(rows)} call rows; request_pass mismatches vs recompute: {mismatches}; {'; '.join(tags)}")
    names = list(found)
    kinds = sorted({k for s in data.values() for k in s if k != "all"}) + ["all"]
    for title, what in (("requests passed", "pass"), ("contacts/m", "cpm"), ("false arrivals", "fa"),
                        ("tilts", "tilts"), ("median call s", "dur")):
        print(f"\n{title}\n| command type | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for kind in kinds:
            print(f"| {kind} | " + " | ".join(cell(data[d][kind], what) if kind in data[d] else "-" for d in names)
                  + " |")


if __name__ == "__main__":
    main()
