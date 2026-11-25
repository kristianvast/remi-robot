"""episodes.csv (evaluate.py) -> driver-eval.csv: one row per (condition, controller, scope).

  python3 sim/isaac/elegoo/train/summarize.py <episodes.csv> .omo/waves/p3/driver-eval.csv [route,route,...]
scope = "all", an area name, or "excluded_routes" = all but the listed routes. Collision kinds: 1 tilt,
2 stall (no progress under drive), 4 off map. Contacts = onsets of the bumper gap <= 1 cm (when the CSV has them).
"""

import csv
import statistics
import sys
from collections import defaultdict

FIELDS = ("condition", "controller", "scope", "episodes", "success_rate", "collision_rate", "timeout_rate",
          "collisions_per_m", "metres", "mean_speed_m_s", "mean_peak_m_s", "max_peak_m_s",
          "success_s_per_planned_m", "success_path_ratio", "tilt", "stall", "off_map",
          "contacts_per_m", "contact_episode_rate", "near_fraction")


def row(key: tuple, episodes: list[dict]) -> dict:
    n = len(episodes)
    metres = sum(e["travel_m"] for e in episodes)
    seconds = sum(e["time_s"] for e in episodes)
    wins = [e for e in episodes if e["success"]]
    crashes = [e for e in episodes if e["collision"]]
    return dict(zip(FIELDS[:3], key)) | {
        "episodes": n, "success_rate": round(len(wins) / n, 3), "collision_rate": round(len(crashes) / n, 3),
        "timeout_rate": round(sum(e["timeout"] or e["unfinished"] for e in episodes) / n, 3),
        "collisions_per_m": round(len(crashes) / metres, 4) if metres else "", "metres": round(metres, 1),
        "mean_speed_m_s": round(metres / seconds, 3) if seconds else "",
        "mean_peak_m_s": round(statistics.mean(e["peak_m_s"] for e in episodes), 3),
        "max_peak_m_s": round(max(e["peak_m_s"] for e in episodes), 3),
        "success_s_per_planned_m": round(statistics.median(e["time_s"] / e["planned_m"] for e in wins), 2) if wins else "",
        "success_path_ratio": round(statistics.median(e["travel_m"] / e["planned_m"] for e in wins), 3) if wins else "",
        "tilt": sum(e["collision_kind"] & 1 > 0 for e in crashes), "stall": sum(e["collision_kind"] & 2 > 0 for e in crashes),
        "off_map": sum(e["collision_kind"] & 4 > 0 for e in crashes),
    } | ({
        "contacts_per_m": round(sum(e["contacts"] for e in episodes) / metres, 4) if metres else "",
        "contact_episode_rate": round(sum(e["contacts"] > 0 for e in episodes) / n, 3),
        "near_fraction": round(sum(e["near_s"] for e in episodes) / seconds, 3) if seconds else "",
    } if "contacts" in episodes[0] else {})


def main(source: str, target: str, excluded: str = "") -> None:
    excluded_routes = {r for r in excluded.split(",") if r}
    groups = defaultdict(list)
    with open(source, newline="") as f:
        for e in csv.DictReader(f):
            for k in ("success", "collision", "collision_kind", "timeout", "unfinished"):
                e[k] = int(e[k])
            for k in ("planned_m", "travel_m", "time_s", "peak_m_s"):
                e[k] = float(e[k])
            if "contacts" in e:
                e["contacts"], e["near_s"] = int(e["contacts"]), float(e["near_s"])
            groups[(e["condition"], e["controller"], "all")].append(e)
            if excluded_routes and e["route"] not in excluded_routes:
                groups[(e["condition"], e["controller"], "excluded_routes")].append(e)
            groups[(e["condition"], e["controller"], e["area"])].append(e)
    rows = [row(key, episodes) for key, episodes in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] != "all", kv[0][2]))]
    with open(target, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["scope"] in ("all", "excluded_routes"):
            print(r)


if __name__ == "__main__":
    main(*sys.argv[1:4])
