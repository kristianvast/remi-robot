"""One trip in macsim:  python -m sim.navigate.demo --from area_01 --to "dining room" [--cold] [--seed N]
    [--driver hand|policy]

The car starts at a random free spot in --from (the navigator is told only the area name, or nothing
with --cold), drives to the named place on its own estimate, and the result is judged by the true pose.
"""

import argparse

import numpy as np

from sim.macsim.world import floor, scene_meta

from .navigator import DRIVERS
from .places import area_name, resolve
from .trip import run_trip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True, help="area id, e.g. area_01")
    ap.add_argument("--to", dest="goal", required=True, help="place name from places.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cold", action="store_true", help="do not tell the navigator the start area")
    ap.add_argument("--driver", choices=DRIVERS, default="hand")
    args = ap.parse_args()
    gx, gy, garea = resolve(args.goal)
    rng = np.random.default_rng(args.seed)
    spawns = [s for s in scene_meta()["spawn_poses"] if s["area"] == args.start]
    if not spawns:
        raise SystemExit(f"No spawn poses in {args.start!r}")
    s = spawns[rng.integers(len(spawns))]
    start = (s["x"], s["y"], float(rng.uniform(-np.pi, np.pi)))
    print(f"start {args.start} ({area_name(args.start)}) at ({start[0]:.2f}, {start[1]:.2f}); "
          f"goal {args.goal!r} = {garea} ({gx:.2f}, {gy:.2f}); prior: {'none' if args.cold else args.start}; "
          f"driver: {args.driver}")

    def show(status):
        if round(status.t * 10) % 10 == 0 or status.state in ("arrived", "failed"):
            print(status.line(), flush=True)

    trip = run_trip(start, args.goal, None if args.cold else args.start, seed=args.seed,
                    driver=args.driver, on_status=show)
    end = trip.truth[-1]
    err = float(np.hypot(end[0] - gx, end[1] - gy))
    verdict = ("SUCCESS" if trip.final.state == "arrived" and err <= .5 else
               "FALSE ARRIVAL" if trip.final.state == "arrived" else f"FAILED ({trip.final.reason})")
    print(f"RESULT {verdict}: navigator says {trip.final.state}{' ' + trip.final.reason if trip.final.reason else ''}"
          f" after {trip.t[-1]:.1f} s; true position ({end[0]:.2f}, {end[1]:.2f}) in {floor().area(*end[:2]) or '?'},"
          f" {err:.2f} m from the goal; drove {trip.driven_m:.1f} m; contacts {trip.contacts}")


if __name__ == "__main__":
    main()
