"""True vs estimated path plots for six evaluation runs (after evaluate.py). From the project root:

    sim/macsim/.venv/bin/python -m sim.localize.plots
"""

import pickle

import numpy as np

from sim.macsim.viewer import save_path

from .evaluate import RUNS, errors, row


class EvaluateUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class("sim.localize.evaluate" if module == "__main__" else module, name)


def main():
    with RUNS.open("rb") as f:
        results = EvaluateUnpickler(f).load()
    by = {}
    for r in results:
        by.setdefault(r.case.experiment, []).append((row(r), r))

    def pick(experiment, key, reverse=False, where=lambda d: True):
        runs = sorted((x for x in by.get(experiment, []) if where(x[0])), key=lambda x: key(x[0]), reverse=reverse)
        return runs[0] if runs else None

    chosen = [
        ("1_tracking_best", pick("tracking", lambda d: d["pos_p95_m"])),
        ("2_tracking_worst", pick("tracking", lambda d: d["pos_p95_m"], reverse=True)),
        ("3_global_fast", pick("global", lambda d: d["converge_s"], where=lambda d: d["converged"])),
        ("4_global_slow", pick("global", lambda d: d["converge_s"], reverse=True, where=lambda d: d["converged"])),
        ("5_global_wrong", pick("global", lambda d: d["pos_final_m"], reverse=True)),
        ("6_furniture_missing_worst", pick("furniture_missing", lambda d: d["pos_p95_m"], reverse=True)),
    ]
    out = RUNS.parent
    for name, item in chosen:
        if item is None:
            print(name, "no run")
            continue
        d, r = item
        pos, _ = errors(r)
        title = (f"{d['experiment']} route {d['route']} ({d['start_area']}): median {d['pos_median_m']} m, "
                 f"final {d['pos_final_m']} m, converge {d['converge_s'] or 'never'} s")
        save_path(out / f"{name}.png", r.truth, r.est, title=title)
        print(name, title)


if __name__ == "__main__":
    main()
