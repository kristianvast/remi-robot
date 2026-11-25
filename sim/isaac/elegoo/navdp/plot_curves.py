"""learning.csv from train.py -> learning_curves.png (policy-driven episodes only for the driving metrics).

  python plot_curves.py run/learning.csv run/learning_curves.png
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(path: str, out: str) -> None:
    rows = list(csv.DictReader(open(path, newline="")))
    minutes = [float(r["elapsed_s"]) / 60 for r in rows]

    def series(key):
        pts = [(m, float(r[key])) for m, r in zip(minutes, rows, strict=True) if r[key] != ""]
        return [p[0] for p in pts], [p[1] for p in pts]

    panels = (("loss_imitation", "diffusion loss"), ("loss_critic", "critic loss"),
              ("success_policy", "success (policy-driven)"), ("collision_policy", "collision rate (policy-driven)"),
              ("contacts_per_m_policy", "contacts / m (policy-driven)"), ("beta", "expert fraction beta"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (key, title) in zip(axes.flat, panels, strict=True):
        x, y = series(key)
        ax.plot(x, y, ".", ms=3, alpha=0.5)
        if len(y) >= 5:
            k = 5
            ax.plot(x[k - 1:], [sum(y[i - k + 1:i + 1]) / k for i in range(k - 1, len(y))], "-")
        ax.set_title(title)
        ax.set_xlabel("minutes")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main(*sys.argv[1:3])
