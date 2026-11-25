"""learning.csv -> learning_curves.png (20-iteration episode-weighted means; stochastic training policy).

  python plot_curves.py <learning.csv> <out.png>
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(source: str, target: str, window: int = 20) -> None:
    rows = list(csv.DictReader(open(source, newline="")))
    xs, success, per_m, speed, seconds, contacts = [], [], [], [], [], []
    for start in range(0, len(rows), window):
        part = [r for r in rows[start:start + window] if int(r["episodes"])]
        n = sum(int(r["episodes"]) for r in part)
        xs.append(int(part[-1]["iteration"]))
        seconds.append(float(part[-1]["elapsed_s"]))
        success.append(sum(float(r["success_rate"]) * int(r["episodes"]) for r in part) / n)
        per_m.append(sum(float(r["collisions_per_m"]) for r in part) / len(part))
        speed.append(sum(float(r["mean_speed_m_s"]) for r in part) / len(part))
        if "contacts_per_m" in part[0]:
            contacts.append(sum(float(r["contacts_per_m"]) for r in part) / len(part))
    boundary = next((int(r["iteration"]) for r in rows if r["segment_seed"] != rows[0]["segment_seed"]), None)
    panels = [(success, "success rate"), (per_m, "collisions per metre"), (speed, "mean speed (m/s, limits 0.5-1)")]
    if contacts:
        panels.append((contacts, "contacts per metre (bumper gap <= 1 cm)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (ys, title) in zip(axes, panels):
        ax.plot(xs, ys, marker="o", ms=3)
        if boundary is not None:
            ax.axvline(boundary, color="grey", ls="--")
        ax.set_title(title)
        ax.set_xlabel("PPO iteration (1024 cars x 32 steps)")
        ax.grid(alpha=0.3)
    if boundary is not None:
        axes[0].text(boundary + 3, min(success), "segment 2: new camera draw,\ncollision penalty 15 -> 25", fontsize=8)
    fig.tight_layout()
    fig.savefig(target, dpi=110)
    print("wrote", target)


if __name__ == "__main__":
    main(*sys.argv[1:3])
