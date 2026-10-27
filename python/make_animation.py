"""Animate the simulated arm (fixed vs fuzzy CTC, torque pulse at t = 6 s) from real simulation output.

Usage: python python/make_animation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from nac.dynamics import RobotParams  # noqa: E402
from nac.simulate import DISTURBANCE_TIME, Q_REF, simulate  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "assets" / "simulation.gif"
FRAME_STEP = 5  # every 5th 10 ms sample -> 20 fps real time


def joints(q, p: RobotParams):
    # Angles are measured from the downward vertical (G = 0 at q = 0).
    elbow = np.array([p.l1 * np.sin(q[0]), -p.l1 * np.cos(q[0])])
    tip = elbow + [p.l2 * np.sin(q[0] + q[1]), -p.l2 * np.cos(q[0] + q[1])]
    return np.array([[0, 0], elbow, tip])


def main() -> None:
    p = RobotParams()
    runs = {"fixed-gain CTC": simulate("fixed", T=9.0, disturbance=1.0),
            "fuzzy CTC": simulate("fuzzy", T=9.0, disturbance=1.0)}
    ref = joints(Q_REF, p)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    lines, titles = [], []
    for ax, name in zip(axes, runs):
        ax.plot(ref[:, 0], ref[:, 1], "o--", color="0.7", label="reference")
        (line,) = ax.plot([], [], "o-", lw=4, color="tab:blue" if "fixed" in name else "tab:orange")
        lines.append(line)
        titles.append(ax.set_title(name))
        ax.set_xlim(-0.22, 0.22)
        ax.set_ylim(-0.22, 0.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)

    def frame(i):
        k = i * FRAME_STEP
        for line, title, (name, r) in zip(lines, titles, runs.items()):
            pts = joints(r.x_true[k, :2], p)
            line.set_data(pts[:, 0], pts[:, 1])
            pulse = DISTURBANCE_TIME <= r.t[k] < DISTURBANCE_TIME + 0.3
            title.set_text(f"{name}   t = {r.t[k]:.1f} s" + ("   pulse!" if pulse else ""))
        return lines

    n = len(next(iter(runs.values())).t) // FRAME_STEP
    FuncAnimation(fig, frame, frames=n, blit=False).save(OUT, writer=PillowWriter(fps=20))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
