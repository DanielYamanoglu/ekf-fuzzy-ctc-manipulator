"""Run fixed-gain vs fuzzy CTC (with EKF in the loop), write metrics and figures.

Usage: python python/run_experiments.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nac.simulate import Q_REF, metrics, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
SEEDS = range(10)
DISTURBANCE_NM = 1.0


def mean_metrics(controller: str, disturbance: float = 0.0) -> dict:
    T = 12.0 if disturbance else 6.0
    runs = [metrics(simulate(controller, T=T, disturbance=disturbance, seed=s)) for s in SEEDS]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def plot_tracking(fixed, fuzzy, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for j in range(2):
        ax = axes[0, j]
        ax.plot(fixed.t, fixed.x_true[:, j], label="fixed-gain CTC")
        ax.plot(fuzzy.t, fuzzy.x_true[:, j], label="fuzzy CTC")
        ax.plot(fuzzy.t, fuzzy.x_est[:, j], "--", lw=1, label="EKF estimate (fuzzy)")
        ax.axhline(Q_REF[j], color="k", lw=0.8, ls=":", label="reference")
        ax.set_ylabel(f"q{j + 1} [rad]")
        ax.grid(alpha=0.3)
        axes[1, j].plot(fuzzy.t, fuzzy.kp[:, j], color="tab:purple")
        axes[1, j].set_ylabel(f"Kp{j + 1} (fuzzy)")
        axes[1, j].set_xlabel("time [s]")
        axes[1, j].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Step response with EKF in the loop and adaptive gains (seed 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_disturbance(fixed, fuzzy, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(fixed.t, fixed.x_true[:, 0] - Q_REF[0], label="fixed-gain CTC")
    ax.plot(fuzzy.t, fuzzy.x_true[:, 0] - Q_REF[0], label="fuzzy CTC")
    ax.axvline(6.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("q1 error [rad]")
    ax.set_title(f"{DISTURBANCE_NM:g} N·m, 100 ms torque pulse on joint 1 at t = 6 s (seed 0)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def table(rows: dict) -> str:
    keys = list(next(iter(rows.values())))
    lines = ["| metric | " + " | ".join(rows) + " |", "|---" * (len(rows) + 1) + "|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(f"{rows[c][k]:.4f}" for c in rows) + " |")
    return "\n".join(lines)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_tracking(simulate("fixed", T=6.0), simulate("fuzzy", T=6.0), FIG_DIR / "step_response.png")
    plot_disturbance(simulate("fixed", T=10.0, disturbance=DISTURBANCE_NM),
                     simulate("fuzzy", T=10.0, disturbance=DISTURBANCE_NM), FIG_DIR / "disturbance.png")

    step = {"fixed": mean_metrics("fixed"), "fuzzy": mean_metrics("fuzzy")}
    dist = {"fixed": mean_metrics("fixed", DISTURBANCE_NM), "fuzzy": mean_metrics("fuzzy", DISTURBANCE_NM)}
    report = (f"# Python port results (mean of {len(SEEDS)} seeds)\n\n## Step to reference\n\n{table(step)}\n\n"
              f"## Step + {DISTURBANCE_NM:g} N·m, 100 ms torque pulse at t = 6 s\n\n{table(dist)}\n")
    (ROOT / "reports" / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
