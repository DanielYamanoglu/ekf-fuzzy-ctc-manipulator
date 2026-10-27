"""Fuzzy gain scheduler: (e, e_dot) -> scaling factors for Kp and Kd.

Port of matlab/fuzzy_ctc/fuzzy_gain_scheduler_optimized.m: 5 Gaussian membership
functions per input, a 5x5 rule base per gain, weighted-average defuzzification.
"""

import numpy as np

MF_CENTERS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # NB, NS, Z, PS, PB
MF_SIGMA = 0.35
OUT_CENTERS = np.array([0.625, 0.875, 1.25, 1.625, 1.875])  # VL, L, M, H, VH

# Rows: error NB..PB, columns: error-rate NB..PB, values index OUT_CENTERS (1-based as in MATLAB).
RULE_KP = np.array([[5, 5, 4, 3, 2], [5, 4, 3, 2, 1], [4, 3, 3, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 5]]) - 1
RULE_KD = np.array([[2, 3, 4, 5, 5], [1, 2, 3, 4, 5], [1, 2, 3, 2, 1], [5, 4, 3, 2, 1], [5, 5, 4, 3, 2]]) - 1

E_MAX = 0.5  # rad
E_DOT_MAX = 2.0  # rad/s
KP_NOMINAL = 100.0
KD_NOMINAL = 90.0
ALPHA_RANGE = (0.5, 1.5)


def _memberships(value: float) -> np.ndarray:
    mu = np.exp(-((value - MF_CENTERS) ** 2) / (2 * MF_SIGMA**2))
    return mu / (mu.sum() + 1e-10)


def scaling(e: float, e_dot: float) -> tuple[float, float]:
    """Return (alpha_Kp, alpha_Kd) for one joint."""
    mu_e = _memberships(np.clip(e / E_MAX, -1, 1))
    mu_ed = _memberships(np.clip(e_dot / E_DOT_MAX, -1, 1))
    w = np.outer(mu_e, mu_ed)  # rule firing strengths
    a_kp = (w * OUT_CENTERS[RULE_KP]).sum() / w.sum()
    a_kd = (w * OUT_CENTERS[RULE_KD]).sum() / w.sum()
    return float(np.clip(a_kp, *ALPHA_RANGE)), float(np.clip(a_kd, *ALPHA_RANGE))


def fuzzy_gains(e: np.ndarray, e_dot: np.ndarray):
    """Diagonal Kp, Kd matrices and the per-joint scaling factors."""
    alphas = np.array([scaling(e[i], e_dot[i]) for i in range(2)])
    Kp = np.diag(KP_NOMINAL * alphas[:, 0])
    Kd = np.diag(KD_NOMINAL * alphas[:, 1])
    return Kp, Kd, alphas
