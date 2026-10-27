"""Closed-loop simulation: EKF-in-the-loop computed-torque control, fixed vs fuzzy gains.

Scenario follows matlab/fuzzy_ctc/fuzzy_ctc_simulation.m: step to q_ref = [0.32, 0.09] rad,
dt = 10 ms, process and measurement noise, torque limit 15 N*m, optional torque pulse disturbance.
"""

from dataclasses import dataclass

import numpy as np

from .dynamics import RobotParams, dynamics, rk4_step
from .ekf import Ekf
from .fuzzy import KD_NOMINAL, KP_NOMINAL, fuzzy_gains

Q_REF = np.array([0.32, 0.09])
DT = 0.01
TAU_MAX = 15.0
P0 = np.diag([0.01, 0.01, 0.5, 0.5])
Q = np.diag([1.5e-4, 1.5e-4, 0.03, 0.012])
R = np.diag([2e-4, 2e-4])
PROCESS_NOISE_STD = 0.02
DISTURBANCE_TIME = 6.0  # s, after the overdamped step response has settled
SETTLE_BAND = 0.03  # rad; measurement noise std is ~0.014 rad, so a 2 % band is below the noise floor


@dataclass
class Result:
    t: np.ndarray
    x_true: np.ndarray  # (N, 4)
    x_est: np.ndarray  # (N, 4)
    tau: np.ndarray  # (N, 2)
    kp: np.ndarray  # (N, 2) diagonal of Kp
    nis: np.ndarray  # (N,)


def simulate(controller: str = "fuzzy", T: float = 8.0, disturbance: float = 0.0,
             disturbance_time: float = DISTURBANCE_TIME, disturbance_duration: float = 0.1,
             seed: int = 0, params: RobotParams = RobotParams()) -> Result:
    """Run one closed loop. controller is "fuzzy" or "fixed"; disturbance is a torque pulse on joint 1 (N*m)."""
    rng = np.random.default_rng(seed)
    n = int(round(T / DT))
    x = np.zeros(4)
    ekf = Ekf([0.02, -0.01, 0.0, 0.0], P0, Q, R, params)
    hist = {k: [] for k in ("x_true", "x_est", "tau", "kp", "nis")}

    for k in range(n):
        e, e_dot = ekf.x[:2] - Q_REF, ekf.x[2:]
        if controller == "fuzzy":
            Kp, Kd, _ = fuzzy_gains(e, e_dot)
        else:
            Kp, Kd = KP_NOMINAL * np.eye(2), KD_NOMINAL * np.eye(2)
        M, C, G = dynamics(ekf.x[:2], ekf.x[2:], params)
        tau = np.clip(M @ (-Kp @ e - Kd @ e_dot) + C + G, -TAU_MAX, TAU_MAX)

        tau_plant = tau.copy()
        if disturbance and disturbance_time <= k * DT < disturbance_time + disturbance_duration:
            tau_plant[0] += disturbance
        x = rk4_step(x, tau_plant, DT, params)
        x[2:] += PROCESS_NOISE_STD * rng.standard_normal(2) * DT
        y = x[:2] + np.sqrt(R[0, 0]) * rng.standard_normal(2)

        ekf.predict(tau, DT)
        nis = ekf.update(y)

        for key, val in (("x_true", x.copy()), ("x_est", ekf.x.copy()), ("tau", tau),
                         ("kp", np.diag(Kp).copy()), ("nis", nis)):
            hist[key].append(val)

    return Result(np.arange(n) * DT, *(np.array(hist[k]) for k in ("x_true", "x_est", "tau", "kp", "nis")))


def settling_time(t: np.ndarray, err: np.ndarray, band: float = SETTLE_BAND) -> float:
    """First time after which |err| stays within band."""
    outside = np.abs(err) > band
    if not outside.any():
        return 0.0
    last = np.nonzero(outside)[0][-1]
    return float(t[last + 1]) if last + 1 < len(t) else float("nan")


def metrics(r: Result) -> dict:
    """Step metrics on joint 1 before DISTURBANCE_TIME, recovery metrics after it."""
    err = r.x_true[:, :2] - Q_REF
    pre = r.t < DISTURBANCE_TIME
    steady = (r.t >= DISTURBANCE_TIME - 1.0) & pre
    post = ~pre
    if r.t[-1] < DISTURBANCE_TIME + 1.0:
        post[:] = False  # run too short to judge recovery
    m = {
        "settling_time_q1_s": settling_time(r.t[pre], err[pre, 0]),
        "overshoot_q1_pct": float(max(0.0, (r.x_true[pre, 0].max() - Q_REF[0]) / Q_REF[0] * 100)),
        "steady_rms_error_rad": float(np.sqrt((err[steady] ** 2).mean())),
        "rms_estimation_error_rad": float(np.sqrt(((r.x_est[:, :2] - r.x_true[:, :2]) ** 2).mean())),
        "mean_nis": float(r.nis.mean()),
        "max_torque_nm": float(np.abs(r.tau).max()),
    }
    if post.any():
        m["max_deviation_after_disturbance_rad"] = float(np.abs(err[post, 0]).max())
        m["recovery_time_s"] = settling_time(r.t[post] - DISTURBANCE_TIME, err[post, 0])
    return m
