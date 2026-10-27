"""2-DOF planar manipulator: M(q) q'' + C(q, q') q' + G(q) = tau.

Mirrors matlab/fuzzy_ctc/robot_dynamics_common.m and matlab/ekf/system_dynamics.m.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotParams:
    m1: float = 1.0  # kg
    m2: float = 1.0  # kg
    l1: float = 0.1  # m
    l2: float = 0.1  # m
    g: float = 9.81  # m/s^2


def dynamics(q: np.ndarray, qd: np.ndarray, p: RobotParams):
    """Return mass matrix M, Coriolis/centrifugal term C(q, qd) qd and gravity vector G."""
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    m11 = (p.m1 + p.m2) * p.l1**2 + p.m2 * p.l2**2 + 2 * p.m2 * p.l1 * p.l2 * c2
    m12 = p.m2 * p.l2**2 + p.m2 * p.l1 * p.l2 * c2
    m22 = p.m2 * p.l2**2
    # Small regularisation, as in the MATLAB code, keeps M well conditioned.
    M = np.array([[m11, m12], [m12, m22]]) + 1e-6 * np.eye(2)

    h = p.m2 * p.l1 * p.l2 * s2
    C = np.array([-h * (2 * qd[0] * qd[1] + qd[1] ** 2), h * qd[0] ** 2])

    s1, s12 = np.sin(q[0]), np.sin(q[0] + q[1])
    G = np.array([(p.m1 + p.m2) * p.l1 * p.g * s1 + p.m2 * p.l2 * p.g * s12, p.m2 * p.l2 * p.g * s12])
    return M, C, G


def state_derivative(x: np.ndarray, tau: np.ndarray, p: RobotParams) -> np.ndarray:
    """x = [q1, q2, qd1, qd2]; returns dx/dt."""
    M, C, G = dynamics(x[:2], x[2:], p)
    qdd = np.linalg.solve(M, tau - C - G)
    return np.concatenate([x[2:], qdd])


def rk4_step(x: np.ndarray, tau: np.ndarray, dt: float, p: RobotParams) -> np.ndarray:
    k1 = state_derivative(x, tau, p)
    k2 = state_derivative(x + dt / 2 * k1, tau, p)
    k3 = state_derivative(x + dt / 2 * k2, tau, p)
    k4 = state_derivative(x + dt * k3, tau, p)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
