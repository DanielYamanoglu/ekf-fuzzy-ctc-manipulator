"""Extended Kalman Filter estimating [q1, q2, qd1, qd2] from noisy joint-angle measurements.

Port of matlab/ekf/ekf_predict.m and ekf_update.m (RK4 prediction, numerical Jacobian,
Joseph-form covariance update, NIS for consistency checks).
"""

import numpy as np

from .dynamics import RobotParams, rk4_step, state_derivative

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])  # only joint angles are measured


def state_jacobian(x: np.ndarray, tau: np.ndarray, p: RobotParams, eps: float = 1e-7) -> np.ndarray:
    F = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        F[:, i] = (state_derivative(x + dx, tau, p) - state_derivative(x - dx, tau, p)) / (2 * eps)
    return F


class Ekf:
    def __init__(self, x0, P0, Q, R, params: RobotParams):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.params = params

    def predict(self, tau: np.ndarray, dt: float) -> None:
        Phi = np.eye(4) + state_jacobian(self.x, tau, self.params) * dt
        self.x = rk4_step(self.x, tau, dt, self.params)
        P = Phi @ self.P @ Phi.T + self.Q
        self.P = 0.5 * (P + P.T)

    def update(self, y: np.ndarray) -> float:
        """Correct with measurement y; returns the normalised innovation squared (NIS)."""
        innovation = y - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return float(innovation @ np.linalg.solve(S, innovation))
