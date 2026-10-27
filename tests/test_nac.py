import numpy as np

from nac import RobotParams, dynamics, fuzzy_gains, metrics, simulate
from nac.ekf import state_jacobian
from nac.fuzzy import ALPHA_RANGE, scaling


def test_mass_matrix_symmetric_positive_definite():
    for q2 in np.linspace(-np.pi, np.pi, 13):
        M, _, _ = dynamics(np.array([0.3, q2]), np.zeros(2), RobotParams())
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_gravity_vanishes_hanging_down():
    _, C, G = dynamics(np.zeros(2), np.zeros(2), RobotParams())
    assert np.allclose(G, 0) and np.allclose(C, 0)


def test_fuzzy_scaling_bounded_and_symmetric():
    for e in np.linspace(-1, 1, 9):
        for ed in np.linspace(-3, 3, 9):
            a_kp, a_kd = scaling(e, ed)
            assert ALPHA_RANGE[0] <= a_kp <= ALPHA_RANGE[1]
            assert ALPHA_RANGE[0] <= a_kd <= ALPHA_RANGE[1]
    assert np.isclose(scaling(0.2, 0.5)[0], scaling(-0.2, -0.5)[0])


def test_fuzzy_gains_are_diagonal():
    Kp, Kd, _ = fuzzy_gains(np.array([0.1, -0.2]), np.array([0.0, 1.0]))
    assert np.count_nonzero(Kp - np.diag(np.diag(Kp))) == 0
    assert np.count_nonzero(Kd - np.diag(np.diag(Kd))) == 0


def test_jacobian_matches_finite_difference_shape():
    F = state_jacobian(np.array([0.1, 0.2, 0.0, 0.0]), np.zeros(2), RobotParams())
    assert F.shape == (4, 4)
    assert np.allclose(F[:2, 2:], np.eye(2))


def test_closed_loop_converges_and_ekf_tracks():
    for controller in ("fixed", "fuzzy"):
        r = simulate(controller, T=6.0, seed=1)
        m = metrics(r)
        assert m["rms_estimation_error_rad"] < 0.03
        assert m["max_torque_nm"] <= 15.0 + 1e-9
        assert m["steady_rms_error_rad"] < 0.05
        assert np.all(np.abs(r.x_true[-50:, :2] - np.array([0.32, 0.09])) < 0.08)


def test_recovers_from_torque_pulse():
    m = metrics(simulate("fuzzy", T=12.0, disturbance=1.0, seed=2))
    assert m["max_deviation_after_disturbance_rad"] > 0.05
    assert m["recovery_time_s"] < 6.0
