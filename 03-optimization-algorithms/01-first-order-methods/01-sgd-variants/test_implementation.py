"""
Test suite for SGD variants implementation (deterministic, fast).
"""

from __future__ import annotations

import numpy as np

from exercise import (
    QuadraticProblem,
    VanillaSGD,
    SGDWithMomentum,
    NesterovSGD,
    AdaGrad,
    RMSprop,
    Adam,
    SVRG,
    optimize_problem,
)


def _numerical_grad(f, x, eps=1e-6):
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def test_quadratic_gradient_matches_finite_differences():
    prob = QuadraticProblem(dim=5, condition_number=20.0, noise_std=0.0)
    x = np.linspace(-0.3, 0.2, 5)
    g = prob.gradient(x)
    g_num = _numerical_grad(prob.objective, x)
    np.testing.assert_allclose(g, g_num, rtol=1e-5, atol=1e-6)


def test_quadratic_optimal_point_has_zero_gradient():
    prob = QuadraticProblem(dim=4, condition_number=15.0, noise_std=0.0)
    x_star = prob.optimal_point()
    g = prob.gradient(x_star)
    assert np.linalg.norm(g) < 1e-8


def test_vanilla_sgd_update_rule():
    opt = VanillaSGD(learning_rate=0.1)
    g = np.array([0.5, -0.25])
    dx = opt.step(g)
    np.testing.assert_allclose(dx, -0.1 * g)


def test_momentum_updates_velocity_correctly():
    opt = SGDWithMomentum(learning_rate=0.1, momentum=0.9)
    g1 = np.array([1.0, 2.0])
    dx1 = opt.step(g1)
    # first: v = 0.1*g1, dx=-v
    np.testing.assert_allclose(dx1, -0.1 * g1)
    g2 = np.array([0.5, -1.0])
    dx2 = opt.step(g2)
    v1 = 0.1 * g1
    v2 = 0.9 * v1 + 0.1 * g2
    np.testing.assert_allclose(dx2, -v2)


def test_nesterov_lookahead_point():
    opt = NesterovSGD(learning_rate=0.1, momentum=0.9)
    x = np.array([1.0, 2.0])
    # no velocity yet
    np.testing.assert_allclose(opt.get_lookahead_point(x), x)
    # after one step, velocity exists
    _ = opt.step(np.array([1.0, 1.0]))
    look = opt.get_lookahead_point(x)
    assert look.shape == x.shape


def test_adagrad_first_step_matches_formula():
    opt = AdaGrad(learning_rate=0.1, eps=1e-8)
    g = np.array([0.5, -0.3])
    dx = opt.step(g)
    expected = -0.1 / np.sqrt(g**2 + 1e-8) * g
    np.testing.assert_allclose(dx, expected, rtol=1e-7, atol=1e-10)


def test_rmsprop_first_step_matches_formula():
    opt = RMSprop(learning_rate=0.1, decay_rate=0.9, eps=1e-8)
    g = np.array([0.5, -0.3])
    dx = opt.step(g)
    v = (1 - 0.9) * g**2
    expected = -(0.1 / (np.sqrt(v) + 1e-8)) * g
    np.testing.assert_allclose(dx, expected, rtol=1e-10)


def test_adam_bias_correction_first_step():
    opt = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    g = np.array([0.5, -0.3])
    dx = opt.step(g)
    # t=1: m=(1-beta1)g, v=(1-beta2)g^2, bias-corrected gives m_hat=g, v_hat=g^2
    expected = -0.001 * g / (np.sqrt(g**2) + 1e-8)
    np.testing.assert_allclose(dx, expected, rtol=1e-10)


def test_optimize_problem_decreases_objective_on_quadratic():
    prob = QuadraticProblem(dim=3, condition_number=5.0, noise_std=0.0)
    x0 = np.array([5.0, -3.0, 2.0])
    opt = VanillaSGD(learning_rate=0.05)
    x_final, hist = optimize_problem(prob, opt, x0, n_iterations=200, batch_size=1)
    assert hist["objective"][0] >= hist["objective"][-1]


def test_svrg_runs_and_improves_objective_on_quadratic():
    prob = QuadraticProblem(dim=5, condition_number=10.0, noise_std=0.0)
    x0 = np.ones(5)
    opt = SVRG(learning_rate=0.1, update_frequency=10)
    x_final, hist = optimize_problem(prob, opt, x0, n_iterations=50, batch_size=1)
    assert hist["objective"][0] >= hist["objective"][-1]
