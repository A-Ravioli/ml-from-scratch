"""
Deterministic tests for Newton method implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from exercise import (
    QuadraticProblem,
    RosenbrockProblem,
    LogisticRegressionProblem,
    NewtonOptimizer,
    DampedNewtonOptimizer,
    TrustRegionNewton,
)


class TestQuadraticNewton:
    def test_newton_solves_quadratic_in_one_step(self):
        np.random.seed(0)
        prob = QuadraticProblem(dim=5, condition_number=20.0)
        x0 = np.random.randn(5)
        opt = NewtonOptimizer(max_iterations=5, tolerance=1e-12, line_search=False)
        x_star, hist = opt.optimize(prob, x0)
        np.testing.assert_allclose(x_star, prob.optimal_point(), rtol=0.0, atol=1e-8)
        assert len(hist["objective"]) >= 1


class TestRosenbrock:
    def test_damped_newton_decreases_objective(self):
        prob = RosenbrockProblem(dim=2)
        x0 = np.array([-1.2, 1.0])
        f0 = prob.objective(x0)
        opt = DampedNewtonOptimizer(max_iterations=25, tolerance=1e-10, line_search=True)
        x_final, _ = opt.optimize(prob, x0)
        assert prob.objective(x_final) < f0


class TestLogReg:
    def test_newton_improves_logistic_objective(self):
        np.random.seed(0)
        prob = LogisticRegressionProblem(n_samples=80, dim=4, regularization=0.1)
        x0 = np.zeros(4)
        f0 = prob.objective(x0)
        opt = NewtonOptimizer(max_iterations=15, tolerance=1e-8, line_search=True)
        x_final, _ = opt.optimize(prob, x0)
        assert prob.objective(x_final) < f0


class TestTrustRegion:
    def test_trust_region_runs_and_decreases_quadratic(self):
        np.random.seed(0)
        prob = QuadraticProblem(dim=6, condition_number=50.0)
        x0 = np.random.randn(6)
        f0 = prob.objective(x0)
        opt = TrustRegionNewton(initial_radius=1.0)
        x_final, _ = opt.optimize(prob, x0, max_iterations=30, tolerance=1e-10)
        assert prob.objective(x_final) < f0

