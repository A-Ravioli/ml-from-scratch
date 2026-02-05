"""
Deterministic tests for quasi-Newton implementations (BFGS / L-BFGS).
"""

from __future__ import annotations

import numpy as np
import pytest

from exercise import QuadraticProblem, RosenbrockProblem, BFGSOptimizer, LBFGSOptimizer


class TestBFGS:
    def test_bfgs_solves_quadratic(self):
        np.random.seed(0)
        prob = QuadraticProblem(dim=6, condition_number=30.0)
        x0 = np.random.randn(6)
        opt = BFGSOptimizer(store_inverse=True, max_iterations=200, tolerance=1e-10, line_search=True)
        x_star, hist = opt.optimize(prob, x0)
        np.testing.assert_allclose(x_star, prob.optimal_point(), rtol=0.0, atol=1e-4)
        assert len(hist["objective"]) >= 1


class TestLBFGS:
    def test_lbfgs_decreases_rosenbrock(self):
        np.random.seed(0)
        prob = RosenbrockProblem(dim=2)
        x0 = np.array([-1.2, 1.0])
        f0 = prob.objective(x0)
        opt = LBFGSOptimizer(history_size=5, max_iterations=200, tolerance=1e-8, line_search=True)
        x_final, _ = opt.optimize(prob, x0)
        assert prob.objective(x_final) < f0

