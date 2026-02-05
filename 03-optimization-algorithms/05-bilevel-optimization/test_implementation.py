"""
Tests for bilevel optimization hypergradients (implicit differentiation).
"""

import numpy as np

from exercise import hypergradient_lambda, finite_difference_hypergradient


def test_hypergradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = rng.normal(size=(40,))
    X_val = rng.normal(size=(20, 5))
    y_val = rng.normal(size=(20,))
    lam = 0.2

    g_impl = hypergradient_lambda(X, y, X_val, y_val, lam)
    g_fd = finite_difference_hypergradient(X, y, X_val, y_val, lam, eps=1e-5)
    assert abs(g_impl - g_fd) < 1e-4

