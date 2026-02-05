"""
Test suite for Convex Optimization implementations.

These tests are deterministic and CPU-friendly.
"""

import numpy as np

from exercise import (
    GradientDescent,
    NewtonMethod,
    ProjectedGradientDescent,
    projection_onto_simplex,
    check_convex_function,
)


def test_projection_onto_simplex_basic():
    x = np.array([0.2, -0.1, 3.0, 0.0])
    z = projection_onto_simplex(x)
    assert z.shape == x.shape
    assert np.all(z >= -1e-12)
    np.testing.assert_allclose(z.sum(), 1.0, atol=1e-10)


def test_check_convex_function_quadratic():
    # f(x) = 1/2 ||x||^2 is convex
    def f(x):
        return 0.5 * float(np.dot(x, x))

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(50, 3))
    results = check_convex_function(f, samples)
    assert results["jensen_condition"]
    assert results["first_order_condition"]
    assert results["second_order_condition"]


def test_gradient_descent_converges_on_strongly_convex_quadratic():
    A = np.diag([1.0, 3.0, 10.0])
    b = np.array([1.0, -2.0, 0.5])

    def f(x):
        return 0.5 * float(x.T @ A @ x) - float(b @ x)

    def g(x):
        return A @ x - b

    x_star = np.linalg.solve(A, b)

    opt = GradientDescent(f, g)
    res = opt.optimize(np.zeros(3), max_iterations=5000, tolerance=1e-8, step_size=0.05)
    np.testing.assert_allclose(res["x_final"], x_star, atol=1e-3)


def test_newton_method_converges_quickly_on_quadratic():
    A = np.diag([1.0, 4.0])
    b = np.array([1.0, -1.0])

    def f(x):
        return 0.5 * float(x.T @ A @ x) - float(b @ x)

    def g(x):
        return A @ x - b

    def H(x):
        return A

    x_star = np.linalg.solve(A, b)

    opt = NewtonMethod(f, g, H)
    res = opt.optimize(np.array([10.0, -10.0]), max_iterations=20, tolerance=1e-12)
    np.testing.assert_allclose(res["x_final"], x_star, atol=1e-8)


def test_projected_gradient_descent_stays_on_simplex():
    # Minimize f(x) = ||x - c||^2 over simplex.
    c = np.array([0.7, 0.2, 0.1])

    def f(x):
        d = x - c
        return float(d @ d)

    def g(x):
        return 2.0 * (x - c)

    pgd = ProjectedGradientDescent(f, g, projection_onto_simplex)
    x0 = np.array([-1.0, 2.0, 0.0])
    res = pgd.optimize(x0, max_iterations=500, tolerance=1e-10, step_size=0.1)
    x = res["x_final"]
    assert np.all(x >= -1e-12)
    np.testing.assert_allclose(x.sum(), 1.0, atol=1e-10)
    # Should be close to c projected onto simplex (c already on simplex)
    np.testing.assert_allclose(x, c, atol=1e-3)

