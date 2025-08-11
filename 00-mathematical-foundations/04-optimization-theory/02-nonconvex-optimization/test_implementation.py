"""
Tests for Nonconvex Optimization utilities.
"""

import numpy as np
import pytest

from exercise import (
    find_critical_points_2d, noisy_gradient_descent, trust_region_step
)


def test_critical_points_rosenbrock():
    def f(x):
        return (1 - x[0])**2 + 100*(x[1]-x[0]**2)**2
    def g(x):
        dx = -2*(1 - x[0]) - 400*x[0]*(x[1]-x[0]**2)
        dy = 200*(x[1]-x[0]**2)
        return np.array([dx, dy])
    def H(x):
        return np.array([[2 - 400*(x[1]-3*x[0]**2), -400*x[0]], [-400*x[0], 200]])
    xs = np.linspace(-2, 2, 50)
    ys = np.linspace(-1, 3, 50)
    res = find_critical_points_2d(f, g, H, (xs, ys))
    assert 'minima' in res


def test_noisy_gd_saddle_escape():
    # Saddle at origin: f(x,y)=x^2 - y^2
    def f(x):
        return x[0]**2 - x[1]**2
    def g(x):
        return np.array([2*x[0], -2*x[1]])
    out = noisy_gradient_descent(f, g, x0=np.array([0.1, 0.0]), step=0.05, noise_std=0.01, iters=500)
    assert 'trajectory' in out


def test_trust_region_cauchy_point():
    gradx = np.array([1.0, 0.0])
    hessx = np.eye(2)
    p = trust_region_step(gradx, hessx, delta=0.5)
    assert np.linalg.norm(p) <= 0.5 + 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


