"""
Tests for Constrained Optimization utilities.
"""

import numpy as np
import pytest

from exercise import (
    solve_quadratic_program, augmented_lagrangian, projection_box
)


def test_qp_simple():
    # minimize 1/2 x^T I x + (-1)^T x subject to x >= 0
    P = np.eye(2)
    q = -np.ones(2)
    G = -np.eye(2)
    h = np.zeros(2)
    x = solve_quadratic_program(P, q, G=G, h=h)
    # Unconstrained optimum is x=1; constraints keep x>=0
    assert np.all(x >= -1e-8)


def test_augmented_lagrangian_equality():
    # minimize (x-1)^2 subject to x = 0 => solution x=0
    def f(x):
        return (x[0]-1.0)**2
    def g(x):
        return np.array([2*(x[0]-1.0)])
    def hfun(x):
        return np.array([x[0]])
    out = augmented_lagrangian(f, g, hfun, x0=np.array([2.0]))
    assert abs(out.get('x', np.array([0.0]))[0]) < 1e-2


def test_box_projection():
    x = np.array([-2.0, 0.5, 3.0])
    y = projection_box(x, lower=0.0, upper=1.0)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


