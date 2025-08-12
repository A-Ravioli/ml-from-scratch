"""
Tests for Stochastic Optimization utilities.
"""

import numpy as np
import pytest

from exercise import (
    sgd, svrg, adam
)


def quad_grad(x, i):
    # f(x)=0.5||x||^2; grad=x
    return x


def test_sgd_basic():
    x0 = np.array([1.0, -2.0])
    def lr(k):
        return 0.1
    out = sgd(quad_grad, x0, iters=100, lr_schedule=lr)
    assert np.linalg.norm(out['x']) < 1e-3


def test_adam_basic():
    x0 = np.array([3.0, -4.0])
    out = adam(quad_grad, x0, iters=500, lr=0.1)
    assert np.linalg.norm(out['x']) < 1e-3


def test_svrg_quadratic():
    # Finite-sum quadratic: (1/n)∑ 0.5||x-a_i||^2 -> grad = x - a_i
    rng = np.random.default_rng(0)
    n = 50
    A = rng.normal(size=(n, 2))
    def grad_full(x):
        return x - np.mean(A, axis=0)
    def grad_i(x, i):
        return x - A[i]
    x0 = np.array([0.0, 0.0])
    out = svrg(grad_full, grad_i, x0, n=n, m=10, eta=0.1, epochs=20)
    assert np.linalg.norm(out['x'] - np.mean(A, axis=0)) < 1e-2


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


