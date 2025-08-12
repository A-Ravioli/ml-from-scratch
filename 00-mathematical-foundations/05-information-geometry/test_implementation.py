"""
Tests for Information Geometry utilities.
"""

import numpy as np
import pytest

from exercise import (
    fisher_information_bernoulli, fisher_information_gaussian,
    natural_gradient_step, gaussian_geodesic_approx
)


def test_fisher_bernoulli():
    I = fisher_information_bernoulli(0.3)
    assert abs(I - (1.0/(0.3*0.7))) < 1e-8


def test_fisher_gaussian():
    F = fisher_information_gaussian(0.0, 2.0)
    assert np.allclose(F, np.array([[1/4.0, 0.0], [0.0, 2.0]]))


def test_natural_gradient():
    F = np.array([[2.0, 0.0],[0.0, 8.0]])
    g = np.array([2.0, 8.0])
    step = natural_gradient_step(g, F, lr=1.0)
    # Preconditioned gradient should be [-1, -1]
    assert np.allclose(step, -np.array([1.0, 1.0]))


def test_geodesic_approx():
    mu0, s0 = 0.0, 1.0
    mu1, s1 = 2.0, 2.0
    mu_half, s_half = gaussian_geodesic_approx(mu0, s0, mu1, s1, t=0.5)
    assert s_half > 1.0 and s_half < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


