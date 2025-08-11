"""
Tests for Concentration Inequalities.
"""

import numpy as np
import pytest

from exercise import (
    is_subgaussian, subgaussian_parameter, hoeffding_bound,
    bernstein_bound, chernoff_bound, verify_bounds_empirically
)


class TestSubGaussian:
    def test_subgaussian_check(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(0, 1, size=5000)
        t_vals = np.linspace(-1.0, 1.0, 21)
        assert is_subgaussian(samples, t_vals, c=1.5)  # loose upper bound
        c_hat = subgaussian_parameter(samples, t_vals)
        assert c_hat < 2.0


class TestClassicBounds:
    def test_hoeffding(self):
        b = hoeffding_bound(n=100, a=0.0, b=1.0, t=0.1)
        assert 0 < b < 1

    def test_bernstein(self):
        b = bernstein_bound(n=100, sigma2=0.25, b=1.0, t=0.1)
        assert 0 < b < 1

    def test_chernoff(self):
        b = chernoff_bound(p=0.5, n=100, delta=0.2)
        assert 0 < b < 1


def test_empirical_verification():
    rng = np.random.default_rng(0)
    def sampler(m):
        return rng.uniform(0, 1, size=m)
    results = verify_bounds_empirically(sampler, n=100, trials=200)
    assert 'hoeffding_violations' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


