"""
Tests for Martingales utilities.
"""

import numpy as np
import pytest

from exercise import (
    is_martingale, optional_stopping_demo, azuma_hoeffding_bound
)


class TestMartingale:
    def test_is_martingale_random_walk(self):
        rng = np.random.default_rng(0)
        n_paths, T = 200, 200
        steps = rng.choice([-1.0, 1.0], size=(n_paths, T))
        paths = np.zeros((n_paths, T+1))
        paths[:, 1:] = np.cumsum(steps, axis=1)
        assert is_martingale(paths, tol=0.2)

    def test_azuma_bound(self):
        b = azuma_hoeffding_bound(t=5.0, c=1.0, n=100)
        assert 0 < b < 1

    def test_optional_stopping(self):
        res = optional_stopping_demo()
        assert 'E_X_tau' in res and 'E_X_0' in res


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


