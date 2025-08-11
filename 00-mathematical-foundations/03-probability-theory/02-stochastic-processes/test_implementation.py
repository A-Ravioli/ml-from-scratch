"""
Tests for Stochastic Processes.
"""

import numpy as np
import pytest

from exercise import (
    simulate_markov_chain, stationary_distribution, mixing_time_upper_bound,
    simulate_poisson_process, simulate_ctmc
)


class TestDTMC:
    def test_stationary_distribution_two_state(self):
        P = np.array([[0.9, 0.1], [0.2, 0.8]])
        pi = stationary_distribution(P)
        assert np.allclose(pi @ P, pi)
        assert abs(np.sum(pi) - 1.0) < 1e-10

    def test_simulation(self):
        P = np.array([[0.5, 0.5], [0.1, 0.9]])
        traj = simulate_markov_chain(P, x0=0, T=100)
        assert len(traj) == 101


class TestPoisson:
    def test_poisson_process(self):
        times = simulate_poisson_process(lmbda=5.0, T=2.0)
        assert np.all(np.diff(times) > -1e-12)
        assert times[0] >= 0 and times[-1] <= 2.0 + 1e-12


class TestCTMC:
    def test_ctmc_uniformization(self):
        # Birth-death on 3 states with rates
        Q = np.array([[-1.0, 1.0, 0.0], [0.5, -1.0, 0.5], [0.0, 1.0, -1.0]])
        states, times = simulate_ctmc(Q, x0=0, T=1.0)
        assert states[0] == 0
        assert len(states) == len(times)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


