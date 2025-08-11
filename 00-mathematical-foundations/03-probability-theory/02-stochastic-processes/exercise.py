"""
Stochastic Processes Exercises
"""

from typing import Tuple
import numpy as np


def simulate_markov_chain(P: np.ndarray, x0: int, T: int, rng: np.random.Generator = None) -> np.ndarray:
    """TODO: Simulate DTMC with transition matrix P for T steps from x0."""
    # TODO: Implement this
    pass


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """TODO: Compute stationary distribution π solving πᵀ = πᵀP, sum π=1."""
    # TODO: Implement this
    pass


def mixing_time_upper_bound(P: np.ndarray, epsilon: float = 1e-2) -> float:
    """TODO: Use spectral gap bound (for reversible P) as rough upper bound."""
    # TODO: Implement this
    pass


def simulate_poisson_process(lmbda: float, T: float, rng: np.random.Generator = None) -> np.ndarray:
    """TODO: Simulate Poisson process on [0, T] via exponential inter-arrival times."""
    # TODO: Implement this
    pass


def simulate_ctmc(Q: np.ndarray, x0: int, T: float, rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """TODO: Simulate CTMC with generator Q up to time T using uniformization."""
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Stochastic Processes Exercises")

