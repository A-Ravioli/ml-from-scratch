"""
Martingales Exercises
"""

from typing import Tuple
import numpy as np


def is_martingale(paths: np.ndarray, tol: float = 1e-2) -> bool:
    """
    TODO: Empirically check martingale property on sample paths.
    paths: shape (n_paths, T+1). Check E[X_{t+1} - X_t | past] ≈ 0.
    """
    # TODO: Implement this
    pass


def optional_stopping_demo(rng: np.random.Generator = None) -> dict:
    """TODO: Simulate a simple random walk and stopping time; compare E[X_τ] and E[X_0]."""
    # TODO: Implement this
    pass


def azuma_hoeffding_bound(t: float, c: float, n: int) -> float:
    """TODO: Return exp(-2 t^2 / (n c^2)) for bounded increments |X_{i}-X_{i-1}| ≤ c."""
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Martingales Exercises")

