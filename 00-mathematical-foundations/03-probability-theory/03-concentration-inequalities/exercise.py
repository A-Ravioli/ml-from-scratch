"""
Concentration Inequalities Exercises
"""

from typing import Tuple
import numpy as np


def is_subgaussian(samples: np.ndarray, t_values: np.ndarray, c: float) -> bool:
    """TODO: Check E[e^{tX}] ≤ e^{c² t² / 2} empirically (centered samples)."""
    # TODO: Implement this
    pass


def subgaussian_parameter(samples: np.ndarray, t_values: np.ndarray) -> float:
    """TODO: Estimate minimal c such that mgf bound holds over t_values."""
    # TODO: Implement this
    pass


def hoeffding_bound(n: int, a: float, b: float, t: float) -> float:
    """TODO: Return Hoeffding bound for deviation t of sample mean."""
    # TODO: Implement this
    pass


def bernstein_bound(n: int, sigma2: float, b: float, t: float) -> float:
    """TODO: Return Bernstein bound with variance sigma2 and range bound b."""
    # TODO: Implement this
    pass


def chernoff_bound(p: float, n: int, delta: float) -> float:
    """TODO: Chernoff bound for Binomial(n,p) deviation by factor (1+delta)."""
    # TODO: Implement this
    pass


def verify_bounds_empirically(dist_sampler, n: int, trials: int = 1000) -> dict:
    """TODO: Empirically compare deviations with bounds for a given sampler."""
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Concentration Inequalities Exercises")

