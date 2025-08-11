"""
Nonconvex Optimization Exercises
"""

from typing import Tuple, Dict
import numpy as np


def find_critical_points_2d(f, grad, hess, grid: Tuple[np.ndarray, np.ndarray]) -> Dict[str, list]:
    """TODO: Search grid for near-stationary points and classify via Hessian eigenvalues."""
    # TODO: Implement this
    pass


def noisy_gradient_descent(f, grad, x0: np.ndarray, step: float, noise_std: float,
                           iters: int = 1000) -> Dict:
    """TODO: Run GD with Gaussian noise; record whether escapes saddles on test functions."""
    # TODO: Implement this
    pass


def trust_region_step(gradx: np.ndarray, hessx: np.ndarray, delta: float) -> np.ndarray:
    """TODO: Cauchy point for trust-region subproblem."""
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Nonconvex Optimization Exercises")

