"""
Information Geometry Exercises
"""

from typing import Callable, Tuple, Dict
import numpy as np


def fisher_information_bernoulli(theta: float) -> float:
    """TODO: Fisher information for Bernoulli(θ): I(θ) = 1/(θ(1-θ))."""
    # TODO: Implement this
    pass


def fisher_information_gaussian(mu: float, sigma: float) -> np.ndarray:
    """TODO: Fisher information matrix for N(μ, σ^2) with params (μ, log σ)."""
    # TODO: Implement this
    pass


def natural_gradient_step(grad_theta: np.ndarray, fisher_matrix: np.ndarray, lr: float) -> np.ndarray:
    """TODO: Return -lr * F^{-1} grad as natural gradient step."""
    # TODO: Implement this
    pass


def gaussian_geodesic_approx(mu0: float, sigma0: float, mu1: float, sigma1: float, t: float) -> Tuple[float, float]:
    """
    TODO: Approximate geodesic between two Gaussians under Fisher metric.
    Use interpolation in (μ/σ, log σ) coordinates as a simple approximation.
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Information Geometry Exercises")

