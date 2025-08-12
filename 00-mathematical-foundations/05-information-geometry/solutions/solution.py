"""
Information Geometry Solutions - Reference Implementation
"""

from typing import Callable, Tuple, Dict
import numpy as np


def fisher_information_bernoulli(theta: float) -> float:
    eps = 1e-12
    theta = float(np.clip(theta, eps, 1 - eps))
    return 1.0 / (theta * (1.0 - theta))


def fisher_information_gaussian(mu: float, sigma: float) -> np.ndarray:
    # Parameters: (mu, log_sigma). Fisher matrix is diag(1/sigma^2, 2)
    s2 = sigma * sigma
    F = np.array([[1.0 / s2, 0.0], [0.0, 2.0]])
    return F


def natural_gradient_step(grad_theta: np.ndarray, fisher_matrix: np.ndarray, lr: float) -> np.ndarray:
    step = - lr * np.linalg.solve(fisher_matrix, grad_theta)
    return step


def gaussian_geodesic_approx(mu0: float, sigma0: float, mu1: float, sigma1: float, t: float) -> Tuple[float, float]:
    # Interpolate in invariant coordinates u=mu/sigma, v=log sigma
    u0, v0 = mu0 / sigma0, np.log(sigma0)
    u1, v1 = mu1 / sigma1, np.log(sigma1)
    u_t = (1 - t) * u0 + t * u1
    v_t = (1 - t) * v0 + t * v1
    sigma_t = float(np.exp(v_t))
    mu_t = float(u_t * sigma_t)
    return mu_t, sigma_t


