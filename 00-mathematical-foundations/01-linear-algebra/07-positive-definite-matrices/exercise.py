"""
Module 7 exercises: positive definite matrices and Cholesky decomposition.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve


def make_positive_definite(dimension: int, epsilon: float = 1e-3, seed: int = 0) -> np.ndarray:
    """Generate a symmetric positive definite matrix."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(dimension, dimension))
    return M.T @ M + epsilon * np.eye(dimension)


def eigenvalue_pd_test(A: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check positive definiteness using eigenvalues."""
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        return False
    return bool(np.all(np.linalg.eigvalsh(A) > tolerance))


def cholesky_pd_test(A: np.ndarray) -> bool:
    """Check positive definiteness by attempting a Cholesky factorization."""
    try:
        np.linalg.cholesky(np.asarray(A, dtype=float))
        return True
    except np.linalg.LinAlgError:
        return False


def quadratic_form(A: np.ndarray, x: np.ndarray) -> float:
    """Evaluate x^T A x."""
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(x.T @ A @ x)


def quadratic_surface(
    A: np.ndarray,
    grid_limits: Tuple[float, float] = (-2.0, 2.0),
    samples: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the quadratic surface x^T A x for 2D inputs."""
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A must be 2x2 for surface visualization.")

    xs = np.linspace(grid_limits[0], grid_limits[1], samples)
    ys = np.linspace(grid_limits[0], grid_limits[1], samples)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)

    for i in range(samples):
        for j in range(samples):
            vector = np.array([X[i, j], Y[i, j]])
            Z[i, j] = quadratic_form(A, vector)
    return X, Y, Z


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using the Cholesky factorization."""
    factor = cho_factor(np.asarray(A, dtype=float), lower=True)
    return cho_solve(factor, np.asarray(b, dtype=float))


def sample_multivariate_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample from N(mean, covariance) using Cholesky."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(covariance)
    z = rng.normal(size=(n_samples, mean.shape[0]))
    return z @ L.T + mean


def benchmark_cholesky_vs_lu(A: np.ndarray, b: np.ndarray, repeats: int = 5) -> Dict[str, float]:
    """Compare Cholesky and LU solve times for the same SPD system."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    cholesky_times = []
    lu_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        cholesky_solve(A, b)
        cholesky_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        lu, piv = lu_factor(A)
        lu_solve((lu, piv), b)
        lu_times.append(time.perf_counter() - start)

    cholesky_seconds = float(np.mean(cholesky_times))
    lu_seconds = float(np.mean(lu_times))
    return {
        "cholesky_seconds": cholesky_seconds,
        "lu_seconds": lu_seconds,
        "speedup": lu_seconds / max(cholesky_seconds, 1e-12),
    }


if __name__ == "__main__":
    A = make_positive_definite(3, seed=1)
    b = np.array([1.0, 2.0, 3.0])
    print("PD via eigenvalues:", eigenvalue_pd_test(A))
    print("Solve:", cholesky_solve(A, b))
