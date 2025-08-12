"""
Spectral Theory Exercises
"""

from typing import List, Tuple
import numpy as np


def symmetric_eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """TODO: Return (eigenvalues, eigenvectors) for symmetric A (columns are eigenvectors)."""
    # TODO: Implement this
    pass


def diagonalize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """TODO: Return (V, D) s.t. A = V D V^{-1} when diagonalizable."""
    # TODO: Implement this
    pass


def power_iteration(A: np.ndarray, max_iters: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """TODO: Dominant eigenpair via power iteration."""
    # TODO: Implement this
    pass


def inverse_iteration(A: np.ndarray, mu: float, max_iters: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """TODO: Eigenpair near shift mu via inverse iteration."""
    # TODO: Implement this
    pass


def rayleigh_quotient(A: np.ndarray, x: np.ndarray) -> float:
    """TODO: Return (xᵀ A x) / (xᵀ x)."""
    # TODO: Implement this
    pass


def rayleigh_quotient_iteration(A: np.ndarray, x0: np.ndarray, max_iters: int = 50, tol: float = 1e-12) -> Tuple[float, np.ndarray]:
    """TODO: Rayleigh quotient iteration for fast convergence (symmetric A)."""
    # TODO: Implement this
    pass


def gershgorin_disks(A: np.ndarray) -> List[Tuple[complex, float]]:
    """TODO: Return list of (center, radius) for Gershgorin disks of A."""
    # TODO: Implement this
    pass


def matrix_function_via_eigen(A: np.ndarray, func) -> np.ndarray:
    """TODO: Compute f(A) for diagonalizable A via eigen-decomposition."""
    # TODO: Implement this
    pass


def graph_laplacian_spectrum(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """TODO: Return (eigenvalues, eigenvectors) of unnormalized Laplacian L = D - A."""
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Spectral Theory Exercises")

