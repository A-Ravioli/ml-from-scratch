"""
Module 5 exercises: eigenvalues, eigenvectors, and the spectral theorem.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def eigendecompose(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors."""
    A = np.asarray(A, dtype=float)
    if np.allclose(A, A.T):
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def transform_vectors(A: np.ndarray, vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Apply A to a collection of vectors."""
    A = np.asarray(A, dtype=float)
    stacked = np.column_stack([np.asarray(v, dtype=float) for v in vectors])
    return A @ stacked


def matrix_power_via_eig(A: np.ndarray, power: int) -> np.ndarray:
    """Compute A^power using eigendecomposition for diagonalizable matrices."""
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(A, dtype=float))
    diagonal_power = np.diag(eigenvalues**power)
    result = eigenvectors @ diagonal_power @ np.linalg.inv(eigenvectors)
    return np.real_if_close(result)


def diagonalize_symmetric(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return Q and Lambda for a symmetric matrix A = Q Lambda Q^T."""
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        raise ValueError("A must be symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return eigenvectors, np.diag(eigenvalues)


def random_symmetric_matrix(dimension: int, seed: int = 0) -> np.ndarray:
    """Generate a random symmetric matrix."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(dimension, dimension))
    return 0.5 * (M + M.T)


def verify_spectral_theorem(A: np.ndarray, tolerance: float = 1e-8) -> Dict[str, bool]:
    """Verify the main claims of the spectral theorem for a symmetric matrix."""
    Q, Lambda = diagonalize_symmetric(A)
    eigenvalues = np.diag(Lambda)
    return {
        "real_eigenvalues": np.all(np.isreal(eigenvalues)),
        "orthogonal_eigenvectors": np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tolerance),
        "reconstruction": np.allclose(Q @ Lambda @ Q.T, A, atol=tolerance),
    }


def is_positive_semidefinite(A: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Check PSD by inspecting eigenvalues."""
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        return False
    eigenvalues = np.linalg.eigvalsh(A)
    return bool(np.all(eigenvalues >= -tolerance))


if __name__ == "__main__":
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    values, vectors = eigendecompose(A)
    print("Eigenvalues:", values)
    print("Eigenvectors:\n", vectors)
