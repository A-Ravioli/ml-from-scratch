"""
Module 6 exercises: SVD, low-rank approximation, and PCA.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def compute_svd(A: np.ndarray, full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the singular value decomposition."""
    return np.linalg.svd(np.asarray(A, dtype=float), full_matrices=full_matrices)


def reconstruct_from_svd(
    U: np.ndarray,
    singular_values: np.ndarray,
    Vt: np.ndarray,
    rank: Optional[int] = None,
) -> np.ndarray:
    """Reconstruct a matrix from its SVD, optionally truncating the rank."""
    singular_values = np.asarray(singular_values, dtype=float)
    if rank is None:
        rank = len(singular_values)
    rank = min(rank, len(singular_values))
    return U[:, :rank] @ np.diag(singular_values[:rank]) @ Vt[:rank, :]


def low_rank_approximation(A: np.ndarray, rank: int) -> np.ndarray:
    """Return the best rank-k approximation from the truncated SVD."""
    U, singular_values, Vt = compute_svd(A, full_matrices=False)
    return reconstruct_from_svd(U, singular_values, Vt, rank=rank)


def reconstruction_error_curve(A: np.ndarray) -> np.ndarray:
    """Compute Frobenius reconstruction errors for all truncation ranks."""
    A = np.asarray(A, dtype=float)
    U, singular_values, Vt = compute_svd(A, full_matrices=False)
    errors = []
    for rank in range(1, len(singular_values) + 1):
        approx = reconstruct_from_svd(U, singular_values, Vt, rank=rank)
        errors.append(np.linalg.norm(A - approx, ord="fro"))
    return np.asarray(errors)


def compress_grayscale_image(image: np.ndarray, rank: int) -> np.ndarray:
    """Compress a grayscale image array with truncated SVD."""
    image = np.asarray(image, dtype=float)
    compressed = low_rank_approximation(image, rank)
    return np.clip(compressed, 0.0, 255.0)


def singular_value_spectrum(A: np.ndarray) -> np.ndarray:
    """Return the singular values in descending order."""
    _, singular_values, _ = compute_svd(A, full_matrices=False)
    return singular_values


def pca_via_svd(X: np.ndarray, n_components: int) -> Dict[str, np.ndarray]:
    """Perform PCA by centering the data and applying SVD."""
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0, keepdims=True)
    centered = X - mean
    U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    transformed = centered @ components.T
    explained_variance = (singular_values**2) / max(len(X) - 1, 1)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance[:n_components] / total_variance
    return {
        "mean": mean,
        "components": components,
        "transformed": transformed,
        "explained_variance_ratio": explained_variance_ratio,
    }


def pseudoinverse_via_svd(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Compute the Moore-Penrose pseudoinverse from the SVD."""
    U, singular_values, Vt = compute_svd(A, full_matrices=False)
    reciprocal = np.array([1.0 / s if s > rcond else 0.0 for s in singular_values])
    return Vt.T @ np.diag(reciprocal) @ U.T


def solve_overdetermined_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve least squares using the SVD pseudoinverse."""
    return pseudoinverse_via_svd(A) @ np.asarray(b, dtype=float)


if __name__ == "__main__":
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    U, s, Vt = compute_svd(A)
    print("Singular values:", s)
