"""
Module 3 exercises: span, subspaces, and change of basis.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space, orth


def linear_independence_via_rank(vectors: Sequence[np.ndarray], tolerance: float = 1e-10) -> bool:
    """Check linear independence by comparing rank to the number of vectors."""
    matrix = np.column_stack([np.asarray(vector, dtype=float) for vector in vectors])
    return np.linalg.matrix_rank(matrix, tol=tolerance) == len(vectors)


def plot_span_2d(
    v1: np.ndarray,
    v2: np.ndarray,
    coefficient_range: Tuple[float, float] = (-2.0, 2.0),
    samples: int = 21,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Visualize the span of two 2D vectors by sampling linear combinations."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if v1.shape != (2,) or v2.shape != (2,):
        raise ValueError("Only 2D vectors are supported.")

    coeffs = np.linspace(coefficient_range[0], coefficient_range[1], samples)
    points = np.array([a * v1 + b * v2 for a in coeffs for b in coeffs])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5)
    ax.arrow(0.0, 0.0, v1[0], v1[1], head_width=0.12, color="tab:orange", length_includes_head=True)
    ax.arrow(0.0, 0.0, v2[0], v2[1], head_width=0.12, color="tab:green", length_includes_head=True)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Sampled Span of Two Vectors")
    return fig, ax, points


def rank_deficiency_report(A: np.ndarray, tolerance: float = 1e-10) -> Dict[str, float]:
    """Report matrix rank and whether the matrix is rank deficient."""
    A = np.asarray(A, dtype=float)
    rank = np.linalg.matrix_rank(A, tol=tolerance)
    full_rank = min(A.shape)
    return {
        "rank": int(rank),
        "full_rank": int(full_rank),
        "rank_deficient": bool(rank < full_rank),
    }


def compute_four_subspaces(A: np.ndarray, tolerance: float = 1e-10) -> Dict[str, np.ndarray]:
    """Compute orthonormal bases for the four fundamental subspaces."""
    A = np.asarray(A, dtype=float)
    return {
        "column_space": orth(A, rcond=tolerance),
        "row_space": orth(A.T, rcond=tolerance),
        "null_space": null_space(A, rcond=tolerance),
        "left_null_space": null_space(A.T, rcond=tolerance),
    }


def verify_rank_nullity(A: np.ndarray, tolerance: float = 1e-10) -> Dict[str, int | bool]:
    """Verify rank-nullity numerically."""
    A = np.asarray(A, dtype=float)
    subspaces = compute_four_subspaces(A, tolerance=tolerance)
    rank = subspaces["column_space"].shape[1]
    nullity = subspaces["null_space"].shape[1]
    n = A.shape[1]
    return {
        "rank": rank,
        "nullity": nullity,
        "dimension": n,
        "holds": rank + nullity == n,
    }


def coordinates_in_basis(v: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    """Express a vector in coordinates of the given basis."""
    v = np.asarray(v, dtype=float)
    basis_matrix = np.asarray(basis_matrix, dtype=float)
    return np.linalg.solve(basis_matrix, v)


def vector_from_coordinates(coordinates: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    """Recover a vector from basis coordinates."""
    return np.asarray(basis_matrix, dtype=float) @ np.asarray(coordinates, dtype=float)


def change_basis_coordinates(
    coordinates: np.ndarray,
    source_basis: np.ndarray,
    target_basis: np.ndarray,
) -> np.ndarray:
    """Convert coordinates from one basis description to another."""
    world_vector = vector_from_coordinates(coordinates, source_basis)
    return coordinates_in_basis(world_vector, target_basis)


def principal_component_basis(X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
    """Return the PCA basis vectors as rows using SVD."""
    X = np.asarray(X, dtype=float)
    centered = X - np.mean(X, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    if n_components is None:
        return Vt
    return Vt[:n_components]


if __name__ == "__main__":
    vectors = [np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    print("Independent?", linear_independence_via_rank(vectors))
