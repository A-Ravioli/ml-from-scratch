"""
Module 4 exercises: projections, QR, and orthogonal transforms.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def project_onto_vector(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Project v onto the direction u."""
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    denominator = np.dot(u, u)
    if abs(denominator) < 1e-12:
        raise ValueError("Cannot project onto the zero vector.")
    return (np.dot(v, u) / denominator) * u


def projection_matrix(A: np.ndarray) -> np.ndarray:
    """Build the projection matrix onto the column space of A."""
    A = np.asarray(A, dtype=float)
    gram = A.T @ A
    return A @ np.linalg.inv(gram) @ A.T


def project_onto_subspace(v: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Project v onto the column space of A."""
    return projection_matrix(A) @ np.asarray(v, dtype=float)


def plot_projection_2d(
    v: np.ndarray,
    u: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize projection of v onto u in 2D."""
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    projected = project_onto_vector(v, u)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    for vector, color, label in [
        (u, "tab:orange", "basis"),
        (v, "tab:blue", "v"),
        (projected, "tab:green", "proj(v)"),
        (v - projected, "tab:red", "residual"),
    ]:
        ax.arrow(0.0, 0.0, vector[0], vector[1], head_width=0.12, color=color, length_includes_head=True)
        ax.text(vector[0], vector[1], label)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Projection as a Shadow")
    return fig, ax


def gram_schmidt(vectors: Sequence[np.ndarray], tolerance: float = 1e-10) -> np.ndarray:
    """Run modified Gram-Schmidt and return orthonormal vectors as columns."""
    vectors = [np.asarray(vector, dtype=float) for vector in vectors]
    basis = []
    for vector in vectors:
        w = vector.copy()
        for q in basis:
            w -= np.dot(q, w) * q
        norm = np.linalg.norm(w)
        if norm > tolerance:
            basis.append(w / norm)
    if not basis:
        raise ValueError("Input vectors did not contain a non-zero direction.")
    return np.column_stack(basis)


def qr_via_gram_schmidt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Factor A into QR using modified Gram-Schmidt."""
    A = np.asarray(A, dtype=float)
    q_columns = []
    R = np.zeros((A.shape[1], A.shape[1]), dtype=float)

    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i, q in enumerate(q_columns):
            R[i, j] = np.dot(q, v)
            v -= R[i, j] * q
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-12:
            raise np.linalg.LinAlgError("Columns are linearly dependent.")
        q_columns.append(v / R[j, j])

    Q = np.column_stack(q_columns)
    return Q, R


def qr_condition_numbers(A: np.ndarray) -> Tuple[float, float]:
    """Return condition numbers of A and its Q factor."""
    Q, _ = qr_via_gram_schmidt(A)
    return float(np.linalg.cond(A)), float(np.linalg.cond(Q))


def rotation_matrix(theta: float) -> np.ndarray:
    """Construct the 2D rotation matrix for angle theta."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )


def compose_rotations(theta_1: float, theta_2: float) -> np.ndarray:
    """Compose two 2D rotations."""
    return rotation_matrix(theta_1) @ rotation_matrix(theta_2)


def preserves_norm(Q: np.ndarray, v: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check whether Q preserves the Euclidean norm of v."""
    Q = np.asarray(Q, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.isclose(np.linalg.norm(Q @ v), np.linalg.norm(v), atol=tolerance)


def attention_scores(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Compute unnormalized attention scores as query-key dot products."""
    return np.asarray(queries, dtype=float) @ np.asarray(keys, dtype=float).T


if __name__ == "__main__":
    v = np.array([2.0, 1.0])
    u = np.array([1.0, 0.0])
    print("Projection:", project_onto_vector(v, u))
