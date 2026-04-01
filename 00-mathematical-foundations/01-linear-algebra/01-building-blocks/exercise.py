"""
Module 1 exercises: vectors, matrices, and special matrix families.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def vector_add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Add two vectors entry by entry."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("Vectors must have the same shape.")
    return u + v


def scalar_multiply(alpha: float, v: np.ndarray) -> np.ndarray:
    """Scale a vector by a scalar."""
    return float(alpha) * np.asarray(v, dtype=float)


def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the Euclidean dot product without np.dot."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError("Vectors must have the same shape.")
    return float(np.sum(u * v))


def basis_vectors(dimension: int) -> np.ndarray:
    """Return the standard basis vectors as rows."""
    return np.eye(dimension, dtype=float)


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a non-zero vector."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return v / norm


def plot_2d_vectors(
    vectors: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D vectors as arrows from the origin."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    labels = labels or [f"v{i + 1}" for i in range(len(vectors))]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    max_extent = 1.0
    for idx, (vector, label) in enumerate(zip(vectors, labels)):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (2,):
            raise ValueError("Only 2D vectors can be plotted.")
        ax.arrow(
            0.0,
            0.0,
            vector[0],
            vector[1],
            head_width=0.12,
            length_includes_head=True,
            color=colors[idx % len(colors)],
        )
        ax.text(vector[0], vector[1], label)
        max_extent = max(max_extent, np.max(np.abs(vector)) + 0.5)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("2D Vector Geometry")
    ax.grid(True, alpha=0.3)
    return fig, ax


def matmul_triple_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices using the textbook triple loop."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must agree.")

    result = np.zeros((A.shape[0], B.shape[1]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            running_total = 0.0
            for k in range(A.shape[1]):
                running_total += A[i, k] * B[k, j]
            result[i, j] = running_total
    return result


def benchmark_matmul(A: np.ndarray, B: np.ndarray, repeats: int = 5) -> Dict[str, float]:
    """Compare naive matrix multiplication with NumPy's optimized implementation."""
    loop_times = []
    numpy_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        matmul_triple_loop(A, B)
        loop_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        np.matmul(A, B)
        numpy_times.append(time.perf_counter() - start)

    loop_seconds = float(np.mean(loop_times))
    numpy_seconds = float(np.mean(numpy_times))
    speedup = loop_seconds / max(numpy_seconds, 1e-12)
    return {
        "loop_seconds": loop_seconds,
        "numpy_seconds": numpy_seconds,
        "speedup": speedup,
    }


def plot_column_space_2d(
    A: np.ndarray,
    coefficient_range: Tuple[float, float] = (-2.0, 2.0),
    samples: int = 21,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Visualize the 2D column space by sampling linear combinations of columns."""
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A must be a 2x2 matrix for 2D column-space visualization.")

    coeffs = np.linspace(coefficient_range[0], coefficient_range[1], samples)
    points = np.array([A @ np.array([c1, c2]) for c1 in coeffs for c2 in coeffs])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5, color="tab:blue")
    origin = np.zeros(2)
    for idx in range(A.shape[1]):
        ax.arrow(
            origin[0],
            origin[1],
            A[0, idx],
            A[1, idx],
            head_width=0.12,
            color="tab:orange",
            length_includes_head=True,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Sampled Column Space")
    ax.grid(True, alpha=0.3)
    return fig, ax, points


def generate_special_matrices(dimension: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """Generate representative identity, diagonal, symmetric, orthogonal, and triangular matrices."""
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(dimension, dimension))

    return {
        "identity": np.eye(dimension),
        "diagonal": np.diag(rng.uniform(0.5, 2.0, size=dimension)),
        "symmetric": 0.5 * (random_matrix + random_matrix.T),
        "orthogonal": np.linalg.qr(rng.normal(size=(dimension, dimension)))[0],
        "triangular": np.triu(rng.normal(size=(dimension, dimension))),
    }


def verify_special_matrix_properties(
    matrices: Dict[str, np.ndarray],
    tolerance: float = 1e-8,
) -> Dict[str, bool]:
    """Check the defining algebraic properties of each matrix family."""
    identity = matrices["identity"]
    diagonal = matrices["diagonal"]
    symmetric = matrices["symmetric"]
    orthogonal = matrices["orthogonal"]
    triangular = matrices["triangular"]

    return {
        "identity": np.allclose(identity @ identity, identity, atol=tolerance),
        "diagonal": np.allclose(diagonal, np.diag(np.diag(diagonal)), atol=tolerance),
        "symmetric": np.allclose(symmetric, symmetric.T, atol=tolerance),
        "orthogonal": np.allclose(orthogonal.T @ orthogonal, np.eye(orthogonal.shape[0]), atol=tolerance),
        "triangular": np.allclose(np.tril(triangular, k=-1), 0.0, atol=tolerance),
    }


if __name__ == "__main__":
    v1 = np.array([1.0, 2.0])
    v2 = np.array([-0.5, 1.5])
    print("v1 + v2 =", vector_add(v1, v2))
    print("2 * v1 =", scalar_multiply(2.0, v1))
    print("v1 · v2 =", dot_product(v1, v2))

    A = np.array([[2.0, 1.0], [0.0, 1.0]])
    B = np.array([[1.0, 3.0], [2.0, 4.0]])
    print("A @ B (triple loop) =\n", matmul_triple_loop(A, B))
    print("Special matrix checks =", verify_special_matrix_properties(generate_special_matrices(3)))
