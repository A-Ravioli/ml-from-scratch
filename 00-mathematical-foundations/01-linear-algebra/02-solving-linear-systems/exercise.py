"""
Module 2 exercises: geometry and algorithms for linear systems.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve


def classify_linear_system(A: np.ndarray, b: np.ndarray, tolerance: float = 1e-10) -> str:
    """Classify Ax=b as unique, underdetermined, or inconsistent."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have compatible shapes.")

    rank_A = np.linalg.matrix_rank(A, tol=tolerance)
    rank_augmented = np.linalg.matrix_rank(np.hstack([A, b]), tol=tolerance)
    n_variables = A.shape[1]

    if rank_A != rank_augmented:
        return "inconsistent"
    if rank_A < n_variables:
        return "underdetermined"
    return "unique"


def solve_with_numpy(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b with NumPy for square, nonsingular A."""
    return np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))


def perturb_rhs_and_solve(A: np.ndarray, b: np.ndarray, delta_b: np.ndarray) -> Dict[str, np.ndarray]:
    """Solve the original and perturbed systems to study sensitivity."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    delta_b = np.asarray(delta_b, dtype=float)

    return {
        "original_solution": solve_with_numpy(A, b),
        "perturbed_solution": solve_with_numpy(A, b + delta_b),
        "delta_b": delta_b,
    }


def plot_2d_system(
    A: np.ndarray,
    b: np.ndarray,
    x_limits: Tuple[float, float] = (-5.0, 5.0),
    points: int = 200,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 2x2 linear system as the intersection of two lines."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.shape != (2, 2) or b.shape != (2,):
        raise ValueError("A must be 2x2 and b must have shape (2,).")

    xs = np.linspace(x_limits[0], x_limits[1], points)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    for row, target in zip(A, b):
        if abs(row[1]) < 1e-12:
            ax.axvline(target / row[0], linewidth=2)
        else:
            ys = (target - row[0] * xs) / row[1]
            ax.plot(xs, ys, linewidth=2)

    if classify_linear_system(A, b) == "unique":
        solution = solve_with_numpy(A, b)
        ax.scatter(solution[0], solution[1], color="black", zorder=5)

    ax.set_xlim(*x_limits)
    ax.set_ylim(*x_limits)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Geometry of Ax = b")
    ax.grid(True, alpha=0.3)
    return fig, ax


def sample_plane_points(
    coefficients: np.ndarray,
    value: float,
    grid_limits: Tuple[float, float] = (-2.0, 2.0),
    samples: int = 25,
) -> np.ndarray:
    """Sample points on a plane ax + by + cz = d."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (3,):
        raise ValueError("Plane coefficients must have shape (3,).")
    if abs(coefficients[2]) < 1e-12:
        raise ValueError("This helper expects a non-zero z coefficient.")

    xs = np.linspace(grid_limits[0], grid_limits[1], samples)
    ys = np.linspace(grid_limits[0], grid_limits[1], samples)
    points = []
    for x in xs:
        for y in ys:
            z = (value - coefficients[0] * x - coefficients[1] * y) / coefficients[2]
            points.append([x, y, z])
    return np.asarray(points)


def gaussian_elimination_partial_pivot(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using Gaussian elimination with partial pivoting."""
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if b.shape != (n,):
        raise ValueError("b must be a 1D vector with matching dimension.")

    for pivot_idx in range(n):
        max_row = pivot_idx + np.argmax(np.abs(A[pivot_idx:, pivot_idx]))
        if abs(A[max_row, pivot_idx]) < 1e-12:
            raise np.linalg.LinAlgError("Matrix is singular to working precision.")
        if max_row != pivot_idx:
            A[[pivot_idx, max_row]] = A[[max_row, pivot_idx]]
            b[[pivot_idx, max_row]] = b[[max_row, pivot_idx]]

        pivot = A[pivot_idx, pivot_idx]
        for row in range(pivot_idx + 1, n):
            factor = A[row, pivot_idx] / pivot
            A[row, pivot_idx:] -= factor * A[pivot_idx, pivot_idx:]
            b[row] -= factor * b[pivot_idx]

    x = np.zeros(n, dtype=float)
    for row in range(n - 1, -1, -1):
        rhs = b[row] - np.dot(A[row, row + 1 :], x[row + 1 :])
        x[row] = rhs / A[row, row]
    return x


def solve_many_rhs_with_lu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Factor A once and solve for multiple right-hand sides."""
    lu, piv = lu_factor(np.asarray(A, dtype=float))
    return lu_solve((lu, piv), np.asarray(B, dtype=float))


def benchmark_lu_reuse(A: np.ndarray, B: np.ndarray, repeats: int = 5) -> Dict[str, float]:
    """Compare LU reuse against repeated direct solves."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    lu_times = []
    repeated_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        solve_many_rhs_with_lu(A, B)
        lu_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        np.column_stack([np.linalg.solve(A, B[:, idx]) for idx in range(B.shape[1])])
        repeated_times.append(time.perf_counter() - start)

    lu_seconds = float(np.mean(lu_times))
    repeated_seconds = float(np.mean(repeated_times))
    return {
        "lu_seconds": lu_seconds,
        "repeated_seconds": repeated_seconds,
        "speedup": repeated_seconds / max(lu_seconds, 1e-12),
    }


if __name__ == "__main__":
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    print("System type:", classify_linear_system(A, b))
    print("Gaussian elimination solution:", gaussian_elimination_partial_pivot(A, b))
