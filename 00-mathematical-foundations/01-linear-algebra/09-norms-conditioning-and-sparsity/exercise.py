"""
Module 9 exercises: norms, conditioning, and sparse matrices.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import sparse


def vector_norms(v: np.ndarray) -> Dict[str, float]:
    """Compute common vector norms."""
    v = np.asarray(v, dtype=float)
    return {
        "l1": float(np.linalg.norm(v, ord=1)),
        "l2": float(np.linalg.norm(v, ord=2)),
        "linf": float(np.linalg.norm(v, ord=np.inf)),
    }


def matrix_norms(A: np.ndarray) -> Dict[str, float]:
    """Compute common matrix norms."""
    A = np.asarray(A, dtype=float)
    return {
        "frobenius": float(np.linalg.norm(A, ord="fro")),
        "spectral": float(np.linalg.norm(A, ord=2)),
        "l1": float(np.linalg.norm(A, ord=1)),
        "linf": float(np.linalg.norm(A, ord=np.inf)),
    }


def unit_ball_points(norm: str, samples: int = 200) -> np.ndarray:
    """Sample the 2D unit-ball boundary for L1, L2, or Linf."""
    if norm == "l2":
        theta = np.linspace(0.0, 2.0 * np.pi, samples)
        return np.column_stack([np.cos(theta), np.sin(theta)])
    if norm == "l1":
        corners = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
        return corners
    if norm == "linf":
        return np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]])
    raise ValueError("norm must be one of {'l1', 'l2', 'linf'}.")


def construct_matrix_with_condition_number(
    dimension: int,
    condition_number: float,
    seed: int = 0,
) -> np.ndarray:
    """Construct a matrix with approximately the requested 2-norm condition number."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    V, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    singular_values = np.geomspace(condition_number, 1.0, dimension)
    return U @ np.diag(singular_values) @ V.T


def perturbation_sensitivity(A: np.ndarray, b: np.ndarray, delta_b: np.ndarray) -> Dict[str, float]:
    """Measure how much the solution changes relative to a perturbation in b."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    delta_b = np.asarray(delta_b, dtype=float)
    x = np.linalg.solve(A, b)
    x_perturbed = np.linalg.solve(A, b + delta_b)
    relative_input = np.linalg.norm(delta_b) / np.linalg.norm(b)
    relative_output = np.linalg.norm(x_perturbed - x) / np.linalg.norm(x)
    return {
        "relative_input_change": float(relative_input),
        "relative_output_change": float(relative_output),
        "amplification": float(relative_output / max(relative_input, 1e-12)),
    }


def gradient_descent_quadratic(Q: np.ndarray, x0: np.ndarray, lr: float, steps: int) -> np.ndarray:
    """Run gradient descent on f(x)=0.5 x^T Q x and return the loss history."""
    Q = np.asarray(Q, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    losses = []
    for _ in range(steps):
        losses.append(0.5 * float(x.T @ Q @ x))
        gradient = Q @ x
        x -= lr * gradient
    return np.asarray(losses)


def build_sparse_formats(A: np.ndarray) -> Dict[str, sparse.spmatrix]:
    """Create COO, CSR, and CSC sparse matrices from a dense array."""
    A = np.asarray(A, dtype=float)
    return {
        "coo": sparse.coo_matrix(A),
        "csr": sparse.csr_matrix(A),
        "csc": sparse.csc_matrix(A),
    }


def sparse_dense_matmul_benchmark(A: np.ndarray, x: np.ndarray, repeats: int = 5) -> Dict[str, float]:
    """Compare sparse and dense matrix-vector multiplication timings."""
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    A_sparse = sparse.csr_matrix(A)
    sparse_times = []
    dense_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        A_sparse @ x
        sparse_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        A @ x
        dense_times.append(time.perf_counter() - start)

    return {
        "sparse_seconds": float(np.mean(sparse_times)),
        "dense_seconds": float(np.mean(dense_times)),
    }


def sparsity_pattern(A: np.ndarray) -> np.ndarray:
    """Return the indices of non-zero entries."""
    return np.argwhere(np.asarray(A) != 0.0)


def graph_laplacian(num_nodes: int, edges: Iterable[Tuple[int, int]], weights: Iterable[float] | None = None) -> sparse.csr_matrix:
    """Build the graph Laplacian L = D - A."""
    edges = list(edges)
    if weights is None:
        weights = [1.0] * len(edges)
    rows = []
    cols = []
    data = []
    degree = np.zeros(num_nodes, dtype=float)

    for (i, j), w in zip(edges, weights):
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])
        degree[i] += w
        degree[j] += w

    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    degree_matrix = sparse.diags(degree)
    return (degree_matrix - adjacency).tocsr()


if __name__ == "__main__":
    A = construct_matrix_with_condition_number(3, 100.0)
    print("Condition number:", np.linalg.cond(A))
