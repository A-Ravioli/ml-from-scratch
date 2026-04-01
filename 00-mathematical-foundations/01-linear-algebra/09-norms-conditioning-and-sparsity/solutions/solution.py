"""
Reference solution for Module 9.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    build_sparse_formats,
    construct_matrix_with_condition_number,
    gradient_descent_quadratic,
    graph_laplacian,
    matrix_norms,
    perturbation_sensitivity,
    sparse_dense_matmul_benchmark,
    sparsity_pattern,
    unit_ball_points,
    vector_norms,
)

__all__ = [
    "build_sparse_formats",
    "construct_matrix_with_condition_number",
    "gradient_descent_quadratic",
    "graph_laplacian",
    "matrix_norms",
    "perturbation_sensitivity",
    "sparse_dense_matmul_benchmark",
    "sparsity_pattern",
    "unit_ball_points",
    "vector_norms",
]
