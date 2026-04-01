"""
Reference solution for Module 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    basis_vectors,
    benchmark_matmul,
    dot_product,
    generate_special_matrices,
    matmul_triple_loop,
    plot_2d_vectors,
    plot_column_space_2d,
    scalar_multiply,
    unit_vector,
    vector_add,
    verify_special_matrix_properties,
)

__all__ = [
    "basis_vectors",
    "benchmark_matmul",
    "dot_product",
    "generate_special_matrices",
    "matmul_triple_loop",
    "plot_2d_vectors",
    "plot_column_space_2d",
    "scalar_multiply",
    "unit_vector",
    "vector_add",
    "verify_special_matrix_properties",
]
