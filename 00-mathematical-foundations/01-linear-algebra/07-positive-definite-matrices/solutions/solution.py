"""
Reference solution for Module 7.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    benchmark_cholesky_vs_lu,
    cholesky_pd_test,
    cholesky_solve,
    eigenvalue_pd_test,
    make_positive_definite,
    quadratic_form,
    quadratic_surface,
    sample_multivariate_gaussian,
)

__all__ = [
    "benchmark_cholesky_vs_lu",
    "cholesky_pd_test",
    "cholesky_solve",
    "eigenvalue_pd_test",
    "make_positive_definite",
    "quadratic_form",
    "quadratic_surface",
    "sample_multivariate_gaussian",
]
