"""
Reference solution for Module 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    benchmark_lu_reuse,
    classify_linear_system,
    gaussian_elimination_partial_pivot,
    perturb_rhs_and_solve,
    plot_2d_system,
    sample_plane_points,
    solve_many_rhs_with_lu,
    solve_with_numpy,
)

__all__ = [
    "benchmark_lu_reuse",
    "classify_linear_system",
    "gaussian_elimination_partial_pivot",
    "perturb_rhs_and_solve",
    "plot_2d_system",
    "sample_plane_points",
    "solve_many_rhs_with_lu",
    "solve_with_numpy",
]
