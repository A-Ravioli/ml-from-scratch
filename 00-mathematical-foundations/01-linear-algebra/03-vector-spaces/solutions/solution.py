"""
Reference solution for Module 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    change_basis_coordinates,
    compute_four_subspaces,
    coordinates_in_basis,
    linear_independence_via_rank,
    plot_span_2d,
    principal_component_basis,
    rank_deficiency_report,
    vector_from_coordinates,
    verify_rank_nullity,
)

__all__ = [
    "change_basis_coordinates",
    "compute_four_subspaces",
    "coordinates_in_basis",
    "linear_independence_via_rank",
    "plot_span_2d",
    "principal_component_basis",
    "rank_deficiency_report",
    "vector_from_coordinates",
    "verify_rank_nullity",
]
