"""
Reference solution for Module 5.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    diagonalize_symmetric,
    eigendecompose,
    is_positive_semidefinite,
    matrix_power_via_eig,
    random_symmetric_matrix,
    transform_vectors,
    verify_spectral_theorem,
)

__all__ = [
    "diagonalize_symmetric",
    "eigendecompose",
    "is_positive_semidefinite",
    "matrix_power_via_eig",
    "random_symmetric_matrix",
    "transform_vectors",
    "verify_spectral_theorem",
]
