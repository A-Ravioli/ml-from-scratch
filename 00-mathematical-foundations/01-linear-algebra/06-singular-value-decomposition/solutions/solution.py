"""
Reference solution for Module 6.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    compress_grayscale_image,
    compute_svd,
    low_rank_approximation,
    pca_via_svd,
    pseudoinverse_via_svd,
    reconstruct_from_svd,
    reconstruction_error_curve,
    singular_value_spectrum,
    solve_overdetermined_system,
)

__all__ = [
    "compress_grayscale_image",
    "compute_svd",
    "low_rank_approximation",
    "pca_via_svd",
    "pseudoinverse_via_svd",
    "reconstruct_from_svd",
    "reconstruction_error_curve",
    "singular_value_spectrum",
    "solve_overdetermined_system",
]
