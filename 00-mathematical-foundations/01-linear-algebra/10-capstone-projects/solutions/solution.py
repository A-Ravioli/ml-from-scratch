"""
Reference solution for Module 10.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    TwoLayerNetScratch,
    add_bias_column,
    linear_regression_four_ways,
    linear_regression_gradient_descent,
    linear_regression_normal_equations,
    linear_regression_qr,
    linear_regression_svd,
    multi_head_attention,
    pca_pipeline,
    scaled_dot_product_attention,
)

__all__ = [
    "TwoLayerNetScratch",
    "add_bias_column",
    "linear_regression_four_ways",
    "linear_regression_gradient_descent",
    "linear_regression_normal_equations",
    "linear_regression_qr",
    "linear_regression_svd",
    "multi_head_attention",
    "pca_pipeline",
    "scaled_dot_product_attention",
]
