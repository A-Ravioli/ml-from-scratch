"""
Reference solution for Module 8.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exercise import (  # noqa: E402,F401
    Value,
    classify_hessian,
    finite_difference_parameter_gradient,
    least_squares_gradient,
    least_squares_loss,
    numerical_gradient,
    numerical_hessian,
    quadratic_curvature_surface,
    two_layer_forward,
    two_layer_gradients,
)

__all__ = [
    "Value",
    "classify_hessian",
    "finite_difference_parameter_gradient",
    "least_squares_gradient",
    "least_squares_loss",
    "numerical_gradient",
    "numerical_hessian",
    "quadratic_curvature_surface",
    "two_layer_forward",
    "two_layer_gradients",
]
