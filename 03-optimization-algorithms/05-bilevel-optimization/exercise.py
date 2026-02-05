"""
Bilevel Optimization — Exercises

Implement a tiny bilevel problem and compute hypergradients via implicit differentiation.

Inner problem (ridge regression):
  w*(λ) = argmin_w (1/n)||Xw - y||^2 + λ ||w||^2

Outer problem:
  L_val(w*(λ)) = (1/m)||X_val w*(λ) - y_val||^2

Goal: compute d/dλ L_val(w*(λ)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Array = np.ndarray


def ridge_solution(X: Array, y: Array, lam: float) -> Array:
    """
    Closed-form ridge solution:
      w = (X^T X / n + lam I)^{-1} (X^T y / n)
    """
    # YOUR CODE HERE
    raise NotImplementedError


def validation_loss(X_val: Array, y_val: Array, w: Array) -> float:
    """Mean squared error on validation set."""
    # YOUR CODE HERE
    raise NotImplementedError


def hypergradient_lambda(X: Array, y: Array, X_val: Array, y_val: Array, lam: float) -> float:
    """
    Compute d/dλ validation_loss(w*(λ)) using implicit differentiation.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def finite_difference_hypergradient(
    X: Array, y: Array, X_val: Array, y_val: Array, lam: float, eps: float = 1e-5
) -> float:
    """Numerical hypergradient for verification."""
    w_plus = ridge_solution(X, y, lam + eps)
    w_minus = ridge_solution(X, y, lam - eps)
    l_plus = validation_loss(X_val, y_val, w_plus)
    l_minus = validation_loss(X_val, y_val, w_minus)
    return (l_plus - l_minus) / (2 * eps)

