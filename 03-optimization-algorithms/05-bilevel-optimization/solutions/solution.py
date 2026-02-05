"""
Bilevel Optimization — Solutions (Reference Implementation)
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def ridge_solution(X: Array, y: Array, lam: float) -> Array:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    A = (X.T @ X) / n + float(lam) * np.eye(d)
    b = (X.T @ y) / n
    return np.linalg.solve(A, b)


def validation_loss(X_val: Array, y_val: Array, w: Array) -> float:
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float)
    w = np.asarray(w, dtype=float)
    r = X_val @ w - y_val
    return float(np.mean(r**2))


def hypergradient_lambda(X: Array, y: Array, X_val: Array, y_val: Array, lam: float) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    n, d = X.shape
    m = X_val.shape[0]

    w = ridge_solution(X, y, lam)
    # Inner optimality: (X^T X / n + lam I) w = X^T y / n
    A = (X.T @ X) / n + float(lam) * np.eye(d)

    # Outer gradient wrt w: ∂L/∂w = (2/m) X_val^T (X_val w - y_val)
    r = X_val @ w - y_val
    dL_dw = (2.0 / m) * (X_val.T @ r)

    # Implicit differentiation: dw/dλ = -A^{-1} (∂(A w - b)/∂λ) = -A^{-1} (I w) = -A^{-1} w
    dw_dlam = -np.linalg.solve(A, w)
    return float(dL_dw @ dw_dlam)


def finite_difference_hypergradient(X: Array, y: Array, X_val: Array, y_val: Array, lam: float, eps: float = 1e-5) -> float:
    w_plus = ridge_solution(X, y, lam + eps)
    w_minus = ridge_solution(X, y, lam - eps)
    l_plus = validation_loss(X_val, y_val, w_plus)
    l_minus = validation_loss(X_val, y_val, w_minus)
    return (l_plus - l_minus) / (2 * eps)

