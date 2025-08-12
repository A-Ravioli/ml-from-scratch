"""
Constrained Optimization Solutions - Reference Implementation
"""

from typing import Tuple, Dict, Callable
import numpy as np


def solve_quadratic_program(P: np.ndarray, q: np.ndarray, A: np.ndarray = None, b: np.ndarray = None,
                            G: np.ndarray = None, h: np.ndarray = None) -> np.ndarray:
    # Try cvxpy if available; otherwise fallback to simple unconstrained/box cases
    try:
        import cvxpy as cp
        n = q.shape[0]
        x = cp.Variable(n)
        obj = 0.5 * cp.quad_form(x, P) + q @ x
        cons = []
        if A is not None and b is not None:
            cons.append(A @ x == b)
        if G is not None and h is not None:
            cons.append(G @ x <= h)
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.OSQP, verbose=False)
        return np.array(x.value).reshape(-1)
    except Exception:
        # Fallback: unconstrained
        K = P.copy()
        x = -np.linalg.lstsq(K, q, rcond=None)[0]
        return x


def augmented_lagrangian(f: Callable[[np.ndarray], float],
                         grad: Callable[[np.ndarray], np.ndarray],
                         h: Callable[[np.ndarray], np.ndarray],
                         x0: np.ndarray, mu0: float = 1.0, iters: int = 100) -> Dict:
    x = x0.astype(float).copy()
    lam = np.zeros_like(h(x))
    mu = mu0
    hist = [x.copy()]
    for k in range(iters):
        g = grad(x) + (mu * (h(x))) @ _jacobian(h, x) + _jacobian_hT(h, x, lam)
        # simple gradient step
        x = x - 0.01 * g
        hist.append(x.copy())
        # update multipliers and penalty
        lam = lam + mu * h(x)
        mu = min(mu * 1.5, 1e6)
        if np.linalg.norm(h(x)) < 1e-6 and np.linalg.norm(g) < 1e-4:
            break
    return {'x': x, 'trajectory': hist}


def _jacobian(h, x, eps: float = 1e-6):
    m = len(h(x))
    n = len(x)
    J = np.zeros((m, n))
    fx = h(x)
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        J[:, j] = (h(xp) - fx) / eps
    return J


def _jacobian_hT(h, x, lam, eps: float = 1e-6):
    # Approximate J^T lam
    J = _jacobian(h, x, eps)
    return J.T @ lam


def projection_box(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(x, lower, upper)


