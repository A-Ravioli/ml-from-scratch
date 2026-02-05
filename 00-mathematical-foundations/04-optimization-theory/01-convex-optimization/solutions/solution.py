"""
Convex Optimization Solutions - Reference Implementation

This mirrors the interfaces in `exercise.py` and provides working
implementations for study and verification.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def is_convex_set(points: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Practical convexity check for a *finite* set of points:
    sample random convex combinations and check they are close to some point in the set.

    Note: a finite set is only truly convex in trivial cases, so this function is
    best understood as a consistency check for discretized convex sets.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return False
    if points.shape[0] == 1:
        return True

    rng = np.random.default_rng(0)
    n = points.shape[0]
    # Build a fast nearest-neighbor check via brute force (n is expected small for exercises)
    for _ in range(200):
        i, j = rng.integers(0, n, size=2)
        t = rng.random()
        z = t * points[i] + (1.0 - t) * points[j]
        d2 = np.sum((points - z) ** 2, axis=1)
        if float(np.min(d2)) > tolerance**2:
            return False
    return True


def check_convex_function(
    f: Callable[[np.ndarray], float], domain_samples: np.ndarray, tolerance: float = 1e-6
) -> Dict[str, bool]:
    domain_samples = np.asarray(domain_samples, dtype=float)
    rng = np.random.default_rng(0)

    def numerical_grad(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            g[i] = (f(xp) - f(xm)) / (2 * eps)
        return g

    # Jensen/convexity inequality check: f(tx+(1-t)y) <= t f(x) + (1-t) f(y)
    jensen_ok = True
    for _ in range(200):
        x = domain_samples[rng.integers(0, domain_samples.shape[0])]
        y = domain_samples[rng.integers(0, domain_samples.shape[0])]
        t = rng.random()
        lhs = f(t * x + (1 - t) * y)
        rhs = t * f(x) + (1 - t) * f(y)
        if lhs - rhs > tolerance:
            jensen_ok = False
            break

    # First-order condition: f(y) >= f(x) + grad(x)^T (y-x)
    first_order_ok = True
    for _ in range(200):
        x = domain_samples[rng.integers(0, domain_samples.shape[0])]
        y = domain_samples[rng.integers(0, domain_samples.shape[0])]
        gx = numerical_grad(x)
        lhs = f(y)
        rhs = f(x) + float(gx @ (y - x))
        if lhs + tolerance < rhs:
            first_order_ok = False
            break

    # Second-order condition: Hessian PSD (approx, via symmetric finite differences + eigenvalues)
    hessian_ok = True
    for _ in range(min(10, domain_samples.shape[0])):
        x = domain_samples[rng.integers(0, domain_samples.shape[0])]
        n = len(x)
        eps = 1e-4
        H = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps)
        H = 0.5 * (H + H.T)
        eigs = np.linalg.eigvalsh(H)
        if np.min(eigs) < -1e-4:
            hessian_ok = False
            break

    return {
        "jensen_condition": jensen_ok,
        "first_order_condition": first_order_ok,
        "second_order_condition": hessian_ok,
    }


class ConvexOptimizer:
    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.objective = objective
        self.gradient = gradient or self._numerical_gradient
        self.hessian = hessian or self._numerical_hessian

    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.objective(x_plus) - self.objective(x_minus)) / (2 * eps)
        return grad

    def _numerical_hessian(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()

                x_pp[i] += eps
                x_pp[j] += eps

                x_pm[i] += eps
                x_pm[j] -= eps

                x_mp[i] -= eps
                x_mp[j] += eps

                x_mm[i] -= eps
                x_mm[j] -= eps

                hess[i, j] = (
                    self.objective(x_pp)
                    - self.objective(x_pm)
                    - self.objective(x_mp)
                    + self.objective(x_mm)
                ) / (4 * eps**2)
        return hess

    def line_search(self, x: np.ndarray, direction: np.ndarray, initial_step: float = 1.0, c1: float = 1e-4) -> float:
        alpha = float(initial_step)
        fx = float(self.objective(x))
        gx = self.gradient(x)
        descent = float(gx @ direction)
        if descent >= 0:
            # Not a descent direction; fall back to small step.
            return 0.0
        while alpha > 1e-16:
            x_new = x + alpha * direction
            if self.objective(x_new) <= fx + c1 * alpha * descent:
                return alpha
            alpha *= 0.5
        return 0.0


class GradientDescent(ConvexOptimizer):
    def optimize(
        self,
        x0: np.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        step_size: Optional[float] = None,
    ) -> Dict:
        x = np.asarray(x0, dtype=float).copy()
        history = {"x": [x.copy()], "objective": [float(self.objective(x))], "grad_norm": []}

        for _ in range(max_iterations):
            g = self.gradient(x)
            gn = float(np.linalg.norm(g))
            history["grad_norm"].append(gn)
            if gn <= tolerance:
                break

            d = -g
            alpha = float(step_size) if step_size is not None else self.line_search(x, d)
            x = x + alpha * d
            history["x"].append(x.copy())
            history["objective"].append(float(self.objective(x)))

        return {"x_final": x, "history": history, "converged": history["grad_norm"][-1] <= tolerance if history["grad_norm"] else True}


class NewtonMethod(ConvexOptimizer):
    def optimize(self, x0: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-8) -> Dict:
        x = np.asarray(x0, dtype=float).copy()
        history = {"x": [x.copy()], "objective": [float(self.objective(x))], "grad_norm": []}

        for _ in range(max_iterations):
            g = self.gradient(x)
            gn = float(np.linalg.norm(g))
            history["grad_norm"].append(gn)
            if gn <= tolerance:
                break

            H = self.hessian(x)
            # Regularize slightly for numerical stability
            H = 0.5 * (H + H.T) + 1e-12 * np.eye(len(x))
            try:
                step_dir = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step_dir = -g

            alpha = self.line_search(x, step_dir)
            x = x + alpha * step_dir
            history["x"].append(x.copy())
            history["objective"].append(float(self.objective(x)))

        return {"x_final": x, "history": history, "converged": history["grad_norm"][-1] <= tolerance if history["grad_norm"] else True}


class ProjectedGradientDescent(ConvexOptimizer):
    def __init__(
        self,
        objective: Callable,
        gradient: Optional[Callable] = None,
        projection: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(objective, gradient)
        self.projection = projection or (lambda x: x)

    def optimize(
        self,
        x0: np.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        step_size: float = 0.01,
    ) -> Dict:
        x = self.projection(np.asarray(x0, dtype=float).copy())
        history = {"x": [x.copy()], "objective": [float(self.objective(x))], "grad_norm": []}

        for _ in range(max_iterations):
            g = self.gradient(x)
            gn = float(np.linalg.norm(g))
            history["grad_norm"].append(gn)
            if gn <= tolerance:
                break
            x = self.projection(x - float(step_size) * g)
            history["x"].append(x.copy())
            history["objective"].append(float(self.objective(x)))

        return {"x_final": x, "history": history, "converged": history["grad_norm"][-1] <= tolerance if history["grad_norm"] else True}


def projection_onto_simplex(x: np.ndarray) -> np.ndarray:
    """
    Project onto the probability simplex {z: z_i >= 0, sum z_i = 1}.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Simplex projection expects a 1D vector")
    n = x.shape[0]
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_candidates = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho_candidates.size == 0:
        # all entries project to uniform
        return np.ones_like(x) / n
    rho = int(rho_candidates[-1])
    theta = cssv[rho] / (rho + 1)
    w = np.maximum(x - theta, 0.0)
    # normalize for numerical drift
    s = w.sum()
    if s <= 0:
        return np.ones_like(x) / n
    return w / s

