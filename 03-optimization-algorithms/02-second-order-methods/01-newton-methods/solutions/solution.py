"""
Newton Methods - Reference Solutions

This file mirrors `exercise.py` and provides complete implementations of:
- Quadratic, Rosenbrock, and logistic regression optimization problems
- Newton, damped Newton, and trust-region Newton optimizers
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


class OptimizationProblem(ABC):
    """Base class for optimization problems with second-order information."""

    def __init__(self, dim: int):
        self.dim = int(dim)

    @abstractmethod
    def objective(self, x: np.ndarray) -> float: ...

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def hessian(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def optimal_point(self) -> np.ndarray: ...


class QuadraticProblem(OptimizationProblem):
    """Quadratic problem: f(x) = 1/2 x^T A x - b^T x + c."""

    def __init__(self, dim: int, condition_number: float = 10.0):
        super().__init__(dim)
        self.A: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.c: float = 0.0
        self._generate_problem_data(float(condition_number))

    def _generate_problem_data(self, condition_number: float):
        rng = np.random.default_rng(0)

        # Random orthogonal Q
        M = rng.normal(size=(self.dim, self.dim))
        Q, _ = np.linalg.qr(M)

        # Eigenvalues from 1..condition_number
        evals = np.linspace(1.0, float(condition_number), self.dim)
        A = Q @ np.diag(evals) @ Q.T
        A = (A + A.T) / 2.0

        self.A = A
        self.b = rng.normal(size=(self.dim,))
        self.c = 0.0

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        assert self.A is not None and self.b is not None
        return float(0.5 * x @ self.A @ x - self.b @ x + self.c)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        assert self.A is not None and self.b is not None
        return self.A @ x - self.b

    def hessian(self, x: np.ndarray) -> np.ndarray:
        assert self.A is not None
        return self.A

    def optimal_point(self) -> np.ndarray:
        assert self.A is not None and self.b is not None
        return np.linalg.solve(self.A, self.b)


class RosenbrockProblem(OptimizationProblem):
    """Rosenbrock function."""

    def __init__(self, dim: int):
        super().__init__(dim)
        if self.dim < 2:
            raise ValueError("Rosenbrock function requires at least 2 dimensions")

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        s = 0.0
        for i in range(self.dim - 1):
            s += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        return float(s)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        g[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
        for i in range(1, self.dim - 1):
            g[i] = (
                200.0 * (x[i] - x[i - 1] ** 2)
                - 400.0 * x[i] * (x[i + 1] - x[i] ** 2)
                - 2.0 * (1.0 - x[i])
            )
        g[self.dim - 1] = 200.0 * (x[self.dim - 1] - x[self.dim - 2] ** 2)
        return g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        H = np.zeros((self.dim, self.dim), dtype=float)
        H[0, 0] = 1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0
        H[0, 1] = -400.0 * x[0]
        for i in range(1, self.dim - 1):
            H[i, i] = 1200.0 * x[i] ** 2 - 400.0 * x[i + 1] + 202.0
            H[i, i - 1] = -400.0 * x[i - 1]
            H[i, i + 1] = -400.0 * x[i]
        H[self.dim - 1, self.dim - 1] = 200.0
        H[self.dim - 1, self.dim - 2] = -400.0 * x[self.dim - 2]
        return H

    def optimal_point(self) -> np.ndarray:
        return np.ones(self.dim)


class LogisticRegressionProblem(OptimizationProblem):
    """L2-regularized logistic regression."""

    def __init__(self, n_samples: int, dim: int, regularization: float = 0.01):
        super().__init__(dim)
        self.n_samples = int(n_samples)
        self.regularization = float(regularization)

        self.features: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self._opt_cache: Optional[np.ndarray] = None
        self._generate_classification_data()

    def _generate_classification_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(self.n_samples, self.dim))
        w_true = rng.normal(size=(self.dim,))
        logits = X @ w_true + 0.2 * rng.normal(size=(self.n_samples,))
        y = np.where(logits >= 0, 1.0, -1.0)
        self.features = X
        self.labels = y

    @staticmethod
    def _sigmoid(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        pos = t >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
        exp_t = np.exp(t[neg])
        out[neg] = exp_t / (1.0 + exp_t)
        return out

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        assert self.features is not None and self.labels is not None
        z = self.labels * (self.features @ x)
        loss = np.mean(np.log1p(np.exp(-z)))
        reg = 0.5 * self.regularization * float(np.dot(x, x))
        return float(loss + reg)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        assert self.features is not None and self.labels is not None
        z = self.labels * (self.features @ x)
        s = self._sigmoid(-z)  # sigmoid(-y x^T w)
        grad = -(self.features.T @ (self.labels * s)) / self.n_samples
        grad = grad + self.regularization * x
        return grad

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        assert self.features is not None and self.labels is not None
        z = self.labels * (self.features @ x)
        s = self._sigmoid(-z)
        w = s * (1.0 - s)
        Xw = self.features * w[:, None]
        H = (self.features.T @ Xw) / self.n_samples
        H = H + self.regularization * np.eye(self.dim)
        return H

    def optimal_point(self) -> np.ndarray:
        if self._opt_cache is not None:
            return self._opt_cache
        from scipy.optimize import minimize

        x0 = np.zeros(self.dim)
        res = minimize(self.objective, x0, jac=self.gradient, hess=self.hessian, method="Newton-CG")
        self._opt_cache = np.asarray(res.x, dtype=float)
        return self._opt_cache


class NewtonOptimizer:
    """Pure Newton's method."""

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6, line_search: bool = True):
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.line_search = bool(line_search)

        self.history: Dict[str, List[float]] = {
            "objective": [],
            "gradient_norm": [],
            "distance_to_opt": [],
            "step_size": [],
            "condition_number": [],
        }

    def optimize(self, problem: OptimizationProblem, x0: np.ndarray) -> Tuple[np.ndarray, Dict]:
        x = np.asarray(x0, dtype=float).copy()
        opt = problem.optimal_point()

        for _ in range(self.max_iterations):
            g = problem.gradient(x)
            H = problem.hessian(x)
            gnorm = float(np.linalg.norm(g))

            self.history["objective"].append(problem.objective(x))
            self.history["gradient_norm"].append(gnorm)
            self.history["distance_to_opt"].append(float(np.linalg.norm(x - opt)))
            self.history["condition_number"].append(float(np.linalg.cond(H)))

            if gnorm < self.tolerance:
                self.history["step_size"].append(0.0)
                break

            p = self._solve_newton_system(H, g)
            alpha = 1.0
            if self.line_search:
                alpha = self._line_search(problem, x, p, initial_step=1.0)

            x = x + alpha * p
            self.history["step_size"].append(float(alpha))

        return x, self.history

    def _solve_newton_system(self, hessian: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        H = np.asarray(hessian, dtype=float)
        g = np.asarray(gradient, dtype=float)
        try:
            c, lower = scipy.linalg.cho_factor(H, lower=True, check_finite=False)
            p = scipy.linalg.cho_solve((c, lower), -g, check_finite=False)
            return np.asarray(p, dtype=float)
        except Exception:
            try:
                return np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                p, *_ = np.linalg.lstsq(H, -g, rcond=None)
                return np.asarray(p, dtype=float)

    def _line_search(self, problem: OptimizationProblem, x: np.ndarray, direction: np.ndarray, initial_step: float = 1.0) -> float:
        alpha = float(initial_step)
        c1 = 1e-4
        rho = 0.5
        f0 = problem.objective(x)
        g0 = problem.gradient(x)
        gTp = float(np.dot(g0, direction))
        if gTp >= 0:
            return 0.0

        while alpha > 1e-10:
            xn = x + alpha * direction
            if problem.objective(xn) <= f0 + c1 * alpha * gTp:
                break
            alpha *= rho
        return float(alpha)


class DampedNewtonOptimizer(NewtonOptimizer):
    """Damped Newton method: solve (H + λI)p = -g with λ chosen for stability."""

    def __init__(self, damping_strategy: str = "adaptive", initial_damping: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.damping_strategy = str(damping_strategy)
        self.initial_damping = float(initial_damping)
        self.current_damping = float(initial_damping)

    def _solve_newton_system(self, hessian: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        H = np.asarray(hessian, dtype=float)
        g = np.asarray(gradient, dtype=float)

        lam = self.current_damping
        if self.damping_strategy == "adaptive":
            try:
                min_eig = float(np.min(np.linalg.eigvalsh((H + H.T) / 2.0)))
            except Exception:
                min_eig = -1.0
            if min_eig <= 0:
                lam = max(lam, -min_eig + self.initial_damping)
            else:
                lam = max(self.initial_damping * 0.1, lam * 0.5)
        elif self.damping_strategy == "fixed":
            lam = self.initial_damping
        elif self.damping_strategy == "levenberg_marquardt":
            # scale with gradient norm
            lam = self.initial_damping * (1.0 + float(np.linalg.norm(g)))

        self.current_damping = float(lam)
        Hreg = H + lam * np.eye(H.shape[0])
        try:
            return np.linalg.solve(Hreg, -g)
        except np.linalg.LinAlgError:
            p, *_ = np.linalg.lstsq(Hreg, -g, rcond=None)
            return np.asarray(p, dtype=float)


class TrustRegionNewton:
    """Trust-region Newton method using a dogleg step."""

    def __init__(
        self,
        initial_radius: float = 1.0,
        max_radius: float = 10.0,
        eta1: float = 0.25,
        eta2: float = 0.75,
        gamma1: float = 0.5,
        gamma2: float = 2.0,
    ):
        self.initial_radius = float(initial_radius)
        self.max_radius = float(max_radius)
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)

        self.current_radius = float(initial_radius)
        self.history: Dict[str, List[float]] = {
            "objective": [],
            "gradient_norm": [],
            "trust_radius": [],
            "predicted_reduction": [],
            "actual_reduction": [],
        }

    def optimize(self, problem: OptimizationProblem, x0: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[np.ndarray, Dict]:
        x = np.asarray(x0, dtype=float).copy()
        Delta = float(self.initial_radius)

        for _ in range(int(max_iterations)):
            f = problem.objective(x)
            g = problem.gradient(x)
            H = problem.hessian(x)
            gnorm = float(np.linalg.norm(g))

            self.history["objective"].append(float(f))
            self.history["gradient_norm"].append(gnorm)
            self.history["trust_radius"].append(Delta)

            if gnorm < float(tolerance):
                self.history["predicted_reduction"].append(0.0)
                self.history["actual_reduction"].append(0.0)
                break

            p = self._dogleg_method(g, H, Delta)
            pred = -(float(g @ p) + 0.5 * float(p @ H @ p))
            xn = x + p
            fn = problem.objective(xn)
            act = float(f - fn)
            rho = act / pred if pred > 0 else -np.inf

            self.history["predicted_reduction"].append(float(pred))
            self.history["actual_reduction"].append(float(act))

            if rho < self.eta1:
                Delta = self.gamma1 * Delta
            else:
                if rho > self.eta2 and abs(np.linalg.norm(p) - Delta) < 1e-8:
                    Delta = min(self.max_radius, self.gamma2 * Delta)

            if rho > self.eta1:
                x = xn

        self.current_radius = float(Delta)
        return x, self.history

    def _solve_trust_region_subproblem(self, gradient: np.ndarray, hessian: np.ndarray) -> np.ndarray:
        # For this educational implementation, use dogleg.
        return self._dogleg_method(gradient, hessian, self.current_radius)

    def _dogleg_method(self, gradient: np.ndarray, hessian: np.ndarray, radius: float) -> np.ndarray:
        g = np.asarray(gradient, dtype=float)
        H = np.asarray(hessian, dtype=float)
        Delta = float(radius)

        # Newton step
        try:
            pN = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            pN = -g

        if np.linalg.norm(pN) <= Delta:
            return pN

        # Cauchy point
        gHg = float(g @ H @ g)
        if gHg <= 0:
            pU = -(Delta / (np.linalg.norm(g) + 1e-12)) * g
            return pU

        alpha = float((g @ g) / gHg)
        pU = -alpha * g

        if np.linalg.norm(pU) >= Delta:
            return -(Delta / (np.linalg.norm(g) + 1e-12)) * g

        # Interpolate between pU and pN: find tau s.t. ||pU + tau*(pN-pU)|| = Delta
        d = pN - pU
        a = float(d @ d)
        b = float(2.0 * pU @ d)
        c = float(pU @ pU - Delta * Delta)
        disc = max(0.0, b * b - 4.0 * a * c)
        tau = (-b + np.sqrt(disc)) / (2.0 * a) if a > 0 else 0.0
        tau = float(np.clip(tau, 0.0, 1.0))
        return pU + tau * d


def compare_newton_methods(problem: OptimizationProblem, methods: Dict[str, object], x0: np.ndarray) -> Dict:
    results = {}
    for name, optimizer in methods.items():
        start_time = time.time()
        x_final, history = optimizer.optimize(problem, x0)
        end_time = time.time()
        results[name] = {
            "final_point": x_final,
            "history": history,
            "runtime": end_time - start_time,
            "final_objective": problem.objective(x_final),
            "final_gradient_norm": float(np.linalg.norm(problem.gradient(x_final))),
        }
    return results


def plot_convergence_comparison(results: Dict, title: str = "Newton Methods Comparison"):
    plt.figure(figsize=(10, 4))
    for name, r in results.items():
        plt.plot(r["history"]["objective"], label=name)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def numerical_vs_analytical_derivatives(problem: OptimizationProblem, x: np.ndarray, eps: float = 1e-6) -> Dict:
    x = np.asarray(x, dtype=float)
    g_true = problem.gradient(x)
    g_num = np.zeros_like(g_true)
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        g_num[i] = (problem.objective(x + d) - problem.objective(x - d)) / (2.0 * eps)
    return {"analytical_grad": g_true, "numerical_grad": g_num, "max_abs_diff": float(np.max(np.abs(g_true - g_num)))}


def exercise_1_quadratic_convergence():
    prob = QuadraticProblem(dim=10, condition_number=50.0)
    x0 = np.random.randn(10)
    opt = NewtonOptimizer(max_iterations=5, tolerance=1e-10, line_search=False)
    x_star, hist = opt.optimize(prob, x0)
    return {"x_final": x_star, "final_obj": prob.objective(x_star), "iters": len(hist["objective"])}


def exercise_2_rosenbrock_optimization():
    prob = RosenbrockProblem(dim=2)
    x0 = np.array([-1.2, 1.0])
    opt = DampedNewtonOptimizer(max_iterations=50, tolerance=1e-10, line_search=True)
    x_final, hist = opt.optimize(prob, x0)
    return {"x_final": x_final, "final_obj": prob.objective(x_final), "iters": len(hist["objective"])}


def exercise_3_logistic_regression():
    prob = LogisticRegressionProblem(n_samples=200, dim=5, regularization=0.1)
    x0 = np.zeros(5)
    opt = NewtonOptimizer(max_iterations=30, tolerance=1e-8, line_search=True)
    x_final, hist = opt.optimize(prob, x0)
    return {"x_final": x_final, "final_obj": prob.objective(x_final), "iters": len(hist["objective"])}


def exercise_4_trust_region_methods():
    prob = RosenbrockProblem(dim=2)
    x0 = np.array([-1.2, 1.0])
    opt = TrustRegionNewton(initial_radius=1.0)
    x_final, hist = opt.optimize(prob, x0, max_iterations=60, tolerance=1e-10)
    return {"x_final": x_final, "final_obj": prob.objective(x_final), "iters": len(hist["objective"])}


def exercise_5_computational_complexity():
    return {"note": "Newton system solve is O(d^3) per iteration for dense Hessians."}


def exercise_6_numerical_stability():
    prob = QuadraticProblem(dim=10, condition_number=1e4)
    x0 = np.random.randn(10)
    opt = DampedNewtonOptimizer(max_iterations=10, tolerance=1e-10, line_search=True)
    x_final, _ = opt.optimize(prob, x0)
    return {"final_obj": prob.objective(x_final)}


if __name__ == "__main__":
    print(exercise_1_quadratic_convergence())

