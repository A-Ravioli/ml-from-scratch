"""
Quasi-Newton Methods - Reference Solutions

This module mirrors `exercise.py` and provides complete implementations of:
- Quadratic, Rosenbrock, logistic regression problems
- Quasi-Newton base optimizer with Armijo line search
- BFGS, L-BFGS, DFP, and SR1-style updates (inverse Hessian form)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class OptimizationProblem(ABC):
    def __init__(self, dim: int):
        self.dim = int(dim)

    @abstractmethod
    def objective(self, x: np.ndarray) -> float: ...

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray: ...

    def hessian(self, x: np.ndarray) -> Optional[np.ndarray]:
        return None

    @abstractmethod
    def optimal_point(self) -> np.ndarray: ...


class QuadraticProblem(OptimizationProblem):
    def __init__(self, dim: int, condition_number: float = 10.0):
        super().__init__(dim)
        self.condition_number = float(condition_number)
        self.A: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self._generate_problem_data()

    def _generate_problem_data(self):
        rng = np.random.default_rng(0)
        M = rng.normal(size=(self.dim, self.dim))
        Q, _ = np.linalg.qr(M)
        evals = np.linspace(1.0, self.condition_number, self.dim)
        A = Q @ np.diag(evals) @ Q.T
        A = (A + A.T) / 2.0
        self.A = A
        self.b = rng.normal(size=(self.dim,))

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        assert self.A is not None and self.b is not None
        return float(0.5 * x @ self.A @ x - self.b @ x)

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
    def __init__(self, dim: int):
        super().__init__(dim)
        if self.dim < 2:
            raise ValueError("Rosenbrock requires at least 2 dimensions")

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

    def optimal_point(self) -> np.ndarray:
        return np.ones(self.dim)


class LogisticRegressionProblem(OptimizationProblem):
    def __init__(self, n_samples: int, dim: int, regularization: float = 0.01):
        super().__init__(dim)
        self.n_samples = int(n_samples)
        self.regularization = float(regularization)
        self.features: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self._opt_cache: Optional[np.ndarray] = None
        self._generate_data()

    def _generate_data(self):
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
        s = self._sigmoid(-z)
        grad = -(self.features.T @ (self.labels * s)) / self.n_samples
        return grad + self.regularization * x

    def optimal_point(self) -> np.ndarray:
        if self._opt_cache is not None:
            return self._opt_cache
        from scipy.optimize import minimize

        x0 = np.zeros(self.dim)
        res = minimize(self.objective, x0, jac=self.gradient, method="BFGS")
        self._opt_cache = np.asarray(res.x, dtype=float)
        return self._opt_cache


class QuasiNewtonOptimizer(ABC):
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6, line_search: bool = True):
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.line_search = bool(line_search)

        self.history: Dict[str, List[float]] = {
            "objective": [],
            "gradient_norm": [],
            "distance_to_opt": [],
            "step_size": [],
            "hessian_condition": [],
        }

    @abstractmethod
    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray): ...

    @abstractmethod
    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def reset(self): ...

    def optimize(self, problem: OptimizationProblem, x0: np.ndarray) -> Tuple[np.ndarray, Dict]:
        x = np.asarray(x0, dtype=float).copy()
        self.reset()
        g_prev = problem.gradient(x)
        opt = problem.optimal_point()

        for _ in range(self.max_iterations):
            f = problem.objective(x)
            gnorm = float(np.linalg.norm(g_prev))

            self.history["objective"].append(float(f))
            self.history["gradient_norm"].append(gnorm)
            self.history["distance_to_opt"].append(float(np.linalg.norm(x - opt)))

            if gnorm < self.tolerance:
                self.history["step_size"].append(0.0)
                self.history["hessian_condition"].append(float("nan"))
                break

            p = self.get_search_direction(g_prev)
            alpha = 1.0
            if self.line_search:
                alpha = self._line_search(problem, x, p, g_prev)

            x_new = x + alpha * p
            g_new = problem.gradient(x_new)

            s = x_new - x
            y = g_new - g_prev

            self.update_hessian_approximation(s, y)

            self.history["step_size"].append(float(alpha))
            # condition number proxy if inverse Hessian available
            cond = float("nan")
            H = getattr(self, "H", None)
            if isinstance(H, np.ndarray):
                try:
                    cond = float(np.linalg.cond(H))
                except Exception:
                    cond = float("nan")
            self.history["hessian_condition"].append(cond)

            x, g_prev = x_new, g_new

        return x, self.history

    def _line_search(self, problem: OptimizationProblem, x: np.ndarray, direction: np.ndarray, gradient: np.ndarray) -> float:
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        f0 = problem.objective(x)
        gTp = float(np.dot(gradient, direction))
        if gTp >= 0:
            return 0.0
        while alpha > 1e-12:
            xn = x + alpha * direction
            if problem.objective(xn) <= f0 + c1 * alpha * gTp:
                break
            alpha *= rho
        return float(alpha)


class BFGSOptimizer(QuasiNewtonOptimizer):
    def __init__(self, store_inverse: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.store_inverse = bool(store_inverse)
        self.H: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None

    def reset(self):
        self.H = None
        self.B = None

    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        ys = float(np.dot(y, s))
        if ys <= 1e-12:
            return

        n = s.size
        if self.store_inverse:
            if self.H is None:
                self.H = np.eye(n)
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            self.H = V @ self.H @ V.T + rho * np.outer(s, s)
        else:
            if self.B is None:
                self.B = np.eye(n)
            Bs = self.B @ s
            sBs = float(np.dot(s, Bs))
            if sBs <= 1e-12:
                return
            self.B = self.B + np.outer(y, y) / ys - np.outer(Bs, Bs) / sBs

    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        g = np.asarray(gradient, dtype=float)
        n = g.size
        if self.store_inverse:
            if self.H is None:
                self.H = np.eye(n)
            return -(self.H @ g)
        if self.B is None:
            self.B = np.eye(n)
        try:
            return np.linalg.solve(self.B, -g)
        except np.linalg.LinAlgError:
            return -g


class LBFGSOptimizer(QuasiNewtonOptimizer):
    def __init__(self, history_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.history_size = int(history_size)
        self.s_list: Deque[np.ndarray] = deque(maxlen=self.history_size)
        self.y_list: Deque[np.ndarray] = deque(maxlen=self.history_size)

    def reset(self):
        self.s_list.clear()
        self.y_list.clear()

    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        if float(np.dot(s, y)) <= 1e-12:
            return
        self.s_list.append(s)
        self.y_list.append(y)

    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        g = np.asarray(gradient, dtype=float)
        if not self.s_list:
            return -g

        q = g.copy()
        alphas: List[float] = []
        rhos: List[float] = []

        for s, y in zip(reversed(self.s_list), reversed(self.y_list)):
            rho = 1.0 / float(np.dot(y, s))
            a = rho * float(np.dot(s, q))
            q = q - a * y
            alphas.append(a)
            rhos.append(rho)

        # initial scaling
        s_last = self.s_list[-1]
        y_last = self.y_list[-1]
        gamma = float(np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-12))
        r = gamma * q

        for (s, y), a, rho in zip(zip(self.s_list, self.y_list), reversed(alphas), reversed(rhos)):
            b = rho * float(np.dot(y, r))
            r = r + s * (a - b)

        return -r


class DFPOptimizer(QuasiNewtonOptimizer):
    """DFP update in inverse Hessian form."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H: Optional[np.ndarray] = None

    def reset(self):
        self.H = None

    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        ys = float(np.dot(y, s))
        if ys <= 1e-12:
            return
        n = s.size
        if self.H is None:
            self.H = np.eye(n)

        Hy = self.H @ y
        yHy = float(y @ Hy)
        if yHy <= 1e-12:
            return
        self.H = self.H + np.outer(s, s) / ys - np.outer(Hy, Hy) / yHy

    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        g = np.asarray(gradient, dtype=float)
        if self.H is None:
            self.H = np.eye(g.size)
        return -(self.H @ g)


class SROptimizer(QuasiNewtonOptimizer):
    """SR1 update in inverse Hessian form."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H: Optional[np.ndarray] = None

    def reset(self):
        self.H = None

    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        n = s.size
        if self.H is None:
            self.H = np.eye(n)

        Hy = self.H @ y
        u = s - Hy
        denom = float(np.dot(u, y))
        if abs(denom) <= 1e-10:
            return
        self.H = self.H + np.outer(u, u) / denom

    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        g = np.asarray(gradient, dtype=float)
        if self.H is None:
            self.H = np.eye(g.size)
        return -(self.H @ g)


def compare_quasi_newton_methods(problem: OptimizationProblem, methods: Dict[str, QuasiNewtonOptimizer], x0: np.ndarray) -> Dict:
    results = {}
    for name, opt in methods.items():
        start = time.time()
        x_final, hist = opt.optimize(problem, x0)
        results[name] = {
            "final_point": x_final,
            "history": hist,
            "runtime": time.time() - start,
            "final_objective": problem.objective(x_final),
            "final_gradient_norm": float(np.linalg.norm(problem.gradient(x_final))),
        }
    return results


def convergence_rate_analysis(problem: OptimizationProblem, optimizer: QuasiNewtonOptimizer, x0: np.ndarray, n_iter: int = 50) -> Dict:
    x = np.asarray(x0, dtype=float).copy()
    opt = problem.optimal_point()
    distances: List[float] = []
    optimizer.reset()
    g = problem.gradient(x)
    for _ in range(int(n_iter)):
        distances.append(float(np.linalg.norm(x - opt)))
        p = optimizer.get_search_direction(g)
        alpha = optimizer._line_search(problem, x, p, g)
        x_new = x + alpha * p
        g_new = problem.gradient(x_new)
        optimizer.update_hessian_approximation(x_new - x, g_new - g)
        x, g = x_new, g_new
    return {"distances": distances}


def plot_convergence_comparison(results: Dict, problem_name: str):
    plt.figure(figsize=(10, 4))
    for name, r in results.items():
        plt.semilogy(r["history"]["objective"], label=name)
    plt.title(problem_name)
    plt.xlabel("Iteration")
    plt.ylabel("Objective (log)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def exercise_1_bfgs_implementation():
    prob = QuadraticProblem(dim=10, condition_number=20.0)
    x0 = np.random.randn(10)
    opt = BFGSOptimizer(store_inverse=True, max_iterations=200, tolerance=1e-10, line_search=True)
    x_final, hist = opt.optimize(prob, x0)
    return {"final_obj": prob.objective(x_final), "iters": len(hist["objective"])}


def exercise_2_lbfgs_efficiency():
    prob = RosenbrockProblem(dim=2)
    x0 = np.array([-1.2, 1.0])
    opt = LBFGSOptimizer(history_size=5, max_iterations=300, tolerance=1e-8, line_search=True)
    x_final, hist = opt.optimize(prob, x0)
    return {"final_obj": prob.objective(x_final), "iters": len(hist["objective"])}


def exercise_3_method_comparison():
    prob = QuadraticProblem(dim=8, condition_number=50.0)
    x0 = np.random.randn(8)
    methods = {
        "BFGS": BFGSOptimizer(store_inverse=True, max_iterations=200, tolerance=1e-10, line_search=True),
        "L-BFGS": LBFGSOptimizer(history_size=7, max_iterations=200, tolerance=1e-10, line_search=True),
        "DFP": DFPOptimizer(max_iterations=200, tolerance=1e-10, line_search=True),
        "SR1": SROptimizer(max_iterations=200, tolerance=1e-10, line_search=True),
    }
    return compare_quasi_newton_methods(prob, methods, x0)


def exercise_4_line_search_impact():
    prob = RosenbrockProblem(dim=2)
    x0 = np.array([-1.2, 1.0])
    opt_ls = BFGSOptimizer(store_inverse=True, max_iterations=200, tolerance=1e-8, line_search=True)
    opt_nols = BFGSOptimizer(store_inverse=True, max_iterations=200, tolerance=1e-8, line_search=False)
    x1, _ = opt_ls.optimize(prob, x0)
    x2, _ = opt_nols.optimize(prob, x0)
    return {"with_ls": prob.objective(x1), "without_ls": prob.objective(x2)}


def exercise_5_conditioning_analysis():
    prob = QuadraticProblem(dim=10, condition_number=1e3)
    x0 = np.random.randn(10)
    opt = LBFGSOptimizer(history_size=10, max_iterations=200, tolerance=1e-8, line_search=True)
    x_final, _ = opt.optimize(prob, x0)
    return {"final_obj": prob.objective(x_final)}


def exercise_6_practical_considerations():
    return {"note": "Quasi-Newton methods trade memory for curvature information and can be effective without explicit Hessians."}


if __name__ == "__main__":
    print(exercise_1_bfgs_implementation())

