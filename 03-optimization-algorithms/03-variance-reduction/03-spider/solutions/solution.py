"""
SPIDER reference solution.

Provides a complete, deterministic implementation matching the public API of the
accompanying `exercise.py`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


class OptimizationProblem(ABC):
    def __init__(self, n_samples: int, dim: int):
        self.n_samples = int(n_samples)
        self.dim = int(dim)

    @abstractmethod
    def objective(self, x: np.ndarray) -> float: ...

    @abstractmethod
    def individual_objective(self, x: np.ndarray, i: int) -> float: ...

    @abstractmethod
    def full_gradient(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def individual_gradient(self, x: np.ndarray, i: int) -> np.ndarray: ...

    @abstractmethod
    def batch_gradient(self, x: np.ndarray, batch_indices: np.ndarray) -> np.ndarray: ...

    def optimal_point(self) -> Optional[np.ndarray]:
        return None


class NonConvexQuadratic(OptimizationProblem):
    """
    Non-convex finite sum:
        f(x) = (1/n) sum_i 0.5 (x-a_i)^T A_i (x-a_i)

    A subset of A_i have negative eigenvalues but the gradient is Lipschitz with constant L.
    """

    def __init__(self, n_samples: int, dim: int, negative_curvature_ratio: float = 0.3):
        super().__init__(n_samples, dim)
        self.negative_curvature_ratio = float(negative_curvature_ratio)
        self.A_matrices: List[np.ndarray] = []
        self.centers: np.ndarray = np.zeros((self.n_samples, self.dim))
        self.L: float = 0.0
        self._generate_problem_data()

    def _generate_problem_data(self):
        rng = np.random.default_rng(0)
        self.centers = rng.normal(size=(self.n_samples, self.dim))
        n_neg = int(np.round(self.n_samples * self.negative_curvature_ratio))

        self.A_matrices = []
        max_abs_eig = 0.0
        for i in range(self.n_samples):
            Q, _ = np.linalg.qr(rng.normal(size=(self.dim, self.dim)))
            # Positive spectrum baseline.
            pos_eigs = np.linspace(0.5, 3.0, self.dim)
            eigs = pos_eigs.copy()
            if i < n_neg:
                # Flip a few eigenvalues negative.
                k = max(1, self.dim // 3)
                eigs[:k] = -np.linspace(0.2, 1.0, k)
            A = Q @ np.diag(eigs) @ Q.T
            A = (A + A.T) / 2
            self.A_matrices.append(A)
            max_abs_eig = max(max_abs_eig, float(np.max(np.abs(np.linalg.eigvalsh(A)))))

        self.L = float(max_abs_eig)

    def individual_objective(self, x: np.ndarray, i: int) -> float:
        x = np.asarray(x, dtype=float)
        A = self.A_matrices[int(i)]
        a = self.centers[int(i)]
        d = x - a
        return 0.5 * float(d.T @ A @ d)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.mean([self.individual_objective(x, i) for i in range(self.n_samples)]))

    def individual_gradient(self, x: np.ndarray, i: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        A = self.A_matrices[int(i)]
        a = self.centers[int(i)]
        return A @ (x - a)

    def full_gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.zeros(self.dim, dtype=float)
        for i in range(self.n_samples):
            g += self.individual_gradient(x, i)
        return g / self.n_samples

    def batch_gradient(self, x: np.ndarray, batch_indices: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        batch_indices = np.asarray(batch_indices, dtype=int)
        g = np.zeros(self.dim, dtype=float)
        for i in batch_indices:
            g += self.individual_gradient(x, int(i))
        return g / max(1, batch_indices.size)


class NonConvexLogistic(OptimizationProblem):
    """
    Logistic loss with a non-convex perturbation term:
        f(x) = mean_i log(1+exp(-y_i z_i^T x)) + (λ/2)||x||^2 + α * mean_j cos(x_j)
    """

    def __init__(self, n_samples: int, dim: int, regularization: float = 0.01, nonconvex_strength: float = 0.1):
        super().__init__(n_samples, dim)
        self.regularization = float(regularization)
        self.nonconvex_strength = float(nonconvex_strength)
        self.features: np.ndarray = np.zeros((self.n_samples, self.dim))
        self.labels: np.ndarray = np.zeros(self.n_samples, dtype=int)
        self._generate_data()

    def _generate_data(self):
        rng = np.random.default_rng(0)
        w_true = rng.normal(size=self.dim)
        w_true /= max(1e-8, np.linalg.norm(w_true))
        X = rng.normal(size=(self.n_samples, self.dim))
        y = np.where((X @ w_true + 0.2 * rng.normal(size=self.n_samples)) >= 0.0, 1, -1)
        self.features = X
        self.labels = y.astype(int)

    def individual_objective(self, x: np.ndarray, i: int) -> float:
        x = np.asarray(x, dtype=float)
        z = self.features[int(i)]
        y = float(self.labels[int(i)])
        margin = y * float(z @ x)
        return float(np.logaddexp(0.0, -margin))

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        data_term = np.mean([self.individual_objective(x, i) for i in range(self.n_samples)])
        reg_term = 0.5 * self.regularization * float(x @ x)
        nonconvex_term = self.nonconvex_strength * float(np.mean(np.cos(x)))
        return float(data_term + reg_term + nonconvex_term)

    def individual_gradient(self, x: np.ndarray, i: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = self.features[int(i)]
        y = float(self.labels[int(i)])
        margin = y * float(z @ x)
        sigma = 1.0 / (1.0 + np.exp(margin))
        return (-y * sigma) * z

    def full_gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.zeros(self.dim, dtype=float)
        for i in range(self.n_samples):
            g += self.individual_gradient(x, i)
        g /= self.n_samples
        g += self.regularization * x
        g += -self.nonconvex_strength * np.sin(x) / max(1, self.dim)
        return g

    def batch_gradient(self, x: np.ndarray, batch_indices: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        batch_indices = np.asarray(batch_indices, dtype=int)
        g = np.zeros(self.dim, dtype=float)
        for i in batch_indices:
            g += self.individual_gradient(x, int(i))
        g /= max(1, batch_indices.size)
        g += self.regularization * x
        g += -self.nonconvex_strength * np.sin(x) / max(1, self.dim)
        return g


class SimpleNeuralNetwork(OptimizationProblem):
    """
    Tiny 1-hidden-layer network with squared loss to provide a non-convex objective.
    Parameters are packed as a single vector.
    """

    def __init__(self, n_samples: int, input_dim: int, hidden_dim: int, output_dim: int):
        param_dim = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim
        super().__init__(n_samples, param_dim)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.X: np.ndarray = np.zeros((self.n_samples, self.input_dim))
        self.y: np.ndarray = np.zeros((self.n_samples, self.output_dim))
        self._generate_data()

    def _generate_data(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(self.n_samples, self.input_dim))
        W = rng.normal(size=(self.input_dim, self.output_dim)) / np.sqrt(self.input_dim)
        self.y = np.tanh(self.X @ W) + 0.05 * rng.normal(size=(self.n_samples, self.output_dim))

    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=float)
        d_in, d_h, d_out = self.input_dim, self.hidden_dim, self.output_dim
        o1 = d_in * d_h
        o2 = o1 + d_h
        o3 = o2 + d_h * d_out
        W1 = params[:o1].reshape(d_in, d_h)
        b1 = params[o1:o2]
        W2 = params[o2:o3].reshape(d_h, d_out)
        b2 = params[o3:]
        return W1, b1, W2, b2

    def _forward(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self._unpack_params(params)
        h = np.tanh(x @ W1 + b1)
        return h @ W2 + b2

    def individual_objective(self, params: np.ndarray, i: int) -> float:
        params = np.asarray(params, dtype=float)
        pred = self._forward(params, self.X[int(i)])
        err = pred - self.y[int(i)]
        return 0.5 * float(np.sum(err * err))

    def objective(self, params: np.ndarray) -> float:
        params = np.asarray(params, dtype=float)
        return float(np.mean([self.individual_objective(params, i) for i in range(self.n_samples)]))

    def individual_gradient(self, params: np.ndarray, i: int) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        x = self.X[int(i)]
        y = self.y[int(i)]
        W1, b1, W2, b2 = self._unpack_params(params)

        pre = x @ W1 + b1
        h = np.tanh(pre)
        pred = h @ W2 + b2
        d_out = pred - y

        gW2 = np.outer(h, d_out)
        gb2 = d_out
        dh = W2 @ d_out
        dpre = (1.0 - np.tanh(pre) ** 2) * dh
        gW1 = np.outer(x, dpre)
        gb1 = dpre

        return np.concatenate([gW1.ravel(), gb1.ravel(), gW2.ravel(), gb2.ravel()])

    def full_gradient(self, params: np.ndarray) -> np.ndarray:
        g = np.zeros(self.dim, dtype=float)
        for i in range(self.n_samples):
            g += self.individual_gradient(params, i)
        return g / self.n_samples

    def batch_gradient(self, params: np.ndarray, batch_indices: np.ndarray) -> np.ndarray:
        batch_indices = np.asarray(batch_indices, dtype=int)
        g = np.zeros(self.dim, dtype=float)
        for i in batch_indices:
            g += self.individual_gradient(params, int(i))
        return g / max(1, batch_indices.size)


class SPIDEROptimizer:
    def __init__(self, batch_size_estimator: int, batch_size_update: int, update_frequency: int, step_size: float):
        self.batch_size_estimator = int(batch_size_estimator)
        self.batch_size_update = int(batch_size_update)
        self.update_frequency = int(update_frequency)
        self.step_size = float(step_size)

        self.estimator: Optional[np.ndarray] = None
        self.iteration_count = 0
        self._rng = np.random.default_rng(0)
        self.history: Dict[str, List] = {
            "objective": [],
            "gradient_norm": [],
            "estimator_norm": [],
            "batch_evaluations": 0,
            "variance_estimate": [],
        }

    def reset(self):
        self.estimator = None
        self.iteration_count = 0
        self.history = {
            "objective": [],
            "gradient_norm": [],
            "estimator_norm": [],
            "batch_evaluations": 0,
            "variance_estimate": [],
        }

    def update_estimator(self, problem: OptimizationProblem, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        batch = self._rng.choice(problem.n_samples, size=min(problem.n_samples, self.batch_size_estimator), replace=False)
        self.estimator = problem.batch_gradient(x, batch)
        self.history["batch_evaluations"] += int(batch.size)

    def compute_spider_gradient(self, problem: OptimizationProblem, x_current: np.ndarray, x_previous: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Estimator not initialized.")
        x_current = np.asarray(x_current, dtype=float)
        x_previous = np.asarray(x_previous, dtype=float)
        batch = self._rng.choice(problem.n_samples, size=min(problem.n_samples, self.batch_size_update), replace=False)
        g_cur = problem.batch_gradient(x_current, batch)
        g_prev = problem.batch_gradient(x_previous, batch)
        self.history["batch_evaluations"] += 2 * int(batch.size)
        return g_cur - g_prev + self.estimator

    def step(self, problem: OptimizationProblem, x: np.ndarray, x_previous: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        need_refresh = self.estimator is None or (self.iteration_count % self.update_frequency == 0) or (x_previous is None)
        if need_refresh:
            self.update_estimator(problem, x)
        else:
            self.estimator = self.compute_spider_gradient(problem, x, np.asarray(x_previous, dtype=float))

        x_new = x - self.step_size * self.estimator
        self.iteration_count += 1
        return x_new


class SPIDERSFOOptimizer(SPIDEROptimizer):
    def __init__(self, batch_size_estimator: int, update_frequency: int, step_size: float):
        super().__init__(batch_size_estimator, 1, update_frequency, step_size)

    def compute_spider_gradient(self, problem: OptimizationProblem, x_current: np.ndarray, x_previous: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Estimator not initialized.")
        idx = int(self._rng.integers(0, problem.n_samples))
        g_cur = problem.individual_gradient(np.asarray(x_current, dtype=float), idx)
        g_prev = problem.individual_gradient(np.asarray(x_previous, dtype=float), idx)
        self.history["batch_evaluations"] += 2
        return g_cur - g_prev + self.estimator


def optimize_with_spider(
    problem: OptimizationProblem,
    optimizer: SPIDEROptimizer,
    x0: np.ndarray,
    n_iterations: int = 10000,
    tolerance: float = 1e-6,
    track_progress: bool = True,
) -> Tuple[np.ndarray, Dict]:
    x = np.asarray(x0, dtype=float).copy()
    x_previous: Optional[np.ndarray] = None
    optimizer.reset()

    for _ in range(int(n_iterations)):
        x_new = optimizer.step(problem, x, x_previous=x_previous)
        if track_progress:
            f = problem.objective(x_new)
            g = problem.full_gradient(x_new)
            optimizer.history["objective"].append(float(f))
            optimizer.history["gradient_norm"].append(float(np.linalg.norm(g)))
            optimizer.history["estimator_norm"].append(float(np.linalg.norm(optimizer.estimator) if optimizer.estimator is not None else 0.0))
            optimizer.history["variance_estimate"].append(float(np.var(optimizer.estimator) if optimizer.estimator is not None else 0.0))

        x_previous = x
        x = x_new
        if float(np.linalg.norm(problem.full_gradient(x))) < float(tolerance):
            break

    return x, optimizer.history


def compare_spider_variants(problem: OptimizationProblem, optimizers: Dict[str, SPIDEROptimizer], x0: np.ndarray, n_iterations: int = 10000) -> Dict:
    results: Dict[str, Dict] = {}
    for name, optimizer in optimizers.items():
        start = time.time()
        x_final, history = optimize_with_spider(problem, optimizer, x0, n_iterations=n_iterations)
        end = time.time()
        results[name] = {
            "final_point": x_final,
            "history": history,
            "runtime": end - start,
            "final_objective": problem.objective(x_final),
            "batch_evaluations": history.get("batch_evaluations", 0),
        }
    return results


def hyperparameter_sensitivity_study(
    problem: OptimizationProblem, batch_sizes: List[int], update_frequencies: List[int], step_sizes: List[float], x0: np.ndarray
) -> Dict:
    results: Dict[str, Dict] = {}
    for b1 in batch_sizes:
        for q in update_frequencies:
            for eta in step_sizes:
                key = f"b1={b1}_q={q}_eta={eta:.3f}"
                opt = SPIDEROptimizer(batch_size_estimator=b1, batch_size_update=max(1, b1 // 2), update_frequency=q, step_size=eta)
                x_final, history = optimize_with_spider(problem, opt, x0, n_iterations=200)
                results[key] = {"final_point": x_final, "final_objective": problem.objective(x_final), "history": history}
    return results


def variance_analysis(problem: OptimizationProblem, spider_optimizer: SPIDEROptimizer, x_trajectory: List[np.ndarray]) -> Dict:
    rng = np.random.default_rng(0)
    analysis = {"spider_variance": [], "sgd_variance": [], "path_integration_effect": [], "distances_to_stationary": []}
    for x in x_trajectory:
        x = np.asarray(x, dtype=float)
        full = problem.full_gradient(x)
        grads = []
        for _ in range(min(200, 20 * problem.n_samples)):
            i = int(rng.integers(0, problem.n_samples))
            grads.append(problem.individual_gradient(x, i))
        grads = np.stack(grads, axis=0)
        diff = grads - full[None, :]
        analysis["sgd_variance"].append(float(np.mean(np.sum(diff * diff, axis=1))))
        analysis["spider_variance"].append(float(np.var(spider_optimizer.estimator) if spider_optimizer.estimator is not None else 0.0))
        analysis["path_integration_effect"].append(float(np.linalg.norm(full)))
        analysis["distances_to_stationary"].append(float(np.linalg.norm(full)))
    return analysis


def convergence_rate_analysis(problem: OptimizationProblem, optimizer: SPIDEROptimizer, x0: np.ndarray) -> Dict:
    x_final, history = optimize_with_spider(problem, optimizer, x0, n_iterations=500)
    g = np.asarray(history.get("gradient_norm", []), dtype=float)
    stationarity_reached = bool(g.size > 0 and np.min(g) < 1e-3)
    iters_to = int(np.argmax(g < 1e-3)) if stationarity_reached else None
    return {"convergence_rate": None, "stationarity_reached": stationarity_reached, "iterations_to_convergence": iters_to, "final_point": x_final}


def computational_complexity_study(problem_sizes: List[Tuple[int, int]], methods: List[str]) -> Dict:
    results = {"problem_sizes": problem_sizes, "methods": methods, "gradient_evaluations": {}, "runtimes": {}}
    for n, d in problem_sizes:
        prob = NonConvexQuadratic(n_samples=n, dim=d, negative_curvature_ratio=0.2)
        x0 = np.zeros(d)
        for method in methods:
            if method.lower() == "spider":
                opt = SPIDEROptimizer(batch_size_estimator=min(n, max(10, n // 2)), batch_size_update=min(n, 5), update_frequency=5, step_size=0.1)
            else:
                opt = SPIDEROptimizer(batch_size_estimator=min(n, max(10, n // 2)), batch_size_update=min(n, 1), update_frequency=1, step_size=0.05)
            start = time.time()
            _, history = optimize_with_spider(prob, opt, x0, n_iterations=100, track_progress=False)
            end = time.time()
            key = f"n={n}_d={d}_{method}"
            results["gradient_evaluations"][key] = history.get("batch_evaluations", 0)
            results["runtimes"][key] = end - start
    return results


def plot_spider_analysis(results: Dict, problem_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    for name, result in results.items():
        history = result["history"]
        ax.plot(history.get("objective", []), label=name)
    ax.set_title("Objective Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    for name, result in results.items():
        history = result["history"]
        ax.semilogy(history.get("gradient_norm", []), label=name)
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||∇f(x)||")
    ax.legend()
    ax.grid(True)

    ax = axes[0, 2]
    for name, result in results.items():
        history = result["history"]
        ax.semilogy(history.get("estimator_norm", []), label=name)
    ax.set_title("Estimator Norm")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||v_k||")
    ax.legend()
    ax.grid(True)

    axes[1, 0].set_title("Efficiency (not plotted)")
    axes[1, 1].set_title("Runtime (not plotted)")
    axes[1, 2].set_title("Variance (not plotted)")
    plt.tight_layout()
    plt.suptitle(f"SPIDER Analysis: {problem_name}")


def exercise_1_basic_spider():
    problem = NonConvexQuadratic(n_samples=50, dim=5, negative_curvature_ratio=0.3)
    x0 = np.ones(problem.dim) * 0.5
    opt = SPIDEROptimizer(batch_size_estimator=25, batch_size_update=5, update_frequency=5, step_size=0.1)
    return optimize_with_spider(problem, opt, x0, n_iterations=500)


def exercise_2_nonconvex_convergence():
    problem = NonConvexLogistic(n_samples=200, dim=10, regularization=0.05, nonconvex_strength=0.1)
    x0 = np.zeros(problem.dim)
    opt = SPIDEROptimizer(batch_size_estimator=100, batch_size_update=10, update_frequency=5, step_size=0.5)
    return optimize_with_spider(problem, opt, x0, n_iterations=300)


def main():
    start = time.time()
    x_final, history = exercise_1_basic_spider()
    end = time.time()
    print(f"Completed demo in {end - start:.3f}s; last f={history['objective'][-1]:.6g}")


if __name__ == "__main__":
    main()

