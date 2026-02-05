"""
SAGA reference solution.

This file provides a complete implementation matching the public API of the
accompanying `exercise.py`, with deterministic behavior suitable for unit tests.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


class FiniteSumProblem(ABC):
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
    def optimal_point(self) -> np.ndarray: ...


class QuadraticFiniteSum(FiniteSumProblem):
    """
    Finite sum of quadratics with ridge term:
        f(x) = (1/n) sum_i 0.5 (x-a_i)^T A_i (x-a_i) + (λ/2)||x||^2
    """

    def __init__(
        self,
        n_samples: int,
        dim: int,
        condition_number: float = 10.0,
        regularization: float = 0.01,
    ):
        super().__init__(n_samples, dim)
        self.regularization = float(regularization)
        self.A_matrices: List[np.ndarray] = []
        self.centers: np.ndarray = np.zeros((self.n_samples, self.dim))
        self.mu: float = 0.0
        self.L: float = 0.0
        self._generate_problem_data(float(condition_number))

    def _generate_problem_data(self, condition_number: float):
        rng = np.random.default_rng(0)
        self.centers = rng.normal(size=(self.n_samples, self.dim))

        eig_min = 1.0
        eig_max = max(eig_min * 1.01, condition_number)
        base_eigs = np.linspace(eig_min, eig_max, self.dim)
        self.A_matrices = []
        for _ in range(self.n_samples):
            Q, _ = np.linalg.qr(rng.normal(size=(self.dim, self.dim)))
            eigs = np.maximum(1e-3, base_eigs * (1.0 + 0.05 * rng.normal(size=self.dim)))
            A = Q @ np.diag(eigs) @ Q.T
            A = (A + A.T) / 2
            self.A_matrices.append(A)

        H = sum(self.A_matrices) / self.n_samples + self.regularization * np.eye(self.dim)
        H = (H + H.T) / 2
        eigs_H = np.linalg.eigvalsh(H)
        self.mu = float(np.min(eigs_H))
        self.L = float(np.max(eigs_H))

    def individual_objective(self, x: np.ndarray, i: int) -> float:
        x = np.asarray(x, dtype=float)
        A = self.A_matrices[int(i)]
        a = self.centers[int(i)]
        d = x - a
        return 0.5 * float(d.T @ A @ d)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        data_term = np.mean([self.individual_objective(x, i) for i in range(self.n_samples)])
        reg_term = 0.5 * self.regularization * float(x @ x)
        return float(data_term + reg_term)

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
        g /= self.n_samples
        g += self.regularization * x
        return g

    def optimal_point(self) -> np.ndarray:
        H = sum(self.A_matrices) / self.n_samples + self.regularization * np.eye(self.dim)
        b = np.zeros(self.dim, dtype=float)
        for i in range(self.n_samples):
            b += self.A_matrices[i] @ self.centers[i]
        b /= self.n_samples
        return np.linalg.solve(H, b)


class LogisticRegressionFiniteSum(FiniteSumProblem):
    def __init__(self, n_samples: int, dim: int, regularization: float = 0.01):
        super().__init__(n_samples, dim)
        self.regularization = float(regularization)
        self.features: np.ndarray = np.zeros((self.n_samples, self.dim))
        self.labels: np.ndarray = np.zeros(self.n_samples, dtype=int)
        self._opt_cache: Optional[np.ndarray] = None
        self._generate_classification_data()

    def _generate_classification_data(self):
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
        return float(data_term + reg_term)

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
        return g

    def optimal_point(self) -> np.ndarray:
        if self._opt_cache is not None:
            return self._opt_cache.copy()
        x = np.zeros(self.dim, dtype=float)
        z_norm_sq = np.sum(self.features**2, axis=1)
        L = 0.25 * float(np.max(z_norm_sq)) + self.regularization
        step = 0.9 / max(1e-6, L)
        for _ in range(1000):
            x = x - step * self.full_gradient(x)
        self._opt_cache = x.copy()
        return x


class SAGAOptimizer:
    """
    SAGA: maintains a table of past gradients and a running average.
    """

    def __init__(self, step_size: float, memory_efficient: bool = False):
        self.step_size = float(step_size)
        self.memory_efficient = bool(memory_efficient)
        self.gradient_table: Optional[np.ndarray] = None
        self.average_gradient: Optional[np.ndarray] = None
        self.iteration_count = 0
        self._rng = np.random.default_rng(0)
        self._last_update: Optional[np.ndarray] = None
        self.history: Dict[str, List] = {
            "objective": [],
            "gradient_norm": [],
            "distance_to_opt": [],
            "table_staleness": [],
            "variance_estimate": [],
        }

    def initialize_table(self, problem: FiniteSumProblem, x0: np.ndarray):
        x0 = np.asarray(x0, dtype=float)
        table = np.zeros((problem.n_samples, problem.dim), dtype=float)
        for i in range(problem.n_samples):
            table[i] = problem.individual_gradient(x0, i)
        self.gradient_table = table
        self.average_gradient = np.mean(table, axis=0)
        self._last_update = np.zeros(problem.n_samples, dtype=int)

    def update_table_entry(self, problem: FiniteSumProblem, x: np.ndarray, sample_idx: int, new_grad: np.ndarray):
        idx = int(sample_idx)
        if self.gradient_table is None or self.average_gradient is None:
            raise RuntimeError("SAGA table not initialized.")
        old = self.gradient_table[idx].copy()
        self.gradient_table[idx] = new_grad
        self.average_gradient = self.average_gradient + (new_grad - old) / problem.n_samples
        if self._last_update is not None:
            self._last_update[idx] = self.iteration_count

    def compute_saga_gradient(self, problem: FiniteSumProblem, x: np.ndarray, sample_idx: int) -> np.ndarray:
        if self.gradient_table is None or self.average_gradient is None:
            raise RuntimeError("SAGA table not initialized.")
        idx = int(sample_idx)
        x = np.asarray(x, dtype=float)
        current_grad = problem.individual_gradient(x, idx)
        reg = getattr(problem, "regularization", 0.0)
        return current_grad - self.gradient_table[idx] + self.average_gradient + float(reg) * x

    def step(self, problem: FiniteSumProblem, x: np.ndarray) -> np.ndarray:
        if self.gradient_table is None or self.average_gradient is None:
            self.initialize_table(problem, x)

        x = np.asarray(x, dtype=float)
        idx = int(self._rng.integers(0, problem.n_samples))

        current_grad = problem.individual_gradient(x, idx)
        reg = getattr(problem, "regularization", 0.0)
        saga_grad = current_grad - self.gradient_table[idx] + self.average_gradient + float(reg) * x

        x_new = x - self.step_size * saga_grad
        self.update_table_entry(problem, x, idx, current_grad)
        self.iteration_count += 1
        return x_new

    def reset(self):
        self.gradient_table = None
        self.average_gradient = None
        self.iteration_count = 0
        self._last_update = None
        self.history = {
            "objective": [],
            "gradient_norm": [],
            "distance_to_opt": [],
            "table_staleness": [],
            "variance_estimate": [],
        }


class MemoryEfficientSAGA(SAGAOptimizer):
    """
    Toy, memory-efficient variant: store only a fixed subset of coordinates per sample.
    """

    def __init__(self, step_size: float, compression_ratio: float = 0.1):
        super().__init__(step_size, memory_efficient=True)
        self.compression_ratio = float(compression_ratio)
        self.compressed_table: Optional[np.ndarray] = None
        self.compression_indices: Optional[np.ndarray] = None

    def initialize_table(self, problem: FiniteSumProblem, x0: np.ndarray):
        rng = np.random.default_rng(0)
        k = max(1, int(np.ceil(problem.dim * self.compression_ratio)))
        self.compression_indices = np.sort(rng.choice(problem.dim, size=k, replace=False))
        self.compressed_table = np.zeros((problem.n_samples, k), dtype=float)
        avg = np.zeros(k, dtype=float)
        for i in range(problem.n_samples):
            g = problem.individual_gradient(np.asarray(x0, dtype=float), i)
            self.compressed_table[i] = g[self.compression_indices]
            avg += self.compressed_table[i]
        self.average_gradient = np.zeros(problem.dim, dtype=float)
        self.average_gradient[self.compression_indices] = avg / problem.n_samples
        self.gradient_table = np.zeros((problem.n_samples, problem.dim), dtype=float)
        self.gradient_table[:, self.compression_indices] = self.compressed_table
        self._last_update = np.zeros(problem.n_samples, dtype=int)

    def update_table_entry(self, problem: FiniteSumProblem, x: np.ndarray, sample_idx: int, new_grad: np.ndarray):
        if self.compression_indices is None or self.compressed_table is None:
            raise RuntimeError("Compressed table not initialized.")
        idx = int(sample_idx)
        new_small = new_grad[self.compression_indices]
        old_small = self.compressed_table[idx].copy()
        self.compressed_table[idx] = new_small

        old_full = np.zeros(problem.dim, dtype=float)
        new_full = np.zeros(problem.dim, dtype=float)
        old_full[self.compression_indices] = old_small
        new_full[self.compression_indices] = new_small
        if self.gradient_table is None or self.average_gradient is None:
            raise RuntimeError("Internal tables not initialized.")
        self.gradient_table[idx] = new_full
        self.average_gradient = self.average_gradient + (new_full - old_full) / problem.n_samples
        if self._last_update is not None:
            self._last_update[idx] = self.iteration_count


class ProximalSAGA(SAGAOptimizer):
    def __init__(self, step_size: float, prox_operator: Callable[[np.ndarray, float], np.ndarray]):
        super().__init__(step_size)
        self.prox_operator = prox_operator

    def step(self, problem: FiniteSumProblem, x: np.ndarray) -> np.ndarray:
        if self.gradient_table is None or self.average_gradient is None:
            self.initialize_table(problem, x)
        x = np.asarray(x, dtype=float)
        idx = int(self._rng.integers(0, problem.n_samples))
        current_grad = problem.individual_gradient(x, idx)
        reg = getattr(problem, "regularization", 0.0)
        saga_grad = current_grad - self.gradient_table[idx] + self.average_gradient + float(reg) * x
        y = x - self.step_size * saga_grad
        x_new = self.prox_operator(y, self.step_size)
        self.update_table_entry(problem, x, idx, current_grad)
        self.iteration_count += 1
        return x_new


def l1_prox_operator(x: np.ndarray, threshold: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = float(threshold)
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def optimize_with_saga(
    problem: FiniteSumProblem,
    optimizer: SAGAOptimizer,
    x0: np.ndarray,
    n_epochs: int = 100,
    track_progress: bool = True,
) -> Tuple[np.ndarray, Dict]:
    x = np.asarray(x0, dtype=float).copy()
    optimizer.reset()
    optimizer.initialize_table(problem, x)
    x_star = problem.optimal_point()

    for _ in range(int(n_epochs)):
        for _ in range(problem.n_samples):
            x = optimizer.step(problem, x)

        if track_progress:
            f = problem.objective(x)
            g = problem.full_gradient(x)
            optimizer.history["objective"].append(f)
            optimizer.history["gradient_norm"].append(float(np.linalg.norm(g)))
            optimizer.history["distance_to_opt"].append(float(np.linalg.norm(x - x_star)))
            if optimizer._last_update is not None:
                optimizer.history["table_staleness"].append(int(np.max(optimizer._last_update)))
            else:
                optimizer.history["table_staleness"].append(0)
            optimizer.history["variance_estimate"].append(float(np.var(optimizer.gradient_table) if optimizer.gradient_table is not None else 0.0))

    return x, optimizer.history


def compare_saga_variants(problem: FiniteSumProblem, optimizers: Dict[str, SAGAOptimizer], x0: np.ndarray, n_epochs: int = 100) -> Dict:
    results: Dict[str, Dict] = {}
    for name, optimizer in optimizers.items():
        start = time.time()
        x_final, history = optimize_with_saga(problem, optimizer, x0, n_epochs=n_epochs)
        end = time.time()
        results[name] = {
            "final_point": x_final,
            "history": history,
            "runtime": end - start,
            "final_objective": problem.objective(x_final),
            "memory_usage": estimate_memory_usage(optimizer),
        }
    return results


def estimate_memory_usage(optimizer: SAGAOptimizer) -> float:
    bytes_used = 0
    if optimizer.gradient_table is not None:
        bytes_used += int(optimizer.gradient_table.nbytes)
    if optimizer.average_gradient is not None:
        bytes_used += int(optimizer.average_gradient.nbytes)
    if hasattr(optimizer, "compressed_table") and getattr(optimizer, "compressed_table") is not None:
        bytes_used += int(getattr(optimizer, "compressed_table").nbytes)
    return float(bytes_used) / (1024.0 * 1024.0)


def analyze_table_staleness(optimizer: SAGAOptimizer, problem: FiniteSumProblem) -> Dict:
    if optimizer._last_update is None:
        return {"max_staleness": None, "mean_staleness": None, "last_update": None}
    current = int(optimizer.iteration_count)
    staleness = current - optimizer._last_update
    return {
        "max_staleness": int(np.max(staleness)),
        "mean_staleness": float(np.mean(staleness)),
        "last_update": optimizer._last_update.copy(),
    }


def variance_reduction_analysis(problem: FiniteSumProblem, saga_optimizer: SAGAOptimizer, x_trajectory: List[np.ndarray]) -> Dict:
    rng = np.random.default_rng(0)
    analysis = {"saga_variance": [], "sgd_variance": [], "full_gradient_norm": [], "distances_to_opt": []}
    x_star = problem.optimal_point()
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
        analysis["full_gradient_norm"].append(float(np.linalg.norm(full)))
        analysis["distances_to_opt"].append(float(np.linalg.norm(x - x_star)))

        if saga_optimizer.gradient_table is not None and saga_optimizer.average_gradient is not None:
            # A crude "variance" proxy from the gradient table spread.
            analysis["saga_variance"].append(float(np.mean(np.var(saga_optimizer.gradient_table, axis=0))))
        else:
            analysis["saga_variance"].append(0.0)
    return analysis


def step_size_sensitivity_study(problem: FiniteSumProblem, step_sizes: List[float], x0: np.ndarray) -> Dict:
    results: Dict[str, Dict] = {}
    for eta in step_sizes:
        opt = SAGAOptimizer(step_size=float(eta))
        x_final, history = optimize_with_saga(problem, opt, x0, n_epochs=20)
        results[str(eta)] = {"final_point": x_final, "final_objective": problem.objective(x_final), "history": history}
    return results


def memory_usage_scaling_study(dimensions: List[int], sample_sizes: List[int]) -> Dict:
    memory_usage = np.zeros((len(dimensions), len(sample_sizes)))
    theoretical = np.zeros((len(dimensions), len(sample_sizes)))
    for di, d in enumerate(dimensions):
        for ni, n in enumerate(sample_sizes):
            opt = SAGAOptimizer(step_size=0.1)
            # table stores n*d floats
            theoretical[di, ni] = float(n * d * 8) / (1024.0 * 1024.0)
            # approximate actual usage of arrays only
            opt.gradient_table = np.zeros((n, d), dtype=float)
            opt.average_gradient = np.zeros(d, dtype=float)
            memory_usage[di, ni] = estimate_memory_usage(opt)
    return {"dimensions": dimensions, "sample_sizes": sample_sizes, "memory_usage": memory_usage, "theoretical_memory": theoretical}


def plot_saga_analysis(results: Dict, problem_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    for name, result in results.items():
        history = result["history"]
        ax.semilogy(history.get("objective", []), label=name)
    ax.set_title("Objective Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    for name, result in results.items():
        history = result["history"]
        ax.semilogy(history.get("gradient_norm", []), label=name)
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||∇f(x)||")
    ax.legend()
    ax.grid(True)

    ax = axes[0, 2]
    names = list(results.keys())
    mem = [results[name]["memory_usage"] for name in names]
    ax.bar(names, mem)
    ax.set_title("Memory Usage")
    ax.set_ylabel("Memory (MB)")
    plt.setp(ax.get_xticklabels(), rotation=45)

    ax = axes[1, 0]
    runtimes = [results[name]["runtime"] for name in names]
    ax.bar(names, runtimes)
    ax.set_title("Runtime")
    ax.set_ylabel("Time (seconds)")
    plt.setp(ax.get_xticklabels(), rotation=45)

    axes[1, 1].set_title("Variance Reduction (not plotted)")
    axes[1, 2].set_title("Table Staleness (not plotted)")
    plt.tight_layout()
    plt.suptitle(f"SAGA Analysis: {problem_name}")


def exercise_1_basic_saga():
    problem = QuadraticFiniteSum(n_samples=30, dim=5, condition_number=10.0, regularization=0.05)
    x0 = np.ones(problem.dim)
    saga = SAGAOptimizer(step_size=0.2)
    x_final, history = optimize_with_saga(problem, saga, x0, n_epochs=30)
    return x_final, history


def exercise_2_convergence_analysis():
    problem = QuadraticFiniteSum(n_samples=50, dim=5, condition_number=30.0, regularization=0.1)
    x0 = np.ones(problem.dim) * 2.0
    optimizers = {"SAGA": SAGAOptimizer(step_size=0.15), "CompressedSAGA": MemoryEfficientSAGA(step_size=0.15, compression_ratio=0.3)}
    return compare_saga_variants(problem, optimizers, x0, n_epochs=20)


def exercise_3_memory_efficiency():
    return memory_usage_scaling_study(dimensions=[10, 50, 100], sample_sizes=[100, 500, 1000])


def exercise_4_variance_reduction():
    problem = QuadraticFiniteSum(n_samples=30, dim=5, regularization=0.1)
    x0 = np.ones(problem.dim)
    saga = SAGAOptimizer(step_size=0.2)
    x_final, history = optimize_with_saga(problem, saga, x0, n_epochs=10)
    return x_final, history


def main():
    start = time.time()
    x_final, history = exercise_1_basic_saga()
    end = time.time()
    print(f"Completed demo in {end - start:.3f}s; final f={history['objective'][-1]:.6g}")


if __name__ == "__main__":
    main()

