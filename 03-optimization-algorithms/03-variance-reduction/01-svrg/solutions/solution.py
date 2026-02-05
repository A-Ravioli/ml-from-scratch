"""
SVRG reference solution.

The accompanying `exercise.py` is intentionally incomplete for learners. This file provides a
complete, deterministic implementation with the same public API, suitable for running under
`scripts/verify_solutions.py`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


class FiniteSumProblem(ABC):
    """Base class for finite-sum optimization problems."""

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
    Finite sum of quadratics:
        f(x) = (1/n) sum_i 0.5 (x-a_i)^T A_i (x-a_i)

    Each A_i is SPD, so the objective is strongly convex.
    """

    def __init__(
        self,
        n_samples: int,
        dim: int,
        condition_number: float = 10.0,
        noise_level: float = 0.1,
    ):
        super().__init__(n_samples, dim)
        self.A_matrices: List[np.ndarray] = []
        self.centers: np.ndarray = np.zeros((self.n_samples, self.dim))
        self.mu: float = 0.0
        self.L: float = 0.0
        self._generate_problem_data(float(condition_number), float(noise_level))

    def _generate_problem_data(self, condition_number: float, noise_level: float):
        rng = np.random.default_rng(0)
        self.centers = rng.normal(size=(self.n_samples, self.dim)) * (1.0 + noise_level)

        # Generate SPD matrices with a controlled spectrum via random orthogonal basis.
        eig_min = 1.0
        eig_max = max(eig_min * 1.01, float(condition_number))
        base_eigs = np.linspace(eig_min, eig_max, self.dim)

        self.A_matrices = []
        for _ in range(self.n_samples):
            Q, _ = np.linalg.qr(rng.normal(size=(self.dim, self.dim)))
            # Small per-sample spectrum perturbation for variety.
            jitter = 1.0 + 0.05 * rng.normal(size=self.dim)
            eigs = np.maximum(1e-3, base_eigs * jitter)
            A = Q @ np.diag(eigs) @ Q.T
            A = (A + A.T) / 2
            self.A_matrices.append(A)

        H = sum(self.A_matrices) / self.n_samples
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

    def optimal_point(self) -> np.ndarray:
        H = sum(self.A_matrices) / self.n_samples
        b = np.zeros(self.dim, dtype=float)
        for i in range(self.n_samples):
            b += self.A_matrices[i] @ self.centers[i]
        b /= self.n_samples
        return np.linalg.solve(H, b)


class LogisticRegressionFiniteSum(FiniteSumProblem):
    """
    Logistic regression:
        f(x) = (1/n) sum_i log(1 + exp(-y_i z_i^T x)) + (λ/2)||x||^2
    """

    def __init__(self, n_samples: int, dim: int, regularization: float = 0.01, data_noise: float = 0.1):
        super().__init__(n_samples, dim)
        self.regularization = float(regularization)
        self.features: np.ndarray = np.zeros((self.n_samples, self.dim))
        self.labels: np.ndarray = np.zeros(self.n_samples, dtype=int)
        self._opt_cache: Optional[np.ndarray] = None
        self._generate_classification_data(float(data_noise))

    def _generate_classification_data(self, noise_level: float):
        rng = np.random.default_rng(0)
        w_true = rng.normal(size=self.dim)
        w_true /= max(1e-8, np.linalg.norm(w_true))
        X = rng.normal(size=(self.n_samples, self.dim))
        logits = X @ w_true + noise_level * rng.normal(size=self.n_samples)
        y = np.where(logits >= 0.0, 1, -1)
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
        # d/dx log(1 + exp(-margin)) = -y z * sigmoid(-margin)
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

        # Deterministic gradient descent with a conservative step.
        x = np.zeros(self.dim, dtype=float)
        z_norm_sq = np.sum(self.features**2, axis=1)
        L = 0.25 * float(np.max(z_norm_sq)) + self.regularization
        step = 0.9 / max(1e-6, L)
        for _ in range(1000):
            g = self.full_gradient(x)
            x = x - step * g
        self._opt_cache = x.copy()
        return x


class SVRGOptimizer:
    """SVRG (Stochastic Variance Reduced Gradient) optimizer."""

    def __init__(self, step_size: float, inner_loop_length: int, snapshot_strategy: str = "fixed"):
        self.step_size = float(step_size)
        self.inner_loop_length = int(inner_loop_length)
        self.snapshot_strategy = str(snapshot_strategy)

        self.snapshot_point: Optional[np.ndarray] = None
        self.full_gradient_at_snapshot: Optional[np.ndarray] = None
        self.iteration_count = 0

        self.history: Dict[str, List] = {
            "objective": [],
            "gradient_norm": [],
            "distance_to_opt": [],
            "variance": [],
            "full_gradient_evaluations": 0,
        }

        self._rng = np.random.default_rng(0)

    def update_snapshot(self, x: np.ndarray, problem: FiniteSumProblem):
        self.snapshot_point = np.asarray(x, dtype=float).copy()
        self.full_gradient_at_snapshot = problem.full_gradient(self.snapshot_point)
        self.history["full_gradient_evaluations"] += 1

    def compute_svrg_gradient(self, x: np.ndarray, sample_idx: int, problem: FiniteSumProblem) -> np.ndarray:
        if self.snapshot_point is None or self.full_gradient_at_snapshot is None:
            self.update_snapshot(np.asarray(x, dtype=float), problem)
        idx = int(sample_idx)
        return (
            problem.individual_gradient(x, idx)
            - problem.individual_gradient(self.snapshot_point, idx)
            + self.full_gradient_at_snapshot
        )

    def step(self, x: np.ndarray, problem: FiniteSumProblem) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.snapshot_point is None or self.full_gradient_at_snapshot is None:
            self.update_snapshot(x, problem)
        # Fixed snapshot schedule: refresh every m steps.
        if self.snapshot_strategy == "fixed" and (self.iteration_count % self.inner_loop_length == 0):
            self.update_snapshot(x, problem)

        sample_idx = int(self._rng.integers(0, problem.n_samples))
        g = self.compute_svrg_gradient(x, sample_idx, problem)
        x_new = x - self.step_size * g
        self.iteration_count += 1
        return x_new

    def reset(self):
        self.snapshot_point = None
        self.full_gradient_at_snapshot = None
        self.iteration_count = 0
        self.history = {
            "objective": [],
            "gradient_norm": [],
            "distance_to_opt": [],
            "variance": [],
            "full_gradient_evaluations": 0,
        }


class SGDOptimizer:
    """Standard SGD (uniform sampling) for comparison."""

    def __init__(self, step_size: float):
        self.step_size = float(step_size)
        self.iteration_count = 0
        self.history: Dict[str, List] = {
            "objective": [],
            "gradient_norm": [],
            "distance_to_opt": [],
            "variance": [],
            "full_gradient_evaluations": 0,
        }
        self._rng = np.random.default_rng(0)

    def step(self, x: np.ndarray, problem: FiniteSumProblem) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        sample_idx = int(self._rng.integers(0, problem.n_samples))
        g = problem.individual_gradient(x, sample_idx)
        x_new = x - self.step_size * g
        self.iteration_count += 1
        return x_new

    def reset(self):
        self.iteration_count = 0
        self.history = {
            "objective": [],
            "gradient_norm": [],
            "distance_to_opt": [],
            "variance": [],
            "full_gradient_evaluations": 0,
        }


def estimate_gradient_variance(problem: FiniteSumProblem, x: np.ndarray, n_samples: int = 1000) -> float:
    """
    Estimate E[||g_i(x) - ∇f(x)||^2] by Monte Carlo sampling of indices.
    """
    rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    full = problem.full_gradient(x)
    vals = []
    for _ in range(int(n_samples)):
        i = int(rng.integers(0, problem.n_samples))
        gi = problem.individual_gradient(x, i)
        vals.append(float(np.sum((gi - full) ** 2)))
    return float(np.mean(vals))


def optimize_with_svrg(
    problem: FiniteSumProblem,
    optimizer: SVRGOptimizer,
    x0: np.ndarray,
    n_epochs: int = 100,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    x = np.asarray(x0, dtype=float).copy()
    optimal_point = problem.optimal_point()
    optimizer.reset()

    total_inner = optimizer.inner_loop_length
    for epoch in range(int(n_epochs)):
        optimizer.update_snapshot(x, problem)
        for _ in range(total_inner):
            x = optimizer.step(x, problem)

        f = problem.objective(x)
        g = problem.full_gradient(x)
        d = float(np.linalg.norm(x - optimal_point))
        v = estimate_gradient_variance(problem, x, n_samples=min(200, 20 * problem.n_samples))
        optimizer.history["objective"].append(f)
        optimizer.history["gradient_norm"].append(float(np.linalg.norm(g)))
        optimizer.history["distance_to_opt"].append(d)
        optimizer.history["variance"].append(v)
        if verbose:
            print(f"[SVRG] epoch={epoch:03d} f={f:.6g} ||g||={np.linalg.norm(g):.3g} dist={d:.3g} var={v:.3g}")

    return x, optimizer.history


def optimize_with_sgd(
    problem: FiniteSumProblem,
    optimizer: SGDOptimizer,
    x0: np.ndarray,
    n_iterations: int = 10000,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    x = np.asarray(x0, dtype=float).copy()
    optimal_point = problem.optimal_point()
    optimizer.reset()

    n_iterations = int(n_iterations)
    track_every = max(1, n_iterations // 100)
    for t in range(n_iterations):
        x = optimizer.step(x, problem)
        if (t + 1) % track_every == 0 or t == 0:
            f = problem.objective(x)
            g = problem.full_gradient(x)
            d = float(np.linalg.norm(x - optimal_point))
            v = estimate_gradient_variance(problem, x, n_samples=min(200, 20 * problem.n_samples))
            optimizer.history["objective"].append(f)
            optimizer.history["gradient_norm"].append(float(np.linalg.norm(g)))
            optimizer.history["distance_to_opt"].append(d)
            optimizer.history["variance"].append(v)
            if verbose:
                print(
                    f"[SGD] iter={t:05d} f={f:.6g} ||g||={np.linalg.norm(g):.3g} dist={d:.3g} var={v:.3g}"
                )

    return x, optimizer.history


def compare_svrg_vs_sgd(
    problem: FiniteSumProblem,
    svrg_params: Dict,
    sgd_params: Dict,
    x0: np.ndarray,
    n_iterations: int = 10000,
) -> Dict:
    results: Dict[str, Tuple[np.ndarray, Dict]] = {}

    svrg = SVRGOptimizer(**svrg_params)
    n_epochs = max(1, int(n_iterations) // max(1, svrg.inner_loop_length))
    x_svrg, history_svrg = optimize_with_svrg(problem, svrg, x0, n_epochs=n_epochs)
    results["SVRG"] = (x_svrg, history_svrg)

    sgd = SGDOptimizer(**sgd_params)
    x_sgd, history_sgd = optimize_with_sgd(problem, sgd, x0, n_iterations=int(n_iterations))
    results["SGD"] = (x_sgd, history_sgd)

    return results


def hyperparameter_sensitivity_study(
    problem: FiniteSumProblem, step_sizes: List[float], inner_loop_lengths: List[int], x0: np.ndarray
) -> Dict:
    results: Dict[str, Dict] = {}
    for eta in step_sizes:
        for m in inner_loop_lengths:
            svrg = SVRGOptimizer(step_size=float(eta), inner_loop_length=int(m))
            x_final, history = optimize_with_svrg(problem, svrg, x0, n_epochs=10)
            key = f"eta={eta}_m={m}"
            results[key] = {
                "final_point": x_final,
                "final_objective": problem.objective(x_final),
                "history": history,
            }
    return results


def plot_convergence_comparison(results: Dict, problem: FiniteSumProblem):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.reshape(-1)

    for name, (x_final, history) in results.items():
        axes[0].plot(history.get("objective", []), label=name)
        axes[1].plot(history.get("distance_to_opt", []), label=name)
        axes[2].semilogy(history.get("gradient_norm", []), label=name)
        axes[3].semilogy(history.get("variance", []), label=name)

    axes[0].set_title("Objective")
    axes[1].set_title("Distance to optimum")
    axes[2].set_title("Gradient norm")
    axes[3].set_title("Gradient variance")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    plt.tight_layout()


def visualize_variance_reduction(problem: FiniteSumProblem, svrg_optimizer: SVRGOptimizer, x_trajectory: List[np.ndarray]):
    variances_svrg: List[float] = []
    variances_sgd: List[float] = []
    distances_to_opt: List[float] = []
    opt = problem.optimal_point()

    sgd = SGDOptimizer(step_size=svrg_optimizer.step_size)

    for x in x_trajectory:
        variances_svrg.append(estimate_gradient_variance(problem, x, n_samples=min(200, 20 * problem.n_samples)))
        variances_sgd.append(estimate_gradient_variance(problem, x, n_samples=min(200, 20 * problem.n_samples)))
        distances_to_opt.append(float(np.linalg.norm(np.asarray(x) - opt)))

    plt.figure(figsize=(10, 6))
    plt.plot(distances_to_opt, variances_svrg, label="SVRG variance (est.)")
    plt.plot(distances_to_opt, variances_sgd, label="SGD variance (est.)", linestyle="--")
    plt.xlabel("Distance to optimum")
    plt.ylabel("Gradient variance estimate")
    plt.title("Variance vs distance-to-optimum")
    plt.grid(True)
    plt.legend()


def exercise_1_basic_svrg():
    problem = QuadraticFiniteSum(n_samples=20, dim=5, condition_number=10.0, noise_level=0.2)
    x0 = np.zeros(problem.dim)
    svrg = SVRGOptimizer(step_size=0.2, inner_loop_length=10)
    x_final, history = optimize_with_svrg(problem, svrg, x0, n_epochs=20, verbose=True)
    return x_final, history


def exercise_2_convergence_analysis():
    problem = QuadraticFiniteSum(n_samples=20, dim=5, condition_number=30.0, noise_level=0.2)
    x0 = np.ones(problem.dim)
    svrg = SVRGOptimizer(step_size=0.15, inner_loop_length=10)
    sgd = SGDOptimizer(step_size=0.03)
    results = compare_svrg_vs_sgd(problem, {"step_size": svrg.step_size, "inner_loop_length": svrg.inner_loop_length}, {"step_size": sgd.step_size}, x0, n_iterations=200)
    return results


def exercise_3_hyperparameter_tuning():
    problem = QuadraticFiniteSum(n_samples=20, dim=5)
    x0 = np.ones(problem.dim)
    return hyperparameter_sensitivity_study(problem, step_sizes=[0.05, 0.1, 0.2], inner_loop_lengths=[5, 10, 20], x0=x0)


def exercise_4_variance_reduction_study():
    problem = QuadraticFiniteSum(n_samples=50, dim=5)
    x0 = np.ones(problem.dim)
    svrg = SVRGOptimizer(step_size=0.2, inner_loop_length=10)
    x, history = optimize_with_svrg(problem, svrg, x0, n_epochs=10)
    return x, history


def exercise_5_practical_problems():
    problem = LogisticRegressionFiniteSum(n_samples=200, dim=10, regularization=0.05, data_noise=0.3)
    x0 = np.zeros(problem.dim)
    svrg = SVRGOptimizer(step_size=0.5, inner_loop_length=20)
    x, history = optimize_with_svrg(problem, svrg, x0, n_epochs=5)
    return x, history


def exercise_6_extensions():
    # Placeholder for extensions; provide a deterministic baseline return.
    return {"status": "extensions-not-implemented-in-toy-solution"}


def main():
    start = time.time()
    x_final, history = exercise_1_basic_svrg()
    end = time.time()
    print(f"Completed demo in {end - start:.3f}s; final f={history['objective'][-1]:.6g}")


if __name__ == "__main__":
    main()

