"""
SGD Variants Solutions - Reference Implementation

This mirrors the interfaces in `exercise.py` and provides working
implementations for study and verification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import time


class OptimizationProblem(ABC):
    """Base class for optimization problems."""

    def __init__(self, dim: int, noise_std: float = 0.1):
        self.dim = dim
        self.noise_std = noise_std

    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        ...

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def stochastic_gradient(self, x: np.ndarray, batch_indices: Optional[np.ndarray] = None) -> np.ndarray:
        ...

    @abstractmethod
    def optimal_point(self) -> np.ndarray:
        ...


class QuadraticProblem(OptimizationProblem):
    """
    Quadratic: f(x) = 0.5 x^T A x + b^T x
    with A positive definite.
    """

    def __init__(self, dim: int, condition_number: float = 10.0, noise_std: float = 0.1):
        super().__init__(dim, noise_std)
        self.A: np.ndarray
        self.b: np.ndarray
        self._rng = np.random.default_rng(0)
        self._generate_A_b(condition_number)

    def _generate_A_b(self, condition_number: float) -> None:
        # Random orthogonal Q
        M = self._rng.standard_normal((self.dim, self.dim))
        Q, _ = np.linalg.qr(M)
        eigs = np.linspace(1.0, float(condition_number), self.dim)
        self.A = Q @ np.diag(eigs) @ Q.T
        self.A = 0.5 * (self.A + self.A.T)
        self.b = self._rng.standard_normal(self.dim)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return 0.5 * float(x.T @ self.A @ x) + float(self.b @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.A @ x + self.b

    def stochastic_gradient(self, x: np.ndarray, batch_indices: Optional[np.ndarray] = None) -> np.ndarray:
        g = self.gradient(x)
        if self.noise_std <= 0:
            return g
        noise = self._rng.normal(scale=self.noise_std, size=g.shape)
        return g + noise

    def optimal_point(self) -> np.ndarray:
        return -np.linalg.solve(self.A, self.b)


class LogisticRegressionProblem(OptimizationProblem):
    """
    Logistic regression on a synthetic dataset.
    """

    def __init__(self, n_samples: int, dim: int, regularization: float = 0.01):
        super().__init__(dim, noise_std=0.0)
        self.n_samples = n_samples
        self.regularization = float(regularization)
        self._rng = np.random.default_rng(0)
        self.Z: np.ndarray
        self.y: np.ndarray
        self._opt_cache: Optional[np.ndarray] = None
        self._generate_data()

    def _generate_data(self) -> None:
        Z = self._rng.standard_normal((self.n_samples, self.dim))
        w_true = self._rng.standard_normal(self.dim)
        logits = Z @ w_true + 0.5 * self._rng.standard_normal(self.n_samples)
        y = np.where(logits >= 0, 1.0, -1.0)
        # Add small label noise so it's not perfectly separable.
        flip = self._rng.random(self.n_samples) < 0.05
        y[flip] *= -1.0
        self.Z = Z
        self.y = y

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        margins = self.y * (self.Z @ x)
        loss = np.logaddexp(0.0, -margins).mean()
        reg = 0.5 * self.regularization * float(x @ x)
        return float(loss + reg)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        margins = self.y * (self.Z @ x)
        # d/dx log(1+exp(-m)) = -y z * sigmoid(-m)
        sigma = 1.0 / (1.0 + np.exp(margins))
        grad = -(self.Z.T @ (self.y * sigma)) / self.n_samples
        grad = grad + self.regularization * x
        return grad

    def stochastic_gradient(self, x: np.ndarray, batch_indices: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if batch_indices is None:
            batch_indices = self._rng.integers(0, self.n_samples, size=min(32, self.n_samples))
        Zb = self.Z[batch_indices]
        yb = self.y[batch_indices]
        margins = yb * (Zb @ x)
        sigma = 1.0 / (1.0 + np.exp(margins))
        grad = -(Zb.T @ (yb * sigma)) / len(batch_indices)
        grad = grad + self.regularization * x
        return grad

    def optimal_point(self) -> np.ndarray:
        if self._opt_cache is not None:
            return self._opt_cache.copy()
        # Simple deterministic GD to approximate optimum (avoid heavy dependencies).
        x = np.zeros(self.dim, dtype=float)
        lr = 0.2
        for _ in range(2000):
            g = self.gradient(x)
            x = x - lr * g
            if np.linalg.norm(g) < 1e-6:
                break
        self._opt_cache = x.copy()
        return x


class SGDOptimizer(ABC):
    """Base class for SGD optimizers."""

    def __init__(self, learning_rate: float, **kwargs):
        self.learning_rate = float(learning_rate)
        self.history = {"objective": [], "gradient_norm": [], "distance_to_opt": []}

    @abstractmethod
    def step(self, gradient: np.ndarray) -> np.ndarray:
        """Return an additive parameter update Δx."""

    @abstractmethod
    def reset(self):
        ...


class VanillaSGD(SGDOptimizer):
    def step(self, gradient: np.ndarray) -> np.ndarray:
        return -self.learning_rate * gradient

    def reset(self):
        return None


class SGDWithMomentum(SGDOptimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = float(momentum)
        self.velocity: Optional[np.ndarray] = None

    def step(self, gradient: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        return -self.velocity

    def reset(self):
        self.velocity = None


class NesterovSGD(SGDOptimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = float(momentum)
        self.velocity: Optional[np.ndarray] = None

    def get_lookahead_point(self, x: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            return x
        return x - self.momentum * self.velocity

    def step(self, gradient: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        return -self.velocity

    def reset(self):
        self.velocity = None


class AdaGrad(SGDOptimizer):
    def __init__(self, learning_rate: float = 0.01, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.eps = float(eps)
        self.sum_squared_gradients: Optional[np.ndarray] = None

    def step(self, gradient: np.ndarray) -> np.ndarray:
        if self.sum_squared_gradients is None:
            self.sum_squared_gradients = np.zeros_like(gradient)
        self.sum_squared_gradients = self.sum_squared_gradients + gradient**2
        denom = np.sqrt(self.sum_squared_gradients + self.eps)
        return -(self.learning_rate / denom) * gradient

    def reset(self):
        self.sum_squared_gradients = None


class RMSprop(SGDOptimizer):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.decay_rate = float(decay_rate)
        self.eps = float(eps)
        self.moving_avg_squared_grad: Optional[np.ndarray] = None

    def step(self, gradient: np.ndarray) -> np.ndarray:
        if self.moving_avg_squared_grad is None:
            self.moving_avg_squared_grad = np.zeros_like(gradient)
        self.moving_avg_squared_grad = self.decay_rate * self.moving_avg_squared_grad + (1.0 - self.decay_rate) * (
            gradient**2
        )
        denom = np.sqrt(self.moving_avg_squared_grad) + self.eps
        return -(self.learning_rate / denom) * gradient

    def reset(self):
        self.moving_avg_squared_grad = None


class Adam(SGDOptimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def step(self, gradient: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient**2)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0


class SVRG(SGDOptimizer):
    def __init__(self, learning_rate: float, update_frequency: int = 100):
        super().__init__(learning_rate)
        self.update_frequency = int(update_frequency)
        self.snapshot_point: Optional[np.ndarray] = None
        self.full_gradient: Optional[np.ndarray] = None
        self.step_count = 0

    def update_snapshot(self, x: np.ndarray, problem: OptimizationProblem):
        self.snapshot_point = np.asarray(x, dtype=float).copy()
        self.full_gradient = problem.gradient(self.snapshot_point)
        self.step_count = 0

    def step(self, gradient: np.ndarray, x: np.ndarray, sample_gradient_at_snapshot: np.ndarray) -> np.ndarray:
        if self.full_gradient is None:
            raise ValueError("SVRG snapshot not initialized")
        v = gradient - sample_gradient_at_snapshot + self.full_gradient
        return -self.learning_rate * v

    def reset(self):
        self.snapshot_point = None
        self.full_gradient = None
        self.step_count = 0


def optimize_problem(
    problem: OptimizationProblem,
    optimizer: SGDOptimizer,
    x0: np.ndarray,
    n_iterations: int = 1000,
    batch_size: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, List]]:
    x = np.asarray(x0, dtype=float).copy()
    history: Dict[str, List] = {"objective": [], "gradient_norm": [], "distance_to_opt": [], "time": []}

    try:
        optimal_point = problem.optimal_point()
    except Exception:
        optimal_point = None

    start_time = time.time()
    rng = np.random.default_rng(0)

    for it in range(n_iterations):
        if isinstance(problem, LogisticRegressionProblem) and batch_size > 1:
            batch_idx = rng.integers(0, problem.n_samples, size=batch_size)
        else:
            batch_idx = None

        if isinstance(optimizer, SVRG):
            if optimizer.snapshot_point is None or (it % optimizer.update_frequency == 0):
                optimizer.update_snapshot(x, problem)
            assert optimizer.snapshot_point is not None
            assert optimizer.full_gradient is not None
            g = problem.stochastic_gradient(x, batch_idx)
            g_snap = problem.stochastic_gradient(optimizer.snapshot_point, batch_idx)
            dx = optimizer.step(g, x, g_snap)
        elif isinstance(optimizer, NesterovSGD):
            lookahead = optimizer.get_lookahead_point(x)
            g = problem.stochastic_gradient(lookahead, batch_idx)
            dx = optimizer.step(g)
        else:
            g = problem.stochastic_gradient(x, batch_idx)
            dx = optimizer.step(g)

        x = x + dx

        fval = problem.objective(x)
        g_full = problem.gradient(x)
        history["objective"].append(float(fval))
        history["gradient_norm"].append(float(np.linalg.norm(g_full)))
        if optimal_point is None:
            history["distance_to_opt"].append(float("nan"))
        else:
            history["distance_to_opt"].append(float(np.linalg.norm(x - optimal_point)))
        history["time"].append(float(time.time() - start_time))

    return x, history


def compare_optimizers(
    problem: OptimizationProblem,
    optimizers: Dict[str, SGDOptimizer],
    x0: np.ndarray,
    n_iterations: int = 1000,
    batch_size: int = 1,
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    results: Dict[str, Tuple[np.ndarray, Dict]] = {}
    for name, opt in optimizers.items():
        opt.reset()
        results[name] = optimize_problem(problem, opt, x0, n_iterations=n_iterations, batch_size=batch_size)
    return results


def plot_convergence(results: Dict[str, Tuple[np.ndarray, Dict]], problem: OptimizationProblem, metrics: List[str] = ["objective"]):
    for metric in metrics:
        plt.figure(figsize=(6, 4))
        for name, (_, hist) in results.items():
            plt.plot(hist[metric], label=name)
        plt.title(metric)
        plt.legend()
        plt.close()


def learning_rate_sensitivity_study(
    problem: OptimizationProblem, optimizer_class: type, learning_rates: List[float], x0: np.ndarray, n_iterations: int = 1000
) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for lr in learning_rates:
        opt = optimizer_class(lr)
        x_final, hist = optimize_problem(problem, opt, x0, n_iterations=n_iterations)
        out[float(lr)] = float(hist["objective"][-1])
    return out


def batch_size_study(
    problem: OptimizationProblem,
    optimizer: SGDOptimizer,
    batch_sizes: List[int],
    x0: np.ndarray,
    n_iterations: int = 1000,
) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for bs in batch_sizes:
        optimizer.reset()
        _, hist = optimize_problem(problem, optimizer, x0, n_iterations=n_iterations, batch_size=bs)
        out[int(bs)] = hist
    return out
