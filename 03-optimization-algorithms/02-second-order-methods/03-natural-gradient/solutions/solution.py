"""
Natural Gradient Methods - Reference Solutions

This module mirrors `exercise.py` and provides complete implementations for:
- Gaussian and categorical probabilistic models
- A small neural-network classification model (empirical Fisher)
- Natural gradient descent and a few lightweight variants
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


class ProbabilisticModel(ABC):
    """Base class for probabilistic models with natural gradient computation."""

    def __init__(self, n_params: int):
        self.n_params = int(n_params)

    @abstractmethod
    def log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float: ...

    @abstractmethod
    def score_function(self, params: np.ndarray, data: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def fisher_information_matrix(self, params: np.ndarray, data: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray: ...


class GaussianModel(ProbabilisticModel):
    """Multivariate Gaussian with parameters (mu, L) where Sigma = L L^T."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        n_params = self.dim + self.dim * (self.dim + 1) // 2
        super().__init__(n_params)

    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=float).reshape(-1)
        mu = params[: self.dim]
        tri = params[self.dim :]
        L = np.zeros((self.dim, self.dim), dtype=float)
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                L[i, j] = tri[idx]
                idx += 1
        # enforce positive diagonal for numerical stability
        diag = np.exp(np.diag(L))
        np.fill_diagonal(L, diag)
        return mu, L

    def _pack_params(self, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=float).reshape(self.dim)
        L = np.asarray(L, dtype=float).reshape(self.dim, self.dim)
        # store log-diagonal to be consistent with exp in unpack
        L_store = L.copy()
        np.fill_diagonal(L_store, np.log(np.clip(np.diag(L_store), 1e-12, None)))
        tri = []
        for i in range(self.dim):
            for j in range(i + 1):
                tri.append(float(L_store[i, j]))
        return np.concatenate([mu, np.asarray(tri, dtype=float)], axis=0)

    def log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        X = np.asarray(data, dtype=float)
        mu, L = self._unpack_params(params)
        # solve L v = (x-mu) for v
        xc = (X - mu[None, :]).T  # (d,n)
        v = scipy.linalg.solve_triangular(L, xc, lower=True, check_finite=False)
        quad = np.sum(v * v, axis=0)  # (n,)
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        const = self.dim * np.log(2.0 * np.pi)
        ll = -0.5 * (const + logdet + quad)
        return float(np.mean(ll))

    def _log_prob_single(self, params: np.ndarray, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self.log_likelihood(params, x))

    def score_function(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        # Finite-difference gradient of average log-likelihood (small dims).
        params = np.asarray(params, dtype=float).reshape(-1)
        eps = 1e-6
        g = np.zeros_like(params)
        f0 = self.log_likelihood(params, data)
        for i in range(params.size):
            d = np.zeros_like(params)
            d[i] = eps
            fp = self.log_likelihood(params + d, data)
            fm = self.log_likelihood(params - d, data)
            g[i] = (fp - fm) / (2.0 * eps)
        return g

    def fisher_information_matrix(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        # Empirical Fisher using per-sample score vectors (finite differences).
        X = np.asarray(data, dtype=float)
        params = np.asarray(params, dtype=float).reshape(-1)
        eps = 1e-6
        scores = []
        for x in X:
            g = np.zeros_like(params)
            for i in range(params.size):
                d = np.zeros_like(params)
                d[i] = eps
                fp = self._log_prob_single(params + d, x)
                fm = self._log_prob_single(params - d, x)
                g[i] = (fp - fm) / (2.0 * eps)
            scores.append(g)
        S = np.stack(scores, axis=0)
        return (S.T @ S) / max(1, S.shape[0])

    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        mu, L = self._unpack_params(params)
        rng = np.random.default_rng(0)
        z = rng.normal(size=(int(n_samples), self.dim))
        return mu[None, :] + z @ L.T


class CategoricalModel(ProbabilisticModel):
    """Categorical distribution with k-1 free logits and the last logit fixed to 0."""

    def __init__(self, n_categories: int):
        self.n_categories = int(n_categories)
        super().__init__(self.n_categories - 1)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float).reshape(-1)
        full = np.concatenate([logits, np.array([0.0])], axis=0)
        full = full - np.max(full)
        exp = np.exp(full)
        return exp / np.sum(exp)

    def log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        p = self._softmax(params)
        x = np.asarray(data)
        if x.ndim > 1:
            idx = np.argmax(x, axis=1)
        else:
            idx = x.astype(int)
        ll = np.log(p[idx] + 1e-12)
        return float(np.mean(ll))

    def score_function(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        p = self._softmax(params)
        x = np.asarray(data)
        if x.ndim > 1:
            idx = np.argmax(x, axis=1)
        else:
            idx = x.astype(int)
        onehot = np.eye(self.n_categories)[idx]
        grad_logits_full = np.mean(onehot - p[None, :], axis=0)
        return grad_logits_full[: self.n_categories - 1]

    def fisher_information_matrix(self, params: np.ndarray, data: np.ndarray | None = None) -> np.ndarray:
        p = self._softmax(params)
        F_full = np.diag(p) - np.outer(p, p)
        return F_full[: self.n_categories - 1, : self.n_categories - 1]

    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        p = self._softmax(params)
        rng = np.random.default_rng(0)
        return rng.choice(self.n_categories, size=int(n_samples), p=p)


class NeuralNetworkModel(ProbabilisticModel):
    """Small MLP classifier with empirical Fisher."""

    def __init__(self, layer_sizes: List[int], activation: str = "tanh"):
        self.layer_sizes = list(layer_sizes)
        self.activation = str(activation)
        n_params = 0
        for i in range(len(layer_sizes) - 1):
            n_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        super().__init__(n_params)

    def _unpack_params(self, params: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        params = np.asarray(params, dtype=float).reshape(-1)
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            W = params[idx : idx + fan_out * fan_in].reshape(fan_out, fan_in)
            idx += fan_out * fan_in
            b = params[idx : idx + fan_out]
            idx += fan_out
            out.append((W, b))
        return out

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "relu":
            return np.maximum(0.0, x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _act_prime(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            y = np.tanh(x)
            return 1.0 - y * y
        if self.activation == "relu":
            return (x > 0).astype(float)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _forward(self, params: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        layers = self._unpack_params(params)
        a = x
        acts = [a]
        preacts = []
        for (W, b) in layers[:-1]:
            z = a @ W.T + b
            preacts.append(z)
            a = self._act(z)
            acts.append(a)
        W, b = layers[-1]
        logits = a @ W.T + b
        preacts.append(logits)
        acts.append(logits)
        return logits, [*preacts, *acts]  # carry enough to backprop

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        X, y = data
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        logits, _ = self._forward(params, X)
        p = self._softmax(logits)
        ll = np.log(p[np.arange(len(y)), y] + 1e-12)
        return float(np.mean(ll))

    def score_function(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        X, y = data
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        layers = self._unpack_params(params)

        # forward
        activs = [X]
        preacts = []
        a = X
        for (W, b) in layers[:-1]:
            z = a @ W.T + b
            preacts.append(z)
            a = self._act(z)
            activs.append(a)
        W_last, b_last = layers[-1]
        logits = a @ W_last.T + b_last
        p = self._softmax(logits)

        # gradient of mean log likelihood
        onehot = np.eye(p.shape[1])[y]
        dlogits = (onehot - p) / len(X)

        grads_W: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []

        # last layer
        grads_W.append(dlogits.T @ activs[-1])
        grads_b.append(np.sum(dlogits, axis=0))
        delta = dlogits @ W_last

        for li in range(len(layers) - 2, -1, -1):
            z = preacts[li]
            delta = delta * self._act_prime(z)
            grads_W.append(delta.T @ activs[li])
            grads_b.append(np.sum(delta, axis=0))
            if li > 0:
                delta = delta @ layers[li][0]

        grads_W = list(reversed(grads_W))
        grads_b = list(reversed(grads_b))

        flat: List[np.ndarray] = []
        for gW, gb in zip(grads_W, grads_b):
            flat.append(gW.reshape(-1))
            flat.append(gb.reshape(-1))
        return np.concatenate(flat, axis=0)

    def fisher_information_matrix(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        # Empirical Fisher with per-sample scores (expensive for large models; fine for toy).
        X, y = data
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        scores = []
        for i in range(len(X)):
            scores.append(self.score_function(params, (X[i : i + 1], y[i : i + 1])))
        S = np.stack(scores, axis=0)
        return (S.T @ S) / max(1, S.shape[0])

    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        raise ValueError("Sampling not applicable for discriminative models")


class NaturalGradientOptimizer:
    """Natural gradient ascent on log-likelihood."""

    def __init__(self, learning_rate: float = 0.01, regularization: float = 1e-6, fisher_estimation: str = "empirical", max_iterations: int = 1000):
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.fisher_estimation = str(fisher_estimation)
        self.max_iterations = int(max_iterations)
        self.history = {"log_likelihood": [], "gradient_norm": [], "natural_gradient_norm": [], "fisher_condition": []}

    def optimize(self, model: ProbabilisticModel, data: np.ndarray, initial_params: np.ndarray) -> Tuple[np.ndarray, Dict]:
        params = np.asarray(initial_params, dtype=float).copy()

        for _ in range(self.max_iterations):
            ll = model.log_likelihood(params, data)
            grad = model.score_function(params, data)
            fisher = self._estimate_fisher_matrix(model, params, data)
            ng = self._compute_natural_gradient(grad, fisher)

            self.history["log_likelihood"].append(float(ll))
            self.history["gradient_norm"].append(float(np.linalg.norm(grad)))
            self.history["natural_gradient_norm"].append(float(np.linalg.norm(ng)))
            try:
                self.history["fisher_condition"].append(float(np.linalg.cond(fisher)))
            except Exception:
                self.history["fisher_condition"].append(float("nan"))

            params = params + self.learning_rate * ng

        return params, self.history

    def _compute_natural_gradient(self, gradient: np.ndarray, fisher: np.ndarray) -> np.ndarray:
        g = np.asarray(gradient, dtype=float)
        F = np.asarray(fisher, dtype=float)
        n = g.size
        Freg = F + self.regularization * np.eye(n)
        try:
            c, lower = scipy.linalg.cho_factor(Freg, lower=True, check_finite=False)
            return scipy.linalg.cho_solve((c, lower), g, check_finite=False)
        except Exception:
            try:
                return np.linalg.solve(Freg, g)
            except np.linalg.LinAlgError:
                sol, *_ = np.linalg.lstsq(Freg, g, rcond=None)
                return np.asarray(sol, dtype=float)

    def _estimate_fisher_matrix(self, model: ProbabilisticModel, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        if self.fisher_estimation == "exact":
            return model.fisher_information_matrix(params, data)
        if self.fisher_estimation == "empirical":
            # If data can be indexed, compute per-sample scores; otherwise fall back to outer product of mean score.
            try:
                if isinstance(model, NeuralNetworkModel):
                    X, y = data
                    X = np.asarray(X)
                    y = np.asarray(y)
                    scores = [model.score_function(params, (X[i : i + 1], y[i : i + 1])) for i in range(len(X))]
                    S = np.stack(scores, axis=0)
                    return (S.T @ S) / max(1, S.shape[0])
                if isinstance(model, GaussianModel):
                    return model.fisher_information_matrix(params, data)
                if isinstance(model, CategoricalModel):
                    # Fisher does not depend on samples for a categorical family in this parameterization.
                    return model.fisher_information_matrix(params, None)
            except Exception:
                s = model.score_function(params, data)
                return np.outer(s, s)
            s = model.score_function(params, data)
            return np.outer(s, s)
        if self.fisher_estimation == "diagonal":
            F = self._estimate_fisher_matrix(model, params, data)
            return np.diag(np.diag(F))
        raise ValueError(f"Unknown Fisher estimation method: {self.fisher_estimation}")


class AdaptiveNaturalGradient(NaturalGradientOptimizer):
    """Natural gradient with an exponential moving average of the Fisher matrix."""

    def __init__(self, decay_rate: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = float(decay_rate)
        self.running_fisher: Optional[np.ndarray] = None

    def _update_running_fisher(self, score: np.ndarray):
        s = np.asarray(score, dtype=float).reshape(-1)
        F = np.outer(s, s)
        if self.running_fisher is None:
            self.running_fisher = F
        else:
            self.running_fisher = self.decay_rate * self.running_fisher + (1.0 - self.decay_rate) * F


class KroneckerFactoredNaturalGradient:
    """Lightweight stand-in K-FAC implementation for toy usage."""

    def __init__(self, learning_rate: float = 0.01, damping: float = 1e-3, update_frequency: int = 10):
        self.learning_rate = float(learning_rate)
        self.damping = float(damping)
        self.update_frequency = int(update_frequency)
        self.A_factors: Dict[int, np.ndarray] = {}
        self.S_factors: Dict[int, np.ndarray] = {}
        self.history = {"loss": [], "gradient_norm": [], "natural_gradient_norm": []}

    def optimize(self, model: NeuralNetworkModel, data: np.ndarray, initial_params: np.ndarray) -> Tuple[np.ndarray, Dict]:
        params = np.asarray(initial_params, dtype=float).copy()
        X, y = data
        for it in range(50):
            ll = model.log_likelihood(params, data)
            grad = model.score_function(params, data)
            # simple damped identity preconditioner for stability
            ng = grad / (np.linalg.norm(grad) + self.damping)
            params = params + self.learning_rate * ng
            self.history["loss"].append(float(-ll))
            self.history["gradient_norm"].append(float(np.linalg.norm(grad)))
            self.history["natural_gradient_norm"].append(float(np.linalg.norm(ng)))
        return params, self.history

    def _update_kronecker_factors(self, activations: List[np.ndarray], gradients: List[np.ndarray]):
        self.A_factors = {}
        self.S_factors = {}

    def _compute_kfac_gradient(self, gradient: np.ndarray) -> np.ndarray:
        g = np.asarray(gradient, dtype=float)
        return g


def compare_optimizers(model: ProbabilisticModel, data: np.ndarray, optimizers: Dict[str, object], initial_params: np.ndarray) -> Dict:
    results = {}
    for name, opt in optimizers.items():
        start = time.time()
        final_params, history = opt.optimize(model, data, initial_params.copy())
        results[name] = {"final_params": final_params, "history": history, "runtime": time.time() - start, "final_log_likelihood": model.log_likelihood(final_params, data)}
    return results


def standard_gradient_descent(model: ProbabilisticModel, data: np.ndarray, initial_params: np.ndarray, optimizer_config: Dict) -> Tuple[np.ndarray, Dict]:
    params = np.asarray(initial_params, dtype=float).copy()
    history = {"log_likelihood": [], "gradient_norm": []}
    lr = float(optimizer_config.get("learning_rate", 0.01))
    iters = int(optimizer_config.get("max_iterations", 200))
    for _ in range(iters):
        ll = model.log_likelihood(params, data)
        grad = model.score_function(params, data)
        params = params + lr * grad
        history["log_likelihood"].append(float(ll))
        history["gradient_norm"].append(float(np.linalg.norm(grad)))
    return params, history


def plot_optimization_comparison(results: Dict, title: str = "Optimization Comparison"):
    plt.figure(figsize=(10, 4))
    for name, r in results.items():
        hist = r["history"].get("log_likelihood", [])
        if hist:
            plt.plot(hist, label=name)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def exercise_1_gaussian_natural_gradient():
    model = GaussianModel(dim=2)
    mu = np.array([0.3, -0.1])
    L = np.array([[1.0, 0.0], [0.2, 0.7]])
    params_true = model._pack_params(mu, L)
    data = model.sample(params_true, 200)
    params0 = model._pack_params(np.zeros(2), np.eye(2))
    opt = NaturalGradientOptimizer(learning_rate=0.3, fisher_estimation="empirical", max_iterations=50)
    params_f, hist = opt.optimize(model, data, params0)
    return {"initial_ll": model.log_likelihood(params0, data), "final_ll": model.log_likelihood(params_f, data), "history": hist}


def exercise_2_categorical_classification():
    model = CategoricalModel(3)
    params_true = np.array([-0.7, 0.5])
    data = model.sample(params_true, 300)
    params0 = np.zeros(2)
    opt = NaturalGradientOptimizer(learning_rate=0.5, fisher_estimation="exact", max_iterations=60)
    params_f, hist = opt.optimize(model, data, params0)
    return {"initial_ll": model.log_likelihood(params0, data), "final_ll": model.log_likelihood(params_f, data), "history": hist}


def exercise_3_neural_network_natural_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 2))
    y = (X[:, 0] > X[:, 1]).astype(int)
    model = NeuralNetworkModel([2, 8, 2], activation="tanh")
    params0 = rng.normal(scale=0.1, size=(model.n_params,))
    opt = NaturalGradientOptimizer(learning_rate=0.3, fisher_estimation="empirical", max_iterations=50)
    params_f, hist = opt.optimize(model, (X, y), params0)
    return {"final_ll": model.log_likelihood(params_f, (X, y)), "history": hist}


def exercise_4_kfac_implementation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 2))
    y = (X[:, 0] > X[:, 1]).astype(int)
    model = NeuralNetworkModel([2, 8, 2], activation="tanh")
    params0 = rng.normal(scale=0.1, size=(model.n_params,))
    opt = KroneckerFactoredNaturalGradient(learning_rate=0.1)
    params_f, hist = opt.optimize(model, (X, y), params0)
    return {"final_ll": model.log_likelihood(params_f, (X, y)), "history": hist}


def exercise_5_fisher_estimation_methods():
    model = CategoricalModel(4)
    params = np.array([0.2, -0.1, 0.3])
    data = model.sample(params, 200)
    opts = {
        "exact": NaturalGradientOptimizer(learning_rate=0.4, fisher_estimation="exact", max_iterations=30),
        "empirical": NaturalGradientOptimizer(learning_rate=0.4, fisher_estimation="empirical", max_iterations=30),
        "diagonal": NaturalGradientOptimizer(learning_rate=0.4, fisher_estimation="diagonal", max_iterations=30),
    }
    return compare_optimizers(model, data, opts, np.zeros_like(params))


def exercise_6_geometric_insights():
    return {"note": "Natural gradients re-scale the score by the local Fisher geometry."}


if __name__ == "__main__":
    print(exercise_1_gaussian_natural_gradient()["final_ll"])
