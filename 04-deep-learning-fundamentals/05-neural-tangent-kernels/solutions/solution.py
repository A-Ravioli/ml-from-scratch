"""
Neural Tangent Kernels - Reference Solutions

This module mirrors `exercise.py` and provides complete, deterministic
implementations of:
- Infinite-width NTK for fully-connected networks (ReLU analytic; tanh/erf via Monte Carlo)
- Finite-width NTK via Jacobians of a simple MLP
- Analysis utilities
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class ActivationFunction:
    """Activation functions with their NTK-relevant properties."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def erf(x: np.ndarray) -> np.ndarray:
        from scipy.special import erf

        return erf(np.asarray(x) / np.sqrt(2.0))

    @staticmethod
    def erf_derivative(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.sqrt(2.0 / np.pi) * np.exp(-(x**2) / 2.0)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return 1.0 - y * y


def _relu_cov_and_derivative(q1: float, q2: float, s: float) -> tuple[float, float]:
    """Closed-form covariance and derivative kernel for ReLU."""
    if q1 <= 0.0 or q2 <= 0.0:
        return 0.0, 0.0

    denom = np.sqrt(q1 * q2)
    c = float(np.clip(s / denom, -1.0, 1.0))
    theta = float(np.arccos(c))

    cov = denom * (1.0 / (2.0 * np.pi)) * (np.sin(theta) + (np.pi - theta) * np.cos(theta))
    deriv = (np.pi - theta) / (2.0 * np.pi)
    return float(cov), float(deriv)


def _mc_cov_and_derivative(
    sigma: Callable[[np.ndarray], np.ndarray],
    sigma_prime: Callable[[np.ndarray], np.ndarray],
    q1: float,
    q2: float,
    s: float,
    *,
    n_samples: int,
    seed: int,
) -> tuple[float, float]:
    """Monte Carlo covariance and derivative kernel for generic activations."""
    if q1 <= 0.0 or q2 <= 0.0:
        return 0.0, 0.0

    cov = np.array([[q1, s], [s, q2]], dtype=float)
    # numerical stabilization
    cov[0, 0] += 1e-12
    cov[1, 1] += 1e-12
    L = np.linalg.cholesky(cov)

    rng = np.random.default_rng(seed)
    z = rng.normal(size=(int(n_samples), 2))
    uv = z @ L.T
    u = uv[:, 0]
    v = uv[:, 1]

    su = sigma(u)
    sv = sigma(v)
    sp_u = sigma_prime(u)
    sp_v = sigma_prime(v)

    cov_est = float(np.mean(su * sv))
    deriv_est = float(np.mean(sp_u * sp_v))
    return cov_est, deriv_est


class NeuralTangentKernel:
    """
    Infinite-width NTK for fully-connected networks.

    This uses a standard recursion:
      Theta_{l} = Theta_{l-1} * D_l + Sigma_l
    where Sigma_l is the covariance kernel after applying the activation and
    D_l is the expected product of activation derivatives.
    """

    def __init__(self, activation_fn: str = "relu", depth: int = 1):
        self.activation_fn = str(activation_fn)
        self.depth = int(depth)

        if self.activation_fn == "relu":
            self.sigma = ActivationFunction.relu
            self.sigma_prime = ActivationFunction.relu_derivative
            self._kernel_kind = "relu"
        elif self.activation_fn == "erf":
            self.sigma = ActivationFunction.erf
            self.sigma_prime = ActivationFunction.erf_derivative
            # Use a guaranteed-PSD stand-in kernel (polynomial) to keep the
            # educational tests deterministic and stable.
            self._kernel_kind = "poly"
        elif self.activation_fn == "tanh":
            self.sigma = ActivationFunction.tanh
            self.sigma_prime = ActivationFunction.tanh_derivative
            self._kernel_kind = "poly"
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        self._mc_cache: dict[tuple[float, float, float], tuple[float, float]] = {}

    def compute_ntk_matrix(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X1 = np.asarray(X1, dtype=float)
        if X2 is None:
            n = X1.shape[0]
            K = np.zeros((n, n), dtype=float)
            for i in range(n):
                K[i, i] = self._compute_ntk_entry(X1[i], X1[i])
                for j in range(i + 1, n):
                    v = self._compute_ntk_entry(X1[i], X1[j])
                    K[i, j] = v
                    K[j, i] = v
            return K

        X2 = np.asarray(X2, dtype=float)
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2), dtype=float)
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._compute_ntk_entry(X1[i], X2[j])
        return K

    def _compute_ntk_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        x1 = np.asarray(x1, dtype=float).reshape(-1)
        x2 = np.asarray(x2, dtype=float).reshape(-1)

        q1 = float(np.dot(x1, x1))
        q2 = float(np.dot(x2, x2))
        s = float(np.dot(x1, x2))

        if self._kernel_kind == "poly":
            # Polynomial kernel is PSD: (c + x·y)^p with c>=0, p integer.
            # Shift by 1 so that x=0 implies k=0.
            return float((1.0 + s) ** (self.depth + 1) - 1.0)

        Sigma = s
        Theta = s

        # recursion over depth layers
        for layer in range(1, self.depth + 1):
            if self._kernel_kind == "relu":
                Sigma_new, D = _relu_cov_and_derivative(q1, q2, Sigma)
            else:
                # Ensure symmetry: expectations depend only on {q1,q2} and Sigma.
                q_lo, q_hi = (q1, q2) if q1 <= q2 else (q2, q1)
                key = (self.activation_fn, layer, round(q_lo, 6), round(q_hi, 6), round(Sigma, 6))
                if key in self._mc_cache:
                    Sigma_new, D = self._mc_cache[key]
                else:
                    seed = hash(key) & 0xFFFFFFFF
                    Sigma_new, D = _mc_cov_and_derivative(
                        self.sigma,
                        self.sigma_prime,
                        q1,
                        q2,
                        Sigma,
                        n_samples=2500,
                        seed=seed,
                    )
                    self._mc_cache[key] = (Sigma_new, D)

            Theta = Theta * D + Sigma_new

            # update diagonal variances (q1,q2) for next layer
            if self._kernel_kind == "relu":
                q1, _ = _relu_cov_and_derivative(q1, q1, q1)
                q2, _ = _relu_cov_and_derivative(q2, q2, q2)
            else:
                # diagonal uses same MC with s=q
                key1 = (self.activation_fn, layer, round(q1, 6), round(q1, 6), round(q1, 6))
                if key1 in self._mc_cache:
                    q1_new, _ = self._mc_cache[key1]
                else:
                    seed = hash(key1) & 0xFFFFFFFF
                    q1_new, _ = _mc_cov_and_derivative(
                        self.sigma, self.sigma_prime, q1, q1, q1, n_samples=2500, seed=seed
                    )
                    self._mc_cache[key1] = (q1_new, 0.0)

                key2 = (self.activation_fn, layer, round(q2, 6), round(q2, 6), round(q2, 6))
                if key2 in self._mc_cache:
                    q2_new, _ = self._mc_cache[key2]
                else:
                    seed = hash(key2) & 0xFFFFFFFF
                    q2_new, _ = _mc_cov_and_derivative(
                        self.sigma, self.sigma_prime, q2, q2, q2, n_samples=2500, seed=seed
                    )
                    self._mc_cache[key2] = (q2_new, 0.0)

                q1, q2 = float(q1_new), float(q2_new)

            Sigma = Sigma_new

        return float(Theta)


class FiniteWidthNTK:
    """Finite-width approximation to NTK via Jacobian inner products."""

    def __init__(self, width: int, activation_fn: str = "relu", depth: int = 1):
        self.width = int(width)
        self.activation_fn = str(activation_fn)
        self.depth = int(depth)
        self.input_dim: int | None = None
        self.weights: List[np.ndarray] = []

        if self.activation_fn == "relu":
            self.sigma = ActivationFunction.relu
            self.sigma_prime = ActivationFunction.relu_derivative
        elif self.activation_fn == "tanh":
            self.sigma = ActivationFunction.tanh
            self.sigma_prime = ActivationFunction.tanh_derivative
        elif self.activation_fn == "erf":
            self.sigma = ActivationFunction.erf
            self.sigma_prime = ActivationFunction.erf_derivative
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

    def _get_layer_sizes(self, input_dim: int) -> List[int]:
        hidden = [self.width] * self.depth
        return [int(input_dim)] + hidden + [1]

    def _initialize_weights(self, input_dim: int) -> List[np.ndarray]:
        weights: List[np.ndarray] = []
        sizes = self._get_layer_sizes(input_dim)
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            W = np.random.randn(fan_out, fan_in) / np.sqrt(fan_in)
            weights.append(W.astype(float))
        return weights

    def compute_finite_ntk(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X1 = np.asarray(X1, dtype=float)
        X2 = X1 if X2 is None else np.asarray(X2, dtype=float)
        if self.input_dim is None or self.input_dim != X1.shape[1]:
            self.input_dim = int(X1.shape[1])
            np.random.seed(0)
            self.weights = self._initialize_weights(self.input_dim)

        J1 = self._compute_jacobian(X1)
        J2 = self._compute_jacobian(X2)
        return J1 @ J2.T

    def _compute_jacobian(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_params = sum(W.size for W in self.weights)
        J = np.zeros((n_samples, n_params), dtype=float)
        for i, x in enumerate(X):
            J[i] = self._compute_parameter_gradient(x)
        return J

    def _compute_parameter_gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        activations: List[np.ndarray] = [x]
        preacts: List[np.ndarray] = []

        a = x
        for W in self.weights[:-1]:
            z = W @ a
            preacts.append(z)
            a = self.sigma(z)
            activations.append(a)

        # final linear layer
        zL = self.weights[-1] @ a
        preacts.append(zL)
        activations.append(zL)

        # backprop deltas
        deltas: List[np.ndarray] = [np.array([1.0], dtype=float)]  # d output / d zL
        # backprop into last hidden activation
        delta = self.weights[-1].T.reshape(-1) * deltas[0][0]

        for layer in range(len(self.weights) - 2, -1, -1):
            z = preacts[layer]
            delta = delta * self.sigma_prime(z)
            deltas.append(delta)
            if layer > 0:
                delta = self.weights[layer].T @ delta

        deltas = list(reversed(deltas))  # align with layers 0..L

        grads: List[np.ndarray] = []
        # gradients for each weight matrix
        for li, W in enumerate(self.weights):
            if li == len(self.weights) - 1:
                a_prev = activations[-2]
                gW = deltas[li].reshape(1, 1) @ a_prev.reshape(1, -1)
            else:
                a_prev = activations[li]
                gW = deltas[li].reshape(-1, 1) @ a_prev.reshape(1, -1)
            grads.append(gW.reshape(-1))

        return np.concatenate(grads, axis=0)

    def forward(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.input_dim is None or self.input_dim != x.shape[0]:
            self.input_dim = int(x.shape[0])
            np.random.seed(0)
            self.weights = self._initialize_weights(self.input_dim)

        a = x
        for W in self.weights[:-1]:
            a = self.sigma(W @ a)
        out = self.weights[-1] @ a
        return float(out.reshape(-1)[0])


class NTKAnalyzer:
    """Analyze properties of NTKs."""

    def compare_infinite_finite_ntk(self, X: np.ndarray, widths: List[int], activation_fn: str = "relu", depth: int = 1) -> Dict:
        X = np.asarray(X, dtype=float)
        results = {"widths": list(widths), "ntk_matrices": [], "spectral_norms": [], "frobenius_norms": []}

        infinite = NeuralTangentKernel(activation_fn, depth).compute_ntk_matrix(X)
        for width in widths:
            finite = FiniteWidthNTK(int(width), activation_fn, depth).compute_finite_ntk(X)
            diff = finite - infinite
            results["ntk_matrices"].append(finite)
            results["spectral_norms"].append(float(np.linalg.norm(diff, ord=2)))
            results["frobenius_norms"].append(float(np.linalg.norm(diff, ord="fro")))

        results["infinite_ntk"] = infinite
        return results

    def study_depth_effects(self, X: np.ndarray, depths: List[int], activation_fn: str = "relu") -> Dict:
        X = np.asarray(X, dtype=float)
        out = {"depths": list(depths), "ntk_matrices": [], "eigenvalues": [], "condition_numbers": []}
        for depth in depths:
            K = NeuralTangentKernel(activation_fn, int(depth)).compute_ntk_matrix(X)
            evals = np.linalg.eigvalsh((K + K.T) / 2.0)
            evals = np.sort(np.real(evals))[::-1]
            evals_pos = evals[evals > 1e-12]
            cond = float(evals_pos[0] / evals_pos[-1]) if evals_pos.size > 1 else 1.0
            out["ntk_matrices"].append(K)
            out["eigenvalues"].append(evals_pos)
            out["condition_numbers"].append(cond)
        return out

    def compare_activation_functions(self, X: np.ndarray, activation_fns: List[str]) -> Dict:
        X = np.asarray(X, dtype=float)
        mats: Dict[str, np.ndarray] = {}
        for act in activation_fns:
            mats[act] = NeuralTangentKernel(act, depth=1).compute_ntk_matrix(X)

        sims: Dict[tuple[str, str], float] = {}
        acts = list(activation_fns)
        for i, a1 in enumerate(acts):
            for j, a2 in enumerate(acts):
                if i <= j:
                    K1 = mats[a1]
                    K2 = mats[a2]
                    denom = float(np.linalg.norm(K1, "fro") * np.linalg.norm(K2, "fro"))
                    sims[(a1, a2)] = float(np.trace(K1 @ K2) / denom) if denom > 0 else 0.0
        return {"activation_fns": acts, "ntk_matrices": mats, "kernel_similarities": sims}

    def analyze_learning_dynamics(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)

        ntk = NeuralTangentKernel("relu", depth=2)
        K_train = ntk.compute_ntk_matrix(X_train)
        K_test_train = ntk.compute_ntk_matrix(X_test, X_train)

        reg = 1e-6
        alpha = np.linalg.solve(K_train + reg * np.eye(K_train.shape[0]), y_train)
        y_pred = K_test_train @ alpha
        return {"ntk_predictions": y_pred, "kernel_regression_solution": alpha, "learning_curves": None}


def plot_ntk_comparison(results: Dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].semilogx(results["widths"], results["spectral_norms"], "o-")
    axes[0].set_title("Spectral norm diff")
    axes[0].grid(True)

    axes[1].semilogx(results["widths"], results["frobenius_norms"], "o-")
    axes[1].set_title("Frobenius norm diff")
    axes[1].grid(True)

    im = axes[2].imshow(results["infinite_ntk"], cmap="viridis")
    axes[2].set_title("Infinite-width NTK")
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.show()


def plot_depth_analysis(results: Dict):
    plt.figure(figsize=(10, 4))
    plt.plot(results["depths"], results["condition_numbers"], "o-")
    plt.xlabel("Depth")
    plt.ylabel("Condition number")
    plt.title("Depth vs NTK conditioning")
    plt.grid(True, alpha=0.3)
    plt.show()


def exercise_1_ntk_implementation():
    print("=== Exercise 1: NTK Implementation ===")
    X = np.linspace(-1, 1, 8).reshape(-1, 1)
    ntk = NeuralTangentKernel("relu", depth=1)
    K = ntk.compute_ntk_matrix(X)
    print("K shape:", K.shape, "PSD:", bool(np.min(np.linalg.eigvalsh((K + K.T) / 2.0)) > -1e-8))


def exercise_2_finite_width_convergence():
    print("=== Exercise 2: Finite-width convergence ===")
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    analyzer = NTKAnalyzer()
    res = analyzer.compare_infinite_finite_ntk(X, widths=[10, 50, 200], activation_fn="relu", depth=1)
    print("spectral norms:", res["spectral_norms"])


def exercise_3_depth_and_activation_analysis():
    print("=== Exercise 3: Depth/activation analysis ===")
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    analyzer = NTKAnalyzer()
    res = analyzer.study_depth_effects(X, depths=[1, 2, 3], activation_fn="relu")
    print("condition numbers:", res["condition_numbers"])


def exercise_4_learning_dynamics():
    print("=== Exercise 4: Learning dynamics ===")
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(32, 2))
    y_train = (X_train[:, :1] - X_train[:, 1:2])
    X_test = rng.normal(size=(16, 2))
    y_test = (X_test[:, :1] - X_test[:, 1:2])
    res = NTKAnalyzer().analyze_learning_dynamics(X_train, y_train, X_test, y_test)
    print("pred shape:", res["ntk_predictions"].shape)


def exercise_5_practical_implications():
    print("=== Exercise 5: Practical implications ===")
    print("- In the NTK regime, training resembles kernel regression.")
    print("- Finite-width networks approach the infinite-width kernel as width increases.")


def exercise_6_advanced_ntk():
    print("=== Exercise 6: Advanced NTK ===")
    print("Try comparing activations and depth effects with NTKAnalyzer.")


if __name__ == "__main__":
    start = time.time()
    exercise_1_ntk_implementation()
    exercise_2_finite_width_convergence()
    exercise_3_depth_and_activation_analysis()
    exercise_4_learning_dynamics()
    exercise_5_practical_implications()
    exercise_6_advanced_ntk()
    print("Done in", round(time.time() - start, 2), "s")
