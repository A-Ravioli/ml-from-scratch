"""
Normalization Techniques - Reference Solutions

This module mirrors `exercise.py` and provides complete implementations for:
- BatchNorm
- LayerNorm
- GroupNorm
- InstanceNorm (for 2D inputs, equivalent to per-sample feature normalization)
- RMSNorm

All implementations are NumPy-based and intended for CPU, educational use.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


class NormalizationLayer(ABC):
    """Base class for normalization layers."""

    def __init__(self):
        self.training = True
        self.epsilon = 1e-8

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass during training or inference."""
        ...

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Backward pass: returns (grad_input, parameter_gradients)."""
        ...

    @abstractmethod
    def update_parameters(self, gradients: Dict, learning_rate: float):
        """Update learnable parameters."""
        ...

    def set_training_mode(self, training: bool):
        self.training = bool(training)


class BatchNormalization(NormalizationLayer):
    """Batch Normalization for 2D inputs (N, C)."""

    def __init__(self, num_features: int, momentum: float = 0.9):
        super().__init__()
        self.num_features = int(num_features)
        self.momentum = float(momentum)

        self.gamma = np.ones(self.num_features, dtype=float)
        self.beta = np.zeros(self.num_features, dtype=float)

        self.running_mean = np.zeros(self.num_features, dtype=float)
        self.running_var = np.ones(self.num_features, dtype=float)

        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_hat = (x - mean) / np.sqrt(var + self.epsilon)
            out = self.gamma * x_hat + self.beta

            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var

            self.cache = {"x": x, "mean": mean, "var": var, "x_hat": x_hat}
            return out

        x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * x_hat + self.beta

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict]:
        grad_output = np.asarray(grad_output, dtype=float)
        x = self.cache["x"]
        mean = self.cache["mean"]
        var = self.cache["var"]
        x_hat = self.cache["x_hat"]

        N = x.shape[0]
        inv_std = 1.0 / np.sqrt(var + self.epsilon)

        grad_beta = np.sum(grad_output, axis=0)
        grad_gamma = np.sum(grad_output * x_hat, axis=0)

        dx_hat = grad_output * self.gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * inv_std**3, axis=0)
        dmean = np.sum(dx_hat * -inv_std, axis=0) + dvar * np.sum(-2.0 * (x - mean), axis=0) / N
        dx = dx_hat * inv_std + dvar * 2.0 * (x - mean) / N + dmean / N

        return dx, {"gamma": grad_gamma, "beta": grad_beta}

    def update_parameters(self, gradients: Dict, learning_rate: float):
        lr = float(learning_rate)
        self.gamma -= lr * gradients["gamma"]
        self.beta -= lr * gradients["beta"]


class LayerNormalization(NormalizationLayer):
    """Layer Normalization for 2D inputs (N, C): normalize across features per sample."""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = int(num_features)

        self.gamma = np.ones(self.num_features, dtype=float)
        self.beta = np.zeros(self.num_features, dtype=float)
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * x_hat + self.beta
        self.cache = {"x": x, "mean": mean, "var": var, "x_hat": x_hat}
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict]:
        grad_output = np.asarray(grad_output, dtype=float)
        x = self.cache["x"]
        mean = self.cache["mean"]
        var = self.cache["var"]
        x_hat = self.cache["x_hat"]

        D = x.shape[1]
        inv_std = 1.0 / np.sqrt(var + self.epsilon)

        grad_beta = np.sum(grad_output, axis=0)
        grad_gamma = np.sum(grad_output * x_hat, axis=0)

        dx_hat = grad_output * self.gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * inv_std**3, axis=1, keepdims=True)
        dmean = np.sum(dx_hat * -inv_std, axis=1, keepdims=True) + dvar * np.sum(-2.0 * (x - mean), axis=1, keepdims=True) / D
        dx = dx_hat * inv_std + dvar * 2.0 * (x - mean) / D + dmean / D
        return dx, {"gamma": grad_gamma, "beta": grad_beta}

    def update_parameters(self, gradients: Dict, learning_rate: float):
        lr = float(learning_rate)
        self.gamma -= lr * gradients["gamma"]
        self.beta -= lr * gradients["beta"]


class GroupNormalization(NormalizationLayer):
    """Group Normalization for 2D inputs (N, C): normalize across feature groups per sample."""

    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_features = int(num_features)
        self.num_groups = int(num_groups)
        if self.num_features % self.num_groups != 0:
            raise ValueError("num_features must be divisible by num_groups")
        self.group_size = self.num_features // self.num_groups

        self.gamma = np.ones(self.num_features, dtype=float)
        self.beta = np.zeros(self.num_features, dtype=float)
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        xg = x.reshape(N, self.num_groups, self.group_size)
        mean = np.mean(xg, axis=2, keepdims=True)
        var = np.var(xg, axis=2, keepdims=True)
        xg_hat = (xg - mean) / np.sqrt(var + self.epsilon)
        x_hat = xg_hat.reshape(N, self.num_features)
        out = self.gamma * x_hat + self.beta
        self.cache = {"x": x, "mean": mean, "var": var, "x_hat": x_hat}
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict]:
        grad_output = np.asarray(grad_output, dtype=float)
        x = self.cache["x"]
        mean = self.cache["mean"]
        var = self.cache["var"]
        x_hat = self.cache["x_hat"]

        N = x.shape[0]
        Dg = self.group_size

        grad_beta = np.sum(grad_output, axis=0)
        grad_gamma = np.sum(grad_output * x_hat, axis=0)

        dx_hat = grad_output * self.gamma
        dxg_hat = dx_hat.reshape(N, self.num_groups, self.group_size)
        xg = x.reshape(N, self.num_groups, self.group_size)

        inv_std = 1.0 / np.sqrt(var + self.epsilon)
        dvar = np.sum(dxg_hat * (xg - mean) * -0.5 * inv_std**3, axis=2, keepdims=True)
        dmean = np.sum(dxg_hat * -inv_std, axis=2, keepdims=True) + dvar * np.sum(-2.0 * (xg - mean), axis=2, keepdims=True) / Dg
        dxg = dxg_hat * inv_std + dvar * 2.0 * (xg - mean) / Dg + dmean / Dg
        dx = dxg.reshape(N, self.num_features)

        return dx, {"gamma": grad_gamma, "beta": grad_beta}

    def update_parameters(self, gradients: Dict, learning_rate: float):
        lr = float(learning_rate)
        self.gamma -= lr * gradients["gamma"]
        self.beta -= lr * gradients["beta"]


class InstanceNormalization(NormalizationLayer):
    """
    Instance Normalization for 2D inputs (N, C).

    In convolutional settings, instance norm normalizes per-sample per-channel across spatial dims.
    For 2D vectors, we normalize per sample across features (same reduction as LayerNorm).
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = int(num_features)
        self.gamma = np.ones(self.num_features, dtype=float)
        self.beta = np.zeros(self.num_features, dtype=float)
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * x_hat + self.beta
        self.cache = {"x": x, "mean": mean, "var": var, "x_hat": x_hat}
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict]:
        grad_output = np.asarray(grad_output, dtype=float)
        x = self.cache["x"]
        mean = self.cache["mean"]
        var = self.cache["var"]
        x_hat = self.cache["x_hat"]

        D = x.shape[1]
        inv_std = 1.0 / np.sqrt(var + self.epsilon)

        grad_beta = np.sum(grad_output, axis=0)
        grad_gamma = np.sum(grad_output * x_hat, axis=0)

        dx_hat = grad_output * self.gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * inv_std**3, axis=1, keepdims=True)
        dmean = np.sum(dx_hat * -inv_std, axis=1, keepdims=True) + dvar * np.sum(-2.0 * (x - mean), axis=1, keepdims=True) / D
        dx = dx_hat * inv_std + dvar * 2.0 * (x - mean) / D + dmean / D
        return dx, {"gamma": grad_gamma, "beta": grad_beta}

    def update_parameters(self, gradients: Dict, learning_rate: float):
        lr = float(learning_rate)
        self.gamma -= lr * gradients["gamma"]
        self.beta -= lr * gradients["beta"]


class RMSNorm(NormalizationLayer):
    """RMSNorm for 2D inputs (N, C): normalize by root-mean-square per sample."""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = int(num_features)
        self.gamma = np.ones(self.num_features, dtype=float)
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True) + self.epsilon)
        x_hat = x / rms
        out = self.gamma * x_hat
        self.cache = {"x": x, "rms": rms, "x_hat": x_hat}
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict]:
        grad_output = np.asarray(grad_output, dtype=float)
        x = self.cache["x"]
        rms = self.cache["rms"]
        x_hat = self.cache["x_hat"]

        D = x.shape[1]

        grad_gamma = np.sum(grad_output * x_hat, axis=0)
        dy = grad_output * self.gamma

        # dx = dy/rms - x * (sum(dy*x)/ (D*rms^3))
        dot = np.sum(dy * x, axis=1, keepdims=True)
        dx = dy / rms - x * (dot / (D * (rms**3)))

        return dx, {"gamma": grad_gamma}

    def update_parameters(self, gradients: Dict, learning_rate: float):
        self.gamma -= float(learning_rate) * gradients["gamma"]


class NormalizationAnalyzer:
    """Analyze the effects of different normalization techniques (lightweight)."""

    def __init__(self):
        self.normalizations = {
            "none": None,
            "batch_norm": lambda: BatchNormalization(50),
            "layer_norm": lambda: LayerNormalization(50),
            "group_norm": lambda: GroupNormalization(50, 10),
            "instance_norm": lambda: InstanceNormalization(50),
            "rms_norm": lambda: RMSNorm(50),
        }

    def analyze_activation_statistics(self, layer_sizes: List[int], activation_func: Callable, n_samples: int = 1000) -> Dict:
        results: Dict[str, List[Dict]] = {}
        X = np.random.randn(int(n_samples), int(layer_sizes[0]))

        for norm_name, norm_factory in self.normalizations.items():
            layer_stats: List[Dict] = []
            current_input = X

            for i in range(len(layer_sizes) - 1):
                fan_in = int(layer_sizes[i])
                fan_out = int(layer_sizes[i + 1])
                W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / (fan_in + fan_out))
                b = np.zeros(fan_out)

                pre = current_input @ W.T + b
                z = pre
                if norm_factory is not None:
                    z = norm_factory().forward(z, training=True)
                a = activation_func(z)

                layer_stats.append(
                    {
                        "layer": i,
                        "pre_norm_stats": self._compute_stats(pre),
                        "post_norm_stats": self._compute_stats(z),
                        "activation_stats": self._compute_stats(a),
                    }
                )
                current_input = a

            results[norm_name] = layer_stats

        return results

    def _compute_stats(self, x: np.ndarray) -> Dict:
        x = np.asarray(x, dtype=float)
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "skewness": float(self._compute_skewness(x)),
            "kurtosis": float(self._compute_kurtosis(x)),
        }

    def _compute_skewness(self, x: np.ndarray) -> float:
        mean = float(np.mean(x))
        std = float(np.std(x))
        if std <= 0:
            return 0.0
        z = (x - mean) / std
        return float(np.mean(z**3))

    def _compute_kurtosis(self, x: np.ndarray) -> float:
        mean = float(np.mean(x))
        std = float(np.std(x))
        if std <= 0:
            return 0.0
        z = (x - mean) / std
        return float(np.mean(z**4) - 3.0)

    def analyze_gradient_flow(self, layer_sizes: List[int], activation_func: Callable, n_epochs: int = 10) -> Dict:
        # Keep extremely lightweight: just return per-layer gradient norms after one synthetic backward pass.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, int(layer_sizes[0])))
        y = rng.normal(size=(64, int(layer_sizes[-1])))

        results: Dict[str, List[float]] = {}
        for norm_name, norm_factory in self.normalizations.items():
            # one forward/backward through linear stack with optional normalization and tanh activation
            weights: List[np.ndarray] = []
            norms: List[Optional[NormalizationLayer]] = []
            for i in range(len(layer_sizes) - 1):
                fan_in = int(layer_sizes[i])
                fan_out = int(layer_sizes[i + 1])
                weights.append(rng.normal(size=(fan_out, fan_in)) * np.sqrt(2.0 / (fan_in + fan_out)))
                norms.append(None if norm_factory is None else norm_factory())

            a = X
            zs: List[np.ndarray] = []
            as_: List[np.ndarray] = [a]
            for W, norm in zip(weights, norms):
                z = a @ W.T
                if norm is not None:
                    z = norm.forward(z, training=True)
                a = activation_func(z)
                zs.append(z)
                as_.append(a)

            delta = (a - y) / len(X)
            grad_norms: List[float] = []
            for li in range(len(weights) - 1, -1, -1):
                a_prev = as_[li]
                gW = delta.T @ a_prev
                grad_norms.append(float(np.linalg.norm(gW)))
                if li > 0:
                    delta = delta @ weights[li]
            results[norm_name] = list(reversed(grad_norms))
        return results

    def analyze_training_stability(self, X_train: np.ndarray, y_train: np.ndarray, learning_rates: List[float]) -> Dict:
        # Minimal: run a few SGD steps and record whether loss is finite.
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        rng = np.random.default_rng(0)
        results: Dict[str, Dict[float, Dict[str, float]]] = {}

        for norm_name, norm_factory in self.normalizations.items():
            per_lr: Dict[float, Dict[str, float]] = {}
            for lr in learning_rates:
                W = rng.normal(size=(y_train.shape[1], X_train.shape[1])) * 0.1
                norm = None if norm_factory is None else norm_factory()
                loss = None
                for _ in range(5):
                    z = X_train @ W.T
                    if norm is not None:
                        z = norm.forward(z, training=True)
                    preds = np.tanh(z)
                    loss = float(0.5 * np.mean((preds - y_train) ** 2))
                    delta = (preds - y_train) / len(X_train)
                    gW = delta.T @ X_train
                    W -= float(lr) * gW
                per_lr[float(lr)] = {"final_loss": float(loss), "finite": float(np.isfinite(loss))}
            results[norm_name] = per_lr
        return results

    def analyze_batch_size_sensitivity(self, X_train: np.ndarray, y_train: np.ndarray, batch_sizes: List[int]) -> Dict:
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        rng = np.random.default_rng(0)
        results: Dict[str, Dict[int, float]] = {}

        for norm_name, norm_factory in self.normalizations.items():
            per_bs: Dict[int, float] = {}
            for bs in batch_sizes:
                bs = int(bs)
                idx = rng.choice(len(X_train), size=min(bs, len(X_train)), replace=False)
                xb = X_train[idx]
                yb = y_train[idx]
                W = rng.normal(size=(yb.shape[1], xb.shape[1])) * 0.1
                norm = None if norm_factory is None else norm_factory()
                z = xb @ W.T
                if norm is not None:
                    z = norm.forward(z, training=True)
                preds = np.tanh(z)
                loss = float(0.5 * np.mean((preds - yb) ** 2))
                per_bs[bs] = loss
            results[norm_name] = per_bs
        return results


def compare_normalization_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_sizes: List[int],
) -> Dict:
    # Minimal comparison: one forward evaluation of activation statistics + simple loss.
    analyzer = NormalizationAnalyzer()
    stats = analyzer.analyze_activation_statistics(layer_sizes, np.tanh, n_samples=min(512, len(X_train)))
    return {"activation_stats": stats}


def plot_normalization_effects(results: Dict):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.reshape(2, 3)

    if "activation_stats" not in results:
        plt.close(fig)
        return

    activation_stats = results["activation_stats"]
    methods = list(activation_stats.keys())
    layer0_pre_std = [activation_stats[m][0]["pre_norm_stats"]["std"] for m in methods]
    layer0_post_std = [activation_stats[m][0]["post_norm_stats"]["std"] for m in methods]
    layer0_act_std = [activation_stats[m][0]["activation_stats"]["std"] for m in methods]

    axes[0, 0].bar(methods, layer0_pre_std)
    axes[0, 0].set_title("Layer0 pre-norm std")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(methods, layer0_post_std)
    axes[0, 1].set_title("Layer0 post-norm std")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[0, 2].bar(methods, layer0_act_std)
    axes[0, 2].set_title("Layer0 activation std")
    axes[0, 2].tick_params(axis="x", rotation=45)

    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def exercise_1_normalization_implementations():
    print("=== Exercise 1: Normalization Implementations ===")
    x = np.random.randn(16, 10)
    bn = BatchNormalization(10)
    ln = LayerNormalization(10)
    gn = GroupNormalization(10, 5)
    rn = RMSNorm(10)
    print("bn std", np.std(bn.forward(x, training=True)))
    print("ln std", np.std(ln.forward(x, training=True)))
    print("gn std", np.std(gn.forward(x, training=True)))
    print("rn std", np.std(rn.forward(x, training=True)))


def exercise_2_activation_statistics():
    print("=== Exercise 2: Activation Statistics ===")
    analyzer = NormalizationAnalyzer()
    res = analyzer.analyze_activation_statistics([50, 50, 50], np.tanh, n_samples=256)
    print(list(res.keys()))


def exercise_3_training_dynamics():
    print("=== Exercise 3: Training Dynamics ===")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 20))
    y = rng.normal(size=(256, 5))
    analyzer = NormalizationAnalyzer()
    res = analyzer.analyze_training_stability(X, y, learning_rates=[0.01, 0.05])
    print("keys", list(res.keys()))


def exercise_4_batch_size_effects():
    print("=== Exercise 4: Batch Size Effects ===")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 20))
    y = rng.normal(size=(256, 5))
    analyzer = NormalizationAnalyzer()
    res = analyzer.analyze_batch_size_sensitivity(X, y, batch_sizes=[4, 16, 64])
    print("batch_norm", res["batch_norm"])


def exercise_5_gradient_flow_analysis():
    print("=== Exercise 5: Gradient Flow Analysis ===")
    analyzer = NormalizationAnalyzer()
    res = analyzer.analyze_gradient_flow([20, 32, 16, 5], np.tanh, n_epochs=1)
    print("layer norms (none)", res["none"])


def exercise_6_practical_applications():
    print("=== Exercise 6: Practical Applications ===")
    print("- BatchNorm: best with moderate/large batches.")
    print("- LayerNorm/RMSNorm: batch-size independent, common in transformers.")
    print("- GroupNorm: robust alternative when batch sizes are small.")


if __name__ == "__main__":
    start = time.time()
    exercise_1_normalization_implementations()
    exercise_2_activation_statistics()
    exercise_3_training_dynamics()
    exercise_4_batch_size_effects()
    exercise_5_gradient_flow_analysis()
    exercise_6_practical_applications()
    print("Done in", round(time.time() - start, 2), "s")

