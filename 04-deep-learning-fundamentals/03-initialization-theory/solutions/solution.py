"""
Neural Network Initialization Theory - Reference Solutions

This module mirrors `exercise.py` and provides completed implementations of
initialization strategies and analysis utilities.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class WeightInitializer(ABC):
    """Base class for weight initialization strategies."""

    @abstractmethod
    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Initialize weight matrix with shape (fan_out, fan_in)."""
        ...

    @abstractmethod
    def initialize_biases(self, size: int) -> np.ndarray:
        """Initialize bias vector with shape (size,)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of initialization method."""
        ...


class ZeroInitializer(WeightInitializer):
    """Zero initialization (useful for demonstrating symmetry breaking issues)."""

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.zeros((fan_out, fan_in), dtype=float)

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return "Zero"


class RandomNormalInitializer(WeightInitializer):
    """Random normal initialization N(0, std^2)."""

    def __init__(self, std: float = 1.0):
        self.std = float(std)

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.normal(0.0, self.std, size=(fan_out, fan_in))

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return f"Normal(σ={self.std})"


class XavierUniformInitializer(WeightInitializer):
    """Xavier/Glorot uniform: U[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]."""

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
        return np.random.uniform(-bound, bound, size=(fan_out, fan_in))

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return "Xavier Uniform"


class XavierNormalInitializer(WeightInitializer):
    """Xavier/Glorot normal: N(0, 2/(fan_in+fan_out))."""

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        std = float(np.sqrt(2.0 / (fan_in + fan_out)))
        return np.random.normal(0.0, std, size=(fan_out, fan_in))

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return "Xavier Normal"


class HeUniformInitializer(WeightInitializer):
    """He uniform: U[-sqrt(6/fan_in), sqrt(6/fan_in)] (good for ReLU)."""

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        bound = float(np.sqrt(6.0 / fan_in))
        return np.random.uniform(-bound, bound, size=(fan_out, fan_in))

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return "He Uniform"


class HeNormalInitializer(WeightInitializer):
    """He normal: N(0, 2/fan_in) (good for ReLU)."""

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        std = float(np.sqrt(2.0 / fan_in))
        return np.random.normal(0.0, std, size=(fan_out, fan_in))

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return "He Normal"


class LeCunInitializer(WeightInitializer):
    """LeCun normal: N(0, 1/fan_in)."""

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        std = float(np.sqrt(1.0 / fan_in))
        return np.random.normal(0.0, std, size=(fan_out, fan_in))

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return "LeCun"


class OrthogonalInitializer(WeightInitializer):
    """Orthogonal initialization using QR decomposition."""

    def __init__(self, gain: float = 1.0):
        self.gain = float(gain)

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        # Return W shape (fan_out, fan_in).
        if fan_out <= fan_in:
            # Make rows orthonormal: W W^T = I
            a = np.random.normal(size=(fan_in, fan_out))
            q, _ = np.linalg.qr(a, mode="reduced")  # (fan_in, fan_out)
            W = q.T  # (fan_out, fan_in)
        else:
            # Make columns orthonormal: W^T W = I
            a = np.random.normal(size=(fan_out, fan_in))
            q, _ = np.linalg.qr(a, mode="reduced")  # (fan_out, fan_in)
            W = q  # already (fan_out, fan_in)
        return self.gain * W.astype(float)

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return f"Orthogonal(gain={self.gain})"


class VarianceScalingInitializer(WeightInitializer):
    """General variance scaling initializer (subsumes Xavier/He/LeCun)."""

    def __init__(self, scale: float = 1.0, mode: str = "fan_in", distribution: str = "normal"):
        self.scale = float(scale)
        self.mode = str(mode)
        self.distribution = str(distribution)

    def initialize_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        if self.mode == "fan_in":
            fan = float(fan_in)
        elif self.mode == "fan_out":
            fan = float(fan_out)
        elif self.mode == "fan_avg":
            fan = float(fan_in + fan_out) / 2.0
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        variance = self.scale / fan

        if self.distribution == "normal":
            std = float(np.sqrt(variance))
            return np.random.normal(0.0, std, size=(fan_out, fan_in))
        if self.distribution == "uniform":
            limit = float(np.sqrt(3.0 * variance))
            return np.random.uniform(-limit, limit, size=(fan_out, fan_in))

        raise ValueError(f"Unknown distribution: {self.distribution}")

    def initialize_biases(self, size: int) -> np.ndarray:
        return np.zeros((size,), dtype=float)

    @property
    def name(self) -> str:
        return f"VarScale(scale={self.scale}, mode={self.mode}, dist={self.distribution})"


class InitializationAnalyzer:
    """Analyze the effects of different initialization strategies."""

    def __init__(self):
        self.initializers: Dict[str, WeightInitializer] = {
            "zero": ZeroInitializer(),
            "normal_1": RandomNormalInitializer(1.0),
            "normal_0.1": RandomNormalInitializer(0.1),
            "xavier_uniform": XavierUniformInitializer(),
            "xavier_normal": XavierNormalInitializer(),
            "he_uniform": HeUniformInitializer(),
            "he_normal": HeNormalInitializer(),
            "lecun": LeCunInitializer(),
            "orthogonal": OrthogonalInitializer(),
        }

    def analyze_activation_statistics(
        self, layer_sizes: List[int], activation_func: Callable, n_samples: int = 1000
    ) -> Dict:
        results: Dict[str, List[Dict]] = {}
        X = np.random.randn(int(n_samples), int(layer_sizes[0]))

        for name, initializer in self.initializers.items():
            layer_stats: List[Dict] = []
            current_input = X

            for i in range(len(layer_sizes) - 1):
                fan_in = int(layer_sizes[i])
                fan_out = int(layer_sizes[i + 1])

                W = initializer.initialize_weights(fan_in, fan_out)  # (fan_out, fan_in)
                b = initializer.initialize_biases(fan_out)

                z = current_input @ W.T + b
                a = activation_func(z)

                layer_stats.append(
                    {
                        "layer": i,
                        "pre_activation": {
                            "mean": float(np.mean(z)),
                            "std": float(np.std(z)),
                            "min": float(np.min(z)),
                            "max": float(np.max(z)),
                        },
                        "post_activation": {
                            "mean": float(np.mean(a)),
                            "std": float(np.std(a)),
                            "min": float(np.min(a)),
                            "max": float(np.max(a)),
                            "saturation_rate": float(self._compute_saturation_rate(a, activation_func)),
                        },
                    }
                )

                current_input = a

            results[name] = layer_stats

        return results

    def _compute_saturation_rate(
        self, activations: np.ndarray, activation_func: Callable, threshold: float = 0.01
    ) -> float:
        a = np.asarray(activations)
        a_min = float(np.min(a))
        a_max = float(np.max(a))

        # Heuristic based on output range.
        if a_min >= -1e-9 and a_max <= 1.0 + 1e-9:
            # likely sigmoid-ish [0,1]
            sat = np.minimum(a, 1.0 - a) < threshold
            return float(np.mean(sat))

        if a_min >= -1.0 - 1e-9 and a_max <= 1.0 + 1e-9:
            # likely tanh-ish [-1,1]
            sat = (1.0 - np.abs(a)) < threshold
            return float(np.mean(sat))

        # fallback: relu-like (dead at 0)
        sat = np.abs(a) < threshold
        return float(np.mean(sat))

    def analyze_gradient_flow(
        self,
        layer_sizes: List[int],
        activation_func: Callable,
        activation_derivative: Callable,
        n_samples: int = 100,
    ) -> Dict:
        results: Dict[str, List[float]] = {}
        X = np.random.randn(int(n_samples), int(layer_sizes[0]))
        y = np.random.randn(int(n_samples), int(layer_sizes[-1]))

        for name, initializer in self.initializers.items():
            # Initialize weights and biases
            weights: List[np.ndarray] = []
            biases: List[np.ndarray] = []
            for i in range(len(layer_sizes) - 1):
                fan_in = int(layer_sizes[i])
                fan_out = int(layer_sizes[i + 1])
                weights.append(initializer.initialize_weights(fan_in, fan_out))
                biases.append(initializer.initialize_biases(fan_out))

            # Forward pass
            a = X
            zs: List[np.ndarray] = []
            activations: List[np.ndarray] = [a]
            for W, b in zip(weights, biases):
                z = a @ W.T + b
                a = activation_func(z)
                zs.append(z)
                activations.append(a)

            # MSE loss gradient at output
            d_out = (a - y) / max(1, int(n_samples))
            delta = d_out * activation_derivative(zs[-1])

            grad_norms: List[float] = []
            for layer_idx in range(len(weights) - 1, -1, -1):
                a_prev = activations[layer_idx]
                grad_W = delta.T @ a_prev  # matches W shape (fan_out, fan_in)
                grad_norms.append(float(np.linalg.norm(grad_W)))

                if layer_idx > 0:
                    delta = (delta @ weights[layer_idx]) * activation_derivative(zs[layer_idx - 1])

            results[name] = list(reversed(grad_norms))

        return results

    def compare_training_dynamics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        layer_sizes: List[int],
        n_epochs: int = 100,
    ) -> Dict:
        results: Dict[str, Dict[str, List[float]]] = {}
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)

        def tanh(x: np.ndarray) -> np.ndarray:
            return np.tanh(x)

        def dtanh(x: np.ndarray) -> np.ndarray:
            y = np.tanh(x)
            return 1.0 - y * y

        lr = 0.05
        for name, initializer in self.initializers.items():
            weights: List[np.ndarray] = []
            biases: List[np.ndarray] = []
            for i in range(len(layer_sizes) - 1):
                weights.append(initializer.initialize_weights(layer_sizes[i], layer_sizes[i + 1]))
                biases.append(initializer.initialize_biases(layer_sizes[i + 1]))

            train_losses: List[float] = []
            test_losses: List[float] = []
            grad_norms: List[float] = []

            for _ in range(int(n_epochs)):
                # forward
                a = X_train
                zs: List[np.ndarray] = []
                activs: List[np.ndarray] = [a]
                for W, b in zip(weights, biases):
                    z = a @ W.T + b
                    a = tanh(z)
                    zs.append(z)
                    activs.append(a)

                loss = float(0.5 * np.mean((a - y_train) ** 2))
                train_losses.append(loss)

                # backward
                delta = (a - y_train) / max(1, len(X_train))
                delta = delta * dtanh(zs[-1])

                total_gn = 0.0
                for li in range(len(weights) - 1, -1, -1):
                    a_prev = activs[li]
                    gW = delta.T @ a_prev
                    gb = np.sum(delta, axis=0)
                    total_gn += float(np.linalg.norm(gW))

                    weights[li] -= lr * gW
                    biases[li] -= lr * gb

                    if li > 0:
                        delta = (delta @ weights[li]) * dtanh(zs[li - 1])

                grad_norms.append(total_gn)

                # test loss
                a = X_test
                for W, b in zip(weights, biases):
                    a = tanh(a @ W.T + b)
                test_losses.append(float(0.5 * np.mean((a - y_test) ** 2)))

            results[name] = {"train_loss": train_losses, "test_loss": test_losses, "grad_norm": grad_norms}

        return results

    def theoretical_analysis(self, layer_sizes: List[int]) -> Dict:
        analysis: Dict[str, List[Dict[str, Union[int, float]]]] = {}
        for name, initializer in self.initializers.items():
            layer_analysis: List[Dict[str, Union[int, float]]] = []
            for i in range(len(layer_sizes) - 1):
                fan_in = int(layer_sizes[i])
                fan_out = int(layer_sizes[i + 1])
                W = initializer.initialize_weights(fan_in, fan_out)
                w_var = float(np.var(W))

                # Rough linear predictions for unit-variance inputs.
                pred_preact_var = float(fan_in) * w_var
                pred_backprop_var = float(fan_out) * w_var

                layer_analysis.append(
                    {
                        "layer": i,
                        "fan_in": fan_in,
                        "fan_out": fan_out,
                        "weight_variance": w_var,
                        "predicted_activation_var": pred_preact_var,
                        "predicted_gradient_var": pred_backprop_var,
                    }
                )
            analysis[name] = layer_analysis
        return analysis


def plot_activation_statistics(results: Dict, layer_idx: int = 0):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = list(results.keys())
    pre_means = [results[m][layer_idx]["pre_activation"]["mean"] for m in methods]
    pre_stds = [results[m][layer_idx]["pre_activation"]["std"] for m in methods]
    post_means = [results[m][layer_idx]["post_activation"]["mean"] for m in methods]
    post_stds = [results[m][layer_idx]["post_activation"]["std"] for m in methods]

    axes[0, 0].bar(methods, pre_means)
    axes[0, 0].set_title("Pre-activation Mean")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(methods, pre_stds)
    axes[0, 1].set_title("Pre-activation Std")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(methods, post_means)
    axes[1, 0].set_title("Post-activation Mean")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].bar(methods, post_stds)
    axes[1, 1].set_title("Post-activation Std")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_gradient_flow(results: Dict):
    plt.figure(figsize=(12, 6))
    for method, gradient_data in results.items():
        if gradient_data:
            layers = list(range(len(gradient_data)))
            plt.semilogy(layers, gradient_data, "o-", label=method)
    plt.xlabel("Layer")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Flow Across Layers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_training_curves(results: Dict):
    plt.figure(figsize=(15, 5))
    for i, (method, data) in enumerate(results.items()):
        plt.subplot(1, 3, 1)
        plt.plot(data["train_loss"], label=method)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 3, 2)
        plt.plot(data["test_loss"], label=method)
        plt.title("Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 3, 3)
        plt.plot(data["grad_norm"], label=method)
        plt.title("Gradient Norm")
        plt.xlabel("Epoch")
        plt.ylabel("Norm")

    for j in [1, 2, 3]:
        plt.subplot(1, 3, j)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def exercise_1_initialization_implementations():
    print("=== Exercise 1: Initialization Implementations ===")
    inits = [
        ZeroInitializer(),
        RandomNormalInitializer(0.1),
        XavierUniformInitializer(),
        XavierNormalInitializer(),
        HeUniformInitializer(),
        HeNormalInitializer(),
        LeCunInitializer(),
        OrthogonalInitializer(),
        VarianceScalingInitializer(scale=1.0, mode="fan_avg", distribution="uniform"),
    ]
    for init in inits:
        W = init.initialize_weights(10, 5)
        print(init.name, "shape", W.shape, "var", round(float(np.var(W)), 6))


def exercise_2_activation_analysis():
    print("=== Exercise 2: Activation Analysis ===")
    analyzer = InitializationAnalyzer()
    layer_sizes = [64, 64, 64]
    tanh = np.tanh
    results = analyzer.analyze_activation_statistics(layer_sizes, tanh, n_samples=512)
    plot_activation_statistics(results, layer_idx=0)


def exercise_3_gradient_flow_study():
    print("=== Exercise 3: Gradient Flow Study ===")
    analyzer = InitializationAnalyzer()
    layer_sizes = [32, 32, 32, 32]
    tanh = np.tanh
    dtanh = lambda z: 1.0 - np.tanh(z) ** 2
    results = analyzer.analyze_gradient_flow(layer_sizes, tanh, dtanh, n_samples=128)
    plot_gradient_flow(results)


def exercise_4_training_dynamics():
    print("=== Exercise 4: Training Dynamics ===")
    analyzer = InitializationAnalyzer()
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(256, 10))
    y_train = X_train[:, :1] - X_train[:, 1:2]
    X_test = rng.normal(size=(128, 10))
    y_test = X_test[:, :1] - X_test[:, 1:2]
    results = analyzer.compare_training_dynamics(X_train, y_train, X_test, y_test, [10, 32, 1], n_epochs=50)
    plot_training_curves(results)


def exercise_5_theoretical_validation():
    print("=== Exercise 5: Theoretical Validation ===")
    analyzer = InitializationAnalyzer()
    analysis = analyzer.theoretical_analysis([128, 64, 32])
    for name, layers in analysis.items():
        print(name, "layer0 weight_var", round(float(layers[0]["weight_variance"]), 6))


def exercise_6_practical_recommendations():
    print("=== Exercise 6: Practical Recommendations ===")
    print("- Tanh/Sigmoid: Xavier/LeCun")
    print("- ReLU: He")
    print("- Recurrent stacks: Orthogonal can help stability")


if __name__ == "__main__":
    start = time.time()
    exercise_1_initialization_implementations()
    exercise_2_activation_analysis()
    exercise_3_gradient_flow_study()
    exercise_4_training_dynamics()
    exercise_5_theoretical_validation()
    exercise_6_practical_recommendations()
    print("Done in", round(time.time() - start, 2), "s")

