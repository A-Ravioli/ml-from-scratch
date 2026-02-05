"""
Neural Network Theory - Reference Solutions

This file mirrors `exercise.py` and provides completed implementations for all
exercise stubs. It focuses on clarity and numerical correctness over speed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class ActivationFunction(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute activation function."""
        ...

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function (w.r.t. its input)."""
        ...


class Sigmoid(ActivationFunction):
    """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)
        return out

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1.0 - s)


class Tanh(ActivationFunction):
    """Hyperbolic tangent."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return 1.0 - y * y


class ReLU(ActivationFunction):
    """Rectified Linear Unit: max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return (x > 0).astype(x.dtype)


class LeakyReLU(ActivationFunction):
    """Leaky ReLU: max(alpha*x, x)."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = float(alpha)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0, 1.0, self.alpha).astype(x.dtype)


class Swish(ActivationFunction):
    """Swish: x * sigmoid(x)."""

    def __init__(self):
        self.sigmoid = Sigmoid()

    def forward(self, x: np.ndarray) -> np.ndarray:
        s = self.sigmoid.forward(x)
        return np.asarray(x) * s

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.sigmoid.forward(x)
        ds = s * (1.0 - s)
        return s + np.asarray(x) * ds


class NeuralNetwork:
    """
    Feedforward neural network from scratch.

    Parameters are stored as:
    - weights[i]: (in_dim, out_dim)
    - biases[i]:  (out_dim,)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[ActivationFunction],
        weight_init: str = "xavier",
        bias_init: str = "zeros",
    ):
        self.layer_sizes = list(layer_sizes)
        self.activations = list(activations)
        self.n_layers = len(layer_sizes) - 1
        if len(self.activations) != self.n_layers:
            raise ValueError("activations must have length len(layer_sizes)-1")

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._initialize_parameters(weight_init, bias_init)

        self.z_values: List[np.ndarray] = []
        self.a_values: List[np.ndarray] = []

    def _initialize_parameters(self, weight_init: str, bias_init: str):
        rng = np.random.default_rng(0)

        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            fan_in = int(self.layer_sizes[i])
            fan_out = int(self.layer_sizes[i + 1])

            scheme = weight_init.lower()
            if scheme == "xavier":
                std = np.sqrt(2.0 / (fan_in + fan_out))
                W = rng.normal(0.0, std, size=(fan_in, fan_out))
            elif scheme == "he":
                std = np.sqrt(2.0 / fan_in)
                W = rng.normal(0.0, std, size=(fan_in, fan_out))
            elif scheme == "normal":
                W = rng.normal(0.0, 0.01, size=(fan_in, fan_out))
            else:
                raise ValueError(f"Unknown weight_init: {weight_init!r}")

            b_scheme = bias_init.lower()
            if b_scheme == "zeros":
                b = np.zeros((fan_out,), dtype=float)
            elif b_scheme == "random":
                b = rng.normal(0.0, 0.01, size=(fan_out,))
            else:
                raise ValueError(f"Unknown bias_init: {bias_init!r}")

            self.weights.append(W.astype(float))
            self.biases.append(b.astype(float))

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self.z_values = []
        self.a_values = [X]

        a = X
        for i in range(self.n_layers):
            W = self.weights[i]
            b = self.biases[i]
            z = a @ W + b
            a = self.activations[i].forward(z)
            self.z_values.append(z)
            self.a_values.append(a)
        return a

    def backward(
        self, X: np.ndarray, y: np.ndarray, output: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation for MSE loss used by `train_step`:
        L = 0.5 * mean((output - y)^2)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        output = np.asarray(output, dtype=float)

        batch_size = int(X.shape[0])
        if batch_size == 0:
            raise ValueError("Empty batch")

        weight_grads = [np.zeros_like(w) for w in self.weights]
        bias_grads = [np.zeros_like(b) for b in self.biases]

        # dL/dout
        d_out = (output - y) / batch_size
        # delta at output layer: dL/dz_L = dL/dout * σ'(z_L)
        delta = d_out * self.activations[-1].derivative(self.z_values[-1])

        for i in range(self.n_layers - 1, -1, -1):
            a_prev = self.a_values[i]
            weight_grads[i] = a_prev.T @ delta
            bias_grads[i] = np.sum(delta, axis=0)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activations[i - 1].derivative(self.z_values[i - 1])

        return weight_grads, bias_grads

    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        output = self.forward(X)
        loss = float(0.5 * np.mean((output - y) ** 2))
        w_grads, b_grads = self.backward(X, y, output)

        for i in range(self.n_layers):
            self.weights[i] -= learning_rate * w_grads[i]
            self.biases[i] -= learning_rate * b_grads[i]

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


class UniversalApproximationExperiment:
    """Experiments to illustrate universal approximation (kept lightweight)."""

    def __init__(self):
        self.target_functions = {
            "polynomial": lambda x: x**3 - 2 * x**2 + x,
            "trigonometric": lambda x: np.sin(3 * x) + 0.5 * np.cos(7 * x),
            "step_function": lambda x: np.where(x < 0, -1, np.where(x < 0.5, 0, 1)),
            "discontinuous": lambda x: np.where(np.abs(x) < 0.3, 1, 0),
            "high_frequency": lambda x: np.sin(20 * x) * np.exp(-x**2),
        }

    def generate_data(self, func_name: str, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        f = self.target_functions[func_name]
        x = rng.uniform(-1.0, 1.0, size=(n_samples,))
        y = f(x)
        y = y + 0.01 * rng.normal(size=y.shape)
        return x.reshape(-1, 1), y.reshape(-1, 1)

    def test_approximation_capacity(self, func_name: str, hidden_sizes: List[int], n_epochs: int = 1000) -> Dict:
        results: Dict[int, Dict[str, float]] = {}
        X_train, y_train = self.generate_data(func_name, 256)
        X_test, y_test = self.generate_data(func_name, 128)

        # Keep this as a demo: cap runtime even if someone calls with huge n_epochs.
        n_epochs = int(min(max(n_epochs, 1), 300))

        for hidden_size in hidden_sizes:
            net = NeuralNetwork([1, int(hidden_size), 1], [Tanh(), Tanh()], weight_init="xavier", bias_init="zeros")
            lr = 0.05
            for _ in range(n_epochs):
                net.train_step(X_train, y_train, lr)
            preds = net.predict(X_test)
            mse = float(np.mean((preds - y_test) ** 2))
            results[int(hidden_size)] = {"mse": mse}
        return results

    def analyze_depth_vs_width(self, func_name: str) -> Dict:
        X_train, y_train = self.generate_data(func_name, 256)
        X_test, y_test = self.generate_data(func_name, 128)

        configs = {
            "shallow_wide": ([1, 100, 1], [Tanh(), Tanh()]),
            "medium": ([1, 50, 50, 1], [Tanh(), Tanh(), Tanh()]),
            "deep_narrow": ([1, 25, 25, 25, 25, 1], [Tanh(), Tanh(), Tanh(), Tanh(), Tanh()]),
        }

        out: Dict[str, float] = {}
        for name, (sizes, acts) in configs.items():
            net = NeuralNetwork(sizes, acts)
            for _ in range(200):
                net.train_step(X_train, y_train, 0.03)
            preds = net.predict(X_test)
            out[name] = float(np.mean((preds - y_test) ** 2))
        return out


class ExpressionCapacityAnalysis:
    """Lightweight expressivity utilities."""

    def count_linear_regions(self, network: NeuralNetwork, input_bounds: Tuple[float, float], resolution: int = 1000) -> int:
        lo, hi = input_bounds
        xs = np.linspace(lo, hi, int(resolution)).reshape(-1, 1)
        ys = network.predict(xs).reshape(-1)
        # Approximate piecewise linear region count by sign changes in discrete second derivative.
        d2 = np.diff(ys, n=2)
        return int(np.sum(np.abs(np.diff(np.sign(d2))) > 0)) + 1

    def measure_function_diversity(self, networks: List[NeuralNetwork], input_grid: np.ndarray) -> float:
        preds = np.stack([net.predict(input_grid).reshape(-1) for net in networks], axis=0)
        return float(np.mean(np.var(preds, axis=0)))

    def lottery_ticket_experiment(self, seed: int = 0) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(64, 2))
        y = (X[:, :1] - X[:, 1:2])
        net = NeuralNetwork([2, 16, 1], [Tanh(), Tanh()])
        for _ in range(50):
            net.train_step(X, y, 0.05)
        base_loss = float(0.5 * np.mean((net.predict(X) - y) ** 2))

        W0 = net.weights[0]
        thresh = np.quantile(np.abs(W0), 0.5)
        mask = (np.abs(W0) >= thresh).astype(float)
        net.weights[0] = W0 * mask
        for _ in range(30):
            net.train_step(X, y, 0.05)
        pruned_loss = float(0.5 * np.mean((net.predict(X) - y) ** 2))
        return {"base_loss": base_loss, "pruned_loss": pruned_loss}


class GradientFlowAnalysis:
    """Lightweight gradient flow checks for this from-scratch network."""

    def analyze_gradient_flow(self, network: NeuralNetwork, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        out = network.forward(X)
        w_grads, _ = network.backward(X, y, out)
        norms = [float(np.linalg.norm(g)) for g in w_grads]
        return {"min_grad_norm": float(np.min(norms)), "max_grad_norm": float(np.max(norms)), "mean_grad_norm": float(np.mean(norms))}

    def study_activation_saturation(self, activation: ActivationFunction, x: np.ndarray) -> Dict[str, float]:
        y = activation.forward(x)
        dy = activation.derivative(x)
        return {
            "output_mean": float(np.mean(y)),
            "output_std": float(np.std(y)),
            "derivative_mean": float(np.mean(dy)),
            "derivative_min": float(np.min(dy)),
        }

    def test_expressivity(self) -> Dict[str, float]:
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(128, 1))
        y = np.sin(5 * X)
        net = NeuralNetwork([1, 64, 1], [Tanh(), Tanh()])
        for _ in range(200):
            net.train_step(X, y, 0.05)
        mse = float(np.mean((net.predict(X) - y) ** 2))
        return {"mse": mse}


def plot_activation_functions(activations: Dict[str, ActivationFunction], x_range: Tuple[float, float] = (-5, 5)):
    x = np.linspace(x_range[0], x_range[1], 1000)
    fig, axes = plt.subplots(2, len(activations), figsize=(4 * len(activations), 6))
    if len(activations) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, (name, activation) in enumerate(activations.items()):
        y = activation.forward(x)
        axes[0, i].plot(x, y)
        axes[0, i].set_title(f"{name}")
        axes[0, i].grid(True)

        dy = activation.derivative(x)
        axes[1, i].plot(x, dy)
        axes[1, i].set_title(f"{name} Derivative")
        axes[1, i].grid(True)

    plt.tight_layout()
    plt.show()


def exercise_1_activation_functions():
    print("=== Exercise 1: Activation Functions ===")
    activations = {
        "Sigmoid": Sigmoid(),
        "Tanh": Tanh(),
        "ReLU": ReLU(),
        "LeakyReLU": LeakyReLU(),
        "Swish": Swish(),
    }
    plot_activation_functions(activations)


def exercise_2_network_implementation():
    print("=== Exercise 2: Neural Network Implementation ===")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 2))
    y = (X[:, :1] + X[:, 1:2])
    net = NeuralNetwork([2, 8, 1], [Tanh(), Tanh()])
    for _ in range(100):
        net.train_step(X, y, 0.05)
    print("train MSE:", float(np.mean((net.predict(X) - y) ** 2)))


def exercise_3_universal_approximation():
    print("=== Exercise 3: Universal Approximation ===")
    exp = UniversalApproximationExperiment()
    results = exp.test_approximation_capacity("trigonometric", hidden_sizes=[8, 32, 64], n_epochs=200)
    print(results)


def exercise_4_expressivity_analysis():
    print("=== Exercise 4: Expressivity Analysis ===")
    ana = ExpressionCapacityAnalysis()
    net = NeuralNetwork([1, 16, 16, 1], [ReLU(), ReLU(), ReLU()])
    regions = ana.count_linear_regions(net, (-1, 1), resolution=400)
    print("approx. linear regions:", regions)


def exercise_5_gradient_flow():
    print("=== Exercise 5: Gradient Flow Analysis ===")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 2))
    y = (X[:, :1] - X[:, 1:2])
    net = NeuralNetwork([2, 32, 32, 1], [Tanh(), Tanh(), Tanh()])
    gf = GradientFlowAnalysis().analyze_gradient_flow(net, X, y)
    print(gf)


def exercise_6_theoretical_properties():
    print("=== Exercise 6: Theoretical Properties ===")
    start = time.time()
    _ = GradientFlowAnalysis().test_expressivity()
    print("demo runtime (s):", round(time.time() - start, 3))


if __name__ == "__main__":
    exercise_1_activation_functions()
    exercise_2_network_implementation()
    exercise_3_universal_approximation()
    exercise_4_expressivity_analysis()
    exercise_5_gradient_flow()
    exercise_6_theoretical_properties()

    print("\nAll exercises completed!")
