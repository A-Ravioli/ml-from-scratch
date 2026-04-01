"""
Module 10 capstones: PCA, regression, attention, and a tiny neural network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def pca_pipeline(X: np.ndarray, n_components: int) -> Dict[str, np.ndarray]:
    """End-to-end PCA using SVD."""
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0, keepdims=True)
    centered = X - mean
    U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    projected = centered @ components.T
    reconstructed = projected @ components + mean
    explained_variance = (singular_values**2) / max(len(X) - 1, 1)
    explained_variance_ratio = explained_variance[:n_components] / np.sum(explained_variance)
    return {
        "mean": mean,
        "components": components,
        "projected": projected,
        "reconstructed": reconstructed,
        "explained_variance_ratio": explained_variance_ratio,
    }


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Augment a design matrix with a bias column."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(X.shape[0]), X])


def linear_regression_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve least squares with the normal equations."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.solve(X.T @ X, X.T @ y)


def linear_regression_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve least squares with QR."""
    Q, R = np.linalg.qr(np.asarray(X, dtype=float))
    return np.linalg.solve(R, Q.T @ np.asarray(y, dtype=float))


def linear_regression_svd(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve least squares with the pseudoinverse."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.pinv(X) @ y


def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    steps: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve least squares with gradient descent."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.zeros(X.shape[1], dtype=float)
    losses = []

    for _ in range(steps):
        residual = X @ weights - y
        losses.append(0.5 * np.mean(residual**2))
        gradient = X.T @ residual / len(X)
        weights -= learning_rate * gradient
    return weights, np.asarray(losses)


def linear_regression_four_ways(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    steps: int = 2000,
) -> Dict[str, np.ndarray]:
    """Solve the same regression problem with four methods."""
    gd_weights, gd_losses = linear_regression_gradient_descent(X, y, learning_rate, steps)
    return {
        "normal_equations": linear_regression_normal_equations(X, y),
        "qr": linear_regression_qr(X, y),
        "svd": linear_regression_svd(X, y),
        "gradient_descent": gd_weights,
        "gradient_descent_losses": gd_losses,
    }


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention."""
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def multi_head_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, num_heads: int) -> np.ndarray:
    """Compute multi-head attention with block reshaping."""
    X = np.asarray(X, dtype=float)
    d_model = X.shape[1]
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")

    head_dim = d_model // num_heads
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    Q_heads = Q.reshape(X.shape[0], num_heads, head_dim).transpose(1, 0, 2)
    K_heads = K.reshape(X.shape[0], num_heads, head_dim).transpose(1, 0, 2)
    V_heads = V.reshape(X.shape[0], num_heads, head_dim).transpose(1, 0, 2)

    outputs = []
    for head in range(num_heads):
        head_output, _ = scaled_dot_product_attention(Q_heads[head], K_heads[head], V_heads[head])
        outputs.append(head_output)
    return np.concatenate(outputs, axis=-1)


@dataclass
class TwoLayerNetScratch:
    """A tiny 2-layer neural network trained without autograd."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(scale=0.3, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(scale=0.3, size=(self.hidden_dim, self.output_dim))
        self.b2 = np.zeros(self.output_dim)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        X = np.asarray(X, dtype=float)
        z1 = X @ self.W1 + self.b1
        h = np.tanh(z1)
        y_hat = h @ self.W2 + self.b2
        return y_hat, {"X": X, "z1": z1, "h": h}

    def loss_and_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        y_hat, cache = self.forward(X)
        error = y_hat - np.asarray(y, dtype=float)
        loss = 0.5 * np.mean(error**2)

        dY = error / len(X)
        dW2 = cache["h"].T @ dY
        db2 = np.sum(dY, axis=0)
        dH = dY @ self.W2.T
        dZ1 = dH * (1.0 - np.tanh(cache["z1"]) ** 2)
        dW1 = cache["X"].T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return float(loss), grads

    def step(self, grads: Dict[str, np.ndarray], learning_rate: float) -> None:
        self.W1 -= learning_rate * grads["W1"]
        self.b1 -= learning_rate * grads["b1"]
        self.W2 -= learning_rate * grads["W2"]
        self.b2 -= learning_rate * grads["b2"]

    def train(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, epochs: int = 2000) -> np.ndarray:
        losses = []
        for _ in range(epochs):
            loss, grads = self.loss_and_gradients(X, y)
            self.step(grads, learning_rate)
            losses.append(loss)
        return np.asarray(losses)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat, _ = self.forward(X)
        return y_hat


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.1
    result = linear_regression_four_ways(X, y)
    print("Normal equations weights:", result["normal_equations"])
