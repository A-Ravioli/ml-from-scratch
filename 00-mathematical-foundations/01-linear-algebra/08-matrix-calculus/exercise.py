"""
Module 8 exercises: gradients, Hessians, and backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute a numerical gradient with central differences."""
    x = np.asarray(x, dtype=float)
    gradient = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        shift = np.zeros_like(x)
        shift[index] = epsilon
        gradient[index] = (f(x + shift) - f(x - shift)) / (2.0 * epsilon)
    return gradient


def least_squares_loss(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Compute ||Ax - b||^2."""
    residual = np.asarray(A, dtype=float) @ np.asarray(x, dtype=float) - np.asarray(b, dtype=float)
    return float(np.dot(residual, residual))


def least_squares_gradient(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Analytic gradient of ||Ax - b||^2 with respect to x."""
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)
    return 2.0 * A.T @ (A @ x - b)


def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Compute a numerical Hessian using central differences."""
    x = np.asarray(x, dtype=float)
    n = x.size
    hessian = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = epsilon
            ej[j] = epsilon
            hessian[i, j] = (
                f(x + ei + ej)
                - f(x + ei - ej)
                - f(x - ei + ej)
                + f(x - ei - ej)
            ) / (4.0 * epsilon**2)
    return hessian


def classify_hessian(H: np.ndarray, tolerance: float = 1e-8) -> str:
    """Classify a Hessian from its eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(np.asarray(H, dtype=float))
    if np.all(eigenvalues > tolerance):
        return "positive_definite"
    if np.all(eigenvalues >= -tolerance):
        return "positive_semidefinite"
    if np.all(eigenvalues < -tolerance):
        return "negative_definite"
    return "indefinite"


def quadratic_curvature_surface(
    Q: np.ndarray,
    grid_limits: Tuple[float, float] = (-2.0, 2.0),
    samples: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the quadratic loss x^T Q x on a 2D grid."""
    Q = np.asarray(Q, dtype=float)
    xs = np.linspace(grid_limits[0], grid_limits[1], samples)
    ys = np.linspace(grid_limits[0], grid_limits[1], samples)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(samples):
        for j in range(samples):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = x.T @ Q @ x
    return X, Y, Z


@dataclass
class Value:
    """Tiny scalar autograd engine in the style of micrograd."""

    data: float
    children: Tuple["Value", ...] = ()
    op: str = ""
    grad: float = 0.0
    _backward: Callable[[], None] = field(default=lambda: None, repr=False)

    def __post_init__(self) -> None:
        self.data = float(self.data)

    def __add__(self, other: float | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-other)

    def __rsub__(self, other: float | "Value") -> "Value":
        return other + (-self)

    def __mul__(self, other: float | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def tanh(self) -> "Value":
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: List[Value] = []
        visited: Set[int] = set()

        def build(node: Value) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node.children:
                build(child)
            topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def two_layer_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: float,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Forward pass for a tiny 2-layer tanh network with scalar output."""
    x = np.asarray(x, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    W2 = np.asarray(W2, dtype=float)

    z1 = W1 @ x + b1
    h = np.tanh(z1)
    y_hat = float(W2 @ h + b2)
    cache = {"x": x, "z1": z1, "h": h, "W2": W2}
    return y_hat, cache


def two_layer_gradients(
    x: np.ndarray,
    target: float,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: float,
) -> Dict[str, np.ndarray | float]:
    """Manual backpropagation for a tiny 2-layer network."""
    y_hat, cache = two_layer_forward(x, W1, b1, W2, b2)
    error = y_hat - target
    dL_dy = error
    dh = dL_dy * cache["W2"]
    dz1 = dh * (1.0 - np.tanh(cache["z1"]) ** 2)

    return {
        "loss": 0.5 * error**2,
        "dW2": dL_dy * cache["h"],
        "db2": dL_dy,
        "dW1": np.outer(dz1, cache["x"]),
        "db1": dz1,
    }


def finite_difference_parameter_gradient(
    x: np.ndarray,
    target: float,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: float,
    parameter_name: str,
    epsilon: float = 1e-6,
) -> np.ndarray | float:
    """Estimate a parameter gradient for the 2-layer network with finite differences."""
    params = {
        "W1": np.asarray(W1, dtype=float).copy(),
        "b1": np.asarray(b1, dtype=float).copy(),
        "W2": np.asarray(W2, dtype=float).copy(),
        "b2": float(b2),
    }

    def loss_from_params() -> float:
        y_hat, _ = two_layer_forward(x, params["W1"], params["b1"], params["W2"], params["b2"])
        return 0.5 * (y_hat - target) ** 2

    parameter = params[parameter_name]
    if np.isscalar(parameter):
        params[parameter_name] = parameter + epsilon
        plus = loss_from_params()
        params[parameter_name] = parameter - epsilon
        minus = loss_from_params()
        return (plus - minus) / (2.0 * epsilon)

    gradient = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        parameter[index] = original + epsilon
        plus = loss_from_params()
        parameter[index] = original - epsilon
        minus = loss_from_params()
        gradient[index] = (plus - minus) / (2.0 * epsilon)
        parameter[index] = original
    return gradient


if __name__ == "__main__":
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 0.0])
    x = np.array([0.5, -1.0])
    print("Analytic gradient:", least_squares_gradient(A, b, x))
