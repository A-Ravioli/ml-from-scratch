"""
Tests for Module 8: Matrix Calculus.
"""

import numpy as np

from exercise import (
    Value,
    classify_hessian,
    finite_difference_parameter_gradient,
    least_squares_gradient,
    least_squares_loss,
    numerical_gradient,
    numerical_hessian,
    quadratic_curvature_surface,
    two_layer_gradients,
)


def test_numerical_gradient_matches_least_squares_gradient():
    A = np.array([[1.0, 2.0], [3.0, -1.0]])
    b = np.array([1.0, -2.0])
    x = np.array([0.5, 1.5])

    numerical = numerical_gradient(lambda z: least_squares_loss(A, b, z), x)
    analytic = least_squares_gradient(A, b, x)
    assert np.allclose(numerical, analytic, atol=1e-5)


def test_numerical_hessian_matches_quadratic():
    Q = np.array([[3.0, 1.0], [1.0, 2.0]])
    f = lambda z: float(z.T @ Q @ z)
    H = numerical_hessian(f, np.array([0.3, -0.7]))
    assert np.allclose(H, 2.0 * Q, atol=1e-4)


def test_hessian_classification():
    assert classify_hessian(np.array([[2.0, 0.0], [0.0, 1.0]])) == "positive_definite"
    assert classify_hessian(np.array([[1.0, 0.0], [0.0, -1.0]])) == "indefinite"


def test_quadratic_curvature_surface_shape():
    X, Y, Z = quadratic_curvature_surface(np.array([[2.0, 0.0], [0.0, 1.0]]), samples=12)
    assert X.shape == Y.shape == Z.shape == (12, 12)


def test_tiny_autograd_engine():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(0.5)
    y = (a * b + c).tanh()
    y.backward()

    manual = 1.0 - np.tanh(2.0 * -3.0 + 0.5) ** 2
    assert np.isclose(a.grad, manual * b.data)
    assert np.isclose(b.grad, manual * a.data)
    assert np.isclose(c.grad, manual)


def test_two_layer_gradients_match_finite_differences():
    x = np.array([0.2, -0.4])
    target = 0.3
    W1 = np.array([[0.5, -0.1], [0.2, 0.3]])
    b1 = np.array([0.0, 0.1])
    W2 = np.array([1.2, -0.7])
    b2 = -0.2

    grads = two_layer_gradients(x, target, W1, b1, W2, b2)
    fd_W1 = finite_difference_parameter_gradient(x, target, W1, b1, W2, b2, "W1")
    fd_b1 = finite_difference_parameter_gradient(x, target, W1, b1, W2, b2, "b1")
    fd_W2 = finite_difference_parameter_gradient(x, target, W1, b1, W2, b2, "W2")
    fd_b2 = finite_difference_parameter_gradient(x, target, W1, b1, W2, b2, "b2")

    assert np.allclose(grads["dW1"], fd_W1, atol=1e-5)
    assert np.allclose(grads["db1"], fd_b1, atol=1e-5)
    assert np.allclose(grads["dW2"], fd_W2, atol=1e-5)
    assert np.isclose(grads["db2"], fd_b2, atol=1e-5)
