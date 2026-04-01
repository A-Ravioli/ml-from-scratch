"""
Tests for Module 10: Capstone Projects.
"""

import numpy as np

from exercise import (
    TwoLayerNetScratch,
    add_bias_column,
    linear_regression_four_ways,
    multi_head_attention,
    pca_pipeline,
    scaled_dot_product_attention,
)


def test_pca_pipeline_shapes_and_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    result = pca_pipeline(X, n_components=2)
    assert result["components"].shape == (2, 4)
    assert result["projected"].shape == (50, 2)
    assert result["reconstructed"].shape == X.shape
    assert np.sum(result["explained_variance_ratio"]) <= 1.0 + 1e-8


def test_linear_regression_four_methods_agree():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    X = add_bias_column(X)
    true_w = np.array([0.5, 1.0, -2.0, 0.75])
    y = X @ true_w
    result = linear_regression_four_ways(X, y, learning_rate=0.1, steps=3000)

    assert np.allclose(result["normal_equations"], true_w, atol=1e-6)
    assert np.allclose(result["qr"], true_w, atol=1e-6)
    assert np.allclose(result["svd"], true_w, atol=1e-6)
    assert np.allclose(result["gradient_descent"], true_w, atol=1e-2)


def test_attention_weights_are_row_stochastic():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 1.0], [1.0, -1.0]])
    V = np.array([[2.0, 0.0], [0.0, 3.0]])
    output, weights = scaled_dot_product_attention(Q, K, V)

    assert output.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_multi_head_attention_output_shape():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 4))
    W_q = np.eye(4)
    W_k = np.eye(4)
    W_v = np.eye(4)
    output = multi_head_attention(X, W_q, W_k, W_v, num_heads=2)
    assert output.shape == X.shape


def test_two_layer_net_training_reduces_loss():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(128, 2))
    y = (X[:, :1] - 0.5 * X[:, 1:2])

    model = TwoLayerNetScratch(input_dim=2, hidden_dim=8, output_dim=1, seed=0)
    losses = model.train(X, y, learning_rate=0.05, epochs=500)
    assert losses[-1] < losses[0]
