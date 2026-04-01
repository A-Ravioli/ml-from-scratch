"""
Tests for Module 6: Singular Value Decomposition.
"""

import numpy as np

from exercise import (
    compress_grayscale_image,
    compute_svd,
    low_rank_approximation,
    pca_via_svd,
    pseudoinverse_via_svd,
    reconstruct_from_svd,
    reconstruction_error_curve,
    singular_value_spectrum,
    solve_overdetermined_system,
)


def test_svd_reconstructs_matrix():
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    U, s, Vt = compute_svd(A)
    reconstructed = reconstruct_from_svd(U, s, Vt)
    assert np.allclose(reconstructed, A)


def test_low_rank_approximation_has_requested_rank():
    A = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 1.0]])
    approx = low_rank_approximation(A, rank=1)
    assert np.linalg.matrix_rank(approx) == 1


def test_reconstruction_error_curve_decreases():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 4))
    errors = reconstruction_error_curve(A)
    assert np.all(np.diff(errors) <= 1e-8)


def test_image_compression_preserves_shape():
    image = np.arange(64, dtype=float).reshape(8, 8)
    compressed = compress_grayscale_image(image, rank=2)
    assert compressed.shape == image.shape


def test_singular_values_are_sorted():
    A = np.array([[3.0, 0.0], [0.0, 1.0]])
    s = singular_value_spectrum(A)
    assert np.all(np.diff(s) <= 0.0)


def test_pca_via_svd_outputs_orthonormal_components():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 3))
    result = pca_via_svd(X, n_components=2)
    components = result["components"]
    assert np.allclose(components @ components.T, np.eye(2), atol=1e-8)
    assert np.all(result["explained_variance_ratio"] >= 0.0)
    assert np.sum(result["explained_variance_ratio"]) <= 1.0 + 1e-8


def test_pseudoinverse_matches_numpy():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(5, 3))
    assert np.allclose(pseudoinverse_via_svd(A), np.linalg.pinv(A))


def test_overdetermined_solution_matches_lstsq():
    rng = np.random.default_rng(8)
    A = rng.normal(size=(8, 3))
    b = rng.normal(size=8)
    x = solve_overdetermined_system(A, b)
    expected, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert np.allclose(x, expected)
