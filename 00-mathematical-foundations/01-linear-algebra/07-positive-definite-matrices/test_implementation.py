"""
Tests for Module 7: Positive Definite Matrices.
"""

import numpy as np

from exercise import (
    benchmark_cholesky_vs_lu,
    cholesky_pd_test,
    cholesky_solve,
    eigenvalue_pd_test,
    make_positive_definite,
    quadratic_form,
    quadratic_surface,
    sample_multivariate_gaussian,
)


def test_generated_matrix_is_positive_definite():
    A = make_positive_definite(4, seed=2)
    assert eigenvalue_pd_test(A)
    assert cholesky_pd_test(A)


def test_quadratic_form_is_positive_for_nonzero_vector():
    A = make_positive_definite(3, seed=3)
    x = np.array([1.0, -2.0, 0.5])
    assert quadratic_form(A, x) > 0.0


def test_quadratic_surface_shape():
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    X, Y, Z = quadratic_surface(A, samples=10)
    assert X.shape == Y.shape == Z.shape == (10, 10)
    assert np.all(Z >= 0.0)


def test_cholesky_solve_matches_numpy():
    A = make_positive_definite(5, seed=4)
    b = np.arange(5, dtype=float)
    assert np.allclose(cholesky_solve(A, b), np.linalg.solve(A, b))


def test_gaussian_sampling_matches_moments_approximately():
    mean = np.array([1.0, -1.0])
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    samples = sample_multivariate_gaussian(mean, covariance, n_samples=5000, seed=0)
    empirical_mean = samples.mean(axis=0)
    empirical_covariance = np.cov(samples.T)
    assert np.allclose(empirical_mean, mean, atol=0.1)
    assert np.allclose(empirical_covariance, covariance, atol=0.15)


def test_benchmark_returns_positive_numbers():
    A = make_positive_definite(6, seed=5)
    b = np.arange(6, dtype=float)
    stats = benchmark_cholesky_vs_lu(A, b, repeats=2)
    assert stats["cholesky_seconds"] > 0.0
    assert stats["lu_seconds"] > 0.0
