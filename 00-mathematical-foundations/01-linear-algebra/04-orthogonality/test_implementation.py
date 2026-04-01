"""
Tests for Module 4: Orthogonality.
"""

import numpy as np

from exercise import (
    attention_scores,
    compose_rotations,
    gram_schmidt,
    plot_projection_2d,
    preserves_norm,
    project_onto_subspace,
    project_onto_vector,
    projection_matrix,
    qr_condition_numbers,
    qr_via_gram_schmidt,
    rotation_matrix,
)


def test_vector_projection():
    v = np.array([2.0, 2.0])
    u = np.array([1.0, 0.0])
    projected = project_onto_vector(v, u)
    assert np.allclose(projected, np.array([2.0, 0.0]))


def test_subspace_projection_matrix_properties():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    P = projection_matrix(A)
    assert np.allclose(P @ P, P)
    assert np.allclose(P, P.T)

    v = np.array([1.0, 2.0, 3.0])
    projected = project_onto_subspace(v, A)
    assert np.allclose(projected, np.array([1.0, 2.0, 0.0]))


def test_projection_plot_returns_figure():
    fig, ax = plot_projection_2d(np.array([2.0, 1.0]), np.array([1.0, 0.0]))
    assert fig is not None
    assert ax is not None


def test_gram_schmidt_returns_orthonormal_columns():
    vectors = [np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])]
    Q = gram_schmidt(vectors)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)


def test_qr_matches_numpy():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    Q, R = qr_via_gram_schmidt(A)
    assert np.allclose(Q @ R, A)
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-8)


def test_q_condition_number_is_one():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(4, 4))
    cond_A, cond_Q = qr_condition_numbers(A)
    assert cond_A >= 1.0
    assert np.isclose(cond_Q, 1.0, atol=1e-8)


def test_rotations_preserve_norm_and_compose():
    theta_1 = np.pi / 6
    theta_2 = np.pi / 4
    combined = compose_rotations(theta_1, theta_2)
    expected = rotation_matrix(theta_1 + theta_2)
    assert np.allclose(combined, expected)

    v = np.array([1.0, -2.0])
    assert preserves_norm(rotation_matrix(theta_1), v)


def test_attention_scores_are_dot_products():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    keys = np.array([[1.0, 1.0], [1.0, -1.0]])
    scores = attention_scores(queries, keys)
    assert np.allclose(scores, np.array([[1.0, 1.0], [1.0, -1.0]]))
