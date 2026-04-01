"""
Tests for Module 3: Vector Spaces.
"""

import numpy as np

from exercise import (
    change_basis_coordinates,
    compute_four_subspaces,
    coordinates_in_basis,
    linear_independence_via_rank,
    plot_span_2d,
    principal_component_basis,
    rank_deficiency_report,
    vector_from_coordinates,
    verify_rank_nullity,
)


def test_linear_independence_via_rank():
    independent = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    dependent = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    assert linear_independence_via_rank(independent)
    assert not linear_independence_via_rank(dependent)


def test_span_plot_returns_sampled_points():
    fig, ax, points = plot_span_2d(np.array([1.0, 0.0]), np.array([0.0, 1.0]), samples=4)
    assert points.shape == (16, 2)
    assert fig is not None
    assert ax is not None


def test_rank_deficiency_report():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    report = rank_deficiency_report(A)
    assert report["rank_deficient"]
    assert report["rank"] == 1


def test_four_fundamental_subspaces():
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    subspaces = compute_four_subspaces(A)
    null_basis = subspaces["null_space"]
    assert null_basis.shape[1] == 2
    assert np.allclose(A @ null_basis, 0.0)


def test_rank_nullity_holds():
    A = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]])
    result = verify_rank_nullity(A)
    assert result["holds"]
    assert result["rank"] + result["nullity"] == result["dimension"]


def test_change_of_basis_round_trip():
    source_basis = np.array([[1.0, 0.0], [0.0, 1.0]])
    target_basis = np.array([[1.0, 1.0], [1.0, -1.0]])
    vector = np.array([2.0, 1.0])

    source_coords = coordinates_in_basis(vector, source_basis)
    target_coords = change_basis_coordinates(source_coords, source_basis, target_basis)
    recovered = vector_from_coordinates(target_coords, target_basis)
    assert np.allclose(recovered, vector)


def test_principal_component_basis_is_orthonormal():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3))
    basis = principal_component_basis(X, n_components=2)
    assert basis.shape == (2, 3)
    assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-8)
