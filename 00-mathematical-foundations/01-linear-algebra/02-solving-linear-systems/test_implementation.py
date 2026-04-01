"""
Tests for Module 2: Solving Linear Systems.
"""

import numpy as np

from exercise import (
    benchmark_lu_reuse,
    classify_linear_system,
    gaussian_elimination_partial_pivot,
    perturb_rhs_and_solve,
    plot_2d_system,
    sample_plane_points,
    solve_many_rhs_with_lu,
    solve_with_numpy,
)


def test_classify_systems():
    A_unique = np.array([[2.0, 1.0], [1.0, 3.0]])
    b_unique = np.array([1.0, 2.0])
    assert classify_linear_system(A_unique, b_unique) == "unique"

    A_under = np.array([[1.0, 1.0], [2.0, 2.0]])
    b_under = np.array([1.0, 2.0])
    assert classify_linear_system(A_under, b_under) == "underdetermined"

    b_inconsistent = np.array([1.0, 3.0])
    assert classify_linear_system(A_under, b_inconsistent) == "inconsistent"


def test_numpy_and_gaussian_elimination_agree():
    A = np.array([[0.0, 2.0, 1.0], [1.0, -2.0, 3.0], [3.0, 1.0, -1.0]])
    b = np.array([4.0, 5.0, 2.0])
    expected = solve_with_numpy(A, b)
    actual = gaussian_elimination_partial_pivot(A, b)
    assert np.allclose(actual, expected)


def test_rhs_perturbation_changes_solution():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    delta_b = np.array([0.1, -0.2])
    result = perturb_rhs_and_solve(A, b, delta_b)
    assert not np.allclose(result["original_solution"], result["perturbed_solution"])


def test_plane_sampling_shape():
    points = sample_plane_points(np.array([1.0, -2.0, 1.0]), 3.0, samples=4)
    assert points.shape == (16, 3)
    assert np.allclose(points[:, 0] - 2.0 * points[:, 1] + points[:, 2], 3.0)


def test_lu_reuse_matches_repeated_solves():
    rng = np.random.default_rng(11)
    A = rng.normal(size=(5, 5))
    A += 5.0 * np.eye(5)
    B = rng.normal(size=(5, 3))

    lu_solution = solve_many_rhs_with_lu(A, B)
    repeated = np.column_stack([np.linalg.solve(A, B[:, idx]) for idx in range(B.shape[1])])
    assert np.allclose(lu_solution, repeated)


def test_lu_benchmark_returns_positive_numbers():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    A += 6.0 * np.eye(6)
    B = rng.normal(size=(6, 4))
    stats = benchmark_lu_reuse(A, B, repeats=2)
    assert stats["lu_seconds"] > 0.0
    assert stats["repeated_seconds"] > 0.0


def test_2d_system_plot_returns_figure():
    fig, ax = plot_2d_system(np.array([[1.0, 1.0], [1.0, -1.0]]), np.array([2.0, 0.0]))
    assert fig is not None
    assert ax is not None
