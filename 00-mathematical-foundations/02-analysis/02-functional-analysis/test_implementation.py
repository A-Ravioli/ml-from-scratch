"""
Test suite for Functional Analysis implementations.
"""

import numpy as np
import pytest

from exercise import (
    NormedSpace, InnerProductSpace, LinearOperator,
    l1_norm, l2_norm, linf_norm, dot_inner,
    orthogonal_projection, matrix_operator
)


def random_points(n: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim) for _ in range(n)]


class TestNorms:
    def test_norm_axioms(self):
        points = random_points(10, 5)
        for norm in [l1_norm, l2_norm, linf_norm]:
            space = NormedSpace(norm)
            assert space.verify_norm_axioms(points)


class TestInnerProduct:
    def test_cauchy_schwarz_and_parallelogram(self):
        points = random_points(8, 4, seed=1)
        H = InnerProductSpace(dot_inner)
        assert H.verify_cauchy_schwarz(points)
        assert H.verify_parallelogram_law(points)

    def test_induced_norm(self):
        H = InnerProductSpace(dot_inner)
        x = np.array([3.0, 4.0])
        assert abs(H.norm(x) - 5.0) < 1e-10


class TestLinearOperator:
    def test_linearity_and_operator_norm(self):
        A = np.array([[3.0, 0.0], [0.0, 2.0]])
        T = matrix_operator(A)
        op = LinearOperator(T, l2_norm, l2_norm)

        x = np.array([1.0, -1.0])
        y = np.array([2.0, 3.0])
        assert op.is_linear(x, y, scalars=[-1.0, 0.5, 2.0])

        samples = [np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2*np.pi, 64)]
        est = op.estimate_operator_norm(samples)
        # True spectral norm is 3.0 for this diagonal matrix
        assert 2.5 < est <= 3.1

    def test_continuity_at_zero(self):
        A = np.array([[1.0, 2.0], [0.0, -1.0]])
        T = matrix_operator(A)
        op = LinearOperator(T, l2_norm, l2_norm)
        assert op.check_continuity_at_zero(deltas=[1e-1, 1e-2, 1e-3])


class TestProjection:
    def test_projection_onto_axis(self):
        e1 = np.array([1.0, 0.0])
        x = np.array([2.0, 3.0])
        p = orthogonal_projection([e1], x)
        assert np.allclose(p, np.array([2.0, 0.0]), atol=1e-8)

    def test_projection_onto_plane(self):
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        x = np.array([1.0, 2.0, 3.0])
        p = orthogonal_projection([e1, e2], x)
        assert np.allclose(p, np.array([1.0, 2.0, 0.0]), atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


