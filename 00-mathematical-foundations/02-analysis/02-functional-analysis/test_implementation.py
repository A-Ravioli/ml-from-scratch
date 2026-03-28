import numpy as np

from exercise import InnerProductSpace, LinearOperator, NormedSpace, l1_norm, l2_norm, linf_norm, matrix_operator, orthogonal_projection


def test_norms_and_inner_product_space():
    space = NormedSpace(l2_norm)
    points = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    assert space.verify_norm_axioms(points)
    inner = InnerProductSpace(lambda x, y: float(np.dot(x, y)))
    assert inner.verify_cauchy_schwarz(points)
    assert l1_norm(np.array([1.0, -2.0])) == 3.0
    assert linf_norm(np.array([1.0, -2.0])) == 2.0


def test_linear_operator_and_projection():
    A = np.array([[1.0, 2.0], [0.0, -1.0]])
    T = matrix_operator(A)
    op = LinearOperator(T, l2_norm, l2_norm)
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    assert op.is_linear(x, y, [2.0, -1.0])
    assert op.estimate_operator_norm([x, y, x + y]) > 0.0

    basis = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    projected = orthogonal_projection(basis, np.array([2.0, -1.0]))
    assert np.allclose(projected, np.array([2.0, -1.0]), atol=1e-6)
