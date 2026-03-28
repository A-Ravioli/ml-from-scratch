import numpy as np

from exercise import kronecker_product, matricize_mode_n, rank1_approximation, tensor_contract


def test_kronecker_product_shape():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    out = kronecker_product(A, B)
    assert out.shape == (4, 4)


def test_tensor_contract_matches_numpy():
    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)
    out = tensor_contract(A, B, axes=([1], [0]))
    assert np.allclose(out, A @ B)


def test_matricize_and_rank1_approximation():
    tensor = np.arange(24.0).reshape(2, 3, 4)
    matricized = matricize_mode_n(tensor, mode=1)
    assert matricized.shape == (3, 8)

    a = np.array([1.0, -2.0, 0.5])
    b = np.array([2.0, 1.0])
    c = np.array([0.5, -1.0, 3.0])
    rank1 = np.einsum('i,j,k->ijk', a, b, c)
    factors = rank1_approximation(rank1)
    recovered = np.einsum('i,j,k->ijk', *factors)
    assert np.allclose(np.abs(rank1 / np.linalg.norm(rank1)), np.abs(recovered / np.linalg.norm(recovered)), atol=1e-2)
