"""
Tensor Algebra Solutions - Reference Implementation
"""

from typing import List, Tuple
import numpy as np


def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)


def khatri_rao_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns for Khatri–Rao product.")
    cols = []
    for i in range(A.shape[1]):
        cols.append(np.kron(A[:, i], B[:, i]))
    return np.stack(cols, axis=1)


def tensor_contract(A: np.ndarray, B: np.ndarray, axes: Tuple[List[int], List[int]]) -> np.ndarray:
    return np.tensordot(A, B, axes=axes)


def matricize_mode_n(X: np.ndarray, mode: int) -> np.ndarray:
    # Move `mode` to front, then flatten the rest
    order = list(range(X.ndim))
    order[0], order[mode] = order[mode], order[0]
    X_perm = np.transpose(X, axes=order)
    return X_perm.reshape(X.shape[mode], -1)


def mode_n_product(X: np.ndarray, U: np.ndarray, mode: int) -> np.ndarray:
    # Matricize, multiply, then fold back
    Xn = matricize_mode_n(X, mode)
    Yn = U @ Xn
    new_shape = list(X.shape)
    new_shape[mode] = U.shape[0]
    # Inverse permutation
    order = list(range(X.ndim))
    order[0], order[mode] = order[mode], order[0]
    inv_order = np.argsort(order)
    Y_perm = Yn.reshape([new_shape[mode]] + [s for i, s in enumerate(X.shape) if i != mode])
    return np.transpose(Y_perm, axes=inv_order)


def rank1_approximation(X: np.ndarray) -> Tuple[np.ndarray, ...]:
    factors = []
    for mode in range(X.ndim):
        Xn = matricize_mode_n(X, mode)
        U, s, Vt = np.linalg.svd(Xn, full_matrices=False)
        u1 = U[:, 0]
        factors.append(u1)
    # Scale factors by overall norm
    lam = np.linalg.norm(X.ravel())
    # Normalize to unit vectors and absorb scale in first factor
    norms = [np.linalg.norm(a) + 1e-12 for a in factors]
    factors = [a / n for a, n in zip(factors, norms)]
    factors[0] = factors[0] * lam
    return tuple(factors)


