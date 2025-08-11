"""
Matrix Decompositions Solutions - Reference Implementation
"""

from typing import Tuple
import numpy as np


def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = A.copy().astype(float)
    n = A.shape[0]
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy()
    for k in range(n-1):
        # Pivot
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if U[pivot, k] == 0:
            continue
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        # Eliminate
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return P, L, U


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def back_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    P, L, U = lu_decomposition(A)
    bp = P @ b
    y = forward_substitution(L, bp)
    x = back_substitution(U, y)
    return x


def determinant_via_lu(A: np.ndarray) -> float:
    P, _, U = lu_decomposition(A)
    detU = float(np.prod(np.diag(U)))
    # Permutation sign is det(P)
    sign = 1 if np.linalg.det(P) > 0 else -1
    return sign * detU


def qr_decomposition(A: np.ndarray, method: str = "mgs") -> Tuple[np.ndarray, np.ndarray]:
    A = A.astype(float)
    m, n = A.shape
    if method == "householder":
        Q, R = np.linalg.qr(A)
        return Q, R
    # Modified Gram–Schmidt
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] < 1e-15:
            Q[:, i] = 0
        else:
            Q[:, i] = V[:, i] / R[i, i]
        for j in range(i+1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return Q, R


def least_squares_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    Q, R = qr_decomposition(A, method="householder")
    y = Q.T @ b
    # Solve upper-triangular Rx = y (least squares uses R[:n,:])
    n = R.shape[0]
    x = back_substitution(R[:n, :n], y[:n])
    return x


def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(A)


def solve_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    L = cholesky_decomposition(A)
    y = forward_substitution(L, b)
    x = back_substitution(L.T, y)
    return x


