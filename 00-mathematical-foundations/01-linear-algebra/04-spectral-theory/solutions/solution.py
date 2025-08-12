"""
Spectral Theory Solutions - Reference Implementation
"""

from typing import List, Tuple
import numpy as np


def symmetric_eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(A)
    return vals, vecs


def diagonalize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eig(A)
    D = np.diag(vals)
    return vecs, D


def power_iteration(A: np.ndarray, max_iters: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    n = A.shape[0]
    x = np.random.randn(n)
    x /= np.linalg.norm(x)
    lam_old = 0.0
    for _ in range(max_iters):
        y = A @ x
        x = y / (np.linalg.norm(y) + 1e-15)
        lam = float(x @ (A @ x))
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return lam, x


def inverse_iteration(A: np.ndarray, mu: float, max_iters: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    n = A.shape[0]
    x = np.random.randn(n)
    x /= np.linalg.norm(x)
    I = np.eye(n)
    for _ in range(max_iters):
        y = np.linalg.solve(A - mu * I, x)
        x_new = y / (np.linalg.norm(y) + 1e-15)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    lam = float(x @ (A @ x) / (x @ x))
    return lam, x


def rayleigh_quotient(A: np.ndarray, x: np.ndarray) -> float:
    return float((x.T @ (A @ x)) / (x.T @ x))


def rayleigh_quotient_iteration(A: np.ndarray, x0: np.ndarray, max_iters: int = 50, tol: float = 1e-12) -> Tuple[float, np.ndarray]:
    x = x0 / (np.linalg.norm(x0) + 1e-15)
    n = A.shape[0]
    I = np.eye(n)
    mu = rayleigh_quotient(A, x)
    for _ in range(max_iters):
        y = np.linalg.solve(A - mu * I, x)
        x = y / (np.linalg.norm(y) + 1e-15)
        mu_new = rayleigh_quotient(A, x)
        if abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new
    return mu, x


def gershgorin_disks(A: np.ndarray) -> List[Tuple[complex, float]]:
    disks = []
    for i in range(A.shape[0]):
        center = A[i, i]
        radius = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        disks.append((center, float(radius)))
    return disks


def matrix_function_via_eigen(A: np.ndarray, func) -> np.ndarray:
    vals, vecs = np.linalg.eig(A)
    F = np.diag([func(v) for v in vals])
    return vecs @ F @ np.linalg.inv(vecs)


def graph_laplacian_spectrum(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    deg = np.diag(np.sum(adj, axis=1))
    L = deg - adj
    vals, vecs = np.linalg.eigh(L)
    return vals, vecs


