"""
Matrix Decompositions Exercises: LU, QR, Cholesky, and Applications
"""

from typing import Tuple
import numpy as np


def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO: Compute LU decomposition with partial pivoting.

    Returns P, L, U such that P @ A = L @ U, L unit-lower-triangular.
    """
    # TODO: Implement this
    pass


def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TODO: Solve Ax=b using LU decomposition with pivoting."""
    # TODO: Implement this
    pass


def determinant_via_lu(A: np.ndarray) -> float:
    """TODO: Compute det(A) from LU factors (track permutation sign)."""
    # TODO: Implement this
    pass


def qr_decomposition(A: np.ndarray, method: str = "mgs") -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: Compute QR decomposition.
    - method="mgs": modified Gram–Schmidt
    - method="householder": you may call np.linalg.qr
    """
    # TODO: Implement this
    pass


def least_squares_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TODO: Solve min_x ||Ax-b|| via QR."""
    # TODO: Implement this
    pass


def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """TODO: Compute Cholesky factor L (A = L L^T) for SPD A."""
    # TODO: Implement this
    pass


def solve_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TODO: Solve Ax=b using Cholesky (forward/back substitution)."""
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Matrix Decompositions Exercises")

