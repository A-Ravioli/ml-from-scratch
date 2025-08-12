"""
Tensor Algebra Exercises
"""

from typing import List, Tuple
import numpy as np


def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """TODO: Compute A ⊗ B."""
    # TODO: Implement this
    pass


def khatri_rao_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """TODO: Column-wise Kronecker product. A,B with same number of columns."""
    # TODO: Implement this
    pass


def tensor_contract(A: np.ndarray, B: np.ndarray, axes: Tuple[List[int], List[int]]) -> np.ndarray:
    """TODO: Generalized contraction using np.tensordot semantics."""
    # TODO: Implement this
    pass


def matricize_mode_n(X: np.ndarray, mode: int) -> np.ndarray:
    """TODO: Unfold tensor along `mode` into a matrix (rows = size(mode))."""
    # TODO: Implement this
    pass


def mode_n_product(X: np.ndarray, U: np.ndarray, mode: int) -> np.ndarray:
    """TODO: Multiply tensor by matrix U along the given mode."""
    # TODO: Implement this
    pass


def rank1_approximation(X: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    TODO: Return rank-1 factors (a1, a2, ..., aN) approximating X ≈ a1 ∘ a2 ∘ ... ∘ aN.
    Hint: Use leading left singular vectors of mode-n unfoldings and normalize.
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Tensor Algebra Exercises")

