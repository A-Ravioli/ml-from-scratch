"""
Functional Analysis Exercises for Machine Learning

Implement utilities for normed/inner product spaces, linear operators,
operator norms, and orthogonal projections relevant to ML.
"""

from typing import Callable, List, Tuple, Optional
import numpy as np


class NormedSpace:
    """
    Normed space with a provided norm function.
    """

    def __init__(self, norm_func: Callable[[np.ndarray], float]):
        self.norm = norm_func

    def verify_norm_axioms(self, points: List[np.ndarray], tolerance: float = 1e-10) -> bool:
        """
        TODO: Verify norm axioms on given sample points.
        1) Non-negativity and identity: ||x|| >= 0 and =0 iff x=0
        2) Absolute homogeneity: ||a x|| = |a| ||x||
        3) Triangle inequality: ||x + y|| <= ||x|| + ||y||
        """
        # TODO: Implement this
        pass

    def induced_metric(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance d(x,y) = ||x-y||."""
        return float(self.norm(x - y))


class InnerProductSpace(NormedSpace):
    """
    Inner product space over ℝ with provided inner product.
    The norm is induced by the inner product.
    """

    def __init__(self, inner_product: Callable[[np.ndarray, np.ndarray], float]):
        self.inner = inner_product
        super().__init__(lambda x: float(np.sqrt(max(self.inner(x, x), 0.0))))

    def verify_cauchy_schwarz(self, points: List[np.ndarray], tolerance: float = 1e-8) -> bool:
        """TODO: Verify |<x,y>| <= ||x|| ||y|| on sample points."""
        # TODO: Implement this
        pass

    def verify_parallelogram_law(self, points: List[np.ndarray], tolerance: float = 1e-8) -> bool:
        """TODO: Verify ||x+y||^2 + ||x-y||^2 = 2||x||^2 + 2||y||^2 approximately."""
        # TODO: Implement this
        pass


class LinearOperator:
    """Linear operator T: ℝ^n → ℝ^m represented as a function."""

    def __init__(self, T: Callable[[np.ndarray], np.ndarray], domain_norm: Callable[[np.ndarray], float], codomain_norm: Callable[[np.ndarray], float]):
        self.T = T
        self.domain_norm = domain_norm
        self.codomain_norm = codomain_norm

    def is_linear(self, x: np.ndarray, y: np.ndarray, scalars: List[float]) -> bool:
        """
        TODO: Check T(ax + by) = aT(x) + bT(y) for a few scalars.
        """
        # TODO: Implement this
        pass

    def estimate_operator_norm(self, samples: List[np.ndarray]) -> float:
        """
        TODO: Estimate ||T|| = sup_{x≠0} ||T x|| / ||x|| over given samples.
        """
        # TODO: Implement this
        pass

    def check_continuity_at_zero(self, deltas: List[float]) -> bool:
        """
        TODO: Empirically check continuity at 0: small ||x|| implies small ||T x||.
        """
        # TODO: Implement this
        pass


def orthogonal_projection(basis: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    """
    TODO: Orthogonal projection of x onto span(basis) in ℝ^n using the standard inner product.
    If basis is empty, return the zero vector of appropriate shape.
    """
    # TODO: Implement this
    pass


def l1_norm(x: np.ndarray) -> float:
    """TODO: L1 norm."""
    # TODO: Implement this
    pass


def l2_norm(x: np.ndarray) -> float:
    """TODO: L2 norm."""
    # TODO: Implement this
    pass


def linf_norm(x: np.ndarray) -> float:
    """TODO: L-infinity norm."""
    # TODO: Implement this
    pass


def dot_inner(x: np.ndarray, y: np.ndarray) -> float:
    """TODO: Standard inner product on ℝ^n."""
    # TODO: Implement this
    pass


def matrix_operator(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return T(x) = A x for a matrix A."""
    def T(x: np.ndarray) -> np.ndarray:
        return A @ x
    return T


if __name__ == "__main__":
    print("Functional Analysis Exercises for ML")

