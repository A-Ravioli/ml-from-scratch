"""
Functional Analysis Solutions - Reference Implementation
"""

from typing import Callable, List, Tuple, Optional
import numpy as np


class NormedSpace:
    def __init__(self, norm_func: Callable[[np.ndarray], float]):
        self.norm = norm_func

    def verify_norm_axioms(self, points: List[np.ndarray], tolerance: float = 1e-10) -> bool:
        # 1) Non-negativity and identity
        for x in points:
            n = self.norm(x)
            if n < -tolerance:
                return False
            if np.allclose(x, 0, atol=tolerance):
                if abs(n) > tolerance:
                    return False
            else:
                if n <= tolerance:
                    return False

        # 2) Absolute homogeneity
        for x in points:
            for a in [-2.0, -0.5, 0.0, 0.3, 1.7]:
                left = self.norm(a * x)
                right = abs(a) * self.norm(x)
                if abs(left - right) > 1e-8:
                    return False

        # 3) Triangle inequality
        for x in points:
            for y in points:
                if self.norm(x + y) > self.norm(x) + self.norm(y) + 1e-8:
                    return False

        return True


class InnerProductSpace(NormedSpace):
    def __init__(self, inner_product: Callable[[np.ndarray, np.ndarray], float]):
        self.inner = inner_product
        super().__init__(lambda x: float(np.sqrt(max(self.inner(x, x), 0.0))))

    def verify_cauchy_schwarz(self, points: List[np.ndarray], tolerance: float = 1e-8) -> bool:
        for x in points:
            for y in points:
                lhs = abs(self.inner(x, y))
                rhs = self.norm(x) * self.norm(y)
                if lhs - rhs > tolerance:
                    return False
        return True

    def verify_parallelogram_law(self, points: List[np.ndarray], tolerance: float = 1e-8) -> bool:
        for x in points:
            for y in points:
                lhs = self.norm(x + y) ** 2 + self.norm(x - y) ** 2
                rhs = 2 * self.norm(x) ** 2 + 2 * self.norm(y) ** 2
                if abs(lhs - rhs) > 1e-6:
                    return False
        return True


class LinearOperator:
    def __init__(self, T: Callable[[np.ndarray], np.ndarray], domain_norm: Callable[[np.ndarray], float], codomain_norm: Callable[[np.ndarray], float]):
        self.T = T
        self.domain_norm = domain_norm
        self.codomain_norm = codomain_norm

    def is_linear(self, x: np.ndarray, y: np.ndarray, scalars: List[float]) -> bool:
        for a in scalars:
            for b in scalars:
                left = self.T(a * x + b * y)
                right = a * self.T(x) + b * self.T(y)
                if not np.allclose(left, right, atol=1e-8):
                    return False
        return True

    def estimate_operator_norm(self, samples: List[np.ndarray]) -> float:
        best = 0.0
        for x in samples:
            n = self.domain_norm(x)
            if n > 1e-12:
                val = self.codomain_norm(self.T(x)) / n
                best = max(best, float(val))
        return best

    def check_continuity_at_zero(self, deltas: List[float]) -> bool:
        # For linear maps on finite-dimensional spaces, continuity at 0 is equivalent to boundedness.
        # Empirically: smaller inputs produce proportionally smaller outputs.
        for d in deltas:
            x = np.random.randn(5)
            x = (d / (np.linalg.norm(x) + 1e-12)) * x
            if self.codomain_norm(self.T(x)) > 10 * d:  # heuristic bound
                return False
        return True


def orthogonal_projection(basis: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    if len(basis) == 0:
        return np.zeros_like(x)
    # Orthonormalize basis via Gram-Schmidt
    B = np.stack(basis, axis=1)  # shape (n, k)
    # QR decomposition for stability
    Q, _ = np.linalg.qr(B)
    # Project: P = Q Q^T
    return Q @ (Q.T @ x)


def l1_norm(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x)))


def l2_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)))


def linf_norm(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def dot_inner(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y))


def matrix_operator(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def T(x: np.ndarray) -> np.ndarray:
        return A @ x
    return T


