"""
Constrained Optimization Exercises
"""

from typing import Tuple, Dict, Callable
import numpy as np


def solve_quadratic_program(P: np.ndarray, q: np.ndarray, A: np.ndarray = None, b: np.ndarray = None,
                            G: np.ndarray = None, h: np.ndarray = None) -> np.ndarray:
    """TODO: Solve simple QP via KKT (small scale) or fall back to cvxpy if available."""
    # TODO: Implement this
    pass


def augmented_lagrangian(f: Callable[[np.ndarray], float],
                         grad: Callable[[np.ndarray], np.ndarray],
                         h: Callable[[np.ndarray], np.ndarray],
                         x0: np.ndarray, mu0: float = 1.0, iters: int = 100) -> Dict:
    """TODO: Basic augmented Lagrangian for equality constraints h(x)=0."""
    # TODO: Implement this
    pass


def projection_box(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """TODO: Project onto box [lower, upper]^n."""
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Constrained Optimization Exercises")

