"""
Stochastic Optimization Exercises
"""

from typing import Callable, Dict
import numpy as np


def sgd(grad_fn: Callable[[np.ndarray, int], np.ndarray], x0: np.ndarray,
        iters: int, lr_schedule: Callable[[int], float]) -> Dict:
    """TODO: Basic SGD loop; grad_fn uses stochastic index/seed internal to function."""
    # TODO: Implement this
    pass


def svrg(grad_full: Callable[[np.ndarray], np.ndarray], grad_i: Callable[[np.ndarray, int], np.ndarray],
         x0: np.ndarray, n: int, m: int, eta: float, epochs: int) -> Dict:
    """TODO: SVRG for finite-sum (1/n)∑ f_i(x)."""
    # TODO: Implement this
    pass


def adam(grad_fn: Callable[[np.ndarray, int], np.ndarray], x0: np.ndarray, iters: int,
         lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Dict:
    """TODO: Adam optimizer with bias correction."""
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Stochastic Optimization Exercises")

