"""
Stochastic Optimization Solutions - Reference Implementation
"""

from typing import Callable, Dict
import numpy as np


def sgd(grad_fn: Callable[[np.ndarray, int], np.ndarray], x0: np.ndarray,
        iters: int, lr_schedule: Callable[[int], float]) -> Dict:
    x = x0.astype(float).copy()
    traj = [x.copy()]
    for k in range(1, iters + 1):
        g = grad_fn(x, k)
        x = x - lr_schedule(k) * g
        traj.append(x.copy())
    return {'x': x, 'trajectory': traj}


def svrg(grad_full: Callable[[np.ndarray], np.ndarray], grad_i: Callable[[np.ndarray, int], np.ndarray],
         x0: np.ndarray, n: int, m: int, eta: float, epochs: int) -> Dict:
    rng = np.random.default_rng(0)
    x = x0.astype(float).copy()
    for _ in range(epochs):
        mu = grad_full(x)
        y = x.copy()
        for _ in range(m):
            i = int(rng.integers(0, n))
            g = grad_i(y, i) - grad_i(x, i) + mu
            y = y - eta * g
        x = y
    return {'x': x}


def adam(grad_fn: Callable[[np.ndarray, int], np.ndarray], x0: np.ndarray, iters: int,
         lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Dict:
    x = x0.astype(float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    traj = [x.copy()]
    for k in range(1, iters + 1):
        g = grad_fn(x, k)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(x.copy())
    return {'x': x, 'trajectory': traj}


