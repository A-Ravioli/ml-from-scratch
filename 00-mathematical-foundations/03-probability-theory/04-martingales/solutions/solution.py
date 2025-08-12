"""
Martingales Solutions - Reference Implementation
"""

from typing import Tuple
import numpy as np


def is_martingale(paths: np.ndarray, tol: float = 1e-2) -> bool:
    # Check E[X_{t+1}-X_t] ≈ 0 across paths for each t
    diffs = paths[:, 1:] - paths[:, :-1]
    means = diffs.mean(axis=0)
    return float(np.max(np.abs(means))) <= tol


def optional_stopping_demo(rng: np.random.Generator = None) -> dict:
    rng = rng or np.random.default_rng(0)
    n_paths, T = 500, 1000
    steps = rng.choice([-1.0, 1.0], size=(n_paths, T))
    paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)], axis=1)
    # Stop when hitting ±10 or at T
    thresholds = 10.0
    taus = np.full(n_paths, T, dtype=int)
    for i in range(n_paths):
        hit = np.where(np.abs(paths[i]) >= thresholds)[0]
        if len(hit) > 0:
            taus[i] = int(hit[0])
    X_tau = np.array([paths[i, taus[i]] for i in range(n_paths)])
    return {'E_X_tau': float(np.mean(X_tau)), 'E_X_0': 0.0, 'taus_mean': float(np.mean(taus))}


def azuma_hoeffding_bound(t: float, c: float, n: int) -> float:
    return float(np.exp(-2.0 * (t**2) / (n * (c**2))))


