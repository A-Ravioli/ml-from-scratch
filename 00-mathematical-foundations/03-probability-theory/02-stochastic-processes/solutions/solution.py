"""
Stochastic Processes Solutions - Reference Implementation
"""

from typing import Tuple
import numpy as np


def simulate_markov_chain(P: np.ndarray, x0: int, T: int, rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = P.shape[0]
    traj = [x0]
    x = x0
    for _ in range(T):
        x = rng.choice(n, p=P[x])
        traj.append(x)
    return np.array(traj, dtype=int)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    # Solve (P^T - I) pi = 0 with sum pi = 1
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.maximum(pi, 0)
    pi = pi / np.sum(pi)
    return pi


def mixing_time_upper_bound(P: np.ndarray, epsilon: float = 1e-2) -> float:
    # Spectral gap bound (reversible or as heuristic): tau_mix ≲ (1-gap)^{-1} log(1/(epsilon*pi_min))
    vals = np.linalg.eigvals(P)
    vals = np.sort(np.abs(vals))
    second = np.max(vals[:-1])
    gap = 1 - second
    if gap <= 0:
        return np.inf
    pi = stationary_distribution(P)
    pi_min = float(np.min(pi) + 1e-12)
    return float(np.log(1.0/(epsilon*pi_min)) / gap)


def simulate_poisson_process(lmbda: float, T: float, rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    times = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / lmbda)
        if t > T:
            break
        times.append(t)
    return np.array([0.0] + times + [T])


def simulate_ctmc(Q: np.ndarray, x0: int, T: float, rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    n = Q.shape[0]
    rates = -np.diag(Q)
    states = [x0]
    times = [0.0]
    t = 0.0
    x = x0
    while t < T:
        rate = rates[x]
        if rate <= 0:
            break
        t += rng.exponential(1.0 / rate)
        if t > T:
            break
        # choose next state according to normalized off-diagonal rates
        probs = np.maximum(Q[x].copy(), 0.0)
        probs[x] = 0.0
        s = probs.sum()
        if s <= 0:
            break
        probs = probs / s
        x = rng.choice(n, p=probs)
        states.append(x)
        times.append(t)
    return np.array(states, dtype=int), np.array(times)


