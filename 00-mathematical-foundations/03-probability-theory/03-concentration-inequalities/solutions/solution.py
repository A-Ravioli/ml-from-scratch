"""
Concentration Inequalities Solutions - Reference Implementation
"""

from typing import Tuple
import numpy as np


def is_subgaussian(samples: np.ndarray, t_values: np.ndarray, c: float) -> bool:
    x = samples - np.mean(samples)
    for t in t_values:
        mgf_hat = float(np.mean(np.exp(t * x)))
        bound = float(np.exp(0.5 * (c**2) * (t**2)))
        if mgf_hat - bound > 1e-2:
            return False
    return True


def subgaussian_parameter(samples: np.ndarray, t_values: np.ndarray) -> float:
    x = samples - np.mean(samples)
    c2_max = 0.0
    for t in t_values:
        if abs(t) < 1e-12:
            continue
        mgf_hat = float(np.mean(np.exp(t * x)))
        c2 = 2.0 * np.log(max(mgf_hat, 1e-12)) / (t**2)
        c2_max = max(c2_max, c2)
    return float(np.sqrt(max(c2_max, 0.0)))


def hoeffding_bound(n: int, a: float, b: float, t: float) -> float:
    return float(np.exp(-2.0 * n * (t**2) / ((b - a)**2)))


def bernstein_bound(n: int, sigma2: float, b: float, t: float) -> float:
    v = n * sigma2
    c = (b) / 3.0
    return float(np.exp(-(t**2) / (2*v + 2*c*t)))


def chernoff_bound(p: float, n: int, delta: float) -> float:
    # P(X >= (1+delta)np) ≤ [e^{delta} / (1+delta)^{1+delta}]^{np}
    if delta <= -1:
        return 0.0
    term = (np.exp(delta) / ((1 + delta)**(1 + delta))) ** (n * p)
    return float(term)


def verify_bounds_empirically(dist_sampler, n: int, trials: int = 1000) -> dict:
    rng = np.random.default_rng(0)
    # Use Uniform[0,1] default-like sampler signature
    means = []
    for _ in range(trials):
        X = dist_sampler(n)
        means.append(float(np.mean(X)))
    means = np.array(means)
    mu_hat = float(np.mean(means))
    t = 0.1
    violations = float(np.mean(means - mu_hat >= t))
    hb = hoeffding_bound(n=n, a=0.0, b=1.0, t=t)
    return {
        'empirical_dev_ge_t': violations,
        'hoeffding_bound': hb,
        'hoeffding_violations': violations <= hb + 0.1
    }


