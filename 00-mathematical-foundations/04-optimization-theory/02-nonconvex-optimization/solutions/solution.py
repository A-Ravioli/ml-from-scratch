"""
Nonconvex Optimization Solutions - Reference Implementation
"""

from typing import Tuple, Dict
import numpy as np


def find_critical_points_2d(f, grad, hess, grid):
    xs, ys = grid
    minima, saddles, maxima = [], [], []
    for x in xs:
        for y in ys:
            p = np.array([x, y], dtype=float)
            g = grad(p)
            if np.linalg.norm(g) < 1e-2:
                H = hess(p)
                vals = np.linalg.eigvals(H)
                if np.all(vals > 1e-6):
                    minima.append(p)
                elif np.all(vals < -1e-6):
                    maxima.append(p)
                else:
                    saddles.append(p)
    return {'minima': minima, 'saddles': saddles, 'maxima': maxima}


def noisy_gradient_descent(f, grad, x0: np.ndarray, step: float, noise_std: float,
                           iters: int = 1000) -> Dict:
    x = x0.astype(float).copy()
    traj = [x.copy()]
    for k in range(iters):
        g = grad(x)
        noise = noise_std * np.random.randn(*x.shape)
        x = x - step * (g + noise)
        traj.append(x.copy())
    return {'trajectory': traj, 'final_value': float(f(x))}


def trust_region_step(gradx: np.ndarray, hessx: np.ndarray, delta: float) -> np.ndarray:
    # Cauchy point: along -grad direction, clipped to trust region
    g = gradx
    if np.linalg.norm(g) == 0:
        return np.zeros_like(g)
    g_norm = np.linalg.norm(g)
    Hg = hessx @ g
    denom = g @ Hg
    if denom <= 0:
        tau = 1.0
    else:
        tau = min(1.0, (g_norm**3) / (delta * denom))
    p = - (tau * delta / g_norm) * g
    if np.linalg.norm(p) > delta:
        p = - delta * g / g_norm
    return p


