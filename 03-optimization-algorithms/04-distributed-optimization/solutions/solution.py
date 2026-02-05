"""
Distributed Optimization — Solutions (Reference Implementation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class PartitionedDataset:
    X_parts: List[Array]
    y_parts: List[Array]

    @property
    def n_workers(self) -> int:
        return len(self.X_parts)


def mean_aggregate(grads: Sequence[Array]) -> Array:
    grads = [np.asarray(g, dtype=float) for g in grads]
    return np.mean(np.stack(grads, axis=0), axis=0)


def weighted_mean_aggregate(grads: Sequence[Array], weights: Sequence[float]) -> Array:
    grads = [np.asarray(g, dtype=float) for g in grads]
    w = np.asarray(list(weights), dtype=float)
    w = w / np.sum(w)
    stacked = np.stack(grads, axis=0)
    return np.tensordot(w, stacked, axes=(0, 0))


def topk_sparsify(grad: Array, k: int) -> Array:
    grad = np.asarray(grad, dtype=float)
    if k <= 0:
        return np.zeros_like(grad)
    if k >= grad.size:
        return grad.copy()
    flat = grad.flatten()
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    out = np.zeros_like(flat)
    out[idx] = flat[idx]
    return out.reshape(grad.shape)


def quadratic_loss_and_grad(A: Array, b: Array, x: Array) -> Tuple[float, Array]:
    loss = 0.5 * float(x.T @ A @ x) - float(b.T @ x)
    grad = A @ x - b
    return loss, grad


@dataclass
class Worker:
    X: Array
    y: Array

    def full_gradient(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        n = self.X.shape[0]
        r = self.X @ x - self.y
        return (2.0 / n) * (self.X.T @ r)


@dataclass
class ParameterServer:
    x: Array

    def apply_update(self, grad: Array, lr: float) -> None:
        self.x = self.x - float(lr) * np.asarray(grad, dtype=float)


def synchronous_data_parallel_sgd(
    dataset: PartitionedDataset,
    x0: Array,
    lr: float,
    n_steps: int,
    *,
    weights: Optional[Sequence[float]] = None,
    topk: Optional[int] = None,
) -> List[Array]:
    x0 = np.asarray(x0, dtype=float)
    ps = ParameterServer(x=x0.copy())
    workers = [Worker(X, y) for X, y in zip(dataset.X_parts, dataset.y_parts)]

    traj = [ps.x.copy()]
    for _ in range(n_steps):
        grads = [w.full_gradient(ps.x) for w in workers]
        if topk is not None:
            grads = [topk_sparsify(g, int(topk)) for g in grads]

        if weights is None:
            g = mean_aggregate(grads)
        else:
            g = weighted_mean_aggregate(grads, weights)

        ps.apply_update(g, lr)
        traj.append(ps.x.copy())
    return traj

