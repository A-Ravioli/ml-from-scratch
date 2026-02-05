"""
Distributed Optimization — Exercises

Implement a minimal synchronous data-parallel SGD simulator to understand
gradient aggregation, synchronization, and (toy) gradient compression.

Run tests:
  python3 -m pytest test_implementation.py -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PartitionedDataset:
    """Simple in-memory dataset partitioned across workers."""

    X_parts: List[Array]
    y_parts: List[Array]

    @property
    def n_workers(self) -> int:
        return len(self.X_parts)


def mean_aggregate(grads: Sequence[Array]) -> Array:
    """Compute the elementwise mean of gradients."""
    # YOUR CODE HERE
    raise NotImplementedError


def weighted_mean_aggregate(grads: Sequence[Array], weights: Sequence[float]) -> Array:
    """Compute a weighted mean of gradients."""
    # YOUR CODE HERE
    raise NotImplementedError


def topk_sparsify(grad: Array, k: int) -> Array:
    """
    Toy gradient compression: keep only top-k entries by magnitude.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def quadratic_loss_and_grad(A: Array, b: Array, x: Array) -> Tuple[float, Array]:
    """
    f(x) = 0.5 x^T A x - b^T x
    """
    loss = 0.5 * float(x.T @ A @ x) - float(b.T @ x)
    grad = A @ x - b
    return loss, grad


@dataclass
class Worker:
    """Computes local gradients on its partition."""

    X: Array
    y: Array

    def full_gradient(self, x: Array) -> Array:
        # Simple linear regression least squares gradient:
        # L = (1/n) ||Xx - y||^2, grad = (2/n) X^T (Xx - y)
        # YOUR CODE HERE
        raise NotImplementedError


@dataclass
class ParameterServer:
    """Holds parameters and applies aggregated gradients."""

    x: Array

    def apply_update(self, grad: Array, lr: float) -> None:
        # YOUR CODE HERE
        raise NotImplementedError


def synchronous_data_parallel_sgd(
    dataset: PartitionedDataset,
    x0: Array,
    lr: float,
    n_steps: int,
    *,
    weights: Optional[Sequence[float]] = None,
    topk: Optional[int] = None,
) -> List[Array]:
    """
    Run synchronous data-parallel SGD with gradient aggregation.

    Returns the parameter trajectory (including x0).
    """
    # YOUR CODE HERE
    raise NotImplementedError

