"""
Flamingo - Reference Solution
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


@dataclass
class TopicConfig:
    input_dim: int
    hidden_dim: int = 4
    scale: float = 0.5


def build_feature_map(x: np.ndarray, scale: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    squared = scale * np.square(x)
    bias = np.ones((x.shape[0], 1), dtype=float)
    return np.concatenate([x, squared, bias], axis=1)


def topic_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-12
    return float(np.dot(x, y) / denom)


class TopicModel:
    def __init__(self, config: TopicConfig):
        self.config = config
        self.weight_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TopicModel":
        features = build_feature_map(x, scale=self.config.scale)
        weight = features.mean(axis=0)
        norm = np.linalg.norm(weight)
        self.weight_ = weight / norm if norm > 0 else weight
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.weight_ is None:
            raise ValueError("Model must be fitted before calling transform")
        features = build_feature_map(x, scale=self.config.scale)
        return features @ self.weight_

    def score(self, x: np.ndarray) -> float:
        projections = self.transform(x)
        return float(np.mean(np.abs(projections)))
