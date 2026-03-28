"""
Feature Learning - Exercises

Implement a compact, deterministic version of the core ideas from the lesson.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Set the NumPy seed used by the exercises."""
    np.random.seed(seed)


@dataclass
class TopicConfig:
    input_dim: int
    hidden_dim: int = 4
    scale: float = 0.5


def build_feature_map(x: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Build a simple nonlinear feature map.

    TODO: return a concatenation of:
    - the original features
    - scaled squared features
    - a bias column of ones
    """
    raise NotImplementedError


def topic_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a normalized similarity score between two vectors.

    TODO: implement cosine similarity with a small numerical stabilizer.
    """
    raise NotImplementedError


class TopicModel:
    """
    Small reference model used throughout the generated topics.
    """

    def __init__(self, config: TopicConfig):
        self.config = config
        self.weight_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TopicModel":
        """
        Fit a compact linear projector on the generated feature map.

        TODO:
        1. Build the feature map.
        2. Compute the mean feature vector.
        3. Normalize it to unit norm and store it in `weight_`.
        """
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Project data onto the learned direction.

        TODO: raise a ValueError if the model is not fitted.
        """
        raise NotImplementedError

    def score(self, x: np.ndarray) -> float:
        """
        Return the average absolute projected magnitude.
        """
        raise NotImplementedError
