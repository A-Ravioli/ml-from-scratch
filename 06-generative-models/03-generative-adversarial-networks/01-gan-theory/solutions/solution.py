from __future__ import annotations

from typing import Dict, List

import numpy as np


class GANGameTheory:
    @staticmethod
    def optimal_discriminator(p_data: np.ndarray, p_gen: np.ndarray, x_points: np.ndarray) -> np.ndarray:
        del x_points
        denom = p_data + p_gen + 1e-12
        return p_data / denom


class ConvergenceAnalyzer:
    @staticmethod
    def compute_gradient_norms(generator_grads: List[np.ndarray], discriminator_grads: List[np.ndarray]) -> Dict[str, List[float]]:
        generator = [float(np.linalg.norm(g)) for g in generator_grads]
        discriminator = [float(np.linalg.norm(g)) for g in discriminator_grads]
        return {"generator": generator, "discriminator": discriminator}


class FDivergenceGAN:
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=float) + 1e-12
        q = np.asarray(q, dtype=float) + 1e-12
        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def reverse_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        return FDivergenceGAN.kl_divergence(q, p)


class GANCapacityAnalysis:
    @staticmethod
    def generator_capacity(generator_architecture: List[int], activation: str = "relu") -> Dict[str, float]:
        linear_regions = float(np.prod([max(width, 1) for width in generator_architecture[1:-1]]) or 1)
        num_parameters = 0
        for in_dim, out_dim in zip(generator_architecture[:-1], generator_architecture[1:]):
            num_parameters += in_dim * out_dim + out_dim
        return {
            "num_parameters": float(num_parameters),
            "linear_regions": linear_regions,
            "activation_bonus": 2.0 if activation == "relu" else 1.0,
        }


def simulate_1d_gan_theory(data_mean: float = 2.0, data_std: float = 0.5, generator_mean: float = 0.0, generator_std: float = 1.0, num_points: int = 128):
    x_points = np.linspace(-4.0, 4.0, num_points)
    p_data = np.exp(-0.5 * ((x_points - data_mean) / data_std) ** 2)
    p_gen = np.exp(-0.5 * ((x_points - generator_mean) / generator_std) ** 2)
    p_data /= p_data.sum()
    p_gen /= p_gen.sum()
    return {"x_points": x_points, "p_data": p_data, "p_gen": p_gen}
