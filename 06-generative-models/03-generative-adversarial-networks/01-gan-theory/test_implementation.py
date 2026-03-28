import numpy as np

from exercise import (
    ConvergenceAnalyzer,
    FDivergenceGAN,
    GANCapacityAnalysis,
    GANGameTheory,
    simulate_1d_gan_theory,
)


def test_optimal_discriminator_range():
    p_data = np.array([0.6, 0.4])
    p_gen = np.array([0.2, 0.8])
    d_star = GANGameTheory.optimal_discriminator(p_data, p_gen, np.array([0.0, 1.0]))
    assert d_star.shape == (2,)
    assert np.all((d_star >= 0.0) & (d_star <= 1.0))


def test_gradient_norm_analysis():
    grads = [np.array([3.0, 4.0]), np.array([0.0, 2.0])]
    out = ConvergenceAnalyzer.compute_gradient_norms(grads, grads)
    assert len(out["generator"]) == 2
    assert out["generator"][0] == 5.0


def test_divergences_are_non_negative():
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    assert FDivergenceGAN.kl_divergence(p, q) >= 0.0
    assert FDivergenceGAN.reverse_kl_divergence(p, q) >= 0.0


def test_capacity_analysis_keys():
    stats = GANCapacityAnalysis.generator_capacity([4, 8, 2], activation="relu")
    assert "num_parameters" in stats
    assert stats["num_parameters"] > 0


def test_simulation_shapes():
    sim = simulate_1d_gan_theory(num_points=32)
    assert sim["x_points"].shape == (32,)
    assert sim["p_data"].shape == (32,)
    assert sim["p_gen"].shape == (32,)
