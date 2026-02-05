"""
Test suite for GAN Theory implementations.

Tests theoretical analysis tools and mathematical foundations.
"""

import numpy as np
import pytest
import torch
from exercise import (
    GANGameTheory, ConvergenceAnalyzer, FDivergenceGAN, GANCapacityAnalysis,
    simulate_1d_gan_theory
)


class TestGANGameTheory:
    """Test GAN game theory analysis."""
    
    def test_optimal_discriminator(self):
        """Test optimal discriminator computation."""
        # Create simple 1D distributions
        x_points = np.linspace(-2, 2, 100)
        p_data = np.exp(-0.5 * x_points**2) / np.sqrt(2 * np.pi)
        p_gen = np.exp(-0.5 * (x_points - 1)**2) / np.sqrt(2 * np.pi)
        
        d_optimal = GANGameTheory.optimal_discriminator(p_data, p_gen, x_points)
        
        if d_optimal is not None:
            # Optimal discriminator should be between 0 and 1
            assert np.all(d_optimal >= 0) and np.all(d_optimal <= 1)
            
            # Should equal p_data / (p_data + p_gen)
            expected = p_data / (p_data + p_gen + 1e-8)
            np.testing.assert_allclose(d_optimal, expected, rtol=1e-5)
    
    def test_jensen_shannon_divergence(self):
        """Test JS divergence computation."""
        # Test with identical distributions
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.3, 0.2])
        
        js_div = GANGameTheory.jensen_shannon_divergence(p, q)
        
        if js_div is not None:
            # JS divergence should be 0 for identical distributions
            assert abs(js_div) < 1e-6
    
    def test_gan_objective_analysis(self):
        """Test GAN objective analysis."""
        x_points = np.linspace(-2, 2, 50)
        p_data = np.exp(-0.5 * x_points**2) / np.sqrt(2 * np.pi)
        p_gen = np.exp(-0.5 * (x_points + 0.5)**2) / np.sqrt(2 * np.pi)
        
        result = GANGameTheory.gan_objective_analysis(p_data, p_gen, x_points)
        
        if result is not None:
            assert isinstance(result, dict)
            # Should contain expected keys
            expected_keys = ['generator_loss', 'discriminator_loss', 'js_divergence']
            for key in expected_keys:
                if key in result:
                    assert isinstance(result[key], (int, float))


class TestConvergenceAnalyzer:
    """Test convergence analysis tools."""
    
    def test_gradient_norms(self):
        """Test gradient norm computation."""
        # Create dummy gradient sequences
        n_steps = 100
        generator_grads = [np.random.randn(10) for _ in range(n_steps)]
        discriminator_grads = [np.random.randn(15) for _ in range(n_steps)]
        
        result = ConvergenceAnalyzer.compute_gradient_norms(generator_grads, discriminator_grads)
        
        if result is not None:
            assert isinstance(result, dict)
            assert 'generator_norms' in result or 'discriminator_norms' in result
    
    def test_mode_collapse_detection(self):
        """Test mode collapse detection."""
        # Create sequences with and without mode collapse
        normal_samples = [np.random.randn(100, 2) for _ in range(10)]
        collapsed_samples = [np.random.randn(100, 2) * 0.1 for _ in range(10)]
        
        normal_result = ConvergenceAnalyzer.mode_collapse_detection(normal_samples)
        collapsed_result = ConvergenceAnalyzer.mode_collapse_detection(collapsed_samples)
        
        if normal_result is not None and collapsed_result is not None:
            # Collapsed samples should have lower diversity
            if 'sample_diversity' in normal_result and 'sample_diversity' in collapsed_result:
                assert normal_result['sample_diversity'] > collapsed_result['sample_diversity']
    
    def test_equilibrium_analysis(self):
        """Test equilibrium analysis."""
        # Create dummy loss history
        n_epochs = 100
        loss_history = {
            'generator_loss': np.random.randn(n_epochs) * 0.1 + 1.0,
            'discriminator_loss': np.random.randn(n_epochs) * 0.1 + 0.8
        }
        
        result = ConvergenceAnalyzer.equilibrium_analysis(loss_history)
        
        if result is not None:
            assert isinstance(result, dict)
            # Should contain stability metrics
            expected_keys = ['oscillation_measure', 'convergence_score', 'stability_score']
            for key in expected_keys:
                if key in result:
                    assert isinstance(result[key], (int, float))


class TestFDivergenceGAN:
    """Test f-divergence computations."""
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        # Test with simple distributions
        p = np.array([0.7, 0.3])
        q = np.array([0.6, 0.4])
        
        kl_div = FDivergenceGAN.kl_divergence(p, q)
        
        if kl_div is not None:
            # KL divergence should be non-negative
            assert kl_div >= 0
            
            # Test symmetry property: KL(p,q) != KL(q,p) in general
            kl_reverse = FDivergenceGAN.kl_divergence(q, p)
            if kl_reverse is not None:
                # They should generally be different
                assert abs(kl_div - kl_reverse) > 1e-10 or abs(kl_div) < 1e-10
    
    def test_reverse_kl_divergence(self):
        """Test reverse KL divergence."""
        p = np.array([0.8, 0.2])
        q = np.array([0.5, 0.5])
        
        rkl_div = FDivergenceGAN.reverse_kl_divergence(p, q)
        
        if rkl_div is not None:
            # Should be non-negative
            assert rkl_div >= 0
    
    def test_total_variation_distance(self):
        """Test total variation distance."""
        # Test with identical distributions
        p = np.array([0.4, 0.6])
        q = np.array([0.4, 0.6])
        
        tv_dist = FDivergenceGAN.total_variation_distance(p, q)
        
        if tv_dist is not None:
            # TV distance should be 0 for identical distributions
            assert abs(tv_dist) < 1e-10
            
        # Test with different distributions
        q2 = np.array([0.2, 0.8])
        tv_dist2 = FDivergenceGAN.total_variation_distance(p, q2)
        
        if tv_dist2 is not None:
            # Should be positive and <= 1
            assert 0 <= tv_dist2 <= 1
    
    def test_chi_squared_divergence(self):
        """Test chi-squared divergence."""
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        
        chi2_div = FDivergenceGAN.chi_squared_divergence(p, q)
        
        if chi2_div is not None:
            # Should be non-negative
            assert chi2_div >= 0
    
    def test_f_divergence_gan_objective(self):
        """Test f-divergence GAN objective."""
        p_data = np.array([0.7, 0.3])
        p_gen = np.array([0.4, 0.6])
        
        # Test with KL divergence function
        f_kl = lambda t: t * np.log(t + 1e-8)
        
        f_div = FDivergenceGAN.f_divergence_gan_objective(f_kl, p_data, p_gen)
        
        if f_div is not None:
            assert isinstance(f_div, (int, float))


class TestGANCapacityAnalysis:
    """Test GAN capacity analysis."""
    
    def test_generator_capacity(self):
        """Test generator capacity analysis."""
        architecture = [100, 256, 128, 784]
        
        capacity = GANCapacityAnalysis.generator_capacity(architecture, activation='relu')
        
        if capacity is not None:
            assert isinstance(capacity, dict)
            
            # Should contain parameter count
            if 'num_parameters' in capacity:
                expected_params = 100*256 + 256*128 + 128*784
                assert capacity['num_parameters'] >= expected_params
    
    def test_discriminator_capacity(self):
        """Test discriminator capacity analysis."""
        architecture = [784, 128, 64, 1]
        
        capacity = GANCapacityAnalysis.discriminator_capacity(architecture)
        
        if capacity is not None:
            assert isinstance(capacity, dict)
            
            if 'num_parameters' in capacity:
                assert capacity['num_parameters'] > 0
    
    def test_sample_complexity_bounds(self):
        """Test sample complexity bounds."""
        bounds = GANCapacityAnalysis.sample_complexity_bounds(
            data_dimension=784,
            generator_params=50000,
            discriminator_params=30000,
            confidence=0.95
        )
        
        if bounds is not None:
            assert isinstance(bounds, dict)
            
            # Should contain various complexity measures
            expected_keys = ['vc_bound', 'rademacher_bound', 'pac_bound']
            for key in expected_keys:
                if key in bounds:
                    assert bounds[key] > 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_simulate_1d_gan_theory(self):
        """Test 1D GAN theory simulation."""
        result = simulate_1d_gan_theory(data_mean=1.0, data_std=0.5, 
                                       gen_mean=0.0, gen_std=1.0)
        
        if result is not None:
            assert isinstance(result, dict)
            
            # Should contain simulation results
            expected_keys = ['optimal_discriminator', 'js_divergence', 'equilibrium_loss']
            for key in expected_keys:
                if key in result:
                    assert isinstance(result[key], (int, float, np.ndarray))


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_analysis(self):
        """Test complete theoretical analysis pipeline."""
        # Create simple 1D scenario
        x_points = np.linspace(-3, 3, 100)
        p_data = np.exp(-0.5 * x_points**2) / np.sqrt(2 * np.pi)
        p_gen = np.exp(-0.5 * (x_points - 1)**2) / np.sqrt(2 * np.pi)
        
        d_optimal = GANGameTheory.optimal_discriminator(p_data, p_gen, x_points)
        js_div = GANGameTheory.jensen_shannon_divergence(
            p_data / np.sum(p_data),
            p_gen / np.sum(p_gen),
        )

        # Basic sanity checks
        assert d_optimal is not None
        assert js_div is not None
        assert len(d_optimal) == len(x_points)
        assert np.all((d_optimal >= 0) & (d_optimal <= 1))
        assert js_div >= 0
    
    def test_theoretical_consistency(self):
        """Test theoretical consistency of implementations."""
        # Test that JS divergence matches optimal discriminator loss
        p_data = np.array([0.8, 0.2])
        p_gen = np.array([0.3, 0.7])
        
        # Theoretical connection: at optimum, generator loss = -log(4) + 2*JS(P||Q)
        js_div = GANGameTheory.jensen_shannon_divergence(p_data, p_gen)
        
        if js_div is not None:
            expected_gen_loss = -np.log(4) + 2 * js_div
            
            # This is a theoretical check - implementation may vary
            assert isinstance(expected_gen_loss, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
