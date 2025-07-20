"""
Test suite for Measure-Theoretic Probability implementations.
"""

import numpy as np
import pytest
from scipy import stats
from exercise import (
    SigmaAlgebra, ProbabilitySpace, RandomVariable, check_independence,
    ConditionalExpectation, demonstrate_convergence_modes,
    empirical_characteristic_function, verify_concentration_inequalities,
    GaussianProcess, monte_carlo_integration, importance_sampling,
    empirical_process_theory_demo, information_theory_connections
)


class TestSigmaAlgebra:
    """Test σ-algebra operations."""
    
    def test_verify_sigma_algebra(self):
        """Test σ-algebra verification."""
        # Valid σ-algebra (power set)
        omega = {1, 2, 3}
        subsets = [set(), {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}]
        sigma = SigmaAlgebra(omega, subsets)
        
        results = sigma.verify_sigma_algebra()
        assert results['contains_omega'] == True
        assert results['closed_under_complement'] == True
        assert results['closed_under_union'] == True
        
        # Invalid σ-algebra (missing complement)
        bad_subsets = [set(), {1}, {2,3}, {1,2,3}]
        bad_sigma = SigmaAlgebra(omega, bad_subsets)
        
        results = bad_sigma.verify_sigma_algebra()
        assert results['closed_under_complement'] == False
    
    def test_generate_from_sets(self):
        """Test σ-algebra generation."""
        omega = {1, 2, 3, 4}
        generators = [{1, 2}, {2, 3}]
        
        sigma = SigmaAlgebra(omega, [])
        generated = sigma.generate_from_sets(generators)
        
        # Should contain generators and be closed
        assert {1, 2} in generated.subsets
        assert {2, 3} in generated.subsets
        
        # Verify it's a valid σ-algebra
        results = generated.verify_sigma_algebra()
        assert all(results.values())


class TestProbabilitySpace:
    """Test probability space operations."""
    
    def test_probability_measure_verification(self):
        """Test probability measure properties."""
        # Dice roll
        omega = {1, 2, 3, 4, 5, 6}
        sigma = SigmaAlgebra(omega, [set()] + [{i} for i in omega] + [omega])
        
        # Uniform distribution
        prob = {frozenset(): 0, frozenset(omega): 1}
        for i in omega:
            prob[frozenset({i})] = 1/6
        
        prob_space = ProbabilitySpace(omega, sigma, prob)
        
        results = prob_space.verify_probability_measure()
        assert results['total_probability_one'] == True
        assert results['non_negative'] == True
        assert results['additive'] == True
    
    def test_compute_probability(self):
        """Test probability computation."""
        # Biased coin
        omega = {'H', 'T'}
        sigma = SigmaAlgebra(omega, [set(), {'H'}, {'T'}, omega])
        
        prob = {
            frozenset(): 0,
            frozenset({'H'}): 0.7,
            frozenset({'T'}): 0.3,
            frozenset(omega): 1
        }
        
        prob_space = ProbabilitySpace(omega, sigma, prob)
        
        assert abs(prob_space.compute_probability({'H'}) - 0.7) < 1e-10
        assert abs(prob_space.compute_probability({'T'}) - 0.3) < 1e-10
        assert abs(prob_space.compute_probability(omega) - 1.0) < 1e-10


class TestRandomVariables:
    """Test random variable operations."""
    
    def test_measurability(self):
        """Test random variable measurability."""
        omega = {1, 2, 3, 4}
        sigma = SigmaAlgebra(omega, [set()] + [{i} for i in omega] + [omega])
        prob = {frozenset({i}): 0.25 for i in omega}
        prob[frozenset()] = 0
        prob[frozenset(omega)] = 1
        
        prob_space = ProbabilitySpace(omega, sigma, prob)
        
        # Measurable function
        X_map = {1: 0, 2: 0, 3: 1, 4: 1}
        X = RandomVariable(prob_space, X_map)
        
        assert X.verify_measurability() == True
    
    def test_expectation_variance(self):
        """Test expectation and variance computation."""
        # Die roll
        omega = {1, 2, 3, 4, 5, 6}
        sigma = SigmaAlgebra(omega, [set()] + [{i} for i in omega] + [omega])
        prob = {frozenset({i}): 1/6 for i in omega}
        prob[frozenset()] = 0
        prob[frozenset(omega)] = 1
        
        prob_space = ProbabilitySpace(omega, sigma, prob)
        
        # Identity random variable
        X_map = {i: i for i in omega}
        X = RandomVariable(prob_space, X_map)
        
        # E[X] = 3.5
        assert abs(X.expectation() - 3.5) < 1e-10
        
        # Var(X) = E[X²] - E[X]² = 91/6 - 49/4 = 35/12
        assert abs(X.variance() - 35/12) < 1e-10
    
    def test_independence(self):
        """Test independence checking."""
        # Two independent coin flips
        omega = {('H','H'), ('H','T'), ('T','H'), ('T','T')}
        sigma = SigmaAlgebra(omega, [set()] + [{outcome} for outcome in omega] + [omega])
        prob = {frozenset({outcome}): 0.25 for outcome in omega}
        prob[frozenset()] = 0
        prob[frozenset(omega)] = 1
        
        prob_space = ProbabilitySpace(omega, sigma, prob)
        
        # First coin
        X_map = {outcome: 1 if outcome[0] == 'H' else 0 for outcome in omega}
        X = RandomVariable(prob_space, X_map)
        
        # Second coin
        Y_map = {outcome: 1 if outcome[1] == 'H' else 0 for outcome in omega}
        Y = RandomVariable(prob_space, Y_map)
        
        assert check_independence(X, Y) == True


class TestConditionalExpectation:
    """Test conditional expectation."""
    
    def test_conditional_expectation_basic(self):
        """Test basic conditional expectation properties."""
        omega = {1, 2, 3, 4}
        sigma = SigmaAlgebra(omega, [set()] + [{i} for i in omega] + [omega])
        prob = {frozenset({i}): 0.25 for i in omega}
        prob[frozenset()] = 0
        prob[frozenset(omega)] = 1
        
        prob_space = ProbabilitySpace(omega, sigma, prob)
        
        # Random variable
        X_map = {1: 1, 2: 2, 3: 3, 4: 4}
        X = RandomVariable(prob_space, X_map)
        
        # Condition on partition {{1,2}, {3,4}}
        sub_sigma = SigmaAlgebra(omega, [set(), {1,2}, {3,4}, omega])
        
        cond_exp = ConditionalExpectation(X, sub_sigma)
        result = cond_exp.compute()
        
        # E[X|{1,2}] = 1.5, E[X|{3,4}] = 3.5
        assert abs(result[1] - 1.5) < 1e-10
        assert abs(result[2] - 1.5) < 1e-10
        assert abs(result[3] - 3.5) < 1e-10
        assert abs(result[4] - 3.5) < 1e-10
        
        # Verify properties
        props = cond_exp.verify_properties(result)
        assert props['is_measurable'] == True
        assert props['tower_property'] == True


class TestGaussianProcess:
    """Test Gaussian process implementation."""
    
    def test_gp_prior_sampling(self):
        """Test GP prior sampling."""
        def zero_mean(x):
            return 0
        
        def rbf_kernel(x1, x2):
            return np.exp(-0.5 * np.sum((x1 - x2)**2))
        
        gp = GaussianProcess(zero_mean, rbf_kernel)
        
        # Sample at points
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        samples = gp.sample_prior(X, n_samples=5)
        
        assert samples.shape == (5, 10)
        
        # Check that samples have expected covariance structure
        K = np.array([[rbf_kernel(x1, x2) for x2 in X] for x1 in X])
        sample_cov = np.cov(samples.T)
        
        # Should be approximately equal (with sampling noise)
        assert np.max(np.abs(sample_cov - K)) < 0.5
    
    def test_gp_posterior(self):
        """Test GP posterior computation."""
        def zero_mean(x):
            return 0
        
        def rbf_kernel(x1, x2):
            return np.exp(-0.5 * np.sum((x1 - x2)**2))
        
        gp = GaussianProcess(zero_mean, rbf_kernel)
        
        # Training data
        X_train = np.array([[0], [1]])
        y_train = np.array([0, 1])
        
        # Test points
        X_test = np.array([[0.5]])
        
        mean, cov = gp.posterior(X_train, y_train, X_test, noise_var=1e-6)
        
        # At x=0.5, should interpolate between 0 and 1
        assert 0.4 < mean[0] < 0.6
        assert cov[0, 0] > 0  # Positive uncertainty


class TestMonteCarloMethods:
    """Test Monte Carlo integration methods."""
    
    def test_monte_carlo_integration(self):
        """Test basic Monte Carlo integration."""
        # Estimate E[X²] for X ~ N(0, 1)
        def f(x):
            return x**2
        
        dist = stats.norm(0, 1)
        estimate, std_err = monte_carlo_integration(f, dist, n_samples=10000)
        
        # True value is 1
        assert abs(estimate - 1.0) < 3 * std_err
    
    def test_importance_sampling(self):
        """Test importance sampling."""
        # Estimate E[f(X)] for X ~ N(5, 1) using proposal N(0, 2)
        def f(x):
            return x
        
        target = stats.norm(5, 1)
        proposal = stats.norm(0, 2)
        
        estimate, std_err = importance_sampling(f, target, proposal, n_samples=10000)
        
        # True value is 5
        assert abs(estimate - 5.0) < 3 * std_err


def test_convergence_demonstrations():
    """Test convergence mode demonstrations."""
    demonstrate_convergence_modes(n_samples=1000)
    # Should produce visualizations without errors


def test_concentration_inequalities():
    """Test concentration inequality verification."""
    verify_concentration_inequalities(n_samples=10000)
    # Should verify inequalities without errors


def test_empirical_process_theory():
    """Test empirical process demonstrations."""
    empirical_process_theory_demo(n_samples=1000)
    # Should demonstrate theorems without errors


def test_information_theory():
    """Test information theory connections."""
    information_theory_connections()
    # Should compute information measures without errors


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])