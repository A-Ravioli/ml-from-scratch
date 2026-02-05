"""
Test Suite for Momentum Methods Implementation

This module provides comprehensive tests for momentum-based optimization algorithms.
Run these tests to verify your implementations are correct.

Author: ML From Scratch Curriculum
Date: 2024
"""

import unittest
import numpy as np
import sys
import os

# Add the current directory to the path to import exercise module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from exercise import (
        SGD, ClassicalMomentum, NesterovMomentum, AdaptiveMomentum, 
        QuasiHyperbolicMomentum, TestFunctions, optimize_function
    )
except ImportError as e:
    print(f"Error importing from exercise.py: {e}")
    print("Make sure you have implemented the required classes and functions.")
    sys.exit(1)


class TestMomentumMethods(unittest.TestCase):
    """Test cases for momentum optimization methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tolerance = 1e-6
        self.max_iterations = 1000
        np.random.seed(42)  # For reproducible tests
    
    def test_sgd_basic(self):
        """Test basic SGD functionality"""
        optimizer = SGD(learning_rate=0.1)
        x = np.array([1.0, 1.0])
        gradient = np.array([0.5, -0.3])
        
        x_new = optimizer.step(x, gradient)
        expected = x - 0.1 * gradient
        
        np.testing.assert_allclose(x_new, expected, rtol=1e-10,
                                 err_msg="SGD update rule is incorrect")
    
    def test_classical_momentum_initialization(self):
        """Test classical momentum velocity initialization"""
        optimizer = ClassicalMomentum(learning_rate=0.01, momentum=0.9)
        x = np.array([1.0, 1.0])
        gradient = np.array([0.5, -0.3])
        
        # First step should initialize velocity
        self.assertIsNone(optimizer.velocity)
        x_new = optimizer.step(x, gradient)
        self.assertIsNotNone(optimizer.velocity)
        self.assertEqual(optimizer.velocity.shape, x.shape)
    
    def test_classical_momentum_update(self):
        """Test classical momentum update rule"""
        optimizer = ClassicalMomentum(learning_rate=0.1, momentum=0.9)
        x = np.array([1.0, 1.0])
        gradient1 = np.array([0.5, -0.3])
        gradient2 = np.array([0.2, 0.1])
        
        # First step
        x1 = optimizer.step(x, gradient1)
        expected_v1 = 0.1 * gradient1
        expected_x1 = x - expected_v1
        np.testing.assert_allclose(x1, expected_x1, rtol=1e-10)
        
        # Second step
        x2 = optimizer.step(x1, gradient2)
        expected_v2 = 0.9 * expected_v1 + 0.1 * gradient2
        expected_x2 = x1 - expected_v2
        np.testing.assert_allclose(x2, expected_x2, rtol=1e-10)
    
    def test_nesterov_momentum_update(self):
        """Test Nesterov momentum update rule"""
        optimizer = NesterovMomentum(learning_rate=0.1, momentum=0.9)
        x = np.array([1.0, 1.0])
        
        # Note: In practice, gradient should be computed at lookahead point
        # For testing, we'll use a simple gradient
        gradient = np.array([0.5, -0.3])
        
        x_new = optimizer.step(x, gradient)
        self.assertIsNotNone(optimizer.velocity)
        self.assertEqual(x_new.shape, x.shape)
    
    def test_quadratic_convergence(self):
        """Test convergence on a simple quadratic function"""
        # Simple quadratic: f(x) = 0.5 * x^T * x
        def quadratic_fn(x):
            return 0.5 * np.sum(x**2), x
        
        x_init = np.array([2.0, -1.5])
        
        # Test SGD
        sgd = SGD(learning_rate=0.1)
        result = optimize_function(sgd, quadratic_fn, x_init, 
                                 max_iterations=100, tolerance=1e-6)
        final_norm = np.linalg.norm(result['x_final'])
        self.assertLess(final_norm, 0.1, "SGD should converge to near origin")
        
        # Test Classical Momentum
        momentum = ClassicalMomentum(learning_rate=0.1, momentum=0.9)
        result = optimize_function(momentum, quadratic_fn, x_init,
                                 max_iterations=100, tolerance=1e-6)
        final_norm = np.linalg.norm(result['x_final'])
        self.assertLess(final_norm, 0.1, "Momentum should converge to near origin")
    
    def test_momentum_coefficient_bounds(self):
        """Test that momentum coefficient is properly bounded"""
        # Test valid momentum values
        valid_momentums = [0.0, 0.5, 0.9, 0.99, 0.999]
        for beta in valid_momentums:
            try:
                optimizer = ClassicalMomentum(learning_rate=0.01, momentum=beta)
                self.assertEqual(optimizer.momentum, beta)
            except Exception as e:
                self.fail(f"Valid momentum {beta} should not raise exception: {e}")
    
    def test_optimizer_reset(self):
        """Test optimizer state reset functionality"""
        optimizer = ClassicalMomentum(learning_rate=0.01, momentum=0.9)
        x = np.array([1.0, 1.0])
        gradient = np.array([0.5, -0.3])
        
        # Take a step to initialize state
        optimizer.step(x, gradient)
        self.assertIsNotNone(optimizer.velocity)
        
        # Reset should clear state
        optimizer.reset()
        self.assertIsNone(optimizer.velocity)
        self.assertEqual(len(optimizer.history['loss']), 0)
    
    def test_ill_conditioned_problem(self):
        """Test momentum methods on ill-conditioned quadratic"""
        # Create ill-conditioned quadratic: f(x) = 0.5 * x^T * A * x
        # where A has large condition number
        condition_number = 100.0
        A = np.array([[condition_number, 0], [0, 1.0]])
        
        def ill_conditioned_quadratic(x):
            return 0.5 * x.T @ A @ x, A @ x
        
        x_init = np.array([10.0, 10.0])
        
        # SGD should struggle more than momentum methods
        sgd = SGD(learning_rate=0.01)
        sgd_result = optimize_function(sgd, ill_conditioned_quadratic, x_init,
                                     max_iterations=500, tolerance=1e-4)
        
        momentum = ClassicalMomentum(learning_rate=0.01, momentum=0.9)
        momentum_result = optimize_function(momentum, ill_conditioned_quadratic, x_init,
                                          max_iterations=500, tolerance=1e-4)
        
        # Momentum should converge faster (fewer iterations)
        # This is not always guaranteed but generally true for ill-conditioned problems
        sgd_norm = np.linalg.norm(sgd_result['x_final'])
        momentum_norm = np.linalg.norm(momentum_result['x_final'])
        
        # Both should converge, but we mainly test that momentum doesn't fail
        self.assertLess(momentum_norm, 1.0, "Momentum should converge reasonably well")
    
    def test_different_momentum_values(self):
        """Test behavior with different momentum coefficient values"""
        def simple_quadratic(x):
            return 0.5 * np.sum(x**2), x
        
        x_init = np.array([1.0, 1.0])
        momentum_values = [0.0, 0.5, 0.9, 0.95]
        
        results = {}
        for beta in momentum_values:
            optimizer = ClassicalMomentum(learning_rate=0.1, momentum=beta)
            result = optimize_function(optimizer, simple_quadratic, x_init,
                                     max_iterations=50, tolerance=1e-8)
            results[beta] = result
        
        # All should converge
        for beta, result in results.items():
            final_norm = np.linalg.norm(result['x_final'])
            self.assertLess(final_norm, 0.1, 
                          f"Momentum with β={beta} should converge")
    
    def test_quasi_hyperbolic_momentum(self):
        """Test QHM update rule"""
        if 'QuasiHyperbolicMomentum' in globals():
            optimizer = QuasiHyperbolicMomentum(learning_rate=0.01, momentum=0.9, nu=0.7)
            x = np.array([1.0, 1.0])
            gradient = np.array([0.5, -0.3])
            
            x_new = optimizer.step(x, gradient)
            self.assertEqual(x_new.shape, x.shape)
            self.assertIsNotNone(optimizer.momentum_buffer)
    
    def test_adaptive_momentum(self):
        """Test adaptive momentum functionality"""
        if 'AdaptiveMomentum' in globals():
            optimizer = AdaptiveMomentum(learning_rate=0.01)
            x = np.array([1.0, 1.0])
            gradient = np.array([0.5, -0.3])
            loss = 1.0
            
            x_new = optimizer.step(x, gradient, loss)
            self.assertEqual(x_new.shape, x.shape)
            
            # Test with decreasing loss
            x_new2 = optimizer.step(x_new, gradient * 0.8, loss * 0.8)
            self.assertEqual(x_new2.shape, x.shape)


class TestOptimizationUtilities(unittest.TestCase):
    """Test optimization utility functions"""
    
    def test_test_functions(self):
        """Test the provided test functions"""
        x = np.array([1.0, 1.0])
        
        # Test quadratic bowl
        f_val, grad = TestFunctions.quadratic_bowl(x)
        self.assertIsInstance(f_val, (float, np.floating))
        self.assertEqual(grad.shape, x.shape)
        
        # Gradient should be correct for f(x) = 0.5 * x^T * x
        expected_grad = x
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-10)
    
    def test_optimization_loop(self):
        """Test the general optimization loop"""
        def simple_quadratic(x):
            return 0.5 * np.sum(x**2), x
        
        optimizer = SGD(learning_rate=0.1)
        x_init = np.array([1.0, 1.0])
        
        result = optimize_function(optimizer, simple_quadratic, x_init,
                                 max_iterations=100, tolerance=1e-6)
        
        # Check result structure
        self.assertIn('x_final', result)
        self.assertIn('iterations', result)
        self.assertIn('converged', result)
        self.assertIn('history', result)
        
        # Check convergence
        final_norm = np.linalg.norm(result['x_final'])
        self.assertLess(final_norm, 0.1)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of implementations"""
    
    def test_large_gradients(self):
        """Test behavior with large gradient values"""
        optimizer = ClassicalMomentum(learning_rate=0.001, momentum=0.9)
        x = np.array([1.0, 1.0])
        large_gradient = np.array([1e6, -1e6])
        
        x_new = optimizer.step(x, large_gradient)
        
        # Should not produce NaN or Inf
        self.assertFalse(np.any(np.isnan(x_new)), "Large gradients should not produce NaN")
        self.assertFalse(np.any(np.isinf(x_new)), "Large gradients should not produce Inf")
    
    def test_small_gradients(self):
        """Test behavior with very small gradient values"""
        optimizer = ClassicalMomentum(learning_rate=0.1, momentum=0.9)
        x = np.array([1.0, 1.0])
        small_gradient = np.array([1e-10, -1e-10])
        
        x_new = optimizer.step(x, small_gradient)
        
        # Should handle small gradients gracefully
        self.assertFalse(np.any(np.isnan(x_new)), "Small gradients should not produce NaN")
        self.assertTrue(np.allclose(x_new, x, atol=1e-9), "Small gradients should produce small changes")
    
    def test_zero_gradients(self):
        """Test behavior with zero gradients"""
        optimizer = ClassicalMomentum(learning_rate=0.1, momentum=0.9)
        x = np.array([1.0, 1.0])
        zero_gradient = np.zeros_like(x)
        
        # First step with non-zero gradient
        nonzero_gradient = np.array([0.5, -0.3])
        x1 = optimizer.step(x, nonzero_gradient)
        
        # Second step with zero gradient
        x2 = optimizer.step(x1, zero_gradient)
        
        # Should apply momentum from previous step
        self.assertFalse(np.allclose(x1, x2), "Zero gradient should still apply momentum")


def run_performance_comparison():
    """
    Compare performance of different momentum methods.
    This is not a unit test but a performance comparison.
    """
    print("\nPerformance Comparison:")
    print("=" * 40)
    
    # Define test problem
    def rosenbrock_simple(x):
        """Simplified 2D Rosenbrock function"""
        f_val = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        grad = np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])
        return f_val, grad
    
    x_init = np.array([-1.0, 1.0])
    max_iterations = 1000
    
    optimizers = {
        'SGD': SGD(learning_rate=0.001),
        'Classical Momentum': ClassicalMomentum(learning_rate=0.001, momentum=0.9),
        'Nesterov': NesterovMomentum(learning_rate=0.001, momentum=0.9)
    }
    
    print(f"{'Method':<20} {'Final Loss':<12} {'Iterations':<12} {'Final Norm':<12}")
    print("-" * 60)
    
    for name, optimizer in optimizers.items():
        try:
            result = optimize_function(optimizer, rosenbrock_simple, x_init,
                                     max_iterations=max_iterations, tolerance=1e-8)
            
            final_loss, _ = rosenbrock_simple(result['x_final'])
            final_norm = np.linalg.norm(result['x_final'] - np.array([1.0, 1.0]))
            
            print(f"{name:<20} {final_loss:<12.6f} {result['iterations']:<12} {final_norm:<12.6f}")
        except Exception as e:
            print(f"{name:<20} {'Error: ' + str(e):<12}")


if __name__ == '__main__':
    print("Testing Momentum Methods Implementation")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance comparison
    run_performance_comparison()
    
    print("\nIf tests pass, your implementation is correct!")
    print("If tests fail, check your implementation against the formulas in lesson.md")
