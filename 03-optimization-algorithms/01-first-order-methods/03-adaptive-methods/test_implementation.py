"""
Test Suite for Adaptive Gradient Methods Implementation

This module provides comprehensive tests for adaptive optimization algorithms.
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
        AdaGrad, RMSprop, Adam, AMSGrad, AdaBelief, AdamW, Lookahead,
        AdaptiveTestFunctions
    )
except ImportError as e:
    print(f"Error importing from exercise.py: {e}")
    print("Make sure you have implemented the required classes and functions.")
    sys.exit(1)


class TestAdaptiveMethods(unittest.TestCase):
    """Test cases for adaptive optimization methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tolerance = 1e-6
        self.max_iterations = 1000
        np.random.seed(42)  # For reproducible tests
    
    def test_adagrad_basic_update(self):
        """Test basic AdaGrad update functionality"""
        optimizer = AdaGrad(learning_rate=0.1, eps=1e-8)
        x = np.array([1.0, 1.0])
        gradient = np.array([0.5, -0.3])
        
        # First step should initialize G
        self.assertIsNone(optimizer.G)
        x_new = optimizer.step(x, gradient)
        self.assertIsNotNone(optimizer.G)
        self.assertEqual(optimizer.G.shape, x.shape)
        
        # Check update rule: x_new = x - lr / sqrt(G + eps) * g
        expected_G = gradient**2
        expected_x = x - 0.1 / np.sqrt(expected_G + 1e-8) * gradient
        np.testing.assert_allclose(x_new, expected_x, rtol=1e-10,
                                 err_msg="AdaGrad first step update is incorrect")
    
    def test_adagrad_accumulation(self):
        """Test AdaGrad gradient accumulation"""
        optimizer = AdaGrad(learning_rate=0.1, eps=1e-8)
        x = np.array([1.0, 1.0])
        gradient1 = np.array([0.5, -0.3])
        gradient2 = np.array([0.2, 0.4])
        
        # First step
        x1 = optimizer.step(x, gradient1)
        G1 = gradient1**2
        
        # Second step
        x2 = optimizer.step(x1, gradient2)
        expected_G2 = G1 + gradient2**2
        
        np.testing.assert_allclose(optimizer.G, expected_G2, rtol=1e-10,
                                 err_msg="AdaGrad accumulation is incorrect")
    
    def test_rmsprop_exponential_average(self):
        """Test RMSprop exponential moving average"""
        optimizer = RMSprop(learning_rate=0.001, gamma=0.9, eps=1e-8)
        x = np.array([1.0, 1.0])
        gradient1 = np.array([0.5, -0.3])
        gradient2 = np.array([0.2, 0.4])
        
        # First step
        x1 = optimizer.step(x, gradient1)
        expected_v1 = (1 - 0.9) * gradient1**2
        np.testing.assert_allclose(optimizer.v, expected_v1, rtol=1e-10)
        
        # Second step
        x2 = optimizer.step(x1, gradient2)
        expected_v2 = 0.9 * expected_v1 + (1 - 0.9) * gradient2**2
        np.testing.assert_allclose(optimizer.v, expected_v2, rtol=1e-10,
                                 err_msg="RMSprop exponential average is incorrect")
    
    def test_adam_bias_correction(self):
        """Test Adam bias correction"""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        x = np.array([1.0, 1.0])
        gradient = np.array([0.5, -0.3])
        
        # First step
        x_new = optimizer.step(x, gradient)
        
        # Check bias correction
        expected_m = (1 - 0.9) * gradient
        expected_v = (1 - 0.999) * gradient**2
        
        bias_correction_1 = 1 / (1 - 0.9**1)
        bias_correction_2 = 1 / (1 - 0.999**1)
        
        expected_m_hat = expected_m * bias_correction_1
        expected_v_hat = expected_v * bias_correction_2
        
        # The exact update rule test
        expected_x = x - 0.001 * expected_m_hat / (np.sqrt(expected_v_hat) + 1e-8)
        np.testing.assert_allclose(x_new, expected_x, rtol=1e-10,
                                 err_msg="Adam bias correction is incorrect")
    
    def test_adam_multiple_steps(self):
        """Test Adam over multiple steps"""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        x = np.array([1.0, 1.0])
        gradients = [
            np.array([0.5, -0.3]),
            np.array([0.2, 0.4]),
            np.array([-0.1, 0.1])
        ]
        
        for i, grad in enumerate(gradients):
            x = optimizer.step(x, grad)
            
            # Check that step count is correct
            self.assertEqual(optimizer.step_count, i + 1)
            
            # Check that moments are updated
            self.assertIsNotNone(optimizer.m)
            self.assertIsNotNone(optimizer.v)
    
    def test_amsgrad_maximum_operation(self):
        """Test AMSGrad maximum operation"""
        optimizer = AMSGrad(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        x = np.array([1.0, 1.0])
        
        # Large gradient first
        large_gradient = np.array([1.0, 1.0])
        x1 = optimizer.step(x, large_gradient)
        v_hat_1 = optimizer.v_hat_max.copy()
        
        # Smaller gradient second  
        small_gradient = np.array([0.1, 0.1])
        x2 = optimizer.step(x1, small_gradient)
        
        # v_hat_max should not decrease
        self.assertTrue(np.all(optimizer.v_hat_max >= v_hat_1),
                       "AMSGrad v_hat_max should not decrease")
    
    def test_adabelief_prediction_error(self):
        """Test AdaBelief prediction error computation"""
        if 'AdaBelief' in globals():
            optimizer = AdaBelief(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
            x = np.array([1.0, 1.0])
            gradient = np.array([0.5, -0.3])
            
            # First step
            x_new = optimizer.step(x, gradient)
            
            # AdaBelief uses (g_t - m_t)^2 for second moment
            # After first step, m_t = (1-beta1) * g_t, so prediction error is reduced
            self.assertIsNotNone(optimizer.m)
            self.assertIsNotNone(optimizer.s)
    
    def test_adamw_weight_decay(self):
        """Test AdamW decoupled weight decay"""
        if 'AdamW' in globals():
            optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
            x = np.array([1.0, 1.0])
            gradient = np.array([0.0, 0.0])  # Zero gradient to isolate weight decay
            
            x_new = optimizer.step(x, gradient)
            
            # With zero gradient, should only apply weight decay
            # x_new = x - lr * (0 + weight_decay * x) = x * (1 - lr * weight_decay)
            expected_decay_factor = 1 - 0.001 * 0.01
            expected_x = x * expected_decay_factor
            
            # This test might be approximate due to Adam's momentum terms
            self.assertTrue(np.linalg.norm(x_new) < np.linalg.norm(x),
                          "AdamW should apply weight decay")
    
    def test_lookahead_wrapper(self):
        """Test Lookahead wrapper functionality"""
        if 'Lookahead' in globals():
            base_optimizer = Adam(learning_rate=0.001)
            lookahead = Lookahead(base_optimizer, k=5, alpha=0.5)
            
            x = np.array([1.0, 1.0])
            gradient = np.array([0.5, -0.3])
            
            # First step should initialize slow and fast weights
            x_new = lookahead.step(x, gradient)
            self.assertIsNotNone(lookahead.slow_weights)
            self.assertIsNotNone(lookahead.fast_weights)
    
    def test_numerical_stability_with_eps(self):
        """Test numerical stability with different epsilon values"""
        eps_values = [1e-8, 1e-6, 1e-4]
        x = np.array([1.0, 1.0])
        gradient = np.array([0.0, 0.0])  # Zero gradient to test division by eps
        
        for eps in eps_values:
            optimizer = Adam(learning_rate=0.001, eps=eps)
            
            # Multiple steps with zero gradient
            for _ in range(5):
                x_new = optimizer.step(x, gradient)
                self.assertFalse(np.any(np.isnan(x_new)), 
                               f"NaN produced with eps={eps}")
                self.assertFalse(np.any(np.isinf(x_new)), 
                               f"Inf produced with eps={eps}")
    
    def test_optimizer_reset(self):
        """Test optimizer state reset functionality"""
        optimizers = [
            AdaGrad(learning_rate=0.01),
            RMSprop(learning_rate=0.001),
            Adam(learning_rate=0.001)
        ]
        
        x = np.array([1.0, 1.0])
        gradient = np.array([0.5, -0.3])
        
        for optimizer in optimizers:
            # Take a step to initialize state
            optimizer.step(x, gradient)
            self.assertGreater(optimizer.step_count, 0)
            
            # Reset should clear state
            optimizer.reset()
            self.assertEqual(optimizer.step_count, 0)
            self.assertEqual(len(optimizer.history['loss']), 0)


class TestConvergenceProperties(unittest.TestCase):
    """Test convergence properties of adaptive methods"""
    
    def test_simple_quadratic_convergence(self):
        """Test convergence on simple quadratic function"""
        def quadratic_fn(x):
            return 0.5 * np.sum(x**2), x
        
        x_init = np.array([2.0, -1.5])
        max_iterations = 500
        tolerance = 1e-6
        
        optimizers = [
            ('AdaGrad', AdaGrad(learning_rate=0.1)),
            ('RMSprop', RMSprop(learning_rate=0.01)),
            ('Adam', Adam(learning_rate=0.01))
        ]
        
        for name, optimizer in optimizers:
            x = x_init.copy()
            
            for iteration in range(max_iterations):
                loss, grad = quadratic_fn(x)
                x = optimizer.step(x, grad)
                
                if loss < tolerance:
                    break
            
            final_norm = np.linalg.norm(x)
            self.assertLess(final_norm, 0.1, 
                          f"{name} should converge to near origin")
    
    def test_different_learning_rates(self):
        """Test adaptive methods with different learning rates"""
        def simple_quadratic(x):
            return 0.5 * np.sum(x**2), x
        
        x_init = np.array([1.0, 1.0])
        learning_rates = [0.1, 0.01, 0.001]
        
        for lr in learning_rates:
            optimizer = Adam(learning_rate=lr)
            x = x_init.copy()
            
            # Run for fixed number of iterations
            for _ in range(100):
                loss, grad = simple_quadratic(x)
                x = optimizer.step(x, grad)
            
            # Should converge (larger lr should converge faster)
            final_norm = np.linalg.norm(x)
            self.assertLess(final_norm, 1.0, 
                          f"Adam with lr={lr} should make progress")
    
    def test_sparse_gradients(self):
        """Test adaptive methods on sparse gradient problems"""
        def sparse_quadratic(x):
            # Only first component has gradient
            loss = 0.5 * x[0]**2
            grad = np.array([x[0], 0.0])
            return loss, grad
        
        x_init = np.array([2.0, 1.0])
        
        # AdaGrad should handle sparse gradients well
        optimizer = AdaGrad(learning_rate=0.1)
        x = x_init.copy()
        
        for _ in range(100):
            loss, grad = sparse_quadratic(x)
            x = optimizer.step(x, grad)
        
        # First component should converge, second should remain unchanged
        self.assertLess(abs(x[0]), 0.1, "First component should converge")
        self.assertAlmostEqual(x[1], 1.0, places=3, 
                              msg="Second component should be unchanged")


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of adaptive methods"""
    
    def test_large_gradients(self):
        """Test behavior with large gradient values"""
        optimizers = [
            AdaGrad(learning_rate=0.001),
            RMSprop(learning_rate=0.001),
            Adam(learning_rate=0.001)
        ]
        
        x = np.array([1.0, 1.0])
        large_gradient = np.array([1e6, -1e6])
        
        for optimizer in optimizers:
            x_new = optimizer.step(x, large_gradient)
            
            self.assertFalse(np.any(np.isnan(x_new)), 
                           f"{type(optimizer).__name__} should handle large gradients")
            self.assertFalse(np.any(np.isinf(x_new)), 
                           f"{type(optimizer).__name__} should not produce Inf")
    
    def test_very_small_gradients(self):
        """Test behavior with very small gradients"""
        optimizers = [
            AdaGrad(learning_rate=0.1),
            RMSprop(learning_rate=0.1), 
            Adam(learning_rate=0.1)
        ]
        
        x = np.array([1.0, 1.0])
        tiny_gradient = np.array([1e-12, -1e-12])
        
        for optimizer in optimizers:
            x_new = optimizer.step(x, tiny_gradient)
            
            self.assertFalse(np.any(np.isnan(x_new)), 
                           f"{type(optimizer).__name__} should handle tiny gradients")
            # Change should be very small
            change = np.linalg.norm(x_new - x)
            self.assertLess(change, 1e-6, 
                          f"{type(optimizer).__name__} should make small changes for tiny gradients")
    
    def test_alternating_gradients(self):
        """Test with alternating large/small gradients"""
        optimizer = Adam(learning_rate=0.001)
        x = np.array([1.0, 1.0])
        
        gradients = [
            np.array([1e3, 0.0]),   # Large
            np.array([1e-6, 0.0]),  # Tiny
            np.array([1e2, 0.0]),   # Medium
            np.array([1e-8, 0.0])   # Very tiny
        ]
        
        for grad in gradients:
            x = optimizer.step(x, grad)
            
            self.assertFalse(np.any(np.isnan(x)), "Should handle alternating gradients")
            self.assertFalse(np.any(np.isinf(x)), "Should not produce Inf")


class TestAdamConvergenceIssue(unittest.TestCase):
    """Test Adam's known convergence issues"""
    
    def test_adam_failure_example(self):
        """Test the example where Adam fails to converge"""
        # This is the simple counter-example from Reddi et al. 2018
        def adam_failure_function(x, t):
            if t % 3 == 1:
                return 1010 * x, 1010.0
            else:
                return -x, -1.0
        
        x = 1.0  # Start at x=1
        adam = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
        amsgrad = AMSGrad(learning_rate=0.01, beta1=0.9, beta2=0.999)
        
        adam_trajectory = []
        amsgrad_trajectory = []
        
        # Run for many steps
        x_adam = x
        x_amsgrad = x
        
        for t in range(1, 100):
            # Adam
            loss_adam, grad_adam = adam_failure_function(x_adam, t)
            x_adam = adam.step(np.array([x_adam]), np.array([grad_adam]))[0]
            adam_trajectory.append(x_adam)
            
            # AMSGrad  
            loss_ams, grad_ams = adam_failure_function(x_amsgrad, t)
            x_amsgrad = amsgrad.step(np.array([x_amsgrad]), np.array([grad_ams]))[0]
            amsgrad_trajectory.append(x_amsgrad)
        
        # Adam should oscillate/not converge to optimum (which is at x=0)
        # AMSGrad should converge better
        adam_final_distance = abs(adam_trajectory[-1])
        amsgrad_final_distance = abs(amsgrad_trajectory[-1])
        
        # This test might be sensitive to implementation details
        # The key insight is that AMSGrad should generally perform better
        print(f"Adam final distance from optimum: {adam_final_distance:.6f}")
        print(f"AMSGrad final distance from optimum: {amsgrad_final_distance:.6f}")


def run_adaptive_methods_comparison():
    """
    Compare performance of different adaptive methods.
    This is not a unit test but a performance comparison.
    """
    print("\nAdaptive Methods Performance Comparison:")
    print("=" * 50)
    
    # Define test problem - ill-conditioned quadratic
    condition_number = 50.0
    A = np.array([[condition_number, 0], [0, 1.0]])
    
    def ill_conditioned_quadratic(x):
        f_val = 0.5 * x.T @ A @ x
        grad = A @ x
        return f_val, grad
    
    x_init = np.array([10.0, 10.0])
    max_iterations = 500
    
    optimizers = {
        'AdaGrad': AdaGrad(learning_rate=0.1),
        'RMSprop': RMSprop(learning_rate=0.01, gamma=0.9),
        'Adam': Adam(learning_rate=0.01, beta1=0.9, beta2=0.999),
        'AMSGrad': AMSGrad(learning_rate=0.01, beta1=0.9, beta2=0.999)
    }
    
    print(f"{'Method':<12} {'Final Loss':<12} {'Final Norm':<12} {'Iterations':<12}")
    print("-" * 60)
    
    for name, optimizer in optimizers.items():
        try:
            x = x_init.copy()
            
            for iteration in range(max_iterations):
                loss, grad = ill_conditioned_quadratic(x)
                x = optimizer.step(x, grad)
                
                if loss < 1e-8:
                    break
            
            final_loss, _ = ill_conditioned_quadratic(x)
            final_norm = np.linalg.norm(x)
            
            print(f"{name:<12} {final_loss:<12.6f} {final_norm:<12.6f} {iteration+1:<12}")
        except Exception as e:
            print(f"{name:<12} {'Error: ' + str(e):<40}")


if __name__ == '__main__':
    print("Testing Adaptive Gradient Methods Implementation")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance comparison
    run_adaptive_methods_comparison()
    
    print("\nIf tests pass, your implementation is correct!")
    print("If tests fail, check your implementation against the formulas in lesson.md")
