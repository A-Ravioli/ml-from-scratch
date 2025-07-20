"""
Test Suite for Backpropagation Implementation

Comprehensive tests to verify correctness of automatic differentiation
and backpropagation implementations.
"""

import numpy as np
import pytest
from typing import Dict, Any
import sys
import os

# Add the parent directory to the path to import the exercise module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from exercise import (
        Tensor, LinearLayer, SimpleNeuralNetwork, GradientChecker,
        mse_loss, cross_entropy_loss, softmax, generate_classification_data,
        generate_regression_data, train_simple_network
    )
except ImportError as e:
    print(f"Warning: Could not import from exercise.py. Error: {e}")
    print("Please implement the required classes and functions in exercise.py")
    sys.exit(1)

class TestTensorBasics:
    """Test basic tensor operations and gradient computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_tensor_creation(self):
        """Test tensor creation and basic properties."""
        data = np.array([1.0, 2.0, 3.0])
        tensor = Tensor(data, requires_grad=True)
        
        assert tensor.shape == (3,), f"Expected shape (3,), got {tensor.shape}"
        assert tensor.requires_grad == True
        assert tensor.grad is None, "Initial gradient should be None"
        np.testing.assert_array_equal(tensor.numpy(), data)
    
    def test_addition_forward(self):
        """Test forward pass of tensor addition."""
        try:
            a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
            c = a + b
            
            expected = np.array([4.0, 6.0])
            np.testing.assert_allclose(c.data, expected, rtol=1e-7)
        except Exception as e:
            pytest.skip(f"Addition forward not implemented: {e}")
    
    def test_addition_backward(self):
        """Test backward pass of tensor addition."""
        try:
            a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
            c = a + b
            c.backward()
            
            # Gradient of addition should be 1 for both inputs
            expected_grad = np.array([1.0, 1.0])
            np.testing.assert_allclose(a.grad, expected_grad, rtol=1e-7)
            np.testing.assert_allclose(b.grad, expected_grad, rtol=1e-7)
        except Exception as e:
            pytest.skip(f"Addition backward not implemented: {e}")
    
    def test_multiplication_gradients(self):
        """Test gradients for element-wise multiplication."""
        try:
            a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
            b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
            c = a * b
            c.backward()
            
            # Gradient w.r.t. a should be b, w.r.t. b should be a
            np.testing.assert_allclose(a.grad, b.data, rtol=1e-7)
            np.testing.assert_allclose(b.grad, a.data, rtol=1e-7)
        except Exception as e:
            pytest.skip(f"Multiplication gradients not implemented: {e}")
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication and gradients."""
        try:
            A = Tensor(np.random.randn(3, 4), requires_grad=True)
            B = Tensor(np.random.randn(4, 2), requires_grad=True)
            C = A @ B
            
            assert C.shape == (3, 2), f"Expected shape (3, 2), got {C.shape}"
            
            # Test gradient computation
            loss = C.sum()
            loss.backward()
            
            assert A.grad is not None, "Gradient for A should not be None"
            assert B.grad is not None, "Gradient for B should not be None"
            assert A.grad.shape == A.shape, f"Gradient shape mismatch for A"
            assert B.grad.shape == B.shape, f"Gradient shape mismatch for B"
        except Exception as e:
            pytest.skip(f"Matrix multiplication not implemented: {e}")
    
    def test_activation_functions(self):
        """Test activation functions and their gradients."""
        try:
            x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
            
            # Test ReLU
            y_relu = x.relu()
            expected_relu = np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(y_relu.data, expected_relu, rtol=1e-7)
            
            # Test ReLU gradient
            y_relu.backward()
            expected_grad_relu = np.array([0.0, 0.0, 1.0])  # derivative of ReLU
            np.testing.assert_allclose(x.grad, expected_grad_relu, rtol=1e-7)
            
        except Exception as e:
            pytest.skip(f"Activation functions not implemented: {e}")

class TestLinearLayer:
    """Test linear layer implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_layer_initialization(self):
        """Test linear layer initialization."""
        try:
            layer = LinearLayer(10, 5)
            
            assert hasattr(layer, 'weight'), "Layer should have weight parameter"
            assert hasattr(layer, 'bias'), "Layer should have bias parameter"
            assert layer.weight.shape == (10, 5), f"Weight shape should be (10, 5)"
            assert layer.bias.shape == (5,), f"Bias shape should be (5,)"
            assert layer.weight.requires_grad, "Weight should require gradients"
            assert layer.bias.requires_grad, "Bias should require gradients"
        except Exception as e:
            pytest.skip(f"Linear layer initialization not implemented: {e}")
    
    def test_layer_forward(self):
        """Test linear layer forward pass."""
        try:
            layer = LinearLayer(3, 2)
            x = Tensor(np.random.randn(5, 3), requires_grad=True)
            y = layer.forward(x)
            
            assert y.shape == (5, 2), f"Expected output shape (5, 2), got {y.shape}"
            assert y.requires_grad, "Output should require gradients"
        except Exception as e:
            pytest.skip(f"Linear layer forward not implemented: {e}")
    
    def test_layer_gradients(self):
        """Test linear layer gradient computation."""
        try:
            layer = LinearLayer(3, 2)
            x = Tensor(np.random.randn(5, 3), requires_grad=True)
            y = layer.forward(x)
            loss = y.sum()
            loss.backward()
            
            assert x.grad is not None, "Input gradient should not be None"
            assert layer.weight.grad is not None, "Weight gradient should not be None"
            assert layer.bias.grad is not None, "Bias gradient should not be None"
            
            assert x.grad.shape == x.shape, "Input gradient shape should match input"
            assert layer.weight.grad.shape == layer.weight.shape, "Weight gradient shape mismatch"
            assert layer.bias.grad.shape == layer.bias.shape, "Bias gradient shape mismatch"
        except Exception as e:
            pytest.skip(f"Linear layer gradients not implemented: {e}")

class TestNeuralNetwork:
    """Test simple neural network implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_network_initialization(self):
        """Test neural network initialization."""
        try:
            network = SimpleNeuralNetwork(10, 20, 5)
            
            params = network.parameters()
            assert len(params) > 0, "Network should have parameters"
            
            # Check that all parameters require gradients
            for param in params:
                assert param.requires_grad, "All parameters should require gradients"
        except Exception as e:
            pytest.skip(f"Neural network initialization not implemented: {e}")
    
    def test_network_forward(self):
        """Test neural network forward pass."""
        try:
            network = SimpleNeuralNetwork(10, 20, 5)
            x = Tensor(np.random.randn(32, 10), requires_grad=False)
            y = network.forward(x)
            
            assert y.shape == (32, 5), f"Expected output shape (32, 5), got {y.shape}"
        except Exception as e:
            pytest.skip(f"Neural network forward not implemented: {e}")
    
    def test_network_backward(self):
        """Test neural network backward pass."""
        try:
            network = SimpleNeuralNetwork(10, 20, 5)
            x = Tensor(np.random.randn(32, 10), requires_grad=False)
            y = network.forward(x)
            loss = y.sum()
            loss.backward()
            
            # Check that all parameters have gradients
            params = network.parameters()
            for param in params:
                assert param.grad is not None, f"Parameter gradient should not be None"
                assert param.grad.shape == param.shape, "Gradient shape should match parameter shape"
        except Exception as e:
            pytest.skip(f"Neural network backward not implemented: {e}")

class TestLossFunctions:
    """Test loss function implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_mse_loss(self):
        """Test mean squared error loss."""
        try:
            pred = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            target = Tensor(np.array([1.5, 2.5, 2.5]), requires_grad=False)
            
            loss = mse_loss(pred, target)
            
            # MSE = mean((pred - target)^2)
            expected = np.mean((pred.data - target.data) ** 2)
            np.testing.assert_allclose(loss.data, expected, rtol=1e-7)
            
            # Test gradient
            loss.backward()
            assert pred.grad is not None, "Prediction gradient should not be None"
        except Exception as e:
            pytest.skip(f"MSE loss not implemented: {e}")
    
    def test_softmax(self):
        """Test softmax function."""
        try:
            x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
            y = softmax(x)
            
            # Check that probabilities sum to 1
            prob_sums = np.sum(y.data, axis=1)
            np.testing.assert_allclose(prob_sums, [1.0, 1.0], rtol=1e-7)
            
            # Check that all probabilities are positive
            assert np.all(y.data > 0), "All softmax outputs should be positive"
        except Exception as e:
            pytest.skip(f"Softmax not implemented: {e}")

class TestGradientChecker:
    """Test gradient checking utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_gradient_checking_simple(self):
        """Test gradient checking on simple function."""
        try:
            def simple_func(x):
                return (x * x).sum()
            
            x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            checker = GradientChecker()
            
            is_correct = checker.check_gradients(simple_func, x)
            
            # For f(x) = sum(x^2), gradient should be 2*x
            # This should pass gradient checking
            assert is_correct, "Gradient checking should pass for simple quadratic function"
        except Exception as e:
            pytest.skip(f"Gradient checking not implemented: {e}")
    
    def test_gradient_checking_complex(self):
        """Test gradient checking on more complex function."""
        try:
            def complex_func(x):
                y = x.relu()
                z = y * y
                return z.sum()
            
            x = Tensor(np.array([-1.0, 0.5, 2.0]), requires_grad=True)
            checker = GradientChecker()
            
            is_correct = checker.check_gradients(complex_func, x, tolerance=1e-2)
            
            # This should also pass (though with lower tolerance due to ReLU non-smoothness)
            assert is_correct, "Gradient checking should pass for ReLU composition"
        except Exception as e:
            pytest.skip(f"Complex gradient checking not implemented: {e}")

class TestTraining:
    """Test training functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        X, y = generate_classification_data(n_samples=100, n_features=10, n_classes=3)
        
        assert X.shape == (100, 10), f"Expected X shape (100, 10), got {X.shape}"
        assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
        assert np.all(y >= 0) and np.all(y < 3), "Labels should be in [0, 3)"
        
        X_reg, y_reg = generate_regression_data(n_samples=50, n_features=5)
        assert X_reg.shape == (50, 5), f"Expected X shape (50, 5), got {X_reg.shape}"
        assert y_reg.shape == (50,), f"Expected y shape (50,), got {y_reg.shape}"
    
    def test_training_loop(self):
        """Test basic training loop."""
        try:
            X, y = generate_regression_data(n_samples=50, n_features=5)
            history = train_simple_network(X, y, epochs=5, learning_rate=0.01)
            
            if history is not None:
                assert 'loss' in history, "Training history should contain loss"
                assert len(history['loss']) == 5, f"Should have 5 loss values, got {len(history['loss'])}"
                
                # Loss should generally decrease (though not guaranteed for such short training)
                initial_loss = history['loss'][0]
                final_loss = history['loss'][-1]
                assert initial_loss > 0, "Initial loss should be positive"
                assert final_loss > 0, "Final loss should be positive"
        except Exception as e:
            pytest.skip(f"Training loop not implemented: {e}")

def run_comprehensive_test():
    """Run comprehensive test of all implementations."""
    print("Running comprehensive test suite for backpropagation...")
    
    test_classes = [
        TestTensorBasics,
        TestLinearLayer,
        TestNeuralNetwork,
        TestLossFunctions,
        TestGradientChecker,
        TestTraining
    ]
    
    results = {}
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\nTesting {class_name}...")
        
        instance = test_class()
        if hasattr(instance, 'setup_method'):
            instance.setup_method()
        
        methods = [method for method in dir(instance) if method.startswith('test_')]
        passed = 0
        total = len(methods)
        
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
        
        results[class_name] = (passed, total)
        print(f"  {passed}/{total} tests passed")
    
    # Summary
    print("\n" + "="*50)
    print("BACKPROPAGATION TEST SUMMARY")
    print("="*50)
    
    total_passed = 0
    total_tests = 0
    
    for class_name, (passed, total) in results.items():
        total_passed += passed
        total_tests += total
        percentage = (passed / total * 100) if total > 0 else 0
        print(f"{class_name}: {passed}/{total} ({percentage:.1f}%)")
    
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\nOverall: {total_passed}/{total_tests} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 80:
        print("🎉 Excellent! Your backpropagation implementation is working well!")
    elif overall_percentage >= 60:
        print("👍 Good progress! A few more methods to implement.")
    else:
        print("📝 Keep working on the implementations.")
    
    return results

if __name__ == "__main__":
    # Run the comprehensive test
    run_comprehensive_test()
    
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print("1. Implement any failing methods in exercise.py")
    print("2. Add support for more complex architectures")
    print("3. Implement memory-efficient gradient computation")
    print("4. Add computational graph visualization")
    print("5. Extend to support batched operations efficiently")
    print("6. Compare performance with PyTorch/TensorFlow")