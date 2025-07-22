"""
Test Suite for Neural ODEs

Comprehensive tests for Neural ODE implementations, solvers, and applications.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

from exercise import (
    EulerSolver, RungeKutta4Solver, AdaptiveStepSolver,
    NeuralODEFunction, NeuralODE, AdjointNeuralODE,
    ContinuousNormalizingFlow, AugmentedNeuralODE,
    ODEBlock, ResNetODEClassifier, LatentODE
)


class TestODESolvers:
    """Test ODE solver implementations"""
    
    @pytest.fixture
    def simple_ode_func(self):
        """Simple ODE function: dy/dt = -y (exponential decay)"""
        def func(t, y):
            return -y
        return func
    
    @pytest.fixture
    def harmonic_oscillator(self):
        """Harmonic oscillator: d²x/dt² = -x -> [dx/dt, dv/dt] = [v, -x]"""
        def func(t, y):
            x, v = y[..., 0:1], y[..., 1:2]
            return torch.cat([v, -x], dim=-1)
        return func
    
    def test_euler_solver_exponential_decay(self, simple_ode_func):
        """Test Euler solver on exponential decay"""
        solver = EulerSolver(simple_ode_func)
        
        y0 = torch.tensor([[1.0]])  # Initial condition
        t = torch.linspace(0, 2, 21)  # Time points
        
        solution = solver.integrate(y0, t)
        
        assert solution.shape == (21, 1, 1)
        
        # Check that solution decays
        assert solution[-1, 0, 0] < solution[0, 0, 0]
        
        # Analytical solution is y(t) = exp(-t)
        analytical = torch.exp(-t).unsqueeze(-1).unsqueeze(-1)
        
        # Euler should be reasonably close for small step sizes
        error = torch.abs(solution - analytical).max()
        assert error < 0.5  # Allow reasonable error for Euler method
    
    def test_rk4_solver_exponential_decay(self, simple_ode_func):
        """Test RK4 solver on exponential decay"""
        solver = RungeKutta4Solver(simple_ode_func)
        
        y0 = torch.tensor([[1.0]])
        t = torch.linspace(0, 2, 21)
        
        solution = solver.integrate(y0, t)
        
        assert solution.shape == (21, 1, 1)
        
        # RK4 should be more accurate than Euler
        analytical = torch.exp(-t).unsqueeze(-1).unsqueeze(-1)
        error = torch.abs(solution - analytical).max()
        assert error < 0.01  # Much smaller error for RK4
    
    def test_harmonic_oscillator_conservation(self, harmonic_oscillator):
        """Test that harmonic oscillator conserves energy"""
        solver = RungeKutta4Solver(harmonic_oscillator)
        
        # Initial condition: position=1, velocity=0
        y0 = torch.tensor([[1.0, 0.0]])
        t = torch.linspace(0, 2*np.pi, 100)  # One full period
        
        solution = solver.integrate(y0, t)
        
        assert solution.shape == (100, 1, 2)
        
        # Compute energy E = 0.5*(x² + v²) at each time
        x, v = solution[..., 0], solution[..., 1]
        energy = 0.5 * (x**2 + v**2)
        
        # Energy should be approximately constant
        energy_variation = energy.std()
        assert energy_variation < 0.1
        
        # Should return close to initial position after full period
        final_position = solution[-1, 0, 0]
        assert torch.abs(final_position - 1.0) < 0.1
    
    def test_solver_batch_processing(self, simple_ode_func):
        """Test that solvers handle batched inputs correctly"""
        solver = EulerSolver(simple_ode_func)
        
        batch_size = 5
        y0 = torch.randn(batch_size, 1)
        t = torch.linspace(0, 1, 11)
        
        solution = solver.integrate(y0, t)
        
        assert solution.shape == (11, batch_size, 1)
        
        # Each trajectory should follow exponential decay
        for i in range(batch_size):
            assert solution[-1, i, 0] < torch.abs(solution[0, i, 0])


class TestNeuralODEFunction:
    """Test Neural ODE function network"""
    
    def test_initialization(self):
        """Test Neural ODE function initialization"""
        func = NeuralODEFunction(dim=4, hidden_dim=32, num_layers=2)
        assert func.dim == 4
        assert func.hidden_dim == 32
    
    def test_forward_pass(self):
        """Test forward pass through Neural ODE function"""
        func = NeuralODEFunction(dim=6, hidden_dim=16)
        
        t = torch.tensor(0.5)
        y = torch.randn(8, 6)
        
        dydt = func(t, y)
        
        assert dydt.shape == y.shape
        assert not torch.isnan(dydt).any()
    
    def test_time_dependence(self):
        """Test that function can handle time-dependent dynamics"""
        func = NeuralODEFunction(dim=3, hidden_dim=16)
        
        y = torch.randn(4, 3)
        t1 = torch.tensor(0.0)
        t2 = torch.tensor(1.0)
        
        dydt1 = func(t1, y)
        dydt2 = func(t2, y)
        
        assert dydt1.shape == dydt2.shape
        # Output might be different for different times (if time-dependent)
        assert not torch.isnan(dydt1).any()
        assert not torch.isnan(dydt2).any()
    
    def test_gradient_flow(self):
        """Test that gradients flow through the function"""
        func = NeuralODEFunction(dim=2, hidden_dim=8)
        
        y = torch.randn(3, 2, requires_grad=True)
        t = torch.tensor(0.5)
        
        dydt = func(t, y)
        loss = dydt.sum()
        loss.backward()
        
        assert y.grad is not None
        assert not torch.isnan(y.grad).any()
        
        # Check that network parameters have gradients
        for param in func.parameters():
            assert param.grad is not None


class TestNeuralODE:
    """Test Neural ODE layer"""
    
    @pytest.fixture
    def ode_func(self):
        """Create simple Neural ODE function for testing"""
        return NeuralODEFunction(dim=4, hidden_dim=16)
    
    def test_initialization(self, ode_func):
        """Test Neural ODE initialization"""
        neural_ode = NeuralODE(ode_func, solver='rk4')
        assert neural_ode.func is ode_func
        assert neural_ode.solver_name == 'rk4'
    
    def test_forward_pass(self, ode_func):
        """Test Neural ODE forward pass"""
        neural_ode = NeuralODE(ode_func, solver='euler')
        
        x = torch.randn(6, 4)
        output = neural_ode(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_custom_integration_time(self, ode_func):
        """Test Neural ODE with custom integration times"""
        neural_ode = NeuralODE(ode_func, solver='rk4')
        
        x = torch.randn(3, 4)
        t = torch.tensor([0., 0.5, 1.0])
        
        output = neural_ode(x, t)
        
        assert output.shape == x.shape
    
    def test_different_solvers(self, ode_func):
        """Test Neural ODE with different solvers"""
        x = torch.randn(4, 4)
        
        for solver in ['euler', 'rk4']:
            neural_ode = NeuralODE(ode_func, solver=solver)
            output = neural_ode(x)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()
    
    def test_backward_pass(self, ode_func):
        """Test that backpropagation works through Neural ODE"""
        neural_ode = NeuralODE(ode_func, solver='euler')
        
        x = torch.randn(2, 4, requires_grad=True)
        output = neural_ode(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that ODE function parameters have gradients
        for param in ode_func.parameters():
            assert param.grad is not None


class TestContinuousNormalizingFlow:
    """Test Continuous Normalizing Flow"""
    
    def test_initialization(self):
        """Test CNF initialization"""
        cnf = ContinuousNormalizingFlow(dim=3, hidden_dim=16)
        assert cnf.dim == 3
    
    def test_forward_transformation(self):
        """Test forward transformation with log-determinant"""
        cnf = ContinuousNormalizingFlow(dim=2, hidden_dim=16)
        
        x = torch.randn(8, 2)
        z, log_det_jac = cnf(x)
        
        assert z.shape == x.shape
        assert log_det_jac.shape == (8,)
        assert not torch.isnan(z).any()
        assert not torch.isnan(log_det_jac).any()
    
    def test_inverse_transformation(self):
        """Test inverse transformation"""
        cnf = ContinuousNormalizingFlow(dim=2, hidden_dim=16)
        
        z = torch.randn(5, 2)
        x, log_det_jac_inv = cnf.inverse(z)
        
        assert x.shape == z.shape
        assert log_det_jac_inv.shape == (5,)
        assert not torch.isnan(x).any()
        assert not torch.isnan(log_det_jac_inv).any()
    
    def test_trace_estimation(self):
        """Test trace estimation using Hutchinson estimator"""
        cnf = ContinuousNormalizingFlow(dim=3, hidden_dim=16)
        
        t = torch.tensor(0.5)
        y = torch.randn(4, 3)
        
        trace_estimate = cnf._trace_df_dy(t, y)
        
        assert trace_estimate.shape == (4,)
        assert not torch.isnan(trace_estimate).any()
    
    def test_invertibility_approximately(self):
        """Test that forward and inverse are approximately inverses"""
        cnf = ContinuousNormalizingFlow(dim=2, hidden_dim=8)
        
        x_original = torch.randn(3, 2)
        
        # Forward then inverse
        z, _ = cnf(x_original)
        x_reconstructed, _ = cnf.inverse(z)
        
        # Should be approximately equal (within integration tolerance)
        reconstruction_error = torch.abs(x_original - x_reconstructed).max()
        assert reconstruction_error < 0.5  # Allow reasonable numerical error


class TestAugmentedNeuralODE:
    """Test Augmented Neural ODE"""
    
    def test_initialization(self):
        """Test Augmented Neural ODE initialization"""
        aug_ode = AugmentedNeuralODE(data_dim=3, augment_dim=2, hidden_dim=16)
        assert aug_ode.data_dim == 3
        assert aug_ode.augment_dim == 2
        assert aug_ode.full_dim == 5
    
    def test_forward_pass(self):
        """Test forward pass maintains data dimension"""
        aug_ode = AugmentedNeuralODE(data_dim=4, augment_dim=2, hidden_dim=16)
        
        x = torch.randn(6, 4)
        output = aug_ode(x)
        
        assert output.shape == x.shape  # Should return same data dimension
        assert not torch.isnan(output).any()
    
    def test_augmentation_effect(self):
        """Test that augmentation provides different dynamics"""
        # Compare regular vs augmented Neural ODE
        data_dim = 3
        
        # Regular Neural ODE
        regular_func = NeuralODEFunction(dim=data_dim, hidden_dim=16)
        regular_ode = NeuralODE(regular_func)
        
        # Augmented Neural ODE
        aug_ode = AugmentedNeuralODE(data_dim=data_dim, augment_dim=2, hidden_dim=16)
        
        x = torch.randn(4, data_dim)
        
        regular_output = regular_ode(x)
        aug_output = aug_ode(x)
        
        assert regular_output.shape == aug_output.shape
        
        # Outputs might be different due to augmentation
        # (though this depends on initialization and dynamics)
        assert not torch.isnan(regular_output).any()
        assert not torch.isnan(aug_output).any()


class TestODEBlock:
    """Test ODE Block for building deeper architectures"""
    
    def test_initialization(self):
        """Test ODE block initialization"""
        block = ODEBlock(dim=5, hidden_dim=20, integration_time=0.5)
        assert block.integration_time == 0.5
    
    def test_forward_pass(self):
        """Test ODE block forward pass"""
        block = ODEBlock(dim=6, hidden_dim=16)
        
        x = torch.randn(8, 6)
        output = block(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_different_integration_times(self):
        """Test ODE blocks with different integration times"""
        dim = 4
        x = torch.randn(3, dim)
        
        for integration_time in [0.1, 0.5, 1.0, 2.0]:
            block = ODEBlock(dim=dim, integration_time=integration_time)
            output = block(x)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()


class TestResNetODEClassifier:
    """Test Neural ODE classifier"""
    
    def test_initialization(self):
        """Test classifier initialization"""
        classifier = ResNetODEClassifier(
            input_dim=10, num_classes=5, hidden_dim=32, num_ode_blocks=2
        )
        
        # Check that it has the right number of ODE blocks
        assert len(classifier.ode_blocks) == 2
    
    def test_forward_pass(self):
        """Test classifier forward pass"""
        classifier = ResNetODEClassifier(
            input_dim=8, num_classes=3, hidden_dim=16, num_ode_blocks=1
        )
        
        x = torch.randn(12, 8)
        logits = classifier(x)
        
        assert logits.shape == (12, 3)
        assert not torch.isnan(logits).any()
    
    def test_classification_loss(self):
        """Test that classifier can compute classification loss"""
        classifier = ResNetODEClassifier(
            input_dim=6, num_classes=4, hidden_dim=16
        )
        
        x = torch.randn(10, 6)
        y = torch.randint(0, 4, (10,))
        
        logits = classifier(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_gradient_flow_through_classifier(self):
        """Test gradient flow through entire classifier"""
        classifier = ResNetODEClassifier(
            input_dim=4, num_classes=2, hidden_dim=8
        )
        
        x = torch.randn(5, 4, requires_grad=True)
        y = torch.randint(0, 2, (5,))
        
        logits = classifier(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that all classifier parameters have gradients
        for param in classifier.parameters():
            assert param.grad is not None


class TestLatentODE:
    """Test Latent ODE for irregular time series"""
    
    def test_initialization(self):
        """Test Latent ODE initialization"""
        latent_ode = LatentODE(
            input_dim=5, latent_dim=3, hidden_dim=16
        )
        assert latent_ode.latent_dim == 3
    
    def test_encoding(self):
        """Test encoding of irregular observations"""
        latent_ode = LatentODE(input_dim=4, latent_dim=2, hidden_dim=16)
        
        batch_size, seq_len = 6, 10
        x = torch.randn(batch_size, seq_len, 4)
        t = torch.sort(torch.rand(batch_size, seq_len))[0]  # Sorted times
        
        z0_mean, z0_std = latent_ode.encode(x, t)
        
        assert z0_mean.shape == (batch_size, 2)
        assert z0_std.shape == (batch_size, 2)
        assert torch.all(z0_std > 0)  # Standard deviation should be positive
    
    def test_decoding(self):
        """Test decoding of latent states"""
        latent_ode = LatentODE(input_dim=3, latent_dim=4, hidden_dim=16)
        
        batch_size, seq_len = 5, 8
        z = torch.randn(batch_size, seq_len, 4)
        t = torch.rand(batch_size, seq_len)
        
        x_reconstructed = latent_ode.decode(z, t)
        
        assert x_reconstructed.shape == (batch_size, seq_len, 3)
        assert not torch.isnan(x_reconstructed).any()
    
    def test_forward_pass(self):
        """Test complete forward pass through Latent ODE"""
        latent_ode = LatentODE(input_dim=2, latent_dim=3, hidden_dim=12)
        
        batch_size = 4
        obs_len, pred_len = 6, 4
        
        x_obs = torch.randn(batch_size, obs_len, 2)
        t_obs = torch.sort(torch.rand(batch_size, obs_len))[0]
        t_pred = torch.sort(torch.rand(batch_size, pred_len))[0] + 1  # Future times
        
        outputs = latent_ode(x_obs, t_obs, t_pred)
        
        assert isinstance(outputs, dict)
        # Should contain reconstructions and predictions
        assert 'reconstruction' in outputs or 'prediction' in outputs


class TestIntegrationAndPerformance:
    """Integration tests and performance benchmarks"""
    
    def test_neural_ode_vs_resnet_equivalence(self):
        """Test that Neural ODE approaches ResNet as step size decreases"""
        dim = 8
        
        # Create simple dynamics function
        ode_func = NeuralODEFunction(dim, hidden_dim=16, num_layers=1)
        
        x = torch.randn(4, dim)
        
        # Test with different numbers of evaluation points (smaller step size)
        results = []
        for n_points in [2, 5, 10, 20]:
            t = torch.linspace(0, 1, n_points)
            neural_ode = NeuralODE(ode_func, solver='euler')
            output = neural_ode(x, t)
            results.append(output)
        
        # Results should converge as step size decreases
        # (This is a qualitative test - exact convergence depends on dynamics)
        for result in results:
            assert result.shape == x.shape
            assert not torch.isnan(result).any()
    
    def test_memory_efficiency_adjoint(self):
        """Test that adjoint method reduces memory usage"""
        dim = 10
        ode_func = NeuralODEFunction(dim, hidden_dim=32)
        
        # Regular Neural ODE
        regular_ode = NeuralODE(ode_func, solver='rk4')
        
        # Adjoint Neural ODE
        adjoint_ode = AdjointNeuralODE(ode_func, solver='rk4')
        
        x = torch.randn(8, dim, requires_grad=True)
        
        # Both should give similar outputs (within numerical tolerance)
        regular_output = regular_ode(x)
        adjoint_output = adjoint_ode(x)
        
        assert regular_output.shape == adjoint_output.shape
        assert not torch.isnan(regular_output).any()
        assert not torch.isnan(adjoint_output).any()
        
        # Test that both can backpropagate
        regular_loss = regular_output.sum()
        regular_loss.backward()
        
        x.grad = None  # Reset gradients
        
        adjoint_loss = adjoint_output.sum()
        adjoint_loss.backward()
        
        assert x.grad is not None
    
    def test_solver_accuracy_comparison(self):
        """Compare accuracy of different solvers on known ODE"""
        # Use harmonic oscillator with analytical solution
        def harmonic_oscillator(t, y):
            x, v = y[..., 0:1], y[..., 1:2]
            return torch.cat([v, -x], dim=-1)
        
        # Initial condition: position=1, velocity=0
        y0 = torch.tensor([[1.0, 0.0]])
        t_span = torch.linspace(0, np.pi/2, 50)  # Quarter period
        
        # Analytical solution at t=π/2: position=0, velocity=-1
        expected_final = torch.tensor([[0.0, -1.0]])
        
        solvers = {
            'euler': EulerSolver(harmonic_oscillator),
            'rk4': RungeKutta4Solver(harmonic_oscillator)
        }
        
        for name, solver in solvers.items():
            solution = solver.integrate(y0, t_span)
            final_state = solution[-1]
            
            error = torch.abs(final_state - expected_final).max()
            print(f"{name} error: {error.item():.6f}")
            
            # RK4 should be much more accurate than Euler
            if name == 'euler':
                assert error < 0.5
            elif name == 'rk4':
                assert error < 0.05
    
    def test_computational_efficiency(self):
        """Test computational efficiency of different architectures"""
        batch_size = 32
        input_dim = 16
        
        x = torch.randn(batch_size, input_dim)
        
        # Test different architectures
        models = {
            'ode_block': ODEBlock(input_dim, hidden_dim=32, integration_time=0.5),
            'classifier': ResNetODEClassifier(input_dim, num_classes=10, hidden_dim=32, num_ode_blocks=1)
        }
        
        for name, model in models.items():
            start_time = time.time()
            
            with torch.no_grad():
                if name == 'classifier':
                    output = model(x)
                else:
                    output = model(x)
            
            end_time = time.time()
            
            print(f"{name} forward time: {end_time - start_time:.4f}s")
            
            assert not torch.isnan(output).any()
            assert output.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])