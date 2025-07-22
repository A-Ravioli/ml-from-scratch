"""
Test suite for Vanilla RNN implementation
"""

import pytest
import numpy as np
import sys
import os

# Add the exercise module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exercise import (VanillaRNNCell, VanillaRNN, RNNLanguageModel, 
                     GradientAnalyzer, RNNTrainer)


class TestVanillaRNNCell:
    """Test vanilla RNN cell implementation"""
    
    def test_initialization(self):
        """Test RNN cell initialization"""
        input_size = 10
        hidden_size = 20
        
        # Test different activations
        for activation in ['tanh', 'relu', 'sigmoid']:
            cell = VanillaRNNCell(input_size, hidden_size, activation)
            
            assert cell.input_size == input_size
            assert cell.hidden_size == hidden_size
            assert cell.activation == activation
            assert cell.W_xh.shape == (hidden_size, input_size)
            assert cell.W_hh.shape == (hidden_size, hidden_size)
            assert cell.b_h.shape == (hidden_size,)
    
    def test_forward_shapes(self):
        """Test forward pass output shapes"""
        input_size = 15
        hidden_size = 25
        batch_size = 8
        
        cell = VanillaRNNCell(input_size, hidden_size)
        
        x = np.random.randn(batch_size, input_size)
        h_prev = np.random.randn(batch_size, hidden_size)
        
        h_new = cell.forward(x, h_prev)
        
        assert h_new.shape == (batch_size, hidden_size)
        assert not np.allclose(h_new, h_prev)  # Should be different
    
    def test_activation_functions(self):
        """Test different activation functions"""
        input_size = 8
        hidden_size = 12
        batch_size = 4
        
        x = np.random.randn(batch_size, input_size) * 2
        h_prev = np.random.randn(batch_size, hidden_size) * 2
        
        # Test tanh activation
        cell_tanh = VanillaRNNCell(input_size, hidden_size, 'tanh')
        h_tanh = cell_tanh.forward(x, h_prev)
        assert np.all((-1 <= h_tanh) & (h_tanh <= 1))  # tanh range
        
        # Test ReLU activation  
        cell_relu = VanillaRNNCell(input_size, hidden_size, 'relu')
        h_relu = cell_relu.forward(x, h_prev)
        assert np.all(h_relu >= 0)  # ReLU non-negative
        
        # Test sigmoid activation
        cell_sigmoid = VanillaRNNCell(input_size, hidden_size, 'sigmoid')
        h_sigmoid = cell_sigmoid.forward(x, h_prev)
        assert np.all((0 <= h_sigmoid) & (h_sigmoid <= 1))  # sigmoid range
        
        # Different activations should produce different outputs
        assert not np.allclose(h_tanh, h_relu, atol=0.1)
        assert not np.allclose(h_tanh, h_sigmoid, atol=0.1)
    
    def test_recurrent_behavior(self):
        """Test that cell exhibits recurrent behavior"""
        input_size = 6
        hidden_size = 10
        batch_size = 3
        
        cell = VanillaRNNCell(input_size, hidden_size)
        
        x = np.random.randn(batch_size, input_size)
        h1 = np.random.randn(batch_size, hidden_size)
        h2 = np.random.randn(batch_size, hidden_size)
        
        # Same input, different hidden states should give different outputs
        out1 = cell.forward(x, h1)
        out2 = cell.forward(x, h2)
        
        assert not np.allclose(out1, out2, atol=0.1)
    
    def test_sigmoid_stability(self):
        """Test sigmoid numerical stability"""
        cell = VanillaRNNCell(10, 10, 'sigmoid')
        
        # Test extreme values
        large_pos = np.array([[100.0] * 10])
        large_neg = np.array([[-100.0] * 10])
        h_zero = np.zeros((1, 10))
        
        sig_pos = cell.forward(large_pos, h_zero)
        sig_neg = cell.forward(large_neg, h_zero)
        
        assert not np.any(np.isnan(sig_pos))
        assert not np.any(np.isnan(sig_neg))
        assert np.all(sig_pos >= 0) and np.all(sig_pos <= 1)
        assert np.all(sig_neg >= 0) and np.all(sig_neg <= 1)


class TestVanillaRNN:
    """Test multi-layer vanilla RNN implementation"""
    
    def test_initialization(self):
        """Test RNN initialization"""
        input_size = 20
        hidden_size = 30
        output_size = 10
        num_layers = 3
        
        # Unidirectional RNN
        rnn = VanillaRNN(input_size, hidden_size, output_size, num_layers)
        assert len(rnn.layers) == num_layers
        assert not hasattr(rnn, 'backward_layers')
        assert rnn.W_ho.shape == (output_size, hidden_size)
        
        # Bidirectional RNN
        birnn = VanillaRNN(input_size, hidden_size, output_size, num_layers, 
                          bidirectional=True)
        assert len(birnn.layers) == num_layers
        assert len(birnn.backward_layers) == num_layers
        assert birnn.W_ho.shape == (output_size, 2 * hidden_size)
    
    def test_forward_with_sequences(self):
        """Test forward pass returning all sequences"""
        input_size = 25
        hidden_size = 35
        output_size = 15
        num_layers = 2
        seq_len = 12
        batch_size = 6
        
        rnn = VanillaRNN(input_size, hidden_size, output_size, num_layers)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        outputs, states = rnn.forward(x, return_sequences=True)
        
        assert outputs.shape == (batch_size, seq_len, output_size)
        assert states.shape == (num_layers, batch_size, hidden_size)
    
    def test_forward_final_only(self):
        """Test forward pass returning final output only"""
        input_size = 20
        hidden_size = 25
        output_size = 8
        seq_len = 15
        batch_size = 4
        
        rnn = VanillaRNN(input_size, hidden_size, output_size, num_layers=1)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        outputs, states = rnn.forward(x, return_sequences=False)
        
        assert outputs.shape == (batch_size, output_size)
        assert states.shape == (1, batch_size, hidden_size)
    
    def test_bidirectional_forward(self):
        """Test bidirectional RNN forward pass"""
        input_size = 18
        hidden_size = 22
        output_size = 5
        seq_len = 10
        batch_size = 3
        
        birnn = VanillaRNN(input_size, hidden_size, output_size, num_layers=2,
                          bidirectional=True)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        outputs, states = birnn.forward(x, return_sequences=True)
        
        assert outputs.shape == (batch_size, seq_len, output_size)
        # 2 layers × 2 directions = 4 state vectors
        assert states.shape == (4, batch_size, hidden_size)
    
    def test_initial_state_handling(self):
        """Test custom initial state handling"""
        input_size = 15
        hidden_size = 20
        output_size = 7
        num_layers = 2
        seq_len = 8
        batch_size = 3
        
        rnn = VanillaRNN(input_size, hidden_size, output_size, num_layers)
        x = np.random.randn(batch_size, seq_len, input_size)
        
        # Test with default initialization (None)
        outputs1, states1 = rnn.forward(x, return_sequences=True)
        
        # Test with custom initialization
        init_states = np.random.randn(num_layers, batch_size, hidden_size) * 0.1
        outputs2, states2 = rnn.forward(x, init_states, return_sequences=True)
        
        assert outputs1.shape == outputs2.shape
        assert states1.shape == states2.shape
        assert not np.allclose(outputs1, outputs2, atol=0.1)  # Should be different
    
    def test_layer_connectivity(self):
        """Test that layers are properly connected"""
        input_size = 10
        hidden_size = 15
        output_size = 5
        seq_len = 6
        batch_size = 2
        
        # Single layer
        rnn1 = VanillaRNN(input_size, hidden_size, output_size, num_layers=1)
        # Multi layer  
        rnn2 = VanillaRNN(input_size, hidden_size, output_size, num_layers=3)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        
        outputs1, _ = rnn1.forward(x, return_sequences=True)
        outputs2, _ = rnn2.forward(x, return_sequences=True)
        
        # Multi-layer should produce different outputs
        assert not np.allclose(outputs1, outputs2, atol=0.1)


class TestRNNLanguageModel:
    """Test RNN language model"""
    
    def test_initialization(self):
        """Test language model initialization"""
        vocab_size = 1000
        embed_dim = 128
        hidden_size = 256
        num_layers = 2
        
        lm = RNNLanguageModel(vocab_size, embed_dim, hidden_size, num_layers)
        
        assert lm.vocab_size == vocab_size
        assert lm.embed_dim == embed_dim
        assert lm.embedding.shape == (vocab_size, embed_dim)
        assert lm.rnn.input_size == embed_dim
        assert lm.rnn.hidden_size == hidden_size
        assert lm.rnn.output_size == vocab_size
    
    def test_forward_pass(self):
        """Test forward pass through language model"""
        vocab_size = 100
        embed_dim = 32
        hidden_size = 64
        seq_len = 20
        batch_size = 4
        
        lm = RNNLanguageModel(vocab_size, embed_dim, hidden_size, num_layers=1)
        
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        logits = lm.forward(input_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not np.any(np.isnan(logits))
    
    def test_generation(self):
        """Test sequence generation"""
        vocab_size = 50
        embed_dim = 16
        hidden_size = 32
        length = 15
        
        lm = RNNLanguageModel(vocab_size, embed_dim, hidden_size, num_layers=1)
        
        generated = lm.generate(start_token=1, length=length, temperature=1.0)
        
        assert len(generated) == length
        assert all(0 <= token < vocab_size for token in generated)
        assert generated[0] == 1  # Start token should be preserved
    
    def test_temperature_effect(self):
        """Test that temperature affects generation"""
        vocab_size = 20
        embed_dim = 8
        hidden_size = 16
        length = 8
        
        lm = RNNLanguageModel(vocab_size, embed_dim, hidden_size, num_layers=1)
        
        # Generate with different temperatures
        np.random.seed(42)
        seq1 = lm.generate(start_token=0, length=length, temperature=0.1)
        
        np.random.seed(42)
        seq2 = lm.generate(start_token=0, length=length, temperature=2.0)
        
        # Sequences should be valid regardless of temperature
        assert all(0 <= token < vocab_size for token in seq1)
        assert all(0 <= token < vocab_size for token in seq2)


class TestGradientAnalyzer:
    """Test gradient analyzer"""
    
    def test_vanishing_gradient_analysis(self):
        """Test vanishing gradient analysis"""
        analyzer = GradientAnalyzer()
        
        hidden_size = 20
        sequence_length = 25
        
        results = analyzer.analyze_vanishing_gradients(
            hidden_size, sequence_length, 'tanh', num_trials=5
        )
        
        assert 'gradient_norms' in results
        assert 'eigenvalues' in results
        assert 'activation' in results
        
        assert len(results['gradient_norms']) == 5  # num_trials
        assert len(results['eigenvalues']) == 5
        assert results['activation'] == 'tanh'
        
        # Each trial should have gradient norms for each timestep
        for norms in results['gradient_norms']:
            assert len(norms) == sequence_length
            assert all(norm >= 0 for norm in norms)  # Norms should be non-negative
    
    def test_compare_activations(self):
        """Test activation comparison"""
        analyzer = GradientAnalyzer()
        
        hidden_size = 15
        sequence_length = 20
        
        results = analyzer.compare_activations(hidden_size, sequence_length)
        
        expected_activations = ['tanh', 'relu', 'sigmoid']
        for activation in expected_activations:
            assert activation in results
            assert 'gradient_norms' in results[activation]
            assert 'eigenvalues' in results[activation]
            
            # Check data structure
            assert len(results[activation]['gradient_norms']) == 50  # default num_trials
            assert len(results[activation]['eigenvalues']) == 50
    
    def test_eigenvalue_computation(self):
        """Test eigenvalue computation"""
        analyzer = GradientAnalyzer()
        
        # Test with known matrix
        hidden_size = 3
        results = analyzer.analyze_vanishing_gradients(
            hidden_size, 10, 'tanh', num_trials=1
        )
        
        eigenvalues = results['eigenvalues']
        assert len(eigenvalues) == 1
        assert eigenvalues[0] >= 0  # Magnitude should be non-negative


class TestRNNTrainer:
    """Test RNN trainer"""
    
    def test_initialization(self):
        """Test trainer initialization"""
        input_size = 10
        hidden_size = 15
        output_size = 5
        
        rnn = VanillaRNN(input_size, hidden_size, output_size)
        trainer = RNNTrainer(rnn, learning_rate=0.01, clip_norm=5.0)
        
        assert trainer.model == rnn
        assert trainer.learning_rate == 0.01
        assert trainer.clip_norm == 5.0
    
    def test_loss_computation(self):
        """Test loss computation"""
        rnn = VanillaRNN(5, 10, 3)
        trainer = RNNTrainer(rnn)
        
        # MSE loss
        predictions = np.random.randn(4, 3)
        targets = np.random.randn(4, 3)
        
        mse_loss = trainer.compute_loss(predictions, targets, 'mse')
        assert mse_loss >= 0
        assert isinstance(mse_loss, float)
        
        # Cross entropy loss
        targets_ce = np.random.randint(0, 3, 4)
        ce_loss = trainer.compute_loss(predictions, targets_ce, 'cross_entropy')
        assert ce_loss >= 0
        assert isinstance(ce_loss, float)
    
    def test_gradient_clipping(self):
        """Test gradient clipping"""
        rnn = VanillaRNN(5, 10, 3)
        trainer = RNNTrainer(rnn, clip_norm=1.0)
        
        # Create large gradients
        gradients = {
            'W_xh': np.random.randn(10, 5) * 10,  # Large gradients
            'W_hh': np.random.randn(10, 10) * 10,
            'b_h': np.random.randn(10) * 10
        }
        
        # Compute initial norm
        total_norm = np.sqrt(sum(np.sum(grad**2) for grad in gradients.values()))
        assert total_norm > 1.0  # Should be large
        
        # Clip gradients
        clipped_gradients = trainer.gradient_clip(gradients, 1.0)
        
        # Compute clipped norm
        clipped_norm = np.sqrt(sum(np.sum(grad**2) for grad in clipped_gradients.values()))
        assert clipped_norm <= 1.0 + 1e-6  # Should be clipped


class TestRNNProperties:
    """Test mathematical and theoretical properties"""
    
    def test_parameter_sharing(self):
        """Test parameter sharing across timesteps"""
        input_size = 8
        hidden_size = 12
        batch_size = 3
        
        cell = VanillaRNNCell(input_size, hidden_size)
        
        # Same parameters should be used for different inputs
        x1 = np.random.randn(batch_size, input_size)
        x2 = np.random.randn(batch_size, input_size)
        h = np.random.randn(batch_size, hidden_size)
        
        # Store original parameters
        W_xh_orig = cell.W_xh.copy()
        W_hh_orig = cell.W_hh.copy()
        b_h_orig = cell.b_h.copy()
        
        # Forward pass should not modify parameters
        _ = cell.forward(x1, h)
        _ = cell.forward(x2, h)
        
        assert np.array_equal(cell.W_xh, W_xh_orig)
        assert np.array_equal(cell.W_hh, W_hh_orig)
        assert np.array_equal(cell.b_h, b_h_orig)
    
    def test_sequence_processing_order(self):
        """Test that sequence processing order matters"""
        input_size = 6
        hidden_size = 8
        output_size = 4
        seq_len = 5
        batch_size = 2
        
        rnn = VanillaRNN(input_size, hidden_size, output_size, num_layers=1)
        
        # Create sequence
        x = np.random.randn(batch_size, seq_len, input_size)
        x_reversed = x[:, ::-1, :]  # Reverse sequence
        
        outputs1, _ = rnn.forward(x, return_sequences=True)
        outputs2, _ = rnn.forward(x_reversed, return_sequences=True)
        
        # Different sequences should produce different outputs
        assert not np.allclose(outputs1, outputs2, atol=0.1)
    
    def test_hidden_state_evolution(self):
        """Test that hidden state evolves over time"""
        input_size = 5
        hidden_size = 10
        seq_len = 8
        batch_size = 2
        
        rnn = VanillaRNN(input_size, hidden_size, hidden_size, num_layers=1)  # output_size = hidden_size
        
        x = np.random.randn(batch_size, seq_len, input_size)
        outputs, final_states = rnn.forward(x, return_sequences=True)
        
        # Outputs should change over time (not constant)
        first_timestep = outputs[:, 0, :]
        last_timestep = outputs[:, -1, :]
        
        assert not np.allclose(first_timestep, last_timestep, atol=0.1)
    
    def test_vanishing_gradient_property(self):
        """Test vanishing gradient property empirically"""
        hidden_size = 20
        sequence_length = 50
        
        analyzer = GradientAnalyzer()
        results = analyzer.analyze_vanishing_gradients(
            hidden_size, sequence_length, 'tanh', num_trials=10
        )
        
        # For most trials, gradients should decay
        decay_count = 0
        for gradient_norms in results['gradient_norms']:
            initial_norm = gradient_norms[0]
            final_norm = gradient_norms[-1]
            if final_norm < initial_norm * 0.5:  # At least 50% decay
                decay_count += 1
        
        # Most trials should show gradient decay
        assert decay_count >= 7  # At least 70% of trials
    
    def test_activation_bounds(self):
        """Test activation function bounds are respected"""
        input_size = 5
        hidden_size = 8
        batch_size = 3
        
        # Large inputs to test saturation
        x_large = np.random.randn(batch_size, input_size) * 10
        h_large = np.random.randn(batch_size, hidden_size) * 10
        
        # Test tanh bounds
        cell_tanh = VanillaRNNCell(input_size, hidden_size, 'tanh')
        h_tanh = cell_tanh.forward(x_large, h_large)
        assert np.all((-1 <= h_tanh) & (h_tanh <= 1))
        
        # Test ReLU bounds
        cell_relu = VanillaRNNCell(input_size, hidden_size, 'relu')
        h_relu = cell_relu.forward(x_large, h_large)
        assert np.all(h_relu >= 0)
        
        # Test sigmoid bounds
        cell_sigmoid = VanillaRNNCell(input_size, hidden_size, 'sigmoid')
        h_sigmoid = cell_sigmoid.forward(x_large, h_large)
        assert np.all((0 <= h_sigmoid) & (h_sigmoid <= 1))


if __name__ == "__main__":
    # Run tests
    import subprocess
    
    try:
        # Try to run with pytest
        result = subprocess.run(["python", "-m", "pytest", __file__, "-v"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except:
        # Fallback: run tests manually
        print("Running tests manually...")
        
        test_classes = [
            TestVanillaRNNCell(),
            TestVanillaRNN(),
            TestRNNLanguageModel(),
            TestGradientAnalyzer(),
            TestRNNTrainer(),
            TestRNNProperties()
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            class_name = test_class.__class__.__name__
            print(f"\n=== {class_name} ===")
            
            for method_name in dir(test_class):
                if method_name.startswith('test_'):
                    total_tests += 1
                    try:
                        method = getattr(test_class, method_name)
                        method()
                        print(f"✓ {method_name}")
                        passed_tests += 1
                    except Exception as e:
                        print(f"✗ {method_name}: {str(e)}")
        
        print(f"\n=== Test Summary ===")
        print(f"Passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {100*passed_tests/total_tests:.1f}%")