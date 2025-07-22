"""
Test suite for GRU implementation
"""

import pytest
import numpy as np
import sys
import os

# Add the exercise module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exercise import GRUCell, GRU, GRULanguageModel, GRUClassifier, GRUPerformanceAnalyzer


class TestGRUCell:
    """Test GRU cell implementation"""
    
    def test_initialization(self):
        """Test GRU cell initialization"""
        input_size = 10
        hidden_size = 20
        
        cell = GRUCell(input_size, hidden_size)
        
        assert cell.input_size == input_size
        assert cell.hidden_size == hidden_size
        assert cell.W.shape == (3, hidden_size, input_size + hidden_size)
        assert cell.b.shape == (3, hidden_size)
        
        # Check update gate bias initialization (should be 1)
        assert np.allclose(cell.b[1], 1.0)
        assert np.allclose(cell.b[0], 0.0)  # Reset gate bias
        assert np.allclose(cell.b[2], 0.0)  # Candidate bias
    
    def test_forward_shapes(self):
        """Test forward pass output shapes"""
        input_size = 15
        hidden_size = 25
        batch_size = 8
        
        cell = GRUCell(input_size, hidden_size)
        
        x = np.random.randn(batch_size, input_size)
        h_prev = np.random.randn(batch_size, hidden_size)
        
        h_new = cell.forward(x, h_prev)
        
        assert h_new.shape == (batch_size, hidden_size)
        assert not np.allclose(h_new, h_prev)  # Should be different
    
    def test_gate_ranges(self):
        """Test that gates produce values in correct ranges"""
        input_size = 10
        hidden_size = 15
        batch_size = 5
        
        cell = GRUCell(input_size, hidden_size)
        
        x = np.random.randn(batch_size, input_size)
        h_prev = np.random.randn(batch_size, hidden_size)
        
        # Access internal gate computations
        combined = np.concatenate([h_prev, x], axis=1)
        gates = np.dot(combined, cell.W[:2].reshape(2 * hidden_size, -1).T) + cell.b[:2].flatten()
        gates = gates.reshape(batch_size, 2, hidden_size)
        
        reset_gate = cell.sigmoid(gates[:, 0, :])
        update_gate = cell.sigmoid(gates[:, 1, :])
        
        # Gates should be in [0, 1]
        assert np.all((0 <= reset_gate) & (reset_gate <= 1))
        assert np.all((0 <= update_gate) & (update_gate <= 1))
    
    def test_sigmoid_stability(self):
        """Test sigmoid numerical stability"""
        cell = GRUCell(10, 10)
        
        # Test extreme values
        large_pos = np.array([100.0])
        large_neg = np.array([-100.0])
        
        sig_pos = cell.sigmoid(large_pos)
        sig_neg = cell.sigmoid(large_neg)
        
        assert np.isclose(sig_pos[0], 1.0, atol=1e-10)
        assert np.isclose(sig_neg[0], 0.0, atol=1e-10)
        assert not np.isnan(sig_pos[0])
        assert not np.isnan(sig_neg[0])
    
    def test_gradient_flow_properties(self):
        """Test that GRU has good gradient flow properties"""
        input_size = 20
        hidden_size = 30
        batch_size = 4
        
        cell = GRUCell(input_size, hidden_size)
        
        # Test that update gate close to 0 preserves previous state
        x = np.random.randn(batch_size, input_size) * 0.001  # Small input
        h_prev = np.random.randn(batch_size, hidden_size)
        
        # Modify update gate bias to favor memory (large negative bias)
        cell.b[1] = -10 * np.ones(hidden_size)
        
        h_new = cell.forward(x, h_prev)
        
        # Should be close to previous state when update gate is close to 0
        similarity = np.mean(np.abs(h_new - h_prev))
        assert similarity < 1.0  # Should be reasonably close


class TestGRU:
    """Test multi-layer GRU implementation"""
    
    def test_initialization(self):
        """Test GRU initialization"""
        input_size = 50
        hidden_size = 64
        num_layers = 3
        
        # Unidirectional
        gru = GRU(input_size, hidden_size, num_layers)
        assert len(gru.forward_layers) == num_layers
        assert not hasattr(gru, 'backward_layers')
        
        # Bidirectional
        bigru = GRU(input_size, hidden_size, num_layers, bidirectional=True)
        assert len(bigru.forward_layers) == num_layers
        assert len(bigru.backward_layers) == num_layers
    
    def test_forward_shapes(self):
        """Test forward pass shapes"""
        input_size = 30
        hidden_size = 40
        num_layers = 2
        seq_len = 15
        batch_size = 6
        
        gru = GRU(input_size, hidden_size, num_layers)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        outputs, final_states = gru.forward(x)
        
        assert outputs.shape == (batch_size, seq_len, hidden_size)
        assert final_states.shape == (num_layers, batch_size, hidden_size)
    
    def test_bidirectional_forward(self):
        """Test bidirectional GRU forward pass"""
        input_size = 25
        hidden_size = 35
        num_layers = 2
        seq_len = 10
        batch_size = 4
        
        bigru = GRU(input_size, hidden_size, num_layers, bidirectional=True)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        outputs, final_states = bigru.forward(x)
        
        # Bidirectional doubles output size
        assert outputs.shape == (batch_size, seq_len, 2 * hidden_size)
        assert final_states.shape == (num_layers, batch_size, 2 * hidden_size)
    
    def test_initial_state_handling(self):
        """Test handling of initial states"""
        input_size = 20
        hidden_size = 25
        num_layers = 2
        seq_len = 8
        batch_size = 3
        
        gru = GRU(input_size, hidden_size, num_layers)
        x = np.random.randn(batch_size, seq_len, input_size)
        
        # Test with default initialization
        outputs1, states1 = gru.forward(x)
        
        # Test with custom initialization
        h_0 = np.random.randn(num_layers, batch_size, hidden_size) * 0.1
        outputs2, states2 = gru.forward(x, h_0)
        
        assert outputs1.shape == outputs2.shape
        assert states1.shape == states2.shape
        assert not np.allclose(outputs1, outputs2)  # Should be different
    
    def test_sequence_processing_consistency(self):
        """Test that GRU processes sequences consistently"""
        input_size = 15
        hidden_size = 20
        seq_len = 10
        batch_size = 2
        
        gru = GRU(input_size, hidden_size, num_layers=1)
        
        # Create test sequence
        x = np.random.randn(batch_size, seq_len, input_size)
        
        # Process full sequence
        full_output, _ = gru.forward(x)
        
        # Process step by step
        step_outputs = []
        h_state = np.zeros((1, batch_size, hidden_size))
        
        for t in range(seq_len):
            step_input = x[:, t:t+1, :]  # Single timestep
            step_output, h_state = gru.forward(step_input, h_state)
            step_outputs.append(step_output[:, 0, :])  # Remove seq dimension
        
        step_outputs = np.stack(step_outputs, axis=1)
        
        # Should match (within numerical precision)
        assert np.allclose(full_output, step_outputs, rtol=1e-5, atol=1e-6)


class TestGRULanguageModel:
    """Test GRU language model"""
    
    def test_initialization(self):
        """Test language model initialization"""
        vocab_size = 1000
        embed_dim = 128
        hidden_size = 256
        num_layers = 2
        
        lm = GRULanguageModel(vocab_size, embed_dim, hidden_size, num_layers)
        
        assert lm.vocab_size == vocab_size
        assert lm.embed_dim == embed_dim
        assert lm.embedding.shape == (vocab_size, embed_dim)
        assert lm.output_projection.shape == (hidden_size, vocab_size)
    
    def test_forward_pass(self):
        """Test forward pass through language model"""
        vocab_size = 100
        embed_dim = 32
        hidden_size = 64
        seq_len = 20
        batch_size = 4
        
        lm = GRULanguageModel(vocab_size, embed_dim, hidden_size, num_layers=1)
        
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
        
        lm = GRULanguageModel(vocab_size, embed_dim, hidden_size, num_layers=1)
        
        generated = lm.generate(start_token=1, length=length, temperature=1.0)
        
        assert len(generated) == length
        assert all(0 <= token < vocab_size for token in generated)
        assert generated[0] == 1  # Start token
    
    def test_temperature_effect(self):
        """Test temperature effect on generation"""
        vocab_size = 20
        embed_dim = 8
        hidden_size = 16
        length = 10
        
        lm = GRULanguageModel(vocab_size, embed_dim, hidden_size, num_layers=1)
        
        # Fix random seed for comparison
        np.random.seed(42)
        low_temp = lm.generate(start_token=0, length=length, temperature=0.1)
        
        np.random.seed(42) 
        high_temp = lm.generate(start_token=0, length=length, temperature=2.0)
        
        # Different temperatures should generally produce different sequences
        # (though this is probabilistic, so we just check they're valid)
        assert all(0 <= token < vocab_size for token in low_temp)
        assert all(0 <= token < vocab_size for token in high_temp)


class TestGRUClassifier:
    """Test GRU classifier"""
    
    def test_initialization(self):
        """Test classifier initialization"""
        input_size = 100
        hidden_size = 64
        num_classes = 5
        
        # Unidirectional
        classifier = GRUClassifier(input_size, hidden_size, num_classes, num_layers=2)
        assert classifier.classifier.shape == (hidden_size, num_classes)
        
        # Bidirectional
        bi_classifier = GRUClassifier(input_size, hidden_size, num_classes, 
                                     num_layers=2, bidirectional=True)
        assert bi_classifier.classifier.shape == (2 * hidden_size, num_classes)
    
    def test_forward_pass(self):
        """Test forward pass"""
        input_size = 50
        hidden_size = 32
        num_classes = 3
        seq_len = 25
        batch_size = 8
        
        classifier = GRUClassifier(input_size, hidden_size, num_classes)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        logits = classifier.forward(x)
        
        assert logits.shape == (batch_size, num_classes)
        assert not np.any(np.isnan(logits))
    
    def test_predictions(self):
        """Test prediction generation"""
        input_size = 30
        hidden_size = 24
        num_classes = 4
        seq_len = 15
        batch_size = 6
        
        classifier = GRUClassifier(input_size, hidden_size, num_classes)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        predictions = classifier.predict(x)
        
        assert predictions.shape == (batch_size,)
        assert np.all((0 <= predictions) & (predictions < num_classes))
    
    def test_masking(self):
        """Test sequence masking"""
        input_size = 20
        hidden_size = 16
        num_classes = 2
        seq_len = 12
        batch_size = 4
        
        classifier = GRUClassifier(input_size, hidden_size, num_classes)
        
        x = np.random.randn(batch_size, seq_len, input_size)
        mask = np.ones((batch_size, seq_len))
        
        # Create variable length sequences
        mask[0, 8:] = 0  # First sequence has length 8
        mask[1, 5:] = 0  # Second sequence has length 5
        
        logits_masked = classifier.forward(x, mask)
        logits_unmasked = classifier.forward(x)
        
        assert logits_masked.shape == logits_unmasked.shape
        # Results should be different due to masking
        assert not np.allclose(logits_masked, logits_unmasked)


class TestGRUPerformanceAnalyzer:
    """Test performance analyzer"""
    
    def test_benchmark_models(self):
        """Test model benchmarking"""
        analyzer = GRUPerformanceAnalyzer()
        
        input_size = 20
        hidden_size = 32
        seq_len = 30
        batch_size = 8
        
        results = analyzer.benchmark_models(input_size, hidden_size, seq_len, 
                                          batch_size, num_runs=3)
        
        assert 'GRU' in results
        assert 'LSTM' in results
        assert 'RNN' in results
        
        for model_name, stats in results.items():
            assert 'avg_time' in stats
            assert 'std_time' in stats
            assert 'memory_usage' in stats
            assert 'params' in stats
            assert stats['avg_time'] > 0
            assert stats['memory_usage'] > 0
            assert stats['params'] > 0
    
    def test_gradient_flow_analysis(self):
        """Test gradient flow analysis"""
        analyzer = GRUPerformanceAnalyzer()
        
        hidden_size = 24
        seq_len = 20
        
        gradient_info = analyzer.analyze_gradient_flow(hidden_size, seq_len)
        
        assert 'magnitude' in gradient_info
        assert 'reset_gate_avg' in gradient_info
        assert 'update_gate_avg' in gradient_info
        
        assert len(gradient_info['magnitude']) == seq_len
        assert len(gradient_info['reset_gate_avg']) == seq_len
        assert len(gradient_info['update_gate_avg']) == seq_len
        
        # Gate values should be in [0, 1]
        assert np.all(np.array(gradient_info['reset_gate_avg']) >= 0)
        assert np.all(np.array(gradient_info['reset_gate_avg']) <= 1)
        assert np.all(np.array(gradient_info['update_gate_avg']) >= 0)
        assert np.all(np.array(gradient_info['update_gate_avg']) <= 1)


class TestGRUProperties:
    """Test mathematical and theoretical properties of GRU"""
    
    def test_update_gate_interpolation(self):
        """Test that update gate performs linear interpolation"""
        input_size = 10
        hidden_size = 15
        batch_size = 3
        
        cell = GRUCell(input_size, hidden_size)
        
        x = np.random.randn(batch_size, input_size)
        h_prev = np.random.randn(batch_size, hidden_size)
        
        # Force update gate to specific values
        cell.b[1] = np.full(hidden_size, -10)  # z ≈ 0 (keep old state)
        h_keep = cell.forward(x, h_prev)
        
        cell.b[1] = np.full(hidden_size, 10)   # z ≈ 1 (use new candidate)
        h_update = cell.forward(x, h_prev)
        
        # With z ≈ 0, should be close to h_prev
        assert np.mean(np.abs(h_keep - h_prev)) < np.mean(np.abs(h_update - h_prev))
    
    def test_reset_gate_functionality(self):
        """Test reset gate functionality"""
        input_size = 8
        hidden_size = 12
        batch_size = 2
        
        cell = GRUCell(input_size, hidden_size)
        
        x = np.random.randn(batch_size, input_size)
        h_prev = np.random.randn(batch_size, hidden_size)
        
        # Test with reset gate ≈ 0 (ignore previous state)
        cell.b[0] = np.full(hidden_size, -10)  # r ≈ 0
        h_reset = cell.forward(x, h_prev)
        
        # Test with reset gate ≈ 1 (use previous state) 
        cell.b[0] = np.full(hidden_size, 10)   # r ≈ 1
        h_noreset = cell.forward(x, h_prev)
        
        # Different reset gate values should produce different results
        assert not np.allclose(h_reset, h_noreset, rtol=1e-3)
    
    def test_parameter_efficiency(self):
        """Test parameter count vs LSTM"""
        input_size = 50
        hidden_size = 100
        num_layers = 2
        
        # GRU parameter count
        def count_gru_params():
            layer1 = 3 * (input_size + hidden_size) * hidden_size + 3 * hidden_size
            layer2 = 3 * (hidden_size + hidden_size) * hidden_size + 3 * hidden_size
            return layer1 + layer2
        
        # LSTM parameter count  
        def count_lstm_params():
            layer1 = 4 * (input_size + hidden_size) * hidden_size + 4 * hidden_size
            layer2 = 4 * (hidden_size + hidden_size) * hidden_size + 4 * hidden_size
            return layer1 + layer2
        
        gru_params = count_gru_params()
        lstm_params = count_lstm_params()
        
        # GRU should have ~25% fewer parameters
        reduction = 1 - gru_params / lstm_params
        assert 0.2 < reduction < 0.3  # Approximately 25% reduction
    
    def test_computational_efficiency(self):
        """Test computational efficiency vs alternatives"""
        input_size = 30
        hidden_size = 50
        seq_len = 40
        batch_size = 8
        
        gru = GRU(input_size, hidden_size, num_layers=1)
        
        # Simple RNN for comparison
        class SimpleRNN:
            def __init__(self, input_size, hidden_size):
                self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
                self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
                self.b_h = np.zeros(hidden_size)
            
            def forward(self, x):
                batch_size, seq_len, input_size = x.shape
                hidden_size = self.W_hh.shape[0]
                
                outputs = []
                h = np.zeros((batch_size, hidden_size))
                
                for t in range(seq_len):
                    h = np.tanh(np.dot(h, self.W_hh) + np.dot(x[:, t], self.W_xh.T) + self.b_h)
                    outputs.append(h)
                
                return np.stack(outputs, axis=1)
        
        rnn = SimpleRNN(input_size, hidden_size)
        x = np.random.randn(batch_size, seq_len, input_size)
        
        # Time both models
        import time
        
        start = time.time()
        gru_out, _ = gru.forward(x)
        gru_time = time.time() - start
        
        start = time.time()
        rnn_out = rnn.forward(x)
        rnn_time = time.time() - start
        
        # GRU should be slower than simple RNN (due to gates) but not excessively so
        assert gru_time > rnn_time  # GRU has more computation
        assert gru_time < rnn_time * 10  # But not more than 10x slower
    
    def test_bidirectional_symmetry(self):
        """Test bidirectional processing symmetry"""
        input_size = 20
        hidden_size = 25
        seq_len = 10
        batch_size = 4
        
        bigru = GRU(input_size, hidden_size, num_layers=1, bidirectional=True)
        
        # Create symmetric sequence (palindrome)
        x = np.random.randn(batch_size, seq_len, input_size)
        x_reversed = x[:, ::-1, :]  # Time-reversed sequence
        
        outputs, _ = bigru.forward(x)
        outputs_rev, _ = bigru.forward(x_reversed)
        
        # Extract forward and backward components
        forward_out = outputs[:, :, :hidden_size]
        backward_out = outputs[:, :, hidden_size:]
        
        forward_out_rev = outputs_rev[:, :, :hidden_size]
        backward_out_rev = outputs_rev[:, :, hidden_size:]
        
        # The structure should be preserved (though not exactly equal due to different parameters)
        assert forward_out.shape == backward_out.shape
        assert not np.allclose(forward_out, backward_out)  # Should be different directions


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
            TestGRUCell(),
            TestGRU(), 
            TestGRULanguageModel(),
            TestGRUClassifier(),
            TestGRUPerformanceAnalyzer(),
            TestGRUProperties()
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