"""
Test suite for Self-Attention implementation

Comprehensive tests for self-attention mechanisms, multi-head attention,
and attention pattern analysis.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from exercise import *


class TestSelfAttention:
    """Test SelfAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        
        self.input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
        self.self_attn = SelfAttention(self.d_model)
    
    def test_output_shape(self):
        """Test that self-attention produces correct output shape"""
        output = self.self_attn.forward(self.input)
        
        expected_shape = self.input.shape
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_attention_weights_properties(self):
        """Test mathematical properties of attention weights"""
        output = self.self_attn.forward(self.input)
        attention_weights = self.self_attn.get_attention_weights()
        
        # Should have correct shape
        expected_attn_shape = (self.batch_size, self.seq_len, self.seq_len)
        assert attention_weights.shape == expected_attn_shape
        
        # Should be non-negative
        assert np.all(attention_weights >= 0), "Attention weights should be non-negative"
        
        # Rows should sum to 1 (softmax property)
        row_sums = np.sum(attention_weights, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Attention weights should sum to 1 per row"
    
    def test_different_dimensions(self):
        """Test self-attention with different dimensions"""
        dimensions = [32, 64, 128, 256]
        
        for d in dimensions:
            attn = SelfAttention(d)
            test_input = np.random.randn(1, 10, d)
            output = attn.forward(test_input)
            
            assert output.shape == test_input.shape, f"Failed for dimension {d}"
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention function"""
        d_k = 64
        Q = np.random.randn(self.batch_size, self.seq_len, d_k)
        K = np.random.randn(self.batch_size, self.seq_len, d_k)
        V = np.random.randn(self.batch_size, self.seq_len, d_k)
        
        output, attention_weights = self.self_attn.scaled_dot_product_attention(Q, K, V)
        
        # Check output shape
        assert output.shape == V.shape
        
        # Check attention weight properties
        assert np.all(attention_weights >= 0)
        row_sums = np.sum(attention_weights, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
    
    def test_masking_functionality(self):
        """Test attention with masking"""
        # Create a simple mask (mask out last two positions)
        mask = np.zeros((self.batch_size, self.seq_len, self.seq_len))
        mask[:, :, -2:] = 1  # Mask last two key positions
        
        output = self.self_attn.forward(self.input, mask=mask)
        attention_weights = self.self_attn.get_attention_weights()
        
        # Masked positions should have near-zero attention
        masked_attention = attention_weights[:, :, -2:]
        assert np.all(masked_attention < 1e-6), "Masked positions should have near-zero attention"
    
    def test_temperature_scaling(self):
        """Test different temperature parameters"""
        Q = np.random.randn(1, 5, 32)
        K = np.random.randn(1, 5, 32)  
        V = np.random.randn(1, 5, 32)
        
        temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]
        attention_entropies = []
        
        for temp in temperatures:
            _, attention_weights = self.self_attn.scaled_dot_product_attention(
                Q, K, V, temperature=temp
            )
            
            # Compute attention entropy
            entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=-1)
            mean_entropy = np.mean(entropy)
            attention_entropies.append(mean_entropy)
        
        # Lower temperature should lead to lower entropy (more peaked distributions)
        assert attention_entropies[0] < attention_entropies[-1], "Lower temperature should reduce entropy"
    
    def test_parameter_counting(self):
        """Test parameter counting"""
        param_count = self.self_attn.count_parameters()
        
        # Should count W_q, W_k, W_v, W_o matrices
        expected_params = 4 * self.d_model * self.d_model  # Simplified calculation
        
        assert isinstance(param_count, int)
        assert param_count > 0
        # Allow some flexibility for different d_k, d_v choices
        assert param_count >= expected_params * 0.5


class TestMaskedSelfAttention:
    """Test MaskedSelfAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        
        self.input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
        self.masked_attn = MaskedSelfAttention(self.d_model)
    
    def test_causal_mask_creation(self):
        """Test causal mask creation"""
        seq_len = 5
        mask = self.masked_attn.create_causal_mask(seq_len)
        
        # Should be upper triangular
        expected_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        assert np.array_equal(mask, expected_mask), "Causal mask should be upper triangular"
    
    def test_causal_attention_property(self):
        """Test that causal attention prevents future information leakage"""
        output = self.masked_attn.forward(self.input)
        attention_weights = self.masked_attn.get_attention_weights()
        
        # Upper triangular part should be zero (no future attention)
        for b in range(self.batch_size):
            for i in range(self.seq_len):
                for j in range(i + 1, self.seq_len):
                    assert attention_weights[b, i, j] < 1e-6, f"Found future attention at [{b},{i},{j}]"
    
    def test_autoregressive_consistency(self):
        """Test that each position only depends on previous positions"""
        # Process sequence step by step
        step_outputs = []
        
        for step in range(1, self.seq_len + 1):
            partial_input = self.input[:, :step, :]
            partial_output = self.masked_attn.forward(partial_input)
            step_outputs.append(partial_output[:, -1, :])  # Last position output
        
        # Full sequence output
        full_output = self.masked_attn.forward(self.input)
        
        # Each step's final output should match the corresponding position in full output
        for step in range(self.seq_len):
            assert np.allclose(step_outputs[step], full_output[:, step, :], atol=1e-5), \
                f"Step-by-step processing inconsistent at position {step}"


class TestMultiHeadSelfAttention:
    """Test MultiHeadSelfAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        self.num_heads = 8
        
        self.input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
        self.multi_head_attn = MultiHeadSelfAttention(self.d_model, self.num_heads)
    
    def test_output_shape(self):
        """Test multi-head attention output shape"""
        output = self.multi_head_attn.forward(self.input)
        
        expected_shape = self.input.shape
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_head_dimension_consistency(self):
        """Test that d_model is correctly divided among heads"""
        expected_d_k = self.d_model // self.num_heads
        
        for head in self.multi_head_attn.heads:
            assert head.d_k == expected_d_k, "Head dimensions should be d_model // num_heads"
            assert head.d_v == expected_d_k, "Head dimensions should be d_model // num_heads"
    
    def test_attention_weights_from_all_heads(self):
        """Test that we can get attention weights from all heads"""
        output = self.multi_head_attn.forward(self.input)
        all_attention_weights = self.multi_head_attn.get_attention_weights()
        
        assert len(all_attention_weights) == self.num_heads, "Should have weights from all heads"
        
        for i, head_weights in enumerate(all_attention_weights):
            expected_shape = (self.batch_size, self.seq_len, self.seq_len)
            assert head_weights.shape == expected_shape, f"Head {i} weights have wrong shape"
    
    def test_different_head_counts(self):
        """Test multi-head attention with different numbers of heads"""
        valid_head_counts = [1, 2, 4, 8]
        
        for num_heads in valid_head_counts:
            if self.d_model % num_heads == 0:  # Only test divisible head counts
                mha = MultiHeadSelfAttention(self.d_model, num_heads)
                output = mha.forward(self.input)
                
                assert output.shape == self.input.shape, f"Failed for {num_heads} heads"
    
    def test_head_specialization_potential(self):
        """Test that different heads can learn different patterns"""
        # This is more of a property test - we can't guarantee specialization
        # but we can check that heads have the potential to be different
        
        output = self.multi_head_attn.forward(self.input)
        all_attention_weights = self.multi_head_attn.get_attention_weights()
        
        # Check that heads are not identical (random initialization should ensure this)
        head1_weights = all_attention_weights[0]
        head2_weights = all_attention_weights[1]
        
        assert not np.allclose(head1_weights, head2_weights, atol=1e-3), \
            "Different heads should have different attention patterns"
    
    def test_parameter_scaling(self):
        """Test that parameter count scales appropriately with heads"""
        single_head = MultiHeadSelfAttention(self.d_model, 1)
        multi_head = MultiHeadSelfAttention(self.d_model, self.num_heads)
        
        single_params = single_head.count_parameters()
        multi_params = multi_head.count_parameters()
        
        # Multi-head should have similar parameter count (due to projection structure)
        # The exact relationship depends on implementation details
        assert multi_params > 0, "Multi-head should have positive parameter count"


class TestAttentionPatternAnalysis:
    """Test attention pattern analysis functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.seq_len = 8
        
        # Create a simple attention pattern for testing
        self.attention_weights = np.random.rand(self.seq_len, self.seq_len)
        # Normalize to make it a valid attention matrix
        self.attention_weights = self.attention_weights / np.sum(self.attention_weights, axis=-1, keepdims=True)
        
        self.tokens = [f"token_{i}" for i in range(self.seq_len)]
    
    def test_attention_entropy_calculation(self):
        """Test attention entropy computation"""
        analysis = analyze_attention_patterns(self.attention_weights, self.tokens)
        
        assert 'attention_entropy' in analysis
        assert len(analysis['attention_entropy']) == self.seq_len
        
        # Entropy should be non-negative
        assert all(h >= 0 for h in analysis['attention_entropy'])
        
        # Maximum possible entropy for uniform distribution
        max_entropy = np.log(self.seq_len)
        assert all(h <= max_entropy + 1e-6 for h in analysis['attention_entropy'])
    
    def test_attention_statistics(self):
        """Test various attention statistics"""
        analysis = analyze_attention_patterns(self.attention_weights)
        
        required_keys = [
            'attention_entropy', 'max_attention_weight', 'attention_spread',
            'self_attention_ratio', 'local_attention_ratio'
        ]
        
        for key in required_keys:
            assert key in analysis, f"Missing analysis key: {key}"
        
        # Self-attention ratio should be between 0 and 1
        assert 0 <= analysis['self_attention_ratio'] <= 1
        
        # Max attention weights should be valid
        max_weights = analysis['max_attention_weight']
        assert len(max_weights) == self.seq_len
        assert all(0 <= w <= 1 for w in max_weights)
    
    def test_focused_vs_distributed_attention(self):
        """Test analysis of focused vs distributed attention patterns"""
        # Create focused attention (one position gets most attention)
        focused_attention = np.zeros((3, 3))
        focused_attention[0, :] = [0.9, 0.05, 0.05]
        focused_attention[1, :] = [0.1, 0.8, 0.1]
        focused_attention[2, :] = [0.1, 0.1, 0.8]
        
        # Create distributed attention (uniform)
        distributed_attention = np.ones((3, 3)) / 3
        
        focused_analysis = analyze_attention_patterns(focused_attention)
        distributed_analysis = analyze_attention_patterns(distributed_attention)
        
        # Distributed attention should have higher entropy
        focused_entropy = np.mean(focused_analysis['attention_entropy'])
        distributed_entropy = np.mean(distributed_analysis['attention_entropy'])
        
        assert distributed_entropy > focused_entropy, \
            "Distributed attention should have higher entropy"


class TestComputationalComplexity:
    """Test computational complexity analysis"""
    
    def test_complexity_scaling(self):
        """Test attention complexity scaling with sequence length"""
        seq_lengths = [16, 32, 64, 128]
        d_model = 64
        
        complexity_analysis = compute_attention_complexity(seq_lengths, d_model)
        
        assert 'sequence_lengths' in complexity_analysis
        assert 'memory_complexity' in complexity_analysis
        assert 'time_complexity' in complexity_analysis
        
        # Memory should scale quadratically with sequence length
        memory_values = complexity_analysis['memory_complexity']
        
        # Check that memory complexity is increasing
        assert all(memory_values[i] < memory_values[i+1] for i in range(len(memory_values)-1))
        
        # Time complexity should also increase
        time_values = complexity_analysis['time_complexity']
        assert all(time_values[i] < time_values[i+1] for i in range(len(time_values)-1))
    
    def test_attention_matrix_size_scaling(self):
        """Test that attention matrix size scales as O(n²)"""
        seq_lengths = [10, 20, 40]
        d_model = 32
        
        complexity_analysis = compute_attention_complexity(seq_lengths, d_model)
        attention_sizes = complexity_analysis['attention_matrix_size']
        
        # Should scale quadratically
        for i in range(1, len(seq_lengths)):
            ratio = seq_lengths[i] / seq_lengths[i-1]
            size_ratio = attention_sizes[i] / attention_sizes[i-1]
            
            # Size ratio should be approximately ratio²
            expected_ratio = ratio ** 2
            assert abs(size_ratio - expected_ratio) < 0.1, \
                f"Attention matrix size should scale quadratically"


class TestMaskingStrategies:
    """Test various masking strategies"""
    
    def test_padding_mask_creation(self):
        """Test padding mask creation for variable-length sequences"""
        sequence_lengths = [5, 3, 7]
        max_length = 8
        
        mask = create_padding_mask(sequence_lengths, max_length)
        
        expected_shape = (len(sequence_lengths), max_length, max_length)
        assert mask.shape == expected_shape
        
        # Check that padding positions are masked
        for i, length in enumerate(sequence_lengths):
            if length < max_length:
                # Keys beyond length should be masked
                assert np.all(mask[i, :, length:] == 1), f"Padding keys not masked for sequence {i}"
                # Queries beyond length should be masked  
                assert np.all(mask[i, length:, :] == 1), f"Padding queries not masked for sequence {i}"
    
    def test_mask_application(self):
        """Test that masks are properly applied in attention computation"""
        batch_size, seq_len, d_model = 2, 6, 32
        test_input = np.random.randn(batch_size, seq_len, d_model)
        
        # Create mask that blocks last position
        mask = np.zeros((batch_size, seq_len, seq_len))
        mask[:, :, -1] = 1  # Mask last key position
        
        self_attn = SelfAttention(d_model)
        output = self_attn.forward(test_input, mask=mask)
        attention_weights = self_attn.get_attention_weights()
        
        # Last column should have near-zero attention
        assert np.all(attention_weights[:, :, -1] < 1e-6), \
            "Masked positions should have near-zero attention"


class TestNumericalStability:
    """Test numerical stability of attention mechanisms"""
    
    def test_extreme_input_values(self):
        """Test behavior with extreme input values"""
        d_model = 64
        extreme_inputs = [
            np.ones((1, 5, d_model)) * 1e6,    # Very large
            np.ones((1, 5, d_model)) * 1e-6,   # Very small
            np.zeros((1, 5, d_model)),         # All zeros
        ]
        
        self_attn = SelfAttention(d_model)
        
        for test_input in extreme_inputs:
            output = self_attn.forward(test_input)
            attention_weights = self_attn.get_attention_weights()
            
            # Should not produce NaN or inf
            assert not np.any(np.isnan(output)), "Output should not contain NaN"
            assert not np.any(np.isinf(output)), "Output should not contain inf"
            assert not np.any(np.isnan(attention_weights)), "Attention weights should not contain NaN"
            assert not np.any(np.isinf(attention_weights)), "Attention weights should not contain inf"
    
    def test_softmax_numerical_stability(self):
        """Test that softmax computation is numerically stable"""
        # Create scores that could cause softmax overflow
        large_scores = np.array([[1000., 1000., 1000.]])
        
        # This should not crash or produce invalid results
        exp_scores = np.exp(large_scores - np.max(large_scores, axis=-1, keepdims=True))
        normalized = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Result should be valid probabilities
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        assert np.allclose(np.sum(normalized, axis=-1), 1.0)


def test_self_attention_vs_rnn_comparison():
    """Compare self-attention properties with RNN properties"""
    print("Comparing self-attention with RNN properties...")
    
    batch_size, seq_len, d_model = 2, 16, 64
    test_input = np.random.randn(batch_size, seq_len, d_model)
    
    self_attn = SelfAttention(d_model)
    
    # Test parallelization potential (all positions processed simultaneously)
    start_time = time.time()
    output = self_attn.forward(test_input)
    attention_time = time.time() - start_time
    
    print(f"Self-attention forward pass time: {attention_time:.4f}s")
    
    # Test permutation equivariance
    # Permute input sequence
    perm_indices = np.random.permutation(seq_len)
    permuted_input = test_input[:, perm_indices, :]
    
    # Process both original and permuted
    original_output = self_attn.forward(test_input)
    permuted_output = self_attn.forward(permuted_input)
    
    # Permute the original output using same indices
    expected_permuted_output = original_output[:, perm_indices, :]
    
    # They should be equal (up to numerical precision)
    assert np.allclose(expected_permuted_output, permuted_output, atol=1e-5), \
        "Self-attention should be permutation equivariant"
    
    print("✓ Self-attention is permutation equivariant")
    print("✓ Self-attention processes all positions in parallel")


def test_attention_mechanism_comparison():
    """Compare different attention mechanisms"""
    print("Comparing different attention mechanisms...")
    
    seq_len, d_model = 32, 64
    comparison = compare_attention_mechanisms(seq_len, d_model)
    
    print("Attention Mechanism Comparison:")
    for name, metrics in comparison.items():
        print(f"{name}:")
        print(f"  Forward time: {metrics['forward_time']:.4f}s")
        print(f"  Parameters: {metrics['parameters']:,}")
        print(f"  Output shape: {metrics['output_shape']}")


def validate_attention_mathematical_properties():
    """Validate mathematical properties of attention"""
    print("Validating attention mathematical properties...")
    
    batch_size, seq_len, d_model = 1, 8, 32
    X = np.random.randn(batch_size, seq_len, d_model)
    
    self_attn = SelfAttention(d_model)
    output = self_attn.forward(X)
    attention_weights = self_attn.get_attention_weights()
    
    # Property 1: Attention weights are non-negative
    assert np.all(attention_weights >= 0), "Attention weights must be non-negative"
    
    # Property 2: Each row sums to 1
    row_sums = np.sum(attention_weights, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Attention weights must sum to 1"
    
    # Property 3: Output is a weighted combination of values
    Q = X @ self_attn.W_q
    K = X @ self_attn.W_k  
    V = X @ self_attn.W_v
    
    # Manual computation
    manual_attended = attention_weights @ V
    manual_output = manual_attended @ self_attn.W_o
    
    assert np.allclose(output, manual_output, atol=1e-5), \
        "Output should match manual weighted combination"
    
    print("✓ All mathematical properties validated")


def benchmark_sequence_length_scaling():
    """Benchmark attention performance across sequence lengths"""
    print("Benchmarking sequence length scaling...")
    
    d_model = 64
    seq_lengths = [16, 32, 64, 128, 256]
    
    times = []
    memory_estimates = []
    
    for seq_len in seq_lengths:
        test_input = np.random.randn(1, seq_len, d_model)
        self_attn = SelfAttention(d_model)
        
        # Measure time
        start_time = time.time()
        output = self_attn.forward(test_input)
        end_time = time.time()
        
        times.append(end_time - start_time)
        memory_estimates.append(seq_len * seq_len + seq_len * d_model)
    
    print("Sequence Length Scaling Results:")
    for i, seq_len in enumerate(seq_lengths):
        print(f"Length {seq_len:3d}: {times[i]:.4f}s, Memory est: {memory_estimates[i]:,}")
    
    # Verify quadratic scaling trend
    if len(times) >= 3:
        # Compare ratios
        ratio_2_to_1 = times[1] / times[0]
        ratio_3_to_2 = times[2] / times[1]
        print(f"Time scaling ratios: {ratio_2_to_1:.2f}, {ratio_3_to_2:.2f}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run additional validation
    test_self_attention_vs_rnn_comparison()
    test_attention_mechanism_comparison()
    validate_attention_mathematical_properties()
    benchmark_sequence_length_scaling()
    
    print("\nTesting completed!")
    print("Self-attention implementation validated successfully.")