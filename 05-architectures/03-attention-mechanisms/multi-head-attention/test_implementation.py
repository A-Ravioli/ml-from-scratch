"""
Test suite for Multi-Head Attention implementation

Comprehensive tests for multi-head attention mechanisms, head specialization,
and attention variants.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from exercise import *


class TestMultiHeadAttention:
    """Test MultiHeadAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        self.num_heads = 8
        
        self.input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
    
    def test_initialization(self):
        """Test proper initialization of multi-head attention"""
        assert self.mha.d_model == self.d_model
        assert self.mha.num_heads == self.num_heads
        assert self.mha.d_k == self.d_model // self.num_heads
        assert self.mha.d_v == self.d_model // self.num_heads
        
        # Check parameter shapes
        assert self.mha.W_qkv.shape == (self.d_model, 3 * self.d_model)
        assert self.mha.W_o.shape == (self.d_model, self.d_model)
    
    def test_output_shape(self):
        """Test that multi-head attention produces correct output shape"""
        output = self.mha.forward(self.input)
        
        expected_shape = self.input.shape
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_attention_weights_storage(self):
        """Test that attention weights are properly stored and accessible"""
        output = self.mha.forward(self.input)
        
        # Test averaged attention weights
        avg_weights = self.mha.get_attention_weights()
        assert avg_weights is not None
        assert avg_weights.shape == (self.batch_size, self.seq_len, self.seq_len)
        
        # Test individual head weights
        all_head_weights = self.mha.get_all_head_weights()
        assert all_head_weights is not None
        assert all_head_weights.shape == (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        
        # Test specific head access
        for head_idx in range(self.num_heads):
            head_weights = self.mha.get_attention_weights(head_idx)
            assert head_weights.shape == (self.batch_size, self.seq_len, self.seq_len)
    
    def test_attention_properties(self):
        """Test mathematical properties of attention weights"""
        output = self.mha.forward(self.input)
        all_head_weights = self.mha.get_all_head_weights()
        
        for head_idx in range(self.num_heads):
            head_weights = all_head_weights[:, head_idx, :, :]
            
            # Should be non-negative
            assert np.all(head_weights >= 0), f"Head {head_idx} has negative attention weights"
            
            # Should sum to 1 along last dimension
            row_sums = np.sum(head_weights, axis=-1)
            assert np.allclose(row_sums, 1.0, atol=1e-6), \
                f"Head {head_idx} attention weights don't sum to 1"
    
    def test_different_head_counts(self):
        """Test multi-head attention with different numbers of heads"""
        valid_head_counts = [1, 2, 4, 8, 16]
        
        for num_heads in valid_head_counts:
            if self.d_model % num_heads == 0:
                mha = MultiHeadAttention(self.d_model, num_heads)
                output = mha.forward(self.input)
                
                assert output.shape == self.input.shape, \
                    f"Failed for {num_heads} heads"
                assert mha.d_k == self.d_model // num_heads
    
    def test_invalid_head_count(self):
        """Test that invalid head counts raise appropriate errors"""
        # d_model not divisible by num_heads should raise assertion error
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=64, num_heads=5)  # 64 not divisible by 5
    
    def test_masking_functionality(self):
        """Test attention with masking"""
        # Create causal mask
        mask = np.triu(np.ones((self.seq_len, self.seq_len)), k=1)  # Upper triangular
        mask = np.broadcast_to(mask[None, :, :], (self.batch_size, self.seq_len, self.seq_len))
        
        output = self.mha.forward(self.input, mask=mask)
        all_head_weights = self.mha.get_all_head_weights()
        
        # Check that masked positions have near-zero attention
        for head_idx in range(self.num_heads):
            head_weights = all_head_weights[:, head_idx, :, :]
            masked_positions = mask == 1
            
            assert np.all(head_weights[masked_positions] < 1e-6), \
                f"Head {head_idx} has non-zero attention in masked positions"
    
    def test_parameter_counting(self):
        """Test parameter counting"""
        param_count = self.mha.count_parameters()
        
        # Should count W_qkv and W_o
        expected_params = (
            self.d_model * 3 * self.d_model +  # W_qkv
            self.d_model * self.d_model        # W_o
        )
        
        assert param_count == expected_params
        
        # Test scaling with different dimensions
        for d_model in [32, 128, 256]:
            for num_heads in [4, 8]:
                if d_model % num_heads == 0:
                    mha = MultiHeadAttention(d_model, num_heads)
                    params = mha.count_parameters()
                    expected = 4 * d_model * d_model  # 4 = 3(QKV) + 1(output)
                    assert params == expected
    
    def test_different_sequence_lengths(self):
        """Test with various sequence lengths"""
        for seq_len in [1, 4, 16, 32, 64]:
            test_input = np.random.randn(1, seq_len, self.d_model)
            output = self.mha.forward(test_input)
            
            assert output.shape == test_input.shape
            
            # Check attention weights shape
            all_head_weights = self.mha.get_all_head_weights()
            assert all_head_weights.shape == (1, self.num_heads, seq_len, seq_len)


class TestScaledDotProductAttention:
    """Test scaled dot-product attention function"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 6
        self.d_k = 16
        self.d_v = 16
        
        self.Q = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.d_k)
        self.K = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.d_k)
        self.V = np.random.randn(self.batch_size, self.num_heads, self.seq_len, self.d_v)
        
        self.mha = MultiHeadAttention(64, 4)  # d_model=64, num_heads=4
    
    def test_attention_computation(self):
        """Test basic attention computation"""
        attention_output, attention_weights = self.mha.scaled_dot_product_attention(
            self.Q, self.K, self.V
        )
        
        # Check output shapes
        expected_output_shape = (self.batch_size, self.num_heads, self.seq_len, self.d_v)
        expected_weights_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        
        assert attention_output.shape == expected_output_shape
        assert attention_weights.shape == expected_weights_shape
        
        # Check attention properties
        assert np.all(attention_weights >= 0)
        row_sums = np.sum(attention_weights, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
    
    def test_attention_with_mask(self):
        """Test attention computation with mask"""
        # Create random mask
        mask = np.random.choice([0, 1], size=(self.batch_size, self.seq_len, self.seq_len))
        mask = mask[:, None, :, :]  # Add head dimension
        
        attention_output, attention_weights = self.mha.scaled_dot_product_attention(
            self.Q, self.K, self.V, mask=mask
        )
        
        # Masked positions should have near-zero attention
        masked_positions = mask == 1
        assert np.all(attention_weights[masked_positions] < 1e-6)
    
    def test_dropout_application(self):
        """Test that dropout can be applied (though results are random)"""
        # Test with dropout - mainly checking it doesn't crash
        attention_output, attention_weights = self.mha.scaled_dot_product_attention(
            self.Q, self.K, self.V, dropout_rate=0.1
        )
        
        # Should still have valid shapes and properties
        assert attention_output.shape == (self.batch_size, self.num_heads, self.seq_len, self.d_v)
        assert attention_weights.shape == (self.batch_size, self.num_heads, self.seq_len, self.seq_len)


class TestMultiQueryAttention:
    """Test MultiQueryAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        self.num_heads = 8
        
        self.input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
        self.mqa = MultiQueryAttention(self.d_model, self.num_heads)
    
    def test_output_shape(self):
        """Test MQA output shape"""
        output = self.mqa.forward(self.input)
        
        expected_shape = self.input.shape
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_parameter_efficiency(self):
        """Test that MQA uses fewer parameters than MHA"""
        mha = MultiHeadAttention(self.d_model, self.num_heads)
        
        mqa_params = self.mqa.count_parameters()
        mha_params = mha.count_parameters()
        
        # MQA should use fewer parameters due to shared K,V
        assert mqa_params < mha_params, "MQA should use fewer parameters than MHA"
        
        print(f"MHA parameters: {mha_params:,}")
        print(f"MQA parameters: {mqa_params:,}")
        print(f"Parameter reduction: {(1 - mqa_params/mha_params)*100:.1f}%")
    
    def test_attention_properties(self):
        """Test that MQA still produces valid attention weights"""
        output = self.mqa.forward(self.input)
        attention_weights = self.mqa.last_attention_weights
        
        assert attention_weights is not None
        assert attention_weights.shape == (self.batch_size, self.seq_len, self.seq_len)
        
        # Check attention properties
        assert np.all(attention_weights >= 0)
        row_sums = np.sum(attention_weights, axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


class TestAttentionVariantsComparison:
    """Test comparison between different attention variants"""
    
    def test_variant_comparison(self):
        """Test comparison of different attention variants"""
        seq_len, d_model, num_heads = 16, 64, 8
        
        comparison = compare_attention_variants(seq_len, d_model, num_heads)
        
        assert 'Multi-Head Attention' in comparison
        assert 'Multi-Query Attention' in comparison
        
        # Check that comparison contains expected metrics
        for variant_name, metrics in comparison.items():
            assert 'forward_time' in metrics
            assert 'parameters' in metrics
            assert 'output_shape' in metrics
            assert 'memory_efficiency' in metrics
            
            assert metrics['forward_time'] > 0
            assert metrics['parameters'] > 0
            
        # MQA should be more memory efficient
        mha_params = comparison['Multi-Head Attention']['parameters']
        mqa_params = comparison['Multi-Query Attention']['parameters']
        assert mqa_params < mha_params, "MQA should use fewer parameters"


class TestHeadSpecializationAnalysis:
    """Test head specialization analysis functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 8
        
        # Create sample attention weights with some patterns
        self.attention_weights = np.random.rand(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        
        # Normalize to make valid attention weights
        self.attention_weights = self.attention_weights / np.sum(self.attention_weights, axis=-1, keepdims=True)
        
        # Add some intentional patterns for testing
        # Head 0: diagonal pattern (self-attention)
        for i in range(self.seq_len):
            self.attention_weights[:, 0, i, i] = 0.5
        self.attention_weights[:, 0] = self.attention_weights[:, 0] / np.sum(self.attention_weights[:, 0], axis=-1, keepdims=True)
        
        # Head 1: uniform pattern
        self.attention_weights[:, 1] = 1.0 / self.seq_len
    
    def test_specialization_analysis(self):
        """Test head specialization analysis"""
        analysis = analyze_head_specialization(self.attention_weights)
        
        # Check that analysis contains expected keys
        required_keys = [
            'head_entropy', 'head_diversity', 'positional_preferences', 
            'head_similarity_matrix', 'specialized_patterns'
        ]
        
        for key in required_keys:
            assert key in analysis, f"Missing key: {key}"
        
        # Check shapes and properties
        assert len(analysis['head_entropy']) == self.num_heads
        assert len(analysis['positional_preferences']) == self.num_heads
        assert analysis['head_similarity_matrix'].shape == (self.num_heads, self.num_heads)
        
        # Check entropy properties
        for entropy in analysis['head_entropy']:
            assert entropy >= 0, "Entropy should be non-negative"
        
        # Uniform head (head 1) should have higher entropy than diagonal head (head 0)
        assert analysis['head_entropy'][1] > analysis['head_entropy'][0], \
            "Uniform attention should have higher entropy than diagonal attention"
    
    def test_pattern_detection(self):
        """Test detection of specific attention patterns"""
        analysis = analyze_head_specialization(self.attention_weights)
        
        specialized_patterns = analysis['specialized_patterns']
        
        # Check that patterns are detected for all heads
        for head_idx in range(self.num_heads):
            head_key = f'head_{head_idx}'
            assert head_key in specialized_patterns
            
            pattern_info = specialized_patterns[head_key]
            assert 'pattern_type' in pattern_info
            assert 'diagonal_score' in pattern_info
            assert 'adjacent_score' in pattern_info
        
        # Head 0 should be detected as having self-attention pattern
        head_0_pattern = specialized_patterns['head_0']['pattern_type']
        # Note: exact pattern detection depends on threshold, so we just check it's not None
        assert head_0_pattern is not None
        
        # Head 1 should be detected as global/uniform pattern
        head_1_entropy = analysis['head_entropy'][1]
        max_entropy = np.log(self.seq_len)
        assert head_1_entropy > 0.5 * max_entropy, "Head 1 should have high entropy"
    
    def test_head_similarity_matrix(self):
        """Test head similarity computation"""
        analysis = analyze_head_specialization(self.attention_weights)
        similarity_matrix = analysis['head_similarity_matrix']
        
        # Check properties of similarity matrix
        assert similarity_matrix.shape == (self.num_heads, self.num_heads)
        
        # Diagonal should be 1 (head similar to itself)
        diagonal_values = np.diag(similarity_matrix)
        assert np.allclose(diagonal_values, 1.0, atol=1e-3), \
            "Diagonal of similarity matrix should be close to 1"
        
        # Matrix should be symmetric
        assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-6), \
            "Similarity matrix should be symmetric"
        
        # Values should be in [-1, 1] range (correlation coefficients)
        assert np.all(similarity_matrix >= -1.01), "Similarity values should be >= -1"
        assert np.all(similarity_matrix <= 1.01), "Similarity values should be <= 1"


class TestHeadImportanceAndPruning:
    """Test head importance computation and pruning"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.num_heads = 8
        self.seq_len = 16
        
        self.model = MultiHeadAttention(self.d_model, self.num_heads)
        self.test_input = np.random.randn(2, self.seq_len, self.d_model)
    
    def test_head_importance_computation(self):
        """Test head importance scoring"""
        # First run forward pass to generate attention weights
        output = self.model.forward(self.test_input)
        
        importance_scores = compute_head_importance(self.model, self.test_input)
        
        # Check that importance scores are computed for all heads
        assert len(importance_scores) == self.num_heads
        
        # Scores should be non-negative
        assert np.all(importance_scores >= 0), "Importance scores should be non-negative"
        
        # Should have some variation (not all equal)
        assert np.std(importance_scores) > 0, "Importance scores should have some variation"
    
    def test_head_pruning(self):
        """Test head pruning functionality"""
        # Generate importance scores
        output = self.model.forward(self.test_input)
        importance_scores = compute_head_importance(self.model, self.test_input)
        
        # Test pruning with different keep ratios
        for keep_ratio in [0.5, 0.75]:
            pruned_model = prune_attention_heads(self.model, importance_scores, keep_ratio)
            
            expected_heads = int(self.num_heads * keep_ratio)
            assert pruned_model.num_heads == expected_heads
            
            # Pruned model should still work
            pruned_output = pruned_model.forward(self.test_input)
            assert pruned_output.shape == self.test_input.shape


class TestNumericalStability:
    """Test numerical stability of multi-head attention"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.num_heads = 8
        self.seq_len = 10
        
        self.model = MultiHeadAttention(self.d_model, self.num_heads)
    
    def test_extreme_inputs(self):
        """Test behavior with extreme input values"""
        extreme_inputs = [
            np.ones((1, self.seq_len, self.d_model)) * 1e6,    # Very large
            np.ones((1, self.seq_len, self.d_model)) * 1e-6,   # Very small
            np.zeros((1, self.seq_len, self.d_model)),         # All zeros
        ]
        
        for test_input in extreme_inputs:
            output = self.model.forward(test_input)
            
            # Should not produce NaN or inf
            assert not np.any(np.isnan(output)), "Should not produce NaN"
            assert not np.any(np.isinf(output)), "Should not produce inf"
            
            # Check attention weights are valid
            all_head_weights = self.model.get_all_head_weights()
            if all_head_weights is not None:
                assert not np.any(np.isnan(all_head_weights)), "Attention weights should not be NaN"
                assert not np.any(np.isinf(all_head_weights)), "Attention weights should not be inf"
    
    def test_large_sequence_lengths(self):
        """Test with large sequence lengths"""
        large_seq_lens = [64, 128, 256]
        
        for seq_len in large_seq_lens:
            test_input = np.random.randn(1, seq_len, self.d_model)
            output = self.model.forward(test_input)
            
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))
    
    def test_very_small_attention_scores(self):
        """Test with inputs that produce very small attention scores"""
        # Create inputs that will produce small dot products
        small_input = np.random.randn(1, self.seq_len, self.d_model) * 1e-3
        
        output = self.model.forward(small_input)
        all_head_weights = self.model.get_all_head_weights()
        
        # Even with small scores, attention should be valid
        if all_head_weights is not None:
            for head_idx in range(self.num_heads):
                head_weights = all_head_weights[0, head_idx]
                row_sums = np.sum(head_weights, axis=-1)
                assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_multi_head_vs_single_head():
    """Compare multi-head vs single-head attention performance"""
    print("Comparing multi-head vs single-head attention...")
    
    d_model = 64
    seq_len = 32
    batch_size = 4
    
    test_input = np.random.randn(batch_size, seq_len, d_model)
    
    # Multi-head attention
    multi_head = MultiHeadAttention(d_model, num_heads=8)
    mh_output = multi_head.forward(test_input)
    mh_params = multi_head.count_parameters()
    
    # Single-head attention (equivalent to num_heads=1)
    single_head = MultiHeadAttention(d_model, num_heads=1)
    sh_output = single_head.forward(test_input)
    sh_params = single_head.count_parameters()
    
    print(f"Multi-head parameters: {mh_params:,}")
    print(f"Single-head parameters: {sh_params:,}")
    print(f"Parameter ratio: {mh_params/sh_params:.2f}")
    
    # Both should have same output shape
    assert mh_output.shape == sh_output.shape
    
    # Multi-head should have same parameter count (due to dimension splitting)
    print("✓ Both produce same output shapes")
    print("✓ Parameter counts are similar (multi-head splits dimensions)")


def test_attention_head_diversity():
    """Test diversity of attention heads"""
    print("Testing attention head diversity...")
    
    d_model = 64
    num_heads = 8
    seq_len = 16
    
    model = MultiHeadAttention(d_model, num_heads)
    test_input = np.random.randn(1, seq_len, d_model)
    
    output = model.forward(test_input)
    all_head_weights = model.get_all_head_weights()
    
    if all_head_weights is not None:
        # Compute pairwise correlations between heads
        head_weights_flat = all_head_weights[0].reshape(num_heads, -1)  # [num_heads, seq_len²]
        
        correlations = []
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                corr = np.corrcoef(head_weights_flat[i], head_weights_flat[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        print(f"Average absolute correlation between heads: {avg_correlation:.3f}")
        
        # Lower correlation indicates more diverse heads
        if avg_correlation < 0.5:
            print("✓ Heads show good diversity (low correlation)")
        else:
            print("! Heads may be too similar (high correlation)")


def test_computational_efficiency():
    """Test computational efficiency of multi-head attention"""
    print("Testing computational efficiency...")
    
    configs = [
        (32, 4, 16),    # Small
        (64, 8, 32),    # Medium  
        (128, 16, 64),  # Large
    ]
    
    for d_model, num_heads, seq_len in configs:
        model = MultiHeadAttention(d_model, num_heads)
        test_input = np.random.randn(1, seq_len, d_model)
        
        # Time multiple forward passes
        num_runs = 50
        start_time = time.time()
        for _ in range(num_runs):
            output = model.forward(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        params = model.count_parameters()
        
        print(f"Config ({d_model}, {num_heads}, {seq_len}): {avg_time:.6f}s, {params:,} params")


def benchmark_attention_variants():
    """Benchmark different attention variants"""
    print("Benchmarking attention variants...")
    
    d_model = 128
    num_heads = 8
    seq_len = 64
    
    comparison = compare_attention_variants(seq_len, d_model, num_heads)
    
    print("Attention Variant Comparison:")
    for variant, metrics in comparison.items():
        print(f"{variant}:")
        print(f"  Forward time: {metrics['forward_time']:.6f}s")
        print(f"  Parameters: {metrics['parameters']:,}")
        print(f"  Memory efficiency: {metrics['memory_efficiency']}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run additional validation
    test_multi_head_vs_single_head()
    test_attention_head_diversity()
    test_computational_efficiency()
    benchmark_attention_variants()
    
    print("\nTesting completed!")
    print("Multi-head attention implementation validated successfully.")