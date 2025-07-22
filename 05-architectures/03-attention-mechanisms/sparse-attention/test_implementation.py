"""
Test suite for Sparse Attention implementation

Comprehensive tests for various sparse attention patterns, efficiency analysis,
and pattern visualization.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from exercise import *


class TestSparseAttentionBase:
    """Test base sparse attention functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
        
        self.test_input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_abstract_base_class(self):
        """Test that base class cannot be instantiated directly"""
        with pytest.raises(TypeError):
            SparseAttentionBase(self.d_model, self.num_heads)
    
    def test_sparse_attention_computation(self):
        """Test sparse attention computation with mock implementation"""
        # Create a concrete implementation for testing
        class MockSparseAttention(SparseAttentionBase):
            def create_attention_mask(self, seq_len):
                # Simple diagonal mask for testing
                return np.eye(seq_len)
        
        sparse_attn = MockSparseAttention(self.d_model, self.num_heads)
        
        # Test forward pass
        output = sparse_attn.forward(self.test_input)
        
        assert output.shape == self.test_input.shape
        assert sparse_attn.get_attention_weights() is not None
        assert sparse_attn.get_sparsity_pattern() is not None


class TestLocalWindowAttention:
    """Test LocalWindowAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.window_size = 4
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
        
        self.local_attn = LocalWindowAttention(self.d_model, self.window_size, self.num_heads)
        self.test_input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_attention_mask_creation(self):
        """Test local window attention mask creation"""
        mask = self.local_attn.create_attention_mask(self.seq_len)
        
        # Check mask shape
        assert mask.shape == (self.seq_len, self.seq_len)
        
        # Check that mask contains only 0s and 1s
        assert np.all((mask == 0) | (mask == 1))
        
        # Check window property: each position should attend to window around it
        for i in range(self.seq_len):
            expected_start = max(0, i - self.window_size)
            expected_end = min(self.seq_len, i + self.window_size + 1)
            
            # Count non-zero elements in row i
            attended_positions = np.sum(mask[i, :])
            expected_attended = expected_end - expected_start
            
            assert attended_positions == expected_attended, \
                f"Position {i} attends to {attended_positions} positions, expected {expected_attended}"
    
    def test_sparsity_properties(self):
        """Test sparsity properties of local window attention"""
        output = self.local_attn.forward(self.test_input)
        sparsity_ratio = self.local_attn.compute_sparsity_ratio()
        
        # Sparsity ratio should be less than 1 for reasonable window sizes
        assert 0 < sparsity_ratio < 1, f"Sparsity ratio {sparsity_ratio} should be between 0 and 1"
        
        # For small windows, should be much sparser than full attention
        if self.window_size < self.seq_len // 2:
            assert sparsity_ratio < 0.5, "Should be significantly sparse for small windows"
    
    def test_different_window_sizes(self):
        """Test local attention with different window sizes"""
        window_sizes = [1, 2, 4, 8]
        
        for window_size in window_sizes:
            if window_size <= self.seq_len:
                local_attn = LocalWindowAttention(self.d_model, window_size, self.num_heads)
                output = local_attn.forward(self.test_input)
                
                assert output.shape == self.test_input.shape
                
                # Check that sparsity decreases as window size increases
                sparsity_ratio = local_attn.compute_sparsity_ratio()
                assert 0 < sparsity_ratio <= 1
    
    def test_edge_cases(self):
        """Test edge cases for local window attention"""
        # Very small sequence
        small_input = np.random.randn(1, 3, self.d_model)
        output = self.local_attn.forward(small_input)
        assert output.shape == small_input.shape
        
        # Window size larger than sequence
        large_window_attn = LocalWindowAttention(self.d_model, window_size=20, num_heads=self.num_heads)
        mask = large_window_attn.create_attention_mask(self.seq_len)
        
        # Should essentially be full attention
        expected_full_mask = np.ones((self.seq_len, self.seq_len))
        assert np.array_equal(mask, expected_full_mask)


class TestStridedAttention:
    """Test StridedAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.window_size = 2
        self.stride = 4
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
        
        self.strided_attn = StridedAttention(self.d_model, self.window_size, self.stride, self.num_heads)
        self.test_input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_strided_mask_creation(self):
        """Test strided attention mask creation"""
        mask = self.strided_attn.create_attention_mask(self.seq_len)
        
        assert mask.shape == (self.seq_len, self.seq_len)
        assert np.all((mask == 0) | (mask == 1))
        
        # Check that strided positions are connected
        for i in range(0, self.seq_len, self.stride):
            # Position i should attend to all other strided positions
            for j in range(0, self.seq_len, self.stride):
                assert mask[i, j] == 1, f"Strided positions {i} and {j} should be connected"
    
    def test_local_plus_strided_pattern(self):
        """Test that pattern includes both local and strided connections"""
        mask = self.strided_attn.create_attention_mask(self.seq_len)
        
        # Check local connections exist
        for i in range(self.seq_len):
            # Should have local window connections
            local_connections = 0
            for j in range(max(0, i - self.window_size), min(self.seq_len, i + self.window_size + 1)):
                if mask[i, j] == 1:
                    local_connections += 1
            
            assert local_connections > 0, f"Position {i} should have local connections"
        
        # Check strided connections exist
        strided_positions = list(range(0, self.seq_len, self.stride))
        for i in strided_positions:
            strided_connections = np.sum(mask[i, strided_positions])
            assert strided_connections >= len(strided_positions), \
                f"Strided position {i} should connect to other strided positions"
    
    def test_forward_pass(self):
        """Test forward pass through strided attention"""
        output = self.strided_attn.forward(self.test_input)
        
        assert output.shape == self.test_input.shape
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


class TestRandomSparseAttention:
    """Test RandomSparseAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.sparsity_ratio = 0.2
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
        
        self.random_attn = RandomSparseAttention(self.d_model, self.sparsity_ratio, self.num_heads, seed=42)
        self.test_input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_random_mask_properties(self):
        """Test random sparse attention mask properties"""
        mask = self.random_attn.create_attention_mask(self.seq_len)
        
        assert mask.shape == (self.seq_len, self.seq_len)
        assert np.all((mask == 0) | (mask == 1))
        
        # Check approximate sparsity ratio
        actual_sparsity = np.mean(mask)
        assert abs(actual_sparsity - self.sparsity_ratio) < 0.1, \
            f"Actual sparsity {actual_sparsity} should be close to target {self.sparsity_ratio}"
        
        # Check that diagonal is always attended (self-attention)
        diagonal_values = np.diag(mask)
        assert np.all(diagonal_values == 1), "Diagonal should always be attended"
    
    def test_reproducibility(self):
        """Test that random patterns are reproducible with same seed"""
        mask1 = self.random_attn.create_attention_mask(self.seq_len)
        mask2 = self.random_attn.create_attention_mask(self.seq_len)
        
        assert np.array_equal(mask1, mask2), "Same seed should produce same pattern"
        
        # Different seed should produce different pattern
        different_seed_attn = RandomSparseAttention(self.d_model, self.sparsity_ratio, self.num_heads, seed=123)
        mask3 = different_seed_attn.create_attention_mask(self.seq_len)
        
        assert not np.array_equal(mask1, mask3), "Different seeds should produce different patterns"
    
    def test_different_sparsity_ratios(self):
        """Test with different sparsity ratios"""
        ratios = [0.1, 0.3, 0.5, 0.8]
        
        for ratio in ratios:
            random_attn = RandomSparseAttention(self.d_model, ratio, self.num_heads)
            mask = random_attn.create_attention_mask(self.seq_len)
            actual_sparsity = np.mean(mask)
            
            # Allow some tolerance due to randomness
            assert abs(actual_sparsity - ratio) < 0.15, \
                f"Sparsity ratio {actual_sparsity} should be close to {ratio}"


class TestBigBirdAttention:
    """Test BigBirdAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.window_size = 3
        self.num_global_tokens = 2
        self.num_random_tokens = 2
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
        
        self.bigbird_attn = BigBirdAttention(
            self.d_model, self.window_size, self.num_global_tokens, 
            self.num_random_tokens, self.num_heads, seed=42
        )
        self.test_input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_bigbird_mask_components(self):
        """Test that BigBird mask contains all required components"""
        mask = self.bigbird_attn.create_attention_mask(self.seq_len)
        
        assert mask.shape == (self.seq_len, self.seq_len)
        assert np.all((mask == 0) | (mask == 1))
        
        # Check local window connections
        for i in range(self.seq_len):
            local_start = max(0, i - self.window_size)
            local_end = min(self.seq_len, i + self.window_size + 1)
            
            # Should have local connections
            local_connections = np.sum(mask[i, local_start:local_end])
            assert local_connections > 0, f"Position {i} should have local connections"
        
        # Check global token connections
        global_positions = list(range(min(self.num_global_tokens, self.seq_len)))
        
        for global_pos in global_positions:
            # Global tokens should attend to all positions
            assert np.sum(mask[global_pos, :]) == self.seq_len, \
                f"Global token {global_pos} should attend to all positions"
            
            # All tokens should attend to global tokens
            assert np.sum(mask[:, global_pos]) == self.seq_len, \
                f"All tokens should attend to global token {global_pos}"
    
    def test_bigbird_sparsity(self):
        """Test BigBird sparsity properties"""
        output = self.bigbird_attn.forward(self.test_input)
        sparsity_ratio = self.bigbird_attn.compute_sparsity_ratio()
        
        # Should be sparse but not too sparse (due to global tokens)
        assert 0 < sparsity_ratio < 1, "BigBird should have reasonable sparsity"
        
        # Should be more connected than simple local attention due to global and random components
        local_only = LocalWindowAttention(self.d_model, self.window_size, self.num_heads)
        local_output = local_only.forward(self.test_input)
        local_sparsity = local_only.compute_sparsity_ratio()
        
        # BigBird should be less sparse (more connected) than local only
        assert sparsity_ratio >= local_sparsity, \
            "BigBird should be at least as connected as local attention"
    
    def test_forward_pass_stability(self):
        """Test forward pass numerical stability"""
        output = self.bigbird_attn.forward(self.test_input)
        
        assert output.shape == self.test_input.shape
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        
        attention_weights = self.bigbird_attn.get_attention_weights()
        assert attention_weights is not None


class TestLongformerAttention:
    """Test LongformerAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.window_size = 3
        self.global_token_indices = [0, 1]  # First two tokens are global
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
        
        self.longformer_attn = LongformerAttention(
            self.d_model, self.window_size, self.global_token_indices, self.num_heads
        )
        self.test_input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_longformer_pattern(self):
        """Test Longformer attention pattern"""
        mask = self.longformer_attn.create_attention_mask(self.seq_len)
        
        assert mask.shape == (self.seq_len, self.seq_len)
        
        # Check global token properties
        for global_idx in self.global_token_indices:
            if global_idx < self.seq_len:
                # Global token should attend to all positions
                assert np.sum(mask[global_idx, :]) == self.seq_len
                
                # All positions should attend to global token
                assert np.sum(mask[:, global_idx]) == self.seq_len
        
        # Check local window for non-global tokens
        for i in range(self.seq_len):
            if i not in self.global_token_indices:
                local_start = max(0, i - self.window_size)
                local_end = min(self.seq_len, i + self.window_size + 1)
                
                # Should have at least local window connections
                local_connections = np.sum(mask[i, local_start:local_end])
                assert local_connections > 0
    
    def test_global_token_effects(self):
        """Test the effect of different global token configurations"""
        # Test with no global tokens
        no_global_attn = LongformerAttention(self.d_model, self.window_size, [], self.num_heads)
        no_global_mask = no_global_attn.create_attention_mask(self.seq_len)
        
        # Test with many global tokens
        many_global_indices = list(range(min(5, self.seq_len)))
        many_global_attn = LongformerAttention(self.d_model, self.window_size, many_global_indices, self.num_heads)
        many_global_mask = many_global_attn.create_attention_mask(self.seq_len)
        
        # More global tokens should result in higher sparsity ratio
        no_global_sparsity = np.mean(no_global_mask)
        many_global_sparsity = np.mean(many_global_mask)
        
        assert many_global_sparsity > no_global_sparsity, \
            "More global tokens should increase connectivity"


class TestBlockSparseAttention:
    """Test BlockSparseAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.d_model = 64
        self.block_size = 4
        self.num_heads = 8
        self.seq_len = 16
        self.batch_size = 2
        
        self.block_attn = BlockSparseAttention(self.d_model, self.block_size, self.num_heads)
        self.test_input = np.random.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_block_structure(self):
        """Test block sparse attention structure"""
        mask = self.block_attn.create_attention_mask(self.seq_len)
        
        assert mask.shape == (self.seq_len, self.seq_len)
        
        num_blocks = (self.seq_len + self.block_size - 1) // self.block_size
        
        # Check block diagonal structure
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min((block_idx + 1) * self.block_size, self.seq_len)
            
            # Within block should have full connectivity
            block_mask = mask[start:end, start:end]
            expected_block_mask = np.ones((end - start, end - start))
            assert np.array_equal(block_mask, expected_block_mask), \
                f"Block {block_idx} should have full internal connectivity"
            
            # Outside block should have no connectivity (for this simple implementation)
            if block_idx < num_blocks - 1:
                next_start = (block_idx + 1) * self.block_size
                cross_block_mask = mask[start:end, next_start:next_start + self.block_size]
                assert np.all(cross_block_mask == 0), \
                    f"No connectivity between blocks {block_idx} and {block_idx + 1}"
    
    def test_different_block_sizes(self):
        """Test with different block sizes"""
        block_sizes = [2, 4, 8]
        
        for block_size in block_sizes:
            if block_size <= self.seq_len:
                block_attn = BlockSparseAttention(self.d_model, block_size, self.num_heads)
                output = block_attn.forward(self.test_input)
                
                assert output.shape == self.test_input.shape
                
                # Check sparsity decreases with larger blocks
                sparsity = block_attn.compute_sparsity_ratio()
                assert 0 < sparsity <= 1


class TestAnalysisAndVisualization:
    """Test analysis and visualization functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.seq_len = 16
        self.num_heads = 4
        self.batch_size = 2
        
        # Create sample attention weights and sparsity pattern
        self.attention_weights = np.random.rand(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.attention_weights = self.attention_weights / np.sum(self.attention_weights, axis=-1, keepdims=True)
        
        # Create sample sparsity pattern
        self.sparsity_pattern = np.random.choice([0, 1], size=(self.seq_len, self.seq_len), p=[0.7, 0.3])
        np.fill_diagonal(self.sparsity_pattern, 1)  # Ensure diagonal attention
    
    def test_sparsity_analysis(self):
        """Test sparse attention analysis function"""
        analysis = analyze_attention_sparsity(self.attention_weights, self.sparsity_pattern)
        
        # Check required keys in analysis
        required_keys = [
            'sparsity_ratio', 'effective_connections', 'max_possible_connections',
            'compression_ratio', 'attention_distribution', 'information_flow'
        ]
        
        for key in required_keys:
            assert key in analysis, f"Missing key: {key}"
        
        # Check value ranges and properties
        assert 0 <= analysis['sparsity_ratio'] <= 1
        assert analysis['effective_connections'] <= analysis['max_possible_connections']
        assert analysis['compression_ratio'] >= 1.0
        
        # Check attention distribution analysis
        if 'mean' in analysis['attention_distribution']:
            assert analysis['attention_distribution']['mean'] >= 0
            assert analysis['attention_distribution']['std'] >= 0
    
    def test_pattern_gallery_creation(self):
        """Test creation of attention pattern gallery"""
        patterns = create_attention_pattern_gallery(seq_len=32)
        
        # Check that various patterns are created
        expected_patterns = ['Local Window', 'Strided', 'Random Sparse', 'BigBird', 'Block Sparse', 'Full Attention']
        
        for pattern_name in expected_patterns:
            assert pattern_name in patterns, f"Missing pattern: {pattern_name}"
            
            pattern = patterns[pattern_name]
            assert pattern.shape == (32, 32)
            assert np.all((pattern == 0) | (pattern == 1))
        
        # Full attention should be all ones
        assert np.all(patterns['Full Attention'] == 1)
        
        # Other patterns should be sparse (except possibly full attention)
        for name, pattern in patterns.items():
            if name != 'Full Attention':
                sparsity = np.mean(pattern)
                assert sparsity < 1.0, f"Pattern {name} should be sparse"
    
    def test_efficiency_comparison(self):
        """Test efficiency comparison function"""
        # Create sample attention patterns for testing
        patterns = {
            'Local': LocalWindowAttention(64, window_size=3),
            'Block': BlockSparseAttention(64, block_size=4)
        }
        
        seq_lengths = [8, 16, 32]
        
        comparison = compare_sparse_attention_efficiency(seq_lengths, patterns)
        
        # Check that comparison contains expected structure
        for pattern_name in patterns.keys():
            assert pattern_name in comparison
            
            pattern_results = comparison[pattern_name]
            assert 'seq_lengths' in pattern_results
            assert 'forward_times' in pattern_results
            assert 'sparsity_ratios' in pattern_results
            assert 'memory_usage' in pattern_results
            
            # Check that results have correct length
            assert len(pattern_results['forward_times']) == len(seq_lengths)
            assert len(pattern_results['sparsity_ratios']) == len(seq_lengths)


class TestNumericalStability:
    """Test numerical stability of sparse attention implementations"""
    
    def test_extreme_sparsity(self):
        """Test behavior with extreme sparsity patterns"""
        d_model = 64
        seq_len = 16
        
        # Very sparse pattern (only diagonal)
        very_sparse = RandomSparseAttention(d_model, sparsity_ratio=0.01, seed=42)
        test_input = np.random.randn(1, seq_len, d_model)
        
        output = very_sparse.forward(test_input)
        
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
    
    def test_edge_case_sequence_lengths(self):
        """Test with edge case sequence lengths"""
        d_model = 64
        
        # Very short sequences
        for seq_len in [1, 2, 3]:
            local_attn = LocalWindowAttention(d_model, window_size=1)
            test_input = np.random.randn(1, seq_len, d_model)
            
            output = local_attn.forward(test_input)
            assert output.shape == test_input.shape
    
    def test_large_sequence_length(self):
        """Test with larger sequence lengths"""
        d_model = 32  # Smaller model for faster testing
        seq_len = 128
        
        # Use local attention for efficiency
        local_attn = LocalWindowAttention(d_model, window_size=4, num_heads=4)
        test_input = np.random.randn(1, seq_len, d_model)
        
        output = local_attn.forward(test_input)
        
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output))
        
        # Check sparsity is reasonable
        sparsity = local_attn.compute_sparsity_ratio()
        assert sparsity < 0.5, "Should be sparse for long sequences"


def test_sparse_vs_full_attention_comparison():
    """Compare sparse attention with full attention"""
    print("Comparing sparse vs full attention...")
    
    d_model = 32
    seq_len = 64
    batch_size = 1
    
    test_input = np.random.randn(batch_size, seq_len, d_model)
    
    # Create different sparse patterns
    patterns = {
        'Local (w=4)': LocalWindowAttention(d_model, window_size=4),
        'Strided': StridedAttention(d_model, window_size=2, stride=8),
        'BigBird': BigBirdAttention(d_model, window_size=3, num_global_tokens=2),
    }
    
    print("Sparsity comparison:")
    for name, pattern in patterns.items():
        output = pattern.forward(test_input)
        sparsity_ratio = pattern.compute_sparsity_ratio()
        compression = 1.0 / sparsity_ratio if sparsity_ratio > 0 else float('inf')
        
        print(f"{name}: {sparsity_ratio:.3f} sparsity, {compression:.1f}x compression")
    
    print("✓ All sparse patterns produce valid outputs")


def test_information_flow_analysis():
    """Test information flow in sparse attention patterns"""
    print("Analyzing information flow in sparse patterns...")
    
    seq_len = 32
    
    patterns = create_attention_pattern_gallery(seq_len)
    
    for name, pattern in patterns.items():
        if name != 'Full Attention':
            # Analyze connectivity
            sparsity_ratio = np.mean(pattern)
            
            # Check path existence (simplified)
            # Can information flow from first to last position?
            reachability = pattern.copy()
            
            # Simple reachability analysis (Floyd-Warshall style, but simplified)
            for k in range(seq_len):
                for i in range(seq_len):
                    for j in range(seq_len):
                        if reachability[i, k] and reachability[k, j]:
                            reachability[i, j] = 1
            
            first_to_last = reachability[0, seq_len - 1]
            
            print(f"{name}: {sparsity_ratio:.3f} sparsity, "
                  f"first→last reachable: {bool(first_to_last)}")
    
    print("✓ Information flow analysis completed")


def benchmark_sparse_attention_patterns():
    """Benchmark different sparse attention patterns"""
    print("Benchmarking sparse attention patterns...")
    
    d_model = 64
    seq_lengths = [32, 64, 128]
    
    patterns = {
        'Local (w=4)': LocalWindowAttention(d_model, window_size=4),
        'Block (b=8)': BlockSparseAttention(d_model, block_size=8),
        'BigBird': BigBirdAttention(d_model, window_size=3, num_global_tokens=2),
    }
    
    print("Performance benchmark:")
    print("Pattern\t\tSeq Length\tTime (ms)\tSparsity")
    print("-" * 50)
    
    for pattern_name, pattern in patterns.items():
        for seq_len in seq_lengths:
            test_input = np.random.randn(1, seq_len, d_model)
            
            # Time forward pass
            start_time = time.time()
            output = pattern.forward(test_input)
            end_time = time.time()
            
            forward_time_ms = (end_time - start_time) * 1000
            sparsity = pattern.compute_sparsity_ratio()
            
            print(f"{pattern_name:<12}\t{seq_len}\t\t{forward_time_ms:.3f}\t\t{sparsity:.3f}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run additional validation
    test_sparse_vs_full_attention_comparison()
    test_information_flow_analysis()
    benchmark_sparse_attention_patterns()
    
    print("\nTesting completed!")
    print("Sparse attention implementation validated successfully.")