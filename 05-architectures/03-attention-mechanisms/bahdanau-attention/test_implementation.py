"""
Test suite for Bahdanau Attention implementation

Comprehensive tests for additive attention mechanisms, encoder-decoder architectures,
and attention alignment analysis.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from exercise import *


class TestBahdanauAttention:
    """Test BahdanauAttention implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.seq_len = 8
        self.decoder_hidden_size = 64
        self.encoder_hidden_size = 128
        self.attention_size = 32
        
        self.decoder_state = np.random.randn(self.batch_size, self.decoder_hidden_size)
        self.encoder_states = np.random.randn(self.batch_size, self.seq_len, self.encoder_hidden_size)
        self.attention = BahdanauAttention(
            self.decoder_hidden_size, self.encoder_hidden_size, self.attention_size
        )
    
    def test_attention_output_shapes(self):
        """Test that Bahdanau attention produces correct output shapes"""
        context_vector, attention_weights = self.attention.compute_attention(
            self.decoder_state, self.encoder_states
        )
        
        expected_context_shape = (self.batch_size, self.encoder_hidden_size)
        expected_weights_shape = (self.batch_size, self.seq_len)
        
        assert context_vector.shape == expected_context_shape
        assert attention_weights.shape == expected_weights_shape
    
    def test_attention_weights_properties(self):
        """Test mathematical properties of attention weights"""
        context_vector, attention_weights = self.attention.compute_attention(
            self.decoder_state, self.encoder_states
        )
        
        # Should be non-negative
        assert np.all(attention_weights >= 0), "Attention weights should be non-negative"
        
        # Should sum to 1 (softmax property)
        row_sums = np.sum(attention_weights, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Attention weights should sum to 1"
        
        # Should be in [0, 1] range
        assert np.all(attention_weights <= 1.0), "Attention weights should be <= 1"
    
    def test_context_vector_computation(self):
        """Test that context vector is correctly computed as weighted sum"""
        context_vector, attention_weights = self.attention.compute_attention(
            self.decoder_state, self.encoder_states
        )
        
        # Manual computation
        manual_context = np.sum(
            attention_weights[:, :, None] * self.encoder_states, axis=1
        )
        
        assert np.allclose(context_vector, manual_context, atol=1e-5), \
            "Context vector should match manual computation"
    
    def test_different_dimensions(self):
        """Test with different dimensional configurations"""
        test_configs = [
            (32, 64, 16),   # Small dimensions
            (128, 256, 64), # Medium dimensions  
            (64, 64, 32),   # Equal encoder/decoder dimensions
        ]
        
        for dec_dim, enc_dim, att_dim in test_configs:
            attention = BahdanauAttention(dec_dim, enc_dim, att_dim)
            
            decoder_state = np.random.randn(1, dec_dim)
            encoder_states = np.random.randn(1, 5, enc_dim)
            
            context, weights = attention.compute_attention(decoder_state, encoder_states)
            
            assert context.shape == (1, enc_dim)
            assert weights.shape == (1, 5)
            assert np.allclose(np.sum(weights), 1.0)
    
    def test_sequence_length_variations(self):
        """Test with different sequence lengths"""
        for seq_len in [1, 5, 10, 20, 50]:
            encoder_states = np.random.randn(1, seq_len, self.encoder_hidden_size)
            decoder_state = np.random.randn(1, self.decoder_hidden_size)
            
            context, weights = self.attention.compute_attention(decoder_state, encoder_states)
            
            assert context.shape == (1, self.encoder_hidden_size)
            assert weights.shape == (1, seq_len)
            assert np.allclose(np.sum(weights), 1.0)
    
    def test_parameter_counting(self):
        """Test parameter counting"""
        param_count = self.attention.count_parameters()
        
        # Should count W_a, U_a, v_a
        expected_params = (
            self.attention_size * self.decoder_hidden_size +  # W_a
            self.attention_size * self.encoder_hidden_size +  # U_a  
            self.attention_size                               # v_a
        )
        
        assert param_count == expected_params
        
    def test_attention_storage(self):
        """Test that attention weights are stored correctly"""
        context, weights = self.attention.compute_attention(self.decoder_state, self.encoder_states)
        stored_weights = self.attention.get_attention_weights()
        
        assert np.array_equal(weights, stored_weights), "Stored weights should match computed weights"


class TestSimpleLSTM:
    """Test SimpleLSTM implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.seq_len = 5
        self.input_size = 32
        self.hidden_size = 64
        
        self.lstm = SimpleLSTM(self.input_size, self.hidden_size)
    
    def test_single_step_forward(self):
        """Test single LSTM step"""
        x_t = np.random.randn(self.batch_size, self.input_size)
        h_prev = np.random.randn(self.batch_size, self.hidden_size)
        c_prev = np.random.randn(self.batch_size, self.hidden_size)
        
        h_t, c_t = self.lstm.forward_step(x_t, h_prev, c_prev)
        
        assert h_t.shape == (self.batch_size, self.hidden_size)
        assert c_t.shape == (self.batch_size, self.hidden_size)
    
    def test_sequence_forward(self):
        """Test processing entire sequence"""
        inputs = np.random.randn(self.batch_size, self.seq_len, self.input_size)
        
        hidden_states, cell_states = self.lstm.forward_sequence(inputs)
        
        assert len(hidden_states) == self.seq_len
        assert len(cell_states) == self.seq_len
        
        for h, c in zip(hidden_states, cell_states):
            assert h.shape == (self.batch_size, self.hidden_size)
            assert c.shape == (self.batch_size, self.hidden_size)
    
    def test_initial_state(self):
        """Test with custom initial state"""
        inputs = np.random.randn(self.batch_size, self.seq_len, self.input_size)
        h_0 = np.random.randn(self.batch_size, self.hidden_size)
        c_0 = np.random.randn(self.batch_size, self.hidden_size)
        
        hidden_states, cell_states = self.lstm.forward_sequence(inputs, (h_0, c_0))
        
        # First output should be influenced by initial state
        assert hidden_states[0].shape == h_0.shape
        assert cell_states[0].shape == c_0.shape


class TestEncoder:
    """Test Encoder implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.vocab_size = 1000
        self.embedding_dim = 128
        self.hidden_size = 256
        self.batch_size = 2
        self.seq_len = 10
        
        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.hidden_size)
    
    def test_encoding_output_shape(self):
        """Test encoder output shape"""
        input_ids = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        encoder_states = self.encoder.forward(input_ids)
        
        # Bidirectional encoder should output 2*hidden_size
        expected_shape = (self.batch_size, self.seq_len, 2 * self.hidden_size)
        assert encoder_states.shape == expected_shape
    
    def test_different_sequence_lengths(self):
        """Test encoder with different sequence lengths"""
        for seq_len in [1, 5, 15, 25]:
            input_ids = np.random.randint(0, self.vocab_size, (1, seq_len))
            encoder_states = self.encoder.forward(input_ids)
            
            expected_shape = (1, seq_len, 2 * self.hidden_size)
            assert encoder_states.shape == expected_shape
    
    def test_embedding_lookup(self):
        """Test that embeddings are looked up correctly"""
        # Test with specific token IDs
        input_ids = np.array([[0, 1, 2]])
        encoder_states = self.encoder.forward(input_ids)
        
        # Should produce valid output
        assert not np.any(np.isnan(encoder_states))
        assert encoder_states.shape == (1, 3, 2 * self.hidden_size)


class TestAttentionDecoder:
    """Test AttentionDecoder implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.vocab_size = 1000
        self.embedding_dim = 128
        self.hidden_size = 256
        self.encoder_hidden_size = 512  # 2 * 256 from bidirectional encoder
        self.batch_size = 2
        self.seq_len = 8
        
        self.decoder = AttentionDecoder(
            self.vocab_size, self.embedding_dim, self.hidden_size, 
            self.encoder_hidden_size
        )
    
    def test_single_decoding_step(self):
        """Test single decoder step with attention"""
        target_token = np.random.randint(0, self.vocab_size, (self.batch_size,))
        decoder_state = np.random.randn(self.batch_size, self.hidden_size)
        cell_state = np.random.randn(self.batch_size, self.hidden_size)
        encoder_states = np.random.randn(self.batch_size, self.seq_len, self.encoder_hidden_size)
        
        output_logits, new_decoder_state, new_cell_state, attention_weights = self.decoder.forward_step(
            target_token, decoder_state, cell_state, encoder_states
        )
        
        # Check shapes
        assert output_logits.shape == (self.batch_size, self.vocab_size)
        assert new_decoder_state.shape == decoder_state.shape
        assert new_cell_state.shape == cell_state.shape
        assert attention_weights.shape == (self.batch_size, self.seq_len)
        
        # Check attention properties
        assert np.allclose(np.sum(attention_weights, axis=1), 1.0)
    
    def test_sequence_decoding(self):
        """Test decoding entire sequence with teacher forcing"""
        target_ids = np.random.randint(0, self.vocab_size, (self.batch_size, 5))
        encoder_states = np.random.randn(self.batch_size, self.seq_len, self.encoder_hidden_size)
        
        output_logits_seq, attention_weights_seq = self.decoder.forward_sequence(
            target_ids, encoder_states
        )
        
        assert len(output_logits_seq) == 5
        assert len(attention_weights_seq) == 5
        
        for logits, weights in zip(output_logits_seq, attention_weights_seq):
            assert logits.shape == (self.batch_size, self.vocab_size)
            assert weights.shape == (self.batch_size, self.seq_len)
    
    def test_attention_history_tracking(self):
        """Test that attention history is tracked correctly"""
        target_ids = np.random.randint(0, self.vocab_size, (1, 3))
        encoder_states = np.random.randn(1, self.seq_len, self.encoder_hidden_size)
        
        self.decoder.attention_history = []  # Reset history
        
        output_logits_seq, attention_weights_seq = self.decoder.forward_sequence(
            target_ids, encoder_states
        )
        
        history = self.decoder.get_attention_history()
        assert len(history) == 3  # Should have 3 attention matrices
        
        for attention in history:
            assert attention.shape == (1, self.seq_len)


class TestSeq2SeqWithAttention:
    """Test complete Seq2Seq model"""
    
    def setup_method(self):
        """Setup test data"""
        self.source_vocab_size = 1000
        self.target_vocab_size = 800
        self.batch_size = 2
        self.source_seq_len = 8
        self.target_seq_len = 6
        
        self.model = Seq2SeqWithAttention(
            self.source_vocab_size, self.target_vocab_size,
            embedding_dim=128, hidden_size=256
        )
    
    def test_forward_pass(self):
        """Test complete forward pass"""
        source_ids = np.random.randint(0, self.source_vocab_size, (self.batch_size, self.source_seq_len))
        target_ids = np.random.randint(0, self.target_vocab_size, (self.batch_size, self.target_seq_len))
        
        output_logits, attention_weights = self.model.forward(source_ids, target_ids)
        
        assert len(output_logits) == self.target_seq_len
        assert len(attention_weights) == self.target_seq_len
        
        for logits, weights in zip(output_logits, attention_weights):
            assert logits.shape == (self.batch_size, self.target_vocab_size)
            assert weights.shape == (self.batch_size, self.source_seq_len)
    
    def test_generation(self):
        """Test sequence generation"""
        source_ids = np.random.randint(0, self.source_vocab_size, (1, self.source_seq_len))
        
        generated_ids, attention_weights = self.model.generate(source_ids, max_length=10)
        
        assert generated_ids.shape[0] == 1  # batch_size
        assert generated_ids.shape[1] <= 10  # max_length
        assert len(attention_weights) == generated_ids.shape[1]
        
        for weights in attention_weights:
            assert weights.shape == (1, self.source_seq_len)
    
    def test_different_temperatures(self):
        """Test generation with different temperatures"""
        source_ids = np.random.randint(0, self.source_vocab_size, (1, 5))
        
        for temp in [0.0, 0.5, 1.0, 2.0]:
            generated_ids, _ = self.model.generate(source_ids, max_length=5, temperature=temp)
            
            assert generated_ids.shape[0] == 1
            assert generated_ids.shape[1] <= 5


class TestAttentionAnalysis:
    """Test attention analysis functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.target_len = 5
        self.source_len = 8
        
        # Create sample attention weights sequence
        self.attention_sequence = []
        for _ in range(self.target_len):
            weights = np.random.rand(2, self.source_len)  # batch_size=2
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            self.attention_sequence.append(weights)
    
    def test_attention_pattern_analysis(self):
        """Test attention pattern analysis"""
        analysis = analyze_attention_patterns(self.attention_sequence)
        
        required_keys = [
            'attention_matrix', 'attention_entropy', 'peak_positions',
            'attention_spread', 'monotonic_score', 'mean_entropy', 'mean_spread'
        ]
        
        for key in required_keys:
            assert key in analysis, f"Missing key: {key}"
        
        # Check shapes and properties
        assert analysis['attention_matrix'].shape == (self.target_len, self.source_len)
        assert len(analysis['attention_entropy']) == self.target_len
        assert len(analysis['peak_positions']) == self.target_len
        assert len(analysis['attention_spread']) == self.target_len
        
        # Check value ranges
        assert 0 <= analysis['monotonic_score'] <= 1
        assert analysis['mean_entropy'] >= 0
        assert analysis['mean_spread'] >= 0
    
    def test_monotonic_attention_detection(self):
        """Test detection of monotonic attention patterns"""
        # Create perfectly monotonic attention
        monotonic_sequence = []
        for t in range(5):
            weights = np.zeros((1, 8))
            weights[0, min(t, 7)] = 1.0  # Attention moves forward
            monotonic_sequence.append(weights)
        
        analysis = analyze_attention_patterns(monotonic_sequence)
        
        # Should have high monotonic score
        assert analysis['monotonic_score'] >= 0.75, "Should detect monotonic pattern"
    
    def test_uniform_attention_analysis(self):
        """Test analysis of uniform attention patterns"""
        # Create uniform attention
        uniform_sequence = []
        for _ in range(3):
            weights = np.ones((1, 5)) / 5  # Uniform distribution
            uniform_sequence.append(weights)
        
        analysis = analyze_attention_patterns(uniform_sequence)
        
        # Uniform attention should have high entropy
        max_entropy = np.log(5)  # log(source_len)
        mean_entropy = analysis['mean_entropy']
        assert mean_entropy > 0.8 * max_entropy, "Uniform attention should have high entropy"


class TestVisualization:
    """Test attention visualization functions"""
    
    def test_attention_alignment_visualization(self):
        """Test attention alignment heatmap creation"""
        attention_matrix = np.random.rand(4, 6)
        attention_matrix = attention_matrix / np.sum(attention_matrix, axis=1, keepdims=True)
        
        source_tokens = [f"src_{i}" for i in range(6)]
        target_tokens = [f"tgt_{i}" for i in range(4)]
        
        # This should not crash
        try:
            # Note: In headless environment, plt.show() might not work
            # but the function should still execute without errors
            visualize_attention_alignment(attention_matrix, source_tokens, target_tokens)
            assert True, "Visualization function executed successfully"
        except Exception as e:
            # Allow for display-related errors in testing environment
            if "display" in str(e).lower() or "tkinter" in str(e).lower():
                assert True, "Display error expected in test environment"
            else:
                raise e


class TestNumericalStability:
    """Test numerical stability of attention mechanisms"""
    
    def test_large_attention_scores(self):
        """Test behavior with large attention scores"""
        attention = BahdanauAttention(64, 128, 32)
        
        # Create decoder/encoder states that might produce large scores
        decoder_state = np.ones((1, 64)) * 100
        encoder_states = np.ones((1, 5, 128)) * 100
        
        context, weights = attention.compute_attention(decoder_state, encoder_states)
        
        # Should not produce NaN or inf
        assert not np.any(np.isnan(context))
        assert not np.any(np.isinf(context))
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
        
        # Weights should still sum to 1
        assert np.allclose(np.sum(weights), 1.0)
    
    def test_zero_inputs(self):
        """Test behavior with zero inputs"""
        attention = BahdanauAttention(64, 128, 32)
        
        decoder_state = np.zeros((1, 64))
        encoder_states = np.zeros((1, 5, 128))
        
        context, weights = attention.compute_attention(decoder_state, encoder_states)
        
        # Should produce valid outputs
        assert not np.any(np.isnan(context))
        assert not np.any(np.isnan(weights))
        assert np.allclose(np.sum(weights), 1.0)
    
    def test_small_inputs(self):
        """Test behavior with very small inputs"""
        attention = BahdanauAttention(64, 128, 32)
        
        decoder_state = np.ones((1, 64)) * 1e-8
        encoder_states = np.ones((1, 5, 128)) * 1e-8
        
        context, weights = attention.compute_attention(decoder_state, encoder_states)
        
        # Should still work correctly
        assert not np.any(np.isnan(context))
        assert not np.any(np.isnan(weights))
        assert np.allclose(np.sum(weights), 1.0)


def test_bahdanau_vs_modern_attention():
    """Compare Bahdanau attention with modern attention concepts"""
    print("Comparing Bahdanau vs modern attention...")
    
    # Test computational complexity
    seq_lengths = [10, 20, 50]
    hidden_size = 128
    
    for seq_len in seq_lengths:
        comparison = compare_attention_mechanisms(seq_len, hidden_size)
        print(f"Sequence length {seq_len}: {comparison}")
    
    # Test key differences
    print("\nKey differences:")
    print("- Bahdanau: Additive attention with learned alignment")
    print("- Modern: Multiplicative attention with query-key-value")
    print("- Bahdanau: Sequential decoder processing")
    print("- Modern: Parallel processing with self-attention")


def test_attention_interpretability():
    """Test interpretability features of Bahdanau attention"""
    print("Testing attention interpretability...")
    
    # Create simple test case
    vocab_size = 100
    model = Seq2SeqWithAttention(vocab_size, vocab_size, embedding_dim=64, hidden_size=128)
    
    # Simulate translation task
    source = np.array([[10, 20, 30, 40, 1]])  # Source sequence with EOS
    target = np.array([[50, 60, 70, 1]])      # Target sequence with EOS
    
    output_logits, attention_weights = model.forward(source, target)
    
    print(f"Source length: {source.shape[1]}")
    print(f"Target length: {target.shape[1]}")
    print(f"Attention weights shape: {attention_weights[0].shape}")
    
    # Analyze attention patterns
    analysis = analyze_attention_patterns(attention_weights)
    print(f"Mean attention entropy: {analysis['mean_entropy']:.3f}")
    print(f"Monotonic alignment score: {analysis['monotonic_score']:.3f}")
    
    print("✓ Attention weights provide interpretable alignment information")


def test_seq2seq_properties():
    """Test sequence-to-sequence model properties"""
    print("Testing seq2seq model properties...")
    
    source_vocab = 1000
    target_vocab = 800
    model = Seq2SeqWithAttention(source_vocab, target_vocab)
    
    # Test variable length sequences
    test_cases = [
        ((2, 5), (2, 3)),   # Short sequences
        ((1, 10), (1, 8)),  # Medium sequences  
        ((1, 20), (1, 15)), # Longer sequences
    ]
    
    for source_shape, target_shape in test_cases:
        source_ids = np.random.randint(0, source_vocab, source_shape)
        target_ids = np.random.randint(0, target_vocab, target_shape)
        
        output_logits, attention_weights = model.forward(source_ids, target_ids)
        
        assert len(output_logits) == target_shape[1]
        assert len(attention_weights) == target_shape[1]
        
        print(f"✓ Handled {source_shape} → {target_shape} successfully")
    
    print("✓ Model handles variable sequence lengths correctly")


def benchmark_attention_mechanisms():
    """Benchmark different attention mechanisms"""
    print("Benchmarking attention mechanisms...")
    
    configs = [
        (32, 64, 16),   # Small
        (128, 256, 64), # Medium
        (256, 512, 128) # Large
    ]
    
    for dec_size, enc_size, att_size in configs:
        attention = BahdanauAttention(dec_size, enc_size, att_size)
        
        decoder_state = np.random.randn(1, dec_size)
        encoder_states = np.random.randn(1, 20, enc_size)
        
        # Time multiple runs
        start_time = time.time()
        for _ in range(100):
            context, weights = attention.compute_attention(decoder_state, encoder_states)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        params = attention.count_parameters()
        
        print(f"Config ({dec_size}, {enc_size}, {att_size}): {avg_time:.6f}s, {params:,} params")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run additional validation
    test_bahdanau_vs_modern_attention()
    test_attention_interpretability()
    test_seq2seq_properties()
    benchmark_attention_mechanisms()
    
    print("\nTesting completed!")
    print("Bahdanau attention implementation validated successfully.")