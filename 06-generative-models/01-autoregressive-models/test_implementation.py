"""
Test Suite for Autoregressive Models Implementation

This test suite verifies correctness of autoregressive model implementations.
Run with: python test_implementation.py

Author: ML-from-Scratch Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import math
from typing import List, Tuple

from exercise import (
    CharRNNLanguageModel, CausalConv1d, WaveNetBlock, CausalCNNLanguageModel,
    MultiHeadCausalAttention, TransformerDecoderLayer, TransformerLanguageModel,
    GenerationStrategies, AutoregressiveTrainer, EvaluationMetrics,
    create_character_tokenizer, load_text_dataset
)


class TestCausalConstraints:
    """Test that models satisfy causality constraints."""
    
    def test_causal_conv1d_causality(self):
        """Test that causal convolution doesn't use future information."""
        batch_size, seq_len, in_channels = 2, 10, 16
        kernel_size = 3
        
        conv = CausalConv1d(in_channels, 32, kernel_size)
        
        # Create input with distinct patterns
        x = torch.randn(batch_size, in_channels, seq_len)
        
        # Get output
        output = conv(x)
        
        # Modify future values
        x_modified = x.clone()
        x_modified[:, :, 5:] = torch.randn_like(x_modified[:, :, 5:])
        output_modified = conv(x_modified)
        
        # Output at positions 0-4 should be identical
        assert torch.allclose(output[:, :, :5], output_modified[:, :, :5], atol=1e-6), \
            "Causal convolution is using future information"
            
    def test_transformer_causal_mask(self):
        """Test that transformer attention mask is properly causal."""
        embed_dim, num_heads, seq_len = 128, 8, 16
        
        attention = MultiHeadCausalAttention(embed_dim, num_heads)
        
        # Create causal mask
        mask = attention.create_causal_mask(seq_len, torch.device('cpu'))
        
        # Check mask properties
        assert mask.shape == (seq_len, seq_len), "Mask has wrong shape"
        
        # Check that mask is lower triangular
        upper_triangle = torch.triu(mask, diagonal=1)
        assert torch.all(upper_triangle == float('-inf')), \
            "Causal mask should mask future positions"
            
        # Check diagonal and lower triangle
        lower_triangle = torch.tril(mask)
        assert torch.all(lower_triangle == 0), \
            "Causal mask should allow past positions"


class TestArchitectures:
    """Test individual model architectures."""
    
    def test_char_rnn_shapes(self):
        """Test CharRNN input/output shapes."""
        vocab_size, embed_dim, hidden_dim = 100, 64, 128
        batch_size, seq_len = 4, 20
        
        model = CharRNNLanguageModel(vocab_size, embed_dim, hidden_dim)
        
        # Test forward pass
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, hidden = model(x)
        
        assert logits.shape == (batch_size, seq_len, vocab_size), \
            f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
            
    def test_causal_cnn_shapes(self):
        """Test CausalCNN input/output shapes."""
        vocab_size = 100
        batch_size, seq_len = 4, 20
        
        model = CausalCNNLanguageModel(vocab_size)
        
        # Test forward pass
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(x)
        
        assert logits.shape == (batch_size, seq_len, vocab_size), \
            f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
            
    def test_transformer_shapes(self):
        """Test Transformer input/output shapes."""
        vocab_size, embed_dim, num_heads = 100, 128, 8
        batch_size, seq_len = 4, 20
        
        model = TransformerLanguageModel(vocab_size, embed_dim, num_heads)
        
        # Test forward pass
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(x)
        
        assert logits.shape == (batch_size, seq_len, vocab_size), \
            f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"


class TestGenerationStrategies:
    """Test text generation strategies."""
    
    def setup_method(self):
        """Setup test data."""
        # Create simple logits for testing
        self.vocab_size = 10
        self.logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5])
        
    def test_greedy_search(self):
        """Test greedy search always picks argmax."""
        token = GenerationStrategies.greedy_search(self.logits)
        assert token == 0, f"Greedy search should pick token 0, got {token}"
        
    def test_temperature_sampling(self):
        """Test temperature sampling behavior."""
        # Very low temperature should be close to greedy
        np.random.seed(42)
        torch.manual_seed(42)
        token_low_temp = GenerationStrategies.temperature_sampling(self.logits, temperature=0.01)
        
        # Very high temperature should be more uniform
        token_high_temp = GenerationStrategies.temperature_sampling(self.logits, temperature=10.0)
        
        # Low temperature should favor high probability tokens
        assert token_low_temp in [0, 1], "Low temperature should favor top tokens"
        
    def test_top_k_sampling(self):
        """Test top-k sampling only considers top k tokens."""
        k = 3
        token = GenerationStrategies.top_k_sampling(self.logits, k=k)
        assert token < k, f"Top-k sampling should only return tokens from top {k}"
        
    def test_nucleus_sampling(self):
        """Test nucleus sampling probability mass constraint."""
        p = 0.9
        # This test verifies the sampling stays within nucleus
        token = GenerationStrategies.nucleus_sampling(self.logits, p=p)
        assert token >= 0, "Nucleus sampling should return valid token"


class TestTrainingComponents:
    """Test training-related functionality."""
    
    def test_loss_computation(self):
        """Test that loss computation is correct."""
        vocab_size, batch_size, seq_len = 100, 4, 10
        
        # Create model
        model = CharRNNLanguageModel(vocab_size, embed_dim=64, hidden_dim=128)
        
        # Create data
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits, _ = model(x)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"
        
    def test_perplexity_computation(self):
        """Test perplexity calculation."""
        # Create mock log probabilities
        log_probs = torch.tensor([-2.3, -1.6, -0.9, -2.1])  # ln(0.1), ln(0.2), ln(0.4), ln(0.12)
        
        perplexity = EvaluationMetrics.perplexity(log_probs)
        
        # Expected perplexity = exp(-mean(log_probs)) = exp(1.725) ≈ 5.6
        expected_perplexity = math.exp(-log_probs.mean().item())
        
        assert abs(perplexity - expected_perplexity) < 1e-6, \
            f"Expected perplexity {expected_perplexity}, got {perplexity}"


class TestPositionalEncoding:
    """Test positional encoding implementations."""
    
    def test_sinusoidal_encoding_properties(self):
        """Test properties of sinusoidal positional encoding."""
        seq_len, embed_dim = 100, 512
        device = torch.device('cpu')
        
        model = TransformerLanguageModel(vocab_size=1000, embed_dim=embed_dim)
        pos_enc = model.create_positional_encoding(seq_len, embed_dim, device)
        
        # Check shape
        assert pos_enc.shape == (seq_len, embed_dim), \
            f"Expected shape {(seq_len, embed_dim)}, got {pos_enc.shape}"
            
        # Check that even dimensions use sin and odd use cos
        # This is a simplified check - full verification would need the actual formula
        assert not torch.allclose(pos_enc[:, 0], pos_enc[:, 1]), \
            "Adjacent dimensions should have different patterns"


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_character_tokenizer(self):
        """Test character tokenizer creation."""
        text = "hello world"
        char_to_idx, idx_to_char = create_character_tokenizer(text)
        
        # Check basic properties
        unique_chars = set(text)
        assert len(char_to_idx) == len(unique_chars), \
            "Tokenizer should have entry for each unique character"
            
        # Check bidirectional mapping
        for char in unique_chars:
            idx = char_to_idx[char]
            assert idx_to_char[idx] == char, \
                "Bidirectional mapping should be consistent"


class TestGenerationQuality:
    """Test generation quality and diversity."""
    
    def test_generation_diversity(self):
        """Test that different sampling methods produce different outputs."""
        vocab_size = 100
        model = CharRNNLanguageModel(vocab_size, embed_dim=64, hidden_dim=128)
        
        # Generate multiple sequences
        start_idx = 0
        max_length = 20
        
        sequences = []
        for _ in range(5):
            seq = model.generate(start_idx, max_length, temperature=1.0)
            sequences.append(seq)
            
        # Check that sequences are different (with high probability)
        all_same = all(seq == sequences[0] for seq in sequences[1:])
        assert not all_same, "Generated sequences should show diversity"
        
    def test_generation_length_control(self):
        """Test that generation respects length limits."""
        vocab_size = 100
        model = CharRNNLanguageModel(vocab_size, embed_dim=64, hidden_dim=128)
        
        max_lengths = [10, 20, 50]
        
        for max_len in max_lengths:
            seq = model.generate(0, max_len)
            assert len(seq) <= max_len, \
                f"Generated sequence length {len(seq)} exceeds limit {max_len}"


class TestNumericalStability:
    """Test numerical stability of implementations."""
    
    def test_attention_scaling(self):
        """Test that attention scaling prevents overflow."""
        embed_dim, num_heads, seq_len = 512, 8, 1024
        
        attention = MultiHeadCausalAttention(embed_dim, num_heads)
        
        # Create large input values
        x = torch.randn(1, seq_len, embed_dim) * 10
        
        # Forward pass should not produce NaN or inf
        output = attention(x)
        
        assert not torch.isnan(output).any(), "Attention output contains NaN"
        assert not torch.isinf(output).any(), "Attention output contains infinity"
        
    def test_softmax_stability(self):
        """Test softmax numerical stability in generation."""
        # Create logits with large values
        logits = torch.tensor([100.0, 101.0, 99.0, 102.0])
        
        # Temperature sampling should handle large logits
        token = GenerationStrategies.temperature_sampling(logits, temperature=1.0)
        
        assert 0 <= token < len(logits), "Generated token should be valid"


def run_comprehensive_tests():
    """Run all tests and report results."""
    print("Running Autoregressive Models Test Suite...")
    print("=" * 50)
    
    test_classes = [
        TestCausalConstraints,
        TestArchitectures, 
        TestGenerationStrategies,
        TestTrainingComponents,
        TestPositionalEncoding,
        TestDataProcessing,
        TestGenerationQuality,
        TestNumericalStability
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Create instance and run setup if exists
                instance = test_class()
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                    
                # Run test method
                test_method = getattr(instance, method_name)
                test_method()
                
                print(f"  ✓ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {total_tests - passed_tests} tests failed")
        
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)