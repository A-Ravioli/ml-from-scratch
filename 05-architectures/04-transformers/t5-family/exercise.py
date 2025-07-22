"""
T5 Family Implementation Exercise

Implementation of T5-style encoder-decoder Transformers for text-to-text tasks
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time
import math
import re


class RelativePositionBias:
    """Relative position bias used in T5 instead of absolute position embeddings"""
    
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        
        # Learnable relative position bias
        self.relative_attention_bias = np.random.randn(num_buckets, num_heads) * 0.02
    
    def _relative_position_bucket(self, relative_position: np.ndarray) -> np.ndarray:
        """Convert relative positions to bucket indices"""
        ret = 0
        n = -relative_position
        
        # Half of buckets are for exact increments in positions
        num_buckets = self.num_buckets
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        # Other half for logarithmically bigger bins
        val_if_large = max_exact + (
            np.log(n.astype(float) / max_exact) / np.log(self.max_distance / max_exact) 
            * (num_buckets - max_exact)
        ).astype(int)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        
        ret += np.where(is_small, n, val_if_large)
        return ret
    
    def forward(self, query_length: int, key_length: int, bidirectional: bool = True) -> np.ndarray:
        """Compute relative position bias"""
        context_position = np.arange(query_length)[:, None]
        memory_position = np.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        
        if not bidirectional:
            relative_position = np.maximum(relative_position, 0)
        
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias[rp_bucket]  # [query_len, key_len, num_heads]
        values = values.transpose(2, 0, 1)  # [num_heads, query_len, key_len]
        
        return values[None, :, :, :]  # [1, num_heads, query_len, key_len]


class T5Attention:
    """T5-style multi-head attention with relative position bias"""
    
    def __init__(self, d_model: int, num_heads: int, is_decoder: bool = False, 
                 has_relative_bias: bool = True):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_decoder = is_decoder
        self.has_relative_bias = has_relative_bias
        
        # Query, Key, Value projections
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # Relative position bias for first layer only
        if has_relative_bias:
            self.position_bias = RelativePositionBias(num_heads)
        else:
            self.position_bias = None
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None, 
                position_bias: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        batch_size, query_len, d_model = query.shape
        key_len = key.shape[1]
        
        # Project to Q, K, V
        Q = np.dot(query, self.W_q).reshape(batch_size, query_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(key, self.W_k).reshape(batch_size, key_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(value, self.W_v).reshape(batch_size, key_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        
        # Add relative position bias
        if self.position_bias is not None:
            bias = self.position_bias.forward(query_len, key_len, bidirectional=not self.is_decoder)
            scores += bias
        elif position_bias is not None:
            scores += position_bias
        
        # Apply mask
        if mask is not None:
            scores += mask * -1e9
        
        attn_weights = self.softmax(scores)
        attn_output = np.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, query_len, d_model)
        output = np.dot(attn_output, self.W_o)
        
        return output, bias if self.position_bias is not None else None
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class T5MLP:
    """T5 MLP with ReLU activation and gating"""
    
    def __init__(self, d_model: int, d_ff: int):
        # T5 uses gated linear unit (GLU) variant
        self.W_1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.W_2 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model) 
        self.W_o = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = np.maximum(0, np.dot(x, self.W_1))  # ReLU activation
        hidden = np.dot(x, self.W_2)
        return np.dot(gate * hidden, self.W_o)  # Gated linear unit


class T5LayerNorm:
    """T5 uses RMSNorm instead of LayerNorm"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.weight = np.ones(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # RMS normalization
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return x * self.weight / np.sqrt(variance + self.eps)


class T5Block:
    """T5 Transformer block (encoder or decoder)"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, is_decoder: bool = False,
                 has_relative_bias: bool = True, is_cross_attention: bool = False):
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention
        
        # Pre-norm architecture
        self.norm1 = T5LayerNorm(d_model)
        self.self_attention = T5Attention(d_model, num_heads, is_decoder, has_relative_bias)
        
        if is_decoder and not is_cross_attention:
            self.norm2 = T5LayerNorm(d_model)
            self.cross_attention = T5Attention(d_model, num_heads, is_decoder=False, has_relative_bias=False)
        
        self.norm_ff = T5LayerNorm(d_model)
        self.mlp = T5MLP(d_model, d_ff)
    
    def forward(self, x: np.ndarray, encoder_output: Optional[np.ndarray] = None,
                self_attn_mask: Optional[np.ndarray] = None,
                cross_attn_mask: Optional[np.ndarray] = None,
                position_bias: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Self-attention
        normed_x = self.norm1.forward(x)
        attn_output, new_position_bias = self.self_attention.forward(
            normed_x, normed_x, normed_x, self_attn_mask, position_bias
        )
        x = x + attn_output
        
        # Cross-attention (decoder only)
        if self.is_decoder and not self.is_cross_attention and encoder_output is not None:
            normed_x = self.norm2.forward(x)
            cross_attn_output, _ = self.cross_attention.forward(
                normed_x, encoder_output, encoder_output, cross_attn_mask
            )
            x = x + cross_attn_output
        
        # Feed-forward
        normed_x = self.norm_ff.forward(x)
        ff_output = self.mlp.forward(normed_x)
        x = x + ff_output
        
        return x, new_position_bias


class T5Model:
    """Complete T5 encoder-decoder model"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, num_layers: int = 6):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Shared embeddings
        self.shared_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Encoder
        self.encoder_blocks = []
        for i in range(num_layers):
            has_bias = (i == 0)  # Only first layer has relative position bias
            self.encoder_blocks.append(T5Block(d_model, num_heads, d_ff, 
                                             is_decoder=False, has_relative_bias=has_bias))
        
        self.encoder_norm = T5LayerNorm(d_model)
        
        # Decoder
        self.decoder_blocks = []
        for i in range(num_layers):
            has_bias = (i == 0)  # Only first layer has relative position bias
            self.decoder_blocks.append(T5Block(d_model, num_heads, d_ff,
                                             is_decoder=True, has_relative_bias=has_bias))
        
        self.decoder_norm = T5LayerNorm(d_model)
        
        # Output head (tied weights with embeddings)
        self.lm_head = self.shared_embedding.T
    
    def encode(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode input sequence"""
        x = self.shared_embedding[input_ids]
        
        position_bias = None
        for i, block in enumerate(self.encoder_blocks):
            x, new_bias = block.forward(x, position_bias=position_bias)
            if i == 0:
                position_bias = new_bias
        
        return self.encoder_norm.forward(x)
    
    def decode(self, decoder_input_ids: np.ndarray, encoder_hidden_states: np.ndarray,
               decoder_attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode with encoder context"""
        x = self.shared_embedding[decoder_input_ids]
        
        # Create causal mask for decoder
        seq_len = decoder_input_ids.shape[1]
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, seq_len, seq_len]
        
        position_bias = None
        for i, block in enumerate(self.decoder_blocks):
            x, new_bias = block.forward(x, encoder_hidden_states, causal_mask, position_bias=position_bias)
            if i == 0:
                position_bias = new_bias
        
        x = self.decoder_norm.forward(x)
        return np.dot(x, self.lm_head)
    
    def forward(self, input_ids: np.ndarray, decoder_input_ids: np.ndarray) -> np.ndarray:
        """Full forward pass"""
        encoder_output = self.encode(input_ids)
        decoder_output = self.decode(decoder_input_ids, encoder_output)
        return decoder_output


class T5SpanDenoiser:
    """Span denoising pre-training for T5"""
    
    def __init__(self, vocab_size: int, sentinel_start: int = 32000):
        self.vocab_size = vocab_size
        self.sentinel_start = sentinel_start
        self.sentinel_tokens = [f"<extra_id_{i}>" for i in range(100)]
    
    def create_span_denoising_data(self, input_ids: np.ndarray, 
                                  corruption_rate: float = 0.15,
                                  mean_span_length: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create span denoising training data"""
        batch_size, seq_len = input_ids.shape
        corrupted_inputs = []
        targets = []
        
        for i in range(batch_size):
            sequence = input_ids[i].copy()
            
            # Determine spans to corrupt
            num_tokens_to_mask = max(1, int(seq_len * corruption_rate))
            spans = self._sample_spans(seq_len, num_tokens_to_mask, mean_span_length)
            
            # Create corrupted input and target
            corrupted_seq, target_seq = self._corrupt_spans(sequence, spans)
            
            corrupted_inputs.append(corrupted_seq)
            targets.append(target_seq)
        
        # Pad sequences to same length
        max_input_len = max(len(seq) for seq in corrupted_inputs)
        max_target_len = max(len(seq) for seq in targets)
        
        padded_inputs = np.zeros((batch_size, max_input_len), dtype=int)
        padded_targets = np.zeros((batch_size, max_target_len), dtype=int)
        
        for i in range(batch_size):
            padded_inputs[i, :len(corrupted_inputs[i])] = corrupted_inputs[i]
            padded_targets[i, :len(targets[i])] = targets[i]
        
        return padded_inputs, padded_targets
    
    def _sample_spans(self, seq_len: int, num_tokens_to_mask: int, 
                     mean_span_length: float) -> List[Tuple[int, int]]:
        """Sample spans for corruption"""
        spans = []
        masked_tokens = 0
        
        while masked_tokens < num_tokens_to_mask:
            # Sample span start
            start = np.random.randint(0, seq_len)
            
            # Sample span length from geometric distribution
            span_length = 1
            while np.random.random() < (1 / mean_span_length) and span_length < 10:
                span_length += 1
            
            end = min(start + span_length, seq_len)
            
            # Check for overlap with existing spans
            overlap = any(not (end <= s or start >= e) for s, e in spans)
            if not overlap:
                spans.append((start, end))
                masked_tokens += (end - start)
        
        return sorted(spans)
    
    def _corrupt_spans(self, sequence: np.ndarray, 
                      spans: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Corrupt spans and create target sequence"""
        corrupted = []
        target = []
        
        sentinel_id = 0
        last_end = 0
        
        for start, end in spans:
            # Add uncorrupted tokens before span
            corrupted.extend(sequence[last_end:start])
            
            # Add sentinel token to corrupted sequence
            corrupted.append(self.sentinel_start + sentinel_id)
            
            # Add sentinel token and original tokens to target
            target.append(self.sentinel_start + sentinel_id)
            target.extend(sequence[start:end])
            
            sentinel_id += 1
            last_end = end
        
        # Add remaining uncorrupted tokens
        corrupted.extend(sequence[last_end:])
        
        # Add final sentinel to target
        target.append(self.sentinel_start + sentinel_id)
        
        return np.array(corrupted), np.array(target)


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_relative_position_bias():
    """Exercise 1: Test relative position bias"""
    print("=== Exercise 1: Relative Position Bias ===")
    
    num_heads = 8
    query_len = 10
    key_len = 12
    
    rel_pos_bias = RelativePositionBias(num_heads)
    bias = rel_pos_bias.forward(query_len, key_len, bidirectional=True)
    
    print(f"Position bias shape: {bias.shape}")
    assert bias.shape == (1, num_heads, query_len, key_len)
    
    print("✓ Relative position bias working correctly")


def exercise_2_t5_attention():
    """Exercise 2: Test T5 attention with relative bias"""
    print("=== Exercise 2: T5 Attention ===")
    
    d_model = 512
    num_heads = 8
    seq_len = 16
    batch_size = 2
    
    attention = T5Attention(d_model, num_heads, has_relative_bias=True)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output, position_bias = attention.forward(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Attention output shape: {output.shape}")
    print(f"Position bias shape: {position_bias.shape if position_bias is not None else None}")
    assert output.shape == x.shape
    
    print("✓ T5 attention working correctly")


def exercise_3_complete_t5():
    """Exercise 3: Test complete T5 model"""
    print("=== Exercise 3: Complete T5 ===")
    
    vocab_size = 32128  # T5 vocab size
    seq_len = 32
    batch_size = 2
    
    model = T5Model(vocab_size, d_model=256, num_layers=4, num_heads=4)
    
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    decoder_input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len//2))
    
    output = model.forward(input_ids, decoder_input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}")
    print(f"T5 output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len//2, vocab_size)
    
    print("✓ Complete T5 working correctly")


def exercise_4_span_denoising():
    """Exercise 4: Test span denoising pre-training"""
    print("=== Exercise 4: Span Denoising ===")
    
    vocab_size = 32128
    denoiser = T5SpanDenoiser(vocab_size)
    
    batch_size = 4
    seq_len = 50
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))  # Use lower IDs for tokens
    
    corrupted_inputs, targets = denoiser.create_span_denoising_data(input_ids)
    
    print(f"Original shape: {input_ids.shape}")
    print(f"Corrupted shape: {corrupted_inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Check that sentinel tokens are used
    max_token = np.max(corrupted_inputs)
    print(f"Max token ID in corrupted input: {max_token}")
    assert max_token >= denoiser.sentinel_start
    
    print("✓ Span denoising working correctly")


def exercise_5_text_to_text_tasks():
    """Exercise 5: Test text-to-text task formulation"""
    print("=== Exercise 5: Text-to-Text Tasks ===")
    
    # Simulate different task prefixes
    tasks = {
        "translate": "translate English to French: Hello world",
        "summarize": "summarize: This is a long document that needs to be summarized into key points.",
        "question": "question: What is the capital of France? context: Paris is the capital and largest city of France.",
        "sentiment": "cola sentence: This movie is absolutely terrible and boring."
    }
    
    vocab_size = 32128
    model = T5Model(vocab_size, d_model=256, num_layers=3, num_heads=4)
    
    for task_type, task_text in tasks.items():
        # Simulate tokenized input (would normally use proper tokenizer)
        input_ids = np.random.randint(0, vocab_size, (1, 20))
        decoder_input_ids = np.random.randint(0, vocab_size, (1, 10))
        
        output = model.forward(input_ids, decoder_input_ids)
        
        print(f"Task: {task_type}")
        print(f"  Input: {task_text[:50]}...")
        print(f"  Output shape: {output.shape}")
        assert output.shape == (1, 10, vocab_size)
    
    print("✓ Text-to-text tasks working correctly")


if __name__ == "__main__":
    exercise_1_relative_position_bias()
    exercise_2_t5_attention()
    exercise_3_complete_t5()
    exercise_4_span_denoising()
    exercise_5_text_to_text_tasks()
    print("\nT5 Family implementation completed!")