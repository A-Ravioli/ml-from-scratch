"""
Vanilla Transformer Implementation Exercise

Complete implementation of the original Transformer from "Attention Is All You Need"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time
import math


class PositionalEncoding:
    """Sinusoidal positional encoding from the original Transformer paper"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to embeddings"""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]


class MultiHeadAttention:
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size, seq_len, d_model = query.shape
        
        Q = np.dot(query, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(key, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(value, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores += mask * -1e9
        
        attn_weights = self.softmax(scores)
        attn_output = np.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(attn_output, self.W_o)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class PositionwiseFeedForward:
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(np.maximum(0, np.dot(x, self.W1) + self.b1), self.W2) + self.b2


class LayerNorm:
    """Layer normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class TransformerEncoder:
    """Transformer encoder stack"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int):
        self.layers = []
        for _ in range(num_layers):
            layer = {
                'self_attn': MultiHeadAttention(d_model, num_heads),
                'feed_forward': PositionwiseFeedForward(d_model, d_ff),
                'norm1': LayerNorm(d_model),
                'norm2': LayerNorm(d_model)
            }
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        for layer in self.layers:
            # Self-attention with residual connection
            attn_out = layer['self_attn'].forward(x, x, x, mask)
            x = layer['norm1'].forward(x + attn_out)
            
            # Feed-forward with residual connection
            ff_out = layer['feed_forward'].forward(x)
            x = layer['norm2'].forward(x + ff_out)
        
        return x


class TransformerDecoder:
    """Transformer decoder stack"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int):
        self.layers = []
        for _ in range(num_layers):
            layer = {
                'self_attn': MultiHeadAttention(d_model, num_heads),
                'enc_dec_attn': MultiHeadAttention(d_model, num_heads),
                'feed_forward': PositionwiseFeedForward(d_model, d_ff),
                'norm1': LayerNorm(d_model),
                'norm2': LayerNorm(d_model),
                'norm3': LayerNorm(d_model)
            }
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray, 
                self_attn_mask: Optional[np.ndarray] = None,
                enc_dec_mask: Optional[np.ndarray] = None) -> np.ndarray:
        for layer in self.layers:
            # Masked self-attention
            self_attn_out = layer['self_attn'].forward(x, x, x, self_attn_mask)
            x = layer['norm1'].forward(x + self_attn_out)
            
            # Encoder-decoder attention
            enc_dec_attn_out = layer['enc_dec_attn'].forward(x, encoder_output, encoder_output, enc_dec_mask)
            x = layer['norm2'].forward(x + enc_dec_attn_out)
            
            # Feed-forward
            ff_out = layer['feed_forward'].forward(x)
            x = layer['norm3'].forward(x + ff_out)
        
        return x


class Transformer:
    """Complete Transformer model"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, 
                 num_layers: int = 6, max_len: int = 5000):
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = np.random.randn(src_vocab_size, d_model) * 0.1
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * 0.1
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers)
        
        # Output projection
        self.output_proj = np.random.randn(d_model, tgt_vocab_size) * 0.1
    
    def forward(self, src_ids: np.ndarray, tgt_ids: np.ndarray) -> np.ndarray:
        # Source embeddings
        src_emb = self.src_embedding[src_ids] * math.sqrt(self.d_model)
        src_emb = self.pos_encoding.forward(src_emb)
        
        # Target embeddings  
        tgt_emb = self.tgt_embedding[tgt_ids] * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding.forward(tgt_emb)
        
        # Create causal mask for decoder
        seq_len = tgt_ids.shape[1]
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        
        # Forward pass
        encoder_output = self.encoder.forward(src_emb)
        decoder_output = self.decoder.forward(tgt_emb, encoder_output, causal_mask)
        
        # Output projection
        return np.dot(decoder_output, self.output_proj)


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_transformer_components():
    """Exercise 1: Implement Transformer components"""
    print("=== Exercise 1: Transformer Components ===")
    
    d_model = 512
    seq_len = 10
    batch_size = 2
    
    # Test positional encoding
    pos_enc = PositionalEncoding(d_model)
    x = np.random.randn(batch_size, seq_len, d_model)
    x_with_pos = pos_enc.forward(x)
    
    print(f"Positional encoding shape: {x_with_pos.shape}")
    assert x_with_pos.shape == x.shape
    
    # Test multi-head attention
    mha = MultiHeadAttention(d_model, num_heads=8)
    attn_out = mha.forward(x, x, x)
    
    print(f"Multi-head attention output shape: {attn_out.shape}")
    assert attn_out.shape == x.shape
    
    print("✓ All components working correctly")


def exercise_2_complete_transformer():
    """Exercise 2: Test complete Transformer model"""
    print("=== Exercise 2: Complete Transformer ===")
    
    src_vocab_size = 1000
    tgt_vocab_size = 800
    seq_len = 20
    batch_size = 2
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=256, num_layers=3)
    
    src_ids = np.random.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt_ids = np.random.randint(0, tgt_vocab_size, (batch_size, seq_len))
    
    output = model.forward(src_ids, tgt_ids)
    
    print(f"Transformer output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, tgt_vocab_size)
    
    print("✓ Complete Transformer working correctly")


if __name__ == "__main__":
    exercise_1_transformer_components()
    exercise_2_complete_transformer()
    print("\nVanilla Transformer implementation completed!")