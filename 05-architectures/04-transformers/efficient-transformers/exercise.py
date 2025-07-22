"""
Efficient Transformers Implementation Exercise

Implementation of various efficiency approaches for scaling Transformers beyond quadratic complexity
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time
import math


class LinearAttention:
    """Linear attention mechanism without softmax"""
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Linear attention: Q(K^T V) / Q(K^T 1)
        # Compute K^T V efficiently
        KV = np.matmul(K.transpose(0, 1, 3, 2), V)  # [batch, heads, d_k, d_k]
        
        # Compute K^T 1 (sum over sequence dimension)
        K_sum = np.sum(K, axis=2, keepdims=True)  # [batch, heads, 1, d_k]
        
        # Apply to queries
        numerator = np.matmul(Q, KV)  # [batch, heads, seq_len, d_k]
        denominator = np.matmul(Q, K_sum.transpose(0, 1, 3, 2))  # [batch, heads, seq_len, 1]
        
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-6)
        
        attn_output = numerator / denominator
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(attn_output, self.W_o)


class PerformerAttention:
    """FAVOR+ attention mechanism from Performer"""
    
    def __init__(self, d_model: int, num_heads: int, num_features: int = 256):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_features = num_features
        
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # Random features for kernel approximation
        self.random_features = np.random.randn(num_features, self.d_k) / np.sqrt(self.d_k)
    
    def feature_map(self, x: np.ndarray) -> np.ndarray:
        """Apply random feature map φ(x) = exp(ω^T x) / sqrt(m)"""
        projection = np.dot(x, self.random_features.T)
        return np.exp(projection - np.max(projection, axis=-1, keepdims=True)) / np.sqrt(self.num_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply feature maps
        Q_prime = np.stack([self.feature_map(Q[:, h]) for h in range(self.num_heads)], axis=1)
        K_prime = np.stack([self.feature_map(K[:, h]) for h in range(self.num_heads)], axis=1)
        
        # Efficient attention computation: (Q'(K')^T)V
        KV = np.matmul(K_prime.transpose(0, 1, 3, 2), V)  # [batch, heads, features, d_k]
        attn_output = np.matmul(Q_prime, KV)  # [batch, heads, seq_len, d_k]
        
        # Normalization
        K_sum = np.sum(K_prime, axis=2, keepdims=True)  # [batch, heads, 1, features]
        normalizer = np.matmul(Q_prime, K_sum.transpose(0, 1, 3, 2))  # [batch, heads, seq_len, 1]
        normalizer = np.maximum(normalizer, 1e-6)
        
        attn_output = attn_output / normalizer
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(attn_output, self.W_o)


class LinformerAttention:
    """Linformer: low-rank approximation of attention"""
    
    def __init__(self, d_model: int, num_heads: int, seq_len: int, k: int = 256):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_len = seq_len
        self.k = k  # Projected dimension
        
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # Low-rank projection matrices
        self.E = np.random.randn(seq_len, k) * 0.1
        self.F = np.random.randn(seq_len, k) * 0.1
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Project K and V to lower dimension
        # K_proj = K @ E^T, V_proj = V @ F^T
        K_proj = np.matmul(K.transpose(0, 1, 3, 2), self.E[:seq_len, :self.k]).transpose(0, 1, 3, 2)
        V_proj = np.matmul(V.transpose(0, 1, 3, 2), self.F[:seq_len, :self.k]).transpose(0, 1, 3, 2)
        
        # Attention with projected keys and values
        scores = np.matmul(Q, K_proj.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        attn_weights = self.softmax(scores)
        attn_output = np.matmul(attn_weights, V_proj)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(attn_output, self.W_o)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SparseAttention:
    """Sparse attention with configurable patterns"""
    
    def __init__(self, d_model: int, num_heads: int, pattern: str = "local"):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.pattern = pattern
        
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def create_sparse_mask(self, seq_len: int) -> np.ndarray:
        """Create sparse attention mask based on pattern"""
        mask = np.full((seq_len, seq_len), -np.inf)
        
        if self.pattern == "local":
            # Local attention (window size = 64)
            window = min(64, seq_len)
            for i in range(seq_len):
                start = max(0, i - window // 2)
                end = min(seq_len, i + window // 2 + 1)
                mask[i, start:end] = 0.0
        
        elif self.pattern == "strided":
            # Strided attention (stride = 64)
            stride = 64
            mask[np.arange(seq_len)[:, None], np.arange(seq_len)[None, :] % stride == 0] = 0.0
            # Also attend to local context
            for i in range(seq_len):
                local_start = max(0, i - 32)
                local_end = min(seq_len, i + 33)
                mask[i, local_start:local_end] = 0.0
        
        elif self.pattern == "bigbird":
            # BigBird pattern: global + random + local
            # Global tokens (first few)
            global_size = min(64, seq_len)
            mask[:, :global_size] = 0.0
            mask[:global_size, :] = 0.0
            
            # Local attention
            for i in range(seq_len):
                local_start = max(0, i - 32)
                local_end = min(seq_len, i + 33)
                mask[i, local_start:local_end] = 0.0
            
            # Random connections
            for i in range(seq_len):
                random_indices = np.random.choice(seq_len, size=min(32, seq_len), replace=False)
                mask[i, random_indices] = 0.0
        
        return mask
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Create sparse mask
        sparse_mask = self.create_sparse_mask(seq_len)
        
        # Project to Q, K, V
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Attention with sparse mask
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        scores = scores + sparse_mask[None, None, :, :]
        
        attn_weights = self.softmax(scores)
        attn_output = np.matmul(attn_weights, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(attn_output, self.W_o)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class FlashAttention:
    """Simplified FlashAttention implementation (memory-efficient)"""
    
    def __init__(self, d_model: int, num_heads: int, block_size: int = 64):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size
        
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Tiled attention computation for memory efficiency"""
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Tiled computation
        output = np.zeros_like(Q)
        
        for i in range(0, seq_len, self.block_size):
            i_end = min(i + self.block_size, seq_len)
            Q_block = Q[:, :, i:i_end, :]
            
            # Initialize block outputs
            O_block = np.zeros_like(Q_block)
            l_block = np.zeros((batch_size, self.num_heads, i_end - i, 1))
            m_block = np.full((batch_size, self.num_heads, i_end - i, 1), -np.inf)
            
            for j in range(0, seq_len, self.block_size):
                j_end = min(j + self.block_size, seq_len)
                K_block = K[:, :, j:j_end, :]
                V_block = V[:, :, j:j_end, :]
                
                # Compute attention for this tile
                S_block = np.matmul(Q_block, K_block.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
                
                # Apply causal mask if needed
                if i >= j:  # Only attend to past
                    causal_mask = np.triu(np.ones((i_end - i, j_end - j)), 
                                        k=j_end - i if j_end > i else 0)
                    S_block = np.where(causal_mask[None, None, :, :] == 0, -np.inf, S_block)
                
                # Online softmax computation
                m_new = np.maximum(m_block, np.max(S_block, axis=-1, keepdims=True))
                P_block = np.exp(S_block - m_new)
                l_new = np.exp(m_block - m_new) * l_block + np.sum(P_block, axis=-1, keepdims=True)
                
                # Update output
                O_block = (np.exp(m_block - m_new) * l_block * O_block + 
                          np.matmul(P_block, V_block)) / l_new
                
                m_block = m_new
                l_block = l_new
            
            output[:, :, i:i_end, :] = O_block
        
        # Concatenate heads and project
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(output, self.W_o)


class EfficientTransformerBlock:
    """Transformer block with choice of efficient attention"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 attention_type: str = "linear", **attention_kwargs):
        self.attention_type = attention_type
        
        # Choose attention mechanism
        if attention_type == "linear":
            self.attention = LinearAttention(d_model, num_heads)
        elif attention_type == "performer":
            self.attention = PerformerAttention(d_model, num_heads, **attention_kwargs)
        elif attention_type == "linformer":
            self.attention = LinformerAttention(d_model, num_heads, **attention_kwargs)
        elif attention_type == "sparse":
            self.attention = SparseAttention(d_model, num_heads, **attention_kwargs)
        elif attention_type == "flash":
            self.attention = FlashAttention(d_model, num_heads, **attention_kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # MLP and normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Pre-norm architecture
        attn_out = self.attention.forward(self.norm1.forward(x))
        x = x + attn_out
        
        mlp_out = self.mlp.forward(self.norm2.forward(x))
        x = x + mlp_out
        
        return x


class LayerNorm:
    """Layer normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class MLP:
    """Simple MLP with GELU activation"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.dot(x, self.W1) + self.b1
        h = self.gelu(h)
        return np.dot(h, self.W2) + self.b2
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_linear_attention():
    """Exercise 1: Test linear attention"""
    print("=== Exercise 1: Linear Attention ===")
    
    d_model = 512
    num_heads = 8
    seq_len = 1024
    batch_size = 2
    
    linear_attn = LinearAttention(d_model, num_heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    start_time = time.time()
    output = linear_attn.forward(x)
    linear_time = time.time() - start_time
    
    print(f"Input shape: {x.shape}")
    print(f"Linear attention output shape: {output.shape}")
    print(f"Linear attention time: {linear_time:.4f}s")
    assert output.shape == x.shape
    
    print("✓ Linear attention working correctly")


def exercise_2_performer_attention():
    """Exercise 2: Test Performer attention"""
    print("=== Exercise 2: Performer Attention ===")
    
    d_model = 512
    num_heads = 8
    seq_len = 1024
    batch_size = 2
    
    performer_attn = PerformerAttention(d_model, num_heads, num_features=256)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    start_time = time.time()
    output = performer_attn.forward(x)
    performer_time = time.time() - start_time
    
    print(f"Input shape: {x.shape}")
    print(f"Performer attention output shape: {output.shape}")
    print(f"Performer attention time: {performer_time:.4f}s")
    assert output.shape == x.shape
    
    print("✓ Performer attention working correctly")


def exercise_3_sparse_attention():
    """Exercise 3: Test sparse attention patterns"""
    print("=== Exercise 3: Sparse Attention ===")
    
    d_model = 256
    num_heads = 4
    seq_len = 512
    batch_size = 1
    
    patterns = ["local", "strided", "bigbird"]
    
    for pattern in patterns:
        sparse_attn = SparseAttention(d_model, num_heads, pattern=pattern)
        x = np.random.randn(batch_size, seq_len, d_model)
        
        output = sparse_attn.forward(x)
        
        print(f"Pattern: {pattern}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == x.shape
    
    print("✓ Sparse attention patterns working correctly")


def exercise_4_efficiency_comparison():
    """Exercise 4: Compare efficiency of different attention mechanisms"""
    print("=== Exercise 4: Efficiency Comparison ===")
    
    d_model = 512
    num_heads = 8
    seq_len = 512
    batch_size = 2
    
    attention_types = [
        ("linear", {}),
        ("performer", {"num_features": 128}),
        ("linformer", {"seq_len": seq_len, "k": 128}),
        ("sparse", {"pattern": "local"}),
        ("flash", {"block_size": 64})
    ]
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    results = {}
    
    for attn_type, kwargs in attention_types:
        try:
            block = EfficientTransformerBlock(d_model, num_heads, d_model * 4, 
                                            attention_type=attn_type, **kwargs)
            
            start_time = time.time()
            output = block.forward(x)
            elapsed_time = time.time() - start_time
            
            results[attn_type] = {
                "time": elapsed_time,
                "shape": output.shape
            }
            
            print(f"{attn_type}: {elapsed_time:.4f}s, shape: {output.shape}")
            assert output.shape == x.shape
            
        except Exception as e:
            print(f"{attn_type}: Failed - {str(e)}")
    
    # Find fastest
    if results:
        fastest = min(results.keys(), key=lambda k: results[k]["time"])
        print(f"\nFastest: {fastest} ({results[fastest]['time']:.4f}s)")
    
    print("✓ Efficiency comparison completed")


def exercise_5_scaling_analysis():
    """Exercise 5: Analyze scaling properties"""
    print("=== Exercise 5: Scaling Analysis ===")
    
    d_model = 256
    num_heads = 4
    batch_size = 1
    
    seq_lengths = [128, 256, 512, 1024]
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # Test linear vs quadratic scaling
        linear_attn = LinearAttention(d_model, num_heads)
        performer_attn = PerformerAttention(d_model, num_heads, num_features=64)
        
        x = np.random.randn(batch_size, seq_len, d_model)
        
        # Linear attention (O(n))
        start_time = time.time()
        linear_out = linear_attn.forward(x)
        linear_time = time.time() - start_time
        
        # Performer attention (O(n))
        start_time = time.time()
        performer_out = performer_attn.forward(x)
        performer_time = time.time() - start_time
        
        print(f"  Linear: {linear_time:.4f}s")
        print(f"  Performer: {performer_time:.4f}s")
        
        # Theoretical complexity analysis
        linear_ops = seq_len * d_model  # O(n)
        quadratic_ops = seq_len * seq_len * d_model  # O(n²)
        
        print(f"  Linear ops (theoretical): {linear_ops}")
        print(f"  Quadratic ops (theoretical): {quadratic_ops}")
        print(f"  Efficiency gain: {quadratic_ops / linear_ops:.1f}x")
    
    print("\n✓ Scaling analysis completed")


if __name__ == "__main__":
    exercise_1_linear_attention()
    exercise_2_performer_attention()
    exercise_3_sparse_attention()
    exercise_4_efficiency_comparison()
    exercise_5_scaling_analysis()
    print("\nEfficient Transformers implementation completed!")