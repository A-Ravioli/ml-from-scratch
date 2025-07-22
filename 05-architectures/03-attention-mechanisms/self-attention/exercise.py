"""
Self-Attention Implementation Exercise

Implement self-attention mechanisms from scratch, focusing on mathematical foundations
and computational efficiency. Study attention patterns and their interpretability.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
from abc import ABC, abstractmethod
import time
import math


class SelfAttention:
    """
    Scaled dot-product self-attention mechanism
    
    Implements: Attention(Q,K,V) = softmax(QK^T / √d_k)V
    """
    
    def __init__(self, d_model: int, d_k: int = None, d_v: int = None):
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        
        # Initialize projection matrices
        self.W_q = np.random.randn(d_model, self.d_k) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, self.d_k) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, self.d_v) * np.sqrt(2.0 / d_model)
        
        # Output projection
        self.W_o = np.random.randn(self.d_v, d_model) * np.sqrt(2.0 / self.d_v)
        
        # Store intermediate values for analysis
        self.last_attention_weights = None
        self.last_queries = None
        self.last_keys = None
        self.last_values = None
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                   mask: Optional[np.ndarray] = None, 
                                   temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention
        
        Args:
            Q: Queries [batch_size, seq_len, d_k]
            K: Keys [batch_size, seq_len, d_k]
            V: Values [batch_size, seq_len, d_v]
            mask: Optional mask [batch_size, seq_len, seq_len] or broadcastable
            temperature: Temperature parameter for softmax (default: √d_k)
            
        Returns:
            output: Attended values [batch_size, seq_len, d_v]
            attention_weights: Attention matrix [batch_size, seq_len, seq_len]
        """
        # TODO: Implement scaled dot-product attention
        # 1. Compute attention scores: Q @ K^T
        # 2. Scale by √d_k (or temperature)
        # 3. Apply mask if provided
        # 4. Apply softmax to get attention weights
        # 5. Apply attention weights to values
        
        batch_size, seq_len, d_k = Q.shape
        
        # Step 1: Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1))  # [batch_size, seq_len, seq_len]
        
        # Step 2: Scale scores
        if temperature == 1.0:
            # Use standard scaling by √d_k
            scores = scores / math.sqrt(d_k)
        else:
            # Use custom temperature
            scores = scores / temperature
        
        # Step 3: Apply mask if provided
        if mask is not None:
            # Add very negative values where mask is True/1
            scores = scores + (mask * -1e9)
        
        # Step 4: Apply softmax
        # Numerically stable softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        
        # Step 5: Apply attention to values
        output = np.matmul(attention_weights, V)  # [batch_size, seq_len, d_v]
        
        return output, attention_weights
    
    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through self-attention
        
        Args:
            X: Input sequence [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Transformed sequence [batch_size, seq_len, d_model]
        """
        # TODO: Implement self-attention forward pass
        # 1. Project input to Q, K, V
        # 2. Apply scaled dot-product attention
        # 3. Project output back to d_model dimensions
        
        batch_size, seq_len, d_model = X.shape
        
        # Step 1: Project to Q, K, V
        Q = np.matmul(X, self.W_q)  # [batch_size, seq_len, d_k]
        K = np.matmul(X, self.W_k)  # [batch_size, seq_len, d_k]
        V = np.matmul(X, self.W_v)  # [batch_size, seq_len, d_v]
        
        # Store for analysis
        self.last_queries = Q
        self.last_keys = K
        self.last_values = V
        
        # Step 2: Apply attention
        attended_values, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Store attention weights for analysis
        self.last_attention_weights = attention_weights
        
        # Step 3: Output projection
        output = np.matmul(attended_values, self.W_o)  # [batch_size, seq_len, d_model]
        
        return output
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get the last computed attention weights"""
        return self.last_attention_weights
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return (self.W_q.size + self.W_k.size + self.W_v.size + self.W_o.size)


class MultiHeadSelfAttention:
    """
    Multi-head self-attention mechanism
    
    Runs multiple attention heads in parallel and concatenates outputs
    """
    
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Create attention heads
        self.heads = [SelfAttention(d_model, self.d_k, self.d_v) for _ in range(num_heads)]
        
        # Final output projection
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through multi-head attention
        
        Args:
            X: Input sequence [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Transformed sequence [batch_size, seq_len, d_model]
        """
        # TODO: Implement multi-head attention
        # 1. Apply each attention head to input
        # 2. Concatenate head outputs
        # 3. Apply final linear projection
        
        batch_size, seq_len, d_model = X.shape
        
        # Step 1: Apply attention heads
        head_outputs = []
        for head in self.heads:
            head_output = head.forward(X, mask)
            head_outputs.append(head_output)
        
        # Step 2: Concatenate head outputs
        concatenated = np.concatenate(head_outputs, axis=-1)  # [batch_size, seq_len, d_model]
        
        # Step 3: Final projection
        output = np.matmul(concatenated, self.W_o)
        
        return output
    
    def get_attention_weights(self) -> List[np.ndarray]:
        """Get attention weights from all heads"""
        return [head.get_attention_weights() for head in self.heads]
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        head_params = sum(head.count_parameters() for head in self.heads)
        return head_params + self.W_o.size


class MaskedSelfAttention(SelfAttention):
    """
    Self-attention with causal masking for autoregressive models
    """
    
    def __init__(self, d_model: int, d_k: int = None, d_v: int = None):
        super().__init__(d_model, d_k, d_v)
    
    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create causal mask to prevent attention to future positions
        
        Args:
            seq_len: Sequence length
            
        Returns:
            mask: Causal mask [seq_len, seq_len]
        """
        # TODO: Create causal mask
        # Upper triangular matrix with 1s where attention should be blocked
        
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask
    
    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass with causal masking
        
        Args:
            X: Input sequence [batch_size, seq_len, d_model]
            mask: Optional additional mask (combined with causal mask)
            
        Returns:
            output: Transformed sequence [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = X.shape
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len)
        
        # Combine with additional mask if provided
        if mask is not None:
            combined_mask = causal_mask + mask
        else:
            combined_mask = causal_mask
        
        return super().forward(X, combined_mask)


class RelativePositionalAttention(SelfAttention):
    """
    Self-attention with relative positional encodings
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 128):
        super().__init__(d_model)
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        self.relative_position_embeddings = np.random.randn(
            2 * max_relative_position + 1, self.d_k
        ) * np.sqrt(2.0 / self.d_k)
    
    def get_relative_positions(self, seq_len: int) -> np.ndarray:
        """
        Get relative position matrix
        
        Args:
            seq_len: Sequence length
            
        Returns:
            relative_positions: Matrix of relative positions [seq_len, seq_len]
        """
        # TODO: Compute relative position matrix
        positions = np.arange(seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to maximum relative position
        relative_positions = np.clip(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to positive indices
        relative_positions = relative_positions + self.max_relative_position
        
        return relative_positions
    
    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass with relative positional attention
        """
        # TODO: Implement relative positional attention
        # This is a simplified version - full implementation requires more complex modifications
        
        # For now, just use standard self-attention
        # In practice, relative positions would modify the attention computation
        return super().forward(X, mask)


def create_padding_mask(sequence_lengths: List[int], max_length: int) -> np.ndarray:
    """
    Create padding mask for variable-length sequences
    
    Args:
        sequence_lengths: List of actual sequence lengths
        max_length: Maximum sequence length (padded length)
        
    Returns:
        mask: Padding mask [batch_size, max_length, max_length]
    """
    batch_size = len(sequence_lengths)
    mask = np.zeros((batch_size, max_length, max_length))
    
    for i, length in enumerate(sequence_lengths):
        # Mask positions beyond actual sequence length
        if length < max_length:
            mask[i, :, length:] = 1  # Mask padding tokens as keys
            mask[i, length:, :] = 1  # Mask padding tokens as queries
    
    return mask


def analyze_attention_patterns(attention_weights: np.ndarray, 
                             tokens: Optional[List[str]] = None) -> Dict:
    """
    Analyze attention patterns for interpretability
    
    Args:
        attention_weights: Attention matrix [seq_len, seq_len]
        tokens: Optional token strings for visualization
        
    Returns:
        analysis: Dictionary containing attention statistics
    """
    seq_len = attention_weights.shape[0]
    
    analysis = {
        'attention_entropy': [],
        'max_attention_weight': [],
        'attention_spread': [],
        'self_attention_ratio': 0,
        'local_attention_ratio': 0
    }
    
    # TODO: Implement attention pattern analysis
    # 1. Compute attention entropy for each position
    # 2. Find maximum attention weights
    # 3. Analyze attention spread/concentration
    # 4. Compute self-attention ratio
    # 5. Analyze local vs global attention patterns
    
    for i in range(seq_len):
        weights = attention_weights[i]
        
        # Attention entropy
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        analysis['attention_entropy'].append(entropy)
        
        # Maximum attention weight
        max_weight = np.max(weights)
        analysis['max_attention_weight'].append(max_weight)
        
        # Attention spread (standard deviation)
        spread = np.std(weights)
        analysis['attention_spread'].append(spread)
    
    # Self-attention ratio (diagonal elements)
    self_attention = np.mean(np.diag(attention_weights))
    analysis['self_attention_ratio'] = self_attention
    
    # Local attention ratio (within distance 2)
    local_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]) <= 2
    local_attention = np.mean(attention_weights[local_mask])
    analysis['local_attention_ratio'] = local_attention
    
    return analysis


def visualize_attention_weights(attention_weights: np.ndarray, 
                              tokens: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
    """
    Visualize attention weight matrix
    
    Args:
        attention_weights: Attention matrix [seq_len, seq_len]
        tokens: Optional token strings for labeling
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    plt.colorbar(im, label='Attention Weight')
    
    # Add labels if tokens provided
    if tokens is not None:
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
    
    plt.xlabel('Key Position')
    plt.ylabel('Query Position') 
    plt.title('Self-Attention Weights')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_attention_heads(multi_head_weights: List[np.ndarray],
                            tokens: Optional[List[str]] = None):
    """
    Visualize attention patterns from multiple heads
    
    Args:
        multi_head_weights: List of attention matrices from different heads
        tokens: Optional token strings for labeling
    """
    num_heads = len(multi_head_weights)
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if num_heads == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, attention_weights in enumerate(multi_head_weights):
        ax = axes[i] if num_heads > 1 else axes[0]
        
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax.set_title(f'Head {i+1}')
        
        if tokens is not None and len(tokens) <= 20:  # Only show tokens for short sequences
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=8)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for i in range(num_heads, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def compute_attention_complexity(seq_lengths: List[int], d_model: int) -> Dict:
    """
    Analyze computational complexity of attention for different sequence lengths
    
    Args:
        seq_lengths: List of sequence lengths to analyze
        d_model: Model dimension
        
    Returns:
        complexity_analysis: Dictionary with complexity metrics
    """
    analysis = {
        'sequence_lengths': seq_lengths,
        'memory_complexity': [],
        'time_complexity': [],
        'attention_matrix_size': []
    }
    
    for n in seq_lengths:
        # Memory complexity: O(n² + nd)
        memory = n * n + n * d_model
        analysis['memory_complexity'].append(memory)
        
        # Time complexity: O(n²d + nd²)
        time = n * n * d_model + n * d_model * d_model
        analysis['time_complexity'].append(time)
        
        # Attention matrix size
        attn_size = n * n
        analysis['attention_matrix_size'].append(attn_size)
    
    return analysis


def compare_attention_mechanisms(sequence_length: int, d_model: int) -> Dict:
    """
    Compare different attention mechanisms
    
    Args:
        sequence_length: Length of input sequence
        d_model: Model dimension
        
    Returns:
        comparison: Dictionary with performance metrics
    """
    # Create sample input
    batch_size = 1
    X = np.random.randn(batch_size, sequence_length, d_model)
    
    mechanisms = {
        'Self-Attention': SelfAttention(d_model),
        'Multi-Head (4 heads)': MultiHeadSelfAttention(d_model, 4),
        'Multi-Head (8 heads)': MultiHeadSelfAttention(d_model, 8),
        'Masked Self-Attention': MaskedSelfAttention(d_model)
    }
    
    comparison = {}
    
    for name, mechanism in mechanisms.items():
        start_time = time.time()
        
        # Forward pass
        output = mechanism.forward(X)
        
        end_time = time.time()
        
        comparison[name] = {
            'forward_time': end_time - start_time,
            'parameters': mechanism.count_parameters(),
            'output_shape': output.shape
        }
    
    return comparison


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_basic_self_attention():
    """
    Exercise 1: Implement basic self-attention
    
    Tasks:
    1. Complete SelfAttention implementation
    2. Test on sample sequences
    3. Verify attention weight properties
    4. Analyze computational complexity
    """
    
    print("=== Exercise 1: Basic Self-Attention ===")
    
    # TODO: Test self-attention implementation
    
    # Create sample input
    batch_size, seq_len, d_model = 2, 8, 64
    X = np.random.randn(batch_size, seq_len, d_model)
    
    # Initialize self-attention
    self_attn = SelfAttention(d_model)
    
    # Forward pass
    output = self_attn.forward(X)
    attention_weights = self_attn.get_attention_weights()
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Verify attention weight properties
    print(f"Attention weights sum (should be ~1.0): {np.mean(np.sum(attention_weights, axis=-1)):.4f}")
    print(f"Attention weights min: {np.min(attention_weights):.4f}")
    print(f"Attention weights max: {np.max(attention_weights):.4f}")
    
    assert output.shape == X.shape, "Output should have same shape as input"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), "Attention weights shape incorrect"
    
    pass


def exercise_2_masked_attention():
    """
    Exercise 2: Implement masked self-attention
    
    Tasks:
    1. Complete MaskedSelfAttention implementation
    2. Test causal masking
    3. Verify no future information leakage
    4. Analyze attention patterns
    """
    
    print("=== Exercise 2: Masked Self-Attention ===")
    
    # TODO: Test masked attention functionality
    
    pass


def exercise_3_multi_head_attention():
    """
    Exercise 3: Implement multi-head attention
    
    Tasks:
    1. Complete MultiHeadSelfAttention implementation
    2. Compare single vs multi-head attention
    3. Analyze different head patterns
    4. Study head specialization
    """
    
    print("=== Exercise 3: Multi-Head Attention ===")
    
    # TODO: Test multi-head attention
    
    pass


def exercise_4_attention_analysis():
    """
    Exercise 4: Analyze attention patterns
    
    Tasks:
    1. Implement attention pattern analysis
    2. Visualize attention weights
    3. Compute attention statistics
    4. Study attention interpretability
    """
    
    print("=== Exercise 4: Attention Analysis ===")
    
    # TODO: Analyze attention patterns
    
    pass


def exercise_5_positional_attention():
    """
    Exercise 5: Implement positional attention variants
    
    Tasks:
    1. Complete RelativePositionalAttention
    2. Compare absolute vs relative positioning
    3. Study position-dependent patterns
    4. Analyze long-range dependencies
    """
    
    print("=== Exercise 5: Positional Attention ===")
    
    # TODO: Test positional attention variants
    
    pass


def exercise_6_efficiency_analysis():
    """
    Exercise 6: Analyze computational efficiency
    
    Tasks:
    1. Study scaling behavior with sequence length
    2. Compare memory usage
    3. Analyze parallelization benefits
    4. Implement optimization techniques
    """
    
    print("=== Exercise 6: Efficiency Analysis ===")
    
    # TODO: Comprehensive efficiency analysis
    
    pass


if __name__ == "__main__":
    # Run all exercises
    exercise_1_basic_self_attention()
    exercise_2_masked_attention()
    exercise_3_multi_head_attention()
    exercise_4_attention_analysis()
    exercise_5_positional_attention()
    exercise_6_efficiency_analysis()
    
    print("\nAll exercises completed!")
    print("Key insights to understand:")
    print("1. Scaled dot-product attention mathematics and implementation")
    print("2. Multi-head attention and head specialization")
    print("3. Masking strategies for different tasks")
    print("4. Computational complexity and optimization techniques")