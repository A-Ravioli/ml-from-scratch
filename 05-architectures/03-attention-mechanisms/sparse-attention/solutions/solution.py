"""
Reference solution (auto-derived from exercise.py).

Matches exercise.py public API with placeholder markers removed.
"""

"""
Sparse Attention Implementation Exercise

Implement various sparse attention patterns for efficient processing of long sequences.
Study the trade-offs between computational efficiency and approximation quality.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
from abc import ABC, abstractmethod
import time


class SparseAttentionBase(ABC):
    """Base class for sparse attention mechanisms"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Attention parameters
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # Store last computed attention for analysis
        self.last_attention_weights = None
        self.last_sparsity_pattern = None
        
    @abstractmethod
    def create_attention_mask(self, seq_len: int) -> np.ndarray:
        """Create sparsity pattern mask for attention"""
        return None
    def compute_sparse_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sparse attention using the provided mask
        
        Args:
            Q, K, V: Query, Key, Value matrices [batch_size, num_heads, seq_len, d_k]
            mask: Sparse attention mask [seq_len, seq_len] where 1=attend, 0=mask
            
        Returns:
            attention_output: [batch_size, num_heads, seq_len, d_v]
            attention_weights: [batch_size, num_heads, seq_len, seq_len] (sparse)
        """
        # Task: Implement efficient sparse attention computation
        # 1. Compute attention scores for allowed positions only
        # 2. Apply softmax with masking
        # 3. Apply attention to values
        
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # Step 1: Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Step 2: Apply sparse mask
        # Convert mask to attention mask (0 -> -inf, 1 -> 0)
        attention_mask = np.where(mask == 1, 0.0, -1e9)
        attention_mask = attention_mask[None, None, :, :]  # Broadcast for batch and heads
        
        masked_scores = scores + attention_mask
        
        # Step 3: Apply softmax
        scores_max = np.max(masked_scores, axis=-1, keepdims=True)
        exp_scores = np.exp(masked_scores - scores_max)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Zero out masked positions in attention weights (for clean visualization)
        mask_broadcast = mask[None, None, :, :].astype(bool)
        attention_weights = attention_weights * mask_broadcast
        
        # Step 4: Apply attention to values
        attention_output = np.matmul(attention_weights, V)
        
        # Store for analysis
        self.last_attention_weights = attention_weights
        self.last_sparsity_pattern = mask
        
        return attention_output, attention_weights
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through sparse attention"""
        batch_size, seq_len, d_model = X.shape
        
        # Create sparsity pattern
        attention_mask = self.create_attention_mask(seq_len)
        
        # Project to Q, K, V
        Q = np.matmul(X, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.matmul(X, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.matmul(X, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply sparse attention
        attention_output, _ = self.compute_sparse_attention(Q, K, V, attention_mask)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        output = np.matmul(attention_output, self.W_o)
        
        return output
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get last computed attention weights"""
        return self.last_attention_weights
    
    def get_sparsity_pattern(self) -> Optional[np.ndarray]:
        """Get last used sparsity pattern"""
        return self.last_sparsity_pattern
    
    def compute_sparsity_ratio(self) -> float:
        """Compute sparsity ratio (fraction of non-zero elements)"""
        if self.last_sparsity_pattern is not None:
            return np.mean(self.last_sparsity_pattern)
        return 1.0  # Full attention


class LocalWindowAttention(SparseAttentionBase):
    """
    Local sliding window attention
    
    Each position attends only to a fixed window around itself
    """
    
    def __init__(self, d_model: int, window_size: int, num_heads: int = 8):
        super().__init__(d_model, num_heads)
        self.window_size = window_size
    
    def create_attention_mask(self, seq_len: int) -> np.ndarray:
        """Create local window attention mask"""
        # Task: Create mask where position i can attend to positions [i-w, i+w]
        
        mask = np.zeros((seq_len, seq_len))
        
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        
        return mask


class StridedAttention(SparseAttentionBase):
    """
    Strided attention pattern from Sparse Transformer
    
    Combines local attention with strided global attention
    """
    
    def __init__(self, d_model: int, window_size: int, stride: int, num_heads: int = 8):
        super().__init__(d_model, num_heads)
        self.window_size = window_size
        self.stride = stride
    
    def create_attention_mask(self, seq_len: int) -> np.ndarray:
        """Create strided attention mask"""
        # Task: Combine local and strided patterns
        # For multi-head: alternate between local and strided patterns
        
        mask = np.zeros((seq_len, seq_len))
        
        # Local pattern (like sliding window)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        
        # Add strided pattern
        for i in range(seq_len):
            for j in range(0, seq_len, self.stride):
                mask[i, j] = 1
                mask[j, i] = 1  # Make symmetric for bidirectional flow
        
        return mask


class RandomSparseAttention(SparseAttentionBase):
    """
    Random sparse attention pattern
    
    Each position attends to a random subset of positions
    """
    
    def __init__(self, d_model: int, sparsity_ratio: float, num_heads: int = 8, seed: int = 42):
        super().__init__(d_model, num_heads)
        self.sparsity_ratio = sparsity_ratio
        self.seed = seed
    
    def create_attention_mask(self, seq_len: int) -> np.ndarray:
        """Create random sparse attention mask"""
        # Task: Create random sparse pattern
        
        np.random.seed(self.seed)  # For reproducibility
        
        mask = np.random.random((seq_len, seq_len)) < self.sparsity_ratio
        
        # Ensure diagonal is always attended (self-attention)
        np.fill_diagonal(mask, True)
        
        return mask.astype(float)


class BigBirdAttention(SparseAttentionBase):
    """
    BigBird attention pattern combining local, global, and random attention
    """
    
    def __init__(self, d_model: int, window_size: int = 3, num_global_tokens: int = 2,
                 num_random_tokens: int = 2, num_heads: int = 8, seed: int = 42):
        super().__init__(d_model, num_heads)
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.num_random_tokens = num_random_tokens
        self.seed = seed
    
    def create_attention_mask(self, seq_len: int) -> np.ndarray:
        """Create BigBird attention mask (local + global + random)"""
        # Task: Implement BigBird pattern
        # 1. Local sliding window
        # 2. Global tokens (first few tokens attend to all, all attend to first few)
        # 3. Random sparse connections
        
        mask = np.zeros((seq_len, seq_len))
        
        # 1. Local sliding window
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        
        # 2. Global tokens
        # First num_global_tokens are global tokens
        global_positions = list(range(min(self.num_global_tokens, seq_len)))
        
        for pos in global_positions:
            mask[pos, :] = 1  # Global tokens attend to all
            mask[:, pos] = 1  # All tokens attend to global tokens
        
        # 3. Random connections
        np.random.seed(self.seed)
        for i in range(seq_len):
            if i not in global_positions:  # Non-global tokens get random connections
                # Sample random positions to attend to
                available_positions = list(range(seq_len))
                available_positions.remove(i)  # Don't include self (already handled by local)
                
                num_random = min(self.num_random_tokens, len(available_positions))
                if num_random > 0:
                    random_positions = np.random.choice(available_positions, num_random, replace=False)
                    mask[i, random_positions] = 1
        
        return mask


class LongformerAttention(SparseAttentionBase):
    """
    Longformer attention pattern with local + global tokens
    """
    
    def __init__(self, d_model: int, window_size: int, global_token_indices: List[int],
                 num_heads: int = 8):
        super().__init__(d_model, num_heads)
        self.window_size = window_size
        self.global_token_indices = global_token_indices
    
    def create_attention_mask(self, seq_len: int) -> np.ndarray:
        """Create Longformer attention mask"""
        # Task: Implement Longformer pattern
        # 1. Local sliding window for all tokens
        # 2. Global attention for specified tokens
        
        mask = np.zeros((seq_len, seq_len))
        
        # 1. Local sliding window
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        
        # 2. Global tokens
        for global_idx in self.global_token_indices:
            if global_idx < seq_len:
                mask[global_idx, :] = 1  # Global token attends to all
                mask[:, global_idx] = 1  # All tokens attend to global token
        
        return mask


class BlockSparseAttention(SparseAttentionBase):
    """
    Block sparse attention - divide sequence into blocks and attend within blocks
    """
    
    def __init__(self, d_model: int, block_size: int, num_heads: int = 8):
        super().__init__(d_model, num_heads)
        self.block_size = block_size
    
    def create_attention_mask(self, seq_len: int) -> np.ndarray:
        """Create block sparse attention mask"""
        # Task: Create block-diagonal attention pattern
        
        mask = np.zeros((seq_len, seq_len))
        
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min((block_idx + 1) * self.block_size, seq_len)
            
            # Within-block attention
            mask[start:end, start:end] = 1
        
        return mask


def analyze_attention_sparsity(attention_weights: np.ndarray, 
                              sparsity_pattern: np.ndarray,
                              tokens: Optional[List[str]] = None) -> Dict:
    """
    Analyze sparse attention patterns and their properties
    
    Args:
        attention_weights: [batch_size, num_heads, seq_len, seq_len]
        sparsity_pattern: [seq_len, seq_len] binary mask
        tokens: Optional token strings
        
    Returns:
        analysis: Dictionary with sparsity analysis
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    analysis = {
        'sparsity_ratio': np.mean(sparsity_pattern),
        'effective_connections': np.sum(sparsity_pattern),
        'max_possible_connections': seq_len * seq_len,
        'compression_ratio': 1.0 / np.mean(sparsity_pattern) if np.mean(sparsity_pattern) > 0 else float('inf'),
        'attention_distribution': {},
        'information_flow': {}
    }
    
    # Task: Implement detailed sparsity analysis
    # 1. Attention distribution within allowed positions
    # 2. Information flow analysis
    # 3. Pattern regularity metrics
    
    # Average attention across batch and heads for analysis
    avg_attention = np.mean(attention_weights, axis=(0, 1))  # [seq_len, seq_len]
    
    # Attention distribution within sparse pattern
    allowed_positions = sparsity_pattern == 1
    if np.any(allowed_positions):
        attention_values = avg_attention[allowed_positions]
        analysis['attention_distribution'] = {
            'mean': np.mean(attention_values),
            'std': np.std(attention_values),
            'min': np.min(attention_values),
            'max': np.max(attention_values),
            'entropy': -np.sum(attention_values * np.log(attention_values + 1e-8))
        }
    
    # Information flow analysis
    # Compute how information can flow through the sparse pattern
    connectivity_matrix = sparsity_pattern
    
    # Count path lengths (simplified)
    direct_connections = np.sum(connectivity_matrix, axis=1)
    analysis['information_flow'] = {
        'avg_direct_connections': np.mean(direct_connections),
        'min_connections': np.min(direct_connections),
        'max_connections': np.max(direct_connections),
        'connection_variance': np.var(direct_connections)
    }
    
    return analysis


def visualize_sparse_attention_pattern(sparsity_pattern: np.ndarray,
                                     attention_weights: Optional[np.ndarray] = None,
                                     title: str = "Sparse Attention Pattern"):
    """
    Visualize sparse attention pattern and weights
    
    Args:
        sparsity_pattern: [seq_len, seq_len] binary mask
        attention_weights: Optional [seq_len, seq_len] attention weights
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2 if attention_weights is not None else 1, 
                            figsize=(12 if attention_weights is not None else 6, 5))
    
    if attention_weights is not None:
        axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
    else:
        axes = [axes]
    
    # Plot sparsity pattern
    im1 = axes[0].imshow(sparsity_pattern, cmap='Blues', aspect='auto')
    axes[0].set_title(f'{title} - Pattern')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot actual attention weights if provided
    if attention_weights is not None:
        # Mask attention weights to show only allowed positions
        masked_attention = attention_weights * sparsity_pattern
        
        im2 = axes[1].imshow(masked_attention, cmap='Reds', aspect='auto')
        axes[1].set_title(f'{title} - Weights')
        axes[1].set_xlabel('Key Position')
        axes[1].set_ylabel('Query Position')
        plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


def compare_sparse_attention_efficiency(seq_lengths: List[int], 
                                      attention_patterns: Dict[str, SparseAttentionBase]) -> Dict:
    """
    Compare computational efficiency of different sparse attention patterns
    
    Args:
        seq_lengths: List of sequence lengths to test
        attention_patterns: Dictionary of pattern name -> attention object
        
    Returns:
        efficiency_comparison: Performance metrics for each pattern
    """
    d_model = 64
    batch_size = 2
    
    results = {}
    
    for pattern_name, attention_model in attention_patterns.items():
        results[pattern_name] = {
            'seq_lengths': seq_lengths,
            'forward_times': [],
            'sparsity_ratios': [],
            'memory_usage': []  # Simplified - actual memory would need profiling
        }
        
        for seq_len in seq_lengths:
            # Create test input
            test_input = np.random.randn(batch_size, seq_len, d_model)
            
            # Time forward pass
            start_time = time.time()
            output = attention_model.forward(test_input)
            end_time = time.time()
            
            forward_time = end_time - start_time
            sparsity_ratio = attention_model.compute_sparsity_ratio()
            
            # Estimate memory usage (theoretical)
            memory_estimate = seq_len * seq_len * sparsity_ratio
            
            results[pattern_name]['forward_times'].append(forward_time)
            results[pattern_name]['sparsity_ratios'].append(sparsity_ratio)
            results[pattern_name]['memory_usage'].append(memory_estimate)
    
    return results


def create_attention_pattern_gallery(seq_len: int = 64) -> Dict[str, np.ndarray]:
    """
    Create a gallery of different sparse attention patterns for visualization
    
    Args:
        seq_len: Sequence length for pattern creation
        
    Returns:
        patterns: Dictionary of pattern name -> attention mask
    """
    patterns = {}
    
    # Task: Create various attention patterns for comparison
    
    # Local window
    local_attention = LocalWindowAttention(64, window_size=5)
    patterns['Local Window'] = local_attention.create_attention_mask(seq_len)
    
    # Strided
    strided_attention = StridedAttention(64, window_size=3, stride=8)
    patterns['Strided'] = strided_attention.create_attention_mask(seq_len)
    
    # Random sparse
    random_attention = RandomSparseAttention(64, sparsity_ratio=0.1)
    patterns['Random Sparse'] = random_attention.create_attention_mask(seq_len)
    
    # BigBird
    bigbird_attention = BigBirdAttention(64, window_size=3, num_global_tokens=4, num_random_tokens=2)
    patterns['BigBird'] = bigbird_attention.create_attention_mask(seq_len)
    
    # Block sparse
    block_attention = BlockSparseAttention(64, block_size=8)
    patterns['Block Sparse'] = block_attention.create_attention_mask(seq_len)
    
    # Full attention (for comparison)
    patterns['Full Attention'] = np.ones((seq_len, seq_len))
    
    return patterns


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_local_attention():
    """
    Exercise 1: Implement local window attention
    
    Tasks:
    1. Complete LocalWindowAttention implementation
    2. Test with different window sizes
    3. Analyze complexity reduction
    4. Study information flow limitations
    """
    
    print("=== Exercise 1: Local Window Attention ===")
    
    # Task: Test local attention implementation
    
    d_model = 64
    seq_len = 32
    batch_size = 2
    
    # Test different window sizes
    window_sizes = [2, 4, 8]
    
    for window_size in window_sizes:
        local_attn = LocalWindowAttention(d_model, window_size)
        
        test_input = np.random.randn(batch_size, seq_len, d_model)
        output = local_attn.forward(test_input)
        
        sparsity_ratio = local_attn.compute_sparsity_ratio()
        
        print(f"Window size {window_size}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Sparsity ratio: {sparsity_ratio:.3f}")
        print(f"  Compression: {1/sparsity_ratio:.1f}x")
    
    assert output.shape == test_input.shape, "Output shape should match input"
    
    return None
def exercise_2_sparse_patterns():
    """
    Exercise 2: Implement various sparse attention patterns
    
    Tasks:
    1. Complete StridedAttention, RandomSparseAttention implementations
    2. Compare pattern properties
    3. Analyze sparsity vs connectivity trade-offs
    4. Visualize different patterns
    """
    
    print("=== Exercise 2: Sparse Attention Patterns ===")
    
    # Task: Test different sparse attention patterns
    
    return None
def exercise_3_bigbird_longformer():
    """
    Exercise 3: Implement BigBird and Longformer attention
    
    Tasks:
    1. Complete BigBirdAttention and LongformerAttention
    2. Study combined pattern effects
    3. Analyze global token importance
    4. Compare with simpler patterns
    """
    
    print("=== Exercise 3: BigBird and Longformer Attention ===")
    
    # Task: Test advanced sparse attention patterns
    
    return None
def exercise_4_efficiency_analysis():
    """
    Exercise 4: Analyze computational efficiency
    
    Tasks:
    1. Benchmark different sparse patterns
    2. Measure actual vs theoretical speedups
    3. Study memory usage patterns
    4. Analyze scalability with sequence length
    """
    
    print("=== Exercise 4: Efficiency Analysis ===")
    
    # Task: Comprehensive efficiency analysis
    
    return None
def exercise_5_pattern_visualization():
    """
    Exercise 5: Visualize and analyze attention patterns
    
    Tasks:
    1. Create attention pattern visualization tools
    2. Analyze information flow properties
    3. Study attention distribution within patterns
    4. Compare pattern effectiveness
    """
    
    print("=== Exercise 5: Pattern Visualization ===")
    
    # Task: Implement pattern visualization and analysis
    
    return None
def exercise_6_adaptive_sparsity():
    """
    Exercise 6: Implement adaptive sparse attention
    
    Tasks:
    1. Design learnable sparsity patterns
    2. Implement top-k attention
    3. Study content-based attention routing
    4. Analyze adaptation during training
    """
    
    print("=== Exercise 6: Adaptive Sparsity ===")
    
    # Task: Implement adaptive sparse attention mechanisms
    
    return None
if __name__ == "__main__":
    # Run all exercises
    exercise_1_local_attention()
    exercise_2_sparse_patterns()
    exercise_3_bigbird_longformer()
    exercise_4_efficiency_analysis()
    exercise_5_pattern_visualization()
    exercise_6_adaptive_sparsity()
    
    print("\nAll exercises completed!")
    print("Key insights to understand:")
    print("1. Sparse attention patterns and their trade-offs")
    print("2. Computational efficiency vs approximation quality")
    print("3. Information flow in sparse attention networks")
    print("4. Pattern design for different applications")