# Sparse Attention: Efficient Attention for Long Sequences

## Prerequisites
- Multi-head attention and scaled dot-product attention
- Computational complexity analysis (big-O notation)
- Sparse matrix operations and graph theory basics
- Information theory (entropy, compression)

## Learning Objectives
- Understand the computational bottleneck of standard attention mechanisms
- Master various sparse attention patterns and their trade-offs
- Implement efficient sparse attention algorithms from scratch
- Analyze the relationship between sparsity patterns and model performance
- Connect sparse attention to graph neural networks and structured attention

## Mathematical Foundations

### 1. The Attention Complexity Problem

#### Standard Attention Complexity
For sequence length n and model dimension d:
- **Time complexity**: O(n²d + nd²)
- **Space complexity**: O(n²) for attention matrix
- **Memory scaling**: Quadratic in sequence length

#### The Long Sequence Challenge
For long sequences (n > 1000):
- **16K tokens**: 256M attention weights per head
- **64K tokens**: 4B attention weights per head  
- **Memory requirements**: Often exceed GPU memory limits
- **Computational cost**: Prohibitive for training and inference

#### Motivation for Sparsity
Most attention weights are small and contribute little to the final output:
- **Attention entropy**: High entropy suggests uniform attention (less informative)
- **Effective attention**: Often only 10-20% of attention weights are "important"
- **Redundancy**: Many attention patterns are redundant across heads

### 2. Sparse Attention Framework

#### Definition 2.1 (Sparse Attention)
A sparse attention mechanism computes attention only for a subset S ⊂ {1,...,n} × {1,...,n} of position pairs:
```
A_sparse[i,j] = {
  softmax(q_i^T k_j / √d_k)  if (i,j) ∈ S
  0                          otherwise
}
```

#### Sparsity Pattern Design Principles

**Locality Preservation**: Maintain local dependencies
```
S_local = {(i,j) : |i-j| ≤ w} for window size w
```

**Long-range Connectivity**: Enable global information flow
```
S_global = {(i,j) : i ≡ 0 (mod s) or j ≡ 0 (mod s)} for stride s
```

**Computational Efficiency**: Reduce complexity from O(n²) to O(n·k) where k << n

### 3. Local Attention Patterns

#### Sliding Window Attention

**Definition 3.1 (Sliding Window Attention)**
Each position attends only to positions within a fixed window:
```
S_window = {(i,j) : max(0, i-w) ≤ j ≤ min(n-1, i+w)}
```

**Properties**:
- **Time complexity**: O(nwd) instead of O(n²d)
- **Space complexity**: O(nw) instead of O(n²)
- **Receptive field growth**: Linear with depth

#### Dilated Attention

**Definition 3.2 (Dilated Attention)**
Attention with gaps, inspired by dilated convolutions:
```
S_dilated = {(i,j) : j = i + k·d for k ∈ {-w,...,w}, d ∈ {1,2,4,...}}
```

**Benefits**:
- **Exponential receptive field**: Covers O(2^L) positions at layer L
- **Hierarchical patterns**: Different dilation rates capture different scales
- **Parameter efficiency**: Maintains O(nw) complexity

#### Local + Random Access

**Longformer Pattern**:
Combines local attention with random global connections:
```
S_longformer = S_window ∪ S_random ∪ S_global
```
where S_random are randomly sampled position pairs.

### 4. Structured Sparse Patterns

#### Strided Attention (Sparse Transformer)

**Definition 4.1 (Strided Attention)**
Two complementary attention patterns:
```
A_local[i,j] = attend if |i-j| ≤ w
A_strided[i,j] = attend if (i-j) ≡ 0 (mod s)
```

**Two-head configuration**:
- **Local head**: Captures local dependencies
- **Strided head**: Enables long-range information transfer

#### Axial Attention

For 2D data (images) with shape H×W, apply attention along axes:
```
A_row[i,j] = attend if row(i) = row(j)
A_col[i,j] = attend if col(i) = col(j)
```

**Complexity reduction**: O(n²) → O(n√n) for 2D data

#### Block-Local Attention

**Definition 4.2 (Block-Local Attention)**
Divide sequence into blocks, apply attention within blocks:
```
S_block = {(i,j) : ⌊i/B⌋ = ⌊j/B⌋} for block size B
```

**Properties**:
- **Complexity**: O(nB) instead of O(n²)
- **Parallelization**: Blocks can be processed independently
- **Information bottleneck**: Limited inter-block communication

### 5. Learnable Sparse Patterns

#### Routing-Based Attention

**Idea**: Learn which tokens should attend to each other
```
Route(q_i, k_j) = MLP([q_i; k_j; pos_i; pos_j])
Attend if Route(q_i, k_j) > threshold
```

#### Adaptive Sparse Attention

**Dynamic sparsity**: Sparsity pattern changes based on input
```
Mask[i,j] = σ(W_mask · [q_i; k_j]) > τ
```

**Benefits**: Task-specific attention patterns
**Challenges**: Irregular computation, hard to optimize

#### Top-K Attention

**Definition 5.1 (Top-K Attention)**
For each query, attend only to top-k keys by attention score:
```
S_topk = {(i,j) : j ∈ topk(q_i^T K, k)}
```

**Properties**:
- **Guaranteed sparsity**: Exactly k attention weights per position
- **Content-based**: Adapts to input content
- **Implementation complexity**: Requires efficient top-k selection

### 6. BigBird: Universal Sparse Pattern

#### BigBird Attention Pattern
Combines multiple sparsity types:
```
S_BigBird = S_local ∪ S_global ∪ S_random
```

**Components**:
- **Local**: Sliding window (w=3 typically)
- **Global**: Few designated global tokens
- **Random**: Sparse random connections

#### Theoretical Properties

**Theorem 6.1 (BigBird Expressivity)**
BigBird attention can approximate any sequence-to-sequence function that full attention can compute, given sufficient random connections.

**Proof Sketch**: Random connections provide "highways" for information transfer, while local connections maintain gradient flow.

### 7. Implementation Considerations

#### Sparse Matrix Operations

**Storage formats**:
- **Coordinate format (COO)**: Store (row, col, value) triplets
- **Compressed Sparse Row (CSR)**: Efficient row-wise operations
- **Block sparse**: Regular sparsity patterns

#### Memory Layout Optimization

**Attention mask precomputation**:
```python
# Precompute attention masks
local_mask = create_local_mask(seq_len, window_size)
global_mask = create_global_mask(seq_len, global_positions)
combined_mask = local_mask | global_mask
```

**Sparse matrix multiplication**:
```python
# Only compute attention for allowed positions
scores = sparse_matmul(Q, K.T, mask=combined_mask)
```

#### GPU Optimization

**Challenges**:
- **Irregular memory access**: Sparse patterns break coalescing
- **Load balancing**: Variable work per thread
- **Kernel fusion**: Difficult to fuse sparse operations

**Solutions**:
- **Block-structured sparsity**: Regular patterns enable efficient kernels
- **Reordering**: Sort by attention pattern for better coalescing
- **Custom CUDA kernels**: Specialized implementations for specific patterns

### 8. Analysis and Performance Trade-offs

#### Approximation Quality

**Information bottleneck analysis**:
```
I_full = I(Output; Input | Full_Attention)
I_sparse = I(Output; Input | Sparse_Attention)
Approximation_gap = I_full - I_sparse
```

#### Complexity Analysis Comparison

| Pattern | Time | Space | Receptive Field Growth |
|---------|------|-------|----------------------|
| Full | O(n²d) | O(n²) | Immediate global |
| Local | O(nwd) | O(nw) | Linear with depth |
| Strided | O(nsd) | O(ns) | Logarithmic with depth |
| Random | O(nrd) | O(nr) | Probabilistic global |
| BigBird | O(n(w+g+r)d) | O(n(w+g+r)) | Global with high prob |

#### Performance Evaluation

**Metrics for sparse attention evaluation**:
1. **Computational efficiency**: Actual speedup vs theoretical
2. **Memory usage**: Peak memory during forward/backward pass
3. **Approximation quality**: Task performance vs full attention
4. **Scalability**: Performance scaling with sequence length

### 9. Advanced Sparse Attention Variants

#### Linformer

**Key insight**: Attention matrix is low-rank
```
K' = K·E,  V' = V·F
Attention(Q,K',V') where K',V' ∈ ℝ^{n×k}, k << n
```

**Linear complexity**: O(nkd) instead of O(n²d)

#### Performer (FAVOR+)

**Random feature approximation**:
```
φ(x) = exp(x)  # Approximate with random features
K(x,y) = φ(x)^T φ(y) ≈ (1/m)Σ_i φ_i(x)φ_i(y)
```

**Linear attention**: O(nd) complexity through kernel approximation

#### Synthesizer

**Replace attention with learned weights**:
```
# Dense synthesizer
A = MLP(X)  # Learn attention directly

# Random synthesizer  
A = Random_matrix  # Fixed random attention
```

### 10. Gradient Flow and Training Dynamics

#### Sparse Gradient Propagation

**Gradient flow through sparse attention**:
```
∂L/∂q_i = Σ_{j:(i,j)∈S} (∂L/∂a_{ij}) · (∂a_{ij}/∂q_i)
```

Only positions in sparsity set S contribute to gradients.

#### Training Stability

**Challenges**:
- **Gradient variance**: Sparse patterns increase gradient noise
- **Optimization landscape**: Different local minima than full attention
- **Learning rate sensitivity**: May require different learning rates

**Solutions**:
- **Gradient accumulation**: Reduce variance through larger batches
- **Curriculum learning**: Start dense, gradually increase sparsity
- **Regularization**: L2/dropout to prevent overfitting to sparse patterns

### 11. Applications and Use Cases

#### Long Document Processing

**Document summarization**:
- Local attention for sentence-level coherence
- Global attention for document-level structure
- Hierarchical patterns for multi-scale processing

#### Code Understanding

**Programming language modeling**:
- Local attention for syntactic dependencies
- Strided attention for scope-based relationships
- Global attention for imports/declarations

#### Scientific Text Processing

**Long research papers**:
- Section-level attention blocks
- Citation-based global connections
- Figure/table reference patterns

### 12. Evaluation and Benchmarking

#### Efficiency Benchmarks

**Metrics**:
- **Throughput**: Tokens processed per second
- **Memory efficiency**: Peak memory usage
- **Energy consumption**: FLOPs and actual energy usage

#### Quality Assessment

**Downstream task evaluation**:
- **Long document QA**: Requires long-range reasoning
- **Document classification**: Tests global understanding
- **Code completion**: Evaluates structured dependencies

#### Ablation Studies

**Component analysis**:
- Effect of different sparsity ratios
- Importance of different pattern components
- Sensitivity to hyperparameters

## Implementation Details

See `exercise.py` for implementations of:
1. Various sparse attention patterns (local, strided, random)
2. Efficient sparse matrix operations
3. BigBird and Longformer attention mechanisms
4. Performance benchmarking tools
5. Attention pattern visualization
6. Gradient analysis for sparse patterns

## Experiments

1. **Complexity Analysis**: Measure actual vs theoretical speedups
2. **Pattern Effectiveness**: Compare different sparse patterns on tasks
3. **Approximation Quality**: Study relationship between sparsity and performance
4. **Scalability Testing**: Performance on sequences of varying length
5. **Memory Profiling**: Detailed memory usage analysis
6. **Hardware Efficiency**: GPU utilization and optimization

## Research Connections

### Foundational Papers
1. **Child et al. (2019)** - "Generating Long Sequences with Sparse Transformers"
   - Introduced strided attention patterns

2. **Beltagy et al. (2020)** - "Longformer: The Long-Document Transformer"
   - Local + global + random attention pattern

3. **Zaheer et al. (2020)** - "Big Bird: Transformers for Longer Sequences"
   - Theoretical analysis of sparse attention universality

### Efficiency-Focused Works
4. **Wang et al. (2020)** - "Linformer: Self-Attention with Linear Complexity"
   - Low-rank approximation of attention

5. **Choromanski et al. (2020)** - "Rethinking Attention with Performers"
   - Kernel approximation for linear attention

6. **Tay et al. (2020)** - "Synthesizer: Rethinking Self-Attention in Transformer Models"
   - Learned and random attention alternatives

### Analysis and Theory
7. **Correia et al. (2019)** - "Adaptively Sparse Transformers"
   - Learnable sparsity patterns

8. **Roy et al. (2021)** - "Efficient Content-Based Sparse Attention with Routing Transformers"
   - Content-based routing for attention

## Resources

### Primary Sources
1. **Child et al. (2019)** - Sparse Transformer paper
2. **Beltagy et al. (2020)** - Longformer: comprehensive long-sequence model
3. **Zaheer et al. (2020)** - BigBird: theoretical foundations

### Implementation Guides
1. **Longformer GitHub** - Official implementation with efficient CUDA kernels
2. **BigBird Implementation** - Google Research implementation
3. **Sparse Attention Patterns** - Various pattern implementations

### Surveys and Analysis
1. **Tay et al. (2020)** - "Efficient Transformers: A Survey"
2. **Qiu et al. (2020)** - "Pre-trained Models for Natural Language Processing"
3. **Lin et al. (2021)** - "A Survey of Transformers"

## Socratic Questions

### Understanding
1. Why does sparse attention work well despite removing most attention connections?
2. How do different sparse patterns affect information flow through the network?
3. What is the relationship between attention sparsity and model interpretability?

### Extension
1. How would you design sparse attention patterns for specific structured data types?
2. Can sparse attention patterns be learned automatically during training?
3. How might sparse attention be combined with other efficiency techniques?

### Research
1. What are the fundamental limits of attention sparsity without quality loss?
2. How can we design hardware-efficient sparse attention patterns?
3. What new applications become feasible with efficient long-sequence attention?

## Exercises

### Theoretical
1. Prove that certain sparse patterns maintain the expressiveness of full attention
2. Analyze the gradient flow properties of different sparse patterns
3. Derive the optimal sparsity ratio for different computational budgets

### Implementation
1. Implement efficient sparse attention with custom CUDA kernels
2. Build adaptive sparsity patterns that change based on input content
3. Create hybrid attention mechanisms combining multiple sparse patterns
4. Develop attention pattern visualization and analysis tools

### Research
1. Study the relationship between attention entropy and optimal sparsity patterns
2. Investigate task-specific optimal sparse attention designs
3. Analyze the trade-offs between different sparse attention variants
4. Explore novel applications enabled by efficient long-sequence processing