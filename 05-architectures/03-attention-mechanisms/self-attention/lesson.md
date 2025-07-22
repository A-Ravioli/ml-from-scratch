# Self-Attention: The Foundation of Modern Deep Learning

## Prerequisites
- Linear algebra (matrix multiplication, eigenvalues, SVD)
- Neural network fundamentals (backpropagation, optimization)
- Sequence modeling basics (RNNs, language modeling)
- Information theory (entropy, mutual information)

## Learning Objectives
- Master the mathematical foundations of self-attention mechanisms
- Understand the attention function and its computational properties
- Implement scaled dot-product attention from scratch
- Analyze computational complexity and parallelization benefits
- Connect self-attention to fundamental machine learning concepts

## Mathematical Foundations

### 1. The Attention Function

#### Definition 1.1 (General Attention)
An attention mechanism is a function that maps queries, keys, and values to an output:
```
Attention(Q, K, V) = f(Q, K, V)
```
where:
- Q ∈ ℝ^{n×d_k} are queries
- K ∈ ℝ^{m×d_k} are keys  
- V ∈ ℝ^{m×d_v} are values
- Output ∈ ℝ^{n×d_v}

#### The Core Insight
Attention computes a weighted average of values, where weights are determined by the compatibility between queries and keys.

### 2. Scaled Dot-Product Attention

#### Definition 2.1 (Scaled Dot-Product Attention)
The fundamental attention mechanism used in Transformers:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Step-by-step breakdown**:
1. **Similarity**: S = QK^T ∈ ℝ^{n×m}
2. **Scaling**: S_scaled = S / √d_k  
3. **Normalization**: A = softmax(S_scaled) ∈ ℝ^{n×m}
4. **Aggregation**: Output = AV ∈ ℝ^{n×d_v}

#### Why Scaling by √d_k?

**Theorem 2.1 (Scaling Necessity)**
Let q, k ∈ ℝ^d with components drawn independently from N(0, 1). Then:
- E[q^T k] = 0
- Var(q^T k) = d

As d increases, dot products have larger variance, pushing softmax into saturation regions.

**Proof**:
```
Var(q^T k) = Var(∑_{i=1}^d q_i k_i) = ∑_{i=1}^d Var(q_i k_i) = ∑_{i=1}^d E[q_i^2]E[k_i^2] = d
```

Scaling by √d_k ensures Var((q^T k)/√d_k) = 1, maintaining gradient flow.

### 3. Self-Attention Mechanism

#### Definition 3.1 (Self-Attention)
When queries, keys, and values all come from the same input sequence:
```
X ∈ ℝ^{n×d}  (input sequence)
Q = XW_Q,  K = XW_K,  V = XW_V
SelfAttention(X) = Attention(Q, K, V)
```

Where W_Q, W_K, W_V ∈ ℝ^{d×d_k} are learned parameter matrices.

#### Properties of Self-Attention

**Permutation Equivariance**: 
For any permutation matrix P:
```
SelfAttention(PX) = P · SelfAttention(X)
```

**Position Independence**: Self-attention has no inherent notion of position or order.

### 4. Attention Weights Analysis

#### Attention Matrix Properties
The attention matrix A = softmax(QK^T/√d_k) satisfies:
1. **Non-negativity**: A_{ij} ≥ 0 for all i,j
2. **Row-wise normalization**: ∑_j A_{ij} = 1 for all i
3. **Differentiability**: Smooth everywhere

#### Information-Theoretic Interpretation

**Definition 4.1 (Attention Entropy)**
For each query position i:
```
H_i = -∑_{j=1}^m A_{ij} log A_{ij}
```

- **High entropy**: Attention is spread across many positions
- **Low entropy**: Attention is focused on few positions

#### Attention Patterns

**Local Attention**: A_{ij} is large only when |i-j| is small
**Global Attention**: A_{ij} can be large for any i,j  
**Sparse Attention**: Most A_{ij} are near zero

### 5. Computational Complexity

#### Time Complexity
**Forward Pass**: O(n²d + nd²)
- QK^T computation: O(n²d)
- Softmax: O(n²)  
- Attention-Value multiplication: O(n²d)
- Linear transformations: O(nd²)

**Backward Pass**: O(n²d + nd²)
Similar complexity due to gradient computation through attention weights.

#### Space Complexity
**Memory**: O(n² + nd)
- Attention matrix: O(n²)
- Activations: O(nd)

#### The Quadratic Bottleneck
For long sequences (n large), the O(n²) term dominates, making standard self-attention prohibitive for very long sequences.

### 6. Gradient Flow Analysis

#### Attention Gradient Computation

**Gradient w.r.t. Values**:
```
∂L/∂V = A^T (∂L/∂Output)
```

**Gradient w.r.t. Attention Weights**:
```
∂L/∂A = (∂L/∂Output) V^T
```

**Gradient w.r.t. Queries/Keys** (via chain rule):
Complex but well-behaved due to softmax properties.

#### Gradient Flow Properties

**Theorem 6.1 (Attention Gradient Flow)**
Self-attention provides direct gradient paths between all pairs of positions, potentially improving long-range dependency learning compared to RNNs.

### 7. Masked Self-Attention

#### Causal Masking
For autoregressive tasks, prevent attention to future positions:
```
Mask[i,j] = {
  0   if j ≤ i
  -∞  if j > i
}
MaskedAttention(Q,K,V) = softmax((QK^T + Mask)/√d_k)V
```

#### Padding Masking
For variable-length sequences, mask padded positions:
```
Mask[i,j] = {
  0   if position j is valid
  -∞  if position j is padding
}
```

### 8. Attention as Soft Dictionary Lookup

#### Dictionary Interpretation
View attention as a differentiable dictionary lookup:
- **Keys**: Dictionary indices
- **Values**: Dictionary content
- **Queries**: Lookup requests
- **Attention weights**: Soft addressing mechanism

#### Nearest Neighbor Connection
In the limit where one attention weight approaches 1:
```
lim_{A_{ij} → 1} ∑_k A_{ik} V_k = V_j
```
This recovers exact nearest neighbor lookup.

### 9. Multi-Query and Multi-Key Extensions

#### Multi-Query Attention
Share keys and values across multiple heads while keeping separate queries:
```
MultiQuery(Q₁,...,Q_h, K, V) = [Attention(Q₁,K,V); ...; Attention(Q_h,K,V)]
```

**Benefits**: Reduced memory usage, faster inference

#### Grouped-Query Attention
Intermediate between multi-head and multi-query:
- Divide heads into groups
- Share K,V within groups
- Keep separate Q for each head

### 10. Advanced Attention Variants

#### Additive Attention (Bahdanau)
```
Attention(Q,K,V) = V^T softmax(v^T tanh(W_Q Q + W_K K))
```
where v, W_Q, W_K are learned parameters.

#### Multiplicative Attention
```
Attention(Q,K,V) = V^T softmax(Q^T W K)
```
where W is a learned compatibility matrix.

#### Scaled Dot-Product Advantages
- **Computational efficiency**: Matrix operations are highly optimized
- **Theoretical properties**: Well-understood scaling behavior
- **Empirical performance**: Excellent results across tasks

### 11. Attention Visualization and Interpretability

#### Attention Weight Interpretation
High attention weights A_{ij} suggest that:
- Position i "attends to" position j
- Information from position j is relevant for position i
- There's a dependency relationship between positions

#### Limitations of Attention Interpretation
1. **Multiple valid explanations**: High attention doesn't imply causation
2. **Head specialization**: Different heads may capture different types of dependencies
3. **Distributed representations**: Information may be spread across multiple attention patterns

### 12. Position Information in Self-Attention

#### The Position Problem
Self-attention is permutation equivariant, lacking inherent position information.

#### Positional Encoding Solutions
1. **Absolute positional encodings**: Add position-dependent vectors to inputs
2. **Relative positional encodings**: Modify attention computation with relative positions
3. **Learned positional embeddings**: Train position-specific parameters

### 13. Self-Attention vs Other Mechanisms

#### vs Recurrent Neural Networks
| Aspect | RNN | Self-Attention |
|--------|-----|----------------|
| **Parallelization** | Sequential | Fully parallel |
| **Long-range deps** | Gradient decay | Direct connections |
| **Memory complexity** | O(1) | O(n²) |
| **Position modeling** | Implicit | Requires encoding |

#### vs Convolutional Networks
| Aspect | CNN | Self-Attention |
|--------|-----|----------------|
| **Receptive field** | Local, growing | Global from layer 1 |
| **Parameter sharing** | Across positions | Across pairs |
| **Inductive bias** | Locality, translation equivariance | Permutation equivariance |

### 14. Theoretical Connections

#### Connection to Kernel Methods
Self-attention can be viewed as a kernel method:
```
K(q_i, k_j) = exp(q_i^T k_j / √d_k) / ∑_l exp(q_i^T k_l / √d_k)
```

#### Connection to Message Passing
Self-attention performs one step of message passing on a complete graph:
- **Nodes**: Sequence positions
- **Messages**: Attention-weighted values
- **Aggregation**: Weighted sum

#### Connection to Retrieval Systems
Self-attention implements a soft retrieval mechanism:
- **Query**: What information do I need?
- **Key matching**: Which positions have relevant information?
- **Value retrieval**: Extract and combine relevant information

## Implementation Details

See `exercise.py` for implementations of:
1. Scaled dot-product attention from scratch
2. Masked self-attention for causal modeling
3. Multi-head self-attention
4. Efficient attention computation optimizations
5. Attention visualization tools
6. Comparative analysis with RNN/CNN approaches

## Experiments

1. **Attention Pattern Analysis**: Visualize learned attention patterns on different tasks
2. **Sequence Length Scaling**: Study computational complexity empirically
3. **Gradient Flow Comparison**: Compare gradient propagation with RNNs
4. **Ablation Studies**: Effect of scaling, masking, and number of heads
5. **Interpretability Analysis**: Relationship between attention weights and task performance

## Research Connections

### Foundational Papers
1. **Bahdanau et al. (2014)** - "Neural Machine Translation by Jointly Learning to Align and Translate"
   - Introduced attention mechanism for sequence-to-sequence models

2. **Vaswani et al. (2017)** - "Attention Is All You Need"
   - Scaled dot-product attention and Transformer architecture

3. **Luong et al. (2015)** - "Effective Approaches to Attention-based Neural Machine Translation"
   - Global vs local attention, attention function variants

### Theoretical Understanding
4. **Clark et al. (2019)** - "What Does BERT Look At? An Analysis of BERT's Attention"
   - Empirical analysis of attention patterns in trained models

5. **Rogers et al. (2020)** - "A Primer on Neural Network Models for Natural Language Processing"
   - Comprehensive survey including attention mechanisms

6. **Elhage et al. (2021)** - "A Mathematical Framework for Transformer Circuits"
   - Mechanistic interpretability of attention

### Extensions and Improvements
7. **Shaw et al. (2018)** - "Self-Attention with Relative Position Representations"
   - Relative positional encoding in self-attention

8. **Shen et al. (2018)** - "DiSAN: Directional Self-Attention Network"
   - Directional self-attention for better position modeling

## Resources

### Primary Sources
1. **Vaswani et al. (2017)** - Original Transformer paper
2. **The Annotated Transformer** - Step-by-step implementation guide
3. **Attention and Memory in Deep Learning** - Distill.pub visual explanation

### Video Resources
1. **CS224N Stanford** - Self-attention lecture by Christopher Manning
2. **The Illustrated Transformer** - Jay Alammar's visual guide
3. **3Blue1Brown** - "Attention in transformers, visually explained"

### Advanced Reading
1. **Tay et al. (2020)** - "Efficient Transformers: A Survey"
2. **Kenton & Toutanova (2019)** - "BERT: Pre-training of Deep Bidirectional Transformers"
3. **Liu et al. (2019)** - "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

## Socratic Questions

### Understanding
1. Why does self-attention use dot products rather than other similarity functions?
2. How does the scaling factor √d_k affect gradient flow and training dynamics?
3. What makes self-attention more parallelizable than RNNs?

### Extension
1. How would you modify self-attention for very long sequences (>10k tokens)?
2. Can self-attention be made more parameter efficient while maintaining performance?
3. How might self-attention be adapted for structured data (graphs, trees)?

### Research
1. What are the fundamental computational limits of attention mechanisms?
2. How can we better understand what attention patterns represent semantically?
3. Can attention be made more interpretable without sacrificing performance?

## Exercises

### Theoretical
1. Derive the gradient of scaled dot-product attention with respect to queries
2. Prove that self-attention is permutation equivariant
3. Analyze the effect of different temperature parameters in softmax attention

### Implementation
1. Implement efficient attention computation using memory-optimized algorithms
2. Build attention visualization tools to analyze learned patterns
3. Create masked attention variants for different sequence modeling tasks
4. Implement relative positional attention mechanisms

### Research
1. Study attention pattern evolution during training on different tasks
2. Compare attention patterns between random and trained networks
3. Investigate the relationship between attention entropy and model performance
4. Analyze how attention patterns change with different architectural choices