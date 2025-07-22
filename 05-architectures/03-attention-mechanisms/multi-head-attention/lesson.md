# Multi-Head Attention: Parallel Attention with Multiple Perspectives

## Prerequisites
- Self-attention mechanisms and scaled dot-product attention
- Linear algebra (matrix multiplication, eigendecomposition)
- Neural network fundamentals (backpropagation, parameter sharing)
- Information theory (capacity, mutual information)

## Learning Objectives
- Understand the motivation and mathematical formulation of multi-head attention
- Master the parallel computation of multiple attention heads
- Implement multi-head self-attention from scratch
- Analyze head specialization and attention diversity
- Connect multi-head attention to ensemble methods and representation learning

## Mathematical Foundations

### 1. Motivation for Multiple Heads

#### Single-Head Attention Limitations
Standard self-attention computes a single alignment between queries and keys:
```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```

**Limitations:**
1. **Single subspace**: Attention operates in one representation subspace
2. **Limited expressivity**: Cannot capture multiple types of relationships simultaneously  
3. **Attention collapse**: May focus on dominant patterns, missing subtle dependencies
4. **Representation bottleneck**: Fixed dimensionality limits information capacity

#### The Multi-Perspective Hypothesis
Different types of relationships require different attention patterns:
- **Syntactic relationships**: Subject-verb agreement, dependency parsing
- **Semantic relationships**: Coreference, entity relationships  
- **Positional patterns**: Local dependencies, long-range connections
- **Task-specific patterns**: Different heads specialize for different aspects

### 2. Multi-Head Attention Definition

#### Definition 2.1 (Multi-Head Attention)
Given input X ∈ ℝ^{n×d_model}, compute h parallel attention heads:

**Step 1 - Parallel Projections**:
```
Q_i = XW_i^Q,  K_i = XW_i^K,  V_i = XW_i^V  for i = 1, ..., h
```

**Step 2 - Parallel Attention**:
```
head_i = Attention(Q_i, K_i, V_i) = softmax(Q_iK_i^T/√d_k)V_i
```

**Step 3 - Concatenation and Projection**:
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
```

**Parameters:**
- W_i^Q, W_i^K, W_i^V ∈ ℝ^{d_model×d_k}: Head-specific projection matrices
- W^O ∈ ℝ^{hd_v×d_model}: Output projection matrix
- Typically: d_k = d_v = d_model/h

#### Key Properties

**Parallel Computation**: All heads computed simultaneously
**Parameter Efficiency**: Total parameters ≈ single-head attention  
**Expressive Power**: Can represent h different attention patterns
**Information Aggregation**: Final projection combines all head outputs

### 3. Detailed Mathematical Analysis

#### Dimension Analysis
With h heads and d_k = d_v = d_model/h:

**Input**: X ∈ ℝ^{n×d_model}
**Per-head projections**: Q_i, K_i, V_i ∈ ℝ^{n×d_k}  
**Per-head attention**: head_i ∈ ℝ^{n×d_v}
**Concatenation**: Concat(heads) ∈ ℝ^{n×(h·d_v)} = ℝ^{n×d_model}
**Output**: MultiHead(X) ∈ ℝ^{n×d_model}

#### Parameter Count Analysis

**Per head**: 3d_model × d_k parameters (W^Q, W^K, W^V)
**All heads**: h × 3d_model × d_k = 3d_model^2 parameters (when d_k = d_model/h)
**Output projection**: (h × d_v) × d_model = d_model^2 parameters  
**Total**: 4d_model^2 parameters

**Comparison with single-head**: Single large head would need ≈ 3d_model^2 + d_model^2 = 4d_model^2 parameters. Multi-head uses same parameter budget more efficiently.

### 4. Information-Theoretic Perspective

#### Representation Capacity

**Theorem 4.1 (Multi-Head Capacity)**
Multi-head attention with h heads can represent any function that single-head attention with dimension hd_k can represent, but with additional structural constraints that may improve generalization.

**Proof Sketch**: The concatenation operation preserves all information from individual heads, and the output projection W^O can reconstruct any linear combination.

#### Attention Diversity

**Definition 4.1 (Attention Diversity)**
For attention heads with weight matrices A^{(1)}, ..., A^{(h)}, diversity can be measured as:
```
Diversity(A^{(1)}, ..., A^{(h)}) = (1/h²) ∑_{i,j} ||A^{(i)} - A^{(j)}||_F
```

Higher diversity indicates heads are capturing different patterns.

#### Information Bottleneck Perspective
Each head creates an information bottleneck:
- **Compression**: Projects d_model → d_k dimensions
- **Prediction**: Maintains information relevant for the task
- **Specialization**: Different heads compress different aspects

### 5. Head Specialization Analysis

#### Empirical Head Patterns

Research has identified common head specialization patterns:

**Positional Heads**: Focus on specific relative positions
- **Adjacent tokens**: Head attends primarily to i±1 positions
- **Syntax heads**: Attend to syntactic relationships (subject→verb)

**Content Heads**: Focus on semantic relationships
- **Entity heads**: Track entity mentions and coreferences
- **Sentiment heads**: Capture emotional or evaluative content

**Global vs Local Heads**:
- **Local heads**: High attention entropy, uniform distribution
- **Global heads**: Low entropy, focused attention

#### Head Importance Analysis

**Definition 5.1 (Head Importance)**
Importance of head i can be measured by performance drop when head i is removed:
```
Importance_i = Performance(all_heads) - Performance(all_heads \ {i})
```

**Pruning Implications**: Heads with low importance can often be removed without significant performance loss.

### 6. Computational Aspects

#### Parallel Computation Benefits

**Traditional Sequential Attention**: O(h × computation_per_head)
**Multi-head Parallel**: O(computation_per_head) with h-fold parallelism

**Memory Layout Optimization**:
```python
# Efficient: Single matrix multiplication for all heads
all_QKV = X @ W_all  # W_all = [W₁^Q W₁^K W₁^V ... W_h^Q W_h^K W_h^V]
Q, K, V = split(all_QKV, dim=-1)  # Split into Q₁,...,Q_h, K₁,...,K_h, V₁,...,V_h
```

#### Attention Computation Optimization

**Batched Matrix Multiplication**:
```
# Instead of loop over heads
for i in range(h):
    head_i = attention(Q_i, K_i, V_i)

# Use batched operations
Q_all = [Q_1; Q_2; ...; Q_h]  # Shape: [h×n×d_k]
K_all = [K_1; K_2; ...; K_h]  # Shape: [h×n×d_k]  
V_all = [V_1; V_2; ...; V_h]  # Shape: [h×n×d_v]
heads = batch_attention(Q_all, K_all, V_all)  # Shape: [h×n×d_v]
```

### 7. Variants and Extensions

#### Multi-Query Attention (MQA)
Share keys and values across heads, separate queries:
```
Q_i = XW_i^Q  for i = 1, ..., h    # Separate queries
K = XW^K,  V = XW^V                # Shared keys/values
head_i = Attention(Q_i, K, V)
```

**Benefits**: Reduced memory usage, faster inference
**Trade-off**: Potentially reduced expressivity

#### Grouped-Query Attention (GQA)
Intermediate between multi-head and multi-query:
- Divide heads into g groups
- Share K,V within groups, separate Q per head
- Reduces parameters while maintaining some diversity

#### Local Multi-Head Attention
Apply different window sizes per head:
```
head_i = LocalAttention(Q_i, K_i, V_i, window_size_i)
```

### 8. Training Dynamics and Optimization

#### Gradient Flow Analysis

**Theorem 8.1 (Multi-Head Gradient Flow)**
Multi-head attention provides multiple gradient paths between positions, potentially improving training stability compared to single-head attention.

**Head-specific Gradients**:
```
∂L/∂W_i^Q = ∂L/∂head_i × ∂head_i/∂Q_i × ∂Q_i/∂W_i^Q
```

Different heads can learn at different rates based on gradient magnitudes.

#### Initialization Strategies

**Xavier/Glorot Initialization** for projection matrices:
```
W_i^Q ~ Uniform(-√(6/(d_model + d_k)), √(6/(d_model + d_k)))
```

**Head Diversity Initialization**: Initialize heads with different random seeds to encourage diverse specialization.

#### Learning Rate Scheduling
Different heads may benefit from different learning rates:
- **Syntax heads**: Often learn quickly, may need lower learning rates
- **Semantic heads**: May require more iterations, higher learning rates

### 9. Analysis and Interpretability

#### Attention Head Visualization

**Head-specific Attention Maps**: Visualize attention patterns for each head separately
**Head Comparison**: Compare attention patterns across heads for same input
**Token-to-Token Analysis**: Track how specific token pairs are attended to by different heads

#### Head Clustering Analysis
Group heads by similarity of attention patterns:
```
similarity(head_i, head_j) = correlation(flatten(A_i), flatten(A_j))
```

#### Probing Head Functionality
- **Syntactic probes**: Test if heads capture syntactic relationships
- **Semantic probes**: Evaluate semantic understanding in head representations
- **Task-specific probes**: Assess head relevance for specific downstream tasks

### 10. Multi-Head vs Single-Head Comparison

#### Expressivity Comparison

| Aspect | Single-Head | Multi-Head |
|--------|-------------|------------|
| **Attention patterns** | One global pattern | Multiple specialized patterns |
| **Parameter efficiency** | Lower (larger dimensions) | Higher (distributed parameters) |
| **Interpretability** | Single attention map | Multiple interpretable heads |
| **Robustness** | Single point of failure | Redundant representations |
| **Training dynamics** | May get stuck in local minima | Multiple learning paths |

#### Performance Analysis

**Empirical Results**: Multi-head consistently outperforms single-head with same parameter budget
**Ablation Studies**: Performance degrades gracefully when heads are removed
**Task Dependence**: Benefits vary by task complexity and structure

### 11. Advanced Applications

#### Hierarchical Multi-Head Attention
Different heads operate at different granularities:
- **Token-level heads**: Fine-grained local patterns
- **Chunk-level heads**: Medium-range dependencies  
- **Document-level heads**: Global document structure

#### Cross-Modal Multi-Head Attention
Different heads specialize for different modalities:
- **Visual heads**: Process image features
- **Textual heads**: Process language features
- **Cross-modal heads**: Align vision and language

#### Task-Specific Head Design
- **Translation**: Heads for source attention, target attention, alignment
- **Summarization**: Content heads, position heads, importance heads
- **Question Answering**: Question heads, context heads, answer heads

### 12. Implementation Considerations

#### Memory Optimization
**Gradient Checkpointing**: Trade computation for memory during backpropagation
**Head Pruning**: Remove less important heads to reduce memory/computation
**Dynamic Head Selection**: Activate different heads for different inputs

#### Numerical Stability
**Attention Weight Clipping**: Prevent extreme attention weights
**Gradient Clipping**: Stabilize training with large models
**Mixed Precision**: Use lower precision for heads with stable attention patterns

#### Efficient Implementation
**Fused Operations**: Combine multiple operations into single kernels
**Memory Layout**: Optimize tensor layouts for hardware efficiency
**Batch Processing**: Maximize throughput with efficient batching

## Implementation Details

See `exercise.py` for implementations of:
1. Multi-head self-attention from scratch
2. Efficient parallel head computation
3. Head analysis and visualization tools
4. Attention head pruning techniques
5. Multi-head attention variants (MQA, GQA)
6. Head specialization analysis tools

## Experiments

1. **Head Specialization Study**: Analyze learned attention patterns across heads
2. **Ablation Analysis**: Study performance impact of removing individual heads
3. **Head Pruning**: Determine minimal head configurations for different tasks
4. **Attention Diversity**: Measure and optimize attention diversity across heads
5. **Computational Efficiency**: Compare single-head vs multi-head efficiency
6. **Cross-Task Transfer**: Study head transferability across different tasks

## Research Connections

### Foundational Papers
1. **Vaswani et al. (2017)** - "Attention Is All You Need"
   - Original multi-head attention in Transformer architecture

2. **Clark et al. (2019)** - "What Does BERT Look At? An Analysis of BERT's Attention"
   - Comprehensive analysis of multi-head attention patterns in BERT

3. **Michel et al. (2019)** - "Are Sixteen Heads Really Better than One?"
   - Head pruning and importance analysis

### Head Analysis and Interpretability
4. **Voita et al. (2019)** - "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned"
   - Head specialization and pruning analysis

5. **Kovaleva et al. (2019)** - "Revealing the Dark Secrets of BERT"
   - Deep analysis of BERT's multi-head attention patterns

6. **Tenney et al. (2019)** - "What do you learn from context? Probing for sentence structure in contextualized word representations"
   - Probing head functionality for linguistic structure

### Efficiency and Variants
7. **Shazeer (2019)** - "Fast Transformer Decoding: One Write-Head is All You Need"
   - Multi-Query Attention (MQA) for efficient inference

8. **Ainslie et al. (2023)** - "GQA: Training Generalized Multi-Query Transformer Models"
   - Grouped-Query Attention balancing efficiency and performance

## Resources

### Primary Sources
1. **Vaswani et al. (2017)** - Original Transformer paper with multi-head attention
2. **The Illustrated Transformer** - Jay Alammar's visual explanation
3. **The Annotated Transformer** - Line-by-line implementation guide

### Analysis and Visualization Tools
1. **BertViz** - Interactive attention visualization tool
2. **Attention Analysis Toolkit** - Comprehensive attention analysis tools
3. **Transformer Interpretability** - Various interpretation methods

### Advanced Reading
1. **Rogers et al. (2020)** - "A Primer on Neural Network Models for Natural Language Processing"
2. **Qiu et al. (2020)** - "Pre-trained Models for Natural Language Processing: A Survey"
3. **Tay et al. (2020)** - "Efficient Transformers: A Survey"

## Socratic Questions

### Understanding
1. Why does multi-head attention work better than single-head attention with the same parameter budget?
2. How do different heads learn to specialize for different types of relationships?
3. What determines the optimal number of attention heads for a given task?

### Extension  
1. How would you design attention heads specifically for structured data (graphs, trees)?
2. Can attention head patterns be transferred between different architectures?
3. How might multi-head attention be adapted for continual learning scenarios?

### Research
1. What are the theoretical limits of attention head specialization?
2. How can we design more interpretable and controllable attention heads?
3. What new applications might benefit from task-specific multi-head designs?

## Exercises

### Theoretical
1. Derive the computational complexity of multi-head vs single-head attention
2. Prove that multi-head attention can represent any single-head attention pattern
3. Analyze the effect of head number on representational capacity

### Implementation
1. Implement efficient multi-head attention with batched operations
2. Build attention head visualization and analysis tools
3. Create head pruning algorithms based on importance scores
4. Implement multi-query and grouped-query attention variants

### Research
1. Study head specialization patterns across different model sizes and tasks
2. Investigate optimal head configurations for specific domains
3. Analyze attention head evolution during training
4. Explore novel multi-head attention architectures for specific applications