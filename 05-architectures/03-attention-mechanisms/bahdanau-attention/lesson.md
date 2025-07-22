# Bahdanau Attention: The First Neural Attention Mechanism

## Prerequisites
- Sequence-to-sequence models and encoder-decoder architectures
- Recurrent neural networks (RNNs, LSTMs, GRUs)
- Neural machine translation fundamentals
- Backpropagation through time
- Information theory basics

## Learning Objectives
- Understand the historical development and motivation for attention mechanisms
- Master the mathematical formulation of Bahdanau (additive) attention
- Implement attention-based sequence-to-sequence models from scratch
- Analyze the alignment and interpretability properties of attention weights
- Connect Bahdanau attention to modern attention mechanisms

## Mathematical Foundations

### 1. Historical Context and Motivation

#### The RNN Bottleneck Problem
Traditional sequence-to-sequence models compress the entire source sequence into a fixed-size context vector:

```
Encoder: h_T = RNN(x_1, x_2, ..., x_T)
Decoder: y_t = f(y_{t-1}, h_T)
```

**Problems:**
1. **Information bottleneck**: All source information must pass through single vector h_T
2. **Long sequence degradation**: Earlier tokens get "forgotten" in long sequences  
3. **Fixed capacity**: Context vector size independent of source sequence length

#### The Alignment Motivation
In neural machine translation, different target words should focus on different source words. Manual alignment is expensive and rigid.

**Goal**: Learn soft alignment automatically during training.

### 2. Bahdanau Attention Mechanism

#### Definition 2.1 (Bahdanau Attention)
For encoder hidden states H = [h_1, h_2, ..., h_T] and decoder state s_t:

**Step 1 - Alignment Scores**:
```
e_t^{(i)} = v_a^T tanh(W_a s_t + U_a h_i)
```

**Step 2 - Attention Weights**:
```
α_t^{(i)} = exp(e_t^{(i)}) / Σ_{j=1}^T exp(e_t^{(j)})
```

**Step 3 - Context Vector**:
```
c_t = Σ_{i=1}^T α_t^{(i)} h_i
```

**Parameters:**
- W_a ∈ ℝ^{d_a×d_s}: Decoder state projection
- U_a ∈ ℝ^{d_a×d_h}: Encoder state projection  
- v_a ∈ ℝ^{d_a}: Attention vector
- d_a: Attention dimension (hyperparameter)

#### Key Properties

**Additive Nature**: Uses tanh(W_a s_t + U_a h_i) rather than dot products
**Learnable Alignment**: Parameters W_a, U_a, v_a learned end-to-end
**Variable Length**: Automatically handles sequences of different lengths

### 3. Detailed Mathematical Analysis

#### Attention Score Computation

**Alignment Function**:
```
align(s_t, h_i) = v_a^T tanh(W_a s_t + U_a h_i)
```

This can be viewed as:
1. **Linear projection**: Project s_t and h_i to common space ℝ^{d_a}
2. **Non-linear combination**: Use tanh to combine projections
3. **Scoring**: Inner product with learned vector v_a

#### Softmax Normalization Properties

**Theorem 3.1 (Attention Weight Properties)**
The attention weights α_t = softmax(e_t) satisfy:
1. **Non-negativity**: α_t^{(i)} ≥ 0 for all i
2. **Normalization**: Σ_i α_t^{(i)} = 1
3. **Differentiability**: ∂α_t^{(i)}/∂e_t^{(j)} exists everywhere

**Gradient Properties**:
```
∂α_t^{(i)}/∂e_t^{(j)} = α_t^{(i)} (δ_{ij} - α_t^{(j)})
```
where δ_{ij} is the Kronecker delta.

#### Context Vector Analysis

**Information Aggregation**:
The context vector c_t is a weighted average of encoder hidden states:
```
c_t = E_α[h_i] = Σ_i α_t^{(i)} h_i
```

**Properties:**
- **Convex combination**: c_t lies in convex hull of {h_1, ..., h_T}
- **Adaptive selection**: Different c_t for each decoder step t
- **Soft addressing**: Unlike hard attention, all h_i contribute

### 4. Sequence-to-Sequence with Attention

#### Complete Architecture

**Encoder**:
```
h_i = BiLSTM(x_i, h_{i-1})  for i = 1, ..., T
```

**Attention-based Decoder**:
```
c_t = Attention(s_t, H)
s_t = LSTM([y_{t-1}; c_t], s_{t-1})
y_t = softmax(W_y [s_t; c_t] + b_y)
```

Where [;] denotes concatenation.

#### Input Feeding
**Enhanced Architecture** (Luong et al.):
Feed previous attention context to current decoder step:
```
s_t = LSTM([y_{t-1}; c_{t-1}], s_{t-1})
c_t = Attention(s_t, H)
y_t = softmax(W_y [s_t; c_t] + b_y)
```

### 5. Training Procedure

#### Loss Function
Standard cross-entropy loss for sequence generation:
```
L = -Σ_{t=1}^{T'} log P(y_t | y_1, ..., y_{t-1}, X)
```

#### Backpropagation Through Attention

**Gradient w.r.t. Context Vector**:
```
∂L/∂c_t = ∂L/∂y_t × ∂y_t/∂c_t
```

**Gradient w.r.t. Attention Weights**:
```
∂L/∂α_t^{(i)} = (∂L/∂c_t)^T h_i
```

**Gradient w.r.t. Alignment Scores**:
```
∂L/∂e_t^{(i)} = (∂L/∂α_t^{(i)}) × α_t^{(i)} × (1 - α_t^{(i)}) + Σ_{j≠i} (∂L/∂α_t^{(j)}) × (-α_t^{(j)}) × α_t^{(i)}
```

#### Teacher Forcing
During training, use ground truth y_{t-1} rather than predicted tokens:
```
s_t = LSTM([y*_{t-1}; c_{t-1}], s_{t-1})
```
where y*_{t-1} is the ground truth token.

### 6. Attention Visualization and Alignment

#### Alignment Matrix
The attention weights α can be visualized as an alignment matrix:
- **Rows**: Target positions (decoder steps)
- **Columns**: Source positions (encoder steps)
- **Values**: Attention weights α_t^{(i)}

#### Interpreting Alignments

**Monotonic Alignment**: α_t^{(i)} is large when i ≈ t (diagonal pattern)
**Non-monotonic Alignment**: Target words attend to distant source words
**One-to-many**: Single source word influences multiple target words
**Many-to-one**: Multiple source words influence single target word

### 7. Theoretical Properties

#### Universal Approximation for Alignment

**Theorem 7.1**: The Bahdanau attention mechanism with sufficient capacity can approximate any alignment function α: S × H → Δ^T where Δ^T is the T-dimensional simplex.

**Proof Sketch**: The function v_a^T tanh(W_a s + U_a h) can approximate any continuous function of (s,h) given sufficient hidden dimension d_a.

#### Information Theoretic Perspective

**Attention as Information Bottleneck**:
The attention mechanism balances:
- **Compression**: Context vector c_t has fixed size
- **Prediction**: Must retain information relevant for next token

**Mutual Information**: 
```
I(c_t; y_t) = H(y_t) - H(y_t | c_t)
```

Higher attention entropy generally corresponds to more uncertain alignment.

### 8. Computational Complexity

#### Time Complexity
**Per decoder step**: O(T × d_a + T × d_h)
- Alignment computation: O(T × d_a)
- Context vector computation: O(T × d_h)

**Total training**: O(T' × T × d_a + T' × T × d_h)
where T' is target sequence length.

#### Space Complexity
**Attention weights storage**: O(T' × T) 
**Hidden states**: O(T × d_h)

#### Comparison with Fixed Context
- **Traditional seq2seq**: O(1) context computation
- **Bahdanau attention**: O(T) context computation per step
- **Trade-off**: Higher computation for better information access

### 9. Variants and Extensions

#### Global vs Local Attention

**Global Attention** (Standard Bahdanau):
Attend to all source positions at each decoder step.

**Local Attention** (Luong et al.):
Attend to a window around predicted position:
```
p_t = S × sigmoid(v_p^T tanh(W_p s_t))
α_t^{(i)} ∝ exp(e_t^{(i)}) × exp(-(i-p_t)²/2σ²)
```

#### Hierarchical Attention
Apply attention at multiple levels:
- **Word-level attention**: Within sentences
- **Sentence-level attention**: Within documents

#### Coverage Mechanism
Track cumulative attention to avoid over/under-translation:
```
Coverage_t = Σ_{t'=1}^{t-1} α_{t'}
e_t^{(i)} = v_a^T tanh(W_a s_t + U_a h_i + W_c Coverage_t^{(i)})
```

### 10. Differences from Modern Attention

#### Bahdanau vs Scaled Dot-Product

| Aspect | Bahdanau | Scaled Dot-Product |
|--------|----------|-------------------|
| **Computation** | Additive (tanh) | Multiplicative (dot product) |
| **Parameters** | W_a, U_a, v_a | W_q, W_k, W_v |
| **Complexity** | O(T×d_a) per query | O(T×d_k) per query |
| **Parallelization** | Sequential (RNN decoder) | Fully parallel |
| **Multiple heads** | Single attention head | Multi-head extension |

#### Mathematical Relationship

**Approximation Property**: For appropriate parameter choices, scaled dot-product attention can approximate Bahdanau attention:
```
v_a^T tanh(W_a s + U_a h) ≈ (s^T W_q)(h^T W_k) / √d_k
```

when tanh is approximately linear in the relevant range.

### 11. Applications and Impact

#### Neural Machine Translation
- **WMT 2014**: Bahdanau et al. achieved competitive results with attention
- **Alignment quality**: Learned alignments often match human intuitions
- **Long sequences**: Dramatic improvement on sentences >30 tokens

#### Other Sequence Tasks
- **Abstractive summarization**: Attend to relevant parts of documents
- **Image captioning**: Attend to relevant image regions
- **Speech recognition**: Align audio frames with text tokens

#### Influence on Field
- **Sparked attention revolution**: Led to Transformer and BERT
- **Interpretability**: Made neural seq2seq models more interpretable
- **Architecture search**: Inspired many attention variants

### 12. Implementation Considerations

#### Numerical Stability
**Softmax overflow prevention**:
```
e_t = e_t - max(e_t)  # Shift before softmax
α_t = exp(e_t) / sum(exp(e_t))
```

#### Efficient Computation
**Batch processing**: Compute attention for entire batch simultaneously
**Memory optimization**: Store only necessary intermediate activations

#### Initialization
**Xavier/Glorot initialization** for W_a, U_a:
```
W_a ~ Uniform(-√(6/(d_s + d_a)), √(6/(d_s + d_a)))
```

### 13. Debugging and Analysis

#### Common Issues

**Attention Collapse**: All attention mass on single position
- **Cause**: Poor initialization or learning rate
- **Solution**: Attention dropout, better initialization

**Uniform Attention**: Attention weights approximately uniform
- **Cause**: Insufficient model capacity or training
- **Solution**: Increase d_a, longer training

#### Evaluation Metrics

**Alignment Error Rate (AER)**: Compare with gold alignment
**Attention Entropy**: Measure attention distribution sharpness
**Coverage**: Ensure all source tokens receive attention

## Implementation Details

See `exercise.py` for implementations of:
1. Bahdanau attention mechanism from scratch
2. Attention-based encoder-decoder architecture  
3. Alignment visualization tools
4. Comparison with modern attention mechanisms
5. Training procedures and optimization techniques

## Experiments

1. **Translation Quality**: Compare attention vs non-attention seq2seq
2. **Alignment Analysis**: Visualize learned alignments on test data
3. **Sequence Length**: Performance on sequences of varying length
4. **Attention Variants**: Compare global, local, and coverage attention
5. **Interpretability**: Analyze attention patterns for linguistic insights

## Research Connections

### Foundational Paper
1. **Bahdanau, Cho & Bengio (2014)** - "Neural Machine Translation by Jointly Learning to Align and Translate"
   - Original attention mechanism for neural machine translation

### Key Extensions
2. **Luong, Pham, Manning (2015)** - "Effective Approaches to Attention-based Neural Machine Translation"
   - Global vs local attention, input feeding, attention variants

3. **Tu et al. (2016)** - "Modeling Coverage for Neural Machine Translation"
   - Coverage mechanism to track attention history

4. **Xu et al. (2015)** - "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
   - Application to computer vision, hard vs soft attention

### Theoretical Understanding
5. **Chorowski et al. (2015)** - "Attention-Based Models for Speech Recognition"
   - Attention in speech processing, location-based attention

6. **Dzmitry et al. (2017)** - "An Actor-Critic Algorithm for Sequence Prediction"
   - Reinforcement learning perspective on attention training

## Resources

### Primary Sources
1. **Bahdanau et al. (2014)** - Original paper introducing neural attention
2. **Luong et al. (2015)** - Comprehensive study of attention variants
3. **Neural Machine Translation and Sequence-to-sequence Models: A Tutorial** - Neubig (2017)

### Video Resources
1. **Stanford CS224N** - Attention and Memory lectures
2. **Deep Learning Specialization (Coursera)** - Sequence models with attention
3. **Attention Mechanism Explained** - Various YouTube tutorials

### Advanced Reading
1. **Rush (2018)** - "The Annotated Encoder-Decoder with Attention"
2. **Koehn (2020)** - "Neural Machine Translation" (comprehensive textbook)
3. **Goldberg (2017)** - "Neural Network Methods for Natural Language Processing"

## Socratic Questions

### Understanding  
1. Why does additive attention use tanh rather than other activation functions?
2. How does the choice of attention dimension d_a affect model capacity and performance?
3. What makes attention weights interpretable as alignment probabilities?

### Extension
1. How would you modify Bahdanau attention for very long sequences (>1000 tokens)?
2. Can attention mechanisms learn non-monotonic alignments effectively?
3. How might attention be extended to structured inputs (trees, graphs)?

### Research
1. What are the theoretical limits of what attention mechanisms can learn?
2. How can we make attention mechanisms more efficient while maintaining quality?
3. What new applications might benefit from attention-like mechanisms?

## Exercises

### Theoretical
1. Derive the gradient of Bahdanau attention with respect to encoder hidden states
2. Analyze the effect of attention dimension d_a on model expressiveness
3. Prove that attention weights form a proper probability distribution

### Implementation
1. Implement Bahdanau attention from scratch without deep learning frameworks
2. Build a complete attention-based neural machine translation system
3. Create alignment visualization tools for analyzing attention patterns
4. Implement and compare different attention variants (global, local, coverage)

### Research
1. Study attention patterns on different language pairs and identify systematic differences
2. Compare attention-based models with Transformer models on various sequence tasks
3. Investigate the relationship between attention entropy and translation quality