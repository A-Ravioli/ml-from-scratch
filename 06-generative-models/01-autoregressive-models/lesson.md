# Autoregressive Models for Generative Modeling

## Prerequisites
- Probability theory (conditional probability, chain rule)
- Information theory (entropy, cross-entropy, perplexity)
- Recurrent neural networks and sequence modeling
- Maximum likelihood estimation
- Basic understanding of language models

## Learning Objectives
- Master the mathematical foundations of autoregressive generation
- Understand the probabilistic chain rule decomposition
- Implement autoregressive models for different data modalities
- Analyze the theoretical properties and limitations
- Connect autoregressive models to modern large language models

## Mathematical Foundations

### 1. The Autoregressive Principle

#### Definition 1.1 (Autoregressive Model)
An autoregressive model decomposes the joint probability of a sequence using the chain rule:

p(x₁, x₂, ..., xₜ) = ∏ᵢ₌₁ᵀ p(xᵢ | x₁, ..., xᵢ₋₁)

**Key insight**: Model complex joint distributions through sequential conditional distributions.

#### Causal Structure
**Constraint**: xᵢ can only depend on x₁, ..., xᵢ₋₁ (past context)
**Benefit**: Enables tractable generation and exact likelihood computation
**Trade-off**: No bidirectional context during generation

#### Universal Approximation Property
**Theorem 1.1**: Any distribution over finite sequences can be exactly represented by an autoregressive model with sufficient capacity.

**Proof sketch**: Chain rule decomposition is exact; need only approximate conditional distributions.

### 2. Maximum Likelihood Training

#### Objective Function
For dataset D = {x⁽¹⁾, x⁽²⁾, ..., x⁽ᴺ⁾}:

L(θ) = ∑ₙ₌₁ᴺ ∑ᵢ₌₁ᵀ log p(xᵢ⁽ⁿ⁾ | x₁⁽ⁿ⁾, ..., xᵢ₋₁⁽ⁿ⁾; θ)

**Benefits**:
- Exact likelihood computation
- Stable training objective
- Principled probabilistic framework

#### Teacher Forcing
**Training**: Use ground truth previous tokens
**Issue**: Exposure bias during generation

**Algorithm 2.1 (Teacher Forcing)**:
```
for each sequence x in batch:
    for t = 1 to T:
        # Use ground truth context
        context = x[1:t-1]  
        prediction = model(context)
        loss += cross_entropy(prediction, x[t])
```

#### Exposure Bias Problem
**Training**: Model sees ground truth context
**Inference**: Model sees its own predictions
**Consequence**: Error accumulation during generation

**Solutions**:
- Scheduled sampling
- Minimum risk training
- Adversarial training

### 3. Architecture Design Principles

#### Causality Constraints
**Requirement**: Future tokens cannot influence past predictions

**Implementation approaches**:
1. **Recurrent**: Natural causal structure (RNNs, LSTMs)
2. **Convolutional**: Causal convolutions with limited receptive field
3. **Attention**: Causal masking in self-attention

#### Context Length
**Trade-off**: Longer context vs computational complexity
- **RNNs**: Unlimited context but sequential computation
- **CNNs**: Fixed context window but parallel computation
- **Transformers**: Quadratic attention but flexible context

#### Parameter Sharing
**Benefit**: Same parameters model all conditional distributions
**Challenge**: Need sufficient capacity for varying conditional complexities

### 4. Recurrent Autoregressive Models

#### Vanilla RNN Formulation
hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ)
p(xₜ₊₁ | x₁:ₜ) = softmax(Wₕᵧhₜ + bᵧ)

**Issues**:
- Vanishing gradients for long sequences
- Limited modeling capacity
- Difficulty with long-range dependencies

#### LSTM/GRU Improvements
**LSTM cell**:
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)  (forget gate)
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)  (input gate)
C̃ₜ = tanh(WC · [hₜ₋₁, xₜ] + bC)  (candidate values)
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ  (cell state)
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)  (output gate)
hₜ = oₜ * tanh(Cₜ)

**Benefits**:
- Better gradient flow
- Selective memory mechanisms
- Improved long-range modeling

#### Modern RNN Variants
**Highway Networks**: Learnable skip connections
**Residual RNNs**: ResNet-style connections for RNNs
**Layer Normalization**: Stabilizes training in recurrent models

### 5. Convolutional Autoregressive Models

#### PixelRNN/PixelCNN
**Motivation**: Model images pixel by pixel

**Ordering**: Raster scan order (top-left to bottom-right)
p(x) = ∏ᵢ₌₁ᴴ ∏ⱼ₌₁ᵂ p(xᵢⱼ | x₁:ᵢ⁻¹, x₁:ⱼ⁻¹)

#### Causal Convolutions
**Standard convolution**: Uses future context
**Causal convolution**: Zero-pad to ensure causality

**1D Causal Conv**: Only use past time steps
**2D Causal Conv**: Mask to use only previous pixels

#### Masked Convolutions
**Type A mask**: For first layer (excludes current pixel)
**Type B mask**: For hidden layers (includes current pixel for other channels)

**Implementation**:
```python
def masked_conv2d(mask_type='A'):
    mask = torch.ones(kernel_size, kernel_size)
    mask[kernel_size//2 + (mask_type=='B'):] = 0
    mask[kernel_size//2, kernel_size//2 + (mask_type=='B'):] = 0
    return mask
```

#### WaveNet Architecture
**Application**: Audio generation
**Innovation**: Dilated causal convolutions

**Dilated convolution**: Exponentially increasing receptive field
Receptive field: 2^L where L is number of layers

**Gated activation**:
tanh(Wf * x) ⊙ σ(Wg * x)

where ⊙ denotes element-wise multiplication.

### 6. Attention-Based Autoregressive Models

#### Transformer Decoder
**Self-attention with causal masking**:
Attention(Q, K, V) = softmax(QK^T/√d + M)V

where M is causal mask: Mᵢⱼ = -∞ if j > i, 0 otherwise.

#### Positional Encoding
**Need**: Attention is permutation-invariant
**Solution**: Add position information

**Sinusoidal encoding**:
PE(pos, 2i) = sin(pos/10000^(2i/d))
PE(pos, 2i+1) = cos(pos/10000^(2i/d))

**Learned encoding**: Learnable position embeddings

#### Multi-Head Attention
**Idea**: Multiple attention "heads" capture different relationships

MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O

where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)

### 7. Modern Large Language Models

#### GPT Architecture Evolution
**GPT-1**: 117M parameters, 12 layers
**GPT-2**: 1.5B parameters, improved training
**GPT-3**: 175B parameters, in-context learning
**GPT-4**: Multimodal capabilities

#### Scaling Laws
**Empirical observations**:
- Performance scales predictably with parameters
- Compute-optimal training requires balanced scaling
- Emergent abilities at scale

**Chinchilla scaling**: Equal allocation to parameters and data

#### In-Context Learning
**Phenomenon**: Models learn new tasks from examples in context
**Mechanism**: Not fully understood theoretically
**Capabilities**: Few-shot learning without gradient updates

### 8. Training Dynamics and Optimization

#### Curriculum Learning
**Strategy**: Start with simpler sequences, increase complexity
**Benefits**: Faster convergence, better final performance
**Implementation**: Sort by length, difficulty metrics

#### Learning Rate Scheduling
**Warm-up**: Gradually increase learning rate
**Decay**: Reduce learning rate during training
**Cosine schedule**: Smooth decrease following cosine curve

#### Gradient Clipping
**Problem**: Exploding gradients in recurrent models
**Solution**: Clip gradient norm

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

### 9. Generation Strategies

#### Greedy Decoding
**Strategy**: Always pick most likely next token
argmax p(xₜ | x₁:ₜ₋₁)

**Issues**: 
- Deterministic output
- Can lead to repetitive text
- Locally optimal but globally suboptimal

#### Sampling Methods
**Ancestral sampling**: Sample from full distribution
xₜ ~ p(xₜ | x₁:ₜ₋₁)

**Temperature scaling**: Control randomness
p'(xₜ | x₁:ₜ₋₁) = softmax(logits / τ)

where τ > 1 increases randomness, τ < 1 decreases it.

#### Top-k Sampling
**Strategy**: Sample from k most likely tokens
**Benefits**: Avoids very low-probability tokens
**Parameter**: k controls diversity

#### Nucleus (Top-p) Sampling
**Strategy**: Sample from smallest set with cumulative probability ≥ p
**Adaptive**: Effective vocabulary size varies by context
**Benefits**: More consistent quality than top-k

#### Beam Search
**Strategy**: Maintain multiple partial sequences
**Objective**: Find sequence with highest probability
**Issues**: Can lead to generic, repetitive outputs

### 10. Evaluation Metrics

#### Perplexity
**Definition**: Exponential of cross-entropy
PPL = exp(-1/T ∑ᵢ₌₁ᵀ log p(xᵢ | x₁:ᵢ₋₁))

**Interpretation**: Average number of equally likely choices
**Lower is better**: Perfect model has perplexity 1

#### BLEU Score
**Application**: Text generation evaluation
**Method**: n-gram overlap with reference text
**Issues**: Doesn't capture semantic similarity well

#### Human Evaluation
**Gold standard**: Human judgments of quality
**Aspects**: Fluency, coherence, relevance
**Challenges**: Expensive, subjective, hard to scale

### 11. Applications and Domains

#### Natural Language Processing
**Language Modeling**: Foundation for many NLP tasks
**Machine Translation**: Encoder-decoder with attention
**Summarization**: Generate summaries autoregressively
**Dialogue**: Conversational AI systems

#### Computer Vision
**Image Generation**: PixelRNN/CNN for images
**Video Generation**: Extend to temporal dimension
**Super-resolution**: Generate high-res from low-res

#### Audio Processing
**Speech Synthesis**: WaveNet and variants
**Music Generation**: Model musical sequences
**Audio Compression**: Learned compression codecs

#### Other Domains
**Molecular Design**: Generate molecular structures
**Code Generation**: Programming language modeling
**Time Series**: Financial data, weather prediction

### 12. Theoretical Analysis

#### Expressivity
**Theorem 12.1**: Autoregressive models with sufficient capacity can represent any discrete distribution exactly.

**Proof**: Chain rule decomposition is exact; universal approximation applies to each conditional.

#### Sample Complexity
**Generalization**: How much data needed for good generalization?
**Factors**: Vocabulary size, sequence length, model capacity
**Trade-offs**: Memorization vs generalization

#### Computational Complexity
**Training**: O(TN) for T-length sequences, N examples
**Generation**: O(T) sequential steps (cannot parallelize)
**Memory**: O(T) for storing context

### 13. Limitations and Failure Modes

#### Exposure Bias
**Problem**: Training/inference mismatch
**Consequence**: Error accumulation during generation
**Mitigation**: Scheduled sampling, RL training

#### Mode Collapse
**Symptom**: Repetitive, low-diversity outputs
**Causes**: Local optima in probability space
**Solutions**: Diverse beam search, stochastic decoding

#### Length Bias
**Problem**: Model may prefer shorter sequences
**Cause**: Length normalization in beam search
**Solution**: Length penalties, coverage mechanisms

#### Computational Inefficiency
**Sequential generation**: Cannot parallelize during inference
**Long sequences**: Memory and compute grow with length
**Real-time applications**: Latency constraints

### 14. Recent Advances

#### Parallel Generation
**Non-autoregressive models**: Generate all tokens simultaneously
**Trade-off**: Speed vs quality
**Methods**: Iterative refinement, mask-based generation

#### Retrieval-Augmented Generation
**Idea**: Combine parametric model with external memory
**Benefits**: Access to up-to-date information
**Challenges**: Integration, consistency

#### Constitutional AI
**Goal**: Align model behavior with human values
**Methods**: Constitutional training, RLHF
**Challenges**: Defining and measuring alignment

#### Mixture of Experts
**Scaling**: Increase parameters without increasing compute
**Architecture**: Route tokens to specialized sub-models
**Benefits**: Efficient scaling to very large models

### 15. Implementation Considerations

#### Memory Management
**Gradient checkpointing**: Trade compute for memory
**Sequence packing**: Efficiently batch variable-length sequences
**Dynamic batching**: Adjust batch size based on sequence length

#### Distributed Training
**Data parallelism**: Distribute batches across GPUs
**Model parallelism**: Split large models across devices
**Pipeline parallelism**: Pipeline different layers

#### Hardware Optimization
**Mixed precision**: Use FP16 for faster training
**Kernel fusion**: Optimize memory access patterns
**Compilation**: XLA, TorchScript for optimization

## Implementation Details

See `exercise.py` for implementations of:
1. Character-level RNN language model
2. Convolutional autoregressive model (PixelCNN-style)
3. Transformer decoder for text generation
4. Various generation strategies (greedy, sampling, beam search)
5. Evaluation metrics (perplexity, BLEU)
6. Training loop with teacher forcing

## Experiments

1. **Model Comparison**: RNN vs CNN vs Transformer on same dataset
2. **Generation Quality**: Compare different sampling strategies
3. **Scaling Study**: Performance vs model size and data
4. **Sequence Length**: How well do models handle long sequences?
5. **Domain Transfer**: Train on one domain, test on another

## Research Connections

### Foundational Papers
1. Bengio et al. (2003) - "A Neural Probabilistic Language Model"
2. Mikolov et al. (2010) - "Recurrent Neural Network Based Language Model"
3. Sutskever et al. (2011) - "Generating Text with Recurrent Neural Networks"
4. Karpathy & Fei-Fei (2015) - "Deep Visual-Semantic Alignments"

### Modern Developments
1. Vaswani et al. (2017) - "Attention Is All You Need"
2. Radford et al. (2018) - "Improving Language Understanding by Generative Pre-Training" (GPT)
3. Brown et al. (2020) - "Language Models are Few-Shot Learners" (GPT-3)
4. Chowdhery et al. (2022) - "PaLM: Scaling Language Modeling with Pathways"

### Theoretical Analysis
1. Arora et al. (2018) - "A Compressed Sensing View of Unsupervised Text Embeddings"
2. Tay et al. (2020) - "Efficient Transformers: A Survey"
3. Wei et al. (2022) - "Emergent Abilities of Large Language Models"

## Resources

### Primary Sources
1. **Goodfellow, Bengio & Courville - Deep Learning (Ch 10)**
   - RNN fundamentals and language modeling
2. **Jurafsky & Martin - Speech and Language Processing**
   - NLP applications and evaluation
3. **Karpathy - The Unreasonable Effectiveness of Recurrent Neural Networks**
   - Intuitive understanding of RNN capabilities

### Video Resources
1. **Stanford CS224N - Language Models and RNNs**
   - Christopher Manning's NLP course
2. **Fast.ai - Natural Language Processing**
   - Practical implementation focus
3. **Andrej Karpathy - Neural Networks: Zero to Hero**
   - Building language models from scratch

### Software Resources
1. **Hugging Face Transformers**: Pre-trained autoregressive models
2. **OpenAI GPT Models**: API and fine-tuning
3. **Google T5**: Text-to-text transfer transformer

## Socratic Questions

### Understanding
1. Why is the autoregressive factorization exact while other generative models make approximations?
2. How does the choice of ordering affect model performance in autoregressive models?
3. What are the fundamental trade-offs between different architectural choices (RNN vs CNN vs Transformer)?

### Extension
1. How would you design an autoregressive model for structured data (graphs, trees)?
2. Can you prove bounds on the sample complexity of autoregressive models?
3. How might quantum computing change autoregressive generation?

### Research
1. What are the theoretical limits of in-context learning in autoregressive models?
2. How can we eliminate the sequential bottleneck in autoregressive generation?
3. What new architectures might replace the transformer for autoregressive modeling?

## Exercises

### Theoretical
1. Prove that autoregressive models can exactly represent any discrete distribution
2. Derive the gradient updates for teacher forcing in RNN language models
3. Analyze the computational complexity of different generation strategies

### Implementation
1. Build a character-level RNN language model from scratch
2. Implement PixelCNN for image generation
3. Create a mini-GPT with causal attention
4. Compare different sampling strategies on the same model

### Research
1. Study the relationship between model size and generation quality
2. Investigate techniques for mitigating exposure bias
3. Explore novel architectures for efficient autoregressive modeling

## Advanced Topics

### Non-Autoregressive Generation
- **Parallel generation**: All tokens simultaneously
- **Iterative refinement**: Gradually improve initial generation
- **Mask-based methods**: BERT-style bidirectional generation

### Controllable Generation
- **Conditional models**: Generate based on attributes
- **Guided generation**: Steer generation toward desired properties
- **Plug-and-play methods**: External control without retraining

### Multimodal Autoregressive Models
- **Vision-language**: Generate text from images
- **Speech-text**: Unified speech and text modeling
- **Cross-modal transfer**: Leverage one modality to improve another

### Efficient Autoregressive Models
- **Sparse attention**: Reduce quadratic complexity
- **Caching mechanisms**: Reuse computations across time steps
- **Low-rank approximations**: Compress attention matrices