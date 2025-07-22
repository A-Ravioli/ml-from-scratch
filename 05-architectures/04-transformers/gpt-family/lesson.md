# GPT Family: Generative Pre-trained Transformers

## Prerequisites
- Transformer decoder architecture
- Causal (autoregressive) language modeling
- Pre-training and fine-tuning paradigms
- Scaling laws for language models

## Learning Objectives
- Master autoregressive language modeling with Transformers
- Understand the evolution from GPT-1 to GPT-4
- Implement causal attention and next-token prediction
- Analyze scaling laws and emergent capabilities
- Connect to modern large language models

## Mathematical Foundations

### 1. Autoregressive Language Modeling

#### Next-Token Prediction
GPT models the probability of a sequence as:
```
P(x₁, x₂, ..., xₙ) = ∏ᵢ₌₁ⁿ P(xᵢ | x₁, ..., xᵢ₋₁)
```

#### Causal Attention
Uses masked self-attention to prevent attending to future tokens:
```
Mask[i,j] = {
  0   if j ≤ i
  -∞  if j > i
}
```

### 2. GPT Architecture Evolution

#### GPT-1 (2018)
- **Size**: 117M parameters
- **Architecture**: 12-layer decoder-only Transformer
- **Context**: 512 tokens
- **Training**: Unsupervised pre-training + supervised fine-tuning

#### GPT-2 (2019)  
- **Size**: 124M to 1.5B parameters
- **Context**: 1024 tokens
- **Innovation**: Zero-shot task performance
- **Training**: Pure unsupervised learning

#### GPT-3 (2020)
- **Size**: 175B parameters
- **Context**: 2048 tokens
- **Innovation**: Few-shot in-context learning
- **Emergent abilities**: Translation, reasoning, code generation

#### GPT-4 (2023)
- **Multimodal**: Text + vision inputs
- **Enhanced reasoning**: Better mathematical and logical capabilities
- **Alignment**: Improved safety and instruction following

### 3. Core Components

#### Transformer Decoder Block
```
x = x + CausalSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

#### Position Embeddings
- **Learned**: Trainable position embeddings
- **Sinusoidal**: Fixed trigonometric encodings
- **Rotary (RoPE)**: Rotary position embedding in modern variants

#### Token Embeddings
- **Vocabulary**: 50,000+ tokens (BPE/SentencePiece)
- **Embedding dimension**: Tied to model dimension
- **Weight sharing**: Output projection often shares weights with embeddings

### 4. Training Methodology

#### Pre-training Objective
Cross-entropy loss for next-token prediction:
```
L = -∑ᵢ log P(xᵢ₊₁ | x₁, ..., xᵢ)
```

#### Data and Scale
- **Datasets**: Common Crawl, WebText, Books, Wikipedia
- **Compute**: Thousands of GPUs, months of training
- **Tokens**: Hundreds of billions to trillions

#### Scaling Laws
Performance scales predictably with:
- Model size (parameters)
- Dataset size (tokens)
- Compute budget (FLOPs)

### 5. Fine-tuning and Adaptation

#### Supervised Fine-tuning
Fine-tune on task-specific datasets for downstream applications

#### RLHF (Reinforcement Learning from Human Feedback)
- Reward modeling from human preferences
- PPO optimization for alignment
- Used in ChatGPT, GPT-4

#### In-Context Learning
Learn tasks from examples in the input prompt without parameter updates

## Implementation Details

See `exercise.py` for implementations of:
1. GPT architecture with causal attention
2. Next-token prediction training
3. Text generation with various decoding strategies
4. Fine-tuning procedures
5. Scaling analysis tools