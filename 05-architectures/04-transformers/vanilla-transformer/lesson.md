# Vanilla Transformer: Attention Is All You Need

## Prerequisites  
- Multi-head attention mechanisms
- Positional encodings and embeddings
- Layer normalization and residual connections
- Sequence-to-sequence modeling

## Learning Objectives
- Master the complete Transformer architecture from the seminal paper
- Understand encoder-decoder attention mechanisms
- Implement positional encodings and feed-forward networks
- Analyze computational complexity and parallelization benefits
- Connect Transformer innovations to modern architectures

## Mathematical Foundations

### 1. Complete Architecture Overview

The Transformer consists of:
- **Encoder**: Stack of N=6 identical layers
- **Decoder**: Stack of N=6 identical layers  
- **Multi-head attention**: Core attention mechanism
- **Position-wise feed-forward**: 2-layer MLP
- **Positional encoding**: Injects sequence order information

### 2. Encoder Architecture

#### Encoder Layer Components
Each encoder layer contains:
```
EncoderLayer(x) = LayerNorm(x + FFN(LayerNorm(x + MultiHeadAttn(x))))
```

**Sub-layers:**
1. Multi-head self-attention
2. Position-wise feed-forward network
3. Residual connections around each sub-layer
4. Layer normalization

#### Mathematical Formulation
```
# Self-attention
Attn_out = MultiHead(Q=x, K=x, V=x)
x₁ = LayerNorm(x + Attn_out)

# Feed-forward  
FFN_out = FFN(x₁)
x₂ = LayerNorm(x₁ + FFN_out)
```

### 3. Decoder Architecture

#### Decoder Layer Components
Each decoder layer contains:
1. Masked multi-head self-attention
2. Multi-head encoder-decoder attention
3. Position-wise feed-forward network
4. Residual connections and layer normalization

#### Mathematical Formulation
```
# Masked self-attention
SelfAttn_out = MaskedMultiHead(Q=y, K=y, V=y)
y₁ = LayerNorm(y + SelfAttn_out)

# Encoder-decoder attention
EncDecAttn_out = MultiHead(Q=y₁, K=encoder_out, V=encoder_out)
y₂ = LayerNorm(y₁ + EncDecAttn_out)

# Feed-forward
FFN_out = FFN(y₂)
y₃ = LayerNorm(y₂ + FFN_out)
```

### 4. Positional Encoding

#### Sinusoidal Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties:**
- **Deterministic**: Same encoding for same position
- **Unique**: Each position has unique encoding
- **Relative positioning**: PE(pos+k) can be expressed as linear function of PE(pos)

### 5. Feed-Forward Networks

#### Position-wise FFN
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Dimensions:**
- Input/Output: d_model = 512
- Hidden: d_ff = 2048 
- Parameters: 2 × d_model × d_ff

### 6. Training and Optimization

#### Loss Function
Cross-entropy loss with label smoothing:
```
Loss = -∑ᵢ (y_i log(ŷᵢ) + (1-α)δᵢⱼ + α/V)
```
where α = 0.1, V = vocabulary size

#### Optimization
- **Optimizer**: Adam with β₁=0.9, β₂=0.98, ε=10⁻⁹
- **Learning rate schedule**: Warm-up for 4000 steps, then decay
- **Regularization**: Dropout=0.1, label smoothing=0.1

## Implementation Details

See `exercise.py` for implementations of:
1. Complete Transformer encoder-decoder architecture
2. Multi-head attention with all variants
3. Positional encoding schemes
4. Layer normalization and residual connections
5. Position-wise feed-forward networks
6. Training loop and optimization