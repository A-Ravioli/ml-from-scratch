# GRU: Gated Recurrent Unit

## Prerequisites
- LSTM architecture and gating mechanisms
- Vanilla RNN limitations and vanishing gradients
- Matrix operations and backpropagation through time
- Understanding of hidden state dynamics

## Learning Objectives
- Master the GRU architecture and its simplified gating mechanism
- Understand the mathematical foundations of reset and update gates
- Compare GRU vs LSTM trade-offs and performance characteristics
- Implement bidirectional and stacked GRU variants
- Analyze computational efficiency and memory requirements

## Mathematical Foundations

### 1. Motivation: Simplifying LSTM Architecture

#### LSTM Complexity Analysis
LSTMs use three gates and two states:
- Forget gate: `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`
- Input gate: `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`
- Output gate: `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`
- Cell state: `C_t = f_t * C_{t-1} + i_t * tanh(W_C · [h_{t-1}, x_t] + b_C)`
- Hidden state: `h_t = o_t * tanh(C_t)`

#### GRU Simplification Philosophy
GRUs achieve similar performance with:
- Only two gates instead of three
- Single state instead of separate hidden/cell states
- Fewer parameters and faster computation

### 2. GRU Architecture

#### Core Components
GRU uses two gates controlling information flow:

**Reset Gate** (`r_t`): Controls how much past information to forget
```
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
```

**Update Gate** (`z_t`): Controls how much new information to accept
```
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
```

#### Hidden State Computation

**Candidate Hidden State**: New information with selective reset
```
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
```

**Final Hidden State**: Linear interpolation between old and new
```
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### 3. Mathematical Analysis

#### Gate Functionality

**Reset Gate Behavior**:
- `r_t ≈ 0`: Ignore previous hidden state (fresh start)
- `r_t ≈ 1`: Fully consider previous hidden state
- Controls how much past context influences new candidate

**Update Gate Behavior**:
- `z_t ≈ 0`: Keep old hidden state unchanged
- `z_t ≈ 1`: Replace with new candidate state
- Acts like a learnable interpolation coefficient

#### Information Flow Analysis

The hidden state update can be rewritten as:
```
h_t = z_t ⊙ h̃_t + (1 - z_t) ⊙ h_{t-1}
```

This is a convex combination where:
- `z_t` acts as a learned "mixing coefficient"
- When `z_t = 0`: perfect memory (copy previous state)
- When `z_t = 1`: complete update (ignore previous state)

### 4. Gradient Flow Properties

#### Gradient Through Update Gate
```
∂h_t/∂h_{t-1} = (1 - z_t) + z_t ⊙ ∂h̃_t/∂h_{t-1}
```

The `(1 - z_t)` term provides a direct path for gradients when the update gate is closed.

#### Long-Range Dependencies
When `z_t ≈ 0` for many timesteps:
```
h_t ≈ h_{t-k}  for small z_{t-k+1}, ..., z_t
```

This creates highway connections similar to ResNets, enabling long-range gradient flow.

#### Vanishing Gradient Mitigation
The direct connection `(1 - z_t) ⊙ h_{t-1}` in the hidden state update allows gradients to flow through unchanged when the update gate learns to preserve information.

### 5. GRU vs LSTM Comparison

#### Parameter Efficiency
**LSTM parameters** (per layer):
- 4 weight matrices: `W_f, W_i, W_o, W_C` each of size `(input_size + hidden_size) × hidden_size`
- 4 bias vectors: `b_f, b_i, b_o, b_C` each of size `hidden_size`
- Total: `4 × (input_size + hidden_size + 1) × hidden_size`

**GRU parameters** (per layer):
- 3 weight matrices: `W_r, W_z, W_h` 
- 3 bias vectors: `b_r, b_z, b_h`
- Total: `3 × (input_size + hidden_size + 1) × hidden_size`

GRU has ~25% fewer parameters than LSTM.

#### Computational Complexity
**Per timestep operations**:
- LSTM: 4 matrix multiplications + 4 element-wise operations
- GRU: 3 matrix multiplications + 3 element-wise operations

GRU is typically 25-30% faster than LSTM.

#### Memory Requirements
- LSTM: Stores both `h_t` and `C_t` (2 × hidden_size per timestep)
- GRU: Stores only `h_t` (1 × hidden_size per timestep)

GRU uses 50% less memory for storing states.

### 6. GRU Variants

#### Minimal Gated Unit (MGU)
Further simplified version with single gate:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
h̃_t = tanh(W_h · [(1-f_t) ⊙ h_{t-1}, x_t] + b_h)
h_t = f_t ⊙ h_{t-1} + (1-f_t) ⊙ h̃_t
```

#### Peephole GRU
Gates can observe the previous hidden state directly:
```
r_t = σ(W_r^x x_t + W_r^h h_{t-1} + b_r)
z_t = σ(W_z^x x_t + W_z^h h_{t-1} + b_z)
```

#### Coupled Input-Forget GRU
Alternative formulation coupling reset and update:
```
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
z_t = 1 - r_t  # Coupled gates
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
h_t = z_t ⊙ h_{t-1} + (1-z_t) ⊙ h̃_t
```

### 7. Bidirectional and Stacked GRUs

#### Bidirectional GRU
Process sequence in both directions:
```
h⃗_t = GRU_forward(x_t, h⃗_{t-1})
h⃖_t = GRU_backward(x_t, h⃖_{t+1})
h_t = [h⃗_t; h⃖_t]  # Concatenation
```

Benefits:
- Access to both past and future context
- Better representation for sequence classification
- Commonly used in BERT-style bidirectional encoders

#### Stacked (Deep) GRU
Multiple GRU layers:
```
h^(1)_t = GRU^(1)(x_t, h^(1)_{t-1})
h^(2)_t = GRU^(2)(h^(1)_t, h^(2)_{t-1})
...
h^(L)_t = GRU^(L)(h^(L-1)_t, h^(L)_{t-1})
```

Design considerations:
- Residual connections between layers
- Layer normalization
- Dropout for regularization

### 8. Training Considerations

#### Initialization Strategies
**Reset Gate Bias**: Initialize to 0 (neutral reset)
**Update Gate Bias**: Initialize to 1 (bias toward memory)
**Weight Matrices**: Xavier or He initialization

#### Gradient Clipping
Essential for stable training:
```
if ||∇|| > threshold:
    ∇ = threshold * ∇ / ||∇||
```

#### Learning Rate Scheduling
GRUs often benefit from:
- Warmup followed by decay
- Different learning rates for gates vs. hidden transformation
- Adaptive optimization (Adam, RMSprop)

### 9. Applications and Use Cases

#### Natural Language Processing
- **Machine Translation**: Encoder-decoder architectures
- **Language Modeling**: Character/word-level generation
- **Sentiment Analysis**: Classification of sequences
- **Named Entity Recognition**: Token-level classification

#### Time Series Analysis
- **Financial Forecasting**: Stock price prediction
- **Weather Modeling**: Meteorological sequences
- **Sensor Data**: IoT and monitoring applications
- **Medical Signals**: ECG, EEG analysis

#### Speech Processing
- **Speech Recognition**: Acoustic modeling
- **Speech Synthesis**: Text-to-speech systems
- **Voice Activity Detection**: Audio segmentation
- **Speaker Recognition**: Identity verification

### 10. Performance Characteristics

#### Empirical Comparisons
Research findings on GRU vs LSTM:
- **Speed**: GRU typically 25-30% faster
- **Memory**: GRU uses ~50% less memory
- **Accuracy**: Task-dependent; often comparable
- **Long sequences**: LSTM sometimes better for very long dependencies
- **Small datasets**: GRU often performs better due to lower parameter count

#### When to Choose GRU
- Limited computational resources
- Shorter to medium-length sequences
- Need for faster training/inference
- Parameter efficiency is important
- Memory constraints exist

#### When to Choose LSTM
- Very long sequences (>1000 timesteps)
- Complex temporal patterns
- Abundant training data
- Maximum accuracy is critical
- Explicit memory control needed

## Implementation Details

See `exercise.py` for implementations of:
1. Basic GRU cell with reset and update gates
2. Multi-layer and bidirectional GRU networks
3. GRU-based language models and sequence classifiers
4. Performance comparisons with LSTM and vanilla RNN
5. Gradient flow analysis and visualization
6. Memory and computational efficiency benchmarks