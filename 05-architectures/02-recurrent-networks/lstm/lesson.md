# LSTM: Long Short-Term Memory Networks

## Prerequisites

- Vanilla RNN architecture and limitations
- Gradient flow and vanishing gradient problem
- Gating mechanisms and sigmoid activation
- Sequence modeling concepts

## Learning Objectives

- Master the LSTM cell design and gating mechanisms
- Understand how LSTMs solve the vanishing gradient problem
- Implement bidirectional and stacked LSTM architectures
- Analyze memory flow and information retention
- Connect to modern sequence modeling approaches

## Mathematical Foundations

### 1. The Vanishing Gradient Problem in RNNs

#### Gradient Flow in Vanilla RNNs

For a vanilla RNN with hidden state recursion:

```LaTeX
h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)
```

The gradient flows as:

```LaTeX
∂h_t/∂h_{t-k} = ∏_{i=1}^k ∂h_{t-i+1}/∂h_{t-i}
```

#### Vanishing Gradient Analysis
Each term in the product:
```
∂h_{t-i+1}/∂h_{t-i} = W_hh^T diag(tanh'(z_{t-i+1}))
```

Since `tanh'(z) ≤ 1`, gradients vanish exponentially with sequence length.

### 2. LSTM Architecture

#### Core Design Principle
LSTMs maintain two states:
- **Hidden state** `h_t`: Short-term memory (output)
- **Cell state** `C_t`: Long-term memory (internal)

#### The Three Gates
1. **Forget Gate**: What to remove from cell state
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

2. **Input Gate**: What new information to store
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

3. **Output Gate**: What to output based on cell state
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
```

#### State Updates
Cell state update (the highway):
```
C_t = f_t * C_{t-1} + i_t * C̃_t
```

Hidden state update:
```
h_t = o_t * tanh(C_t)
```

### 3. Information Flow Analysis

#### Cell State as Highway
The cell state `C_t` acts as an "information highway":
- Uninterrupted gradient flow when `f_t ≈ 1`
- Selective information addition via `i_t * C̃_t`
- Gradient magnitude preserved across many timesteps

#### Gate Functionality
- **Forget gate**: `f_t ≈ 0` removes information, `f_t ≈ 1` preserves it
- **Input gate**: `i_t ≈ 0` ignores new input, `i_t ≈ 1` incorporates it
- **Output gate**: `o_t` controls how much cell state influences output

### 4. Gradient Flow in LSTMs

#### Cell State Gradient
```
∂C_t/∂C_{t-1} = f_t
```

When `f_t ≈ 1`, gradients flow unimpeded through the cell state.

#### Hidden State Gradient
```
∂h_t/∂h_{t-1} = o_t * tanh'(C_t) * i_t * C̃'_t * W_h^i + 
                  o_t * tanh'(C_t) * f_t * ∂C_{t-1}/∂h_{t-1} +
                  o_t * tanh'(C_t) * i_t * C̃'_t * W_h^f * f'_t +
                  o'_t * tanh(C_t) * W_h^o
```

The multiple pathways prevent complete gradient vanishing.

### 5. LSTM Variants

#### Peephole Connections
Gates can "peek" at cell state:
```
f_t = σ(W_f · [h_{t-1}, x_t, C_{t-1}] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t, C_{t-1}] + b_i)
o_t = σ(W_o · [h_{t-1}, x_t, C_t] + b_o)
```

#### Coupled Input-Forget Gates
Simplify by coupling input and forget:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = 1 - f_t
```

#### GRU (Gated Recurrent Unit)
Simplified alternative with only two gates:
```
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)  # Reset gate
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)  # Update gate
h̃_t = tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

### 6. Bidirectional and Stacked LSTMs

#### Bidirectional LSTM
Process sequence in both directions:
```
h⃗_t = LSTM_forward(x_t, h⃗_{t-1}, C⃗_{t-1})
h⃖_t = LSTM_backward(x_t, h⃖_{t+1}, C⃖_{t+1})
h_t = [h⃗_t; h⃖_t]  # Concatenation
```

#### Stacked (Deep) LSTM
Stack multiple LSTM layers:
```
h^(1)_t = LSTM^(1)(x_t, h^(1)_{t-1}, C^(1)_{t-1})
h^(2)_t = LSTM^(2)(h^(1)_t, h^(2)_{t-1}, C^(2)_{t-1})
...
h^(L)_t = LSTM^(L)(h^(L-1)_t, h^(L)_{t-1}, C^(L)_{t-1})
```

### 7. Training and Optimization

#### Backpropagation Through Time (BPTT)
- Unfold LSTM across time steps
- Compute gradients via chain rule
- Truncated BPTT for long sequences

#### Gradient Clipping
Prevent exploding gradients:
```
if ||∇|| > threshold:
    ∇ = threshold * ∇ / ||∇||
```

#### Initialization Strategies
- Forget gate bias: Initialize to 1 (default to remembering)
- Other biases: Initialize to 0
- Weights: Xavier or orthogonal initialization

### 8. Applications and Use Cases

#### Sequence-to-Sequence Tasks
- Machine translation
- Text summarization  
- Speech recognition
- Time series forecasting

#### Sequence Classification
- Sentiment analysis
- Document classification
- Activity recognition

#### Language Modeling
- Character-level generation
- Word-level prediction
- Code generation

## Implementation Details

See `exercise.py` for implementations of:
1. Basic LSTM cell with all gates
2. Bidirectional LSTM networks
3. Stacked LSTM architectures
4. Gradient flow analysis
5. Performance comparison with vanilla RNNs