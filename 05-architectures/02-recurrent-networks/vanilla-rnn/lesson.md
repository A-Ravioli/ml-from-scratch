# Vanilla RNN: Recurrent Neural Networks

## Prerequisites
- Basic neural network concepts and backpropagation
- Sequence modeling and temporal dependencies
- Matrix calculus and chain rule
- Understanding of gradient flow and vanishing gradients

## Learning Objectives
- Master the fundamental RNN architecture and mathematical formulation
- Understand the vanishing gradient problem and its implications
- Implement various RNN variants including Elman and Jordan networks
- Analyze computational patterns and memory limitations
- Connect RNN concepts to modern sequence modeling approaches

## Mathematical Foundations

### 1. Sequential Data and Temporal Dependencies

#### The Need for Memory in Neural Networks
Traditional feedforward networks process inputs independently:
```
y = f(x; θ)
```

Sequential data requires models that can:
- Maintain information across time steps
- Process variable-length sequences
- Capture temporal dependencies and patterns

#### Sequential Modeling Problem
Given a sequence `x_1, x_2, ..., x_T`, we want to:
- Predict next elements: `P(x_{t+1} | x_1, ..., x_t)`
- Classify sequences: `P(y | x_1, ..., x_T)`
- Generate sequences: Sample from `P(x_1, ..., x_T)`

### 2. Vanilla RNN Architecture

#### Core Recurrence Relation
The fundamental RNN equation:
```
h_t = f(W_hh h_{t-1} + W_xh x_t + b_h)
```

Where:
- `h_t`: Hidden state at time `t` (memory)
- `x_t`: Input at time `t`
- `W_hh`: Hidden-to-hidden weight matrix (recurrent weights)
- `W_xh`: Input-to-hidden weight matrix
- `b_h`: Hidden bias vector
- `f`: Activation function (typically `tanh` or `ReLU`)

#### Output Generation
```
o_t = W_ho h_t + b_o
y_t = g(o_t)
```

Where:
- `o_t`: Output logits at time `t`
- `W_ho`: Hidden-to-output weight matrix
- `g`: Output activation (softmax for classification, linear for regression)

#### Complete Forward Pass
For sequence processing:
```
h_0 = 0  (or learned initialization)
for t = 1 to T:
    h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)
    o_t = W_ho h_t + b_o
    y_t = softmax(o_t)  # for classification
```

### 3. RNN Unfolding and Computational Graph

#### Temporal Unfolding
RNNs can be "unfolded" through time, creating a feedforward network:

```
x_1 → [RNN] → h_1 → y_1
      ↗ ↘
x_2 → [RNN] → h_2 → y_2  
      ↗ ↘
x_3 → [RNN] → h_3 → y_3
```

#### Shared Parameters
Key insight: The same parameters `W_hh, W_xh, W_ho` are used at every timestep.
This enables:
- Processing variable-length sequences
- Parameter efficiency
- Translation invariance in time

#### Memory Limitations
The hidden state `h_t` must encode all relevant information from `x_1, ..., x_t`.
This creates a compression bottleneck for long sequences.

### 4. Training via Backpropagation Through Time (BPTT)

#### Loss Function
For sequence classification:
```
L = Σ_{t=1}^T L_t(y_t, ŷ_t)
```

For sequence prediction (next-token):
```
L = -Σ_{t=1}^{T-1} log P(x_{t+1} | x_1, ..., x_t)
```

#### Gradient Computation
Using chain rule through time:
```
∂L/∂W_hh = Σ_{t=1}^T ∂L_t/∂h_t · ∂h_t/∂W_hh
```

#### Recursive Gradient Flow
```
∂h_t/∂h_{t-1} = W_hh^T · diag(f'(z_t))

∂h_t/∂h_k = ∏_{i=k+1}^t ∂h_i/∂h_{i-1} = ∏_{i=k+1}^t W_hh^T · diag(f'(z_i))
```

This product structure leads to the vanishing gradient problem.

### 5. The Vanishing Gradient Problem

#### Mathematical Analysis
For gradients flowing back `k` timesteps:
```
||∂h_t/∂h_{t-k}|| ≤ ||W_hh||^k · ∏_{i=1}^k ||diag(f'(z_{t-i+1}))||
```

#### Conditions for Vanishing
**For tanh activation**: `|f'(z)| ≤ 1`, so gradients decay as:
```
||∂h_t/∂h_{t-k}|| ≤ ||W_hh||^k
```

If `||W_hh|| < 1`, gradients vanish exponentially.

**Spectral Radius Bound**: If largest eigenvalue of `W_hh` is less than 1, gradients vanish.

#### Exploding Gradients
If `||W_hh|| > 1`, gradients can explode exponentially, causing training instability.

#### Practical Implications
- Long-term dependencies (>10-20 timesteps) are difficult to learn
- Early sequence elements have minimal influence on loss
- Training becomes ineffective for long sequences

### 6. RNN Variants and Extensions

#### Elman Networks (Standard RNN)
```
h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)
y_t = W_ho h_t + b_o
```

#### Jordan Networks
Context units store output instead of hidden state:
```
c_t = y_{t-1}  # context from output
h_t = tanh(W_ch c_t + W_xh x_t + b_h)
y_t = W_ho h_t + b_o
```

#### Bidirectional RNNs
Process sequences in both directions:
```
h⃗_t = RNN_forward(x_t, h⃗_{t-1})
h⃖_t = RNN_backward(x_t, h⃖_{t+1})
h_t = [h⃗_t; h⃖_t]  # concatenation
```

#### Deep (Stacked) RNNs
Multiple RNN layers:
```
h^(1)_t = RNN^(1)(x_t, h^(1)_{t-1})
h^(2)_t = RNN^(2)(h^(1)_t, h^(2)_{t-1})
...
y_t = f(h^(L)_t)
```

### 7. Activation Functions and Their Effects

#### Tanh Activation
```
tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})
```

**Properties**:
- Range: (-1, 1)
- Derivative: `tanh'(x) = 1 - tanh²(x)`
- Zero-centered output
- Saturates for large inputs (contributing to vanishing gradients)

#### ReLU Activation
```
ReLU(x) = max(0, x)
```

**Properties**:
- Range: [0, ∞)
- Derivative: 1 if x > 0, 0 if x ≤ 0
- Mitigates vanishing gradients
- Can cause "dying ReLU" problem
- Less common in RNNs due to unbounded activation

#### Sigmoid Activation
```
σ(x) = 1/(1 + e^{-x})
```

**Properties**:
- Range: (0, 1)
- Derivative: `σ'(x) = σ(x)(1-σ(x))`
- Severe vanishing gradient problem
- Rarely used as main RNN activation

### 8. Initialization Strategies

#### Xavier/Glorot Initialization
```
W ~ Uniform(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
```

#### He Initialization (for ReLU)
```
W ~ Normal(0, √(2/n_in))
```

#### Orthogonal Initialization for Recurrent Weights
Initialize `W_hh` as orthogonal matrix to maintain gradient flow:
```
U, S, V = SVD(random_matrix)
W_hh = U  # orthogonal matrix
```

#### Identity Initialization
Initialize `W_hh` close to identity matrix:
```
W_hh = α * I + (1-α) * random_matrix
```
where `α ≈ 0.9`.

### 9. Training Techniques and Regularization

#### Gradient Clipping
Prevent exploding gradients:
```
if ||∇|| > threshold:
    ∇ = threshold * ∇ / ||∇||
```

Common thresholds: 1.0 to 10.0.

#### Truncated Backpropagation Through Time (TBPTT)
Limit gradient flow to `k` timesteps:
- Forward pass: Full sequence
- Backward pass: Only last `k` steps
- Reduces computational cost and memory usage

#### Dropout in RNNs
**Standard dropout** (not recommended for recurrent connections):
```
h_t = tanh(W_hh · dropout(h_{t-1}) + W_xh x_t + b_h)
```

**Variational dropout**: Same dropout mask across all timesteps.

#### Teacher Forcing
During training, use ground truth instead of predictions:
```
Training: h_t = RNN(x_t, h_{t-1})  # use true x_t
Inference: h_t = RNN(ŷ_{t-1}, h_{t-1})  # use predicted ŷ_{t-1}
```

### 10. Computational Complexity

#### Time Complexity
- Forward pass: O(T · (H² + HI + HO))
  - T: sequence length
  - H: hidden size
  - I: input size
  - O: output size

- Backward pass (BPTT): O(T · (H² + HI + HO))

#### Space Complexity
- Parameters: O(H² + HI + HO)
- Activations (for BPTT): O(T · H)
- Can use checkpointing to trade computation for memory

#### Parallelization Challenges
RNNs are inherently sequential:
- Cannot parallelize across time dimension
- Limited GPU utilization compared to feedforward networks
- Motivated development of Transformers and other architectures

### 11. Applications and Use Cases

#### Natural Language Processing
- **Language Modeling**: Character/word-level prediction
- **Machine Translation**: Encoder-decoder architectures
- **Text Classification**: Sentiment analysis, topic classification
- **Named Entity Recognition**: Token-level sequence labeling

#### Time Series Analysis
- **Financial Forecasting**: Stock price prediction
- **Weather Prediction**: Meteorological time series
- **Signal Processing**: Audio and sensor data analysis
- **Anomaly Detection**: Identifying unusual patterns

#### Speech and Audio
- **Speech Recognition**: Acoustic modeling
- **Speech Synthesis**: Text-to-speech generation
- **Music Generation**: Melodic and rhythmic patterns
- **Audio Classification**: Sound recognition tasks

#### Control Systems
- **Robot Control**: Sequential action planning
- **Game Playing**: Strategy games with temporal elements
- **Autonomous Systems**: Navigation and decision making

### 12. Limitations and Historical Context

#### Key Limitations
1. **Vanishing Gradients**: Cannot learn long-term dependencies
2. **Sequential Processing**: Limited parallelization
3. **Memory Bottleneck**: Fixed-size hidden state
4. **Training Instability**: Sensitive to initialization and hyperparameters

#### Historical Impact
- **1980s**: Hopfield networks and early RNN concepts
- **1990**: Elman and Jordan networks
- **1997**: LSTM addresses vanishing gradient problem
- **2000s**: RNNs dominate sequence modeling
- **2014**: GRU simplifies LSTM architecture
- **2017**: Transformers challenge RNN dominance
- **Present**: RNNs still used for specific applications

#### Modern Alternatives
- **Transformers**: Parallel processing, attention mechanisms
- **CNNs**: For sequence modeling (e.g., WaveNet)
- **State Space Models**: Efficient long sequence modeling
- **Memory Networks**: External memory mechanisms

### 13. Connection to Modern Architectures

#### Attention Mechanisms
RNNs + Attention addresses information bottleneck:
```
c_t = Σ_i α_{t,i} h_i  # weighted combination of all hidden states
y_t = f(c_t, h_t)      # output uses both current and context
```

#### Transformer Relationship
Transformers can be viewed as:
- RNNs with perfect memory (attention to all positions)
- Parallel computation of all timesteps
- Self-attention replaces recurrent connections

#### Residual Connections
Highway networks and ResNets inspired by RNN gating:
```
h_t = f(x_t, h_{t-1}) + h_{t-1}  # residual connection
```

## Implementation Details

See `exercise.py` for implementations of:
1. Basic vanilla RNN cell with tanh and ReLU activations
2. Multi-layer and bidirectional RNN architectures  
3. Backpropagation through time (BPTT) implementation
4. Gradient clipping and truncated BPTT
5. RNN-based language models and sequence classifiers
6. Vanishing gradient analysis and visualization
7. Comparison with LSTM/GRU architectures
8. Various initialization strategies and their effects