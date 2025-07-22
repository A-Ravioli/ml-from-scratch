# Advanced RNN Architectures

## Prerequisites
- LSTM and GRU architectures
- Attention mechanisms and memory systems
- Neural Turing Machine concepts
- Differentiable programming principles

## Learning Objectives
- Master advanced RNN variants: Neural Turing Machines, Differentiable Neural Computers
- Understand external memory mechanisms and differentiable read/write operations
- Implement attention-based memory addressing and content-based lookup
- Analyze algorithmic reasoning capabilities and systematic generalization
- Connect to modern memory-augmented neural networks

## Mathematical Foundations

### 1. Memory-Augmented Neural Networks

#### Limitations of Standard RNNs
Standard RNNs suffer from:
- **Fixed Memory Capacity**: Hidden state size limits memory
- **Interference**: New information overwrites old information
- **No Explicit Structure**: Cannot model algorithmic operations
- **Poor Generalization**: Struggle with systematic reasoning

#### External Memory Paradigm
Memory-augmented networks separate:
- **Controller**: Neural network (RNN/LSTM/GRU) for processing
- **Memory**: External storage with structured access
- **Interface**: Differentiable read/write mechanisms

### 2. Neural Turing Machines (NTM)

#### Architecture Overview
NTM consists of:
1. **Controller Network**: LSTM/GRU processing unit
2. **Memory Matrix**: `M ∈ ℝ^{N×M}` (N locations, M-dimensional)
3. **Read/Write Heads**: Attention-based memory access

#### Memory Operations

**Reading from Memory**:
```
r_t = Σ_i w_t^r(i) M_t(i)
```
where `w_t^r(i)` is attention weight for location `i`.

**Writing to Memory**:
```
M_t(i) = M_{t-1}(i) [1 - w_t^w(i) e_t] + w_t^w(i) a_t
```
where:
- `e_t`: Erase vector (what to remove)
- `a_t`: Add vector (what to add)
- `w_t^w(i)`: Write attention weights

#### Attention Mechanisms

**Content-Based Addressing**:
```
w_c^t(i) = exp(β_t K[k_t, M_t(i)]) / Σ_j exp(β_t K[k_t, M_t(j)])
```
where:
- `k_t`: Key vector produced by controller
- `K[u,v]`: Similarity function (cosine similarity)
- `β_t`: Key strength (sharpening parameter)

**Location-Based Addressing**:
```
w_g^t = g_t w_c^t + (1-g_t) w_{t-1}
```
where `g_t` is interpolation gate (0 = use previous, 1 = use content).

**Shifting**:
```
w̃_t(i) = Σ_j w_g^t(j) s_t((i-j) mod N)
```
where `s_t` is shift distribution.

**Sharpening**:
```
w_t(i) = (w̃_t(i))^{γ_t} / Σ_j (w̃_t(j))^{γ_t}
```
where `γ_t ≥ 1` controls concentration.

### 3. Differentiable Neural Computer (DNC)

#### Improvements over NTM
DNC addresses NTM limitations:
- **Memory Allocation**: Dynamic memory management
- **Temporal Links**: Track write order for sequential access
- **Usage Vector**: Prevent interference between reads/writes

#### Memory Allocation

**Usage Vector**: Tracks memory utilization
```
u_t = (u_{t-1} + w_{t-1}^w - u_{t-1} ⊙ w_{t-1}^w) ⊙ ψ_t
```
where `ψ_t` is retention vector.

**Allocation Weights**:
```
a_t = (1 - u_t) ∏_{j=1}^{i-1} u_t(j)  # for i-th location
```

**Write Weights**:
```
w_t^w = α_t^w [β_t c_t + (1-β_t) a_t]
```
where:
- `α_t^w`: Write gate
- `β_t`: Allocation gate
- `c_t`: Content-based weights

#### Temporal Link Matrix

**Link Matrix**: `L_t ∈ ℝ^{N×N}` tracks write precedence
```
L_t[i,j] = (1 - w_t^w(i) - w_t^w(j)) L_{t-1}[i,j] + w_t^w(i) p_{t-1}(j)
```

**Precedence Vector**: `p_t(i) = (1 - Σ_j w_t^w(j)) p_{t-1}(i) + w_t^w(i)`

**Forward/Backward Weights**:
```
f_t = L_t^T w_{t-1}^r  # Forward direction
b_t = L_t w_{t-1}^r    # Backward direction
```

#### Read Operations

**Read Weights**:
```
w_t^{r,i} = π_t^i [β_t^i c_t^i + (1-β_t^i)(f_t^i + b_t^i)]
```

**Read Vectors**:
```
r_t^i = Σ_j w_t^{r,i}(j) M_t(j)
```

### 4. Attention-Based Memory Models

#### Memory Networks
Components:
1. **Input Module**: I(x) - converts input to internal representation
2. **Generalization Module**: G(I,m) - updates memories
3. **Output Module**: O(I,m) - produces response
4. **Response Module**: R(o) - generates final answer

#### End-to-End Memory Networks

**Multiple Hops**: Iterative attention over memory
```
u^1 = Σ_i p_i^1 m_i  # First hop
u^{k+1} = Σ_i p_i^{k+1} m_i + u^k  # Subsequent hops
```

**Attention Weights**:
```
p_i^k = softmax(u^{k-1} · m_i)
```

#### Dynamic Memory Networks

**Episodic Memory Module**:
1. **Attention Mechanism**: Select relevant facts
2. **Memory Update**: Update episode representation
3. **Iteration**: Multiple passes over memory

**Question Module**: Processes question into vector representation
**Input Module**: Converts facts into memory representations
**Answer Module**: Generates response from memory

### 5. Transformer-Style Memory

#### Self-Attention Memory
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Multi-Head Attention**:
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### Memory-Efficient Attention

**Linformer**: Low-rank approximation
```
Attention ≈ softmax(QK^T P_K/√d_k) P_V V
```

**Performer**: Random feature approximation
```
K(x,y) ≈ E[φ(x)^T φ(y)]
```

### 6. Recurrent Memory Architectures

#### Differentiable Stack/Queue

**Stack Operations**:
- **Push**: Add to top with strength `s_t`
- **Pop**: Remove from top with strength `s_t`  
- **No-op**: Maintain current state

**Stack State Update**:
```
V_t^j = max(0, V_{t-1}^j - max(0, s_t - Σ_{i=j+1}^t V_{t-1}^i))
```

**Stack Read**:
```
r_t = Σ_j V_t^j v_j
```

#### Adaptive Computation Time

**Ponder Cost**: Additional computation for thinking
```
C_t = Σ_{n=1}^{N(t)} p_n^t + R(1 - Σ_{n=1}^{N(t)} p_n^t)
```

**Halting Probability**: When to stop computing
```
h_n^t = σ(W_h s_n^t + b_h)
```

### 7. Graph-Structured Memory

#### Graph Neural Networks for Memory

**Message Passing**:
```
m_{ij}^{(l+1)} = M^{(l)}(h_i^{(l)}, h_j^{(l)}, e_{ij})
h_i^{(l+1)} = U^{(l)}(h_i^{(l)}, Σ_{j∈N(i)} m_{ij}^{(l+1)})
```

**Memory Graph Update**:
- Nodes represent memory slots
- Edges represent relationships
- Messages propagate information

#### Relational Memory

**Relation Network**:
```
o_i = f_φ(Σ_j g_θ(x_i, x_j))
```

**Multi-Head Relational Memory**:
```
MHA(X) = Concat(head_1,...,head_h)W^O
head_i = Attention(XW_i^Q, XW_i^K, XW_i^V)
```

### 8. Meta-Learning with Memory

#### Model-Agnostic Meta-Learning (MAML)
```
θ' = θ - α∇_θ L_τ(f_θ)
θ^* = θ - β∇_θ Σ_τ L_τ(f_{θ'})
```

#### Memory-Augmented Meta-Learning

**External Memory for Few-Shot Learning**:
- Store task-specific information
- Retrieve relevant patterns
- Adapt quickly to new tasks

**Differentiable Plasticity**:
```
w_{ij}^{new} = w_{ij} + η A_{ij} Hebb_{ij}
```

### 9. Training Advanced RNNs

#### Curriculum Learning
- Start with simple sequences
- Gradually increase complexity
- Focus on algorithmic patterns

#### Auxiliary Losses
- **Copy Task**: Test memory capacity
- **Associative Recall**: Test content addressing
- **Priority Sort**: Test algorithmic reasoning

#### Regularization Techniques
- **Dropout**: On controller outputs
- **Memory Regularization**: Encourage sparse access
- **Temporal Consistency**: Smooth attention weights

### 10. Applications and Use Cases

#### Algorithmic Tasks
- **Sorting**: Priority queues and comparison sorting
- **Graph Traversal**: BFS/DFS with memory
- **Dynamic Programming**: Optimal substructure
- **Parsing**: Context-free grammar recognition

#### Question Answering
- **Reading Comprehension**: Multi-hop reasoning
- **Knowledge Base QA**: Structured knowledge access
- **Visual QA**: Spatial reasoning with memory

#### Program Synthesis
- **Code Generation**: Structured output generation
- **Neural Programming**: Differentiable interpreters
- **Inductive Programming**: Learning from examples

#### Reasoning Tasks
- **Logic Puzzles**: Systematic constraint solving
- **Mathematical Reasoning**: Multi-step problem solving
- **Causal Reasoning**: Temporal relationship modeling

## Implementation Details

See `exercise.py` for implementations of:
1. Neural Turing Machine with content and location addressing
2. Differentiable Neural Computer with memory allocation
3. Memory Networks for question answering
4. Attention-based external memory systems
5. Differentiable stack and queue data structures
6. Adaptive computation time mechanisms
7. Training procedures and evaluation metrics
8. Benchmark tasks for algorithmic reasoning