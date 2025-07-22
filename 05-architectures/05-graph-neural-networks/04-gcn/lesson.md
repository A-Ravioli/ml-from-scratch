# Graph Convolutional Networks (GCN)

## Prerequisites
- Graph theory basics and matrix representations
- Convolutional neural networks fundamentals
- Spectral graph theory and eigendecompositions
- Message passing frameworks

## Learning Objectives
- Master spectral and spatial approaches to graph convolutions
- Understand the connection between CNNs and GCNs
- Implement Chebyshev polynomials and localized filters
- Analyze inductive vs transductive learning on graphs
- Connect to modern graph neural network architectures

## Mathematical Foundations

### 1. Graph Representation and Notation

#### Graph Definition
Graph `G = (V, E)` where:
- `V`: Set of nodes `{v_1, v_2, ..., v_N}` with `|V| = N`
- `E`: Set of edges `{(v_i, v_j)}` with `|E| = M`

#### Key Matrices
**Adjacency Matrix**: `A ∈ {0,1}^{N×N}`
```
A_ij = {1 if (v_i, v_j) ∈ E
        0 otherwise}
```

**Degree Matrix**: `D ∈ ℝ^{N×N}` (diagonal)
```
D_ii = Σ_j A_ij  (degree of node i)
```

**Laplacian Matrix**: `L = D - A`

**Normalized Laplacian**: `L_norm = I - D^{-1/2} A D^{-1/2}`

#### Node Features
- Input features: `X ∈ ℝ^{N×F}` (F features per node)
- Output features: `H ∈ ℝ^{N×F'}` (F' output features)

### 2. Spectral Graph Theory Foundations

#### Graph Laplacian Eigendecomposition
```
L = UΛU^T
```
where:
- `U = [u_1, u_2, ..., u_N]`: Eigenvectors (graph Fourier basis)
- `Λ = diag(λ_1, λ_2, ..., λ_N)`: Eigenvalues (0 ≤ λ_1 ≤ ... ≤ λ_N)

#### Graph Fourier Transform
**Forward Transform**:
```
x̂ = U^T x
```

**Inverse Transform**:
```
x = U x̂
```

#### Convolution in Spectral Domain
For signals `x, y` on graph:
```
(x *_G y) = U((U^T x) ⊙ (U^T y))
```

### 3. Spectral Convolutional Networks

#### Spectral CNNs (Bruna et al., 2014)
**Convolution Operation**:
```
y = σ(U g_θ U^T x)
```
where `g_θ = diag(θ_1, θ_2, ..., θ_N)` are learnable parameters.

**Issues**:
- O(N²) parameters
- Not localized in space
- Computationally expensive (eigen-decomposition)

#### ChebNet (Defferrard et al., 2016)

**Chebyshev Polynomial Approximation**:
```
g_θ(Λ) ≈ Σ_{k=0}^{K-1} θ_k T_k(Λ̃)
```

where:
- `T_k`: Chebyshev polynomial of order k
- `Λ̃ = 2Λ/λ_max - I`: Scaled eigenvalues

**Recursive Definition**:
```
T_0(x) = 1
T_1(x) = x  
T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
```

**Convolution without Eigendecomposition**:
```
y = Σ_{k=0}^{K-1} θ_k T_k(L̃) x
```

where `L̃ = 2L/λ_max - I`.

### 4. Graph Convolutional Networks (Kipf & Welling, 2017)

#### Simplification of ChebNet
**First-order approximation** (K=1):
```
g_θ(Λ) ≈ θ_0 + θ_1 Λ
```

**With λ_max ≈ 2 assumption**:
```
g_θ(L) ≈ θ_0 I + θ_1 L
```

#### Renormalization Trick
To prevent exploding/vanishing gradients:
```
θ = θ_0 = -θ_1
```

**Final GCN Layer**:
```
H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})
```

where:
- `Ã = A + I` (add self-connections)
- `D̃_ii = Σ_j Ã_ij` (degrees of Ã)

#### Multi-layer GCN
```
H^{(0)} = X
H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})
Z = softmax(H^{(L)})
```

### 5. Spatial Perspective

#### Message Passing Interpretation
GCN can be viewed as message passing:

**Message**: From neighbor j to node i
```
m_ij^{(l)} = (D̃^{-1/2} Ã D̃^{-1/2})_ij H_j^{(l)} W^{(l)}
```

**Aggregation**: Sum messages
```
h_i^{(l+1)} = σ(Σ_{j∈N(i)∪{i}} m_ij^{(l)})
```

#### Localized Filters
**K-localized filter**: Only considers K-hop neighbors
```
g_θ(L) = Σ_{k=0}^{K-1} θ_k L^k
```

**Receptive Field**: Grows with number of layers
- 1 layer: 1-hop neighbors
- 2 layers: 2-hop neighbors  
- L layers: L-hop neighbors

### 6. Variants and Extensions

#### Graph Attention Networks (GAT)
**Attention Mechanism**:
```
α_ij = exp(LeakyReLU(a^T [W h_i || W h_j]))
      / Σ_{k∈N(i)} exp(LeakyReLU(a^T [W h_i || W h_k]))
```

**Message Passing**:
```
h_i' = σ(Σ_{j∈N(i)} α_ij W h_j)
```

#### GraphSAGE
**Sampling and Aggregation**:
```
h_i^{(l+1)} = σ(W^{(l)} · CONCAT(h_i^{(l)}, AGG({h_j^{(l)}, ∀j ∈ N(i)})))
```

**Aggregators**:
- Mean: `AGG = mean({h_j, ∀j ∈ N(i)})`
- LSTM: `AGG = LSTM({h_j, ∀j ∈ N(i)})`
- Pool: `AGG = max({σ(W h_j + b), ∀j ∈ N(i)})`

#### FastGCN
**Importance Sampling**:
```
P(v) = ||h_v||₂² / Σ_u ||h_u||₂²
```

Sample nodes proportional to their feature magnitude.

### 7. Training Procedures

#### Full-Batch Training
- Use entire graph in each iteration
- Memory requirement: O(N²) for dense graphs
- Suitable for small/medium graphs

#### Mini-Batch Training
**Neighbor Sampling** (FastGCN, GraphSAINT):
- Sample subgraph for each batch
- Approximate full receptive field
- Scalable to large graphs

**Subgraph Sampling**:
```
G_batch = Sample(G, batch_nodes, K_hops)
```

#### Inductive vs Transductive

**Transductive**: Fixed graph during training
- Learn embeddings for specific nodes
- Cannot handle new nodes at test time

**Inductive**: Learn generalizable function
- Can handle unseen nodes/graphs
- Requires node features (not just structure)

### 8. Theoretical Analysis

#### Expressiveness
**Universal Approximation**: GCNs can approximate any function on graphs (with sufficient depth/width)

**Weisfeiler-Lehman Test**: GCNs are at most as powerful as 1-WL test for graph isomorphism

#### Over-smoothing Problem
**Analysis**: Deep GCNs suffer from over-smoothing
```
lim_{l→∞} H^{(l)} = constant vector
```

**Solutions**:
- Residual connections: `H^{(l+1)} = H^{(l)} + GCN(H^{(l)})`
- Dense connections: Skip connections between all layers
- Dropedge: Randomly drop edges during training

### 9. Computational Complexity

#### Forward Pass
- **Time**: O(|E| · F · F') per layer
- **Space**: O(N · F) for node features

#### Sparse Implementation
```python
# Efficient sparse matrix multiplication
AH = A @ H  # O(|E| · F)
AHW = AH @ W  # O(N · F · F')
```

#### Memory Optimization
- **Gradient Checkpointing**: Store subset of activations
- **Subgraph Sampling**: Reduce memory per batch
- **Quantization**: Lower precision computation

### 10. Applications

#### Node Classification
**Semi-supervised Learning**:
- Given: Partial node labels
- Goal: Predict labels for unlabeled nodes
- Examples: Citation networks, social networks

#### Graph Classification  
**Graph-level Prediction**:
- Pooling: Global mean/max pooling
- Hierarchical: Learnable graph coarsening
- Set2Set: Attention-based global pooling

#### Link Prediction
**Edge Score Function**:
```
score(i,j) = σ(h_i^T h_j)
```

#### Graph Generation
**Variational Graph Autoencoders**:
```
Encoder: μ, σ = GCN(A, X)
Decoder: Â = σ(Z Z^T)
```

## Implementation Details

See `exercise.py` for implementations of:
1. Spectral GCN with Chebyshev polynomials
2. Kipf-Welling GCN with efficient sparse operations
3. Multi-layer GCN with residual connections
4. Inductive learning with GraphSAGE-style sampling
5. Node and graph classification tasks
6. Training procedures and optimization
7. Over-smoothing analysis and mitigation
8. Computational efficiency optimizations