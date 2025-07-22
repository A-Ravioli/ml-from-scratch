# Efficient Transformers: Scaling Beyond Quadratic Complexity

## Prerequisites
- Standard Transformer architecture
- Computational complexity analysis
- Sparse attention mechanisms
- Linear algebra (low-rank approximations)

## Learning Objectives
- Master various approaches to efficient Transformer computation
- Understand linear attention mechanisms and approximations
- Implement memory-efficient architectures
- Analyze trade-offs between efficiency and performance
- Connect to practical deployment constraints

## Mathematical Foundations

### 1. Efficiency Approaches

#### Four Main Categories
1. **Sparse Attention**: Reduce attention to O(n√n) or O(n log n)
2. **Linear Attention**: Approximate attention with O(n) complexity
3. **Low-Rank Approximation**: Factorize attention matrix
4. **Hardware-Aware**: Optimize for specific hardware constraints

### 2. Linear Attention Methods

#### Performer (FAVOR+)
Approximate softmax attention with random features:
```
K(x,y) = E[φ(x)ᵀφ(y)] where φ(x) = exp(ωᵀx)
Attention ≈ (Qφ)(φᵀK)V / ((Qφ)(φᵀ1))
```

#### Linear Transformer
Remove softmax entirely:
```
Attention(Q,K,V) = Q(KᵀV) / Q(Kᵀ1)
```

#### FNet
Replace attention with Fourier transforms:
```
FNet(x) = FFT(x) ⊙ learnable_weights
```

### 3. Low-Rank Methods

#### Linformer
Project keys and values to lower dimensions:
```
K' = KE, V' = VF where E,F ∈ ℝⁿˣᵏ, k≪n
Attention(Q,K',V') has O(nk) complexity
```

#### Nyströmformer
Use Nyström approximation of attention matrix:
```
A ≈ A₁A₂⁻¹A₃ where A₁,A₂,A₃ are submatrices
```

### 4. Memory-Efficient Training

#### Gradient Checkpointing
- Trade computation for memory
- Recompute activations during backward pass
- Essential for training large models

#### Mixed Precision Training
- Use FP16 for forward/backward pass
- Use FP32 for parameter updates
- 2x memory reduction, 1.5-2x speedup

#### ZeRO Optimizer States
- Partition optimizer states across GPUs
- Reduce memory per GPU
- Enable training larger models

### 5. Hardware-Aware Optimizations

#### FlashAttention
- IO-aware algorithm
- Tile attention computation
- Reduce HBM memory accesses
- 2-4x speedup on modern GPUs

#### Memory Hierarchies
- **Registers**: ~1KB, 1 cycle
- **L1 Cache**: ~32KB, 1-3 cycles  
- **L2 Cache**: ~256KB, 10-20 cycles
- **HBM**: ~80GB, 200-800 cycles

### 6. Model Architecture Efficiency

#### MobileBERT
- Bottleneck structure
- Knowledge distillation
- 4x smaller, 5.5x faster than BERT

#### DistilBERT
- 6 layers instead of 12
- Knowledge distillation during pre-training
- 60% size, 60% faster, 97% performance

#### ALBERT
- Parameter sharing across layers
- Factorized embeddings
- 18x fewer parameters than BERT-Large

## Implementation Details

See `exercise.py` for implementations of:
1. Linear attention mechanisms
2. Sparse attention patterns
3. Memory-efficient training techniques
4. Hardware-aware optimizations
5. Model compression methods