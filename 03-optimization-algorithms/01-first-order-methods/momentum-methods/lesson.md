# Momentum Methods for Optimization

## Prerequisites
- Stochastic gradient descent fundamentals
- Convex analysis and optimization theory
- Linear algebra (eigenvalues, quadratic forms)
- Understanding of conditioning and convergence rates

## Learning Objectives
- Master the theory behind momentum methods and acceleration
- Understand why momentum helps optimization landscape navigation
- Implement and tune momentum-based optimizers from scratch
- Connect momentum to physical intuition and mathematical acceleration theory
- Analyze convergence rates and practical performance characteristics

## Mathematical Foundations

### 1. Classical Momentum (Heavy Ball Method)

#### Motivation: Gradient Descent Limitations
Standard gradient descent: x_{k+1} = x_k - η∇f(x_k)

**Problems**:
- Slow convergence on ill-conditioned problems
- Oscillations around optimal path
- Gets stuck in narrow valleys

#### The Heavy Ball Method (Polyak 1964)
Inspired by physical systems with inertia:

```
v_{k+1} = βv_k + η∇f(x_k)
x_{k+1} = x_k - v_{k+1}
```

**Intuition**: 
- v_k represents "velocity" 
- β ∈ [0,1) controls momentum strength
- Accumulates gradients to maintain direction

#### Equivalent Formulation
x_{k+1} = x_k - η∇f(x_k) + β(x_k - x_{k-1})

The term β(x_k - x_{k-1}) is the "momentum" from previous step.

### 2. Nesterov Accelerated Gradient (NAG)

#### The Lookahead Trick
Key insight: Evaluate gradient at the "lookahead" point.

```
y_k = x_k + β(x_k - x_{k-1})    # Lookahead position
x_{k+1} = y_k - η∇f(y_k)        # Gradient step from lookahead
```

#### Equivalent Velocity Form
```
v_{k+1} = βv_k + η∇f(x_k + βv_k)
x_{k+1} = x_k - v_{k+1}
```

#### Why This Works
- Momentum might overshoot
- Looking ahead allows correction before committing
- More stable than classical momentum

### 3. Theoretical Analysis

#### Quadratic Functions
For f(x) = (1/2)x^T A x - b^T x where A ≻ 0:

**Classical Momentum Convergence**:
Choose β = ((√κ - 1)/(√κ + 1))^2 where κ = λ_max/λ_min

Convergence rate: O((√κ - 1)/(√κ + 1))^k vs O((κ-1)/(κ+1))^k for GD

**Nesterov Acceleration**:
Optimal rate: O(1/k^2) for convex functions
O(exp(-k/√κ)) for strongly convex functions

#### Theorem 3.1 (Nesterov Acceleration Rate)
For L-smooth, μ-strongly convex f, NAG with proper parameters achieves:

E[f(x_k) - f*] ≤ ((√L - √μ)/(√L + √μ))^k · (f(x_0) - f*)

This is optimal for first-order methods.

**Proof Sketch**:
1. Define Lyapunov function combining function value and momentum
2. Show this decreases geometrically
3. Use estimate sequence technique

### 4. Momentum in Stochastic Settings

#### SGD with Momentum (SGDM)
```
v_{k+1} = βv_k + η∇F(x_k, ξ_k)
x_{k+1} = x_k - v_{k+1}
```

where ∇F(x_k, ξ_k) is stochastic gradient.

#### Variance Reduction Property
Momentum reduces variance of gradient updates:
- Smooths out noisy gradients
- Provides implicit averaging
- More stable convergence paths

#### Convergence Analysis
Under standard assumptions (bounded variance σ^2, smoothness L):

E[||∇f(x_k)||^2] ≤ O(1/(k+1)) + O(η σ^2/(1-β))

Trade-off: Higher β reduces gradient noise but may slow adaptation.

### 5. Adaptive Momentum Methods

#### Adam-style Momentum
First moment estimation:
m_k = β_1 m_{k-1} + (1-β_1)g_k

With bias correction:
m̂_k = m_k/(1 - β_1^k)

#### Momentum vs. Adam's First Moment
- Classical momentum: Exponential moving average of gradients
- Adam momentum: Bias-corrected exponential moving average
- Similar effect but different normalization

### 6. Advanced Momentum Variants

#### AdaBelief Momentum Component
Combines momentum with adaptive learning rates based on gradient prediction error.

#### Lookahead Optimizer
k inner updates with momentum, then interpolate:
φ_{t+1} = φ_t + α(θ_{t+1,k} - φ_t)

#### Quasi-Hyperbolic Momentum (QHM)
ν_t = β_1 ν_{t-1} + g_t
θ_{t+1} = θ_t - α[(1-β_1)g_t + β_1 ν_t]

Weighted average of current gradient and momentum.

### 7. Practical Considerations

#### Hyperparameter Selection
**β (momentum coefficient)**:
- Start with β = 0.9
- Increase to 0.95-0.99 for well-conditioned problems
- Lower (0.5-0.8) for noisy/ill-conditioned problems

**Learning Rate η**:
- Often can use larger learning rates with momentum
- Typical range: 0.001 to 0.1
- May need warmup period

#### Initialization
- Initialize velocity v_0 = 0
- Some methods benefit from warm-up
- Can inherit momentum from previous training

#### When Momentum Helps
✅ **Good for**:
- High-dimensional optimization
- Ill-conditioned problems
- When gradients are consistent in direction
- Escaping shallow local minima

❌ **Problematic for**:
- Non-stationary objectives
- Very sparse gradients
- When frequent direction changes needed

### 8. Connection to Physical Systems

#### Differential Equation View
Gradient descent: ẋ = -∇f(x)
Heavy ball: ẍ + γẋ + ∇f(x) = 0

This is a damped oscillator in the potential f(x).

#### Energy Perspective
Total energy: E = (1/2)||ẋ||^2 + f(x)
- Kinetic energy: (1/2)||ẋ||^2  
- Potential energy: f(x)
- Damping reduces kinetic energy

#### Nesterov as Optimal Control
NAG can be derived from optimal control theory:
Minimize ∫_0^T ||u(t)||^2 dt subject to ẍ = u, x(T) at minimum

## Implementation Details

See `exercise.py` for implementations of:
1. Classical momentum (Heavy Ball)
2. Nesterov accelerated gradient
3. SGD with momentum
4. Adaptive momentum variants
5. Hyperparameter tuning utilities
6. Convergence analysis tools
7. Visualization of momentum effects

## Experiments

1. **Conditioning Study**: Compare momentum vs vanilla SGD on ill-conditioned quadratics
2. **Momentum Coefficient**: Effect of β on convergence speed and stability  
3. **Stochastic vs Deterministic**: Momentum benefits in noisy vs clean settings
4. **Learning Rate Interaction**: How momentum affects optimal learning rate choice
5. **Landscape Visualization**: 2D optimization with momentum trajectory plotting

## Research Connections

### Seminal Papers
1. **Polyak (1964)** - "Some Methods of Speeding up Convergence of Iteration Methods"
   - First rigorous analysis of momentum methods
   
2. **Nesterov (1983)** - "A Method for Unconstrained Convex Minimization Problem with the Rate of Convergence O(1/k²)"
   - Introduced optimal acceleration for convex optimization
   
3. **Sutskever et al. (2013)** - "On the Importance of Initialization and Momentum in Deep Learning"
   - Showed momentum's crucial role in deep learning

### Modern Developments
4. **Kidambi et al. (2018)** - "On the Insufficiency of Existing Momentum Schemes for Stochastic Optimization"
   - Analysis of momentum in stochastic settings

5. **Gadat & Panloup (2017)** - "Long Time Behavior of a Stochastic Momentum Method"
   - Continuous-time analysis of stochastic momentum

6. **Wilson et al. (2017)** - "The Marginal Value of Adaptive Gradient Methods"
   - Comparison of momentum vs adaptive methods

## Resources

### Primary Sources
1. **Nesterov - Introductory Lectures on Convex Optimization** (Chapter 2)
   - Definitive treatment of acceleration theory
2. **Polyak - Introduction to Optimization** 
   - Original momentum method development
3. **Bubeck - Convex Optimization: Algorithms and Complexity**
   - Modern perspective on acceleration

### Video Resources
1. **Sebastien Bubeck - Convex Optimization Lectures** (MIT)
   - Excellent coverage of acceleration theory
2. **Ben Recht - Optimization for Machine Learning** (Berkeley)
   - Practical perspective on momentum methods
3. **Francis Bach - Optimization and Learning** (ENS)
   - Mathematical foundations

### Advanced Reading
1. **Su, Boyd & Candès (2016)** - "A Differential Equation for Modeling Nesterov's Accelerated Gradient Method"
2. **Flammarion & Bach (2015)** - "From Averaging to Acceleration"
3. **Allen-Zhu (2016)** - "Katyusha: Accelerated Variance Reduction"

## Socratic Questions

### Understanding
1. Why does classical momentum sometimes overshoot while Nesterov is more stable?
2. How does the momentum coefficient β relate to the condition number of the problem?
3. When can momentum actually hurt convergence?

### Extension  
1. Can you derive the optimal momentum coefficient for quadratic functions?
2. How would you modify momentum for constrained optimization problems?
3. What happens to momentum methods in infinite-dimensional spaces?

### Research
1. Is there a fundamental limit to acceleration for non-convex optimization?
2. How do momentum methods behave near saddle points vs local minima?
3. Can we design adaptive momentum that adjusts β during training?

## Exercises

### Theoretical
1. Derive the convergence rate of classical momentum for quadratic functions
2. Prove that Nesterov acceleration achieves O(1/k²) rate for convex functions
3. Analyze the effect of momentum on gradient noise in stochastic settings

### Implementation
1. Implement all momentum variants from scratch with proper vectorization
2. Build visualization tools for momentum trajectories on 2D functions
3. Create automatic hyperparameter tuning for momentum methods

### Research
1. Empirically study momentum's effect on different neural network architectures
2. Investigate momentum in federated learning settings
3. Compare momentum methods on non-convex optimization landscapes