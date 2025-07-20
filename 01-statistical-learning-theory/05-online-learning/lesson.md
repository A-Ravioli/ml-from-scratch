# Online Learning and Regret Minimization

## Prerequisites
- Convex optimization (gradients, projections)
- Probability theory (concentration inequalities)
- Game theory basics

## Learning Objectives
- Master online learning framework and regret bounds
- Understand online-to-batch conversion
- Implement efficient online algorithms
- Connect to adversarial machine learning

## Mathematical Foundations

### 1. Online Learning Framework

#### Protocol
For T rounds:
1. Learner chooses action xₜ ∈ 𝒳
2. Environment reveals loss function ℓₜ: 𝒳 → ℝ
3. Learner suffers loss ℓₜ(xₜ)

#### Adversarial Setting
- Environment can be adversarial
- No distributional assumptions
- Worst-case analysis

#### Examples
- **Online classification**: Predict labels, observe true labels
- **Online convex optimization**: Choose point, observe convex function
- **Bandit problems**: Choose arm, observe reward for chosen arm only

### 2. Regret Measures

#### Definition 2.1 (Regret)
Regret with respect to comparator u ∈ 𝒳:
R_T(u) = ∑ₜ₌₁ᵀ ℓₜ(xₜ) - ∑ₜ₌₁ᵀ ℓₜ(u)

#### Definition 2.2 (Worst-case Regret)
R_T = max_u∈𝒳 R_T(u) = ∑ₜ₌₁ᵀ ℓₜ(xₜ) - min_u∈𝒳 ∑ₜ₌₁ᵀ ℓₜ(u)

#### Regret Minimization
Algorithm is *no-regret* if R_T/T → 0 as T → ∞.

### 3. Online Gradient Descent

#### Algorithm
For convex constraint set 𝒳:
1. Choose x₁ ∈ 𝒳 arbitrarily
2. For t = 1, ..., T:
   - Play xₜ, observe ℓₜ
   - Update: xₜ₊₁ = Π_𝒳(xₜ - ηₜ∇ℓₜ(xₜ))

where Π_𝒳 is projection onto 𝒳.

#### Theorem 3.1 (OGD Regret Bound)
For convex losses with ||∇ℓₜ(x)|| ≤ G and diam(𝒳) ≤ D:
R_T ≤ (D²/2η) + (ηGT)/2

**Optimal choice**: η = D/(G√T) gives R_T ≤ DG√T.

**Proof Sketch**:
1. Use projection property: ||xₜ₊₁ - u||² ≤ ||xₜ - ηₜ∇ℓₜ(xₜ) - u||²
2. Expand and use convexity: ℓₜ(xₜ) - ℓₜ(u) ≤ ∇ℓₜ(xₜ)ᵀ(xₜ - u)
3. Telescope the bound □

### 4. Follow the Regularized Leader

#### Algorithm (FTRL)
Choose xₜ₊₁ = argmin_{x∈𝒳} [∑ₛ₌₁ᵗ ℓₛ(x) + R(x)]

where R(x) is a regularization function.

#### Examples
- **L2 regularization**: R(x) = (1/2η)||x||²
- **Entropy regularization**: R(x) = η∑ᵢ xᵢ log xᵢ (for simplex)

#### Theorem 4.1 (FTRL Regret Bound)
For α-strongly convex regularizer R:
R_T ≤ (R(u) - R(x₁))/α + ∑ₜ₌₁ᵀ⁻¹ (∇ℓₜ(xₜ₊₁) - ∇ℓₜ(xₜ))ᵀ(xₜ₊₁ - xₜ)

### 5. Multiplicative Weights / Hedge

#### Setting
Actions are probability distributions over n experts.
Loss is linear: ℓₜ(p) = pᵀlₜ where lₜ ∈ [0,1]ⁿ.

#### Algorithm
Initialize w₁ᵢ = 1 for all i.
For t = 1, ..., T:
1. Play pₜᵢ = wₜᵢ/∑ⱼ wₜⱼ
2. Observe losses lₜ
3. Update: wₜ₊₁,ᵢ = wₜᵢ exp(-ηlₜᵢ)

#### Theorem 5.1 (Hedge Regret Bound)
R_T ≤ (log n)/η + ηT/8

**Optimal choice**: η = √(8 log n/T) gives R_T ≤ √(T log n/2).

### 6. Online-to-Batch Conversion

#### Theorem 6.1 (Online-to-Batch)
If online algorithm has regret R_T, then for iid data:
E[f(x̄) - min_u f(u)] ≤ R_T/T

where x̄ = (1/T)∑ₜ xₜ and f(x) = E[ℓ(x, z)].

This connects online regret to generalization in stochastic setting.

### 7. Bandit Learning

#### Multi-Armed Bandits
- K arms with unknown reward distributions
- At each time, pull one arm and observe reward
- Goal: Minimize cumulative regret

#### UCB Algorithm
Choose arm with highest upper confidence bound:
aₜ = argmax_i [μ̂ᵢ + √(2 log t/nᵢ)]

where μ̂ᵢ is empirical mean and nᵢ is number of pulls.

#### Theorem 7.1 (UCB Regret)
R_T ≤ 8∑ᵢ₌₂ᴷ (log T)/Δᵢ + (1 + π²/3)∑ᵢ₌₂ᴷ Δᵢ

where Δᵢ is gap between optimal and i-th arm.

### 8. Contextual Bandits

#### Setting
- Context xₜ ∈ 𝒳 revealed at time t
- Choose action aₜ ∈ 𝒜
- Observe reward r(xₜ, aₜ) + noise

#### LinUCB Algorithm
Assume linear rewards: E[r(x,a)] = xᵀθₐ
Maintain confidence sets for θₐ and choose optimistically.

#### Thompson Sampling
Maintain posterior over parameters, sample, and act according to sample.

### 9. Adversarial Examples Connection

#### Adversarial Training as Online Learning
- Learner chooses model parameters
- Adversary chooses perturbations
- Minimax formulation leads to online algorithms

#### Robust Optimization
min_θ max_{||δ||≤ε} E[ℓ(f_θ(x + δ), y)]

Can be solved using online learning techniques.

### 10. Advanced Topics

#### Adaptive Regret
Regret over any interval [s,t]:
R_{s,t} = ∑_{u=s}^t ℓᵤ(xᵤ) - min_x ∑_{u=s}^t ℓᵤ(x)

#### Strongly Convex Losses
For μ-strongly convex losses:
R_T = O(log T)

Much better than general convex case.

#### Non-convex Online Learning
Recent work on online learning for non-convex problems:
- Stationary point guarantees
- Local regret bounds
- Connections to SGD

### 11. Practical Applications

#### Online Advertising
- Ad placement decisions
- Revenue optimization
- Real-time bidding

#### Recommendation Systems
- Online collaborative filtering
- Exploration vs exploitation
- Cold start problems

#### Portfolio Management
- Online portfolio selection
- Risk management
- Transaction costs

## Implementation Details

See `exercise.py` for implementations of:
1. Online gradient descent variants
2. Follow-the-regularized-leader
3. Multiplicative weights algorithm
4. UCB and Thompson sampling for bandits
5. Contextual bandit algorithms
6. Regret analysis tools

## Experiments

1. **Regret Comparison**: Different algorithms on same sequence
2. **Adversarial vs Stochastic**: Performance in different settings
3. **Parameter Sensitivity**: Effect of learning rates
4. **Online-to-Batch**: Verify theoretical predictions

## Research Connections

### Foundational Papers
1. Freund & Schapire (1997) - "A Decision-Theoretic Generalization of On-Line Learning"
2. Zinkevich (2003) - "Online Convex Programming and Generalized Infinitesimal Gradient Ascent"
3. Hazan et al. (2007) - "Logarithmic Regret Algorithms for Online Convex Optimization"

### Modern Developments
1. McMahan & Streeter (2010) - "Adaptive Bound Optimization for Online Convex Optimization"
2. Rakhlin et al. (2012) - "Online Learning: Random Averages, Combinatorial Parameters"
3. Foster et al. (2018) - "Contextual Bandits with Surrogate Losses"

## Resources

### Primary Sources
1. **Shalev-Shwartz - Online Learning and Online Convex Optimization**
   - Comprehensive treatment
2. **Cesa-Bianchi & Lugosi - Prediction, Learning, and Games**
   - Game-theoretic perspective
3. **Hazan - Introduction to Online Convex Optimization**
   - Modern algorithmic approach

### Advanced Reading
1. **Lattimore & Szepesvári - Bandit Algorithms**
   - Comprehensive bandit book
2. **Rakhlin et al. - Statistical Learning Theory**
   - Connections to statistical learning
3. **Bubeck & Cesa-Bianchi - Regret Analysis**
   - Survey of regret bounds

## Socratic Questions

### Understanding
1. Why is the adversarial setting more challenging than stochastic?
2. How does regret relate to generalization?
3. What's the role of regularization in online learning?

### Extension
1. Can you design online algorithms for structured prediction?
2. How do we handle non-convex online problems?
3. What happens with delayed or partial feedback?

### Research
1. How can we get adaptive regret bounds?
2. What's the connection between online learning and game theory?
3. How do we design practical online algorithms for deep learning?

## Exercises

### Theoretical
1. Prove the regret bound for online gradient descent
2. Derive the multiplicative weights update rule
3. Show the online-to-batch conversion theorem

### Implementation
1. Implement online learning algorithms from scratch
2. Build bandit simulation environment
3. Create regret visualization tools

### Research
1. Study online learning for neural networks
2. Investigate adaptive learning rate schemes
3. Explore applications to reinforcement learning