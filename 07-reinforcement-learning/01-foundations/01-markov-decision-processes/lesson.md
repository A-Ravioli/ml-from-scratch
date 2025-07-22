# Markov Decision Processes

## Prerequisites
- Probability theory (conditional probability, expectations, stochastic processes)
- Linear algebra (matrix operations, eigenvalues, linear systems)
- Real analysis (sequences, convergence, fixed points)
- Basic optimization theory (dynamic programming principles)

## Learning Objectives
- Master the mathematical formalism of Markov Decision Processes
- Understand the Markov property and its implications for sequential decision making
- Analyze finite and infinite horizon problems with different optimality criteria
- Implement MDP algorithms for policy evaluation and optimization
- Connect MDPs to real-world sequential decision problems

## Mathematical Foundations

### 1. Markov Decision Process Definition

#### Definition 1.1 (Markov Decision Process)
A Markov Decision Process is a 5-tuple (S, A, P, R, γ) where:
- **S**: State space (finite or countably infinite)
- **A**: Action space (finite or countably infinite)  
- **P**: Transition probability function P(s'|s,a) = ℙ[S_{t+1} = s' | S_t = s, A_t = a]
- **R**: Reward function R(s,a,s') or R(s,a) or R(s)
- **γ**: Discount factor γ ∈ [0,1]

#### The Markov Property
**Definition 1.2**: A stochastic process {X_t} satisfies the Markov property if:
ℙ[X_{t+1} = x' | X_t = x, X_{t-1} = x_{t-1}, ..., X_0 = x_0] = ℙ[X_{t+1} = x' | X_t = x]

**Interpretation**: The future depends only on the present state, not the history.

#### State Space Considerations
- **Finite**: |S| < ∞ (tabular methods applicable)
- **Countably infinite**: S = ℕ (function approximation needed)
- **Continuous**: S ⊆ ℝⁿ (requires special treatment)

### 2. Policies and Value Functions

#### Definition 2.1 (Policy)
A **policy** π is a mapping from states to action distributions:
- **Deterministic**: π(s) ∈ A
- **Stochastic**: π(a|s) = ℙ[A_t = a | S_t = s]

#### Definition 2.2 (State Value Function)
The value of state s under policy π:
V^π(s) = 𝔼^π[G_t | S_t = s]

where G_t is the return from time t.

#### Definition 2.3 (Action Value Function)  
The value of taking action a in state s under policy π:
Q^π(s,a) = 𝔼^π[G_t | S_t = s, A_t = a]

#### Relationship Between Value Functions
V^π(s) = ∑_a π(a|s) Q^π(s,a)
Q^π(s,a) = ∑_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

### 3. Return Formulations

#### Episodic Tasks
For episodes of length T:
G_t = R_{t+1} + R_{t+2} + ... + R_T

#### Continuing Tasks with Discounting
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = ∑_{k=0}^∞ γᵏR_{t+k+1}

#### Average Reward Formulation
For infinite horizon without discounting:
ρ^π = lim_{n→∞} (1/n) 𝔼^π[∑_{t=0}^{n-1} R_t]

#### Properties of Discounting
**Theorem 3.1**: If rewards are bounded by R_max, then discounted return is bounded:
|G_t| ≤ R_max/(1-γ)

**Proof**: |G_t| ≤ ∑_{k=0}^∞ γᵏR_max = R_max ∑_{k=0}^∞ γᵏ = R_max/(1-γ) □

### 4. Bellman Equations

#### Bellman Expectation Equations
**For V^π**:
V^π(s) = ∑_a π(a|s) ∑_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

**For Q^π**:
Q^π(s,a) = ∑_{s'} P(s'|s,a)[R(s,a,s') + γ ∑_{a'} π(a'|s')Q^π(s',a')]

#### Matrix Form (Finite State Space)
V^π = R^π + γP^π V^π

Where:
- V^π ∈ ℝ^|S|: Vector of state values
- R^π ∈ ℝ^|S|: Expected immediate rewards under π
- P^π ∈ ℝ^|S|×|S|: Transition matrix under π

#### Theorem 4.1 (Existence and Uniqueness)
For finite state space and γ < 1, the Bellman expectation equation has a unique solution:
V^π = (I - γP^π)^{-1}R^π

**Proof**: (I - γP^π) is invertible since ρ(γP^π) = γρ(P^π) ≤ γ < 1 □

### 5. Optimal Policies and Value Functions

#### Definition 5.1 (Optimal Value Functions)
**Optimal state value function**:
V*(s) = max_π V^π(s) for all s ∈ S

**Optimal action value function**:
Q*(s,a) = max_π Q^π(s,a) for all s ∈ S, a ∈ A

#### Bellman Optimality Equations
**For V***:
V*(s) = max_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]

**For Q***:
Q*(s,a) = ∑_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]

#### Theorem 5.1 (Existence of Optimal Policy)
For finite MDPs, there exists at least one deterministic optimal policy π* such that:
V^{π*}(s) = V*(s) for all s ∈ S

#### Greedy Policy Extraction
Given optimal value function V* or Q*:
π*(s) = argmax_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
π*(s) = argmax_a Q*(s,a)

### 6. Policy Partial Ordering

#### Definition 6.1 (Policy Ordering)
Policy π ≥ π' if V^π(s) ≥ V^{π'}(s) for all s ∈ S.

#### Theorem 6.1 (Policy Improvement)
Let π' be the greedy policy with respect to V^π:
π'(s) = argmax_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

Then π' ≥ π. If there exists s such that the inequality is strict in the Bellman equation, then π' > π.

**Proof**: 
Q^π(s,π'(s)) ≥ V^π(s) by definition of π'
But Q^π(s,π'(s)) ≤ V^{π'}(s) by definition of value function
Therefore V^{π'}(s) ≥ V^π(s) □

### 7. Finite Horizon Problems

#### Dynamic Programming Recursion
For finite horizon T:
V_T(s) = 0 (terminal condition)
V_t(s) = max_a ∑_{s'} P(s'|s,a)[R(s,a,s') + V_{t+1}(s')]

#### Optimal Policy
π*_t(s) = argmax_a ∑_{s'} P(s'|s,a)[R(s,a,s') + V_{t+1}(s')]

**Note**: Optimal policy is time-dependent in finite horizon case.

### 8. Partially Observable MDPs (POMDPs)

#### POMDP Definition
A POMDP extends MDP with observations: (S, A, P, R, Ω, O, γ) where:
- **Ω**: Observation space
- **O**: Observation function O(o|s,a) = ℙ[O_t = o | S_t = s, A_{t-1} = a]

#### Belief States
Since agent doesn't observe state directly, maintain belief:
b_t(s) = ℙ[S_t = s | O_0, A_0, O_1, A_1, ..., O_t]

#### POMDP Policy
π(a|b) where b is belief state distribution over S.

#### Complexity
**Theorem 8.1**: Computing optimal POMDP policy is PSPACE-complete.

### 9. Continuous State and Action Spaces

#### Function Approximation Necessity
For continuous spaces, exact tabular representation impossible.
Must use function approximation: V(s) ≈ V̂(s; θ)

#### Hamilton-Jacobi-Bellman Equation
For continuous time:
ρ + H(s, ∇V(s)) = 0

where H is the Hamiltonian and ρ is the value.

#### Linear Quadratic Regulator (LQR)
**Dynamics**: s_{t+1} = As_t + Ba_t + w_t
**Cost**: c(s,a) = s^T Q s + a^T R a
**Optimal policy**: π*(s) = -R^{-1}B^T P s (linear in state)

### 10. Multi-Objective MDPs

#### Vector-Valued Rewards
R: S × A × S → ℝ^k (k objectives)

#### Pareto Optimal Policies
Policy π dominates π' if V^π_i(s) ≥ V^{π'}_i(s) for all i, s with strict inequality for some i, s.

#### Scalarization Approach
Weighted sum: r(s,a,s') = w^T R(s,a,s') where w ∈ ℝ^k_+

### 11. Computational Complexity

#### State Space Explosion
Number of states often grows exponentially with problem size.
Example: n binary variables → 2^n states

#### Curse of Dimensionality
For continuous spaces, discretization requires exponential number of grid points.

#### Approximation Algorithms
- **Function approximation**: V(s) ≈ φ(s)^T θ
- **Hierarchical decomposition**: Temporal abstractions
- **State aggregation**: Group similar states

## Implementation Details

See `exercise.py` for implementations of:
1. Basic MDP class with finite state/action spaces
2. Policy representation (deterministic and stochastic)
3. Value function computation for given policies
4. Bellman equation solvers
5. Policy evaluation algorithms
6. Optimal policy extraction from value functions
7. Finite horizon dynamic programming
8. Simple POMDP belief updating
9. Visualization tools for small MDPs

## Experiments

1. **Grid World**: Implement classic grid world with obstacles and goals
2. **Gambler's Problem**: Finite MDP with optimal policy computation
3. **Car Rental**: Multi-location inventory management MDP
4. **Finite Horizon**: Compare finite vs infinite horizon solutions
5. **Discount Factor**: Analyze effect of γ on optimal policies
6. **POMDP**: Simple tiger problem with belief state tracking

## Research Connections

### Foundational Papers
1. **Bellman (1957)** - "Dynamic Programming"
   - Original formulation of dynamic programming and optimality principle

2. **Howard (1960)** - "Dynamic Programming and Markov Processes"
   - First comprehensive treatment of MDPs

3. **Puterman (1994)** - "Markov Decision Processes: Discrete Stochastic Dynamic Programming"
   - Definitive reference for MDP theory

### Modern Developments
4. **Sutton & Barto (2018)** - "Reinforcement Learning: An Introduction"
   - Modern perspective connecting MDPs to RL

5. **Bertsekas (2012)** - "Dynamic Programming and Optimal Control"
   - Advanced treatment with approximation methods

6. **Kaelbling, Littman & Cassandra (1998)** - "Planning and Acting in Partially Observable Stochastic Domains"
   - POMDP survey and algorithms

### Theoretical Advances
7. **Tsitsiklis & Van Roy (1997)** - "An Analysis of Temporal-Difference Learning"
   - Convergence analysis for approximate dynamic programming

8. **Munos (2003)** - "Error Bounds for Approximate Policy Iteration"
   - Sample complexity and approximation error analysis

## Resources

### Primary Sources
1. **Puterman - Markov Decision Processes**
   - Comprehensive mathematical treatment
2. **Bertsekas - Dynamic Programming and Optimal Control**
   - Advanced algorithms and approximation methods
3. **Sutton & Barto - Reinforcement Learning: An Introduction**
   - Accessible introduction with practical focus

### Video Resources
1. **David Silver - RL Course (DeepMind)**
   - Excellent MDP foundations lecture
2. **Emma Brunskill - CS234 (Stanford)**
   - Rigorous treatment of MDP theory
3. **Sergey Levine - CS285 (Berkeley)**
   - Modern perspective connecting to deep RL

### Advanced Reading
1. **Szepesvári (2010)** - "Algorithms for Reinforcement Learning"
2. **Meyn (2007)** - "Control Techniques for Complex Networks"
3. **Hernández-Lerma & Lasserre (1996)** - "Discrete-Time Markov Control Processes"

## Socratic Questions

### Understanding
1. Why is the Markov property crucial for tractable sequential decision making?
2. How does the discount factor γ affect the relative importance of immediate vs future rewards?
3. What is the relationship between finite horizon and infinite horizon optimal policies?

### Extension
1. How would you modify the MDP framework for risk-sensitive decision making?
2. Can you design an MDP where the optimal policy is randomized?
3. How do continuous state spaces change the computational complexity of finding optimal policies?

### Research
1. What are the fundamental limits of learning in partially observable environments?
2. How can we design MDPs that better model real-world sequential decision problems?
3. What role does the reward function design play in achieving desired behaviors?

## Exercises

### Theoretical
1. Prove that the Bellman optimality equation has a unique solution for finite MDPs
2. Show that any optimal policy for an MDP must be greedy with respect to the optimal value function
3. Derive the finite horizon dynamic programming recursion from first principles

### Implementation
1. Implement a general MDP class that can handle arbitrary finite state and action spaces
2. Build visualization tools for small MDPs showing state transitions and optimal policies
3. Create solvers for both finite and infinite horizon problems
4. Implement belief state updating for simple POMDPs

### Research
1. Study the effect of discount factor on policy convergence and stability
2. Compare different reward function designs for the same underlying task
3. Investigate approximation methods for large state space MDPs