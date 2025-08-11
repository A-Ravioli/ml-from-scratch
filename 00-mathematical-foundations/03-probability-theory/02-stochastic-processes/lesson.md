# Stochastic Processes for Machine Learning

## Prerequisites

- Measure-theoretic probability basics
- Random variables and expectations

## Learning Objectives

- Understand discrete-time Markov chains, stationary distributions, and mixing
- Simulate Poisson processes and continuous-time chains
- Connect ergodicity and mixing to generalization and MCMC

## Core Concepts

### 1. Discrete-Time Markov Chains (DTMC)

State space S, transition matrix P where P_{ij} = P(X_{t+1}=j | X_t=i).

#### Intuition: Memoryless Dynamics

- Next step depends only on current state.
- Stationary π satisfies πᵀ = πᵀ P; mixing means P^t → 1πᵀ.

#### ML Connection

- MCMC samplers (Gibbs/Metropolis) are Markov chains; mixing controls sample quality.

### 2. Continuous-Time Markov Chains (CTMC)

Generator Q with off-diagonals q_{ij} ≥ 0, rows summing to 0; P(t) = e^{tQ}.

#### Intuition: Exponential Holding Times

- Wait exponential time in state i, then jump according to rates.

#### ML Connection

- Birth–death processes model queues; used in stochastic modeling of systems.

### 3. Poisson Process

Independent increments; N(t) ~ Poisson(λt).

#### Intuition

- “Clock” of random arrivals with constant rate λ.

#### ML Connection

- Event modeling, renewal processes, thinning/superposition for simulation.

## Implementation Details

See `exercise.py` for:

1. `simulate_markov_chain`, `stationary_distribution`, `mixing_time_upper_bound`
2. `simulate_ctmc` via uniformization
3. `simulate_poisson_process` and thinning

## Exercises

- Verify detailed balance for reversible chains
- Estimate mixing time empirically for simple chains
- Simulate CTMCs via uniformization and compare with expm


