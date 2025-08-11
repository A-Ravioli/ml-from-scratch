# Martingales for Machine Learning

## Prerequisites

- Measure-theoretic probability, filtrations, conditional expectation

## Learning Objectives

- Understand martingales, super/submartingales, and stopping times
- Apply optional stopping and Azuma–Hoeffding inequality
- Use martingale tools in online learning and bandits

## Core Concepts

### 1. Filtrations and Martingales

Adapted processes (X_t) with E[|X_t|] < ∞ and E[X_{t+1} | 𝔽_t] = X_t.

#### Intuition

- “Fair game”: best prediction at next step is current value.

### 2. Optional Stopping

Under mild conditions, E[X_τ] = E[X_0] for a stopping time τ.

### 3. Azuma–Hoeffding Inequality

For martingales with bounded increments: P(X_n − X_0 ≥ t) ≤ exp(−2t² / ∑c_i²).

## ML Connections

- Regret bounds in online learning; concentration for dependent sequences.

## Implementation Details

See `exercise.py` for:

1. `is_martingale` via conditional expectation checks (empirical)
2. `optional_stopping_demo`
3. `azuma_hoeffding_bound` and empirical verification


