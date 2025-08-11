# Measure Theory for Machine Learning

## Prerequisites

- Real analysis foundations (limits, sequences, continuity)
- Set theory and probability basics

## Learning Objectives

- Understand σ-algebras, measures, and measurable functions
- Compute integrals on finite measure spaces and connect to expectation
- Apply basic inequalities (Markov, Chebyshev) for generalization bounds

## Mathematical Foundations

### 1. σ-Algebras and Measurable Spaces

#### Definition 1.1 (σ-Algebra)

A collection 𝔽 ⊆ 2^Ω is a σ-algebra if:

1. Ω ∈ 𝔽 and ∅ ∈ 𝔽
2. A ∈ 𝔽 ⇒ A^c ∈ 𝔽
3. If A₁, A₂, ... ∈ 𝔽 then ⋃ₙ Aₙ ∈ 𝔽

#### Intuitive Understanding: Why σ-Algebras?

- Specify which events are “observable/measurable.”
- Closed under natural operations (complement, unions), so probabilities/integrals behave consistently.

### 2. Measures and Probability Measures

#### Definition 2.1 (Measure)

μ: 𝔽 → [0, ∞] is a measure if μ(∅) = 0 and μ(⋃ disjoint Aₙ) = ∑ μ(Aₙ).

#### Intuitive Understanding: Measuring Size Beyond Length

- Generalizes length/area/volume to abstract sets; in ML, assigns “mass” over hypotheses or data.

### 3. Measurable Functions and Integration

#### Definition 3.1 (Measurable Function)

f: Ω → ℝ is measurable if {ω: f(ω) ≤ t} ∈ 𝔽 for all t ∈ ℝ.

#### Definition 3.2 (Integral on Finite Spaces)

On finite (Ω, 𝔽, μ), ∫ f dμ = ∑_{ω∈Ω} f(ω) μ({ω}). With probability measure, this is expectation.

#### Intuitive Understanding: Integration as Weighted Average

- On finite spaces, integration is just a weighted sum.
- Monotone sequences of nonnegative functions have increasing integrals (MCT intuition).

### 4. Inequalities

- Markov: P(X ≥ a) ≤ E[X]/a for X ≥ 0, a > 0
- Chebyshev: P(|X − E[X]| ≥ t) ≤ Var(X)/t²

#### Intuitive Understanding: Tail Bounds from Moments

- Large deviations are controlled by average size (Markov) or variance (Chebyshev).

## ML Connections

- Generalization and concentration bounds use measure-theoretic inequalities.
- Loss functions as measurable functions; risk as an integral over data distribution.

## Implementation Details

See `exercise.py` for:

1. σ-algebra verification and generation on finite Ω
2. Measure verification and simple integration/expectation
3. Measurability checks and inequalities (Markov/Chebyshev)

## Exercises

- Show that σ({A}) = {∅, A, A^c, Ω} on finite Ω
- Verify countable additivity reduces to finite additivity on finite partitions
- Implement integral and verify linearity and monotone convergence for simple sequences
