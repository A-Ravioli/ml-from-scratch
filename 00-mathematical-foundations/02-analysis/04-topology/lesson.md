# Topology for Machine Learning

## Prerequisites

- Metric spaces and basic analysis
- Set theory basics

## Learning Objectives

- Understand topological spaces, open/closed sets, and continuity
- Work with closure, interior, boundary on finite spaces
- Recognize compactness and connectedness in simple settings

## Mathematical Foundations

### 1. Topological Spaces

#### Definition 1.1 (Topology)

A topology τ on a set X is a collection of subsets (open sets) such that:

1. ∅, X ∈ τ
2. Any union of open sets is open
3. Finite intersections of open sets are open

#### Intuitive Understanding: Open Sets as “Wiggle Rooms”

- Being in an open set means you can “wiggle a bit” and still stay inside.
- Topology abstracts continuity without distances.

### 2. Continuity

f: X → Y is continuous if preimage of every open set in Y is open in X.

#### Intuitive Understanding: Preimage-Preserving Openness

- Continuity means open-ness is respected by pulling back sets through f.

### 3. Compactness and Connectedness

- Compact: every open cover has a finite subcover.
- Connected: the space cannot be split into two disjoint nonempty open sets.

#### Intuitive Understanding

- Compactness: no infinite “escape”; you can cover with finitely many patches.
- Connectedness: no nontrivial “tear” into separated open regions.

## ML Connections

- Compactness underpins existence of minimizers for continuous losses.
- Topological continuity generalizes robustness of models under perturbations.

## Implementation Details

See `exercise.py` for:

1. Verifying topological axioms on finite sets
2. Closure, interior, boundary computations
3. Continuity checks via preimages
4. Compactness and connectedness checks on finite spaces

## Exercises

- Verify that discrete topology and trivial topology satisfy axioms
- Show that continuous image of a compact space is compact (finite case)
- Construct a non-connected finite topological space
