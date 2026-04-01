# Module 8: Matrix Calculus

## Prerequisites

- Modules 1-7
- Multivariable calculus

## Learning Objectives

- Compute gradients, Jacobians, and Hessians for common ML objectives
- Interpret Hessian eigenvalues as local curvature information
- Understand backpropagation as structured Jacobian chain rule

## Lesson 8.1: Derivatives with Respect to Vectors

### Concept

For scalar-valued functions, the gradient collects partial derivatives into a vector. For vector-valued functions, the Jacobian collects them into a matrix.

### Intuition

The gradient points uphill fastest. The Jacobian generalizes “local linearization” to vector outputs.

### Code Focus

- Implement a numerical gradient checker
- Derive the gradient of `||Ax - b||^2`
- Verify analytic and numerical gradients agree

## Lesson 8.2: Hessians and Second-Order Structure

### Concept

The Hessian is the matrix of second derivatives. It measures curvature and links matrix calculus back to positive definiteness.

### Intuition

- Positive eigenvalues: bowl
- Mixed signs: saddle
- Near-zero eigenvalues: flat directions

### Code Focus

- Compute Hessians numerically
- Check definiteness from eigenvalues
- Visualize curvature for a quadratic loss

## Lesson 8.3: Backprop as Jacobian Chain Rule

### Concept

Backpropagation repeatedly applies the chain rule, but in a way that uses Jacobian-vector products instead of explicitly materializing full Jacobians.

### Intuition

Computation graphs let us cache local derivatives and push sensitivities backward efficiently.

### Code Focus

- Build a tiny scalar autograd engine
- Extend ideas to vector-valued parameters
- Derive and verify gradients for a 2-layer network manually

## Study Questions

1. Why is numerical differentiation useful even when we already know the analytic answer?
2. How does Hessian definiteness connect to positive definite matrices?
3. Why do modern autodiff systems avoid forming full Jacobians?
