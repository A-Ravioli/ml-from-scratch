# Module 10: Capstone Projects

## Prerequisites

- Modules 1-9

## Learning Objectives

- Integrate the full chapter into end-to-end ML workflows
- Compare multiple linear algebra formulations of the same problem
- See attention and neural networks as organized matrix calculus rather than magic

## Project 1: PCA From Scratch

Build a full PCA pipeline using only SVD:

- center data
- compute the SVD
- project onto principal directions
- reconstruct from the reduced coordinates
- plot explained variance

Apply it first to synthetic data and then to a real dataset such as MNIST.

## Project 2: Linear Regression, Four Ways

Solve the same regression task via:

- normal equations
- QR decomposition
- SVD / pseudoinverse
- gradient descent

Compare accuracy, numerical stability, and runtime. Use ill-conditioned design matrices to make the tradeoffs visible.

## Project 3: Attention Mechanism From Scratch

Implement scaled dot-product attention as matrix operations:

`softmax(QK^T / sqrt(d)) V`

Then extend it to multi-head attention by splitting the model dimension into orthogonal head subspaces.

## Project 4: Neural Net From Scratch

Implement a 2-layer neural network without autograd:

- forward pass with matrix multiplications
- backward pass using the chain rule
- gradient updates with plain gradient descent

Track the loss over time and relate the update equations back to the Jacobian ideas from Module 8.

## Suggested Extensions

1. Run PCA on image patches and visualize the learned basis vectors.
2. Compare regression methods under added noise and collinearity.
3. Add masking and causal attention.
4. Visualize a 2-parameter slice of the neural network loss landscape.
