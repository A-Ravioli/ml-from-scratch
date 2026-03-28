# VAE Theory

## Prerequisites
- Probability, KL divergence, and latent-variable models
- Basic neural-network notation and optimization
- Familiarity with the vanilla VAE construction

## Learning Objectives
- Derive the evidence lower bound from first principles
- Understand the encoder/decoder factorization used in VAEs
- Implement reparameterization, reconstruction, and KL terms in a small deterministic setting
- Connect latent-space geometry to diagnostics such as posterior collapse

## Mathematical Foundations

### 1. Latent-variable modeling
A variational autoencoder introduces a latent variable `z` and models data with `p_theta(x, z) = p_theta(x | z) p(z)`.
Exact marginal likelihood is intractable in general, so we introduce an approximate posterior `q_phi(z | x)`.

### 2. ELBO derivation
Starting from `log p_theta(x)`, insert `q_phi(z | x)` and apply Jensen's inequality:
`log p_theta(x) >= E_q[log p_theta(x | z)] - KL(q_phi(z | x) || p(z))`.
The first term rewards faithful reconstructions, while the second keeps the latent representation close to the prior.

### 3. Reparameterization
To differentiate through samples, write `z = mu(x) + sigma(x) * eps` with `eps ~ N(0, I)`.
This separates stochasticity from the encoder parameters and turns sampling into a deterministic computation graph with random input.

### 4. Failure modes
Posterior collapse appears when the decoder becomes so expressive that it ignores the latent code.
Studying KL statistics and latent variances is therefore a useful diagnostic tool.

## Implementation Details
The exercise centers on a compact encoder, decoder, full VAE wrapper, and a few analysis utilities.
The goal is not to match a production VAE, but to make the ELBO pieces visible in code and easy to test.

## Suggested Experiments
1. Change the latent dimension and observe reconstruction quality.
2. Compare ELBO components across different synthetic datasets.
3. Track which latent dimensions collapse toward the prior.

## Research Connections
- Vanilla VAE theory underlies beta-VAE, hierarchical VAE, and discrete latent-variable models.
- The main research questions concern expressivity, disentanglement, posterior collapse, and scalable likelihood estimation.
