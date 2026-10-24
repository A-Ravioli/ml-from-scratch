# score matching · lab notes

numbers measured with the reference solution on an Apple M2 laptop CPU (one thread, as in CI) while the machine was heavily shared; the whole lab took about 28 s. the data are synthetic (a two-component Gaussian mixture generated in the lab), so offline runs are the same experiments with the same numbers. sampling noise on a fraction from 5,000 chains is about ±0.006.

## experiment 1: the exact score is not enough

**what people usually predict:** with the exact score, Langevin samples the mixture exactly: 20% of the chains in the small component.

**what happens:**

| sampler | fraction in the small component | chains starting on its side |
|---|---|---|
| Langevin, exact score (2,000 steps, h = 0.1) | **0.490** | 0.500 |
| annealed Langevin, exact scores (10 levels, 20 → 0.1) | **0.258** | 0.500 |
| true mixture | 0.200 | |

plain Langevin ends with 49% of its chains in the 20% component: essentially the fraction that *started* on that side of the diagonal. annealing brings it to 26%, much closer to the truth but not exactly there.

**why:** near either mode, the mixture's score is that component's score, $-(x - \mu_k)$, multiplied by a posterior responsibility that is 1 to many decimal places; the weights only matter where both components contribute, near the diagonal, where the density is about $e^{-25}$ relative to the modes. a chain would have to climb through that region to cross, which in 2,000 steps of size 0.1 essentially never happens. so each chain samples the component on its own side, and the proportions are inherited from the initialization. at $\sigma = 20$ the blurred mixture is one broad blob whose mass is still skewed towards the heavy component, and chains mix across it easily; as $\sigma$ shrinks the blob splits, and the chains are already apportioned roughly by the weights. the residual 6-point excess comes from the levels around $\sigma \approx 3$–5, where the modes separate: there, 100 steps per level are not enough for the chains to re-equilibrate between the halves before the next level freezes them in (more levels or more steps per level reduce it).

## experiment 2: where a learned score can be trusted

**what people usually predict:** the learned score is accurate near the data and garbage far away; samples from it look broken.

**what happens** (2,000 training steps per network, about 1 s each after warm-up):

- median relative error of the one-level ($\sigma = 0.1$) score: **0.35** within distance 2 of a mode, **0.59** more than 5 from both. the error map is fairly uniform, with error bands near the modes and along the diagonal.
- samples from uniform noise:

| samples from uniform noise | small component | within 3 of a mode |
|---|---|---|
| one level, Langevin (2,000 steps, h = 0.01) | 0.540 | 0.954 |
| ladder, annealed (10 levels × 100 steps) | 0.232 | 0.954 |
| true mixture | 0.200 | 0.989 |

the far-field error is only moderately worse than the near-field one, and the one-level samples *look* fine (95% within 3 of a mode, like the ladder's). what's wrong is the proportions: 54% in the small component, against 23% for the annealed ladder.

**why:** an MLP with smooth activations extrapolates a nearly linear field like this one reasonably well, so on this toy the "garbage far away" failure is mild (it's much worse for complex data in high dimension, where most of the space is far from any data). the error *near* the data is large for a different reason: at $\sigma = 0.1$ the regression target $-z/\sigma$ has a standard deviation of 10 per coordinate while the true blurred score near a mode is of order 1, so 2,000 minibatches average a very noisy target (the irreducible loss of idea 1 is large relative to the signal). the decisive failure is the one experiment 1 showed with the exact score: plain Langevin can't move mass between separated modes, so its proportions are wrong however accurate the score. the ladder fixes both problems at once: large-$\sigma$ training data cover the plane, and annealed sampling apportions the chains while the modes are still connected.

## experiment 3: sliced score matching, and what DSM really learns

**what people usually predict:** DSM at a small $\sigma$ is the most accurate; sliced score matching is a noisy approximation; DSM at $\sigma = 1$ is "a bit blurry".

**what happens** (relative squared error on 5,000 fresh data points):

| training | vs the clean score | vs the score blurred at σ = 1 | ms per step |
|---|---|---|---|
| sliced SM (no noise) | **0.028** | 0.648 | 1.62 |
| DSM, σ = 0.1 | 0.119 | 1.137 | 0.71 |
| DSM, σ = 1 | 0.263 | **0.009** | 0.91 |

sliced score matching gets closest to the clean score, four times closer than DSM at $\sigma = 0.1$, at about twice the cost per step. DSM at $\sigma = 1$ is almost perfect, for the blurred score, and 26% off the clean one.

**why:** for unit-variance components, blurring by $\sigma = 1$ doubles each component's variance, so near a mode the blurred score is $-(x - \mu)/2$, half the clean score: the relative squared error between the two is exactly $0.25$, and the DSM row (0.263 against the clean score, 0.009 against the blurred one) shows that DSM learned precisely what the theory says it learns. sliced score matching has no blur and, here, a lower-variance estimator than DSM at a small $\sigma$, whose targets $-z/\sigma$ are dominated by noise (experiment 2); its costs are the extra backward pass for $v^\top\nabla_x(v^\top s)$ (here about 2× per step), a projection noise that grows with dimension, and no built-in multi-scale ladder. you'd choose it when you need the score of the clean data (e.g. for score-based density estimation or a test statistic), and DSM when you want a sampler, where the blur is exactly what annealing needs.

## references

- Song & Ermon (2019), "Generative modeling by estimating gradients of the data distribution", *NeurIPS*. section 3 has this experiment's mixture. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
- Song, Garg, Shi & Ermon (2019), "Sliced score matching: a scalable approach to density and score estimation", *UAI*. [arXiv:1905.07088](https://arxiv.org/abs/1905.07088)
