# 12 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and the non-centered posterior matches `eight_schools_exact` within its Monte Carlo error | all pass, plus your own checks: e.g. a model whose posterior you know (a conjugate normal, a scale by quadrature) recovered by both HMC and VI, and a deliberately broken Jacobian that your tests catch |
| **results** | no numbers | posterior means only | posterior means, sds and intervals for $\mu$, $\tau$ and the $\theta_j$ from both parameterizations and VI, each with ESS, $\hat R$ and a Monte Carlo error, next to the exact values | the same, plus the funnel figure with the divergent states marked, $P(\tau < 1)$ and $\mathbb E[\log\tau]$ against the exact values, and a timing comparison of your one-pass gradient with `mlfs.autodiff.grad` |
| **analysis** | none | describes the tables | explains the centered sampler's failure with the funnel's geometry (step size versus curvature) and the fix with the non-centered parameterization; explains VI's smaller spread with the direction of the KL | also tests an explanation with an experiment: e.g. a smaller step size or longer warm-up for the centered model (does it help, and at what cost?), the centered model with much smaller $\sigma_j$ (where it should beat the non-centered one), or a full-covariance VI fit |
| **clarity** | hard to follow | readable | labelled figures and tables; all five prompts answered with numbers | a reader could reproduce every number (config, seeds, step sizes) and could reuse your library on a new model from your description of its interface |

## what graders look for

- **ground truth first.** the exact posterior from quadrature is the yardstick; every sampler result is compared with it, with an error bar.
- **diagnostics that are read, not just printed.** which of ESS, $\hat R$, divergences and the $\log\tau$ histogram flagged the centered sampler, and which didn't? what would you have concluded without the exact answer?
- **the geometry.** a clear account of *why* the funnel breaks a fixed step size, in terms of the curvature at small $\tau$, and why the non-centered parameterization doesn't have it (with the data this weak).
- **honest failures.** a run where even the non-centered sampler struggles, a VI fit that's overconfident, a chain that got stuck: reported and explained, not hidden by a lucky seed.
- **the library as a library.** a model interface someone else could use, with clear rules (what a model function may call, what must be Dual-friendly, how constraints are handled).
