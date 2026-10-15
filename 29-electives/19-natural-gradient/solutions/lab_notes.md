# natural gradient · lab notes

numbers are from the reference solution with the lab's defaults (one CPU thread, float64) on real MNIST for experiment 3; experiments 1 and 2 use synthetic data. wall-clock times were measured on a shared, busy machine; take them as ratios, not absolutes.

## experiment 1: three descriptions of one Gaussian

**what people usually predict:** gradient descent's three runs differ a bit but end in about the same place; natural gradient's runs are identical.

**what happens:** (MLE $\hat\mu = 1.991$, $\hat\sigma = 0.298$; positions as $(\mu, \sigma)$)

| step size | coordinates | GD after 60 steps | NGD after 10 steps | NGD after 60 steps |
|---|---|---|---|---|
| 0.1 | $(\mu, \sigma)$ | (1.546, 1.068) | (1.297, 2.063) | (1.988, 0.340) |
| 0.1 | $(\mu, \log\sigma)$ | (1.991, 0.303) | (1.297, 2.076) | (1.988, 0.344) |
| 0.1 | $(\mu, 1/\sigma^2)$ | (1.988, 0.728) | (1.297, 2.102) | (1.988, 0.354) |
| 0.5 | $(\mu, \sigma)$ | diverged ($\sigma \le 0$) at step 15 | (1.989, 0.355) | (1.991, 0.298) |
| 0.5 | $(\mu, \log\sigma)$ | (3.306, 1.652), oscillating | (1.989, 0.398) | (1.991, 0.298) |
| 0.5 | $(\mu, 1/\sigma^2)$ | (1.991, 0.466) | (1.989, 0.509) | (1.991, 0.298) |
| 0.02 | all three | (0.27–0.60, 1.57–2.76) | (0.364, 2.82–2.83) | (1.399, 1.90–1.91) |

at step size 0.1, gradient descent's three runs end in three different places: one has nearly converged, one is still at $\sigma \approx 0.73$, one at $\sigma \approx 1.07$ with $\mu$ far from the data. their paths in the picture have visibly different shapes (the precision run drops $\sigma$ abruptly in its first step). the three natural-gradient paths lie on top of each other: after 10 steps they agree on $\mu$ to three digits and on $\sigma$ to within 2%. at step size 0.5 the $\sigma$ run of gradient descent drives $\sigma$ through zero and dies, the $\log\sigma$ run oscillates in $\mu$, while all three natural-gradient runs reach the MLE; their 10-step positions differ more than at 0.1 (0.355 against 0.509 for $\sigma$).

**why:** gradient descent adds a gradient to a position, and the two transform oppositely under a change of coordinates. in $(\mu, \sigma)$ coordinates $\partial L/\partial\mu = -(\bar x - \mu)/\sigma^2$, so as $\sigma$ shrinks the effective step on $\mu$ grows like $1/\sigma^2$, overshoots, and the $\sigma$ step (which grows like $1/\sigma^3$) eventually throws $\sigma$ negative. in $\log\sigma$ coordinates the $\mu$ step still has the $1/\sigma^2$ factor, hence the oscillation at step size 0.5. the natural gradient multiplies by $F^{-1}$, which has exactly the compensating factors ($\sigma^2$ for $\mu$, $\sigma^2/2$ for $\sigma$), so the $\mu$ step is $\eta(\bar x - \mu)$ whatever $\sigma$ is: dimensionally consistent, and a fraction $\eta$ of the way to the sample mean in every coordinate system. the *direction* of each step is the same in all three systems; what differs is the curve a finite step follows (a straight line in $\sigma$, in $\log\sigma$, or in $1/\sigma^2$), which is why the gap between the three natural-gradient runs grows with the step size and shrinks to nothing as it goes to zero (at 0.02 they agree to two or three digits). natural gradient also rarely produces $\sigma \le 0$: its step on $\sigma$ is proportional to $\sigma$ itself, a multiplicative change.

## experiment 2: the empirical Fisher as a preconditioner

**what people usually predict:** the empirical Fisher is a noisier but reasonable stand-in, so it converges a bit slower than the Fisher but much faster than gradient descent.

**what happens:** (excess loss over least squares; 40 steps)

| real noise | method | after 1 step | after 5 | after 40 | first step below $10^{-6}$ |
|---|---|---|---|---|---|
| 1 (correct) | NGD, Fisher | $9\cdot10^{-15}$ | 0 | 0 | 1 |
| 1 (correct) | NGD, empirical Fisher | 9.46 | 8.19 | $2\cdot10^{-11}$ | 38 |
| 1 (correct) | GD, best step size (0.3) | 7.4 | 2.86 | 0.021 | not within 40 |
| 0.3 | NGD, empirical Fisher | 9.5 | 8.23 | 0.56 (bouncing) | not within 40 |
| 3 | NGD, empirical Fisher | 9.38 | 8.23 | 0.65 | not within 40 |

the Fisher run solves the problem in one step at every noise level. the empirical-Fisher run barely moves for 30 steps (after 5 steps it has removed 15% of the excess loss, where gradient descent has removed 70%), then with the correct noise level converges abruptly and very fast. with the noise at 0.3 it reaches the neighbourhood of the solution and then bounces around it, never converging; with the noise at 3 it creeps towards it far more slowly than gradient descent.

**why:** for a linear-Gaussian model the Fisher is $\frac1N\Phi^\top\Phi$, the Hessian of the loss, so natural gradient is Newton's method and solves a quadratic in one step (on paper 3). the empirical Fisher weights each $\phi_n\phi_n^\top$ by the squared residual $r_n^2$. at the start the residuals are about 4.5 times larger than the noise (mean squared residual about 20), so the empirical Fisher is about 20 times too large and each step about 20 times too short; as the fit improves the residuals shrink, the steps lengthen, and progress accelerates: a step size that's inversely proportional to the loss, which is backwards. near the solution the residuals are just the noise, so the empirical Fisher is about $\sigma_{\text{real}}^2F$ and the step about $1/\sigma_{\text{real}}^2$ times Newton's. with the correct noise level that factor is 1 and the last few steps are Newton steps; with noise 0.3 they're 11 times too long and overshoot (the run bounces until the residuals grow, the steps shrink, and it falls back in); with noise 3 they're 9 times too short. the true Fisher depends only on the model, never on the residuals, so it doesn't care. this is the core of Kunstner et al.'s (2019) argument: the empirical Fisher equals the Fisher only for a well-specified model at its optimum, and away from there it can be worse than no preconditioning at all.

## experiment 3: natural gradient on a real network

**what people usually predict:** natural gradient needs fewer steps, maybe 2–3 times fewer, and loses badly in wall-clock time because each step is expensive.

**what happens:** (3 seeds; Adam's step size $10^{-2}$ was the best of the four on seed 0; target training loss 0.2; CG iterations 10 unless noted)

| method | final training loss (median) | steps to 0.2 | seconds to 0.2 | ms per step | test accuracy |
|---|---|---|---|---|---|
| SGD (0.5) | 0.154 | 32–33 | 0.04–0.10 | 1.2–2.7 | 0.842 ± 0.003 |
| Adam ($10^{-2}$) | 0.126 | 29–30 | 0.04–0.08 | 1.1–2.6 | 0.830 ± 0.004 |
| NGD, 10 CG iterations | 0.0004 | 4 | 0.10–0.16 | 20–24 | 0.852 ± 0.001 |
| NGD, 3 CG iterations | 0.0067 | 9–10 | 0.14–0.15 | 15 | 0.855 ± 0.002 |
| NGD, 30 CG iterations | 0.0004 | 3 | 0.26–0.39 | 32 | 0.838 ± 0.002 |

(ranges are over seeds and over two repeated runs of the lab; the steps and the accuracies are deterministic.)

natural gradient reaches the target in 4 steps against Adam's 30, and after 40 steps its training loss is 300 times lower. each of its steps costs 15–20 Adam steps, though, so in wall-clock time to a training loss of 0.2 Adam and SGD are about twice as fast. test accuracy is about the same for all three, a point or two either way.

**why:** the network is small and the batch is the whole training set, so the curvature estimate is exact and the quadratic model is good: each natural-gradient step does what dozens of first-order steps do, because it rescales every direction by its curvature (the Fisher's eigenvalues span many orders of magnitude, topic 18's experiment 1) instead of using one step size for all. the price is roughly $1 + 3k$ backward-sized passes for $k$ CG iterations plus overhead, which this experiment measures at 15–20 times an Adam step. so the step advantage (7–10×) doesn't survive the cost per step at a loose target; it would at a tight one (Adam needs far more than 40 steps to reach a loss of 0.001), and it shrinks with minibatches (noisy curvature, so more damping and smaller steps) and with bigger networks (more CG iterations needed). fewer CG iterations make each step cheaper but less effective, more make it more expensive for little gain: the CG budget is a real knob. and the test accuracy is the reminder that optimization speed is not generalization: driving the training loss on 1,000 images to $4\cdot10^{-4}$ memorizes them, and buys at most a point of test accuracy. topic 20's K-FAC attacks the cost per step: it gets a curvature-aware step for a few times the cost of an Adam step.
