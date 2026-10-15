# trust regions & PPO · lab notes

numbers are from the reference solution with the lab's defaults (one CPU thread).

## experiment 1: does the clip keep you in the trust region?

**what people usually predict:** with the clip, ratios stay at or just below $1 + \varepsilon = 1.2$ however many epochs you run; the KL stays roughly constant once the clip engages.

**what happens:** (mean KL(old ‖ new) / largest ratio / fraction of ratios outside $[0.8, 1.2]$, on the batch)

| epochs | clip $\varepsilon = 0.2$ | no clip |
|---|---|---|
| 1 | 0.0001 / 1.04 / 0.00 | 0.0001 / 1.04 / 0.00 |
| 5 | 0.0017 / 1.20 / 0.00 | 0.0017 / 1.20 / 0.00 |
| 10 | 0.0032 / 1.33 / 0.02 | 0.0087 / 1.48 / 0.13 |
| 20 | 0.0038 / 1.38 / 0.03 | 0.0483 / 2.40 / 0.38 |
| 40 | 0.0048 / 1.42 / 0.06 | 0.2313 / 4.92 / 0.66 |

for the first five epochs the two runs are identical: no ratio has reached the clip yet, so the clipped and plain objectives have the same gradient. after that the clip clearly helps (at 40 epochs the KL is 50 times smaller than without it, and the largest ratio 1.42 instead of 4.9) but it doesn't hold the line: the largest ratio passes 1.2 and keeps creeping up, 6% of the samples end up outside the range, and the KL keeps growing with every epoch.

**why:** the clip removes a sample's *own* incentive to move further, nothing more. a sample whose ratio is past $1 + \varepsilon$ contributes zero gradient, but its probability still changes whenever the parameters change, and they keep changing because of the samples that are still inside the range (all samples share the network's weights, and nearby states have similar outputs), and because Adam's momentum carries earlier gradients forward. so the clipped objective slows the drift once ratios hit the boundary rather than stopping it; with a larger learning rate or more epochs the leak grows (Engstrom et al., 2020, and Ilyas et al., 2020, report ratios far outside the range in MuJoCo runs). a real bound needs a real constraint: TRPO's KL constraint with a line search, or the cheap backstop most PPO implementations add, early stopping when the approximate KL passes a target (your `target_kl`). "proximal" is a fair name for PPO; "trust region" overstates it.

## experiment 2: implementation details that matter (on CartPole)

**what people usually predict:** every detail helps a bit, most of all advantage normalization and gradient clipping (the ones blog posts emphasize).

**what happens:** (3 seeds, mean ± standard error)

| variant | last 20 episodes | mean over training | per seed |
|---|---|---|---|
| PPO (all details) | 500 ± 0 | 246 ± 11 | 234, 269, 236 |
| no advantage normalization | 459 ± 13 | 211 ± 19 | 181, 246, 204 |
| no gradient clipping | 500 ± 0 | 247 ± 20 | 206, 271, 263 |
| no lr annealing | 500 ± 0 | 250 ± 11 | 236, 271, 242 |
| 1 epoch per batch | 371 ± 29 | 134 ± 9 | 118, 147, 138 |
| no clipping ($\varepsilon = 10^9$) | 287 ± 109 | 100 ± 16 | 82, 132, 87 |

two removals hurt clearly: the clip (mean return over training 100 against 246, and two of three seeds far from solving) and sample reuse (134). advantage normalization helps a little (211 against 246, a difference about the size of the noise with three seeds). gradient clipping and learning-rate annealing make no measurable difference.

**why:** these settings reuse each batch for 20 epochs at lr $3\cdot10^{-3}$. that's where the clip earns its keep: without it, 20 epochs of steps on the plain surrogate push the policy far from the one that collected the data (experiment 1's no-clip column, at four times the learning rate), and the importance-weighted objective stops describing the policy being updated; updates overshoot, a bad policy collects a bad batch, and the run limps. the clip doesn't hold the policy inside a hard trust region, but it slows the drift enough to make aggressive reuse safe. reuse itself is the point of PPO: one epoch per batch is essentially A2C with large batches, and it learns half as fast per environment step.

the details that don't matter here matter elsewhere: CartPole's rewards are all $+1$, so advantages have a sensible scale without normalization (it matters most when reward scales are large or change over training); the network is tiny and the inputs are small, so gradients rarely spike (clipping guards against rare huge gradients, which come with large networks, long horizons, or bad batches); and 60,000 steps is short, so annealing never reaches the phase where a large learning rate would knock a near-optimal policy off its peak. with the gentler setting of 10 epochs at lr $10^{-3}$, the reference solution finds only the number of epochs making a difference; the clip is then harmless to remove. Andrychowicz et al. (2021) make the general point with 250,000 runs: the details that matter depend on the task and on the other choices.
