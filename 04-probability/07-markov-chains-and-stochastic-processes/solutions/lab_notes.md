# markov chains & stochastic processes · lab notes

## experiment 1: mixing time tracks the spectral gap

**what people usually predict:** mixing time grows roughly with the number of nodes for every graph, maybe a bit faster for sparse graphs.

**what happens:**

| graph | n = 6 | n = 16 | n = 40 |
|---|---|---|---|
| cycle | 4 (1/γ = 4.0) | 25 (26.3) | 152 (162) |
| complete | 2 (1.7) | 2 (1.9) | 2 (2.0) |
| barbell | 8 (9.8) | 50 (69) | 291 (417) |

the complete graph mixes in 2 steps at every size. the cycle and the barbell both grow like $n^2$, and at $n = 40$ the barbell needs about 150 times as long as the complete graph, although it has about half as many edges (381 against 780): a factor of 2 in edges can't account for a factor of 150 in time. across all 18 chains, the mixing time is within a factor of about 1.5 of $1/\gamma$: the points hug the diagonal on the log–log plot.

**why:** the complete graph has no structure to get stuck in: one step (two for the lazy walk) lands almost anywhere. on a cycle, the walk moves like a 1-D random walk, which needs about $n^2$ steps to travel distance $n$ (topic 05's $\sqrt t$ spread, inverted). the spectral gap knows this: $\lambda_2 = \tfrac12 + \tfrac12\cos(2\pi/n)$, so $\gamma \approx \pi^2/n^2$. the barbell's problem is a **bottleneck**: inside each clique the walk mixes instantly, but to cross it must be at the one bridge node (probability about $2/n$ of the time) *and* pick the bridge edge (probability about $1/n$ for the lazy walk), so crossings happen about once every $n^2/2$ steps. until the walk has had a chance to cross, it can't know which half it's "supposed" to be in. the spectral gap detects bottlenecks in general (Cheeger's inequality relates $\gamma$ to the narrowest cut of the graph), which is why it predicts mixing so well. this is exactly the trouble with MCMC on multimodal posteriors: two modes separated by a low-probability region are a barbell, and a sampler can spend its entire run in one mode while every within-mode diagnostic looks fine.

## experiment 2: time averages vs distributions

**what people usually predict:** for the periodic chain, either "nothing converges" or "both converge, just more slowly".

**what happens:** for chain A, $p_t$ converges to $\pi \approx (0.257, 0.360, 0.157, 0.227)$ within a few steps (mixing time 2), and the time fractions along one path converge to the same numbers (0.253, 0.360, 0.161, 0.227 after 20,000 steps). for chain B, $p_t$ never converges: at even times the chain is certainly in $\{0, 1\}$ and at odd times in $\{2, 3\}$, so $p_t$ swings back and forth forever. `mixing_time` returns `None` and the spectral gap is 0 ($\lambda_2 = -1$). yet the time fractions along one path converge just fine, to $\pi \approx (0.174, 0.326, 0.248, 0.252)$ (0.169, 0.331, 0.247, 0.253 after 20,000 steps).

**why:** these are two different kinds of convergence. $p_t \to \pi$ is a statement about the distribution at a *fixed time* $t$, and for a periodic chain the fixed time carries information forever (its parity tells you which group you're in). the ergodic theorem is about *averages over time*, which wash out the oscillation: half the steps are spent in each group, and within each group the chain visits states in the right proportions. irreducibility is enough for the ergodic theorem; aperiodicity is needed only for $p_t \to \pi$. MCMC needs the ergodic theorem, because it averages over the path, but in practice samplers are made aperiodic anyway (for example by adding a "stay" probability, as in the lazy walks above) so that the state at a fixed late time is also a good sample. $\lambda_2 = -1$ is the spectral signature of period 2: the eigenvector for $-1$ is $+1$ on one group and $-1$ on the other, a pattern that flips sign every step and never decays.

## experiment 3: how much time does Brownian motion spend above zero?

**what people usually predict:** "about half the time, by symmetry", with a bell-shaped histogram around 1/2. for $P(\max \ge 1)$, people usually say about 0.16, the probability that $B_1 \ge 1$.

**what happens:** the histogram is U-shaped, matching the arcsine density $1/(\pi\sqrt{x(1-x)})$: 1/2 is the *least* likely fraction. 41.5% of paths spend more than 90% of their time on one side of zero (the arcsine law predicts 41.0%), and only 12.7% are between 40% and 60% (predicted 12.8%). and $P(\max_{t \le 1}B_t \ge 1) = 0.311$, about twice $P(B_1 \ge 1) = 0.159$ (the exact answer is $0.317$; the discrete grid misses some excursions above 1 between grid points).

**why:** symmetry says the *average* fraction is 1/2, not that a typical path splits its time evenly. once a Brownian path wanders away from 0, the time it takes to come back has an extremely heavy tail: returns are rare and excursions are long. so a typical path spends a long stretch on one side and comparatively little time switching. Lévy's arcsine law makes this exact: the fraction of time positive has CDF $\frac{2}{\pi}\arcsin\sqrt x$. the same law applies, approximately, to a gambler's running winnings, or to which of two equally good teams is "ahead" over a season: one side usually leads for most of the time, and that's not evidence that it's better. for the maximum, the **reflection principle**: take any path that reaches 1 and reflect everything after its first hitting time across the level 1. reflection is a bijection between paths ending above 1 and paths that hit 1 and end below it, and both sets are equally likely by symmetry of the increments. so $P(\max \ge 1) = P(B_1 \ge 1) + P(\text{hit 1, end below}) = 2P(B_1 \ge 1)$. this is the start of a long story about first-passage times and stopping, which continues with martingales in the advanced track.

reference: Lévy (1939), "Sur certains processus stochastiques homogènes", *Compositio Mathematica* 7: 283–339.
