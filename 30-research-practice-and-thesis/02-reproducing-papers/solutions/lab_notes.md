# reproducing papers · lab notes

numbers below are from the reference solution. experiments 1 and 2 are simulations with fixed seeds (2,000 and 4,000 simulated reproductions per cell, so rates carry a Monte Carlo error of about ±0.01). experiment 3 uses real MNIST from the cache on a laptop CPU (`OMP_NUM_THREADS=1`; about 0.1 s per run, deterministic on the same machine). the whole lab takes about 11 seconds. offline, experiment 3 runs on a synthetic stand-in and its verdicts mean nothing.

## experiment 1: "the paper's number is inside my interval"

**what people predict.** that the naive rule rarely accepts a reproduction that's 2 margins off (most guess 5–20%), and that more seeds make every criterion more reliable in the same direction.

**what happens.** rates of "reproduced", seed noise equal to the margin:

| truth off by | rule | 3 seeds | 5 | 10 | 20 |
|---|---|---|---|---|---|
| 0 margins | naive | 0.95 | 0.95 | 0.95 | 0.94 |
| 0 margins | equivalence | 0.12 | 0.30 | 0.80 | 0.99 |
| 1 margin | naive | 0.82 | 0.60 | 0.20 | 0.01 |
| 1 margin | equivalence | 0.05 | 0.04 | 0.05 | 0.04 |
| 2 margins | naive | 0.53 | 0.10 | 0.00 | 0.00 |
| 2 margins | equivalence | 0.00 | 0.00 | 0.00 | 0.00 |

with 3 seeds, the naive rule declares a reproduction that's **2 margins off** a success **53%** of the time, and one that's 1 margin off 82% of the time. its false-success rate *falls* as you add seeds. your equivalence verdict never accepts the 2-margin miss, accepts the 1-margin case (the truth sits exactly at the band's edge) about 5% of the time at every sample size, and accepts a correct reproduction 12% of the time at 3 seeds, 80% at 10, 99% at 20.

**why.** the naive rule is a test of "my result differs from the paper's" that counts a failure to detect a difference as success. with 3 seeds the 95% $t$ interval is $\bar x \pm 4.30\, s/\sqrt 3 = \bar x \pm 2.5$ margins wide on each side, so almost any paper value lands inside it. that's on-paper problem 2 with the $t$ quantile instead of 1.96. the equivalence verdict turns the burden around: "reproduced" has to be *shown*, so with too few seeds it says "inconclusive" instead (88% of the time at 3 seeds for a correct reproduction). that's the price: an honest test of a small experiment mostly says "I can't tell". and at the edge of the margin it's calibrated: the 4–5% rate is the $\alpha = 0.05$ the 90% interval promises. the lesson for R1–R3: never accept "within my error bars" as a reproduction criterion, and size the experiment (experiment 2) so that the honest test can say something.

## experiment 2: is `seeds_needed` honest?

**what people predict.** about 5 seeds for sd = margin, and that the approximation is either spot on or wildly off.

**what happens.**

| sd / margin | seeds_needed | simulated power | one fewer seed | its power |
|---|---|---|---|---|
| 0.25 | 3 | 0.982 | 2 | 0.556 |
| 0.5 | 4 | 0.822 | 3 | 0.571 |
| 1 | 10 | 0.799 | 9 | 0.726 |
| 2 | 36 | 0.802 | 35 | 0.802 |

at the recommended number of seeds the simulated power is 0.80–0.98, and one seed fewer drops below 0.8 except at sd/margin = 2 (35 and 36 seeds give the same 0.80 within Monte Carlo error). sd = margin needs **10** seeds.

**why.** the formula replaces the random sample sd $s$ by the true $\sigma$. at small $n$, $s$ is very noisy, but the $t$ quantile's heavy tail at $n - 1$ degrees of freedom roughly compensates, and the integer jump from $n - 1$ to $n$ is big when $n$ is small (going from 2 to 3 seeds cuts $t_{0.95}$ from 6.31 to 2.92), which is why the power overshoots to 0.98 at sd/margin = 0.25. at large $n$ the approximation is essentially exact and each extra seed adds little. for the explain prompt: with a pilot sd of 0.6 points and a margin of 0.5, `seeds_needed(0.6, 0.5)` gives 14 seeds; with 5, a correct reproduction is confirmed only 16% of the time (simulated; at $n = 5$ the half-width alone is about $2.13 \times 0.6/\sqrt5 = 0.57$, more than the margin, so it fits only when $s$ happens to come out small), so most likely you'd report "inconclusive". remember the pilot sd is itself a noisy estimate from a few runs; budget some slack.

## experiment 3: the unreported learning rate

**what people predict.** one or two verdicts across the sweep; most expect the "reasonable" learning rates (0.01, 0.03) to reproduce the paper.

**what happens.** Adam alone reaches 89.67% (sd 0.45 over 5 seeds). the paired difference Adam − SGD, 90% $t$ intervals, against the claim $4 \pm 1.5$ points:

| SGD learning rate | SGD accuracy | Adam − SGD (points) | verdict |
|---|---|---|---|
| 0.001 | 75.18% | +14.49 [12.40, 16.59] | partly reproduced |
| 0.003 | 85.33% | +4.34 [4.00, 4.67] | **reproduced** |
| 0.01 | 88.98% | +0.69 [0.15, 1.23] | partly reproduced |
| 0.03 | 89.82% | −0.15 [−1.02, 0.71] | not reproduced |
| 0.1 | 85.99% | +3.68 [1.42, 5.95] | direction reproduced, size inconclusive |

the same claim, the same code and the same seeds get **four different verdicts** from five guesses, and the effect ranges from −0.15 to +14.5 points depending on a number the paper didn't report. only the 0.003 guess "reproduces" it, and at the learning rate where SGD does best (0.03), the paper's claim is flatly not reproduced: SGD matches Adam. the two "partly" cases are partly in opposite directions: at 0.001 the effect is shown to be far *larger* than claimed, at 0.01 far smaller. the 0.1 case is different again: its interval is above 0 but so wide (two unstable seeds) that it overlaps the band, so the direction is confirmed and the size is simply unresolved; calling it "partly reproduced" would treat a failure to confirm the size as evidence against it.

**why.** one epoch is a short budget, and SGD's progress in it depends strongly on the learning rate (too small: it barely moves; too large: it's unstable, which is why the 0.1 interval is wide: its five seeds range from 82.6% to 88.3%). Adam's per-parameter scaling makes it far less sensitive. so "Adam beats SGD by 4 points" is really "Adam beats *badly tuned* SGD by 4 points", and the paper's missing detail is the whole story. what to publish: the full sweep, with the verdict at each value, and the conclusion that the claim holds only for learning rates near 0.003 and fails for SGD's best learning rate. the value that reproduces the paper doesn't make the paper right: it shows which (unreported) choice the paper's number corresponds to, and topic 03's tuning-budget experiment shows why comparing a tuned method with an untuned baseline is the usual way such claims arise. the generated report for the 0.01 guess shows the format: "partly reproduced", one gap, and 6 of 10 checklist items still unanswered, visibly.
