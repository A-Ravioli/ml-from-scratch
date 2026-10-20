# K-FAC · lab notes

numbers are from the reference solution with the lab's defaults: experiment 1 on synthetic blobs (float64), experiment 2 on real MNIST (float32, one CPU thread). losses, steps and accuracies are deterministic; wall-clock times were measured twice on a shared, busy machine and are given as ranges over both runs and the three seeds. read them as ratios.

## experiment 1: how good is the Kronecker approximation?

**what people usually predict:** an approximation that's off by about half its size should give a direction about halfway between the gradient and the natural gradient, or worse.

**what happens:** (fraction of the exact natural gradient's quadratic-model decrease, each direction with its best step length)

| network | direction | λ = 10⁻⁴ | 10⁻³ | 10⁻² | 10⁻¹ |
|---|---|---|---|---|---|
| early (20 Adam steps) | gradient | 0.65 | 0.68 | 0.74 | 0.87 |
| early | K-FAC (factored damping) | 0.91 | 0.89 | 0.85 | 0.83 |
| early | block-diagonal exact Fisher | 0.98 | 0.98 | 0.95 | 0.95 |
| trained (150 Adam steps) | gradient | 0.62 | 0.64 | 0.77 | 0.97 |
| trained | K-FAC (factored damping) | 0.81 | 0.81 | 0.82 | 0.93 |
| trained | block-diagonal exact Fisher | 0.98 | 0.95 | 0.96 | 0.99 |

the Kronecker blocks differ from the exact blocks by 63%, 70% and 45% (early network; 82%, 77% and 37% trained) in relative Frobenius norm, and the between-layer blocks that K-FAC drops hold 63% (54%) of the Fisher's norm. even so, at small damping K-FAC's direction achieves 81–91% of the natural gradient's decrease, against 62–68% for the gradient. dropping the between-layer blocks alone costs almost nothing (95–98%). as the damping grows, all three directions approach each other, and at $\lambda = 0.1$ the gradient catches up with (early) or overtakes (trained) K-FAC.

**why:** the Frobenius error is dominated by the largest entries of the Fisher, i.e. its top eigen-directions, where the curvature is large and the gradient's step is already short. the natural gradient's benefit comes from the other end of the spectrum: the many low-curvature directions where the gradient's step is far too short. what matters for the step is whether the approximation gets the *relative* scale of curvature right across directions, roughly right, over many orders of magnitude, and K-FAC's two factors capture the dominant structure (which inputs are large and correlated, $A$; which outputs the loss is sensitive to, $G$) even when the entries are off. that's Martens & Grosse's (2015) argument too: the approximation to $F^{-1}$ is better than the approximation to $F$ looks. dropping the between-layer blocks matters little here for a related reason: the inverse of the Fisher is much closer to block diagonal than the Fisher itself. damping pulls every preconditioner towards a multiple of the identity, so for large $\lambda$ all three directions converge to the gradient's; K-FAC's factored damping adds the cross terms of idea 1 on top of $\lambda I$, so it gets there faster, which is why the gradient can overtake it at the largest damping, particularly for the trained network, whose curvature is smaller relative to $\lambda$.

## experiment 2 (project): K-FAC against a tuned Adam on chapter 15's network

**what people usually predict:** K-FAC needs several times fewer steps but is several times slower in wall-clock time, so Adam wins overall; test accuracy is about the same.

**tuning runs** (seed 0; final training loss on 2,000 fixed training images / validation accuracy on 5,000 held-out images):

| Adam lr | $3\cdot10^{-4}$ | $10^{-3}$ | $3\cdot10^{-3}$ | $10^{-2}$ |
|---|---|---|---|---|
| | 0.243 / 0.917 | 0.135 / 0.950 | **0.078 / 0.963** | 0.106 / 0.943 |

| K-FAC damping \ KL clip | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ |
|---|---|---|---|
| $10^{-2}$ | **0.013 / 0.963** | 0.079 / 0.945 | 0.343 / 0.918 |
| $10^{-1}$ | 0.070 / 0.962 | 0.023 / 0.961 | 0.083 / 0.948 |
| $1$ | 0.197 / 0.933 | 0.078 / 0.961 | 0.050 / 0.957 |

both selection rules pick the same config for each optimizer: Adam with step size $3\cdot10^{-3}$ (an interior optimum of its grid) and K-FAC with damping $10^{-2}$ and KL clip $10^{-4}$ (a corner of its grid; see the caveats below).

**the comparison** (3 seeds; target = Adam's median final training loss, 0.0819):

| run | final training loss (median) | steps to target | seconds to target | ms per step | test accuracy (mean ± s.e.) |
|---|---|---|---|---|---|
| Adam, lr $3\cdot10^{-3}$ | 0.0819 | 204–210 | 0.19–0.28 | 1.0–1.3 | 0.9597 ± 0.0028 |
| K-FAC, λ = 0.01, κ = $10^{-4}$ | 0.0134 | 114–126 | 0.44–0.74 | 3.9–6.2 | 0.9656 ± 0.0006 |
| K-FAC, empirical Fisher | 0.0177 | 120–126 | 0.43–0.63 | 3.6–5.0 | 0.9652 ± 0.0006 |
| K-FAC, cheap curvature (factors every 10, inverses every 50) | 0.0132 | 114–126 | 0.23–0.33 | 2.2–2.7 | 0.9675 ± 0.0002 |

- **iterations:** K-FAC reaches Adam's final training loss in about 120 steps instead of about 207, 1.7 times fewer, and ends 6 times lower (0.013 against 0.082). the curves show that K-FAC is slower than Adam for the first ~50 steps (the KL clip holds its early steps back) and then pulls away steadily.
- **wall-clock:** full K-FAC costs 4–5 Adam steps per step, so it reaches the target 2–3 times *later* than Adam in seconds. the cheap-curvature variant halves the cost per step without losing anything per step, and comes within 1.2–1.7× of Adam (0.23–0.33 s against 0.19–0.28 s here; ms per step × steps gives about 1.2×, and an independent reviewer measured 1.7×): still slower than Adam at this target, but not by much. K-FAC's advantage grows at tighter targets (Adam doesn't reach 0.013 within the run at all).
- **test accuracy:** K-FAC is better, 96.6–96.8% against 96.0%, a difference of about two standard errors with three seeds.
- **empirical Fisher:** on this problem the empirical-Fisher K-FAC is only slightly worse than the true-Fisher K-FAC (0.018 against 0.013 final training loss, the same test accuracy).

**why:** each K-FAC step uses curvature to rescale every layer's update, input whitening by $A^{-1}$ and output-sensitivity scaling by $G^{-1}$, so a single step makes the progress of several Adam steps; but a step costs a second forward and backward pass (sampled labels), the outer products ($785\times785$ from 256 examples for the first layer), and an inverse of that $785\times785$ matrix every 20 steps. on one CPU thread with a batch of 256, that's 4–5 Adam steps' worth of work, and the 1.7× step advantage doesn't cover it. refreshing the curvature less often is nearly free in steps (the factors change slowly, as experiment 1's averaged statistics suggest), and brings the cost to about 2 Adam steps. larger batches, GPUs (where the extra matrix products are cheap relative to memory traffic) and tighter loss targets all move the balance towards K-FAC (Zhang et al., 2019, show K-FAC's advantage over Adam grows with the batch size); tiny batches move it back.

the empirical-Fisher result is milder than topics 18 and 19 might suggest. two plausible reasons, which this lab doesn't isolate: with minibatches of 256 the empirical $G$ is dominated by the examples the network gets wrong, and early in training that's most of them, so it isn't far from the sampled $G$; the KL clip also limits how much a mis-scaled preconditioner can hurt, since it fixes the step's predicted KL. the empirical Fisher's failure modes (topic 19's experiment 2) are about scale, and K-FAC with a KL clip is largely insensitive to scale. that's a reason not to over-generalize either way from one problem.

**caveats for a fair reading.** K-FAC got a 9-point grid and Adam a 4-point grid, so K-FAC had more tuning budget. its winner sits at a corner of the grid; a separate probe (not in the lab) along the ridge where smaller damping pairs with a smaller KL clip (damping $10^{-3}$, KL clip $10^{-5}$ to $3\cdot10^{-4}$) found training losses down to 0.008 but no validation accuracy above 0.964, so the corner isn't hiding a much better K-FAC. in the lab, Adam's $\beta$s and $\epsilon$ were left at their defaults, and neither optimizer used weight decay or a learning-rate schedule. a reviewer's follow-up runs (not reproduced in the lab) closed the tuning-budget gap: Adam on an equal 9-point grid (step size $\{1.5, 3, 5\}\cdot10^{-3}$ × $\beta_2 \in \{0.98, 0.99, 0.999\}$) picked the same validation winner and reached 95.83 ± 0.21% test accuracy over 5 seeds, against K-FAC's 96.63 ± 0.06%; Adam with a linear learning-rate decay to 0 reached 96.01 ± 0.08%. the gap survives at about 3.5 combined standard errors. it's still one network on 20,000 images without weight decay, so treat it as a solid result for this setup rather than a general one. the honest summary is that K-FAC optimizes this network clearly better per step, stays within 1.2–1.7× of Adam per second only once its curvature updates are amortized (and 2–3× behind otherwise), and here also generalized better.
