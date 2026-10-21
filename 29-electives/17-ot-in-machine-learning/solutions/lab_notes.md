# OT in machine learning · lab notes

numbers are from the reference solution on an Apple M2 laptop CPU (one thread, as in CI); the whole lab takes about 95 s (experiment 1: 15 s, experiment 2: 25 s, the project: 55 s). all data are synthetic and seeded, so the tables reproduce exactly; only timings vary with machine load.

## experiment 1: WGAN revisited

**what people usually predict:** the critic is a rough estimate (WGAN papers warn that it's not a reliable distance); minibatch exact transport is the gold standard; for two samples of the same distribution everything is about 0, and the *debiased* Sinkhorn divergence certainly is.

**what happens** (1,000-point samples; minibatch columns: mean ± sd over 6 pairs of batches of 256):

| generator | exact W1 | WGAN-GP critic (2 seeds) | minibatch W1 | exact W2² | minibatch W2² | your S_ε, ε = 1 |
|---|---|---|---|---|---|---|
| far | 0.691 | 0.705, 0.698 | 0.712 ± 0.018 | 0.481 | 0.513 ± 0.026 | 0.268 ± 0.065 |
| near | 0.216 | 0.181, 0.162 | 0.335 ± 0.084 | 0.070 | 0.268 ± 0.141 | 0.116 ± 0.090 |
| same | 0.065 | 0.037, 0.039 | 0.235 ± 0.091 | 0.035 | 0.262 ± 0.155 | 0.108 ± 0.098 |

with the dropdown (same batches): ε = 0.1 gives 0.516 / 0.270 / 0.263 (far / near / same), ε = 0.3 gives 0.442 / 0.227 / 0.223, ε = 3 gives 0.225 / 0.078 / 0.070.

**why:** the critic does well when the distributions are far apart (0.70 vs an exact 0.691; it can even slightly overshoot, because the gradient penalty only softly enforces the Lipschitz constraint, and only on interpolates) and under-estimates when they're close (0.17 vs 0.216): resolving a 5° rotation needs a finely shaped potential, and 1,500 steps of a small MLP on minibatches don't find it. the exact full-sample values are themselves above the true distances (the "same" pair has $W_1 = 0.065$ between two finite samples of one distribution), and the minibatch values are much further up: at 256 points per batch the sampling floor ($W_1 \approx 0.24$, $W_2^2 \approx 0.26$) is larger than the near pair's whole distance, so minibatch exact transport can't tell "near" from "same" at all, and a generator trained to minimize it is pushed towards a biased optimum (Fatras et al., 2020). the debiased Sinkhorn divergence doesn't fix that: debiasing removes the *entropic* bias ($S_\varepsilon(\mu, \mu) = 0$), not the *sampling* bias between two different minibatches. at ε = 0.1 it is the minibatch $W_2^2$ almost exactly (0.516 / 0.270 / 0.263 vs 0.513 / 0.268 / 0.262); as ε grows it shrinks towards an MMD, whose sampling error decays faster ($n^{-1/2}$, Genevay et al., 2019), so the "far" pair stands out more clearly against the floor (ratio far/same: 2.0 at ε = 0.1, 2.5 at ε = 1, 3.2 at ε = 3), but it also compresses real differences (0.268 for "far" against $W_2^2 = 0.48$) and still can't separate "near" from "same" at this batch size. a training loss doesn't need to *equal* a distance, only to have the right minimizer and useful gradients, which the debiased divergence has in expectation; a *reported* distance needs the floor next to it.

## experiment 2: domain adaptation

**what people usually predict:** OT fixes small rotations and may hurt at large ones, where it maps pieces of one moon onto the other.

**what happens** (5-NN accuracy on the target, mean ± sd over 5 seeds):

| method | 20° | 40° | 60° | 90° |
|---|---|---|---|---|
| source only | 0.888 ± 0.012 | 0.682 ± 0.012 | 0.502 ± 0.013 | 0.269 ± 0.015 |
| OT, exact | 0.971 ± 0.010 | 0.867 ± 0.007 | 0.746 ± 0.016 | 0.524 ± 0.014 |
| OT, ε = 0.05 | 0.994 ± 0.005 | 0.888 ± 0.012 | 0.775 ± 0.010 | 0.534 ± 0.016 |
| oracle (target labels) | 0.999 ± 0.002 | 0.999 ± 0.002 | 0.999 ± 0.002 | 0.999 ± 0.002 |

**why:** in this setup OT never hurts: it beats the source-only classifier at every angle, by 0.08 to 0.26, with small seed spreads. but its gain shrinks with the angle, and at 90° it only climbs from 0.27 (worse than chance: the rotation moved each moon to where the other one was) to 0.52, chance level, against 0.999 with target labels. the cheapest transport onto a strongly rotated target is not the rotation: it moves points the shortest way, which near the moons' tips means onto the *other* moon (the "after" plot at 40° already shows red source points mapped onto the blue moon's end). OT is label-blind; it works when the shift is small relative to the separation between classes. the entropic plan (ε = 0.05) helps a little everywhere: each source point lands on an average of several nearby target points instead of one, which smooths out the sampling noise of an exact one-to-one matching. class-aware costs (Courty et al., 2017; JDOT) or a handful of target labels are the fixes for large shifts.

## experiment 3 (project): OT couplings straighten flow-matching paths

**what people usually predict:** OT pairs help a little at few steps; bigger OT blocks are better at everything; maybe the final samples get better too, since the model "learns the optimal map".

**what happens** (8 Gaussians; per-seed values for training seeds 0, 1, 2; W2 is the exact distance between 1,000 samples and 1,000 held-out points; the floor between two held-out samples is **0.186**):

| coupling | pairing cost | straightness | W2, 1 step | W2, 2 steps | W2, 4 steps | W2, 8 steps | W2, 64 steps | train s |
|---|---|---|---|---|---|---|---|---|
| independent | 6.12 | 1.493, 1.516, 1.529 | 1.80, 1.91, 1.90 | 0.71, 0.70, 0.68 | 0.33, 0.29, 0.33 | 0.32, 0.24, 0.35 | 0.33, 0.22, 0.37 | 2.3 |
| OT, k = 8 | 2.41 | 0.222, 0.221, 0.203 | 0.78, 0.76, 0.75 | 0.38, 0.33, 0.34 | 0.38, 0.28, 0.32 | 0.41, 0.28, 0.34 | 0.43, 0.29, 0.34 | 3.5 |
| OT, k = 32 | 1.40 | 0.053, 0.052, 0.053 | 0.49, 0.44, 0.48 | 0.33, 0.32, 0.36 | 0.34, 0.33, 0.39 | 0.35, 0.34, 0.40 | 0.35, 0.35, 0.41 | 3.3 |
| OT, k = 128 | 1.10 | 0.014, 0.013, 0.011 | 0.37, 0.32, 0.35 | 0.36, 0.28, 0.35 | 0.37, 0.28, 0.37 | 0.38, 0.28, 0.38 | 0.39, 0.28, 0.39 | 7.0 |

paired by seed (W2 differences, negative = better than independent; mean [min, max] over the 3 seeds):

| comparison | 1 step | 2 steps | 64 steps | straightness ratio |
|---|---|---|---|---|
| OT, k = 8 − independent | −1.11 [−1.16, −1.02] | −0.35 [−0.37, −0.33] | +0.04 [−0.03, +0.09] | ×0.143 [0.133, 0.149] |
| OT, k = 32 − independent | −1.40 [−1.47, −1.31] | −0.36 [−0.39, −0.31] | +0.06 [+0.02, +0.13] | ×0.035 [0.034, 0.036] |
| OT, k = 128 − independent | −1.52 [−1.59, −1.43] | −0.37 [−0.42, −0.33] | +0.04 [+0.01, +0.06] | ×0.009 [0.007, 0.009] |

**why:** the coupling does exactly what the theory says to the *paths*. the pairing cost of a batch falls from 6.12 (independent) to 1.10 with blocks of 128, and straightness falls by a factor of 7 (k = 8), 28 (k = 32) and about 110 (k = 128), with tiny seed spreads. the paths panel shows it: independent trajectories head for the centre and then hook towards their modes; OT trajectories are straight lines. with one Euler step, independent samples collapse onto the data mean (W2 = 1.80–1.91), while the k = 128 flow is already within 0.32–0.37, as good as it gets at any number of steps; every seed improves, by 1.43 to 1.59. at two steps the independent flow still trails by 0.33–0.42. that's the project's headline: **minibatch OT couplings buy few-step sampling, in proportion to the block size.**

they don't buy better final samples. at 64 steps the OT-coupled flows are slightly *worse* than the independent one in every seed for k = 32 and k = 128 (+0.01 to +0.13), and all models sit well above the floor of 0.186 (the best: 0.22, independent coupling, seed 1). the seed-to-seed spread at 64 steps (0.22–0.37 for the independent coupling alone) is larger than any difference between couplings, which is why only the paired differences say anything, and three seeds can't establish the sign of a 0.04 effect with confidence. a plausible reason it goes this way: every coupling generates the same distribution in principle (ch 22's on paper 1, with this topic's on paper 1 showing that minibatch pairs are a coupling), so final quality is set by how well the network fits its velocity field, and the OT field is the harder one here. the OT map from a Gaussian onto 8 separated modes is discontinuous (noise on either side of a boundary goes to different modes), and a small MLP smooths the discontinuity, so samples near boundaries land between modes. larger OT blocks also cost training time: 7.0 s against 2.3 s, almost all of it the Hungarian algorithm on blocks of 128.

the limits of this evidence: one 2-D dataset, one network size, 2,000 training steps, three seeds; the metric is a sample-to-sample $W_2$ whose floor (0.186) is a large fraction of every value measured. the published comparisons (Tong et al., 2024; Pooladian et al., 2023) report the same qualitative picture, straighter paths and better low-step quality, on larger problems; they report final quality as good as or better than the independent coupling's, which this lab's small, briefly trained network doesn't reproduce; whether network size, training length or the dataset is responsible is something this lab can't tell.
