# experimental rigor · lab notes

numbers below are from the reference solution on an Apple M-series laptop (CPU, `OMP_NUM_THREADS=1`), real MNIST from the cache: 5,000 training images drawn from the first 50,000, the 10,000-image test set, a $784 \to 128 \to 10$ MLP trained for 3 epochs with SGD (learning rate 0.05, momentum 0.9, batch 64). one run takes about 0.12 s; the whole lab about 30 s. torch on CPU is deterministic, so the same seeds give the same numbers on the same machine; another machine may differ in the last digits. offline the lab runs on a 2,000-image synthetic stand-in and its numbers mean nothing.

## experiment 1: where does the variance come from?

**what people predict.** that initialization dominates ("the seed" usually means the init in people's heads), or that data order is negligible. and that "all three" is roughly the sum of the three, or at least clearly bigger than any one.

**what happens.** with 20 seeds per source:

| source | mean accuracy, 95% CI (%) | sd of one run (points) | max − min (points) |
|---|---|---|---|
| init only | 93.14 [92.88, 93.41] | 0.57 | 2.43 |
| order only | 92.54 [92.04, 93.04] | 1.07 | 4.11 |
| subset only | 92.64 [92.35, 92.93] | 0.62 | 2.08 |
| all three | 92.96 [92.66, 93.26] | 0.64 | 2.47 |

every single source on its own spreads test accuracy over about 2–4 points, and "all three together" is *not* bigger than the largest single source. the order-only column's larger sd comes from two bad runs (90.4% and 89.5%). to check whether that's a property of data order or luck, the same measurement with 60 seeds per source (run separately with the lab's `train_mlp`) gives sds of 0.46 (init), 0.81 (order), 0.87 (subset) and 0.75 (all three), and the order-only sd computed on three disjoint blocks of 20 seeds is 1.07, 0.51 and 0.66. so the ranking of the sources in the 20-seed table is mostly noise: the runs have heavy tails (an occasional bad run), and an sd estimated from 20 heavy-tailed values is uncertain by much more than the ±16% the normal-theory formula $\mathrm{sd}/\sqrt{2(n - 1)}$ gives.

**why.** the sources don't add like independent noise terms. training is chaotic: any perturbation, whether a different first minibatch, a different initial weight, or a different handful of training images, sends the optimizer to a different, roughly equally good solution, and the spread of "equally good solutions" is the variance you see. one source is enough to sample that whole distribution; more sources don't widen it much. this is what Summers & Dinneen (2021) found for larger image classifiers, down to a one-bit change of the initialization. the practical consequences: (1) fixing a seed doesn't remove variance, it hides it, and a reader running your code with any other seed (or on other hardware) will see it; (2) to estimate the variance you need many runs, and more if the metric has heavy tails; (3) the init-only intervals here (±0.27 points for the mean at 20 seeds) are what a careful single-source study would report, while a reader who changes the data subset gets the wider spread too: vary every source across seeds.

## experiment 2: the single-seed ablation

**what people predict.** that most single-seed tables agree with the 20-seed table, because the three components have clearly different effects (most guess 15–18 of 20).

**what happens.**

| config | mean ± 95% CI | n | Δ vs full ± 95% CI |
|---|---|---|---|
| **full** | **0.930 ± 0.003** | **20** | **(baseline)** |
| no momentum | 0.884 ± 0.013 | 20 | -0.046 ± 0.012 |
| no standardization | 0.916 ± 0.002 | 20 | -0.014 ± 0.003 |
| width 32 | 0.908 ± 0.005 | 20 | -0.022 ± 0.005 |

the 20-seed order of importance is no momentum > width 32 > no standardization, and every interval excludes 0. only **11 of 20** single-seed tables rank all three the same way; 18 of 20 get the most important one (momentum) right. the confusion is between width 32 and no standardization: the paired difference between those two ablations is +0.79 points (bootstrap 95% CI [+0.35, +1.27], permutation p = 0.003, `compare_runs` on the 20 shared seeds), well resolved at 20 seeds, but small next to one run's sd of 0.6–1.0 points.

**why.** a single-seed table compares single runs, and each run carries the full seed-to-seed noise of experiment 1 (sd ≈ 0.6–1 points). two ablations whose true effects differ by 0.8 points are confused often; one whose effect is 4.6 points almost never. note what the paired intervals buy: the "no standardization" row's own interval (±0.2 points) is narrower than the full model's, and its Δ (±0.3) is far narrower than the 1-point sd of a single run, because the seed's effect cancels in the difference. a paper that ran each row once should say "we ran one seed per configuration; differences under about 1–2 points are within seed noise" (or, better, run more seeds for the rows that matter).

## experiment 3: who wins depends on the tuning budget

**what people predict.** one optimizer is simply better; the curves are parallel. some expect SGD to win everywhere ("well-tuned SGD beats Adam"), others Adam ("Adam just works").

**what happens.** at $k = 1$ (a random learning rate from the search space) the expected best is 72.8% for Adam and 70.5% for SGD: many learning rates on a log scale from $10^{-4}$ to 1 fail for both. at 3 trials Adam leads by **+1.74 points** (90.3% vs 88.6%; 90% bootstrap bands over the trials [87.2, 91.6] and [84.6, 90.8]). SGD's curve passes Adam's at **$k = 6$**, and at 30 trials SGD leads by **1.20 points** (93.6% [92.9, 93.7] vs 92.4% [92.2, 92.5]). the unfair comparison (SGD with 5 trials, Adam with 30) has Adam "winning" by +0.91 points; at equal budgets of 30, it loses by 1.2.

**why.** Adam is less sensitive to the learning rate: 15 of its 30 trials reach 90% validation accuracy, against 10 of SGD's. a few random draws are therefore more likely to hit a good Adam setting, which is what the left end of the curves measures. but the best SGD setting in this search (learning rate 0.035 with momentum 0.9) beats the best Adam setting (0.0037) by about a point at this short budget, and with enough trials random search finds it. neither optimizer is "better": the answer depends on the budget, which is why Dodge et al. (2019) ask for the whole curve. the comparison "ours (tuned) vs baseline (defaults)" is a comparison of a point near the right end of one curve with a point near the left end of the other. two caveats belong in any write-up of this: the curves come from 30 trials each, so their right ends are uncertain (the bands), and the result depends on the search space (a log-uniform learning rate over four decades) and on the 2-epoch budget per trial.
