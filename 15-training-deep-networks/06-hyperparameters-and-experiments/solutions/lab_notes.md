# hyperparameters & experiments · lab notes

numbers are for real MNIST (standardized pixels), measured with the reference solution: a $784 \to \text{width} \to 10$ ReLU network trained with Adam (batch 32) on 1,000 training images, scored on 2,000 held-out ones. offline (the synthetic stand-in that `verify.py` uses) every configuration reaches 100% validation accuracy with zero spread, so the lab runs but has nothing to say; use the real data.

## experiment 1: two learning rates, ten seeds each

**what people usually predict:** a seed-to-seed spread of a few tenths of a percentage point, and that a single run picks the better learning rate most of the time.

**what happens:**

| learning rate | runs | mean accuracy | std over seeds | range |
|---|---|---|---|---|
| 1e-3 | 10 | 0.9033 | 0.0056 | 0.897 – 0.916 |
| 2e-3 | 10 | 0.9058 | 0.0099 | 0.892 – 0.922 |

the means differ by 0.0025; the seeds of *one* learning rate span 1.9 and 2.9 percentage points. of the 100 possible single-seed comparisons (one run of each), 55% rank the learning rates the same way as the 10-seed means: a coin flip.

**why:** the difference between the configurations is a quarter to a half of one seed's standard deviation, so any single pair of runs is dominated by seed noise. even with 10 seeds each, the standard error of the difference is about $\sqrt{0.0056^2/10 + 0.0099^2/10} = 0.0036$, larger than the difference itself: the honest conclusion is "no detectable difference". for each mean's standard error to be a quarter of the difference (0.000625), you'd need $(0.0056/0.000625)^2 \approx 80$ seeds of the first learning rate and $(0.0099/0.000625)^2 \approx 250$ of the second. and on top of the seed noise, the accuracy of even a fixed model, measured on 2,000 images, has a binomial standard error of $\sqrt{0.9 \cdot 0.1/2000} = 0.0067$ as an estimate of its accuracy on new data. both sources are about the same size, and both are routinely ignored in claims like your colleague's.

## experiment 2: where does the variance come from?

**what people usually predict:** that the initialization dominates (it's "the randomness of the network"), and that data order hardly matters after 3 full epochs.

**what happens:**

| what varies | mean | std | variance (×1e-5) |
|---|---|---|---|
| initialization only | 0.9086 | 0.0039 | 1.56 |
| data order only | 0.8948 | 0.0060 | 3.56 |
| both | 0.9021 | 0.0060 | 3.62 |

the minibatch order alone produces at least as much spread as the initialization alone (here more), and varying both gives about the same spread as varying the order alone, not the sum. note also that the means differ by more than a standard deviation between the rows: the "initialization only" runs all share data order 0 and the "data order only" runs all share initialization 0, and each shared choice happens to be a good or bad draw.

**why:** training is a chaotic dynamical system: tiny differences in the sequence of updates (which examples are seen together, in which order) grow over thousands of steps into different final weights, and the final accuracy inherits that spread. after 3 epochs of 32 steps each, nothing has "averaged out". "both" isn't the sum because variances estimated from 10 runs are themselves very noisy: the standard error of a sample variance from 10 normal draws is about $\sqrt{2/9} \approx 47\%$ of the variance, so 1.56, 3.56 and 3.62 are all compatible with a common value around 3. and the different row means are the same phenomenon at the next level: fixing one source of randomness at one value fixes a random draw that you then can't average over. so when you report a result, vary every source of randomness at once (one seed per run that seeds everything), use several runs, and report the mean with its spread or standard error; and when comparing two configurations, use the *same* seeds for both (a paired comparison) to cancel part of the noise.

## experiment 3: hyperband versus random search, same compute

**what people usually predict:** hyperband, with twice as many configurations, finds a better configuration every time.

**what happens:** the winner's validation loss in the search, and (in parentheses) its mean over 5 fresh seeds at the full budget:

| search seed | hyperband (17 configs, 78 epochs) | random search (8 configs, 72 epochs) |
|---|---|---|
| 0 | lr 2.5e-3, width 128: 0.277 (0.297) | lr 2.7e-3, width 128: 0.277 (0.297) |
| 1 | lr 2.0e-3, width 128: 0.287 (0.295) | lr 4.4e-3, width 128: 0.304 (0.306) |
| 2 | lr 4.1e-4, width 128: 0.285 (0.293) | lr 7.3e-4, width 128: 0.281 (0.287) |

hyperband wins once, ties once and loses once in the search, and after re-training with fresh seeds the two methods' winners are within 0.001–0.011 of each other. every winner looks better in the search than with fresh seeds (by 0.003 to 0.020). all six searches agree on the important part: the widest network and a learning rate between $4 \times 10^{-4}$ and $4 \times 10^{-3}$.

**why:** hyperband's first bracket evaluates 9 configurations after a single epoch and keeps only 3: cheap, which is how it affords 17 configurations, but risky if one-epoch losses don't predict nine-epoch losses (here they mostly do, which is why it never does badly). on a small search space with a broad optimum like this one, 8 random configurations already include something near the optimum, so the extra configurations buy little; hyperband's advantage grows with the size of the space and the ratio of maximum to minimum budget (Li et al. report speedups of an order of magnitude on larger problems). the gap between search loss and fresh-seed loss is the winner's curse: each search picked the configuration whose *seed-0 run* was lowest, which selects for luck as well as quality, and the luck doesn't repeat with new seeds. given experiment 1's spread, differences between winners of 0.01 in loss are within what seeds alone produce; to call one search method better you'd need many repetitions of the whole search, which is exactly how the hyperband paper evaluates it.
