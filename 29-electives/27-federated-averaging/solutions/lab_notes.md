# federated averaging · lab notes

numbers are from the reference solution on the cached MNIST data, one CPU thread (the lab takes 70–115 s depending on machine load). with two seeds per setting, read differences of less than about ±0.01 in final accuracy as noise; the per-seed values are in each table. on the synthetic offline stand-in that `verify.py` uses, the numbers differ but the lab runs the same way.

## experiment 1: local steps vs communication rounds

**what people usually predict:** E = 20 does 20× the local work, so it should need roughly 20× fewer rounds; or, the opposite intuition, that local overfitting makes more epochs useless. on the non-IID shards, most predict that more local work makes things worse because each client drifts toward a 2-digit classifier.

**what happens:**

| split | E | rounds to target (per seed) | final accuracy after 25 rounds (per seed) |
|---|---|---|---|
| IID (target 85%) | 1 | 25, >25 | 0.847 (0.851, 0.844) |
| IID | 5 | 8, 8 | 0.884 (0.883, 0.886) |
| IID | 20 | 3, 5 | 0.891 (0.890, 0.892) |
| shards (target 70%) | 1 | 21, 17 | 0.711 (0.674, 0.747) |
| shards | 5 | 12, 7 | 0.760 (0.716, 0.804) |
| shards | 20 | 12, 7 | 0.777 (0.732, 0.822) |

on IID data, going from 1 to 5 local epochs cuts the rounds to 85% by about 3× (≥25 → 8), and 20 epochs cuts them by 5–8× (to 3–5). that's a big saving but far from 20×. the striking picture is the top-right panel: plotted against **total local steps**, the three IID curves lie nearly on top of each other (E = 20's first point, after a single round of 200 steps per client, sits about 5 points below the others at the same step count; after that they coincide). while the clients agree, a FedAvg round with $u$ local steps is worth about $u$ steps of centralized SGD (each client's trajectory is a nearly unbiased sample of the same path), so more local work moves you further along the same curve per round. the returns diminish once the model is near the accuracy this small training set (10,000 images, a 64-unit MLP) supports: E = 5 and E = 20 both end at 0.88–0.89.

on the shards, local work still helps per round (E = 1 is the slowest by far) but less, and the seed-to-seed spread is huge (0.73 vs 0.82 at E = 20). the curves zig-zag because each round's 10 clients cover only a few digits and the average model is pulled toward whatever those happened to be. per local step (bottom right), E = 1 is now the most efficient: the extra local steps are partly wasted pulling clients toward their own 2-digit optima.

**why:** averaging two-digit experts works because every client starts each round from the *same* global model and moves only a limited distance, so their updates are compatible displacements from a shared point rather than independent models; the average of the displacements is a reasonable step for the union of their data. McMahan et al. (2017) report the same surprise: on their pathological non-IID MNIST, more local computation still reduced the rounds needed. topic 28 shows where this breaks: with enough local steps and enough heterogeneity, the average converges to the wrong point (client drift), and you can compute exactly how wrong. the right axis for a phone on a slow uplink is rounds (or bytes); local steps are almost free.

## experiment 2: how many bits does an update need?

**what people usually predict:** a few percent lost at 8 bits, a lot lost at 2 bits, and 1 bit "can't possibly work".

**what happens:** (IID, E = 5, 25 rounds, two seeds)

| bits | MB uploaded in 25 rounds | MB uploaded to reach 85% (per seed) | final accuracy (per seed) |
|---|---|---|---|
| 32 (none) | 50.89 | 16.28, 16.28 | 0.884 (0.883, 0.886) |
| 8 | 12.72 | 3.56, 3.56 | 0.878 (0.878, 0.877) |
| 4 | 6.36 | 1.78, 1.78 | 0.879 (0.881, 0.878) |
| 2 | 3.18 | 0.89, 1.02 | 0.879 (0.882, 0.876) |
| 1 | 1.59 | 0.70, 0.83 | 0.866 (0.868, 0.863) |

8, 4 and 2 bits are within about half a point of full precision, for 4×, 8× and 16× less upload. even 1 bit (every entry becomes the update's min or max, at random) loses less than 2 points, for 32× less traffic. per megabyte, the low-bit runs win by a wide margin: float32 FedAvg reaches 85% in 8 rounds, about 16 MB of uploads; at 2 bits it gets there after about 1 MB.

**why:** stochastic rounding is unbiased, so the quantized update is the true update plus zero-mean noise, and the server averages 10 such independent noises (variance down 10×). SGD already tolerates plenty of zero-mean noise in its steps: that's what a minibatch gradient is. the cost of quantization is variance, not bias, and variance only slows convergence a little. the error per entry is proportional to the update's range $\max\Delta - \min\Delta$, so a single huge entry (an outlier coordinate, or a layer with much larger updates than the rest) makes the grid coarse for everyone. the fixes are to quantize per layer or per block with its own range, or to apply a random rotation first, which spreads the outliers' energy evenly across coordinates (Suresh et al., 2017; Konečný et al., 2016).

## experiment 3: how many clients per round?

**what people usually predict:** 5× the clients is 5× the data per round, so close to 5× faster.

**what happens:** (IID, E = 5, one seed)

| C | clients per round | rounds to 85% | final accuracy | MB uploaded |
|---|---|---|---|---|
| 0.01 | 1 | 20 | 0.861 | 5.1 |
| 0.05 | 5 | 9 | 0.877 | 25.4 |
| 0.1 | 10 | 8 | 0.883 | 50.9 |
| 0.5 | 50 | 7 | 0.886 | 254.4 |

one client per round is clearly worse (it's sequential training on a 100-image shard per round, and each round's step is noisy), but beyond 5 clients the curves are nearly indistinguishable: 50 clients reach 85% in 7 rounds against 8 for 10 clients, at 5× the bytes. per megabyte, the small cohorts win.

**why:** averaging more client updates reduces the variance of the round's step but doesn't make the step longer or better aimed; once the variance is small next to the step itself, extra clients buy nothing. McMahan et al. found the same: little gain beyond $C = 0.1$ in their MNIST experiments. this changes completely in topic 29: with differential privacy the server adds noise whose size is fixed per round, and the only way to make it small relative to the signal is to average over many more clients. there, a large cohort is what makes private training work at all.
