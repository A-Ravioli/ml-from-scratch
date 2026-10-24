# evaluating generative models · lab notes

numbers are for real MNIST, measured with the reference solution on an Apple M2 laptop while other jobs were running (load average about 20). the feature network was trained once by `cached_feature_net` on MPS (2 epochs, 73 s, test accuracy 0.974) and then loaded from the cache; every number below is in *its* feature space, so it is comparable with the rest of this chapter and with nothing else. the lab itself ran in 40 s (wall clock) once the network was cached. offline (the synthetic stand-in that `verify.py` uses), the lab runs end to end in under a minute, but its numbers say nothing about MNIST.

## experiment 1: the FID of real data against real data

**what people usually predict:** 0, or "small" (a few units), since the two halves come from the same distribution.

**what happens:** FID between $n$ images from each of two disjoint halves of the test set (mean and standard deviation over 5 random splits):

| n | FID | sd |
|---|---|---|
| 50 | 123.2 | 32.0 |
| 100 | 47.7 | 12.3 |
| 250 | 25.7 | 6.2 |
| 500 | 10.5 | 4.9 |
| 1,000 | 5.6 | 1.9 |
| 2,500 | 2.1 | 0.4 |
| 5,000 | 1.0 | 0.1 |

real data against real data scores an FID of **123 at $n = 50$**, worse than the "model" with three missing digit classes in experiment 2 (about 98). the bias falls roughly like $1/n$ (a straight line of slope $-1$ on the log-log plot) and is still 1.0 at $n = 5{,}000$. `fid_infinity`, fitted on 20 subset sizes from 1,000 to 5,000 images of one half against the full other half, extrapolates to **FID∞ = 0.54 ± 0.25** over five random splits (0.18 to 0.88): well below the biased values, but **not 0**.

**why:** FID is computed from *estimated* means and covariances. two sample means of $n$ points differ by noise of expected squared size $2\operatorname{Tr}(C)/n$, and two sample covariances differ too; with 128-dimensional features and $n = 50$ or $100$, the covariance estimates are not even full rank (rank at most $n - 1$), and the trace term sees large spurious differences. both terms shrink like $1/n$, which is what the fit $a + b/n$ exploits (Chong & Forsyth, 2020). but it only extrapolates the *generated* side: the 5,000-image reference half is fixed, and its own estimation noise stays in the intercept, about half a unit here (a reference half against 10,000 training images scores 1.59, with both sides biased). the $1/n$ law also needs $n$ well above the 128 feature dimensions: fitted on $n$ from 500, the same data gave a misleading intercept near 0, because the small-$n$ points bend the curve. so FID∞ is comparable across models that share the same reference set, and a single fit is noisy enough (±0.25 here) to need several splits. the practical lesson: an FID is meaningless without its $n$, two FIDs at different $n$ can't be compared, and at the $n$ of a typical quick evaluation (a few hundred samples) the bias can be larger than the differences between models.

## experiment 2: two flawed "models"

**what people usually predict:** the noisy model B is worse (the noise is visible, while model A's samples are perfect digits), or the two are about equal.

**what happens:** FID against all 10,000 test images (mean of 3 draws), $\sigma = 0.1$:

| n | model A: no 7, 8, 9 | model B: noisy |
|---|---|---|
| 100 | 117.8 | 49.0 |
| 300 | 101.5 | 31.6 |
| 1,000 | 101.5 | 21.6 |
| 3,000 | 98.4 | 19.8 |
| FID∞ | 97.9 | 19.3 |

FID prefers the noisy model by a factor of five at every $n$. but the ranking depends steeply on $\sigma$: on all 10,000 pool images, model B scores 18.9 at $\sigma = 0.1$, 83.7 at 0.15, **189 at 0.2** and 410 at 0.3, against model A's 98. the two swap places at about $\sigma = 0.16$.

**why:** FID measures distance in *this network's* feature space, so it weighs failures by how much they move the features. missing classes shift the feature mean and remove whole directions of variance, a large and $\sigma$-independent penalty. pixel noise is something the network never saw in training (it was trained on clean digits): mild noise barely moves its features, but beyond a threshold it pushes every feature vector off the clean-data manifold and the penalty explodes. a network trained with noise augmentation would rank these models differently. and a single number can't say *which* failure a model has: an FID of about 100 could be model A, or model B at $\sigma \approx 0.16$.

## experiment 3: precision and recall

**what people usually predict:** A has high precision and recall around 0.7 (30% of the classes are missing); B has lower precision (noisy samples look unreal) but full recall; C has high precision and very low recall.

**what happens** ($k = 3$, 3,000 points per set; mean ± sd over 5 random subsets):

| model | precision | recall | FID (one draw) |
|---|---|---|---|
| A: no 7, 8, 9 | 0.910 ± 0.006 | 0.742 ± 0.039 | 99.3 |
| B: noisy, $\sigma = 0.1$ | 0.933 ± 0.007 | 0.754 ± 0.005 | 21.1 |
| C: only 1s | 0.918 ± 0.013 | 0.121 ± 0.002 | 1,098 |
| real training images | 0.911 ± 0.004 | 0.907 ± 0.006 | 1.4 |

model C's collapse is unmistakable in recall (0.12). but models A and B, whose FIDs differ by a factor of five, get almost **the same precision and recall**: A's missing 30% of the data costs only 0.17 of recall relative to real images, and B's noise costs about as much recall and no precision at all. even real images score a precision and recall of about 0.91, not 1.

**why:** the $k$-NN balls are estimates of the manifold, and they're generous. A's samples include 1s, 4s and other digits that share strokes with 7s and 9s, and the balls around A's points reach into the regions where real 7s and 9s live, so many real 7–9s still fall inside some ball: recall overestimates coverage when the missing modes sit near present ones in feature space. B's noise shifts every feature vector slightly in a consistent direction rather than scattering it, so B's points stay inside real balls (precision unchanged) while some real points fall outside B's balls (recall down). and with finite samples, even two draws from the same distribution miss each other's tails, which is why real data scores 0.91 rather than 1. precision and recall separate fidelity from coverage only as well as the feature space and $k$ allow; read them next to FID, never instead of it.

## references

- Chong & Forsyth (2020), "Effectively unbiased FID and Inception Score and where to find them", *CVPR*. [arXiv:1911.07023](https://arxiv.org/abs/1911.07023)
- Kynkäänniemi, Karras, Laine, Lehtinen & Aila (2019), "Improved precision and recall metric for assessing generative models", *NeurIPS*. [arXiv:1904.06991](https://arxiv.org/abs/1904.06991)
