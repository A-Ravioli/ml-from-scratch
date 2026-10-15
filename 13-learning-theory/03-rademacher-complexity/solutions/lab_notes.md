# rademacher complexity · lab notes

numbers are from the reference solution at the shipped seeds; experiment 3 uses the cached MNIST training set (with `MLFS_OFFLINE=1` it runs on the synthetic stand-in and the numbers differ). Monte Carlo estimates carry errors of a few thousandths.

## experiment 1: how complexity grows with the number of hypotheses

**what people usually predict:** that complexity grows roughly in proportion to the number of hypotheses (or to its log), and that any two classes with the same number of labelings are equally complex.

**what happens:** on $n = 100$ points:

| N | Monte Carlo | Massart | ratio |
|---|---|---|---|
| 1 | −0.0008 | 0 | – |
| 10 | 0.154 | 0.215 | 0.72 |
| 100 | 0.252 | 0.304 | 0.83 |
| 1,000 | 0.322 | 0.372 | 0.87 |
| 10,000 | 0.380 | 0.429 | 0.89 |

from 10 to 10,000 random hypotheses (a factor of 1,000) the complexity grows by a factor of **2.5**, and Massart's bound is within 11–28% of it, getting tighter as $N$ grows. the 200 two-sided thresholds score **0.187**, while 200 random hypotheses score **0.273** (Massart for both: about 0.33).

**why:** each random hypothesis gives a correlation $\sigma^\top h/n$ that's approximately Gaussian with sd $1/\sqrt n$, nearly independent across hypotheses. the expected maximum of $N$ independent standard Gaussians is $\approx\sqrt{2\ln N}$ for large $N$ (minus lower-order terms, which is why the ratio creeps up towards 1). Massart's proof is a union bound that treats the hypotheses as independent, so it's nearly exact for random ones. thresholds are as numerous but far from independent: neighbouring thresholds disagree on a single point, so their correlations with $\sigma$ are built from the partial sums of one random walk, and the best of them is set by that walk's largest excursion, which is of order $\sqrt n$: the complexity is about $C/\sqrt n$ (here $C \approx 1.9$), with no $\sqrt{\ln N}$ factor however many thresholds there are. counting labelings (Massart, VC) can't see this structure; the Rademacher complexity can.

## experiment 2: complexity against norm constraints

**what people usually predict:** that both balls grow with the dimension at a similar rate, or that the dimension doesn't matter for either because the norm is bounded.

**what happens:** with $n = 100$ and random $\pm1$ features:

| d | L2 ball | $\sqrt{d/n}$ | L1 ball | $\sqrt{2\ln 2d/n}$ |
|---|---|---|---|---|
| 1 | 0.080 | 0.100 | 0.080 | 0.118 |
| 10 | 0.309 | 0.316 | 0.187 | 0.245 |
| 100 | 1.001 | 1.000 | 0.275 | 0.326 |
| 1,000 | 3.163 | 3.162 | 0.340 | 0.390 |
| 10,000 | 9.994 | 10.000 | 0.395 | 0.445 |

the L2 ball follows $\sqrt{d/n}$ almost exactly, passing 1 at $d = n = 100$ and reaching 10 at $d = 10{,}000$. the L1 ball grows from 0.19 to 0.40 over the same range, tracking (and staying below) $\sqrt{2\ln(2d)/n}$. at $d = 1$ the two coincide at $0.080 \approx \sqrt{2/(\pi n)}$. (with $n = 400$ on the dropdown every value halves, the $1/\sqrt n$ rate.)

**why:** the L2 ball's complexity is $\frac1n\mathbb E\|\sum_i\sigma_ix_i\|_2$. with $\pm1$ features each of the $d$ coordinates of $\sum_i\sigma_ix_i$ is a sum of $n$ random signs, of size about $\sqrt n$, so the Euclidean norm is about $\sqrt{nd}$ (and concentrates, which is why Jensen's bound is tight here). here $\|x_i\|_2 = \sqrt d$ grows with the dimension, so the bound $BR/\sqrt n$ is "dimension-free" only in the sense that the dimension enters through $R$: fix $\|x\| \le 1$ and the L2 ball stops growing. the L1 ball's dual is the $\ell_\infty$ norm, the *largest* of those $d$ sums, and the maximum of $d$ roughly Gaussian variables grows like $\sqrt{2\ln d}$. that's why $\ell_1$ constraints (the lasso, sparse models) can cope with $d \gg n$ when $\|x\|_\infty$ is bounded: complexity grows logarithmically in the number of features, so you pay for features you might use, not for features you have. at $d = 1$ both balls are $\frac1n\mathbb E|\sum\sigma_i|$, the mean absolute value of a random walk, $\approx\sqrt{2/(\pi n)}$, which Jensen bounds by $1/\sqrt n$.

## experiment 3: fitting random labels

**what people usually predict:** that a model which fits random labels perfectly must overfit badly on real labels, and that logistic regression starts fitting random labels perfectly only when the number of features reaches the number of examples ($k \approx n$).

**what happens:** with 1,000 training digits (even vs odd, test on 2,000 others, $\delta = 0.05$):

| model | random-label correlation | train error | test error | gap | bound |
|---|---|---|---|---|---|
| logistic, k = 5 | 0.061 | 0.204 | 0.213 | +0.010 | 0.394 |
| logistic, k = 10 | 0.070 | 0.180 | 0.179 | −0.001 | **0.379** |
| logistic, k = 50 | 0.177 | 0.123 | 0.157 | +0.034 | 0.429 |
| logistic, k = 200 | 0.387 | 0.000 | 0.182 | +0.182 | 0.516 |
| logistic, k = 400 | 0.629 | 0.000 | 0.183 | +0.183 | 0.757 |
| logistic, k = 500 | 0.914 | 0.000 | 0.180 | +0.180 | 1.043 |
| logistic, k = 593 | **0.999** | 0.000 | 0.180 | +0.180 | 1.128 |
| 1-NN, 784 pixels | **1.000** | 0.000 | **0.071** | +0.071 | 1.129 |

the random-label correlation rises slowly up to $k \approx 100$, then jumps between $k = 400$ and $k = 500$ and reaches 0.999 at $k = 593$ (all the dimensions the 1,000 centred digits span). the bound is tightest at $k = 10$ (0.379, about twice the test error) and passes 1 at $k = 500$. the real-label gap grows from 0 to 0.18 and then stays flat, even as the random-label fit goes from 0.4 to 1. 1-NN fits random labels perfectly (complexity 1, bound 1.13) and has the *lowest* test error of all, 7.1%.

**why:** the jump near $k = 500$ is Cover's theorem in action: $n$ points in general position in $k$ dimensions (with a bias, $k + 1$ parameters) are linearly separable for almost every labeling when $n < 2(k + 1)$ and for almost none when $n > 2(k + 1)$, with a sharp transition at $n = 2(k + 1)$, here $k \approx 499$ (the dotted line). so linear classifiers start "memorizing" at half the sample size, not at $k = n$. the bound $\text{train error} + \hat{\mathfrak R} + 0.13$ is informative only while the class can't fit noise; past $k \approx 500$ it can and the bound is vacuous, exactly as on paper 4 predicts. but a class that *can* fit noise needn't *do* so on real labels: on real digits, the labels are a smooth function of the image, the learner finds that structure, and the gap is set by how far the learned function is from the noise-fitting extremes, which the class-wide supremum doesn't see. 1-NN is the extreme case: it shatters every sample, so every uniform bound is vacuous, yet it generalizes because nearby digits share labels. strictly, these correlations are **lower bounds** on the complexity of linear classifiers (logistic regression isn't an exact 0-1 ERM), so the "bound" column isn't certified; with an exact ERM the numbers could only be higher. the lesson of Zhang et al. (2017) in miniature: uniform complexity measures explain why simple models generalize, and say nothing about why flexible ones do.
