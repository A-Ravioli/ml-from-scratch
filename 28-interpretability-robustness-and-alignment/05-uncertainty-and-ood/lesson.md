# uncertainty & OOD

## why this matters

a model that is wrong 10% of the time is usable if you know *which* 10%. a radiology model that flags its unsure cases for a human, a self-driving stack that slows down when its perception is uncertain, a language model that says "I don't know": all of them need predictions that come with an honest measure of confidence. modern networks are bad at this out of the box. Guo et al. (2017) showed that deep image classifiers had become steadily more accurate *and* more overconfident: a ResNet that says "99%" is right far less than 99% of the time. and confidence on inputs unlike anything in training is worse: networks routinely assign high confidence to noise, to rotated digits, to photos of things they were never taught.

this topic builds three tools, each with a different kind of promise. **temperature scaling** fixes the average miscalibration of a trained classifier with one parameter. **conformal prediction** turns any model's scores into prediction *sets* (or intervals) that contain the truth with a probability you choose, guaranteed, with no assumptions about the model and only one about the data: that calibration and test points are exchangeable. **ensembles** split uncertainty into what's irreducible noise and what's the model's own ignorance, and the second kind is a signal that an input may be **out of distribution**. the lab tests each promise, and finds exactly where each one stops.

## warm-up: predict first

1. a network reaches 100% training accuracy and 42% test accuracy. its average confidence on test images: closer to 42% or to 90%?
2. you divide every logit by the same number $T > 0$. can the model's accuracy change?
3. a conformal method promises 90% coverage. you calibrate it once on 500 images and evaluate on 1,500. what's the chance your measured coverage is below 90%?
4. a model trained on digits sees a photo of a cat. will it be unsure?

## prerequisites check

- [ ] you know cross-entropy / NLL, and the expected calibration error and reliability diagrams (→ ch 07 · 03, `mlfs.metrics`)
- [ ] you know Newton's method in one dimension, and convexity (→ ch 06 · 01–02)
- [ ] you know empirical quantiles and order statistics (→ ch 05 · 01)
- [ ] you know entropy and mutual information (→ ch 05 · 05)
- [ ] you know ROC AUC as the probability that a positive outscores a negative (→ ch 07 · 03)

## core ideas

### idea 1: calibration, and temperature scaling

a classifier is **calibrated** if, among all inputs where it says "70%", 70% are of the predicted class. the expected calibration error (chapter 07) bins predictions by confidence and averages $|\text{accuracy} - \text{confidence}|$ over the bins. networks trained to zero training loss are typically **overconfident**: cross-entropy keeps rewarding larger logits on training points long after they're classified correctly, the logits grow, and the softmax becomes sharper than the test accuracy justifies.

the simplest fix, and in Guo et al.'s comparison one of the best, is **temperature scaling**: divide the logits by a single $T > 0$, chosen to minimise the NLL on a held-out validation set,

$$T^\star = \arg\min_{T > 0} \; \frac{1}{n}\sum_i \Big[\log\sum_c e^{z_{ic}/T} - z_{i y_i}/T\Big].$$

$T > 1$ softens overconfident predictions, $T < 1$ sharpens underconfident ones. dividing by a positive number never changes the argmax, so accuracy is untouched (warm-up 2): only the confidence changes. the problem is one-dimensional and, written in $\beta = 1/T$, **convex** (on paper 1): the NLL's derivative in $\beta$ is $\frac1n\sum_i(\mathbb{E}_{p_i}[z_i] - z_{iy_i})$ and its second derivative is the average variance of the logits under the predicted distribution, never negative. Newton's method on $\beta$, started at $\beta = 1$ and kept positive, converges in a handful of steps. what one parameter can't fix: miscalibration that differs between classes, or between easy and hard inputs.

> **spec.** `fit_temperature(logits, labels, *, max_iters=100)`: NumPy `logits` `(n, k)` and integer `labels` `(n,)` (lists accepted; `ValueError` for mismatched shapes); returns the $T > 0$ (a Python float) that minimises the mean NLL of $\mathrm{softmax}(z / T)$, using at most `max_iters` iterations of an optimizer that starts from $T = 1$.

> **▶ build: stage 1.** implement `fit_temperature`.

### idea 2: split conformal prediction

temperature scaling makes confidences right *on average*. conformal prediction gives a different kind of promise: a set of labels $C(x)$ that contains the true label with probability at least $1 - \alpha$, for any model, any data distribution, any sample size. the price is that you get sets, not single answers, and the promise is *marginal*: averaged over test points and over the calibration data.

**split conformal** needs a **score** $s(x, y)$ that's large when $y$ looks wrong for $x$; for a classifier, $s = 1 - \hat p_y(x)$. compute it on $n$ held-out **calibration** points $(x_i, y_i)$, and let $\hat q$ be the $\lceil (n+1)(1-\alpha) \rceil$-th smallest of the $n$ scores (infinite if that rank exceeds $n$). careful with floating point: $10 \cdot (1 - 0.7)$ is `3.0000000000000004`, and a plain `ceil` turns rank 3 into 4. the prediction set for a new $x$ is every label whose score is at most $\hat q$:

$$C(x) = \{\, y : s(x, y) \le \hat q \,\}.$$

why it works: if the test point is exchangeable with the calibration points, its true score is equally likely to land in any of the $n + 1$ gaps between the sorted calibration scores, so the probability that it falls at or below the $k$-th smallest is $k/(n+1)$, with $k = \lceil (n+1)(1-\alpha) \rceil \ge (n+1)(1-\alpha)$ (on paper 2). for continuous scores the expected coverage is *exactly* $k/(n+1)$, and slightly above $1 - \alpha$; the $+1$ in $n + 1$ is the test point itself. use the exact order statistic, not an interpolated quantile, and an inclusive $\le$.

the guarantee doesn't care how good the model is: a bad model gets big sets, a good one small sets. and it says nothing about individual inputs or classes. some classes can be covered 80% of the time and others 99%, as long as the average is right.

> **spec.** `conformal_quantile(scores, alpha)`: the $\lceil (n+1)(1-\alpha) \rceil$-th smallest of the $n$ scores, a float, or `inf` if that rank exceeds $n$; `ValueError` unless $0 < \alpha < 1$; the input isn't modified. `conformal_prediction_sets(cal_probs, cal_labels, test_probs, alpha=0.1)`: scores $1 - \hat p_{y_i}$ on the calibration set, $\hat q$ from `conformal_quantile`, and a bool array `(n_test, k)` that is True where $1 - \hat p_c \le \hat q$.

> **🔮 predict:** with 99 calibration points and $\alpha = 0.1$, what is the expected coverage, exactly?

> **▶ build: stage 2.** implement `conformal_quantile` and `conformal_prediction_sets`.

### idea 3: conformal regression

the recipe works for any score. for regression with a point predictor $\hat\mu$, use the absolute residual $s(x, y) = |y - \hat\mu(x)|$: $\hat q$ is the conformal quantile of the calibration residuals, and the prediction set is the interval $[\hat\mu(x) - \hat q,\; \hat\mu(x) + \hat q]$. it covers with probability at least $1 - \alpha$ whether $\hat\mu$ is a linear model, a neural net or a deliberately biased guess, and whatever the noise distribution, heavy tails included. a biased or noisy model gets wider intervals; nothing else changes. every interval has the same width, which is the weakness of this score: real uncertainty often varies with $x$, and better scores (dividing by a predicted spread, or conformalising quantile regression; Romano et al., 2019) adapt the width while keeping the guarantee.

> **spec.** `conformal_intervals(cal_pred, cal_y, test_pred, alpha=0.1)`: NumPy arrays; $\hat q$ = `conformal_quantile` of $|y_i - \hat\mu_i|$ over the calibration set; returns `(lo, hi)` = `(test_pred - q, test_pred + q)` (infinite if there are too few calibration points).

> **▶ build: stage 3.** implement `conformal_intervals`.

### idea 4: two kinds of uncertainty, and out-of-distribution inputs

some uncertainty is in the data: a blurry digit that could be a 4 or a 9, a coin flip. more data won't remove it; that's **aleatoric** uncertainty. some is in the model: it hasn't seen inputs like this, so different plausible models would answer differently. that's **epistemic** uncertainty, and it's the kind that should rise on out-of-distribution (OOD) inputs.

a **deep ensemble** (Lakshminarayanan et al., 2017), $M$ networks trained from different random seeds, gives a practical handle on both. with member predictions $p_m(y \mid x)$ and their mean $\bar p$, the entropy of the mean splits into

$$\underbrace{H(\bar p)}_{\text{total}} = \underbrace{\frac1M\sum_m H(p_m)}_{\text{aleatoric}} + \underbrace{H(\bar p) - \frac1M\sum_m H(p_m)}_{\text{epistemic}}.$$

the last term is the mutual information between the prediction and the choice of member (the BALD score of Houlsby et al., 2011): it's zero when all members agree, whatever they say, and large when confident members disagree. on paper 4 you'll show it's never negative.

to use any of these as an OOD detector, score inputs and threshold; to evaluate the detector without choosing a threshold, report the **AUROC** of in-distribution versus OOD scores: the probability that a random OOD input scores higher than a random in-distribution one (ties count half). a single model's $1 - \max_c p_c$ is the classic baseline (Hendrycks & Gimpel, 2017). none of these is a guarantee: a model can be confidently wrong on inputs far from anything it saw, because nothing in training ever told it what to do there.

> **spec.** `ensemble_uncertainty(probs)`: `probs` `(M, n, k)` (`ValueError` for another number of dimensions); returns `{"mean": (n, k), "total": (n,), "aleatoric": (n,), "epistemic": (n,)}` as above, in nats, with $0 \log 0 = 0$. `ood_auroc(scores_in, scores_out)`: ROC AUC with the OOD inputs as positives and higher score = more OOD (ties count 1/2), a float.

> **🔮 predict:** two ensemble members, one says (1, 0), the other (0, 1). what are the aleatoric and epistemic uncertainty?

> **▶ build: stage 4.** implement `ensemble_uncertainty` and `ood_auroc`.

## worked example

**a conformal set.** four calibration points with predicted probabilities of their true class $0.7, 0.8, 0.4, 0.4$: scores $0.3, 0.2, 0.6, 0.6$. with $\alpha = 0.4$, the rank is $\lceil 5 \cdot 0.6 \rceil = 3$, so $\hat q$ is the third smallest score, 0.6. a test point with probabilities $(0.4, 0.45, 0.15)$ gets the set $\{0, 1\}$ (scores 0.6 and 0.55 are $\le 0.6$; 0.85 isn't); a confident test point $(0.9, 0.05, 0.05)$ gets $\{0\}$. with $\alpha = 0.05$ the rank would be $\lceil 4.75 \rceil = 5 > 4$: $\hat q = \infty$ and every set is every class, the honest answer when four points can't support 95%.

**ten scores.** for the scores $0.1, 0.2, \dots, 1.0$ and $\alpha = 0.2$: rank $\lceil 11 \cdot 0.8 \rceil = \lceil 8.8 \rceil = 9$, $\hat q = 0.9$. an interpolated quantile (`np.quantile(s, 0.8)` = 0.82) would give a smaller $\hat q$ and break the guarantee.

**an interval.** calibration predictions $(1, 2, 3, 4)$ with targets $(1.5, 1, 3.2, 6)$: residuals $0.5, 1, 0.2, 2$; at $\alpha = 0.4$ the third smallest is 1, so a prediction of 10 becomes $[9, 11]$.

**the decomposition.** members $(0.5, 0.5)$ and $(0.5, 0.5)$: total = aleatoric = $\log 2 = 0.693$, epistemic 0. members $(1, 0)$ and $(0, 1)$: aleatoric 0, epistemic $\log 2$.

your functions must reproduce all of these.

## on paper

1. **convexity in $\beta$.** with $\beta = 1/T$ and $\ell_i(\beta) = \log\sum_c e^{\beta z_{ic}} - \beta z_{iy_i}$, show that $\ell_i'(\beta) = \mathbb{E}_{p_i}[z_i] - z_{iy_i}$ and $\ell_i''(\beta) = \mathrm{Var}_{p_i}[z_i] \ge 0$, where $p_i = \mathrm{softmax}(\beta z_i)$.

<details><summary>solution</summary>

$\frac{d}{d\beta}\log\sum_c e^{\beta z_c} = \sum_c z_c e^{\beta z_c} / \sum_{c'} e^{\beta z_{c'}} = \mathbb{E}_p[z]$. differentiating again, $\frac{d}{d\beta}\sum_c z_c p_c = \sum_c z_c p_c (z_c - \mathbb{E}_p[z]) = \mathbb{E}_p[z^2] - \mathbb{E}_p[z]^2$. the NLL is an average of convex functions, so it's convex in $\beta$ and Newton's method on $\beta$ has no spurious minima. (in $T$ it isn't convex, which is one reason to work in $\beta$ or $\log T$.)

</details>

2. **the coverage guarantee.** let $S_1, \dots, S_n, S_{n+1}$ be exchangeable with no ties, and $\hat q$ the $k$-th smallest of $S_1, \dots, S_n$. show $P(S_{n+1} \le \hat q) = k / (n+1)$.

<details><summary>solution</summary>

by exchangeability, the rank of $S_{n+1}$ among all $n+1$ scores is uniform on $\{1, \dots, n+1\}$. $S_{n+1} \le \hat q$ exactly when fewer than $k$ of the calibration scores are below $S_{n+1}$, i.e. when its rank among all $n + 1$ is at most $k$. that has probability $k/(n+1)$. with $k = \lceil (n+1)(1-\alpha)\rceil$ this is at least $1 - \alpha$ and less than $1 - \alpha + 1/(n+1)$.

</details>

3. **the spread across calibration sets.** given one calibration set, the coverage on an infinite test set is $F(\hat q)$, where $F$ is the distribution function of the score. show that $F(\hat q) \sim \mathrm{Beta}(k, n + 1 - k)$, and compute its standard deviation for $n = 500$, $\alpha = 0.1$.

<details><summary>solution</summary>

$F(S_i)$ are $n$ independent uniforms, and $F(\hat q)$ is their $k$-th order statistic, which is $\mathrm{Beta}(k, n+1-k)$. for $n = 500$, $k = 451$: mean $451/501 = 0.900$, variance $\frac{k(n+1-k)}{(n+1)^2(n+2)} = \frac{451 \cdot 50}{501^2 \cdot 502} \approx 1.79 \times 10^{-4}$, standard deviation 0.013. a finite test set of 1,500 points adds binomial noise on top (a beta-binomial). so a single calibration split routinely covers 88% or 92%: the guarantee is about the average.

</details>

4. **epistemic uncertainty is never negative.** show that $H(\bar p) \ge \frac1M\sum_m H(p_m)$.

<details><summary>solution</summary>

entropy is concave, so by Jensen's inequality the entropy of the average is at least the average of the entropies. equivalently, the difference is the mutual information $I(Y; M)$ between the label and a uniformly chosen member index, which is a KL divergence and hence $\ge 0$, with equality exactly when all members predict the same distribution.

</details>

## experiment

the lab measures the overconfidence of an MLP that memorised 5,000 CIFAR-10 images and fixes it with your temperature; runs your conformal sets over 1,000 random calibration splits and compares the spread of coverages with the theory, class by class; and trains a five-member ensemble on MNIST to see how well total and epistemic uncertainty flag natural photos and rotated digits as unusual.

## check your understanding

1. temperature scaling reduced the ECE from 0.43 to 0.03. did the model get better?

<details><summary>answer</summary>

its probabilities got better (lower NLL, honest confidences) and its decisions didn't change at all: accuracy, ranking of classes per input, and AUROC per class are identical. calibration is about what the numbers mean, not about which answer is chosen.

</details>

2. why must the temperature be fitted on held-out data and not on the training set?

<details><summary>answer</summary>

on the training set the overfitted model is right and confident almost everywhere, so the NLL there is minimised by $T \le 1$, making it *more* confident. the miscalibration only shows on data the model hasn't memorised.

</details>

3. your conformal sets cover 87% on your test set at $\alpha = 0.1$. is the implementation broken?

<details><summary>answer</summary>

not necessarily: for one calibration set of 500 points the coverage has a standard deviation of about 1.3 points, so 87% is within the normal spread. check the average over many random splits (it should be $\lceil (n+1)(1-\alpha) \rceil/(n+1)$), and check exchangeability: a test set from a different distribution breaks the guarantee.

</details>

4. a conformal predictor at $\alpha = 0.1$ outputs sets of 5 classes on average. what does that tell you?

<details><summary>answer</summary>

that the underlying model is weak: to contain the truth 90% of the time it has to include about half the classes. the coverage is still guaranteed; the set size is where model quality shows up.

</details>

5. why does conformal prediction need the calibration data to be exchangeable with the test data, and what breaks it?

<details><summary>answer</summary>

the proof rests on the test score being equally likely to fall anywhere among the calibration scores. distribution shift (a new hospital, a later time period, adversarial inputs) breaks that, and coverage can fall arbitrarily far. weighted versions handle known covariate shift.

</details>

6. why can a single network be confident on an OOD input, and why might an ensemble do better?

<details><summary>answer</summary>

nothing in training constrains the network far from the data; its logits there are an extrapolation, and ReLU networks' logits grow linearly far from the data, so confidence tends to *increase*. members trained from different seeds extrapolate differently, so they disagree there, which the epistemic term detects. but members can also agree on an OOD input (rotated digits often look like other digits to all of them).

</details>

## common bugs

- **optimising $T$ without keeping it positive**, or in a parameterisation where the problem isn't convex and a fixed-step gradient method diverges. work in $\beta = 1/T$ or $\log T$.
- **an interpolated quantile** (`np.quantile` with default settings) instead of the $\lceil (n+1)(1-\alpha) \rceil$-th order statistic: coverage slightly below the target.
- **$n$ instead of $n + 1$**: the test point counts.
- **a strict inequality** in the set: a label whose score equals $\hat q$ belongs in the set.
- **$0 \log 0$ as NaN** in the entropy of a one-hot prediction.
- **OOD AUROC with the wrong orientation**: decide which set is positive and whether higher means more OOD.

## going deeper

- Guo et al. (2017), "On calibration of modern neural networks", *ICML* (temperature scaling). [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*, Springer (the origin of conformal prediction). [doi:10.1007/b106715](https://doi.org/10.1007/b106715)
- Angelopoulos & Bates (2021), "A gentle introduction to conformal prediction and distribution-free uncertainty quantification". [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
- Lei et al. (2018), "Distribution-free predictive inference for regression", *JASA* (split conformal regression). [arXiv:1604.04173](https://arxiv.org/abs/1604.04173)
- Romano, Patterson & Candès (2019), "Conformalized quantile regression", *NeurIPS*. [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)
- Lakshminarayanan, Pritzel & Blundell (2017), "Simple and scalable predictive uncertainty estimation using deep ensembles", *NeurIPS*. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
- Houlsby et al. (2011), "Bayesian active learning for classification and preference learning" (BALD). [arXiv:1112.5745](https://arxiv.org/abs/1112.5745)
- Hendrycks & Gimpel (2017), "A baseline for detecting misclassified and out-of-distribution examples in neural networks", *ICLR*. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)

## connections

- ← builds on: calibration metrics (ch 07 · 03), Newton's method and convexity (ch 06), order statistics and the bootstrap's way of thinking about sampling variation (ch 05), entropy and mutual information (ch 05 · 05), Bayesian model averaging (ch 12).
- → used in: topic 06's membership-inference attack is a score threshold judged by AUROC, like the OOD detector here; topic 07's evals report every number with an interval; chapter 30's research projects can wrap any model's predictions in conformal sets to report guaranteed coverage.
