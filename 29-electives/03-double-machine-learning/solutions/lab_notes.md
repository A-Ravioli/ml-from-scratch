# double machine learning · lab notes

numbers below are from the reference solution with the lab's seeds. experiments 1 and 2 are fully simulated; experiment 3's numbers come from the **real** California housing covariates (the cached `california_housing.csv`). offline (`MLFS_OFFLINE=1`, as in `verify.py`) the loader returns a synthetic stand-in and experiment 3 gives different, though qualitatively similar, numbers. your own ch 08/10 models give similar numbers if they match the contracts.

## experiment 1: regularization bias

**what people usually predict:** that the naive plug-in and DML agree, since they use the same lasso; or that DML is a minor refinement. many expect OLS to be biased because $p$ is so close to $n$.

**what happens:** at lasso penalty $\alpha = 0.1$, over 80 datasets (bias ± its standard error):

| estimator | bias | sd |
|---|---|---|
| OLS on $(d, x)$ | −0.085 ± 0.029 | 0.258 |
| naive plug-in (lasso) | **+0.176 ± 0.008** | 0.068 |
| DML (lasso, 5 folds) | +0.011 ± 0.009 | 0.079 |

and across penalties (30 datasets each except $\alpha = 0.1$):

| lasso $\alpha$ | naive bias | DML bias | DML 95% CI coverage |
|---|---|---|---|
| 0.05 | +0.065 | +0.018 | 93% |
| 0.1 | +0.176 | +0.011 | 94% |
| 0.2 | +0.426 | +0.081 | 87% |
| 0.4 | +0.830 | +0.171 | 27% |

with 30 datasets, each coverage figure is uncertain by about ±5 to ±8 points. the naive estimate is off by 35% of the true effect at the default penalty, with a tight spread that makes it look trustworthy, and its bias grows almost linearly with $\alpha$. DML is within about one standard error of the truth at small and moderate penalties and degrades only when the lasso becomes very poor.

OLS is unbiased: on 1,000 further datasets (the lab prints this line) its bias is **−0.001 ± 0.008** with sd 0.243. the −0.085 on the 80 shown datasets is a 2.9-standard-error fluke of a very wide distribution. OLS's sd is about 3 times DML's: with 180 controls and 200 units it spends almost all its degrees of freedom estimating nuisance coefficients.

**why:** the lasso shrinks the confounders' coefficients towards zero, so $\hat g$ misses part of $3x^\top\beta$, and the missing part is exactly the part correlated with $d = 1.5x^\top\beta + v$. the naive score attributes that leftover confounding to $d$: bias $\approx \mathbb E[m\,\delta_g]/\mathbb Ed^2$, first order in the lasso's error (on paper 3). DML partials the lasso fit out of *both* $y$ and $d$; what remains is the product of the two lasso errors, which is small while each is moderately small. at $\alpha = 0.4$ both lassos shrink so hard that the product is no longer negligible, and DML's intervals stop covering: orthogonality relaxes the accuracy you need from the nuisance models; it doesn't remove it.

## experiment 2: why cross-fit?

**what people usually predict:** more flexible trees fit better, so both estimates improve, and the most flexible trees give the best answer; cross-fitting matters a little.

**what happens:** 20 datasets of $n = 400$ per setting (truth 0.5; the cross-fitted means have standard errors of about 0.015–0.027):

| min per leaf | in-sample | cross-fitted | in-sample mean of $(d - \hat m)^2$ | out-of-fold mean of $(d - \hat m)^2$ |
|---|---|---|---|---|
| 50 | 1.367 ± 0.095 | 1.373 ± 0.121 | 1.761 | 2.086 |
| 20 | 1.040 ± 0.106 | 1.133 ± 0.116 | 1.379 | 1.967 |
| 10 | 0.784 ± 0.118 | 0.902 ± 0.098 | 1.046 | 1.871 |
| 5 | 0.561 ± 0.097 | 0.681 ± 0.107 | 0.685 | 1.962 |
| 2 | **0.222 ± 0.084** | **0.510 ± 0.087** | 0.220 | 2.246 |
| 1 | undefined (every residual is 0) | 0.461 ± 0.061 | 0.000 | 2.451 |

(± is the sd over the 20 datasets.) heavily regularized trees ruin both versions (estimates near 1.37, almost three times the truth). as the trees grow, the cross-fitted estimate falls towards the truth and is best at **2 per leaf** (bias +0.010 ± 0.019). fully grown trees overshoot to **0.461**, a bias of −0.039 ± 0.014, about 2.9 standard errors below the truth. the in-sample estimate passes the truth by accident around 5 per leaf, keeps falling to 0.22, and is undefined for fully grown trees.

**why:** the true noise variance is $\mathrm{Var}(v) = 1$.
- **in-sample**, the residual variance falls to 0.69, 0.22 and finally 0: the trees memorize the noise, so the in-sample residuals are a shrunken, distorted version of $v$ (on paper 4's leaky smoother, at its extreme). that bias comes from overfitting, not from the models being wrong, and no amount of data removes it.
- **out-of-fold**, the residual variance is never close to 1: it's 1.87–2.45, so $\mathbb E[\delta_m^2]$ is roughly 0.9–1.5 at every setting. what changes is the *kind* of error. on paper 3 the bias is $\mathbb E[\delta_m\delta_g]/(\mathbb Ev^2 + \mathbb E\delta_m^2)$ with $\delta_g = \delta_\ell - \theta\delta_m$. a shallow tree's error is *systematic* (it can't represent $\sin 2x_0$ or $x_1^2$) and the same shape in $\hat\ell$ and $\hat m$, so $\delta_m$ and $\delta_g$ line up and the bias is large and positive. as the trees grow, the systematic part shrinks and the error becomes mostly *noise*: each prediction copies a few neighbours' $d$ (or $y$) values, noise included.
- **why one per leaf attenuates.** noise in $\hat m$ enters $\delta_\ell$ only if $\hat\ell$ copies the *same* neighbours, scaled by $\theta$. the two trees are grown on different targets ($y$ and $d$), so they split differently and their leaves hold different units; the $-\theta\,\delta_m$ part of $\delta_g$ then isn't cancelled, and its contribution $-\theta\,\mathbb E[\delta_m^2]$ to the numerator pulls the estimate towards zero. that's the classic attenuation of a regression on a noisy regressor. single-unit leaves maximise that noise (the out-of-fold error is largest at 2.45), so they attenuate most; two units per leaf halve the noise while staying flexible, which is why that's the best setting.
- **shared folds matter for the same reason.** an earlier version of `dml_plr` drew separate fold partitions for $\hat\ell$ and $\hat m$, which decorrelates their errors further; it gave 0.436 at 1 per leaf and 0.475 at 2. sharing one partition per repetition (as Chernozhukov et al. and DoubleML do) gives 0.461 and 0.510. at the regularized settings, where the errors are systematic and positively aligned, sharing makes the estimates about 0.03 *higher*: the product term is estimated honestly in both directions.

cross-fitting is what lets you use flexible models at all; a nuisance error that is systematic (too much regularization) or very noisy (too little) still costs you.

## experiment 3: a semi-synthetic study

**what people usually predict:** DML with linear nuisances does at least as well as OLS (it's "doubly robust"); the boosted DML has close to 95% coverage.

**what happens:** 10 replicates of 1,500 real districts each (of 20,433 available):

| estimator | bias | sd over replicates | mean reported se | 95% CI coverage |
|---|---|---|---|---|
| naive difference | +1.267 | 0.122 | — | — |
| OLS on $(d, x)$ | −0.293 | 0.070 | — | — |
| DML-IRM, linear + logistic | −0.321 | 0.058 | 0.090 | 0 of 10 |
| DML-IRM, gradient boosting | **−0.033** | 0.084 | 0.085 | 9 of 10 |

the naive difference is off by more than the effect itself (the ATE is about 1.0). OLS and linear DML are both off by about 0.3, in the same direction, with intervals that never cover. boosted DML is within half a standard error of the truth on average, and its reported se (0.085) matches the actual spread across replicates (0.084).

**why:** double robustness needs *one* of the nuisance models to be right. the outcome depends on $\sin(2\,\text{lat})$, $\text{age}^2$ and an interaction, and the propensity on $\text{age}^2$ and $\tanh(\text{lat})$: neither a linear regression nor a logistic regression on the raw covariates can represent them, so the product of errors is large and linear DML has no advantage over OLS. gradient boosting learns both nonlinear nuisances well enough for the product to be small. the price is variance (sd 0.084 against 0.058 for linear DML), a good trade when the alternative is a bias four times larger. with 10 replicates the coverage of 9 of 10 is consistent with anything from about 55% to 100% (an exact binomial 95% interval), so this run shows the intervals are roughly calibrated, not that they're exactly 95%; a proper simulation study would use hundreds of replicates. the same numbers hold for the synthetic offline stand-in qualitatively: naive +1.14, OLS −0.31, linear DML −0.38, boosted DML −0.03.
