# potential outcomes & weighting

## why this matters

in the 1970s the US ran a randomized trial of a job-training programme, the National Supported Work demonstration: people were randomly assigned to training or not, and their later earnings compared. randomization makes the difference in means an honest estimate of the effect. LaLonde (1986) then did something clever. he threw away the randomized control group, replaced it with comparison people drawn from large national surveys, and asked whether the standard econometric methods of the day, applied to this *observational* data, could recover the experimental answer. mostly they couldn't. estimates scattered widely around the experimental benchmark, often with the wrong sign, and the methods gave no warning about which ones to trust.

the trouble is who signs up. people who enrol in training differ from the general population in age, education, past earnings, and many things nobody recorded. the comparison you want, *the same people* with and without training, never appears in any dataset. this topic builds the language for that missing comparison (**potential outcomes**), states the assumptions under which observational data can stand in for an experiment, and builds the estimators that use them: **inverse propensity weighting**, which reweights the data to look randomized, **outcome regression**, which models what would have happened, and the **doubly robust** AIPW estimator, which combines both so that only one of them has to be right. you'll also see the failure that gives LaLonde's story its bite: when treated and untreated people barely overlap, weighting estimators become wildly unstable, and their error bars don't always tell you.

## warm-up: predict first

1. every unit has two potential outcomes, $Y(0)$ and $Y(1)$. how many of them can you ever observe for one unit?
2. a unit had a 2% chance of being treated, and was treated. in an inverse-propensity-weighted average, how many units does it stand in for?
3. you add a constant 100 to every outcome. does an IPW estimate of the ATE change?
4. you have two models, one for who gets treated and one for the outcome. you know one of them is wrong but not which. can you still get a consistent estimate?
5. propensity scores near 0 or 1: is that a problem for the bias of IPW, its variance, or both?

## prerequisites check

- [ ] you can fit and use `LogisticRegression` and `LinearRegression` from `mlfs.linear` (→ ch 08 · topics 01 and 04)
- [ ] you can compute expectations by iterating: $\mathbb E[Y] = \mathbb E[\mathbb E[Y \mid X]]$ (→ ch 04 · topic 03)
- [ ] you know what a standard error is and how the central limit theorem gives $\bar x \pm 1.96\,s/\sqrt n$ (→ ch 05 · topic 03)
- [ ] you know the backdoor adjustment formula $\mathbb E[Y \mid \mathrm{do}(x)] = \sum_z \mathbb E[Y \mid x, z]P(z)$ (→ ch 29 · topic 01)

## core ideas

### idea 1: potential outcomes, ignorability and the propensity score

give every unit $i$ two numbers: $Y_i(1)$, the outcome it would have under treatment, and $Y_i(0)$, the outcome without (Neyman, 1923; Rubin, 1974). the individual effect is $Y_i(1) - Y_i(0)$. you observe only the one that matches the treatment the unit actually got,

$$Y_i = T_iY_i(1) + (1 - T_i)Y_i(0),$$

which is the **consistency** assumption (together with "no interference": one unit's treatment doesn't change another's outcome, often bundled as SUTVA). the other potential outcome is missing for every unit, always (warm-up 1). Holland (1986) calls this the **fundamental problem of causal inference**. so you give up on individual effects and aim for averages:

$$\text{ATE} = \mathbb E[Y(1) - Y(0)],\qquad \text{ATT} = \mathbb E[Y(1) - Y(0) \mid T = 1].$$

the ATE is the effect of treating everyone versus no one; the ATT is the effect on the people who actually got treated. they differ when the effect varies and the treated are different people.

causal inference is a missing-data problem, and missing data can be handled when it's missing *for reasons you've measured*. two assumptions make that precise:

- **ignorability** (unconfoundedness): $\big(Y(0), Y(1)\big) \perp T \mid X$. within a group of units with the same covariates $X$, treatment is as good as randomly assigned. in graph terms (topic 01), $X$ satisfies the backdoor criterion.
- **positivity** (overlap): $0 < e(x) < 1$ for every $x$, where $e(x) = P(T = 1 \mid X = x)$ is the **propensity score**. every kind of unit could have ended up in either group.

ignorability is untestable from the data; it's a judgement about what you measured. positivity is checkable, and it fails more often than people admit.

the propensity score is a one-number summary that does all of $X$'s work. Rosenbaum & Rubin (1983) showed that if ignorability holds given $X$, it also holds given $e(X)$ alone, because $X \perp T \mid e(X)$: units with the same score have the same covariate distribution in both groups. so you don't need to match on ten covariates, only on one score. in practice $e$ is unknown and you estimate it, usually with a logistic regression of $T$ on $X$.

weighting estimators reuse some units many times over, which costs precision. a good summary of how much is Kish's **effective sample size** of weights $w_i$,

$$\mathrm{ESS} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2},$$

which is $n$ for equal weights, 1 when a single unit carries all the weight, and roughly the number of equally weighted samples that would give the same variance for a weighted mean.

> **spec.** `propensity_scores(X, t, *, alpha=0.0, clip=None)` fits `LogisticRegression(alpha=alpha)` (from `mlfs.linear`) to `(X, t)` and returns `predict_proba(X)[:, 1]`, shape `(n,)`; if `clip` is a number, the scores are clipped to `[clip, 1 - clip]`. `effective_sample_size(w)` returns Kish's ESS as a float.

> **▶ build: stage 1.** implement `propensity_scores` and `effective_sample_size`.

### idea 2: inverse propensity weighting

a treated unit with propensity $e(x) = 0.2$ is one of a kind that is usually *not* treated. among all units like it, only one in five shows up in the treated group, so it should stand in for five (warm-up 2: a unit with $e = 0.02$ stands in for 50). weighting each treated unit by $1/e(X_i)$ rebuilds, from the treated group alone, a pseudo-population that looks like the whole population. the formal version is two lines. by consistency, then iterated expectations and ignorability,

$$\mathbb E\left[\frac{TY}{e(X)}\right] = \mathbb E\left[\frac{T\,Y(1)}{e(X)}\right] = \mathbb E\left[\frac{\mathbb E[T \mid X]\;\mathbb E[Y(1) \mid X]}{e(X)}\right] = \mathbb E\big[\mathbb E[Y(1) \mid X]\big] = \mathbb E[Y(1)].$$

the same argument with $(1 - T)/(1 - e(X))$ gives $\mathbb E[Y(0)]$, so the **Horvitz–Thompson** IPW estimator (Horvitz & Thompson, 1952) is

$$\hat\tau_\text{HT} = \frac1n\sum_i\frac{T_iY_i}{e_i} - \frac1n\sum_i\frac{(1 - T_i)Y_i}{1 - e_i}.$$

it's unbiased when $e$ is the true propensity, but it has a flaw you can see in warm-up 3. the weights $T_i/e_i$ average to 1 only in expectation; in a given sample their sum is not $n$, so adding 100 to every outcome adds $100\big(\frac1n\sum T_i/e_i - \frac1n\sum(1 - T_i)/(1 - e_i)\big)$ to the estimate. the **Hájek** (self-normalized) estimator divides each weighted sum by its own total weight instead of by $n$:

$$\hat\tau_\text{Hájek} = \frac{\sum_i T_iY_i/e_i}{\sum_i T_i/e_i} - \frac{\sum_i(1 - T_i)Y_i/(1 - e_i)}{\sum_i(1 - T_i)/(1 - e_i)}.$$

it's slightly biased in small samples but shift-invariant and almost always less variable.

for the ATT you only need to rebuild the *controls* to look like the treated. the treated keep weight 1; a control unit gets weight $e/(1 - e)$, the odds of treatment, which up-weights controls that look like typical treated units (on paper 2):

$$\hat\tau_\text{ATT} = \frac{\sum_iT_iY_i}{\sum_iT_i} - \frac{\sum_i(1 - T_i)\,Y_i\,e_i/(1 - e_i)}{\sum_i(1 - T_i)\,e_i/(1 - e_i)}.$$

now the problem. the weights are $1/e$ and $1/(1 - e)$. if some units have $e = 0.001$, any one of them that happens to be treated gets weight 1,000, and the estimate is at the mercy of a single noisy outcome. near-violations of positivity don't bias IPW, they make its variance explode (warm-up 5). two common patches are clipping (trimming) the scores at, say, $[0.05, 0.95]$, which bounds the weights but introduces bias, and restricting the analysis to the region of overlap, which changes the question you're answering (Crump et al., 2009).

> **spec.** `ipw_ate(y, t, e, *, normalized=False)` returns $\hat\tau_\text{HT}$, or $\hat\tau_\text{Hájek}$ when `normalized=True`. `ipw_att(y, t, e)` returns $\hat\tau_\text{ATT}$ above. all inputs are arrays of length $n$; return floats.

> **🔮 predict:** in a randomized trial with $e = 0.5$ for everyone, how do the Horvitz–Thompson and Hájek estimates relate to the plain difference in means?

> **▶ build: stage 2.** implement `ipw_ate` and `ipw_att`.

### idea 3: outcome regression (the g-formula)

the other route models the outcomes instead of the treatment. under ignorability, $\mathbb E[Y(1) \mid X] = \mathbb E[Y(1) \mid X, T = 1] = \mathbb E[Y \mid X, T = 1] =: \mu_1(X)$, a regression you can fit on the treated units. so

$$\mathbb E[Y(1)] = \mathbb E[\mu_1(X)],\qquad \hat\tau_\text{reg} = \frac1n\sum_i\big(\hat\mu_1(X_i) - \hat\mu_0(X_i)\big).$$

fit $\hat\mu_1$ on the treated and $\hat\mu_0$ on the controls, then predict *both* for every unit and average the difference. this is Robins' **g-formula**, and it's the backdoor formula of topic 01 with a model in place of stratum means: $\sum_z\mathbb E[Y \mid x, z]P(z)$ becomes $\frac1n\sum_i\hat\mu_x(X_i)$, the average over the empirical distribution of $X$. in the machine-learning literature fitting one model per arm is called a **T-learner**.

outcome regression doesn't divide by anything, so poor overlap doesn't make it explode. instead it fails quietly: where there are no treated units, $\hat\mu_1$ is an *extrapolation*, and its quality depends entirely on whether the model's functional form is right out there. a linear model fitted on treated units with large $x$ will happily predict for untreated units with small $x$, with no warning. IPW is honest about missing overlap (its variance blows up); regression papers over it.

> **spec.** `outcome_models(X, y, t, make_model=None)` fits `make_model()` (default `LinearRegression` from `mlfs.linear`) on the control rows and a second `make_model()` on the treated rows, and returns `(mu0, mu1)`: both models' predictions for **all** $n$ rows, as float arrays.

> **▶ build: stage 3.** implement `outcome_models`.

### idea 4: doubly robust estimation (AIPW)

you now have two estimators, each relying on one model. the **augmented IPW** estimator (Robins, Rotnitzky & Zhao, 1994) uses both. start from regression, then correct it with weighted residuals:

$$\hat\tau_\text{AIPW} = \frac1n\sum_i\psi_i,\qquad \psi_i = \hat\mu_1(X_i) - \hat\mu_0(X_i) + \frac{T_i\big(Y_i - \hat\mu_1(X_i)\big)}{\hat e_i} - \frac{(1 - T_i)\big(Y_i - \hat\mu_0(X_i)\big)}{1 - \hat e_i}.$$

read it two ways. as regression plus correction: if the outcome model is biased, the weighted residuals estimate that bias and remove it. as IPW plus correction: if the propensities are off, the regression term absorbs the error. the bias of the treated half works out to (on paper 3)

$$\mathbb E\left[\frac{\big(e(X) - \hat e(X)\big)\big(\mu_1(X) - \hat\mu_1(X)\big)}{\hat e(X)}\right],$$

a product of the two models' errors. if *either* model is right, one factor is zero and the estimator is consistent: that's **double robustness** (warm-up 4: yes). the product structure also explains its later fame: if each model's error shrinks like $n^{-1/4}$, the product shrinks like $n^{-1/2}$, fast enough for valid confidence intervals even when both models are flexible machine-learning fits. topic 03 builds on exactly this.

the scores $\psi_i$ are (estimates of) the **efficient influence function** of the ATE plus the ATE itself. one practical consequence: $\hat\tau_\text{AIPW}$ is a sample mean of the $\psi_i$, so its standard error is the standard error of a mean,

$$\widehat{\mathrm{se}} = \frac{\mathrm{sd}(\psi_1, \dots, \psi_n)}{\sqrt n},$$

and $\hat\tau \pm 1.96\,\widehat{\mathrm{se}}$ is an approximate 95% interval. "approximate" is doing real work here: the central limit theorem needs the $\psi_i$ to be reasonably well-behaved, and with propensities near 0 or 1 they're heavy-tailed. the lab checks how much that matters.

double robustness is not a free pass. when both models are wrong, the errors multiply rather than cancel, and Kang & Schafer (2007) showed simulated cases where AIPW does worse than either single-model estimator.

> **spec.** `aipw_ate(y, t, e, mu0, mu1)` returns a tuple `(estimate, se)`: the mean of the scores $\psi_i$ above and `std(psi, ddof=1) / sqrt(n)`, as floats.

> **🔮 predict:** if you set `mu0 = mu1 = 0`, what does `aipw_ate` reduce to? and if each outcome model fits its own arm perfectly ($\hat\mu_1 = y$ on the treated, $\hat\mu_0 = y$ on the controls)?

> **▶ build: stage 4.** implement `aipw_ate`.

## worked example

six units with known propensity scores:

| unit | $t$ | $y$ | $e$ | weight | $\hat\mu_0$ | $\hat\mu_1$ | $\psi$ |
|---|---|---|---|---|---|---|---|
| A | 1 | 4 | 0.80 | $1/0.8 = 1.25$ | 1 | 3 | $2 + 1/0.8 = 3.25$ |
| B | 1 | 6 | 0.50 | $1/0.5 = 2$ | 2 | 4 | $2 + 2/0.5 = 6$ |
| C | 1 | 2 | 0.25 | $1/0.25 = 4$ | 1 | 3 | $2 - 1/0.25 = -2$ |
| D | 0 | 1 | 0.80 | $1/0.2 = 5$ | 2 | 4 | $2 + 1/0.2 = 7$ |
| E | 0 | 3 | 0.50 | $1/0.5 = 2$ | 2 | 4 | $2 - 1/0.5 = 0$ |
| F | 0 | 2 | 0.20 | $1/0.8 = 1.25$ | 1 | 3 | $2 - 1/0.8 = 0.75$ |

**Horvitz–Thompson.** the weighted treated outcomes are $5 + 12 + 8 = 25$ and the weighted control outcomes $5 + 6 + 2.5 = 13.5$, so $\hat\tau_\text{HT} = 25/6 - 13.5/6 = 1.9167$.

**Hájek.** the treated weights sum to $7.25$, not 6, and the control weights to $8.25$. so $\hat\tau_\text{Hájek} = 25/7.25 - 13.5/8.25 = 3.4483 - 1.6364 = 1.8119$.

**ATT.** the treated mean is 4. the control odds weights $e/(1 - e)$ are $4, 1, 0.25$ (sum $5.25$), and the weighted control outcomes are $4 + 3 + 0.5 = 7.5$, so $\hat\tau_\text{ATT} = 4 - 7.5/5.25 = 2.5714$.

**ESS** of the treated weights $(1.25, 2, 4)$: $7.25^2/(1.5625 + 4 + 16) = 52.5625/21.5625 = 2.44$. three treated units, worth about two and a half.

**AIPW**, with the outcome models in the table (they predict an effect of exactly 2 everywhere): unit A contributes $2 + (4 - 3)/0.8 = 3.25$, unit D contributes $2 - (1 - 2)/(1 - 0.8) = 7$, and so on. the mean of the six scores is $15/6 = 2.5$, and the standard error is $\mathrm{sd}(\psi)/\sqrt6 = 3.539/2.449 = 1.445$. your code must reproduce all of these.

## on paper

1. **IPW is unbiased.** with the true propensity $e(x)$, show $\mathbb E\big[(1 - T)Y/(1 - e(X))\big] = \mathbb E[Y(0)]$. which assumption does each step use?

<details><summary>solution</summary>

$(1 - T)Y = (1 - T)Y(0)$ by consistency. condition on $X$: $\mathbb E[(1 - T)Y(0) \mid X] = \mathbb E[1 - T \mid X]\,\mathbb E[Y(0) \mid X] = (1 - e(X))\,\mathbb E[Y(0) \mid X]$, by ignorability ($T \perp Y(0) \mid X$). dividing by $1 - e(X)$ needs positivity ($e(X) < 1$). taking the outer expectation gives $\mathbb E[Y(0)]$.

</details>

2. **ATT weights.** show that weighting controls by $e(X)/(1 - e(X))$ makes their covariate distribution match the treated: $p(x \mid T = 0)\,\frac{e(x)}{1 - e(x)} \propto p(x \mid T = 1)$.

<details><summary>solution</summary>

by Bayes' rule $p(x \mid T = t) = P(T = t \mid x)p(x)/P(T = t)$. so $p(x \mid T = 0)\frac{e(x)}{1 - e(x)} = \frac{(1 - e(x))p(x)}{P(T = 0)}\cdot\frac{e(x)}{1 - e(x)} = \frac{e(x)p(x)}{P(T = 0)} = \frac{P(T = 1)}{P(T = 0)}\,p(x \mid T = 1)$. the constant disappears when you normalize the weights.

</details>

3. **double robustness.** write $\hat\mu_1$ and $\hat e$ for fixed (possibly wrong) models and $\mu_1$, $e$ for the truth. show that the treated part of the AIPW score has expectation $\mathbb E[Y(1)] + \mathbb E\big[(e - \hat e)(\mu_1 - \hat\mu_1)/\hat e\big]$, and conclude.

<details><summary>solution</summary>

$\mathbb E\big[\hat\mu_1 + T(Y - \hat\mu_1)/\hat e \mid X\big] = \hat\mu_1 + e\,(\mu_1 - \hat\mu_1)/\hat e$, using $\mathbb E[TY \mid X] = e\mu_1$ (consistency + ignorability). subtract $\mu_1$: $(\hat\mu_1 - \mu_1) + \frac e{\hat e}(\mu_1 - \hat\mu_1) = \frac{(e - \hat e)(\mu_1 - \hat\mu_1)}{\hat e}$. taking expectations over $X$, and using $\mathbb E[\mu_1(X)] = \mathbb E[Y(1)]$, gives the claim. if $\hat e = e$ or $\hat\mu_1 = \mu_1$, the product vanishes. the control part is symmetric.

</details>

4. **weighting is adjustment.** suppose $X$ is discrete and you use the empirical propensity $\hat e(x)$ = the fraction treated among units with $X = x$. show that the Hájek treated mean $\sum_iT_iY_i/\hat e_i \big/ \sum_iT_i/\hat e_i$ equals $\sum_x\hat P(x)\,\bar Y_{x, T=1}$, the backdoor formula of topic 01.

<details><summary>solution</summary>

in stratum $x$ with $n_x$ units of which $n_{x1}$ are treated, $\hat e = n_{x1}/n_x$, so each treated unit gets weight $n_x/n_{x1}$. the stratum's weighted treated sum is $\frac{n_x}{n_{x1}}\sum_{\text{treated in }x}Y_i = n_x\bar Y_{x,1}$, and its total weight is $n_x$. summing over strata: $\sum_xn_x\bar Y_{x,1}/\sum_xn_x = \sum_x\hat P(x)\bar Y_{x,1}$. (Horvitz–Thompson gives the same here, since the total weight is exactly $n$.)

</details>

## experiment

`lab.py` runs your estimators on simulated data with a known ATE:

1. as treated and untreated units overlap less and less, what happens to each estimator's bias and spread, and to the effective sample size?
2. with a nonlinear truth, every combination of right and wrong propensity and outcome models: when is AIPW unbiased, and what happens when both are wrong?
3. how often does the AIPW 95% interval actually cover the truth, at two sample sizes and three levels of overlap?

## check your understanding

1. why isn't the naive difference $\bar Y_{T=1} - \bar Y_{T=0}$ an estimate of the ATE in observational data? what is it an estimate of?

<details><summary>answer</summary>

it estimates $\mathbb E[Y(1) \mid T = 1] - \mathbb E[Y(0) \mid T = 0]$ = ATT + (selection bias) $\mathbb E[Y(0) \mid T = 1] - \mathbb E[Y(0) \mid T = 0]$. the second term is how different the treated would have been *without* treatment. randomization makes it zero.

</details>

2. (the predict callout in idea 2.) with $e = 0.5$ everywhere, how do HT and Hájek relate to the difference in means?

<details><summary>answer</summary>

Hájek is exactly the difference in means (constant weights cancel in each ratio). HT is $\frac2n\sum T_iY_i - \frac2n\sum(1 - T_i)Y_i$, which equals the difference in means only if exactly half the units are treated; otherwise it's off by a factor $2n_1/n$ on the treated side, which is why Hájek is preferred.

</details>

3. ignorability and positivity: which can you check in data, and how?

<details><summary>answer</summary>

positivity: estimate propensities and look at their distribution in each group (histograms, the smallest and largest scores, the effective sample size of the weights). ignorability can't be checked; it's a claim about unmeasured variables. sensitivity analyses ask how strong an unmeasured confounder would have to be to change the conclusion.

</details>

4. why does a perfect propensity model (one that predicts treatment almost exactly) make IPW *worse*?

<details><summary>answer</summary>

if the model separates the groups, scores approach 0 and 1, the weights explode, and the ESS collapses. predicting treatment well means the groups are very different, which is exactly the situation where there's little information about the counterfactual. for weighting you want a model that captures confounders, not one that adds pure predictors of treatment (instruments), which only add variance.

</details>

5. (the predict callout in idea 4.) what does AIPW reduce to with $\hat\mu_0 = \hat\mu_1 = 0$, and with outcome models that fit their own arm perfectly?

<details><summary>answer</summary>

with zero outcome models it's exactly Horvitz–Thompson IPW. with $\hat\mu_1 = Y$ on the treated and $\hat\mu_0 = Y$ on the controls, every residual the correction uses, $Y_i - \hat\mu_{T_i}(X_i)$, is zero, so AIPW collapses to $\frac1n\sum_i(\hat\mu_1(X_i) - \hat\mu_0(X_i))$: pure regression. (each model's predictions for the *other* arm are not $Y$, so this isn't zero.) that's the real danger of overfitting the outcome models on the same data: the residual correction, the part that protects you when the outcome model is wrong, silently switches off, and you lose double robustness without any warning. topic 03 adds cross-fitting for exactly this reason: out-of-fold residuals can't be fitted away.

</details>

6. you estimate the ATT and the ATE on the same data and get different answers. is something wrong?

<details><summary>answer</summary>

not necessarily: if effects vary with $X$ and the treated have different $X$ than the population, the two estimands differ. in the lab's experiment 1 the effect is $2 + x_0$ and treated units tend to have large $x_0$, so the ATT is above 2.

</details>

## common bugs

- **weighting controls by $1/e$.** symptom: wild control means. controls get $1/(1 - e)$ for the ATE and $e/(1 - e)$ for the ATT.
- **the wrong `predict_proba` column.** symptom: scores are $1 - e$ and every estimate has the wrong sign or size. column 1 is $P(T = 1)$ because classes are sorted.
- **fitting one outcome model on everyone without $T$.** symptom: $\hat\mu_1 = \hat\mu_0$ and the regression estimate is 0. fit one model per arm (or include $T$ and its interactions as features).
- **predicting only on each arm's own rows.** symptom: `mu0` and `mu1` have different lengths. the g-formula averages *both* predictions over *all* units.
- **scores of exactly 0 or 1.** symptom: `inf` or `nan` estimates. with strong separation a logistic regression returns probabilities that round to 1; clip, or reconsider the estimand.
- **`ddof=0` in the standard error.** symptom: a tiny mismatch with the checks. use `ddof=1`.

## going deeper

- Rubin (1974), "Estimating causal effects of treatments in randomized and nonrandomized studies", *Journal of Educational Psychology* 66(5). [doi:10.1037/h0037350](https://doi.org/10.1037/h0037350)
- Holland (1986), "Statistics and causal inference", *JASA* 81(396). the fundamental problem. [doi:10.1080/01621459.1986.10478354](https://doi.org/10.1080/01621459.1986.10478354)
- Rosenbaum & Rubin (1983), "The central role of the propensity score in observational studies for causal effects", *Biometrika* 70(1). [doi:10.1093/biomet/70.1.41](https://doi.org/10.1093/biomet/70.1.41)
- Horvitz & Thompson (1952), "A generalization of sampling without replacement from a finite universe", *JASA* 47(260). [doi:10.1080/01621459.1952.10483446](https://doi.org/10.1080/01621459.1952.10483446)
- Robins, Rotnitzky & Zhao (1994), "Estimation of regression coefficients when some regressors are not always observed", *JASA* 89(427). AIPW. [doi:10.1080/01621459.1994.10476818](https://doi.org/10.1080/01621459.1994.10476818)
- Kang & Schafer (2007), "Demystifying double robustness", *Statistical Science* 22(4). when both models are wrong. [doi:10.1214/07-STS227](https://doi.org/10.1214/07-STS227)
- Crump, Hotz, Imbens & Mitnik (2009), "Dealing with limited overlap in estimation of average treatment effects", *Biometrika* 96(1). [doi:10.1093/biomet/asn055](https://doi.org/10.1093/biomet/asn055)
- LaLonde (1986), "Evaluating the econometric evaluations of training programs with experimental data", *American Economic Review* 76(4).
- Hernán & Robins (2020), *Causal Inference: What If*, chapters 1–3 (potential outcomes), 12–13 (IPW and standardization). [free online](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
- Imbens & Rubin (2015), *Causal Inference for Statistics, Social, and Biomedical Sciences*, Cambridge University Press, chapters 12–14 (propensity scores, overlap, trimming).

## connections

- ← builds on: logistic and linear regression (ch 08 · topics 01 and 04), standard errors and the CLT (ch 05 · topic 03), the backdoor formula (topic 01), which on paper 4 shows is the same thing as weighting.
- → used in: double machine learning (topic 03), which plugs flexible models into the AIPW-style orthogonal scores with cross-fitting; the block's project (topic 04), which estimates an effect after discovering the adjustment set.
