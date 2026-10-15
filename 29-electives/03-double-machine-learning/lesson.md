# double machine learning

## why this matters

a retailer wants the effect of a discount $d$ on weekly sales $y$. discounts weren't randomized: managers set them based on season, region, stock levels, competitor prices and a hundred other recorded variables $X$, which also drive sales. topic 02 says what to do: adjust for $X$. but *how*? a linear regression of $y$ on $d$ and $X$ assumes every confounder enters linearly. you know machine learning, so the next idea is obvious: use a flexible model to soak up $X$'s effect, then read off the discount effect.

that obvious idea is subtly broken. flexible models are *regularized*: a lasso shrinks coefficients, a random forest averages over leaves, boosting stops early. regularization trades a little bias for a lot less variance, which is the right trade for prediction. but in an effect estimate the bias in $\hat g(X)$ doesn't wash out. it lands directly in $\hat\theta$, and it shrinks more slowly than the standard error, so with more data your confidence interval tightens around the *wrong* number. the lab shows a lasso-based plug-in estimate off by a third of the true effect, with a spread small enough to make that look like a precise answer.

**double machine learning** (Chernozhukov et al., 2018) fixes this with two ideas. **orthogonalization**: estimate the effect with a score that is insensitive, to first order, to errors in the machine-learning fits, so that their biases enter only as a product of two small errors. **cross-fitting**: never let a unit's nuisance prediction come from a model that saw that unit. together they let you plug almost any model you built in part II (lasso, forests, boosting) into an effect estimate and still get honest confidence intervals.

## warm-up: predict first

1. you regress $y$ on $d$ and 50 controls with OLS. then you regress $y$ on the controls, keep the residuals, regress $d$ on the controls, keep those residuals, and regress the first residuals on the second. do you get the same coefficient on $d$?
2. a lasso with a well-chosen penalty predicts $y$ from $X$ nearly perfectly. you use it as $\hat g$ and estimate $\theta = \sum d(y - \hat g)/\sum d^2$. biased or not?
3. you fit a fully grown decision tree for $m(X) = \mathbb E[d \mid X]$ and use its *in-sample* residuals $d - \hat m(X)$. what do those residuals look like?
4. DML splits the data into 5 folds. does that cost you 80% of your sample size in precision?
5. two nuisance models each have an error that shrinks like $n^{-1/4}$, much slower than the parametric $n^{-1/2}$. can the effect estimate still converge at $n^{-1/2}$?

## prerequisites check

- [ ] you can fit `Lasso`, `Ridge` and `LinearRegression` from `mlfs.linear`, and know what the lasso penalty does to coefficients (→ ch 08 · topics 01–02)
- [ ] you can fit `DecisionTreeRegressor`, `GradientBoostingRegressor` and `GradientBoostingClassifier` from `mlfs.trees` (→ ch 10 · topics 01 and 04)
- [ ] you know K-fold splitting and why you evaluate on held-out data (→ ch 07 · topic 02)
- [ ] you know the AIPW estimator and its doubly robust bias $\mathbb E[(e - \hat e)(\mu - \hat\mu)/\hat e]$ (→ ch 29 · topic 02)
- [ ] you know the sandwich (heteroskedasticity-robust) idea for standard errors, or are happy to derive it here (→ ch 05 · topic 03)

## core ideas

### idea 1: partialling out, and why you cross-fit

the **partially linear model** (PLR) keeps the effect of interest simple and lets everything else be arbitrary:

$$y = \theta\,d + g(X) + u,\qquad d = m(X) + v,\qquad \mathbb E[u \mid X, d] = 0,\quad \mathbb E[v \mid X] = 0.$$

$\theta$ is a constant effect; $g$ and $m$ are unknown functions, called **nuisance functions** because you need them only to get at $\theta$. under the topic-01 assumptions ($X$ blocks every backdoor path from $d$ to $y$), $\theta$ is causal.

in linear regression there's an old trick, the **Frisch–Waugh–Lovell theorem**: the OLS coefficient on $d$ in a regression of $y$ on $(d, X)$ equals the slope of the regression of "$y$ with $X$ partialled out" on "$d$ with $X$ partialled out" (warm-up 1: yes, exactly). Robinson (1988) generalized the idea to the PLR. take conditional expectations given $X$: $\ell(X) := \mathbb E[y \mid X] = \theta m(X) + g(X)$. subtract:

$$y - \ell(X) = \theta\,\big(d - m(X)\big) + u.$$

so $\theta$ is the slope of a no-intercept regression of $\tilde y = y - \ell(X)$ on $\tilde d = d - m(X)$, and both $\ell$ and $m$ are plain prediction problems you can hand to any regressor. the "double" in double ML is that you learn *two* nuisance functions, one for the outcome and one for the treatment.

now the trap. suppose $\hat m$ is a fully grown decision tree fitted on all the data. on its own training points it's perfect: $\hat m(X_i) = d_i$, so every in-sample residual $\tilde d_i$ is zero (warm-up 3), and $\theta = \sum\tilde d\tilde y/\sum\tilde d^2$ is $0/0$. less flexible models do the same thing partially: in-sample residuals are too small, because the model has absorbed part of the noise $v_i$ it should have left alone, and $\hat m(X_i)$ now depends on $v_i$ itself. that correlation between a unit's nuisance error and its own noise is a bias term that no rate condition removes.

**cross-fitting** cuts it. split the units into $K$ folds; for each fold, fit the nuisance model on the other $K - 1$ folds and predict the held-out fold. every unit gets an **out-of-fold** prediction from a model that never saw it, so its nuisance error is independent of its own noise. unlike a single sample split, every unit is used both for fitting (in $K - 1$ of the models) and for estimation, so you don't throw data away (warm-up 4: no). the price is $K$ fits instead of one.

> **spec.** `cross_fit_predict(make_model, X, y, *, n_folds=5, response="predict", folds=None, rng=None)` returns a float array `(n,)` of out-of-fold predictions. folds are `np.array_split(as_rng(rng).permutation(n), n_folds)`, used in that order, unless `folds` is given (a list of test-index arrays), in which case exactly those folds are used and `rng` and `n_folds` are ignored; for each fold, `make_model()` is fitted on all other rows and predicts that fold, with `predict(X)` for `response="predict"` or `predict_proba(X)[:, 1]` for `response="proba"`. any other `response` raises `ValueError`. without `folds`, each call draws exactly one permutation from the generator.

> **▶ build: stage 1.** implement `cross_fit_predict`.

### idea 2: orthogonal scores, and the naive score they replace

most estimators solve an estimating equation: pick $\hat\theta$ so that the average of a **score** $\psi(W; \theta, \eta)$ is zero, where $W = (y, d, X)$ is one unit's data and $\eta$ the nuisances. two scores for the PLR:

$$\psi_\text{naive} = \big(y - \theta d - g(X)\big)\,d,\qquad \psi_\perp = \big(y - \ell(X) - \theta(d - m(X))\big)\big(d - m(X)\big).$$

with the true nuisances both have mean zero at the true $\theta$. the difference is what happens when you plug in *estimated* nuisances. perturb the nuisance in some direction, $g \to g + r\,h$, and differentiate the expected score at $r = 0$. for the naive score,

$$\frac{\partial}{\partial r}\,\mathbb E\big[(y - \theta d - g(X) - r\,h(X))\,d\big]\Big|_{r=0} = -\mathbb E[h(X)\,d] = -\mathbb E[h(X)\,m(X)],$$

which isn't zero: any error in $\hat g$ that correlates with $m(X)$ moves the estimate one-for-one. that's the **regularization bias** of the naive plug-in: solving $\sum\psi_\text{naive} = 0$ gives $\hat\theta - \theta = \sum d_i\,(g - \hat g)(X_i)/\sum d_i^2 + \text{noise}$, first order in the nuisance error (warm-up 2: biased, by however much the lasso's shrinkage of $g$ correlates with $d$). for $\psi_\perp$ the same derivative in either nuisance direction is zero (on paper 2). a score with this property is **Neyman orthogonal**: nuisance errors affect it only at second order.

solving $\frac1n\sum\psi_\perp = 0$ gives the **partialling-out estimator** and a standard error. with residuals $\tilde y_i = y_i - \hat\ell(X_i)$ and $\tilde d_i = d_i - \hat m(X_i)$:

$$\hat\theta = \frac{\sum_i\tilde d_i\tilde y_i}{\sum_i\tilde d_i^2},\qquad \hat\psi_i = \tilde d_i\big(\tilde y_i - \hat\theta\tilde d_i\big),\qquad \widehat{\mathrm{se}} = \sqrt{\frac1n\cdot\frac{\frac1n\sum_i\hat\psi_i^2}{\big(\frac1n\sum_i\tilde d_i^2\big)^2}}.$$

the standard error is the **sandwich** formula: the variance of the score ("meat") divided by the square of its slope in $\theta$ ("bread", $J = -\partial_\theta\,\mathbb E\psi_\perp = \mathbb E\tilde d^2$). it's robust to heteroskedastic noise; with in-sample OLS residuals it reproduces OLS's HC0 standard error exactly.

for comparison you'll also build the naive estimator in its usual iterative form: start at $\theta = 0$, fit $\hat g$ to $y - \theta d$, set $\theta = \sum d(y - \hat g)/\sum d^2$, repeat. with a linear $g$ this is backfitting, and it converges to the OLS coefficient; with a regularized $g$ it inherits the bias above.

> **spec.** `partialling_out(y_res, d_res)` returns `(theta, se)` from the formulas above (means divide by $n$). `naive_plugin(X, y, d, make_model, *, n_iters=10)` starts from `theta = 0.0` and repeats `n_iters` times: `g = make_model().fit(X, y - theta * d).predict(X)`, then `theta = sum(d * (y - g)) / sum(d ** 2)`; it returns the final `theta` as a float (no cross-fitting: this is the estimator DML replaces).

> **🔮 predict:** in the lab, a lasso is used both inside the naive plug-in and inside DML. as you raise the lasso penalty, which estimate drifts away from the truth faster?

> **▶ build: stage 2.** implement `partialling_out` and `naive_plugin`.

### idea 3: the DML estimator

put the pieces together. **DML for the PLR**:

1. split the units into $K$ random folds.
2. get out-of-fold predictions $\hat\ell(X_i)$ (a regression of $y$ on $X$) and $\hat m(X_i)$ (a regression of $d$ on $X$), using the same folds for both.
3. form residuals and solve the orthogonal score: `partialling_out`.

why does this work with slow machine-learning rates? with cross-fitted nuisances, write $\delta_m = m - \hat m$ and $\delta_g = (\ell - \hat\ell) - \theta(m - \hat m)$ for the errors. the estimator's error is (on paper 3)

$$\hat\theta - \theta \;\approx\; \underbrace{\frac{\frac1n\sum_i v_iu_i}{\mathbb E[v^2]}}_{\text{the oracle's error, }O(n^{-1/2})} \;+\; \underbrace{\frac{\mathbb E[\delta_m\,\delta_g]}{\mathbb E[v^2]}}_{\text{product of nuisance errors}}.$$

if each nuisance error shrinks like $n^{-1/4}$ (in root-mean-square), the product shrinks like $n^{-1/2}$ and is swamped by the first term, so $\hat\theta$ behaves as if you'd known $\ell$ and $m$ (warm-up 5: yes). $n^{-1/4}$ is a modest demand that lasso, forests, boosting and neural nets meet under reasonable conditions. the naive estimator would need its single nuisance error to be $o(n^{-1/2})$, which essentially only correctly specified parametric models deliver.

one random split is one draw of the folds, and the estimate moves a little with the split. **repeated cross-fitting** runs the whole procedure $S$ times with fresh folds and aggregates by the median, adding the between-split spread to the standard error (Chernozhukov et al., 2018, §3.4):

$$\hat\theta = \operatorname{median}_s\hat\theta_s,\qquad \widehat{\mathrm{se}} = \sqrt{\operatorname{median}_s\big(\widehat{\mathrm{se}}_s^2 + (\hat\theta_s - \hat\theta)^2\big)}.$$

this isn't a guarantee of good results. orthogonality removes the *first-order* effect of nuisance errors, but if both nuisance models are badly wrong (a heavily regularized tree, or a linear model of a nonlinear world), the product term is not small, and DML is biased too. it relaxes how good the models must be; it doesn't remove the need for good ones.

> **spec.** `dml_plr(X, y, d, make_model_y, make_model_d, *, n_folds=5, n_repeats=1, rng=None)` makes **one** generator `g = as_rng(rng)`; in each repeat it draws **one** partition `folds = np.array_split(g.permutation(n), n_folds)`, shared by both nuisances, calls `cross_fit_predict(make_model_y, X, y, folds=folds)` and `cross_fit_predict(make_model_d, X, d, folds=folds)`, and runs `partialling_out` on the residuals. it returns a dict with `"theta"` and `"se"` (the median aggregation above; with one repeat, just that repeat's values), `"ci"` = `(theta - 1.96 se, theta + 1.96 se)`, and `"thetas"`, the array of per-repeat estimates.

> **▶ build: stage 3.** implement `dml_plr`.

### idea 4: the interactive model (cross-fitted AIPW)

the PLR assumes a constant effect $\theta$. for a binary treatment you can drop that assumption. the **interactive regression model** (IRM) allows any effect heterogeneity:

$$y = \mu_d(X) + u,\qquad d \in \{0, 1\},\qquad P(d = 1 \mid X) = e(X),$$

and the target is the ATE, $\mathbb E[\mu_1(X) - \mu_0(X)]$. you already know an orthogonal score for it: the AIPW score of topic 02,

$$\psi_i = \hat\mu_1(X_i) - \hat\mu_0(X_i) + \frac{d_i\,(y_i - \hat\mu_1(X_i))}{\hat e(X_i)} - \frac{(1 - d_i)(y_i - \hat\mu_0(X_i))}{1 - \hat e(X_i)} - \theta.$$

its double robustness *is* Neyman orthogonality: the bias was the product $\mathbb E[(e - \hat e)(\mu - \hat\mu)/\hat e]$, second order in the errors. so DML for the IRM is AIPW with cross-fitted nuisances: in each fold, fit $\hat\mu_0$ on the training controls, $\hat\mu_1$ on the training treated, and a classifier $\hat e$ on all training units; predict the held-out fold; average the scores. propensities are clipped to $[c, 1 - c]$ because a flexible classifier can output probabilities of exactly 0 or 1, and the score divides by them. clipping trades a little bias for a lot of stability, as you saw in topic 02.

> **spec.** `dml_irm(X, y, d, make_model_y, make_classifier_d, *, n_folds=5, clip=0.01, rng=None)` uses one permutation `as_rng(rng).permutation(n)` split with `np.array_split` into `n_folds` folds. for each fold: `make_model_y()` fitted on the training rows with `d == 0` gives `mu0` on the fold, another on the training rows with `d == 1` gives `mu1`, and `make_classifier_d()` fitted on all training rows gives `e = predict_proba(X_fold)[:, 1]`. clip `e` to `[clip, 1 - clip]`, form the AIPW scores, and return a dict with `"theta"` (their mean), `"se"` (`std(ddof=1) / sqrt(n)`), `"ci"` and `"scores"`.

> **🔮 predict:** in the lab's semi-synthetic study, will DML with *linear* outcome models and a *logistic* propensity model beat plain OLS?

> **▶ build: stage 4.** implement `dml_irm`.

## worked example

**partialling out by hand.** four units with residuals

| unit | $\tilde y$ | $\tilde d$ | $\tilde d\tilde y$ | $\tilde d^2$ |
|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 |
| 2 | 2 | 1 | 2 | 1 |
| 3 | −1 | −1 | 1 | 1 |
| 4 | 0 | −1 | 0 | 1 |

$\hat\theta = 4/4 = 1$. the scores $\hat\psi_i = \tilde d_i(\tilde y_i - \tilde d_i)$ are $0, 1, 0, -1$, so the meat is $\frac14(0 + 1 + 0 + 1) = 0.5$ and the bread $J = \frac14\sum\tilde d^2 = 1$. the standard error is $\sqrt{0.5/1^2/4} = \sqrt{0.125} = 0.354$, and the 95% interval is $1 \pm 0.693$.

**the naive score's first step.** with a model that predicts a constant, `naive_plugin` starts at $\theta = 0$, fits $\hat g = \bar y$, and sets $\theta_1 = \sum d_i(y_i - \bar y)/\sum d_i^2$; with $d = (1, 2, 3)$ and $y = (1, 3, 2)$ that's $(1 \cdot (-1) + 2 \cdot 1 + 3 \cdot 0)/14 = 1/14$. note the denominator is $\sum d^2$, not $\sum(d - \bar d)^2$: the naive score doesn't centre $d$, which is precisely its non-orthogonality.

**folds.** with $n = 23$ and 5 folds, `np.array_split` gives folds of sizes 5, 5, 5, 4, 4, so the five nuisance models train on 18, 18, 18, 19 and 19 units. your code must reproduce the residual example, and the fold sizes are checked.

## on paper

1. **Frisch–Waugh–Lovell.** let $\hat\beta_d$ be the OLS coefficient on $d$ in $y \sim d + X$ (with an intercept in $X$). let $\tilde y$, $\tilde d$ be the residuals of OLS regressions of $y$ and $d$ on $X$. show $\hat\beta_d = \sum\tilde d\tilde y/\sum\tilde d^2$.

<details><summary>solution</summary>

write $y = \hat\beta_dd + X\hat\beta_X + \hat u$ with $X^\top\hat u = 0$ and $d^\top\hat u = 0$ (the normal equations). apply the residual-maker $M = I - X(X^\top X)^{-1}X^\top$, which kills $X$: $My = \hat\beta_d\,Md + M\hat u$. since $X^\top\hat u = 0$, $M\hat u = \hat u$. so $\tilde y = \hat\beta_d\tilde d + \hat u$. multiply by $\tilde d^\top$: $\tilde d^\top\hat u = d^\top M\hat u = d^\top\hat u = 0$, hence $\tilde d^\top\tilde y = \hat\beta_d\,\tilde d^\top\tilde d$.

</details>

2. **Neyman orthogonality.** show that $\frac{\partial}{\partial r}\mathbb E\big[\psi_\perp(W; \theta, \ell + rh_\ell, m + rh_m)\big]\big|_{r=0} = 0$ for any directions $h_\ell, h_m$ (functions of $X$).

<details><summary>solution</summary>

differentiating $(y - \ell - rh_\ell - \theta(d - m - rh_m))(d - m - rh_m)$ at $r = 0$ gives $(-h_\ell + \theta h_m)(d - m) + (y - \ell - \theta(d - m))(-h_m)$. at the truth, $d - m = v$ and $y - \ell - \theta(d - m) = u$. so the expectation is $\mathbb E[(-h_\ell + \theta h_m)v] - \mathbb E[u\,h_m]$. both vanish by iterated expectations: $\mathbb E[v \mid X] = 0$ and $\mathbb E[u \mid X] = 0$, and $h_\ell, h_m$ depend only on $X$.

</details>

3. **the product of errors.** with cross-fitted nuisances, $\tilde d = v + \delta_m$ and $\tilde y = \theta v + u + \delta_\ell$, where $\delta_m = m - \hat m$, $\delta_\ell = \ell - \hat\ell$ depend only on $X$ and the other folds. find the probability limit of $\hat\theta = \sum\tilde d\tilde y/\sum\tilde d^2$ and show that its bias is $\mathbb E[\delta_m\delta_g]/(\mathbb E v^2 + \mathbb E\delta_m^2)$ with $\delta_g = \delta_\ell - \theta\delta_m$. then do the same for the naive estimator.

<details><summary>solution</summary>

numerator: $\mathbb E[(v + \delta_m)(\theta v + u + \delta_\ell)] = \theta\mathbb Ev^2 + \mathbb E[\delta_m\delta_\ell]$, because $u$ and $v$ have conditional mean zero given $X$ (and given the other folds, which is where cross-fitting is used), so every cross term vanishes. denominator: $\mathbb Ev^2 + \mathbb E\delta_m^2$. so $\hat\theta - \theta \to \frac{\mathbb E[\delta_m\delta_\ell] - \theta\mathbb E\delta_m^2}{\mathbb Ev^2 + \mathbb E\delta_m^2} = \frac{\mathbb E[\delta_m(\delta_\ell - \theta\delta_m)]}{\mathbb Ev^2 + \mathbb E\delta_m^2}$. naive: $y - \hat g = \theta d + \delta_g + u$ with $\delta_g = g - \hat g$, so $\tilde\theta - \theta \to \mathbb E[d\,\delta_g]/\mathbb Ed^2 = \mathbb E[m\,\delta_g]/\mathbb Ed^2$: first order in $\delta_g$, multiplied by the full signal $m$, not by another error.

</details>

4. **why in-sample residuals fail.** let $\hat m$ be 1-nearest-neighbour regression and use in-sample residuals. what is $\tilde d$? now take a "smoother" $\hat m(X_i) = (1 - \lambda)\,m(X_i) + \lambda\,d_i$ with $0 < \lambda < 1$, which leaks a fraction $\lambda$ of each unit's own $d_i$. with $\hat\ell$ exact, what does $\hat\theta$ converge to?

<details><summary>solution</summary>

1-NN on its training set returns each unit's own $d_i$, so $\tilde d = 0$ and $\hat\theta$ is undefined. with the leaky smoother, $\tilde d_i = d_i - (1 - \lambda)m_i - \lambda d_i = (1 - \lambda)v_i$ and $\tilde y = \theta v + u$, so $\hat\theta \to \theta/(1 - \lambda)$: too big, by a factor that doesn't shrink with $n$. this is the leak that cross-fitting removes, since an out-of-fold prediction can't depend on the unit's own $d_i$.

</details>

## experiment

`lab.py` uses your DML code with your part II models:

1. a high-dimensional partially linear model where a lasso is the natural nuisance learner: the naive plug-in versus DML versus OLS, at several lasso penalties.
2. decision trees from heavily regularized to fully grown as nuisance models, with in-sample residuals and with cross-fitting.
3. a semi-synthetic study on real California housing covariates with a simulated binary treatment: the naive difference, OLS, and your cross-fitted AIPW with linear and with boosted nuisances.

## check your understanding

1. why is predicting $y$ from $(d, X)$ well not the same as estimating $\theta$ well?

<details><summary>answer</summary>

a predictor is judged by its total error; it happily trades bias in how it attributes the signal between $d$ and $X$ for lower variance. $\theta$ is one specific attribution, and regularization bias in the $X$ part leaks into it (on paper 3, naive case) without hurting prediction much.

</details>

2. (the predict callout in idea 2.) which estimate drifts faster as the lasso penalty grows?

<details><summary>answer</summary>

the naive plug-in: its bias is first order in the lasso's error in $g$, which grows with the penalty. DML's bias is the product of the errors in $\hat\ell$ and $\hat m$; it grows too, but much more slowly while the errors are small. at the largest penalty in the lab both models are bad and DML's bias becomes visible as well.

</details>

3. does cross-fitting make the estimate random? is that a problem?

<details><summary>answer</summary>

yes, the estimate depends on the fold assignment. the spread across splits is usually small compared with the standard error; repeated cross-fitting with median aggregation reduces the dependence and adds the spread to the reported uncertainty.

</details>

4. the PLR gives one $\theta$. what does it estimate if the true effect varies with $X$?

<details><summary>answer</summary>

a weighted average effect, with weights proportional to the conditional variance of the treatment, $\mathrm{Var}(d \mid X)$: units whose treatment is hard to predict from $X$ count most. that's often not the ATE, which is one reason to use the IRM (stage 4) for binary treatments.

</details>

5. why does the IRM need the propensity clipped while the PLR doesn't?

<details><summary>answer</summary>

the AIPW score divides by $\hat e$ and $1 - \hat e$; a classifier can output exactly 0 or 1. the PLR score only multiplies by residuals, so there's nothing to blow up (it fails differently: if $d$ is nearly a deterministic function of $X$, the residuals are tiny and $\sum\tilde d^2$ is small, the PLR version of poor overlap).

</details>

6. you run DML with a gradient-boosting model you tuned for test-set $R^2$. is that the right tuning target?

<details><summary>answer</summary>

it's a reasonable one: the bias depends on the nuisances' mean squared errors, and out-of-fold $R^2$ measures exactly that. what matters is *each* nuisance's error, so tune $\hat\ell$ and $\hat m$ separately, inside the training folds.

</details>

## common bugs

- **in-sample nuisance predictions.** symptom: estimates that get worse as the models get more flexible, or divide-by-zero with very flexible ones. predict each fold with a model that didn't train on it.
- **one model for all folds.** symptom: cross-fitting is a no-op. call `make_model()` inside the fold loop, so each fold gets a fresh, unfitted model.
- **different folds for the two nuisances.** symptom: `dml_plr` doesn't match the checks. draw one partition per repeat and give it to both `cross_fit_predict` calls (as Chernozhukov et al. and DoubleML do): each unit's $\hat\ell$ and $\hat m$ then come from models fitted on the same data, so their errors line up and the product term is estimated honestly.
- **centring $d$ in the naive score.** symptom: `naive_plugin` disagrees with the checks. the naive estimator divides by $\sum d^2$, uncentred; the orthogonal one uses residuals.
- **dividing by $n - 1$ in the sandwich.** symptom: a standard error slightly off from the FWL/HC0 check. both means divide by $n$.
- **fitting $\hat\mu_0$, $\hat\mu_1$ on everyone.** symptom: the IRM's regression part estimates zero effect. fit each on its own treatment arm within the training folds.

## going deeper

- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018), "Double/debiased machine learning for treatment and structural parameters", *The Econometrics Journal* 21(1). [doi:10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097) · [arXiv:1608.00060](https://arxiv.org/abs/1608.00060)
- Robinson (1988), "Root-N-consistent semiparametric regression", *Econometrica* 56(4). partialling out in the partially linear model. [doi:10.2307/1912705](https://doi.org/10.2307/1912705)
- Frisch & Waugh (1933), "Partial time regressions as compared with individual trends", *Econometrica* 1(4). [doi:10.2307/1907330](https://doi.org/10.2307/1907330); Lovell (1963), "Seasonal adjustment of economic time series and multiple regression analysis", *JASA* 58(304). [doi:10.1080/01621459.1963.10480682](https://doi.org/10.1080/01621459.1963.10480682)
- Belloni, Chernozhukov & Hansen (2014), "Inference on treatment effects after selection among high-dimensional controls", *Review of Economic Studies* 81(2). the lasso version of the problem in experiment 1. [doi:10.1093/restud/rdt044](https://doi.org/10.1093/restud/rdt044)
- Kennedy (2022), "Semiparametric doubly robust targeted double machine learning: a review". influence functions and orthogonality, from the ground up. [arXiv:2203.06469](https://arxiv.org/abs/2203.06469)
- Bach, Chernozhukov, Kurz & Spindler (2022), "DoubleML: an object-oriented implementation of double machine learning in Python", *JMLR* 23. [arXiv:2104.03220](https://arxiv.org/abs/2104.03220)
- Hill (2011), "Bayesian nonparametric modeling for causal inference", *Journal of Computational and Graphical Statistics* 20(1). the semi-synthetic IHDP benchmark recipe used in experiment 3. [doi:10.1198/jcgs.2010.08162](https://doi.org/10.1198/jcgs.2010.08162)

## connections

- ← builds on: lasso, linear and logistic regression (ch 08), trees and gradient boosting (ch 10), K-fold splitting (ch 07 · topic 02), the AIPW estimator and double robustness (topic 02), adjustment sets (topic 01).
- → used in: the block's project (topic 04), which estimates an effect after learning which variables to adjust for; any later work that puts a neural network inside a causal estimate (the same orthogonal scores apply, with ch 15's networks as nuisance models).
