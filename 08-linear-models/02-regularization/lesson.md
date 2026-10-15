# regularization

## why this matters

topic 01 ended with a degree-14 polynomial whose coefficients were in the trillions and whose test error was in the thousands. it fit 15 points exactly, and it was useless. the problem wasn't the basis: it was that least squares, given enough freedom, uses all of it, including to fit noise. and you don't need polynomials to get there. with 40 rows and 30 features, ordinary least squares on perfectly reasonable data has a test error several times the noise level, and with more features than rows it has no unique answer at all.

**regularization** changes the question. instead of "which $w$ fits best?", it asks "which $w$ fits well *and* is small?", with a dial $\alpha$ that sets the exchange rate. two ways of measuring "small" give two very different models. the squared length $\|w\|^2$ gives **ridge regression**: every coefficient shrinks smoothly, the problem always has a unique solution, and it's still a single linear solve. the sum of absolute values $\|w\|_1$ gives the **lasso**: coefficients don't just shrink, they hit *exactly* zero, so the model selects features. you'll build both, their blend (the **elastic net**), the whole path of solutions as $\alpha$ varies, and see each of them as a Bayesian prior in disguise.

## warm-up: predict first

1. you fit ridge with a huge $\alpha$. what does every coefficient go to? what does the intercept go to?
2. one feature is measured in meters, then re-measured in millimeters. does the ridge fit change? does the least-squares fit?
3. as $\alpha$ decreases from $\infty$, ridge coefficients appear gradually. do lasso coefficients appear gradually too, or all at once?
4. two features are identical copies. least squares has no unique answer. what does ridge do with them? the lasso?
5. a ridge model with $\alpha > 0$ has $d = 10$ coefficients. is it really "using" 10 degrees of freedom?

## prerequisites check

- [ ] you've built `LinearRegression` and understand centering for the intercept (→ topic 01, idea 3)
- [ ] you know the thin SVD $X = U\operatorname{diag}(s)V^\top$ and what it says about least squares (→ ch 02 · topic 06)
- [ ] you can write down a Gaussian posterior up to a constant (→ ch 04 · topic 04, ch 05 · topic 02)
- [ ] you know subgradients, or are happy to take "the derivative of $|w|$ at 0 is anything in $[-1, 1]$" on trust (→ ch 06 · topic 06)
- [ ] you can explain the bias–variance trade-off (→ ch 07 · topic 04)

## core ideas

### idea 1: ridge regression

ridge adds a penalty on the squared length of the weights:

$$\min_{w, b}\ \|y - Xw - b\mathbf 1\|^2 + \alpha\|w\|^2, \qquad \alpha \ge 0.$$

(this is the scaling `mlfs` and scikit-learn use for ridge: a **sum** of squared errors, not a mean. the lasso below uses a mean. it's a historical quirk, so the two $\alpha$s are on different scales.) the intercept $b$ is not penalized: shrinking it would make the model depend on where you put the origin of $y$. so, exactly as in topic 01, center $X$ and $y$, solve for $w$ on the centered data, and recover $b = \bar y - \bar x^\top w$. setting the gradient of the centered objective to zero gives

$$(X^\top X + \alpha I)\,w = X^\top y.$$

for $\alpha > 0$ the matrix $X^\top X + \alpha I$ is positive definite even when $X$ has more columns than rows or duplicate columns: its eigenvalues are those of $X^\top X$ plus $\alpha$. so the solution always exists, is unique, and comes from one Cholesky solve. the penalty also improves conditioning: $\kappa = (s_{\max}^2 + \alpha)/(s_{\min}^2 + \alpha)$.

what does it cost? bias: as $\alpha \to \infty$, $w \to 0$ and the model predicts $\bar y$ everywhere (warm-up 1). what does it buy? variance: the directions in which the data say little are exactly the ones least squares gets most wrong, and ridge damps them (idea 2 makes this precise). somewhere in between is a sweet spot, which you find by cross-validation.

**scale matters now.** least squares is equivariant to rescaling a feature (topic 01, check your understanding 4). ridge is not: a feature in millimeters needs a 1,000× smaller coefficient than in meters, which the penalty barely charges for, so it's effectively unpenalized (warm-up 2). standardize features before ridge, unless their units are already comparable. that's why chapter 07's `StandardScaler` exists.

**the Bayesian view.** suppose $y = Xw + \varepsilon$ with $\varepsilon \sim \mathcal N(0, \sigma^2 I)$, and before seeing data you believe $w \sim \mathcal N(0, \tau^2 I)$: coefficients are probably small. by Bayes' rule, the log posterior is

$$\log p(w \mid y) = -\frac{1}{2\sigma^2}\|y - Xw\|^2 - \frac{1}{2\tau^2}\|w\|^2 + \text{const},$$

so the most probable $w$ (which, for a Gaussian, is also the posterior mean) minimizes $\|y - Xw\|^2 + \frac{\sigma^2}{\tau^2}\|w\|^2$: ridge with $\alpha = \sigma^2/\tau^2$ (on paper 2). the penalty *is* a prior, and its strength is the noise-to-signal ratio. noisy data or a skeptical prior means more shrinkage. the lab checks this numerically.

> **🔮 predict:** the true coefficients are drawn from $\mathcal N(0, 0.3^2)$ and the noise has $\sigma = 1$. what $\alpha$ should give the best predictions on average? does the answer depend on the number of rows?

> **▶ build: stage 1.** implement `Ridge(alpha=1.0, *, fit_intercept=True)`: center with the given `center` helper, solve $(X^\top X + \alpha I)w = X^\top y$ with your `cholesky` and `cho_solve`, recover the intercept, and predict with the given `linear_predict`.

### idea 2: ridge through the SVD, and the whole path at once

write the thin SVD $X = U\operatorname{diag}(s)V^\top$ with singular values $s_1 \ge \dots \ge s_r > 0$, left singular vectors $u_j$ and right singular vectors $v_j$. substituting into $(X^\top X + \alpha I)w = X^\top y$ gives

$$w(\alpha) = \sum_{j=1}^r \frac{s_j}{s_j^2 + \alpha}\,(u_j^\top y)\,v_j, \qquad \hat y(\alpha) = Xw(\alpha) = \sum_{j=1}^r \frac{s_j^2}{s_j^2 + \alpha}\,(u_j^\top y)\,u_j.$$

compare with least squares ($\alpha = 0$), where each factor $s_j^2/(s_j^2 + \alpha)$ is 1. ridge keeps the directions with large singular values almost untouched and shrinks the ones with small singular values the most. those are the directions in which the data barely vary, so the data carry little information about the slope there, and least squares divides by a tiny $s_j$ to estimate it: that's where its variance comes from. ridge is a smooth version of throwing those directions away.

two consequences. first, **the whole path is cheap**: one SVD, then each $\alpha$ costs a few vector operations. second, the sum of the shrinkage factors measures how many directions the model really fits:

$$\operatorname{df}(\alpha) = \sum_{j=1}^r \frac{s_j^2}{s_j^2 + \alpha} = \operatorname{tr}\, X(X^\top X + \alpha I)^{-1}X^\top.$$

this is the **effective degrees of freedom**: $r$ at $\alpha = 0$ (all directions, like least squares with $r$ parameters), falling smoothly to 0 as $\alpha \to \infty$ (warm-up 5). it's the trace of the "hat matrix" that maps $y$ to $\hat y$, the same quantity that counted parameters in topic 01's $E[\text{RSS}] = \sigma^2(n - p)$.

in floating point, a rank-deficient $X$ doesn't give singular values that are exactly 0, but values around $10^{-16}s_1$. with $\alpha = 0$, dividing by them would turn rounding noise into enormous coefficients, so treat any $s_j \le 10^{-12}s_1$ as zero (like your chapter 02 `pinv` did).

> **▶ build: stage 2.** implement `ridge_path(X, y, alphas)` (no intercept: `coefs` of shape `(len(alphas), d)` from one `np.linalg.svd`) and `ridge_dof(X, alphas)`. you built an SVD in chapter 02; here you use NumPy's, which is the same thing, faster.

### idea 3: the lasso, one coordinate at a time

the lasso swaps the squared length for the sum of absolute values:

$$\min_w\ \frac{1}{2n}\|y - Xw\|^2 + \alpha\|w\|_1, \qquad \|w\|_1 = \sum_j |w_j|.$$

(the intercept is handled by centering, as always.) the absolute value has a kink at 0, and that kink is the whole point. near 0, the squared penalty $\alpha w_j^2$ is flat, so it pushes a small coefficient towards zero with a vanishing force and never gets it there. the $\ell_1$ penalty pushes with the same force $\alpha$ all the way down, and at zero it can *hold* the coefficient there against any pull smaller than $\alpha$. that's how exact zeros appear.

there's no closed form, but there's a beautifully simple algorithm. fix every coefficient except $w_j$ and minimize over $w_j$ alone. with the residual $r = y - Xw$ and $z_j = \frac1n\|X_j\|^2$ (where $X_j$ is column $j$), the one-dimensional problem is a parabola plus $\alpha|w_j|$, and its solution is (on paper 3)

$$w_j \leftarrow \frac{S\left(\rho_j,\ \alpha\right)}{z_j}, \qquad \rho_j = \frac1n X_j^\top r + z_j w_j, \qquad S(\rho, t) = \operatorname{sign}(\rho)\max(|\rho| - t, 0).$$

$\rho_j$ is how much feature $j$ would like its coefficient to be, measured as its correlation with the residual left after all the *other* features (adding $z_jw_j$ puts feature $j$'s own contribution back). the **soft-threshold** $S$ moves it towards 0 by $\alpha$ and stops at 0: if feature $j$'s correlation with what's left is weaker than $\alpha$, its coefficient is exactly 0.

**cyclic coordinate descent** sweeps $j = 0, 1, \dots, d - 1$ over and over, updating each $w_j$ in turn and keeping $r$ up to date with one vector operation per change (`r -= X[:, j] * (new - old)`). each update can only decrease the objective, the objective is convex, and the iteration converges to the global minimum. stop when a whole sweep moves no coefficient by more than `tol`.

the **elastic net** adds a ridge term to the lasso: $\frac{1}{2n}\|y - Xw\|^2 + \ell_1\|w\|_1 + \frac{\ell_2}{2}\|w\|^2$. the coordinate update gains one term in the denominator, $w_j \leftarrow S(\rho_j, \ell_1)/(z_j + \ell_2)$, so one function handles both.

> **🔮 predict:** two features are exact copies of each other, with true coefficients $1$ and $1$. what will coordinate descent do with them from a start at zero, and does the answer depend on which column comes first?

> **▶ build: stage 3.** implement `soft_threshold(z, t)` and `coordinate_descent(X, y, l1, l2=0.0, *, w0=None, max_iters=1000, tol=1e-06)`, returning `(w, n_sweeps)`. keep the residual up to date instead of recomputing `X @ w` for every coordinate, give an all-zero column the coefficient 0, and copy `w0`.

### idea 4: lasso and elastic net estimators

with the solver in hand, the estimators are short: center, call `coordinate_descent`, recover the intercept. `Lasso(alpha)` uses $\ell_1 = \alpha$ and $\ell_2 = 0$. `ElasticNet(alpha, l1_ratio)` splits one strength between the two penalties,

$$\frac{1}{2n}\|y - Xw - b\mathbf 1\|^2 + \alpha\,\rho\,\|w\|_1 + \frac{\alpha(1 - \rho)}{2}\|w\|^2, \qquad \rho = \texttt{l1\_ratio} \in [0, 1],$$

so $\rho = 1$ is the lasso and $\rho = 0$ is ridge (with the $\frac{1}{2n}$ scaling, so `ElasticNet(alpha, 0)` matches `Ridge(n * alpha)`).

why blend? the lasso has a weakness with correlated features (warm-up 4): if two features carry the same signal, the $\ell_1$ penalty is indifferent between giving the weight to one or splitting it ($|a| + |b|$ is the same for every split with the same sign), so the lasso tends to pick one, more or less arbitrarily, and drop the other. ridge's $\|w\|^2$ strictly prefers an even split ($a^2 + b^2$ is smallest when $a = b$). the elastic net inherits sparsity from one and this **grouping effect** from the other. the lab shows the difference.

> **▶ build: stage 4.** implement `Lasso(alpha=1.0, *, fit_intercept=True, max_iters=1000, tol=1e-06)` and `ElasticNet(alpha=1.0, l1_ratio=0.5, *, ...)`, each storing `coef_`, `intercept_` and `n_iter_`.

### idea 5: paths and warm starts

you rarely know $\alpha$ in advance. the standard practice is to compute the solution for a whole grid of values, from "everything is zero" down to "almost least squares", and pick one by cross-validation. two facts make this cheap.

**where the path starts.** at $w = 0$ the residual is $y$, so the coordinate update gives $S(X_j^\top y / n, \alpha)/z_j$, which is zero for every $j$ exactly when $\alpha \ge \max_j |X_j^\top y|/n$ (on paper 4). so

$$\alpha_{\max} = \frac1n\max_j |X_j^\top y|$$

is the smallest $\alpha$ with an all-zero solution, and the first feature to enter as $\alpha$ drops below it is the one with the largest $|X_j^\top y|$ (the one most correlated with $y$, when the columns have equal norms, e.g. after standardizing). a standard grid is 100 values from $\alpha_{\max}$ down to $10^{-3}\alpha_{\max}$, evenly spaced on a log scale.

**warm starts.** the solution changes continuously with $\alpha$ (it's piecewise linear, for the lasso). so the solution at one $\alpha$ is an excellent starting point for the next, slightly smaller one: coordinate descent then needs only a few sweeps instead of starting from zero. going from the largest $\alpha$ down also means the early solves are on very sparse, easy problems. computing a 100-point path with warm starts often costs about as much as a few cold solves.

> **▶ build: stage 5.** implement `lasso_alpha_max(X, y)` and `lasso_path(X, y, alphas, *, max_iters=1000, tol=1e-06)`: solve from the largest $\alpha$ to the smallest, each from the previous solution, and return the rows in the order the alphas were given.

## worked example

three points with one feature, already centered: $x = (-1, 0, 1)$, $y = (-2, 1, 1)$ (mean 0). so $\sum x_i^2 = 2$, $\sum x_iy_i = 2 + 0 + 1 = 3$, $n = 3$.

**least squares:** $w = 3/2 = 1.5$.

**ridge** (sum scaling): $w = \frac{\sum x_iy_i}{\sum x_i^2 + \alpha} = \frac{3}{2 + \alpha}$. at $\alpha = 1$, $w = 1$. the SVD view agrees: $s = \sqrt2$, $u = x/\sqrt2$, $u^\top y = 3/\sqrt2$, and $\frac{s}{s^2 + \alpha}u^\top y = \frac{\sqrt2}{3}\cdot\frac{3}{\sqrt2} = 1$. the effective degrees of freedom are $\frac{2}{2 + \alpha} = \frac23$.

**lasso** (mean scaling): $z = \frac{2}{3}$ and, starting from $w = 0$, $\rho = \frac{1}{3}\sum x_iy_i = 1$. one coordinate update solves it: $w = S(1, \alpha)/z = \frac32\max(1 - \alpha, 0)$. at $\alpha = 0.5$, $w = 0.75$; at $\alpha = 1 = \alpha_{\max}$ it's exactly 0, and it stays 0 for every larger $\alpha$.

**elastic net** with $\alpha = 0.5$, `l1_ratio` $= 0.5$: $\ell_1 = \ell_2 = 0.25$, so $w = S(1, 0.25)/(z + 0.25) = 0.75/0.9\overline{16} \approx 0.818$.

the intercept is $\bar y - \bar x w = 0$ in every case. the tests check all of these.

## on paper

1. **ridge's normal equations.** derive $(X^\top X + \alpha I)w = X^\top y$ for centered data, and show that $X^\top X + \alpha I$ is positive definite for every $\alpha > 0$, whatever $X$ is.

<details><summary>solution</summary>

the gradient of $\|y - Xw\|^2 + \alpha\|w\|^2$ is $-2X^\top(y - Xw) + 2\alpha w$; setting it to zero gives the equation. for any $v \ne 0$, $v^\top(X^\top X + \alpha I)v = \|Xv\|^2 + \alpha\|v\|^2 \ge \alpha\|v\|^2 > 0$. the objective is strictly convex, so this stationary point is the unique minimizer.

</details>

2. **ridge as a posterior.** with $y \mid w \sim \mathcal N(Xw, \sigma^2 I)$ and prior $w \sim \mathcal N(0, \tau^2 I)$, show that the posterior is Gaussian with mean equal to the ridge solution with $\alpha = \sigma^2/\tau^2$, and find its covariance.

<details><summary>solution</summary>

$\log p(w \mid y) = -\frac{1}{2\sigma^2}\|y - Xw\|^2 - \frac{1}{2\tau^2}\|w\|^2 + c = -\frac{1}{2\sigma^2}\left(w^\top(X^\top X + \alpha I)w - 2w^\top X^\top y\right) + c'$ with $\alpha = \sigma^2/\tau^2$. a log density that's quadratic in $w$ is Gaussian; completing the square, the precision is $(X^\top X + \alpha I)/\sigma^2$ and the mean is $(X^\top X + \alpha I)^{-1}X^\top y$, the ridge solution. the covariance is $\sigma^2(X^\top X + \alpha I)^{-1}$. since the posterior mean minimizes the expected squared error under the posterior, ridge with this $\alpha$ is the best estimator *on average over $w$ drawn from the prior*, which is what the lab measures. chapter 12 develops this into Bayesian linear regression.

</details>

3. **the soft-threshold.** for $z > 0$, show that $\min_{v}\ \frac z2 v^2 - \rho v + \alpha|v|$ is attained at $v = S(\rho, \alpha)/z$. then show that this is the coordinate-wise minimizer of the lasso objective, with $z = \frac1n\|X_j\|^2$ and $\rho = \frac1n X_j^\top r + zw_j$.

<details><summary>solution</summary>

for $v > 0$ the derivative is $zv - \rho + \alpha$, zero at $v = (\rho - \alpha)/z$, valid when $\rho > \alpha$. for $v < 0$: $v = (\rho + \alpha)/z$, valid when $\rho < -\alpha$. if $|\rho| \le \alpha$, the subdifferential at 0, $-\rho + \alpha[-1, 1]$, contains 0, so $v = 0$. together, $v = S(\rho, \alpha)/z$. for the lasso, write the residual without feature $j$ as $r_{-j} = r + X_jw_j$; then $\frac{1}{2n}\|r_{-j} - X_jv\|^2 = \frac{z}{2}v^2 - \frac1n X_j^\top r_{-j}\,v + \text{const}$, and $\frac1n X_j^\top r_{-j} = \frac1n X_j^\top r + zw_j = \rho$.

</details>

4. **where the lasso path starts.** show that $w = 0$ minimizes the lasso objective (no intercept) if and only if $\alpha \ge \frac1n\max_j|X_j^\top y|$.

<details><summary>solution</summary>

the objective is convex, so $0$ is a minimizer iff $0$ is in its subdifferential there: $0 \in -\frac1n X^\top y + \alpha\,\partial\|0\|_1 = -\frac1n X^\top y + \alpha[-1, 1]^d$. that holds iff $|\frac1n X_j^\top y| \le \alpha$ for every $j$.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. the whole ridge path and the whole lasso path on data where only three of ten features matter.
2. two nearly identical features, resampled 40 times, under the lasso, ridge and the elastic net.
3. data drawn from ridge's own Bayesian model: which $\alpha$ is best, and what cross-validation picks.

## check your understanding

1. why is the intercept left out of the penalty? what would go wrong if you penalized it?

<details><summary>answer</summary>

the intercept just sets the level of the predictions. penalizing it would pull predictions towards 0 rather than towards $\bar y$, so the fit would depend on the arbitrary origin of $y$ (degrees Celsius versus Fahrenheit would give different slopes). centering makes the intercept exactly $\bar y - \bar x^\top w$ and leaves the penalty to act only on the slopes.

</details>

2. ridge shrinks every coefficient. can a ridge coefficient ever be *larger* in magnitude than the least-squares one?

<details><summary>answer</summary>

yes. the shrinkage is in the singular directions $v_j$, not coordinate by coordinate: the *vector* $w(\alpha)$ gets shorter, and each component along $v_j$ shrinks, but an individual coordinate can grow when correlated features stop cancelling each other. (the lab's experiment 1 has a gray, irrelevant coefficient that first moves away from zero as $\alpha$ grows.)

</details>

3. your lasso solution with $\alpha = 0.1$ has 4 nonzero coefficients. you standardize the features and refit with the same $\alpha$: still 4?

<details><summary>answer</summary>

not necessarily. the lasso penalty depends on each feature's scale (as ridge's does), and $\alpha_{\max} = \max_j|X_j^\top y|/n$ changes with the scales too. a feature with a large spread needs only a small coefficient and is cheap to include; after standardizing, every feature pays the same price per unit of effect.

</details>

4. coordinate descent updates one coefficient at a time. why doesn't it get stuck the way coordinate-wise methods can on non-smooth problems?

<details><summary>answer</summary>

the non-smooth part, $\alpha\|w\|_1$, is *separable*: a sum of terms that each involve one coordinate. for a convex objective of the form "smooth + separable", every point where no single coordinate can improve is a global minimum (Tseng, 2001). a non-separable kink (like $|w_1 - w_2|$) could trap it.

</details>

5. the ridge effective degrees of freedom at your chosen $\alpha$ is 4.2 for 10 features. what does that tell you?

<details><summary>answer</summary>

the fit has about as much flexibility as least squares with 4 parameters: the penalty has almost switched off the weakly determined directions. it's the number to plug in wherever a "number of parameters" appears, for example in an unbiased noise estimate $\text{RSS}/(n - \text{df})$ or in model-selection criteria like AIC.

</details>

6. why does the lasso path need warm starts but the ridge path doesn't?

<details><summary>answer</summary>

the ridge path has a closed form: after one SVD, each $\alpha$ is a few vector operations, so there's nothing to iterate. the lasso needs an iterative solver for each $\alpha$, and warm starts make each solve start almost at its answer.

</details>

7. with $n = 50$ rows and $d = 200$ features, how many nonzero coefficients can a lasso solution have?

<details><summary>answer</summary>

at most $n = 50$ (for features in general position there is always a solution with at most $\min(n, d)$ nonzeros). the lasso can't select more features than it has rows, which is one of the motivations for the elastic net, which can.

</details>

## common bugs

- **penalizing the intercept.** including the ones column in the penalty, or not centering. symptom: shifting $y$ by a constant changes the slopes; a huge $\alpha$ sends predictions to 0 instead of $\bar y$.
- **mixing up the two scalings.** a mean-squared-error ridge or a sum-squared-error lasso. symptom: coefficients that match scikit-learn only at $\alpha = 0$, off by a factor of $n$ in $\alpha$.
- **recomputing the residual for every coordinate.** `r = y - X @ w` inside the inner loop. symptom: correct but $d$ times slower, so lab paths take minutes.
- **forgetting to put $z_jw_j$ back.** using $\rho_j = X_j^\top r/n$ with the residual that still contains feature $j$. symptom: coefficients that never settle, or that converge to the wrong values; the KKT check fails.
- **dividing by a zero column's norm.** $z_j = 0$ for a constant feature after centering. symptom: `nan` everywhere after the first sweep.
- **a path in the wrong order.** returning rows sorted by $\alpha$ instead of in the given order, or warm-starting from small to large $\alpha$. symptom: the path is scrambled, or each solve starts from a dense solution far from the sparse answer.

## going deeper

- Hoerl & Kennard (1970), "Ridge regression: biased estimation for nonorthogonal problems", *Technometrics* 12(1): 55–67. where ridge comes from. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Tibshirani (1996), "Regression shrinkage and selection via the lasso", *Journal of the Royal Statistical Society B* 58(1): 267–288. [doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Zou & Hastie (2005), "Regularization and variable selection via the elastic net", *Journal of the Royal Statistical Society B* 67(2): 301–320. [doi:10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- Friedman, Hastie & Tibshirani (2010), "Regularization paths for generalized linear models via coordinate descent", *Journal of Statistical Software* 33(1). the glmnet algorithm: coordinate descent, warm starts, $\alpha_{\max}$. [doi:10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- Tseng (2001), "Convergence of a block coordinate descent method for nondifferentiable minimization", *Journal of Optimization Theory and Applications* 109(3): 475–494. why separable penalties are safe. [doi:10.1023/A:1017501703105](https://doi.org/10.1023/A:1017501703105)
- Hastie, Tibshirani & Wainwright (2015), *Statistical Learning with Sparsity: The Lasso and Generalizations*, CRC Press, chapters 2–5. free online. [hastie.su.domains/StatLearnSparsity](https://hastie.su.domains/StatLearnSparsity/)

## connections

- ← builds on: least squares and centering (topic 01), Cholesky and the SVD (ch 02), Gaussian posteriors (ch 04–05), subgradients and proximal steps (ch 06 · topic 06), the bias–variance trade-off and cross-validation (ch 07).
- → used in: $\ell_2$-penalized logistic and softmax regression (topics 04–05) and GLMs (topic 06), the capstone's text classifier, kernel ridge regression and Gaussian processes (ch 09, where ridge meets the kernel trick), Bayesian linear regression (ch 12, which keeps the whole posterior), weight decay in neural networks (ch 15), and sparse coding and compressed sensing ideas in later chapters.
