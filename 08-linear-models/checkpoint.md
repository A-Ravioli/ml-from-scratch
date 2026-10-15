# 08 · linear models · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. why should you never compute $(X^\top X)^{-1}X^\top y$ to fit least squares? what do you do instead?

<details><summary>answer</summary>

forming $X^\top X$ squares the condition number, so a problem with $\kappa(X) = 10^8$ can lose all 16 digits, and an explicit inverse is slower and less accurate than a solve. use QR (or an SVD-based `lstsq`), whose error grows like $\kappa(X)$; if you use the normal equations at all, factor with Cholesky and solve. (topic 01, idea 2)

</details>

2. why does `LinearRegression` center $X$ and $y$ instead of adding a column of ones, and how is the intercept recovered?

<details><summary>answer</summary>

minimizing over the intercept first gives $b = \bar y - \bar x^\top w$, which turns the problem into least squares on centered data, exactly equivalent. numerically it's much better: a ones column next to features with large means is nearly parallel to them, so the normal equations become hopelessly ill-conditioned. and it's what keeps the intercept unpenalized in ridge and lasso. (topic 01, idea 3)

</details>

3. gradient descent on least squares: what is the largest step size that converges, and how does the number of steps depend on feature scaling?

<details><summary>answer</summary>

it converges for $\eta < 2/L$ with $L = \lambda_{\max}(X^\top X/n)$. with $\eta = 1/L$ the error shrinks by $1 - 1/\kappa$ per step, so it needs about $\kappa\log(1/\varepsilon)$ steps. multiplying a feature by $s$ multiplies its eigenvalue by $s^2$, so badly scaled features make $\kappa$ (and the step count) explode; standardizing fixes it. (topic 01, idea 4)

</details>

4. least squares is maximum likelihood under which noise model? why is $\text{RSS}/n$ a biased estimate of the noise variance?

<details><summary>answer</summary>

independent Gaussian noise with constant variance. the fit chooses $\hat w$ to make the training residuals small, so they underestimate the noise: $E[\text{RSS}] = \sigma^2(n - p)$, and the unbiased estimate divides by $n - p$. (topic 01, idea 5)

</details>

5. ridge regression shrinks coefficients. which directions in feature space does it shrink most, and why is that a good idea?

<details><summary>answer</summary>

in the SVD basis, the component along $v_j$ is multiplied by $s_j^2/(s_j^2 + \alpha)$: directions with small singular values (where the data barely vary) are shrunk most. those are exactly the directions least squares estimates worst (it divides by $s_j$), so shrinking them trades a little bias for a large cut in variance. (topic 02, idea 2)

</details>

6. what Bayesian model makes ridge the posterior mean, and what is $\alpha$ in it?

<details><summary>answer</summary>

$y \mid w \sim \mathcal N(Xw, \sigma^2 I)$ with prior $w \sim \mathcal N(0, \tau^2 I)$: the posterior is Gaussian with mean $(X^\top X + \alpha I)^{-1}X^\top y$, $\alpha = \sigma^2/\tau^2$, the noise-to-signal ratio. (topic 02, idea 1)

</details>

7. why does the lasso set coefficients exactly to zero while ridge doesn't? state the coordinate-descent update.

<details><summary>answer</summary>

the $\ell_1$ penalty has a kink at 0, which can hold a coefficient there against any pull weaker than $\alpha$; the squared penalty is flat at 0 and never has the force to. the update is $w_j \leftarrow S(\rho_j, \alpha)/z_j$ with $\rho_j = \frac1nX_j^\top r + z_jw_j$, $z_j = \frac1n\|X_j\|^2$ and the soft-threshold $S(\rho, t) = \operatorname{sign}(\rho)\max(|\rho| - t, 0)$: zero whenever $|\rho_j| \le \alpha$. (topic 02, idea 3)

</details>

8. two features are near-copies of each other. what do the lasso, ridge and the elastic net do with them?

<details><summary>answer</summary>

the lasso is nearly indifferent to how it splits the weight, so it tends to keep one twin (which one changes between resamples) and zero the other; ridge splits the weight evenly; the elastic net groups them like ridge while still zeroing irrelevant features. the predictions are similar; the coefficients aren't. (topic 02, idea 4 and lab)

</details>

9. state the perceptron convergence theorem. what does it *not* depend on?

<details><summary>answer</summary>

if every $\|x_i\| \le R$ and some unit $w^\star$ separates the data with margin $y_i{w^\star}^\top x_i \ge \gamma > 0$, the perceptron makes at most $(R/\gamma)^2$ updates, in any order. it doesn't depend on the number of points or (directly) on the dimension. with an intercept, apply it to the augmented points $(x_i, 1)$. (topic 03, idea 4)

</details>

10. on data no line separates, why is the averaged perceptron better than the plain one?

<details><summary>answer</summary>

the plain perceptron never stops updating, and each mistake moves the weights by a whole data point, so the final weights depend heavily on when you stop (test accuracy can swing by tens of points between epochs). averaging all the iterates cancels the jumps and gives a stable classifier near the middle of the region they bounce around. (topic 03, idea 3 and lab)

</details>

11. derive the gradient of logistic regression's mean cross-entropy in one line. why is the loss convex?

<details><summary>answer</summary>

the derivative of $-y\log\sigma(z) - (1 - y)\log\sigma(-z)$ with respect to $z$ is $\sigma(z) - y$, so $\nabla L = \frac1nA^\top(s - y)$ (+ the penalty term). the Hessian $\frac1nA^\top\operatorname{diag}(s(1 - s))A$ has nonnegative weights, so it's positive semidefinite. (topic 04, ideas 1–2)

</details>

12. what happens to logistic regression with $\alpha = 0$ on perfectly separable data, and what do the solvers report?

<details><summary>answer</summary>

the minimum doesn't exist: scaling a separating $w$ up drives the loss towards 0 forever. the gradient also shrinks exponentially, so IRLS and gradient descent eventually report "converged" at weights set by `tol` and `max_iters`, with absurdly confident probabilities. a penalty $\alpha > 0$ gives a unique finite optimum. (topic 04, idea 4 and lab)

</details>

13. why is IRLS "iteratively reweighted least squares"? what are the weights and the working response for logistic regression?

<details><summary>answer</summary>

each Newton step solves a weighted (ridge) least-squares problem $(A^\top WA + n\Lambda)\theta = A^\top Wz$ with weights $W = \operatorname{diag}(s(1 - s))$ and working response $z = A\theta + (y - s)/(s(1 - s))$; then the weights and targets are updated and it repeats. (topic 04, idea 4; topic 06, idea 2 for any GLM)

</details>

14. softmax regression has one weight column per class. what can't the data determine about them, and what pins it down?

<details><summary>answer</summary>

adding the same vector to every class's column changes no probability, so the unpenalized loss has a line of equal minima. a ridge penalty selects the representative whose rows sum to zero across classes; gradient descent from zero also stays on it. with two classes, softmax with penalty $\alpha$ is logistic regression with $\alpha/2$. (topic 05, idea 2)

</details>

15. a Poisson GLM fitted to counts gives a Pearson dispersion of 4. are the coefficients wrong? are the standard errors?

<details><summary>answer</summary>

the coefficients are still consistent if the mean model is right (the estimating equation $A^\top(y - \mu) = 0$ only uses the mean). the standard errors assume $\operatorname{Var}(y) = \mu$, so they're too small by about $\sqrt4 = 2$, and 95% intervals cover far less often. multiply them by $\sqrt{\hat\phi}$, or use a negative binomial model, the bootstrap or sandwich standard errors. (topic 06, idea 4 and lab)

</details>

## 2. by hand

1. **ridge and lasso in one dimension.** centered data: $x = (-2, -1, 0, 1, 2)$, $y = (-3, -1, 0, 2, 2)$. compute the least-squares slope, the ridge slope for $\alpha = 3$ (sum scaling: $\|y - xw\|^2 + \alpha w^2$), the lasso slope for $\alpha = 1$ (mean scaling: $\frac{1}{2n}\|y - xw\|^2 + \alpha|w|$), and the lasso's $\alpha_{\max}$.

<details><summary>solution</summary>

$\sum x_i^2 = 10$ and $\sum x_iy_i = 6 + 1 + 0 + 2 + 4 = 13$. least squares: $13/10 = 1.3$. ridge: $13/(10 + 3) = 1$. lasso: $z = 10/5 = 2$, $\rho = 13/5 = 2.6$, so $w = S(2.6, 1)/2 = 1.6/2 = 0.8$. $\alpha_{\max} = |x^\top y|/n = 2.6$: for $\alpha \ge 2.6$ the slope is exactly 0. the intercept is $\bar y = 0$ throughout.

</details>

2. **one Newton step for logistic regression.** $x = (0, 1, 2, 3)$, $y = (0, 0, 1, 1)$, with an intercept, starting from $\theta = 0$. compute the first Newton (IRLS) step. what happens if you keep iterating with $\alpha = 0$?

<details><summary>solution</summary>

at $\theta = 0$ all $s_i = \tfrac12$ and all weights $\tfrac14$, so the step is least squares on the working response $z = 4(y - \tfrac12) = (-2, -2, 2, 2)$. with $\bar x = 1.5$ and $\bar z = 0$: slope $= \frac{\sum(x_i - \bar x)z_i}{\sum(x_i - \bar x)^2} = \frac{3 + 1 + 1 + 3}{5} = 1.6$, intercept $= 0 - 1.6 \cdot 1.5 = -2.4$. so $\theta_1 = (-2.4, 1.6)$, a boundary at $x = 1.5$. the data are separable (any threshold between 1 and 2 classifies them perfectly), so with $\alpha = 0$ further steps keep increasing the slope without bound: the MLE doesn't exist.

</details>

3. **softmax cross-entropy and its gradient.** one example with scores $z = (2, 1, 0)$ for classes $0, 1, 2$, true class 1. compute the probabilities, the loss and the gradient with respect to the scores.

<details><summary>solution</summary>

$e^z = (7.389, 2.718, 1)$, sum 11.107, so $p \approx (0.665, 0.245, 0.090)$. the loss is $-\log p_1 = \log 11.107 - 1 \approx 1.408$. the gradient is $p - e_1 \approx (0.665, -0.755, 0.090)$: raise class 1's score, lower the others in proportion to their probabilities. it sums to zero, as it must.

</details>

## 3. blind rebuild (60 minutes)

with no notes and no looking at your old code, reimplement in a fresh copy of the notebooks:

- **coordinate descent for the lasso and elastic net** (topic 02, stages 3–4): `soft_threshold`, `coordinate_descent` with a maintained residual and the stopping rule, and the `Lasso` and `ElasticNet` estimators with an unpenalized intercept;
- **logistic regression by IRLS** (topic 04, stages 1, 2 and 4): `logistic_loss`, `logistic_gradient`, `logistic_hessian`, and `irls` with capped step halving.

then run the checks:

```bash
pytest 08-linear-models/02-regularization -k "stage3 or stage4"
pytest 08-linear-models/04-logistic-regression -k "stage1 or stage2 or stage4"
```

(copy the notebooks somewhere first, or stash your solutions, so you really start from the stubs.)

## 4. readiness

move on to chapter 09 if you got at least 80% of sections 1 and 2 together (say, 12 of the 15 concept questions and all three problems, or 13 and two of them) and every test in section 3 passes. the same checkpoint works as a placement test: if you pass it cold, skim the lessons for the conventions (ridge's sum scaling versus the lasso's mean scaling, `coef_` shapes `(d,)` and `(d, k)`, `classes_` order, unpenalized intercepts), make sure your `mlfs.linear` passes every topic's tests, and go on.
