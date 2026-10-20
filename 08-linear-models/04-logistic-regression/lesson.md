# logistic regression

## why this matters

most prediction problems people care about are yes/no questions: will this customer churn, is this email spam, does this scan show a tumour. a classifier that only says "yes" or "no" throws away the most useful thing you could know, which is *how sure* it is. logistic regression gives you a probability, from a linear model, with a convex loss that has one minimum and a solver that finds it in a handful of steps. it's the default first model for binary classification in medicine, economics and industry, the output layer of nearly every neural network classifier, and the model your capstone uses to classify text.

it also has a sharp edge that surprises people who learned it as "the classification version of linear regression". if a line separates the classes perfectly, there *is* no best model: the loss keeps falling as the weights grow, and your solver will report whatever it had when it stopped. you'll build the loss, its gradient and Hessian, a gradient-descent fit and the Newton method statisticians call **IRLS**, then watch them run into exactly this, and see what regularization does about it.

## warm-up: predict first

1. a model says $P(y = 1 \mid x) = 0.9$ and the label is 0. a second says 0.999999 and the label is 0. how much worse is the second, in cross-entropy?
2. why not fit a line to 0/1 labels with least squares and threshold at 0.5? what could go wrong?
3. is the logistic loss convex in the weights? how about with a penalty?
4. Newton's method on a quadratic converges in one step. on the logistic loss, how many steps should you expect?
5. the training data are perfectly separable. what does maximum likelihood say the weights should be?

## prerequisites check

- [ ] you've built stable `sigmoid` and `log_sigmoid` and know why `np.log(1 + np.exp(z))` overflows (→ ch 00 · topic 03)
- [ ] you know the Bernoulli distribution and maximum likelihood (→ ch 04 · topic 02, ch 05 · topic 02)
- [ ] you know cross-entropy (→ ch 05 · topic 05)
- [ ] you can compute gradients and Hessians of functions of $A\theta$ (→ ch 02 · topic 07)
- [ ] you know Newton's method, its quadratic convergence and why it needs a line search (→ ch 06 · topic 05)
- [ ] you've built `LinearRegression` and `Ridge` (→ topics 01–02)

## core ideas

### idea 1: a probability from a line

a linear score $z = w^\top x + b$ can be any real number; a probability must be in $(0, 1)$. the **logistic** (sigmoid) function maps one to the other:

$$P(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \log\frac{P(y = 1 \mid x)}{P(y = 0 \mid x)} = z.$$

the second form says what the model assumes: the **log-odds** are linear in $x$. each unit of feature $j$ multiplies the odds by $e^{w_j}$, which is how a doctor reads a logistic model ("smoking doubles the odds"). the boundary $P = 1/2$ is $z = 0$, a hyperplane, so logistic regression is a linear classifier like the perceptron, but one that says how far from the boundary each point is, in units of log-odds.

**the loss** comes from maximum likelihood. with $s_i = \sigma(z_i)$, each label is a Bernoulli draw, with probability $s_i^{y_i}(1 - s_i)^{1 - y_i}$. the mean negative log-likelihood is the **cross-entropy** between labels and predictions:

$$L(\theta) = -\frac1n\sum_i \Big[y_i\log s_i + (1 - y_i)\log(1 - s_i)\Big] + \frac12\sum_j \lambda_j\theta_j^2.$$

here $\theta = (b, w)$ stacks the intercept and the weights, the design matrix $A$ has a first column of ones so that $z = A\theta$, and $\lambda = (0, \alpha, \dots, \alpha)$ applies the ridge penalty $\frac\alpha2\|w\|^2$ to the weights but not the intercept (topic 02's reasoning). a confident mistake is expensive: predicting 0.999999 for a 0 costs $-\log(10^{-6}) \approx 13.8$, against 2.3 for predicting 0.9 (warm-up 1).

compute it stably. $\log s = \log\sigma(z)$ and $\log(1 - s) = \log\sigma(-z)$, and your chapter 00 `log_sigmoid` evaluates those without ever forming $e^{|z|}$, which overflows for $|z| > 709$.

**the gradient** is the nicest formula in machine learning. since $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, the derivative of each term with respect to $z_i$ collapses to $s_i - y_i$ (on paper 1), so

$$\nabla L(\theta) = \frac1n A^\top (s - y) + \lambda \odot \theta.$$

"prediction minus target, times the input": the same shape as least squares' $\frac1n A^\top(A\theta - y)$, with the probability in place of the linear prediction. topic 05 and chapter 14 will meet it again.

> **▶ build: stage 1.** implement `logistic_loss(theta, A, y, penalty)` and `logistic_gradient(theta, A, y, penalty)`. use `log_sigmoid` and `sigmoid` from `mlfs.utils`, so scores of $\pm10^4$ give finite answers.

### idea 2: curvature and convexity

differentiate once more. $\partial s_i/\partial\theta = s_i(1 - s_i)a_i$, so

$$\nabla^2 L(\theta) = \frac1n A^\top S A + \operatorname{diag}(\lambda), \qquad S = \operatorname{diag}\big(s_i(1 - s_i)\big).$$

every weight $s_i(1 - s_i)$ is in $(0, \tfrac14]$, so $v^\top\nabla^2L\,v = \frac1n\sum_i s_i(1 - s_i)(a_i^\top v)^2 + \sum_j\lambda_jv_j^2 \ge 0$: the loss is **convex** (warm-up 3). there are no bad local minima, every stationary point is a global minimum, and with $\alpha > 0$ (and an intercept column that isn't all-zero) it's strictly convex, so the minimizer is unique and exists. two more facts fall out:

- **a curvature bound.** since $s(1 - s) \le 1/4$, the Hessian is never larger than $L_{\max} = \lambda_{\max}(A^\top A)/(4n) + \alpha$, so gradient descent with $\eta \le 1/L_{\max}$ is always stable. that's the step size the lab uses.
- **curvature vanishes for confident points.** a point with $s_i \approx 0$ or $1$ contributes almost nothing to the Hessian (and, if it's on the correct side, almost nothing to the gradient). confidently right points stop pulling on the model. this is the root of both the behaviour on separable data (idea 4) and logistic regression's robustness to easy outliers, which least squares lacks (warm-up 2, lab experiment 3).

never build $S$ as an $n \times n$ matrix: $A^\top S A$ is `(A.T * weights) @ A`, scaling the columns of $A^\top$ (the rows of $A$) by the weights.

> **▶ build: stage 2.** implement `logistic_hessian(theta, A, penalty)`. the check compares it with finite differences of your gradient and watches its memory use.

### idea 3: the estimator

`LogisticRegression(*, alpha=0.0, solver="irls", lr=0.1, max_iters=100, tol=1e-08, fit_intercept=True)` wraps all this in chapter 07's `Classifier` protocol. `fit` finds the two classes (any two labels: `classes_[1]` is the one whose probability the model outputs), codes the targets as 0/1, builds $A$ and the penalty vector with the given helpers, minimizes $L$, and unpacks $\theta$ into `intercept_` and `coef_`. `decision_function` returns the logit $z = Xw + b$, `predict_proba` returns the two columns $(\sigma(-z), \sigma(z))$ (note $1 - \sigma(z) = \sigma(-z)$, which is accurate even when $\sigma(z)$ rounds to 1), and `predict` and `score` come free from `Classifier`.

with `solver="gd"`, the minimization is your chapter 06 `gradient_descent` with `lr`, `max_iters` and `tol` passed through, exactly as in topic 01. it works, but each step only uses the slope, so on correlated features it creeps (lab experiment 2). the default solver uses the curvature too.

> **🔮 predict:** you standardize your features and fit with `solver="gd"`, `lr=0.1`. roughly how many steps until the gradient norm is below $10^{-8}$: tens, hundreds, thousands?

> **▶ build: stage 3.** implement `LogisticRegression` with `solver="gd"`: `fit` (setting `classes_`, `coef_`, `intercept_`, `n_iter_` and `converged_`), `decision_function` and `predict_proba`.

### idea 4: Newton's method, alias IRLS

Newton's method minimizes the local quadratic model at each step: $\theta \leftarrow \theta - H^{-1}g$, with $g$ and $H$ the gradient and Hessian at $\theta$. near the minimum it converges **quadratically**: the number of correct digits roughly doubles every step, so 5 to 10 steps are typical, whatever the conditioning (warm-up 4). on this problem each step costs an $O(np^2)$ Hessian and a $p \times p$ solve, cheap for the tens or hundreds of features of most tabular problems.

statisticians know the same algorithm as **iteratively reweighted least squares**. rearranging the Newton step (on paper 3),

$$\theta_{\text{new}} = \big(A^\top SA + n\Lambda\big)^{-1}A^\top S\,u, \qquad u = A\theta + S^{-1}(y - s),$$

which is a *weighted ridge regression* of a "working response" $u$ on $A$, with weights $s_i(1 - s_i)$. each step re-solves a least-squares problem with new weights and new targets. from $\theta = 0$ (all $s_i = \tfrac12$) the first step is ordinary least squares on the targets $4(y - \tfrac12) = \pm 2$: the worked example checks this. the IRLS view is why topic 06 can fit every GLM with the same loop.

pure Newton can overshoot when the quadratic model is poor, far from the minimum. it really happens: with a couple of high-leverage rows (points far out along some direction), a full Newton step from $\theta = 0$ can increase the loss and then run away to weights of $10^{22}$. standardizing the features doesn't help, because Newton's method is affine-invariant: rescaling or rotating the features rescales the step exactly, so the iterates (and the overshoot) are the same. what causes it is a few rows whose leverage dominates the quadratic model. so add **step halving**: try the full step, and halve it until the loss doesn't increase (with a cap on the number of halvings, so a useless direction costs at most ~30 extra loss evaluations). with that safeguard, every step decreases the loss, and once the full step is accepted, convergence is quadratic. one subtlety: near the minimum, the true decrease of a Newton step is far smaller than the rounding error in the loss (about $10^{-16}$ of its value). a strict test `f_new < f` then rejects good steps, and the iteration stalls with a gradient norm around $10^{-9}$ to $10^{-8}$; `f_new <= f` almost never stalls. a small allowance makes it safe: accept a step whose loss is at most $f + 10^{-12}(1 + |f|)$, i.e. no increase, up to rounding.

**separable data.** if some $w$ classifies every training point correctly, then scaling it up, $cw$ with $c \to \infty$, drives every $s_i$ to its label and the loss to 0, without ever reaching it. so with $\alpha = 0$ the minimum doesn't exist: the maximum-likelihood weights are infinite (warm-up 5, on paper 4). your solvers will still stop, because the gradient also goes to 0 (exponentially fast in the margin), and "$\|g\| \le$ `tol`" eventually holds. where they stop depends on `tol` and `max_iters`, not on the data, and the probabilities they report are absurdly confident. the cure is the penalty: with $\alpha > 0$ the objective grows like $\frac\alpha2\|w\|^2$ in every direction, so it has a unique, finite minimizer. this happens far more often than you'd think, because with many features (text, genomics) almost every dataset is separable.

> **🔮 predict:** on separable data with $\alpha = 0$, you run IRLS with `tol=1e-4` and then with `tol=1e-12`. how different will the two $\|w\|$ be?

> **▶ build: stage 4.** implement `irls(A, y, penalty, *, max_iters=100, tol=1e-08)`, returning `(theta, n_steps, converged)`: Newton steps from $\theta = 0$ with capped step halving, stopping on the gradient norm. if the Hessian solve fails, fall back to `np.linalg.lstsq`. `LogisticRegression(solver="irls")` uses it.

## worked example

four points with one feature: $x = (-2, -1, 1, 2)$, $y = (0, 1, 0, 1)$, with an intercept, so $A = [\mathbf 1, x]$ and $\theta = (b, w)$. no penalty.

**at $\theta = 0$.** every $s_i = \tfrac12$, so $L = \log 2 \approx 0.6931$. the gradient is $\frac14A^\top(s - y) = \frac14A^\top(\tfrac12, -\tfrac12, \tfrac12, -\tfrac12)$: the intercept entry is $\frac14(\tfrac12 - \tfrac12 + \tfrac12 - \tfrac12) = 0$, the slope entry $\frac14(-1 + \tfrac12 + \tfrac12 - 1) = -0.25$. so $g = (0, -0.25)$: the loss falls if the slope grows.

**the Hessian** at $\theta = 0$: every weight is $\tfrac14$, and $A^\top A = \begin{pmatrix}4 & 0\\0 & 10\end{pmatrix}$, so $H = \frac{1}{4 \cdot 4}A^\top A = \begin{pmatrix}0.25 & 0\\0 & 0.625\end{pmatrix}$.

**the first Newton step** is $\theta_1 = -H^{-1}g = (0, 0.25/0.625) = (0, 0.4)$. check it the IRLS way: least squares of the targets $4(y - \tfrac12) = (-2, 2, -2, 2)$ on $[\mathbf 1, x]$ gives intercept $0$ (the targets average to 0 and $\bar x = 0$) and slope $\frac{\sum x_it_i}{\sum x_i^2} = \frac{4 - 2 - 2 + 4}{10} = 0.4$. the loss drops to 0.6421.

**the rest.** the next steps give slopes 0.41949 and 0.41962, with gradient norms $1.1 \times 10^{-2}$, $6.7 \times 10^{-5}$, $2.7 \times 10^{-9}$, $10^{-16}$: the exponent roughly doubles each step. the answer, $b = 0$ and $w \approx 0.4196$, is a weak positive slope (two of the four points disagree with it).

the tests check the loss, gradient and Hessian at 0, and the first step.

## on paper

1. **the gradient.** with $s = \sigma(z)$, show that $\frac{d}{dz}\big[-y\log\sigma(z) - (1 - y)\log\sigma(-z)\big] = \sigma(z) - y$, and deduce $\nabla L$.

<details><summary>solution</summary>

$\frac{d}{dz}\log\sigma(z) = \frac{\sigma(z)(1 - \sigma(z))}{\sigma(z)} = 1 - \sigma(z)$, and $\frac{d}{dz}\log\sigma(-z) = -(1 - \sigma(-z)) = -\sigma(z)$. so the derivative is $-y(1 - s) + (1 - y)s = s - y$. by the chain rule with $z_i = a_i^\top\theta$, the data term's gradient is $\frac1n\sum_i(s_i - y_i)a_i = \frac1nA^\top(s - y)$; the penalty adds $\lambda \odot \theta$.

</details>

2. **convexity.** show that the Hessian is $\frac1nA^\top SA + \operatorname{diag}(\lambda)$ and that it is positive semidefinite. when is it positive definite?

<details><summary>solution</summary>

differentiating $\frac1n\sum_i(s_i - y_i)a_i$ with $\partial s_i/\partial\theta = s_i(1 - s_i)a_i$ gives $\frac1n\sum_i s_i(1 - s_i)a_ia_i^\top = \frac1nA^\top SA$. for any $v$, $v^\top Hv = \frac1n\sum_i s_i(1 - s_i)(a_i^\top v)^2 + \sum_j\lambda_jv_j^2 \ge 0$. it is positive definite if $A$ has full column rank (all $s_i(1 - s_i) > 0$ for finite $\theta$), or if every $\lambda_j > 0$; with an unpenalized intercept and $\alpha > 0$, the only direction the penalty misses is the intercept, and the ones column covers it.

</details>

3. **Newton is weighted least squares.** show that $\theta - H^{-1}g$ equals $(A^\top SA + n\Lambda)^{-1}A^\top Su$ with $u = A\theta + S^{-1}(y - s)$ and $\Lambda = \operatorname{diag}(\lambda)$.

<details><summary>solution</summary>

multiply the Newton update by $nH = A^\top SA + n\Lambda$: $nH\theta_{\text{new}} = nH\theta - ng = A^\top SA\theta + n\Lambda\theta - A^\top(s - y) - n\Lambda\theta = A^\top S\big(A\theta + S^{-1}(y - s)\big) = A^\top Su$. so $\theta_{\text{new}} = (A^\top SA + n\Lambda)^{-1}A^\top Su$: the normal equations of a weighted ridge regression of $u$ on $A$ with weights $S$.

</details>

4. **no maximum on separable data.** suppose $w^\star, b^\star$ classify every training point correctly with a positive margin. show that with $\alpha = 0$ the loss along $c(w^\star, b^\star)$ tends to 0 as $c \to \infty$, so the infimum 0 is not attained. what changes for $\alpha > 0$?

<details><summary>solution</summary>

each point has $m_i = (2y_i - 1)(w^{\star\top}x_i + b^\star) > 0$, and its loss is $\log(1 + e^{-cm_i}) \to 0$. every finite $\theta$ has all $s_i \in (0, 1)$ and so a strictly positive loss: the infimum 0 is approached, never reached, and no minimizer exists. with $\alpha > 0$, $L(\theta) \ge \frac\alpha2\|w\|^2$ grows without bound in every direction of $w$ (and the intercept is pinned by the data), so the continuous, strictly convex objective has a unique minimizer.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. perfectly separable data: what IRLS and gradient descent return with $\alpha = 0$, at several tolerances, and with a small penalty.
2. the gradient norm, step by step, for IRLS and for gradient descent on correlated features.
3. least squares as a classifier versus logistic regression, before and after adding points that are easy to classify.

## check your understanding

1. your model outputs 0.73 for an email. what exactly is it claiming, and how would you check whether claims like that are any good?

<details><summary>answer</summary>

that among emails like this one (same features), about 73% are spam. it's a calibration claim: check it with chapter 07's reliability diagram or expected calibration error on held-out data. logistic regression is often well calibrated on data like its training data, because the gradient condition $A^\top(s - y) = 0$ forces the predicted probabilities to sum to the observed counts (overall, and along every feature); a penalty or a distribution shift breaks that.

</details>

2. why does the intercept make the average predicted probability on the training set equal the fraction of positives (with no penalty)?

<details><summary>answer</summary>

the intercept's entry of the gradient is $\frac1n\sum_i(s_i - y_i)$. at the optimum it's 0, so $\frac1n\sum_is_i = \frac1n\sum_iy_i$.

</details>

3. a colleague's logistic regression returns coefficients in the thousands and probabilities of exactly 0 and 1. what's your first guess, and what do you suggest?

<details><summary>answer</summary>

separable (or quasi-separable) training data with no penalty: the weights are wherever the solver stopped. check the training accuracy (100%?), then add a ridge penalty chosen by cross-validation, or remove the feature that separates the data if it's a leak (often it is).

</details>

4. why does IRLS converge in a handful of steps when gradient descent needs hundreds on the same problem?

<details><summary>answer</summary>

Newton's step rescales the gradient by the inverse curvature, which removes the conditioning problem: in a quadratic neighbourhood of the minimum the error is squared every step. gradient descent's progress per step is governed by the condition number of the Hessian, which correlated features make large.

</details>

5. when is IRLS a bad idea?

<details><summary>answer</summary>

when $p$ is large: the Hessian has $p^2$ entries and the solve costs $p^3$, so for 100,000 text features it's impossible. then use L-BFGS (chapter 06), (stochastic) gradient methods, or coordinate descent; scikit-learn's default is L-BFGS. it's also overkill for a quick, loose fit.

</details>

6. you fit logistic regression to 0/1 labels with least squares instead (a "linear probability model"). name two problems.

<details><summary>answer</summary>

the predictions aren't probabilities: they leave $[0, 1]$ for extreme inputs. and the squared loss penalizes points for being *too* correct (a prediction of 3 for a 1 costs as much as $-1$), so easy points far from the boundary drag the line around (lab experiment 3). least squares also implicitly assumes constant noise variance, while a 0/1 outcome has variance $p(1 - p)$.

</details>

7. what is the decision boundary of a logistic regression model, and does the penalty move it?

<details><summary>answer</summary>

the hyperplane $w^\top x + b = 0$, where $P = \tfrac12$. the penalty changes $w$ and $b$, so yes, it moves (and tilts) the boundary, though often less than it changes the probabilities: shrinking $w$ mostly flattens the sigmoid, making predictions less confident.

</details>

## common bugs

- **`np.log(1 + np.exp(z))`.** overflows to `inf` for $z > 709$, and `np.log(sigmoid(z))` gives `-inf` for $z < -745$. symptom: `nan` losses on well-separated data. use `log_sigmoid`.
- **labels as $\pm1$** in a formula written for 0/1 (or the reverse). symptom: the gradient check passes but the fit predicts the wrong class, or the loss is negative.
- **penalizing the intercept.** a penalty vector of all $\alpha$. symptom: a mismatch with scikit-learn that grows with $\alpha$; the mean predicted probability drifts towards $\tfrac12$.
- **an $n \times n$ weight matrix.** `np.diag(s * (1 - s))`. symptom: memory and time explode for a few thousand rows.
- **an uncapped line search.** `while loss(new) > loss(old): t /= 2`. it does terminate, but badly: when no step size helps, `t` halves about 1,075 times until it underflows to 0, and then the "step" is no step at all. symptom: slow fits that silently stop moving (`converged_` False after `max_iters` zero steps). cap the halvings and stop explicitly.
- **`1 - p` for the other class.** computing the class-0 probability as `1 - sigmoid(z)`. symptom: probabilities of exactly 0 where $\sigma(-z)$ would give $10^{-20}$, and an infinite log loss downstream.

## going deeper

- Cox (1958), "The regression analysis of binary sequences", *Journal of the Royal Statistical Society B* 20(2): 215–242. logistic regression's origin. [doi:10.1111/j.2517-6161.1958.tb00292.x](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, section 4.4: logistic regression, IRLS and its ridge-penalized version. [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Albert & Anderson (1984), "On the existence of maximum likelihood estimates in logistic regression models", *Biometrika* 71(1): 1–10. separation, quasi-separation and when the MLE exists. [doi:10.1093/biomet/71.1.1](https://doi.org/10.1093/biomet/71.1.1)
- Soudry, Hoffer, Nacson, Gunasekar & Srebro (2018), "The implicit bias of gradient descent on separable data", *JMLR* 19(70). on separable data, gradient descent's weights diverge in the direction of the max-margin separator. [jmlr.org/papers/v19/18-188](https://www.jmlr.org/papers/v19/18-188.html)
- Murphy (2022), *Probabilistic Machine Learning: An Introduction*, MIT Press, chapter 10. [probml.github.io/pml-book/book1](https://probml.github.io/pml-book/book1.html)

## connections

- ← builds on: stable sigmoids (ch 00), Bernoulli likelihood and cross-entropy (ch 04–05), Newton's method and line search (ch 06), the `Classifier` protocol and calibration metrics (ch 07), least squares and ridge (topics 01–02), the perceptron's margin (topic 03).
- → used in: softmax regression, its multiclass generalization (topic 05), GLMs, which run this same IRLS loop for other distributions (topic 06), the capstone's spam classifier, SVMs and the hinge loss (ch 09), Bayesian logistic regression and the Laplace approximation (ch 12), every network's sigmoid/cross-entropy output layer (ch 14–15), and linear probes (ch 25).
