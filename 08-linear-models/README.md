# 08 · linear models

> master the models everything else is compared against.

every model in the rest of this course has to beat a linear one, and on a surprising number of problems it doesn't beat it by much. linear models are fast, convex, and come with closed forms or solvers that converge in a handful of steps, a century of statistical theory, and coefficients you can read. they are also where the numerical and statistical traps of machine learning are easiest to see. the textbook least-squares formula loses every digit on a problem QR solves to 8 or 9 digits. a degree-14 polynomial fits 15 points exactly and is off by thousands in between. the lasso's "which features matter" answer flips when you resample. logistic regression reports convergence with weights in the hundreds when a line separates the classes. a Poisson model's 95% intervals cover the truth 57% of the time when the counts are noisier than it assumes. you'll build every model in this family yourself, see each of these failures in your own code, and learn the fix for each: QR and centering, ridge and the lasso with their regularization paths, the perceptron and its convergence theorem, logistic and softmax regression with Newton's method and minibatch SGD, and generalized linear models that fit them all with one loop. the capstone turns raw SMS messages into a spam filter, from tokenizer to error analysis.

**you will build:** `LinearRegression` with normal-equation, QR and gradient-descent solvers, Gaussian maximum likelihood and basis expansions; `Ridge` (closed form, SVD path, effective degrees of freedom), coordinate descent with soft-thresholding, `Lasso`, `ElasticNet`, `ridge_path` and a warm-started `lasso_path`; the `Perceptron`, averaged perceptron, margins and Novikoff's mistake bound; `LogisticRegression` by gradient descent and by IRLS; `SoftmaxRegression` by full-batch gradient descent and minibatch SGD, and a `OneVsRest` wrapper; three exponential families and a `GLM` fitted by IRLS, with $D^2$ and the Pearson dispersion. all of it joins one module, **`mlfs.linear`** (topics 01–06), which chapters 09–13 and 25 build on (kernel ridge, SVM baselines, boosting with logistic loss, Bayesian linear regression, linear probes).
**prerequisites:** chapter 02 (Cholesky, QR, the SVD, conditioning, matrix calculus), chapter 04 (Bernoulli, Poisson and categorical distributions), chapter 05 (maximum likelihood, cross-entropy, the bootstrap and permutation tests), chapter 06 (gradient descent and its rate, Newton's method, SGD), chapter 07 (the estimator protocol, cross-validation, metrics, pipelines, grid search). chapter 00's stable `sigmoid`, `softmax` and `log_softmax` are used everywhere.
**time · scaffolding:** about 70 hours · L2→L3. topics 01–03 are L2 (supported): signatures with detailed docstrings, three hint levels. topics 04–06 are L3 (independent): signatures with short docstrings, two hint levels, and more of the design left to you. the capstone gives you the data loading and the experiment harness; you write the text pipeline, the statistics and the error analysis.
**newly unlocked library components:** after this chapter you may use scikit-learn's `linear_model` (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `lasso_path`, `Perceptron`, `LogisticRegression`, `PoissonRegressor`, `TweedieRegressor`), `multiclass.OneVsRestClassifier`, and `feature_extraction.text.CountVectorizer` / `TfidfTransformer` in labs and experiments, since you've built what they implement. inside `mlfs` modules, later chapters use *your* versions through `mlfs.linear`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md). scikit-learn's kernel methods, neighbours, trees and ensembles stay locked until you've built them.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [linear regression](01-linear-regression/) | `mse_loss`, `mse_gradient`, `solve_normal_equations`, `solve_qr`, `LinearRegression` → `mlfs.linear`, `solve_gd`, `gaussian_nll`, `noise_variance`, `polynomial_basis`, `rbf_basis` | a degree-14 polynomial through 15 points has test RMSE over 10,000 and coefficients of $5 \times 10^{12}$; the normal equations lose every digit by degree 12 while QR keeps 8; scaling one feature by 10 costs gradient descent 100× the steps |
| 02 | [regularization](02-regularization/) | `Ridge`, `ridge_path`, `ridge_dof`, `soft_threshold`, `coordinate_descent`, `Lasso`, `ElasticNet`, `lasso_alpha_max`, `lasso_path` → `mlfs.linear` | not one of 600 ridge coefficients is exactly zero while lasso features enter one at a time; the lasso keeps either one of two twin features at random (42% of resamples zero one); the best $\alpha$ lands exactly on $\sigma^2/\tau^2$ |
| 03 | [the perceptron](03-the-perceptron/) | `perceptron_step`, `Perceptron` (plain and averaged) → `mlfs.linear`, `margin`, `mistake_bound` | the mistake bound holds by factors of 10–240, and the real count jumps when the intercept has to move; on noisy labels the final weights' test accuracy swings from 0.44 to 0.75 between epochs while the averaged ones stay at 0.82 |
| 04 | [logistic regression](04-logistic-regression/) | `logistic_loss`, `logistic_gradient`, `logistic_hessian`, `LogisticRegression` → `mlfs.linear`, `irls` | on separable data IRLS "converges" to $\|w\| = 56$, 162 or 270 depending on `tol`; Newton needs 6 steps where gradient descent needs 905; 40 easy points tilt the least-squares classifier and cost it 7 points of accuracy, logistic none |
| 05 | [softmax regression](05-softmax-regression/) | `softmax_loss`, `softmax_gradient`, `SoftmaxRegression` (full batch and minibatch SGD) → `mlfs.linear`, `OneVsRest` | a linear model on raw pixels gets 0.914 [0.908, 0.919] of MNIST test digits from 20,000 images (0.922 from all 60,000), with weights that look like digit templates; one-vs-rest loses up to 23 points on the middle classes of a line; batches of 100 reach 0.904 in one epoch while full-batch GD, after one step, is at 0.735, and batches of 10 plateau lower than batches of 100 |
| 06 | [generalized linear models](06-generalized-linear-models/) | `Gaussian`, `Bernoulli`, `Poisson`, `glm_loss`, `glm_gradient`, `irls_step`, `GLM` → `mlfs.linear`, `d2_score`, `pearson_dispersion` | least squares predicts negative bike rentals on 1 day in 13 and has twice the Poisson GLM's deviance; with overdispersed counts the textbook 95% interval covers the truth 57% of the time |

## capstone

*a text classifier from raw text*: a tokenizer, a bag-of-words `CountVectorizer` with n-grams, a `TfidfTransformer`, and your `LogisticRegression` in one pipeline, with the penalty chosen by cross-validated F1, ablations (counts versus TF-IDF, bigrams, `min_df`), one look at the test set with bootstrap CIs and a paired permutation test against the majority baseline, and an error analysis of the top features and the most confident mistakes, on the SMS Spam Collection. in the full config the penalty lands on a flat plateau around $\alpha = 10^{-6}$, and the test set gives accuracy 0.985 [0.977, 0.991] against the baseline's 0.866, spam recall 0.913 [0.863, 0.955] and precision 0.971 [0.939, 0.993]. small CPU config: a few seconds (offline). full config: about a minute.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (ridge and lasso in one dimension, a Newton step for logistic regression, a softmax gradient), and a 60-minute blind rebuild of lasso coordinate descent and logistic IRLS.

## notation

| symbol | meaning |
|---|---|
| $X$ (`X`), $x_i$ | the data matrix $(n, d)$, row $i$ |
| $y$, $\hat y$; $Y$ | targets and predictions; one-hot labels $(n, k)$ |
| $w$ (`coef_`), $b$ (`intercept_`) | weights $(d,)$ (or $(d, k)$ for softmax) and intercept |
| $A = [\mathbf 1, X]$, $\theta = (b, w)$ | the design matrix with a ones column, and the stacked parameters (topics 04–06); $\Theta$ $(p, k)$ for softmax |
| $\bar x$, $\bar y$ | column means of $X$ and the mean of $y$ (centering) |
| $L(\theta)$, $\nabla L$, $H$ | the training objective, its gradient and Hessian |
| $\alpha$, $\lambda$ | the penalty strength, and the per-coefficient penalty vector (0 for the intercept) |
| $\kappa$, $s_j$, $u_j$, $v_j$ | condition number; singular values and singular vectors of $X$ |
| $\sigma^2$, $\tau^2$ | noise variance; prior variance of the weights (ridge's Bayesian view) |
| $S(\rho, t)$ | the soft-threshold $\operatorname{sign}(\rho)\max(\lvert\rho\rvert - t, 0)$ |
| $z_j$, $\rho_j$ | in coordinate descent: $\frac1n\|X_j\|^2$ and the partial correlation with the residual |
| $\gamma$, $R$ | a margin; the radius of the data (perceptron) |
| $\sigma(z)$, $\operatorname{softmax}(z)$, $s$, $P$ | the sigmoid and softmax; predicted probabilities |
| $\eta$, $\mu$, $g$, $V(\mu)$ | linear predictor, mean, link function and variance function (GLMs); $\eta$ is also a step size in topics 01 and 05 |
| $W$, $z$ | IRLS weights and working response |
| $d(y, \mu)$, $D^2$, $\phi$ | unit deviance, deviance explained, dispersion |

## how this connects

- builds on: least squares, QR, Cholesky, the SVD and conditioning (ch 02), distributions and maximum likelihood (ch 04–05), gradient descent, Newton's method and SGD (ch 06), the estimator protocol, cross-validation, metrics, pipelines and search (ch 07), stable sigmoid and softmax (ch 00).
- used later in: $k$-NN baselines, kernel ridge regression, SVMs and Gaussian processes, where ridge meets the kernel trick (ch 09); gradient boosting with logistic and Poisson losses (ch 10); PCA and its relation to least squares (ch 11); Bayesian linear and logistic regression (ch 12); margin-based generalization bounds (ch 13); every network's softmax output layer and weight decay (ch 14–15); linear probes on learned representations (ch 25).

## core references

- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, chapters 3 (linear regression, ridge, lasso) and 4 (logistic regression, separating hyperplanes). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Trefethen & Bau (1997), *Numerical Linear Algebra*, SIAM, lectures 11 and 18–19: least squares and its conditioning. [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
- Hastie, Tibshirani & Wainwright (2015), *Statistical Learning with Sparsity: The Lasso and Generalizations*, CRC Press. free online. [hastie.su.domains/StatLearnSparsity](https://hastie.su.domains/StatLearnSparsity/)
- Friedman, Hastie & Tibshirani (2010), "Regularization paths for generalized linear models via coordinate descent", *Journal of Statistical Software* 33(1). [doi:10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- McCullagh & Nelder (1989), *Generalized Linear Models*, 2nd ed., Chapman & Hall/CRC.
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, chapters 3–4: linear models for regression and classification, from the probabilistic side. [microsoft.com/en-us/research/people/cmbishop/prml-book](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
