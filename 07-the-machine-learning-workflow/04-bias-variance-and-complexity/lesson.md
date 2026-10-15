# bias–variance & complexity

## why this matters

you fit a polynomial to 30 noisy points. degree 1 misses the curve. degree 3 looks right. degree 12 passes through nearly every point, has the lowest training error of all, and predicts nonsense between and beyond them. you've seen this picture before. this topic turns it into numbers you can measure: how much of a model's error comes from being too simple (**bias**), how much from being too sensitive to the particular sample it was trained on (**variance**), and how much is noise no model can remove.

that split is the most useful diagnostic in classical machine learning. it tells you whether more data will help (it cures variance, not bias), whether a more flexible model will help (it cures bias and costs variance), and why regularization works (it buys a big drop in variance with a little bias). the two curves you'll build, **learning curves** and **validation curves**, are how you read the diagnosis off real data, where you can't resample the world.

you'll build a polynomial regressor with a complexity knob, a simulation that measures bias² and variance directly, and `learning_curve` and `validation_curve` for `mlfs.model_selection`.

## warm-up: predict first

1. as the degree of a least-squares polynomial goes up, what happens to its *training* error? its test error?
2. you double the training set. which part of the error goes down: bias, variance, or noise?
3. can a model with *higher* bias have *lower* expected error than an unbiased one?
4. a degree-10 polynomial fitted to 30 points: where on $[-1, 1]$ would you trust it least?
5. your training and validation errors are both high and close together. more data, or a more flexible model?

## prerequisites check

- [ ] you know least squares and can fit a polynomial with a Vandermonde matrix (→ ch 02 · topic 04)
- [ ] you know $\mathbb E$, $\mathrm{Var}$, and $\mathrm{Var}(X) = \mathbb E[X^2] - (\mathbb E X)^2$ (→ ch 04 · topic 03)
- [ ] you know what an estimator's bias and variance are, and the MSE = bias² + variance identity for estimating a number (→ ch 05 · topic 01)
- [ ] you've built `clone`, `KFold` and `cross_validate` (→ topics 01–02)
- [ ] you know MSE and $R^2$ (→ topic 03, idea 4)

## core ideas

### idea 1: complexity, and a model with a knob

a **hypothesis class** can be small (constant functions, lines) or large (polynomials of degree 12, deep networks). its **complexity** is, loosely, how many different shapes it can fit. polynomials of degree $d$ in one variable are the cleanest example: the class grows with $d$ (degree $d$ contains degree $d - 1$), and the number of parameters is $d + 1$.

fit by least squares, exactly as in chapter 02: build the design matrix

$$A = \begin{pmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^d\\ \vdots & & & & \vdots\\ 1 & x_n & x_n^2 & \cdots & x_n^d\end{pmatrix}, \qquad \hat c = \arg\min_c \|Ac - y\|^2,$$

and predict $\hat h(x) = \sum_j \hat c_j x^j$. because the classes are nested, the training error can only go down as $d$ grows (the bigger class contains the smaller class's best fit), and at $d = n - 1$ it reaches zero: the polynomial interpolates every point. the test error doesn't follow (warm-up 1). it first drops, as the model becomes flexible enough for the true shape (**underfitting** ends), then rises, as the extra flexibility is spent fitting noise (**overfitting** begins). to study that trade-off you need a model whose complexity is one hyperparameter, and `PolynomialRegressor(degree)` is it.

> **▶ build: stage 1.** implement `PolynomialRegressor(degree=1)`, a `Regressor` for one feature: `fit` solves least squares on `np.vander(x, degree + 1, increasing=True)` (with `np.linalg.lstsq`, the fast equivalent of your chapter 02 `lstsq`) and stores `coef_`, lowest power first; `predict` evaluates the polynomial.

### idea 2: the bias–variance decomposition

imagine the whole process as random: a training set $D$ of $n$ points is drawn, a model $\hat h_D$ is fitted, and it's tested at a point $x$ where the target is $y = f(x) + \varepsilon$, with noise $\mathbb E\varepsilon = 0$, $\mathrm{Var}\,\varepsilon = \sigma^2$, independent of $D$. the expected squared error at $x$, over both the training set and the noise, splits into three parts:

$$\mathbb E_{D,\varepsilon}\big[(y - \hat h_D(x))^2\big] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{\big(f(x) - \bar h(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb E_D\big[(\hat h_D(x) - \bar h(x))^2\big]}_{\text{variance}}, \qquad \bar h(x) = \mathbb E_D\,\hat h_D(x).$$

(on paper 1 derives it: add and subtract $f(x)$ and $\bar h(x)$, and the cross terms vanish.) each term answers a different question:

- **noise**: how unpredictable $y$ is given $x$. no model can do better than $\sigma^2$.
- **bias²**: how far the *average* model is from the truth. it measures what the hypothesis class can't represent (plus systematic effects of the fitting procedure, like regularization).
- **variance**: how much the fitted model changes from one training set to another. it measures how much the model chases the particular sample.

average over the test points $x$ and you get the decomposition of the expected test MSE. in real life you can't compute it, because you get one $D$ and don't know $f$. in a simulation you can do it directly: draw many training sets, fit a fresh clone on each, and look at the spread of the predictions. with $R$ rounds and predictions $\hat h_r(x)$ at test points $x_1..x_m$:

$$\widehat{\text{bias}^2} = \frac1m\sum_j\big(\bar h(x_j) - f(x_j)\big)^2, \qquad \widehat{\text{variance}} = \frac1m\sum_j \frac1R\sum_r\big(\hat h_r(x_j) - \bar h(x_j)\big)^2,$$

with $\bar h(x_j) = \frac1R\sum_r \hat h_r(x_j)$. the two pieces add up *exactly* to the average squared distance between the fits and $f$, so the expected error is their sum plus $\sigma^2$.

one caution about this estimator: $\bar h$ is itself an average of only $R$ noisy fits, so it misses the true average model by a random amount with variance $\text{variance}/R$ at each point, and squaring that error makes it *positive on average*. the plug-in bias² is biased upwards:

$$\mathbb E\big[\widehat{\text{bias}^2}\big] = \text{bias}^2 + \frac{\text{variance}}{R}.$$

when the variance is small this is negligible; when it's enormous (an overfit model), the "bias" you measure can be mostly variance in disguise. subtracting $\widehat{\text{variance}}/(R - 1)$ removes the extra term on average.

> **🔮 predict:** you can observe $f(x) = x$ at $x = 0$ and $x = 1$, with noise of variance 1. which predicts better on average at those two points: the line through both observations (unbiased), or the constant at their mean (biased)?

> **▶ build: stage 2.** implement `bias_variance_decomposition(estimator, sample_data, X_test, f_test, *, n_rounds=100, noise_var=0.0, rng=None)`: `n_rounds` times, draw `X, y = sample_data(gen)`, fit `clone(estimator)`, and predict at `X_test`; return `bias2`, `variance`, `noise` and `expected_error` as floats.

### idea 3: learning curves

the decomposition explains *why* models fail; **learning curves** let you see which failure you have on real data. train on increasing fractions of the training set and record the training error and the validation error (from cross-validation) at each size.

- **high bias** (underfitting): both curves flatten quickly and meet at a plateau well above the noise level. the training error is already as high as the validation error, so the model isn't even fitting the data it sees. more data won't help: the plateau *is* the bias (plus noise). you need a richer model or better features (warm-up 5).
- **high variance** (overfitting): the training error is far below the validation error, and the gap shrinks as data grows. more data helps, and so would regularization or a simpler model.
- **about right**: the curves converge near the noise floor.

why does more data cure variance but not bias? for least squares with $p$ parameters, the variance of the fit averaged over the training inputs is exactly $\sigma^2 p/n$ (on paper 3): it falls like $1/n$. the bias is a property of the hypothesis class and doesn't depend on $n$ at all.

the implementation is cross-validation in a loop: for each fold and each size, fit on the first `size` rows of the fold's training indices and score on those rows and on the fold's test rows. use the *smallest* training fold to decide what "100%" means, so every fold can supply every size, and pass a *shuffled* splitter, because the subsets are prefixes of each fold.

> **🔮 predict:** a straight line fitted to $\sin(\pi x)$ plus noise: draw its training and validation error as the training set grows from 30 to 300 points. where do the two curves end up?

> **▶ build: stage 3.** implement `learning_curve(estimator, X, y, *, train_sizes=(0.1, 0.325, 0.55, 0.775, 1.0), cv=5, scoring=None)` on top of your `cross_validate`, returning `(sizes, train_scores, val_scores)` with score arrays of shape `(n_sizes, n_folds)`. the given helper `materialize_splits` turns `cv` into a list of `(train, test)` pairs.

### idea 4: validation curves, and choosing complexity

a **validation curve** sweeps one hyperparameter instead of the training size: for each value, cross-validate a clone with that value set, and record training and validation scores. for polynomial degree it traces the U-shape from idea 1: training error falling forever, validation error falling, bottoming out, rising. the best validation score is the natural choice of hyperparameter.

three cautions.

**use the same folds for every value.** if each value is scored on different random folds, some of the differences between values are just differences between folds. compute the splits once and reuse them.

**the curve is noisy.** with 30 points, the validation error near the bottom of the U is often flat within its own noise, and the minimum jumps around between datasets. a common rule of thumb, the **one-standard-error rule**, picks the *simplest* model whose score is within one standard error of the best (Hastie, Tibshirani & Friedman, 2009, §7.10). it trades a little expected fit for a lot of stability.

**choosing is fitting.** the best validation score of the chosen value is optimistically biased, for the same reason the best stump on noise looked good in topic 01: you picked the maximum of noisy estimates. that's harmless for *choosing*, but it must not be reported as the model's performance. topic 06 builds the honest version, nested cross-validation.

> **▶ build: stage 4.** implement `validation_curve(estimator, X, y, param_name, param_range, *, cv=5, scoring=None)`: one `clone(estimator).set_params(**{param_name: v})` per value, all scored by `cross_validate` on the *same* materialized splits, returning `(train_scores, val_scores)`, each of shape `(n_values, n_folds)`.

## worked example

the 🔮 from idea 2, by hand. the truth is $f(x) = x$, observed at $x = 0$ and $x = 1$: $y_0 = \varepsilon_0$, $y_1 = 1 + \varepsilon_1$, with independent noise of variance $\sigma^2 = 1$. we test at the same two inputs.

**the line through both points** predicts $\hat h(0) = y_0$ and $\hat h(1) = y_1$. its average is $\bar h(0) = 0$, $\bar h(1) = 1$: no bias. its variance at each point is $\mathrm{Var}(\varepsilon) = 1$. expected error: $0 + 1 + \sigma^2 = 2$.

**the constant** predicts $\frac{y_0 + y_1}{2} = \frac12 + \frac{\varepsilon_0 + \varepsilon_1}2$ everywhere. its average is $\frac12$ at both points, so its bias² is $\frac12\big((\tfrac12 - 0)^2 + (\tfrac12 - 1)^2\big) = 0.25$. its variance is $\mathrm{Var}\big(\frac{\varepsilon_0 + \varepsilon_1}{2}\big) = \frac{\sigma^2}{2} = 0.5$. expected error: $0.25 + 0.5 + 1 = 1.75$.

the wrong model wins (warm-up 3): averaging two noisy points halves the variance, and that's worth more than the bias it introduces. with less noise the line wins; on paper 2 finds the crossover. the tests run your `bias_variance_decomposition` on exactly this setup and check bias² ≈ 0.25 and variance ≈ 0.5 for the constant, bias² ≈ 0 and variance ≈ 1 for the line.

## on paper

1. **the decomposition.** derive it. write $y - \hat h_D(x) = \varepsilon + (f(x) - \bar h(x)) + (\bar h(x) - \hat h_D(x))$, square, take expectations over $\varepsilon$ and $D$, and show that every cross term is zero.

<details><summary>solution</summary>

the three terms are: $\varepsilon$ (zero mean, independent of $D$), the constant $b = f(x) - \bar h(x)$, and $v_D = \bar h(x) - \hat h_D(x)$ (zero mean over $D$ by definition of $\bar h$). cross terms: $\mathbb E[\varepsilon b] = b\,\mathbb E\varepsilon = 0$; $\mathbb E[\varepsilon v_D] = \mathbb E\varepsilon\,\mathbb E v_D = 0$ by independence; $\mathbb E[b\,v_D] = b\,\mathbb E_D v_D = 0$. what's left is $\mathbb E\varepsilon^2 + b^2 + \mathbb E v_D^2 = \sigma^2 + \text{bias}^2 + \text{variance}$.

</details>

2. **the crossover.** in the worked example with general noise variance $\sigma^2$, show that the constant's expected error is $0.25 + 1.5\sigma^2$ and the line's is $2\sigma^2$. for which $\sigma^2$ is the constant better?

<details><summary>solution</summary>

constant: bias² $0.25$, variance $\sigma^2/2$, noise $\sigma^2$. line: bias² 0, variance $\sigma^2$, noise $\sigma^2$. the constant wins when $0.25 + 1.5\sigma^2 < 2\sigma^2$, i.e. $\sigma^2 > 0.5$. the noisier the data, the more a simple, biased model is worth.

</details>

3. **variance of least squares.** for a fixed design $A$ ($n \times p$, full column rank) and $y = Ac^* + \varepsilon$ with $\mathrm{Cov}(\varepsilon) = \sigma^2I$, show that the fitted values $\hat y = A\hat c$ have $\mathrm{Cov}(\hat y) = \sigma^2 H$ with $H = A(A^\top A)^{-1}A^\top$, and that their variance averaged over the $n$ training points is $\sigma^2 p/n$.

<details><summary>solution</summary>

$\hat y = Hy = Ac^* + H\varepsilon$, so $\mathrm{Cov}(\hat y) = H\,\sigma^2 I\,H^\top = \sigma^2 H$ (a projection: $H^\top = H$, $H^2 = H$). the average variance is $\frac{\sigma^2}{n}\operatorname{tr}H$, and $\operatorname{tr}H = \operatorname{tr}\big((A^\top A)^{-1}A^\top A\big) = \operatorname{tr}I_p = p$. each extra parameter adds $\sigma^2/n$ of variance; doubling $n$ halves it. the diagonal $H_{ii}$ (the **leverage**) says which points get more: for polynomials, the ones near the edges of the data.

</details>

4. **averaging $k$ neighbours.** a regressor predicts at $x$ the mean of the $y$'s of the $k$ training points nearest to $x$ (chapter 09's $k$-NN). treating those $k$ inputs as fixed, what are its variance and its bias at $x$? how do they move with $k$?

<details><summary>solution</summary>

the prediction is $\frac1k\sum_{i \in N_k(x)}(f(x_i) + \varepsilon_i)$. variance: $\sigma^2/k$. bias: $\frac1k\sum_{i \in N_k(x)} f(x_i) - f(x)$, the error of averaging $f$ over a neighbourhood, which grows as $k$ grows (the neighbours spread out). small $k$: low bias, high variance; large $k$: the opposite. $k$ plays the role of $1/\text{degree}$.

</details>

## experiment

`lab.py` uses your code on $y = \sin(\pi x) + \text{noise}$:

1. bias², variance and expected error against polynomial degree, with the single-dataset CV curve from your `validation_curve` on top.
2. 25 fits of the same degree on different training sets: where the variance lives.
3. learning curves for degrees 1, 3 and 9.

## check your understanding

1. the training error of a degree-$d$ least-squares polynomial never increases with $d$. why not?

<details><summary>answer</summary>

degree $d$ contains every polynomial of degree $d - 1$ (set the top coefficient to zero), so the least-squares minimum over the bigger class can't be larger. that's true for any nested family fitted by exact ERM, and it's why training error can't choose complexity.

</details>

2. why does the bias–variance decomposition need *many training sets* while ordinary evaluation needs only one?

<details><summary>answer</summary>

bias and variance are properties of the fitting *procedure* over the distribution of training sets: $\bar h$ is an average over $D$, and the variance is the spread over $D$. one training set gives one fitted model, which shows the total error but can't separate the parts. in practice learning curves and validation curves are the proxies.

</details>

3. a model's learning curves show training MSE 0.02 and validation MSE 0.40 at the largest size, with the gap still shrinking. what would you try?

<details><summary>answer</summary>

that's high variance: more data (the gap is still closing), regularization, fewer features, or a simpler model. a more complex model would make it worse.

</details>

4. why must `validation_curve` reuse the same folds for every parameter value?

<details><summary>answer</summary>

so that differences between values reflect the values, not the luck of different splits. with a shuffled splitter that draws from a random generator, each call to `split` gives new folds; scoring each value on different folds adds noise that can reorder close candidates. it's a paired comparison, like the paired tests of chapter 05.

</details>

5. regularization (chapter 08) adds bias on purpose. why would anyone want that?

<details><summary>answer</summary>

because it removes more variance than the bias it adds, whenever the variance is the larger problem: few data points, many parameters, noisy targets. the worked example is the smallest case: the constant is a "maximally regularized" line, and it wins when $\sigma^2 > 0.5$.

</details>

6. bias and variance were defined for squared error. does the same trade-off exist for classification with 0–1 loss?

<details><summary>answer</summary>

the intuition carries over (simple classes underfit, flexible ones are unstable), but the clean additive decomposition doesn't: for 0–1 loss, variance can *reduce* the error of a biased classifier at some points. several competing definitions exist (Domingos, 2000). for probabilistic predictions scored by log loss, a decomposition along the same lines holds.

</details>

## common bugs

- **fitting the same estimator object every round.** `estimator.fit(...)` in the loop instead of `clone(estimator).fit(...)`: the predictions are fine, but anything stateful leaks between rounds, and the caller's object ends up fitted. clone it.
- **variance with `ddof=1`.** `np.var(..., ddof=1)` in the variance term breaks the exact identity bias² + variance = mean squared distance to $f$. use the population variance (`ddof=0`).
- **noisy targets as `f_test`.** passing $y$ (with noise) instead of $f(x)$ moves the noise into the bias term. bias² is measured against the noiseless function.
- **unshuffled learning curves.** prefixes of an unshuffled fold of sorted data are one end of the input range: the "small training set" is also a *different* distribution, and the curve shows extrapolation error.
- **different folds per value.** passing a shuffled splitter with a `Generator` and calling `split` once per value. symptom: a jagged validation curve whose minimum moves when you rerun it.

## going deeper

- Geman, Bienenstock & Doursat (1992), "Neural networks and the bias/variance dilemma", *Neural Computation* 4(1): 1–58. the paper that made the decomposition a central idea in machine learning. [doi:10.1162/neco.1992.4.1.1](https://doi.org/10.1162/neco.1992.4.1.1)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, sections 2.9 and 7.2–7.3 (model complexity, the bias–variance decomposition), 7.10 (cross-validation and the one-standard-error rule). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Domingos (2000), "A unified bias-variance decomposition and its applications", *ICML*. bias and variance for general losses, including 0–1 loss. [homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf](https://homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf)
- Belkin, Hsu, Ma & Mandal (2019), "Reconciling modern machine-learning practice and the classical bias–variance trade-off", *PNAS* 116(32): 15849–15854. "double descent": what happens past the interpolation threshold (chapter 19 picks this up). [doi:10.1073/pnas.1903070116](https://doi.org/10.1073/pnas.1903070116)
- Perlich, Provost & Simonoff (2003), "Tree induction vs. logistic regression: a learning-curve analysis", *JMLR* 4: 211–255. learning curves crossing: which model is better depends on $n$. [jmlr.org/papers/v4/perlich03a](https://www.jmlr.org/papers/v4/perlich03a.html)

## connections

- ← builds on: least squares and Vandermonde matrices (ch 02), expectations and variances (ch 04), bias and variance of estimators (ch 05), `clone`, splitters and `cross_validate` (topics 01–02), MSE (topic 03).
- → used in: hyperparameter search (topic 06, which is a validation curve over many parameters), regularization paths for ridge and lasso (ch 08), the $k$ of $k$-NN and kernel bandwidths (ch 09), bagging as variance reduction and boosting as bias reduction (ch 10), learning theory's approximation/estimation trade-off (ch 13), and double descent and scaling laws (ch 19).
