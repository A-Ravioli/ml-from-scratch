# hyperparameter search

## why this matters

every model you'll meet has knobs that aren't learned by `fit`: the number of neighbours, a polynomial degree, a regularization strength, a learning rate, the depth of a tree. topic 04 showed that one knob can move the test error by an order of magnitude, so you have to choose. "choose by cross-validation" sounds simple, and three things make it subtle.

first, cost. a grid of 5 values for each of 5 hyperparameters is 3,125 settings, each fitted $k$ times. second, *how* you spend a fixed budget matters more than you'd think: Bergstra and Bengio (2012) showed that trying settings at random beats an evenly spaced grid when only a few of the knobs really matter, which is the usual case. third, and most important for this chapter: the score of the winner is not an honest estimate of how good the winner is. you picked it *because* it scored highest among many noisy scores, and topic 01's stump experiment showed what picking does to a score. the honest number needs **nested** cross-validation: the whole search, run inside each outer fold.

you'll build `ParameterGrid`, `GridSearchCV`, `RandomizedSearchCV` and `nested_cross_validate`, completing `mlfs.model_selection`.

## warm-up: predict first

1. you tune 3 hyperparameters with 4 values each by 5-fold CV. how many models do you fit?
2. of two hyperparameters, one matters and one barely does. with a budget of 9 trials, how many different values of the important one does a 3 × 3 grid try? how many do 9 random trials try?
3. a grid search over 20 settings reports a best CV accuracy of 0.87. is 0.87 your estimate of the tuned model's accuracy on new data?
4. how many random trials do you need to have a 95% chance of hitting the top 5% of settings?
5. in nested cross-validation with 5 outer and 5 inner folds and 20 candidates, how many models are fitted?

## prerequisites check

- [ ] you've built `clone`, `get_params` and `set_params`, including `"step__param"` names (→ topic 01; topic 05, idea 6)
- [ ] you've built splitters and `cross_validate`, and know why candidates should share folds (→ topic 02; topic 04, idea 4)
- [ ] you've seen ERM on noise pick a lucky winner (→ topic 01's lab, experiment 3)
- [ ] you can use `itertools.product` (→ ch 00 · topic 01)
- [ ] you know $P(\text{all of } n \text{ independent events}) = \prod P$ (→ ch 04 · topic 01)

## core ideas

### idea 1: the grid

a **grid search** lists values for each hyperparameter and tries every combination: the Cartesian product. `ParameterGrid({"degree": [1, 2], "alpha": [0.1, 1.0]})` yields

$$\{\alpha{:}\,0.1, d{:}\,1\},\ \{\alpha{:}\,0.1, d{:}\,2\},\ \{\alpha{:}\,1.0, d{:}\,1\},\ \{\alpha{:}\,1.0, d{:}\,2\}.$$

the order is a convention, but a fixed one matters: it decides which of several equally good candidates "wins" a tie, and it makes results reproducible across implementations. the convention (scikit-learn's) is to sort the keys and enumerate in row-major order, so the *last* key varies fastest. a list of dicts is a union of grids, for hyperparameters that only make sense together (a `degree` only for the polynomial kernel, say).

the size is the product of the list lengths: $m$ values for each of $h$ hyperparameters gives $m^h$ settings, times $k$ folds of fitting (warm-up 1: $4^3 \times 5 = 320$ fits, plus a final refit). that exponential growth is the grid's fundamental problem.

> **▶ build: stage 1.** implement `ParameterGrid.__iter__` (sorted keys, `itertools.product`, last key fastest, dicts of a list in order; an empty dict yields `{}`) and `__len__`. the given `grids()` checks the input.

### idea 2: search as an estimator

a search is itself an estimator: you give it a model and a space of hyperparameters, `fit(X, y)` runs the cross-validation and keeps the winner, and `predict` uses the winner. the loop:

1. compute the CV splits **once**, so every candidate is scored on the same folds (topic 04's lesson: a paired comparison is far less noisy).
2. for each candidate setting $\theta$: cross-validate `clone(model).set_params(**θ)` and record the mean and standard deviation of its fold scores.
3. pick $\hat\theta = \arg\max$ of the mean score (ties: the first candidate).
4. **refit**: fit `clone(model).set_params(**θ̂)` on *all* the data. the CV models were each trained on $\frac{k-1}{k}$ of it; the final model should use everything.

the results go into fitted attributes: `cv_results_` (the parameters and scores of every candidate, the raw material for plots and sanity checks), `best_params_`, `best_score_`, and `best_estimator_`. because the search is an `Estimator` whose hyperparameters include another estimator, `clone(search)` clones the inner model too (topic 01's recursive clone), which is exactly what nested CV (idea 4) needs.

nothing here is specific to a model, so the shared loop lives in a base class, `BaseSearchCV`, and each kind of search only says which candidates to try (`candidate_params()`). and through `"step__param"` names the candidates can reach into a pipeline: `{"knn__k": [5, 10], "scale__with_std": [True, False]}`.

one practical wrinkle: `fit(X, y)` has no `groups` argument, so a search with `cv=GroupKFold(5)` can't split (it raises "needs groups"). the workaround is to precompute the folds and pass them as a list, which the search accepts like any iterable of `(train, test)` pairs: `cv=list(GroupKFold(5).split(X, y, groups))`.

> **▶ build: stage 2.** implement `BaseSearchCV.fit` (the four steps above), `predict` and `score` (through the refitted winner; `score` uses `scoring` if one was given), and `GridSearchCV.candidate_params`.

### idea 3: random search

**random search** replaces the grid by $n$ independent draws from a distribution over settings: a list means "pick uniformly", and a function `value(rng)` means "draw from this", typically log-uniform for scale parameters, like `lambda g: 10 ** g.uniform(-4, 0)` for a regularization strength.

why would random beat a carefully spaced grid? suppose only one of two hyperparameters matters (warm-up 2). a 3 × 3 grid tries just 3 distinct values of the important one and spends two thirds of its budget re-testing them with different values of the unimportant one. 9 random trials try 9 distinct values of each. in high dimensions the effect is stronger: the response of most models depends on a few hyperparameters, a property Bergstra & Bengio call **low effective dimensionality**, and random search projects to a well-spread sample in *every* subspace, while a grid projects to a handful of points.

random search also has a guarantee that doesn't depend on the dimension at all. if a fraction $q$ of the search space is "good" (say the top 5%), the chance that $n$ independent trials all miss it is $(1 - q)^n$, so

$$P(\text{best of } n \text{ is in the top } q) = 1 - (1 - q)^n, \qquad 1 - 0.95^{59} \approx 0.95,$$

about 60 trials for 95% confidence of the top 5% (warm-up 4), whether there are 2 hyperparameters or 20. the catch is in "top 5% *of the space you sampled*": if your ranges are wrong, so is the search.

> **🔮 predict:** with a budget of 9 trials, a 3 × 3 grid over ($k$, $p$) for a nearest-neighbour model where $p$ barely matters, and a random search over the same ranges: which one finds a better $k$, and how often?

> **▶ build: stage 3.** implement `RandomizedSearchCV.candidate_params`: `n_iter` dicts drawn with one generator `as_rng(self.rng)`, names in sorted order, callables called with the generator, lists sampled uniformly. `fit` is inherited.

### idea 4: the winner's curse, and nested cross-validation

`best_score_` is the maximum of many cross-validation scores, and each score is noisy. even if every candidate were equally good, the maximum would sit above their common true value, by roughly the spread of the scores times a factor that grows with the number of candidates (for $m$ roughly independent noisy scores, about $\sigma\sqrt{2\log m}$). auction theorists call it the **winner's curse**. so `best_score_` is fine for *choosing*, and wrong for *reporting* (warm-up 3). the more candidates, and the noisier the scores (small data), the worse it gets.

the fix applies the leakage rule from topic 02 one level up: the choice of hyperparameters is learned from data, so it must be learned inside the training fold. **nested cross-validation** runs the whole search as the "model" inside an outer cross-validation:

- for each outer fold: run the search (with its own inner CV) on the outer training rows only, refit the winner on them, and score it on the outer test rows.

the outer scores estimate the performance of the *procedure* "tune by inner CV, then refit", the thing you'll actually deploy, and none of them has seen its test rows during tuning. with the pieces you already have, it's one line: `cross_validate(search, X, y, cv=outer)`, because `cross_validate` clones the search per outer fold and the search does the rest. it's expensive (warm-up 5: $5 \times (20 \times 5 + 1) = 505$ fits), and the outer folds may choose *different* hyperparameters, which is fine: you're evaluating the tuning procedure, not one setting. for the final model, run the search once on all the data.

> **🔮 predict:** the labels are coin flips, and a search tries 20 settings of a nearest-neighbour classifier with 5-fold CV on 80 points. what `best_score_` will it report? what will nested CV report?

> **▶ build: stage 4.** implement `nested_cross_validate(search, X, y, *, cv=5, scoring=None)`: the outer test scores of cross-validating the search.

## worked example

a grid search by hand. the model is "predict the training mean plus a shift" with one hyperparameter, the shift $s$, scored by negative MSE with 3 contiguous folds on six targets $y = (1, 3, 2, 6, 5, 7)$ (topic 02's worked example, where $s = 0$ gave fold MSEs $10, 4, 10$).

the training means of the three folds are $5, 4, 3$, and the test targets are $\{1, 3\}, \{2, 6\}, \{5, 7\}$. with shift $s$, the prediction in fold $f$ is $m_f + s$, and the test MSE is the mean of $(y - m_f - s)^2$:

| shift $s$ | fold 1 ($m = 5$) | fold 2 ($m = 4$) | fold 3 ($m = 3$) | mean MSE |
|---|---|---|---|---|
| $-1$ | $\frac{9 + 1}{2} = 5$ | $\frac{1 + 9}{2} = 5$ | $\frac{9 + 25}{2} = 17$ | 9 |
| $0$ | 10 | 4 | 10 | **8** |
| $+1$ | $\frac{25 + 9}{2} = 17$ | $\frac{9 + 1}{2} = 5$ | $\frac{1 + 9}{2} = 5$ | 9 |

the grid $s \in \{-1, 0, 1\}$ picks $s = 0$ with `best_score_` $= -8$, and the refit predicts the overall mean, 4. notice how much each fold's verdict differs: fold 1 alone would have picked $s = -1$, and fold 3 alone $s = +1$. averaging over folds is what makes the choice stable, and the spread between folds is the noise behind the winner's curse.

the tests run your `GridSearchCV` on exactly this example (mean scores $-9, -8, -9$, the refit predicting 4), and check the same bookkeeping, every candidate's mean and standard deviation over the same folds, on other data.

## on paper

1. **grid coverage.** with a budget of $B = m^h$ trials on $h$ hyperparameters, how many distinct values of any *one* hyperparameter does a grid try? how many does random search try (with continuous ranges)?

<details><summary>solution</summary>

the grid tries $m = B^{1/h}$ values per hyperparameter: with $B = 64$ and $h = 3$, only 4. random search tries $B$ distinct values of every hyperparameter (with probability 1 for continuous ranges). if only one hyperparameter matters, random search explores it $B^{1 - 1/h}$ times more finely for the same cost.

</details>

2. **sixty trials.** show that $n \ge \log(1 - c)/\log(1 - q)$ trials are enough to land in the top fraction $q$ with probability $c$. evaluate for $q = 0.05$, $c = 0.95$, and for $q = 0.01$, $c = 0.95$.

<details><summary>solution</summary>

$1 - (1 - q)^n \ge c \iff (1 - q)^n \le 1 - c \iff n \log(1 - q) \le \log(1 - c)$, and dividing by the negative $\log(1 - q)$ flips the inequality. $q = 0.05$: $\log 0.05/\log 0.95 \approx 58.4$, so 59. $q = 0.01$: $\log 0.05/\log 0.99 \approx 298.1$, so 299.

</details>

3. **the winner's curse, in its simplest form.** $m$ candidates all have true score $\mu$, and their CV estimates are independent $\mathcal N(\mu, \sigma^2)$. show that $\mathbb E[\max] > \mu$ for $m \ge 2$, and compute it for $m = 2$.

<details><summary>solution</summary>

$\max(S_1, \dots, S_m) \ge S_1$ always, with strict inequality with positive probability, so $\mathbb E\max > \mathbb E S_1 = \mu$. for $m = 2$: $\max(a, b) = \frac{a + b}{2} + \frac{|a - b|}{2}$, and $a - b \sim \mathcal N(0, 2\sigma^2)$ has $\mathbb E|a - b| = \sqrt{2\sigma^2}\sqrt{2/\pi} = 2\sigma/\sqrt\pi$, so $\mathbb E\max = \mu + \sigma/\sqrt\pi \approx \mu + 0.56\sigma$. for large $m$ it grows like $\sigma\sqrt{2\log m}$.

</details>

4. **the cost of nesting.** count the fits in nested CV with $K$ outer folds, $k$ inner folds and $m$ candidates, including every refit.

<details><summary>solution</summary>

each outer fold runs a search: $mk$ inner fits plus 1 refit of the winner. that's $K(mk + 1)$ in total, plus one more search on all the data ($mk + 1$) if you then build the final model. with $K = k = 5$, $m = 20$: $5 \times 101 = 505$, and $606$ with the final search.

</details>

## experiment

`lab.py` runs your searches on a nearest-neighbour model:

1. grid versus random search with the same budget of 9 trials, when one of two hyperparameters matters.
2. on coin-flip labels: `best_score_` versus nested cross-validation.
3. how often the best of $n$ random trials lands in the top 5%, against $1 - 0.95^n$.

## check your understanding

1. why does `BaseSearchCV.fit` compute the splits once instead of passing `cv` to `cross_validate` for each candidate?

<details><summary>answer</summary>

with a shuffled splitter driven by a `Generator`, each call to `split` gives different folds, so candidates would be compared on different data, adding noise that can reorder close candidates. the same folds make it a paired comparison. (with an int seed or an unshuffled splitter the folds happen to be identical, but the code shouldn't rely on that.)

</details>

2. why refit the best setting on all the data, rather than keeping one of the $k$ CV models?

<details><summary>answer</summary>

each CV model saw only $\frac{k-1}{k}$ of the data; the refit uses all of it, which usually makes a slightly better model. which of the $k$ CV models would you keep anyway? they're all valid and all different. the CV scores were an estimate for the procedure, which the refit then carries out on the full data.

</details>

3. you have a grid over a regularization strength $\alpha \in \{0.001, 0.01, 0.1, 1\}$, and the best is $\alpha = 1$. what should you do?

<details><summary>answer</summary>

extend the range: the optimum may lie beyond the edge of the grid. a winner on the boundary is a sign the search space is wrong. this is also why log-spaced values (and log-uniform random draws) are standard for scale parameters.

</details>

4. the five outer folds of a nested CV chose five different hyperparameter settings. is the nested estimate invalid?

<details><summary>answer</summary>

no. it estimates the performance of the *tuning procedure*, which may legitimately choose different settings on different data. if the choices vary wildly, that says the CV scores are flat or noisy around the optimum, which is useful to know: the exact choice probably doesn't matter much.

</details>

5. random search with 60 trials lands in the top 5% with probability 0.95. why might that still give a bad model?

<details><summary>answer</summary>

"top 5%" is relative to the distribution you sample from. if the ranges are too narrow, or the scale is wrong (uniform on $[0, 1]$ for a parameter that matters on a log scale), the top 5% of that space can be poor. it's also a statement about the CV scores, which are noisy: the true ranking may differ.

</details>

6. you report `best_score_` from a search over 200 settings on 150 examples. what would a careful reviewer ask for?

<details><summary>answer</summary>

a nested CV estimate (or a held-out test set that was never used for tuning), with its uncertainty, compared against a baseline. with 200 candidates and 150 examples, the winner's curse can be several points of accuracy.

</details>

## common bugs

- **different folds per candidate.** see question 1. symptom: rerunning the search changes the winner even with fixed seeds for everything else.
- **tuning the passed-in model.** `self.estimator.set_params(**θ).fit(...)` instead of a clone. symptom: after the search, the model you passed in has the last candidate's settings, and nested CV leaks across folds.
- **reporting `best_score_`.** symptom: numbers that don't hold up on new data, worst for big searches on small data.
- **a new generator per candidate.** `as_rng(self.rng)` inside the loop restarts an int seed, so every random candidate is identical. symptom: `cv_results_` lists the same setting `n_iter` times.
- **forgetting `refit`.** keeping the best CV model instead of refitting, or refitting with the default parameters instead of `best_params_`. symptom: `best_estimator_` disagrees with `best_params_`.
- **uniform draws for scale parameters.** `uniform(0, 1)` for a regularization strength puts 90% of the trials in $[0.1, 1]$. use log-uniform draws.

## going deeper

- Bergstra & Bengio (2012), "Random search for hyper-parameter optimization", *JMLR* 13: 281–305. the low-effective-dimensionality argument and the experiments behind experiment 1. [jmlr.org/papers/v13/bergstra12a](https://www.jmlr.org/papers/v13/bergstra12a.html)
- Cawley & Talbot (2010), "On over-fitting in model selection and subsequent selection bias in performance evaluation", *JMLR* 11: 2079–2107. how large the winner's curse can be, and why nested CV is needed. [jmlr.org/papers/v11/cawley10a](https://www.jmlr.org/papers/v11/cawley10a.html)
- Varma & Simon (2006), "Bias in error estimation when using cross-validation for model selection", *BMC Bioinformatics* 7: 91. nested CV on small biomedical datasets. [doi:10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)
- Snoek, Larochelle & Adams (2012), "Practical Bayesian optimization of machine learning algorithms", *NeurIPS*. using a Gaussian-process model of the score to choose the next trial (chapter 09 builds the GP). [arXiv:1206.2944](https://arxiv.org/abs/1206.2944)
- Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2018), "Hyperband: a novel bandit-based approach to hyperparameter optimization", *JMLR* 18(185): 1–52. stopping bad trials early (chapter 15 builds it). [jmlr.org/papers/v18/16-558](https://www.jmlr.org/papers/v18/16-558.html)

## connections

- ← builds on: `clone` and nested parameters (topics 01 and 05), splitters and `cross_validate` (topic 02), validation curves (topic 04), ERM's optimism on noise (topic 01's lab), probability of independent events (ch 04).
- → used in: the capstone (tuning a pipeline with nested CV), regularization strength for ridge, lasso and logistic regression (ch 08), $k$, kernel widths and $C$ for kernel methods (ch 09, where Gaussian processes also power Bayesian optimization), tree depth and boosting rounds (ch 10), learning-rate schedules and hyperband (ch 15), and the multiple-comparisons view of model selection in research practice (ch 30).
