# what is learning?

## why this matters

here's a classifier that gets 100% accuracy on every dataset you give it: store each training example with its label, and when asked about an input, look it up. it's also useless: on a new input it can only guess.

that gap between "fits the data I have" and "works on the data I'll get" is what machine learning is about. this chapter builds the tools for measuring it honestly. so before any clever model, you need three things: a precise statement of what "learning" means, a common interface every model speaks (so one evaluation routine works for all of them), and **baselines**, the dumbest possible models, so that a number like "87% accuracy" means something.

in this topic you build `mlfs.base`: the `Estimator` protocol (`fit` / `predict` / `score`), the base classes that every classical model from chapter 08 to chapter 13 subclasses, `clone`, and the dummy baselines. then you build the lookup table and its opposite, a model that can only draw one line, and find out which one generalizes.

## warm-up: predict first

1. a model gets 100% accuracy on its training data. what does that tell you about its accuracy on new data?
2. a test for a disease that 5% of patients have is right 95% of the time. is that a good test?
3. a regressor always predicts the mean of the training targets. what's its $R^2$ on the training data? can $R^2$ ever be negative?
4. you have 100,000 labelled examples, each described by 10 yes/no features. will a lookup table generalize?
5. the labels are coin flips, unrelated to the 200 features. you search over every "one feature, one threshold" rule for the one with the best training accuracy. roughly what training accuracy do you find with 50 examples?

## prerequisites check

- [ ] you can write a Python class with `__init__`, methods and `super()` (→ ch 00 · topic 01)
- [ ] you can index, sort and `cumsum` NumPy arrays without loops (→ ch 00 · topic 02)
- [ ] you know `as_rng` and why randomness is passed in, never global (→ ch 00 · topic 05)
- [ ] you can compute an expectation and know that the mean minimizes expected squared error (→ ch 04 · topic 03)
- [ ] you know Hoeffding's inequality: an average of $n$ bounded variables is within $O(1/\sqrt n)$ of its mean with high probability (→ ch 04 · topic 06)

## core ideas

### idea 1: the learning problem, and what goes in

**supervised learning** starts from examples $(x_i, y_i)$, $i = 1..n$: an input $x_i \in \mathbb{R}^d$ (a row of **features**) and a **target** $y_i$, either a real number (**regression**) or one of $k$ labels (**classification**). the examples are assumed to be independent draws from some unknown distribution $P(x, y)$, the **data-generating distribution**. the goal is a function $h$ that predicts $y$ from $x$ well *on new draws from $P$*. **unsupervised learning** gets only the $x_i$ and looks for structure: clusters, low-dimensional directions, densities (chapter 11).

in code, the inputs are a matrix `X` of shape `(n_samples, n_features)`, one row per example and one column per feature, and a vector `y` of shape `(n_samples,)`. the convention prevents real bugs: a 1-D `x` that should have been a column, a `y` of shape `(n, 1)` that broadcasts against predictions of shape `(n,)` into an $n \times n$ matrix, a `NaN` that spreads silently.

so every `fit` starts the same way: validate and standardize the input, and **fail loudly**. a model that trains on garbage and returns garbage costs you an afternoon. a `ValueError` that says "X has 100 rows but y has 99" costs you ten seconds.

> **🔮 predict:** `y` has shape `(5, 1)` and your predictions have shape `(5,)`. what's the shape of `y - y_pred`? what does `np.mean((y - y_pred)**2)` compute?

> **▶ build: stage 1.** implement `check_X_y(X, y=None)`: X to a float64 `(n, d)` array (a 1-D `X` is one feature, so it becomes a column), y to a 1-D array of the same length (a `(n, 1)` column is flattened, string labels are kept), and a `ValueError` for anything else: NaN or inf, mismatched lengths, wrong numbers of dimensions, no rows.

### idea 2: estimators, hyperparameters, and clone

a model has two kinds of numbers. **hyperparameters** are chosen before seeing data: the degree of a polynomial, the number of neighbours, a regularization strength. **parameters** (or the fitted state) are learned from data: the polynomial's coefficients, the stored training set. confusing the two is the root of many evaluation bugs, so the protocol keeps them apart by construction:

- `__init__` takes only hyperparameters and stores each one **unchanged** as an attribute of the same name. no validation, no computation, no data.
- `fit(X, y)` learns, stores everything it learned in attributes whose names end in `_` (`coef_`, `classes_`, `constant_`), and returns `self`, so you can write `model.fit(X, y).predict(X_new)`.
- `predict(X)` and `score(X, y)` use the fitted state.

why so strict? because of what the rest of the course does with estimators. cross-validation fits the same model five times on five different training sets. hyperparameter search fits it hundreds of times with different settings. each of those fits needs a **fresh, unfitted copy with the same hyperparameters**, and to make one, code that knows nothing about your model has to be able to read and write its hyperparameters. that's what `get_params` and `set_params` are for, and `clone` is built on them:

$$\texttt{clone(est)} = \texttt{type(est)(**est.get\_params())}.$$

`get_params` doesn't need you to list anything: it reads the parameter names from the signature of your class's `__init__` (with `inspect.signature`) and looks each one up with `getattr`. this is why `__init__` must store its arguments under the same names: the signature *is* the list of hyperparameters. if `__init__` computed something from `alpha` and stored only that, `get_params` would fail, and `clone` would quietly lose the setting.

two details make `clone` safe. hyperparameters can themselves be estimators, so `clone` recurses into them, including estimators inside lists and tuples. everything else is deep-copied, so a clone never shares a mutable list or a random generator with the original. what you must *not* do is `copy.deepcopy(est)`: that would copy the fitted state too: a "fresh" model in fold 2 would already know fold 1.

`set_params(alpha=0.1)` is just `setattr` with a safety check: a misspelled name raises `ValueError`, because a search over `alhpa` that silently does nothing is the worst kind of bug. a name with a double underscore, `"step__alpha"`, reaches into a sub-estimator: `self.step.set_params(alpha=0.1)`. that's how you'll tune a model inside a pipeline in topic 05.

> **▶ build: stage 2.** implement `Estimator.get_params`, `Estimator.set_params` and `clone`. `clone` copies each hyperparameter: estimators (anything with a `get_params` method) recursively, lists, tuples and dicts element by element, everything else with `copy.deepcopy`. it skips keys containing `"__"`, which describe sub-estimators rather than constructor arguments.

### idea 3: losses, risk, and score

to say what a "good" $h$ is, pick a **loss** $\ell(y, \hat y)$: the cost of predicting $\hat y$ when the truth is $y$. the classics are the **squared loss** $(y - \hat y)^2$ for regression and the **0–1 loss** $\mathbb{1}[y \ne \hat y]$ for classification. the quantity you care about is the **risk**, the expected loss on a new draw:

$$R(h) = \mathbb{E}_{(x, y) \sim P}\big[\ell(y, h(x))\big],$$

and the only thing you can compute is the **empirical risk** on your $n$ examples:

$$\hat R(h) = \frac1n \sum_{i=1}^n \ell(y_i, h(x_i)).$$

the whole chapter lives in the difference between those two.

every model reports a default **score**, higher is better, so generic code can compare models without knowing what they are. classifiers report **accuracy**, $1 - \hat R_{0\text{-}1}$. regressors report the **coefficient of determination**:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2} = 1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}.$$

$R^2$ is the squared error measured in units of "the error of always predicting $\bar y$". a perfect model scores 1, the mean predictor scores 0, and a model worse than the mean scores below zero, with no lower limit (warm-up 3). if all the $y_i$ are equal, $\text{SS}_\text{tot} = 0$, and the convention is 1.0 for a perfect prediction and 0.0 otherwise.

classifiers have one more convention, and it matters for everything downstream: `fit` stores `classes_`, the **sorted** distinct labels, and `predict_proba(X)` returns an $(n, k)$ matrix whose column $j$ is the probability of `classes_[j]`. then `predict` is the same for every classifier, the label of the most probable column:

$$\hat y = \texttt{classes\_}\big[\arg\max_j \texttt{predict\_proba}(x)_j\big].$$

labels can be anything sortable: integers, strings like `"spam"`. metrics know where "the positive class" is (column 1 of a binary classifier), and ensembles can average probabilities column by column. transformers (scalers, encoders, PCA) share a different bit of behaviour: `fit_transform(X, y)` is `fit(X, y).transform(X)`.

> **▶ build: stage 3.** implement `Regressor.score` ($R^2$), `Classifier.predict` (argmax of `predict_proba`, returning labels, not column indices), `Classifier.score` (accuracy) and `Transformer.fit_transform`. these are the methods every model in chapters 08–13 inherits.

### idea 4: baselines

"my model has 95% accuracy" is meaningless on its own. if 95% of emails aren't spam, a filter that lets everything through also has 95% accuracy (warm-up 2). every number needs a **baseline**: the score of the simplest thing that could possibly work, computed on the same data with the same protocol. the simplest things ignore $X$ entirely:

- **regression.** predict a constant. the constant that minimizes the squared loss on the training set is the **mean** $\bar y$; for the absolute loss it's the **median**. the mean predictor has $R^2 = 0$ on its training data, by definition.
- **classification.** with class frequencies $p_1, \dots, p_k$ (the **prior**):
  - `most_frequent`: always predict the majority class. accuracy $\max_j p_j$.
  - `prior`: report the frequencies as probabilities. same predictions, but a meaningful `predict_proba`, which matters for log loss in topic 03.
  - `stratified`: guess a class at random in proportion to the prior. expected accuracy $\sum_j p_j^2$, which is *worse* than `most_frequent` unless the classes are balanced (on paper 2).
  - `uniform`: guess uniformly at random. expected accuracy $1/k$.

a model that doesn't beat `most_frequent` by more than the noise in your estimate hasn't learned anything reliable. from this chapter on, every number you report comes with a baseline and an uncertainty estimate.

> **🔮 predict:** 70% of your training labels are class 0. what accuracy does `stratified` get, on average? and `most_frequent`?

> **▶ build: stage 4.** implement `DummyRegressor` (`mean`, `median`) and `DummyClassifier` (`most_frequent`, `prior`, `uniform`, `stratified`). they use $X$ only for its length. the random strategies draw from `rng_ = as_rng(self.rng)`, created in `fit`, so a fixed seed gives reproducible guesses.

### idea 5: empirical risk minimization, and the memorizer

**empirical risk minimization** (ERM) is the template for almost every learning algorithm in this course: fix a set of candidate functions, the **hypothesis class** $\mathcal H$, and pick the one with the smallest training loss:

$$\hat h = \arg\min_{h \in \mathcal H} \hat R(h).$$

linear regression is ERM over linear functions with the squared loss. a neural network is (approximately) ERM over a huge parametric class. the choice of $\mathcal H$ is everything, and the two extremes show why.

**the lookup table** takes $\mathcal H$ = *every* function. it memorizes each training row's label, so $\hat R = 0$ (unless identical rows have different labels). on a new row it has no information at all and falls back to the majority class. with continuous features, every test row is new, so it's exactly the `most_frequent` baseline in disguise: perfect on the training set, a coin flip (or worse) on everything else.

**the decision stump** takes a tiny $\mathcal H$: one feature $j$, one threshold $t$, one direction $s \in \{+1, -1\}$,

$$h(x) = \begin{cases}\text{class 1} & \text{if } s\,(x_j - t) > 0\\ \text{class 0} & \text{otherwise.}\end{cases}$$

ERM over stumps is exact and cheap. on a feature with sorted values $v_1 < v_2 < \dots$, any two thresholds between the same pair of neighbours make identical predictions, so only the midpoints $(v_i + v_{i+1})/2$ (plus one threshold below everything) need checking. sort once, and the mistakes for *every* split come from two cumulative sums: with the $i$ smallest values on the left and $s = +1$, the mistakes are the positives on the left plus the negatives on the right.

why can the small hypothesis class (stumps) beat the big one (every function)? the training error $\hat R(h)$ of a *fixed* $h$ is an average of $n$ independent 0–1 losses, so by Hoeffding's inequality it's within $\sqrt{\log(2/\delta)/(2n)}$ of $R(h)$ with probability $1 - \delta$. but ERM doesn't evaluate a fixed $h$: it picks the best of many, and the best of many noisy estimates is optimistically biased. a union bound over a finite class fixes this, for every $h \in \mathcal H$ at once:

$$R(h) \le \hat R(h) + \sqrt{\frac{\log|\mathcal H| + \log(2/\delta)}{2n}} \quad\text{with probability at least } 1 - \delta.$$

for stumps on $n$ points, $|\mathcal H| \le 2d(n + 1)$ effective rules, so the penalty grows only like $\sqrt{\log(dn)/n}$. for the lookup table on continuous inputs, $\mathcal H$ is effectively infinite and the bound says nothing. the same arithmetic explains warm-up 5: on pure noise, searching 200 features finds a stump that looks good by chance. chapter 13 turns this union bound into a theory; this chapter turns it into a habit: **anything you choose by looking at the data, you must evaluate on data you didn't look at.**

> **🔮 predict:** with 10 yes/no features there are only $2^{10} = 1{,}024$ possible inputs. with 100,000 training examples, how often is a test row one the lookup table has never seen?

> **▶ build: stage 5.** implement `LookupTableClassifier` (a dict from `tuple(row)` to the majority label of that row, with the overall majority class as the fallback) and `StumpClassifier` (exact ERM over all stumps, two classes, ties broken by smallest feature, then smallest threshold, then $s = +1$), both as `Classifier`s.

## worked example

**the stump.** six points on one feature, $x = (1, 2, 3, 4, 5, 6)$ with labels $y = (0, 0, 1, 0, 1, 1)$. the candidate thresholds are $-\infty, 1.5, 2.5, 3.5, 4.5, 5.5$. with $s = +1$ (class 1 on the right), count the positives on the left plus the negatives on the right:

| threshold | left | right | mistakes |
|---|---|---|---|
| $-\infty$ | – | 0 0 1 0 1 1 | 3 (the three 0s) |
| 1.5 | 0 | 0 1 0 1 1 | 2 |
| 2.5 | 0 0 | 1 0 1 1 | **1** |
| 3.5 | 0 0 1 | 0 1 1 | 2 |
| 4.5 | 0 0 1 0 | 1 1 | **1** |
| 5.5 | 0 0 1 0 1 | 1 | 2 |

with $s = -1$ the mistakes are $6 - $ those, at best 3. two candidates tie at one mistake; the rule picks the smaller threshold, so ERM returns feature 0, threshold 2.5, $s = +1$, training error $1/6$. it predicts class 0 for $x = 2.4$ and class 1 for $x = 2.6$. the same table falls out of the cumulative sums: the positives on the left are `cumsum` of the sorted labels, $(0, 0, 0, 1, 1, 2, 3)$ for $i = 0..6$ points on the left, and the negatives on the right are $3 - (\text{negatives on the left})$.

**the lookup table.** train on rows $0, 0, 0, 1, 2, 2, 3$ (one feature) with labels `b, b, a, c, a, c, c`. the table is $\{0 \mapsto b \text{ (2 votes to 1)},\ 1 \mapsto c,\ 2 \mapsto a \text{ (a tie between a and c goes to the smaller label)},\ 3 \mapsto c\}$, and the fallback is the overall majority, $c$ (3 of 7). so it predicts `b, c, a` on rows $0, 1, 2$ and `c` on an unseen row 7. notice that the fallback isn't the smallest label, and row 0's majority isn't the label it was seen with last: both are classic bugs.

the tests check both.

## on paper

1. **the best constant.** show that $c = \bar y$ minimizes $\sum_i (y_i - c)^2$, and conclude that the mean predictor has $R^2 = 0$ on its training data. which constant minimizes $\sum_i |y_i - c|$?

<details><summary>solution</summary>

the derivative is $-2\sum_i(y_i - c)$, zero at $c = \bar y$, and the function is a convex parabola, so that's the minimum. with $\hat y_i = \bar y$, $\text{SS}_\text{res} = \text{SS}_\text{tot}$ and $R^2 = 0$. for the absolute loss, the derivative (where it exists) is $\#\{y_i < c\} - \#\{y_i > c\}$, which changes sign at the median: any median minimizes it.

</details>

2. **stratified guessing.** with class frequencies $p_1, \dots, p_k$, show that guessing a class drawn from the prior has expected accuracy $\sum_j p_j^2$ on data from the same distribution, and that this never beats $\max_j p_j$.

<details><summary>solution</summary>

the guess and the truth are independent, so $P(\text{correct}) = \sum_j P(\text{truth} = j)P(\text{guess} = j) = \sum_j p_j^2$. and $\sum_j p_j^2 \le \sum_j p_j \max_l p_l = \max_l p_l$, with equality only when every class with $p_j > 0$ has the same frequency. at 70/30 that's $0.49 + 0.09 = 0.58$ against $0.70$.

</details>

3. **the finite-class bound.** using Hoeffding's inequality, $P(|\hat R(h) - R(h)| > \varepsilon) \le 2e^{-2n\varepsilon^2}$ for a fixed $h$, and a union bound, show that with probability at least $1 - \delta$ every $h$ in a finite class satisfies $|\hat R(h) - R(h)| \le \sqrt{(\log|\mathcal H| + \log(2/\delta))/(2n)}$.

<details><summary>solution</summary>

$P(\exists h: |\hat R(h) - R(h)| > \varepsilon) \le \sum_{h \in \mathcal H} 2e^{-2n\varepsilon^2} = 2|\mathcal H|e^{-2n\varepsilon^2}$. setting that equal to $\delta$ and solving gives $\varepsilon = \sqrt{\log(2|\mathcal H|/\delta)/(2n)}$. the union bound is what you pay for *choosing*: it holds for whichever $h$ ERM picks, because it holds for all of them at once.

</details>

4. **how many stumps?** on $n$ points with distinct values in each of $d$ features, how many different labelings of the training set can stumps produce? why is this, not the number of real thresholds, the size that matters?

<details><summary>solution</summary>

per feature: $n + 1$ split positions times 2 signs, but "everything is class 1" and "everything is class 0" are shared by all features and both signs, so at most $2d(n - 1) + 2$ distinct labelings. on the training set two hypotheses that label it identically are indistinguishable, so ERM effectively chooses among these finitely many. counting labelings instead of functions is the idea behind the growth function and the VC dimension (chapter 13).

</details>

## experiment

`lab.py` runs your code on three questions:

1. a memorizer, a stump and the dummy baselines on the same noisy data: training versus test accuracy.
2. when does memorizing generalize? the lookup table as the number of possible inputs shrinks.
3. ERM on pure noise: how good does the best stump *look* as the number of features grows, and how good is it really?

## check your understanding

1. why does `__init__` store its arguments unchanged instead of validating them or precomputing things?

<details><summary>answer</summary>

because `get_params` reads them back by name to rebuild the object (`clone`), and `set_params` changes them after construction. anything `__init__` computed from an argument would go stale after `set_params`, and anything stored under a different name is invisible to `get_params`. validation belongs in `fit`, where a bad value is reported when it's actually used (including after a search sets it).

</details>

2. a colleague implements `clone` as `copy.deepcopy(est)`. it passes their tests. what goes wrong in cross-validation?

<details><summary>answer</summary>

if the estimator was already fitted, every "fresh" copy starts with the fitted attributes of the original, trained on data that may include the current test fold. many models refit from scratch and overwrite them, so the bug hides, but any model that warm-starts from its current state, or a `predict` before `fit`, silently uses information it shouldn't have. `clone` builds a new object from hyperparameters only.

</details>

3. why must `classes_` be sorted, and why must `predict_proba`'s columns follow it?

<details><summary>answer</summary>

so that code that knows nothing about the model can interpret the probabilities: column 1 of a binary classifier is the positive class for ROC curves and log loss, averaging probabilities across models (ensembles, folds) lines up columns, and `predict` is just the argmax. sorting gives every model the same order for the same labels, whatever order they appeared in the data.

</details>

4. a model has $R^2 = -0.3$ on the test set. what does that mean in plain words?

<details><summary>answer</summary>

its squared error on the test set is 30% *larger* than that of predicting the mean of the test targets. it's worse than the constant baseline. this happens to overfit models and to models evaluated on a shifted distribution.

</details>

5. the lookup table gets near the best possible accuracy (0.90) on a problem with 10 binary features and 100,000 examples. what changed compared to continuous features?

<details><summary>answer</summary>

the number of possible inputs (1,024) is small compared with $n$, so nearly every test row has been seen, many times, and the table stores the *majority* label for each one: it's estimating $P(y \mid x)$ for every $x$ by counting, which is the Bayes-optimal rule in the limit. memorization fails when the input space is too large to cover, not because memorizing is bad in itself.

</details>

6. why is ERM over a larger hypothesis class not always better, even though its training error can only go down?

<details><summary>answer</summary>

the training error of the chosen hypothesis is a biased estimate of its risk, and the bias grows with the size (more precisely, the richness) of the class: with more candidates, the best-looking one is more likely to look good by chance. the finite-class bound makes this concrete: the gap term grows like $\sqrt{\log|\mathcal H|/n}$. a bigger class lowers the best achievable risk (less **approximation error**) but raises the risk of picking the wrong one (more **estimation error**). topic 04 is the regression version of this trade-off.

</details>

## common bugs

- **work in `__init__`.** computing or validating in `__init__` (`self.alpha = float(alpha)`, `self.k = min(k, 10)`) breaks `get_params`/`set_params` round trips. symptom: a grid search reports the same score for every setting, or `clone` produces a different model.
- **fitted state without the underscore.** storing `self.coef` instead of `self.coef_`, or a hyperparameter that `fit` overwrites. symptom: `clone` copies learned weights, or refitting gives a different answer than a fresh fit.
- **returning column indices from `predict`.** `np.argmax(proba, axis=1)` gives 0, 1, 2, not `"bird", "cat", "dog"`. symptom: accuracy is 0 whenever the labels aren't $0..k-1$.
- **a `y` of shape `(n, 1)`.** `y - y_pred` broadcasts to `(n, n)`. symptom: a loss or $R^2$ that is plausible-looking but wrong. `check_X_y` flattens it.
- **a dummy with a fixed random stream.** `np.random.default_rng(0)` inside `predict` repeats the same "random" guesses on every call. draw from `rng_ = as_rng(self.rng)`, made in `fit`.

## going deeper

- Shalev-Shwartz & Ben-David (2014), *Understanding Machine Learning: From Theory to Algorithms*, Cambridge University Press, chapters 2–4 (ERM, finite hypothesis classes, the union-bound argument). free online. [cs.huji.ac.il/~shais/UnderstandingMachineLearning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, chapter 2 (overview of supervised learning: least squares versus nearest neighbours, the curse of dimensionality). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Buitinck et al. (2013), "API design for machine learning software: experiences from the scikit-learn project". the estimator protocol you just built, and why it looks the way it does. [arXiv:1309.0238](https://arxiv.org/abs/1309.0238)
- Holte (1993), "Very simple classification rules perform well on most commonly used datasets", *Machine Learning* 11: 63–90. one-feature rules, close cousins of the stump, came within a few points of far more complex models on most benchmarks of the time: a lesson about baselines. [doi:10.1023/A:1022631118932](https://doi.org/10.1023/A:1022631118932)
- Vapnik (1999), "An overview of statistical learning theory", *IEEE Transactions on Neural Networks* 10(5): 988–999. ERM and the generalization gap. [doi:10.1109/72.788640](https://doi.org/10.1109/72.788640)

## connections

- ← builds on: classes and NumPy (ch 00), `as_rng` (ch 00 · topic 05), expectations (ch 04), Hoeffding's inequality and the sample mean (ch 04–05).
- → used in: every topic of this chapter (splitters and `cross_validate` clone estimators, metrics read `classes_`, pipelines are estimators made of transformers), every model in chapters 08–13 (they subclass `Regressor`, `Classifier` or `Transformer`), the dummy baselines in every evaluation from here on, and the generalization bounds of chapter 13.
