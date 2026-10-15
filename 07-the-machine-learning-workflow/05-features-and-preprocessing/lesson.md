# features & preprocessing

## why this matters

a model only sees numbers in a matrix, and how you put them there changes what it learns. record income in dollars instead of thousands, and a nearest-neighbour classifier that worked suddenly guesses at chance, because "nearest" is now decided by income alone. number your cities 0 to 5, and the model believes city 2 is between cities 1 and 3. leave a `NaN` in a column, and the fit fails, or worse, silently propagates it. this is unglamorous work, and in practice it decides as much of a model's performance as the choice of algorithm.

it's also where most leakage happens. a scaler, an imputer, an encoder: each one *learns* something from data (a mean, a set of categories), so each one belongs inside cross-validation, refitted on every training fold. the tool that makes that automatic is the **pipeline**: a chain of transformers ending in a model, packaged as one estimator. cross-validate the pipeline, and every step is refitted per fold without you thinking about it.

you'll build `mlfs.preprocessing`: two scalers, a one-hot encoder, an imputer, polynomial features, and `Pipeline` with `make_pipeline`, including the `"step__param"` naming that lets topic 06 tune any step.

## warm-up: predict first

1. two features: age in years (spread about 12) and income in dollars (spread about 20,000). which one dominates a Euclidean distance between two people?
2. you standardize your features. does a decision tree's accuracy change? does $k$-NN's?
3. you one-hot encode a column with 6 categories and fit a linear model with an intercept. what goes wrong if you keep all 6 columns?
4. 30% of a column is missing and you fill it with the column mean. what happens to that column's variance, and to its correlation with the target?
5. how many columns does `PolynomialFeatures(degree=3)` produce from 10 features?

## prerequisites check

- [ ] you've built `Transformer`, `fit_transform` and `check_X_y` (→ topic 01)
- [ ] you can explain why preprocessing fitted on all the data leaks (→ topic 02, idea 5)
- [ ] you know rank, linear dependence and why a rank-deficient least-squares problem has no unique solution (→ ch 02 · topics 01 and 04)
- [ ] you know how the condition number slows gradient descent (→ ch 06 · topic 02)
- [ ] you know mean, median, variance and correlation (→ ch 05 · topic 01)

## core ideas

### idea 1: scaling

many models compare features *with each other*: $k$-NN and kernels through distances $\|x - x'\|^2 = \sum_j (x_j - x'_j)^2$, regularized linear models through a penalty $\lambda\sum_j w_j^2$ that treats every coefficient the same, gradient descent through a loss surface whose curvature in each direction scales with that feature's spread (chapter 06: a badly conditioned problem). none of them knows that a difference of 10 in age and 10 in dollars aren't comparable. so you choose a scale for them.

**standardization** subtracts each feature's mean and divides by its standard deviation, both learned from the training data:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}, \qquad \mu_j = \frac1n\sum_i x_{ij}, \quad \sigma_j = \sqrt{\frac1n\sum_i (x_{ij} - \mu_j)^2}.$$

every feature then has mean 0 and standard deviation 1 on the training set. a constant feature has $\sigma_j = 0$; divide by 1 instead, so it becomes all zeros rather than `NaN`.

**min–max scaling** maps each feature's training minimum and maximum to a range, usually $[0, 1]$:

$$x'_j = \text{lo} + \frac{x_j - \min_j}{\max_j - \min_j}(\text{hi} - \text{lo}).$$

it keeps zeros in sparse data and bounded inputs bounded, but its scale depends on just two values per feature, so one extreme outlier squeezes everything else into a sliver (the lab measures how much). new data can land outside $[\text{lo}, \text{hi}]$; that's fine, and clipping it would hide information.

both are **affine and per-feature**, so both are undone exactly by `inverse_transform`, and neither changes the *ordering* of values within a feature. that's why a decision tree, which only asks "is $x_j$ above a threshold?", doesn't care (warm-up 2), while $k$-NN can change completely.

> **🔮 predict:** income is irrelevant to the label. you record it in *millions* of dollars (spread 0.02) and run $k$-NN on the raw features, and then on standardized features. which does better?

> **▶ build: stage 1.** implement `StandardScaler(*, with_mean=True, with_std=True)` (`mean_`, `scale_`; zero spread → scale 1) and `MinMaxScaler(feature_range=(0.0, 1.0))` (`data_min_`, `data_max_`; zero range → 1), each with `fit`, `transform` and `inverse_transform`, all statistics from `fit`.

### idea 2: encoding categories

a category (a city, a colour, a product type) is a label, not a quantity. coding cities as $0, 1, \dots, 5$ invents an order and distances that don't exist: to a distance-based or linear model, city 5 is "five times" city 1. **one-hot encoding** gives each category its own indicator column instead:

$$\text{colour} \in \{\text{blue}, \text{green}, \text{red}\} \quad\longmapsto\quad \text{red} = (0, 0, 1),\ \text{green} = (0, 1, 0),\ \text{blue} = (1, 0, 0).$$

any two different categories are now equally far apart ($\sqrt2$), and a linear model gets one free coefficient per category. the categories are learned in `fit`, sorted, per input column (`categories_`), which raises the question of **unseen categories** at prediction time. either fail loudly (`handle_unknown="error"`, the default: a new category often means a data problem) or encode it as an all-zero block (`"ignore"`: "none of the categories I know").

**dropping a column.** the $k$ indicators of one feature always sum to 1, the same as the intercept column. with an intercept, the design matrix is rank-deficient (warm-up 3, on paper 3), the least-squares coefficients aren't unique, and unregularized fits become unstable. `drop="first"` removes each block's first column, so the first category is represented by all zeros and its effect is absorbed into the intercept. regularized models (and trees, and $k$-NN) don't need it.

> **🔮 predict:** customers in cities 0, 2 and 4 behave alike, and so do those in 1, 3 and 5. a nearest-neighbour model sees the city as one integer column, or as six one-hot columns. which does better, and why?

> **▶ build: stage 2.** implement `OneHotEncoder(*, handle_unknown="error", drop=None)`: `categories_` from `fit` (sorted, per column), a dense float `transform` (blocks in column order, a 1 in the row's category, `drop="first"` and `"ignore"` as above), and `inverse_transform` back to an object array (an all-zero block is the first category with `drop="first"`, and `None` without it).

### idea 3: missing values

real tables have holes. most models can't take `NaN`, so you must fill them (**impute**) or drop rows (which throws away data and can bias the sample). the simplest imputers replace each missing entry with one statistic of its column, learned from the training data: the **mean**, the **median** (robust to outliers), the **most frequent** value (for discrete columns), or a **constant**.

simple imputation has costs you should know about (warm-up 4). filling with the mean adds values with zero deviation, so it shrinks the column's variance and weakens its correlation with everything else (on paper 4). and *why* the value is missing matters. if values are **missing completely at random**, filling them mostly costs precision. if missingness depends on the value itself (high earners skip the income question), the mean is systematically wrong, and the *fact* that a value is missing can be predictive. a common remedy is to add an indicator column "was missing" next to the imputed one. either way, the imputer is a learned step like any other: fit it on the training fold only.

> **▶ build: stage 3.** implement `SimpleImputer(strategy="mean", *, fill_value=None)` for numeric data: `statistics_` per column from the non-missing values (`mean`, `median`, `most_frequent` with ties to the smallest, `constant`), and a `transform` that fills a *copy*.

### idea 4: polynomial features

a linear model can only draw flat boundaries in the space it's given. give it products of features, and it can fit curves and interactions while staying linear in its parameters: least squares on

$$(1,\ x_0,\ x_1,\ x_0^2,\ x_0x_1,\ x_1^2)$$

fits any quadratic surface in $(x_0, x_1)$. that's topic 04's `PolynomialRegressor` generalized to many inputs, and it's the simplest example of a **feature map** $\phi(x)$, an idea chapter 09 takes to infinite dimensions with kernels.

the column order is a convention worth matching exactly (scikit-learn's): by total degree, and within a degree in the order `itertools.combinations_with_replacement(range(d), k)` lists the feature indices. `interaction_only=True` uses `combinations` instead, keeping products of *distinct* features ($x_0x_1$ but not $x_0^2$). store the exponents as `powers_`, one row per output column, and `transform` is a product of powers.

the catch is growth. with $d$ features and degree $k$ there are $\binom{d + k}{k}$ monomials (on paper 2): 286 for $d = 10$, $k = 3$ (warm-up 5), over 23,000 for $d = 50$. high powers of unscaled features also have wildly different scales, so polynomial features are usually followed (or preceded) by a scaler, and paired with regularization (chapter 08).

> **▶ build: stage 4.** implement `PolynomialFeatures(degree=2, *, include_bias=True, interaction_only=False)` with `powers_` (an int array, one row of exponents per output column, in scikit-learn's order) and a vectorized `transform`.

### idea 5: pipelines

each preprocessing step has `fit` (learn statistics) and `transform` (apply them), and the leakage rule from topic 02 says: *learn on the training fold, apply to the test fold*. doing that by hand for three steps inside a cross-validation loop is tedious and easy to get wrong. a **pipeline** does it structurally:

- `fit(X, y)`: for every step but the last, `X = step.fit_transform(X, y)`; then fit the last step on the transformed `X`.
- `predict(X)` (and `predict_proba`, `score`): pass `X` through every earlier step's `transform` (never `fit`), then call the last step.

because `Pipeline` is an `Estimator`, `cross_validate` clones and refits the *whole chain* per fold: the scaler's mean, the imputer's statistics and the encoder's categories are all learned from training rows only. the lab of topic 02 did this by hand with a feature selector inside `fit`; a pipeline makes it the default.

`make_pipeline(StandardScaler(), model)` names the steps for you, by lowercased class name (`"standardscaler"`), adding `-1`, `-2` to repeated names.

> **▶ build: stage 5.** implement `Pipeline.fit` (unique names, transformers before the last step), `predict`, `predict_proba`, `transform` (every step, for a pipeline of transformers) and `score`, plus `make_pipeline`. the helpers `named_steps`, `classes_` and `transform_before_last` are given.

### idea 6: parameters of a pipeline

to tune a pipeline, a search has to reach the hyperparameters of its steps. the convention is the double underscore from topic 01: `"knn__k"` is the parameter `k` of the step named `knn`. so a pipeline's `get_params()` lists `"steps"` (its only constructor argument) *and* every step's parameters under `"name__param"`, and `set_params(knn__k=7)` forwards to `named_steps["knn"].set_params(k=7)`. nesting works for free: in a pipeline inside a pipeline, `"pre__scale__with_mean"` passes through two levels. `set_params(knn=OtherModel())` replaces a whole step.

`clone` copes with all this because it passes only the keys *without* `"__"` back to the constructor, and clones the estimators inside the `steps` list recursively (topic 01, stage 2). so `clone(pipeline)` is a fresh, unfitted pipeline with every step's settings, which is exactly what cross-validation and search need.

> **▶ build: stage 6.** implement `Pipeline.get_params` (`"steps"` plus every step's `"name__param"`) and `Pipeline.set_params` (`"steps"`, a step name, or `"name__param"`; unknown names raise `ValueError`; replace a step by assigning a new list).

## worked example

four rows, two numeric features and one colour:

| | $x_0$ | $x_1$ | colour |
|---|---|---|---|
| row 1 | 1 | 100 | red |
| row 2 | 2 | 300 | green |
| row 3 | 3 | 200 | blue |
| row 4 | 6 | 400 | green |

**standardization.** $\mu = (3, 250)$. the deviations of $x_0$ are $(-2, -1, 0, 3)$, so $\sigma_0^2 = (4 + 1 + 0 + 9)/4 = 3.5$; for $x_1$, $(-150, 50, -50, 150)$ gives $\sigma_1^2 = 12{,}500$. row 4 becomes $(3/\sqrt{3.5}, 150/\sqrt{12500}) \approx (1.60, 1.34)$.

**min–max.** $x_0$: $(x - 1)/5$ gives $(0, 0.2, 0.4, 1)$. $x_1$: $(x - 100)/300$ gives $(0, 2/3, 1/3, 1)$.

**one-hot.** categories, sorted: blue, green, red. the colours become $(0,0,1), (0,1,0), (1,0,0), (0,1,0)$; with `drop="first"` the blue column goes, and blue is $(0, 0)$.

**polynomial features** of degree 2 for the point $(x_0, x_1) = (2, 3)$: $(1, x_0, x_1, x_0^2, x_0x_1, x_1^2) = (1, 2, 3, 4, 6, 9)$.

the tests check all of these.

## on paper

1. **scaling is a choice of metric.** show that Euclidean distance between standardized vectors equals $\sqrt{\sum_j (x_j - x'_j)^2/\sigma_j^2}$, a weighted distance on the original features. what weights does min–max scaling use?

<details><summary>solution</summary>

$z_j - z'_j = (x_j - \mu_j)/\sigma_j - (x'_j - \mu_j)/\sigma_j = (x_j - x'_j)/\sigma_j$; square and sum. the means cancel, so centering never affects distances; only the scale does. min–max uses weights $1/(\max_j - \min_j)^2$ (times $(\text{hi} - \text{lo})^2$). choosing a scaler *is* choosing which differences count as large.

</details>

2. **counting monomials.** show that the number of monomials of total degree at most $k$ in $d$ variables (including the constant) is $\binom{d + k}{k}$. evaluate it for $d = 10$, $k = 3$.

<details><summary>solution</summary>

a monomial $x_1^{a_1}\cdots x_d^{a_d}$ with $\sum a_j \le k$ corresponds to a solution of $a_0 + a_1 + \dots + a_d = k$ in nonnegative integers (with a slack $a_0$). by stars and bars there are $\binom{k + d}{d} = \binom{d + k}{k}$ of them. for $d = 10$, $k = 3$: $\binom{13}{3} = 286$.

</details>

3. **the dummy-variable trap.** a design matrix has an intercept column of ones and the full one-hot encoding of a 3-category feature. show that it is rank-deficient, and that with `drop="first"` it has full rank (assuming every category occurs).

<details><summary>solution</summary>

the three indicator columns sum to the all-ones column, so $\mathbf 1 - e_1 - e_2 - e_3 = 0$ is a nontrivial linear combination: rank at most (number of columns) $- 1$. without the first indicator, suppose $a\mathbf 1 + b e_2 + c e_3 = 0$. rows in category 1 give $a = 0$, rows in category 2 give $b = 0$, rows in category 3 give $c = 0$: independent. the intercept now means "category 1", and $b, c$ are differences from it.

</details>

4. **mean imputation shrinks variance.** a column has $n$ entries, $m$ of them missing, and the $n - m$ observed values have (population) variance $s^2$. after filling the $m$ missing entries with the observed mean, what is the column's variance?

<details><summary>solution</summary>

the mean is unchanged, the filled entries contribute zero squared deviation, and the observed ones contribute $(n - m)s^2$ in total. so the variance is $\frac{n - m}{n}s^2$: with 30% missing, 70% of the true spread. covariances with other columns shrink by the same factor, while the other column's variance doesn't, so correlations are biased towards 0.

</details>

## experiment

`lab.py` uses your transformers and pipelines with a $k$-nearest-neighbour classifier:

1. kNN on raw, standardized and min–max scaled features, with income recorded in different units.
2. a few data-entry errors (outliers) in one feature: StandardScaler versus MinMaxScaler.
3. a categorical feature as an integer code versus one-hot encoded.

## check your understanding

1. you scaled your features with statistics computed on the full dataset, then cross-validated. topic 02's lab said that's usually harmless. when isn't it?

<details><summary>answer</summary>

when the statistics carry real information about the test rows: small datasets, heavy-tailed features (one test outlier moves the max or the std), data with distribution shift between folds (time, groups), and especially steps that use labels (target encoding, feature selection). the pipeline makes the question moot: fit inside the fold and you never have to argue about it.

</details>

2. why does `StandardScaler` divide by 1 for a constant column instead of raising an error?

<details><summary>answer</summary>

a constant column is legal data (and can easily happen inside one CV fold), carries no information, and should become harmless zeros rather than `NaN`s that poison every later step. raising would make pipelines fail on otherwise fine data.

</details>

3. `OneHotEncoder(handle_unknown="ignore", drop="first")`: what's ambiguous about an all-zero block?

<details><summary>answer</summary>

with `drop="first"`, an all-zero block means "the first category". with `"ignore"`, it also means "a category never seen in fit". the two become indistinguishable, so unknowns silently look like the first category. scikit-learn warns about this combination.

</details>

4. why must a pipeline call `transform`, never `fit_transform`, on the earlier steps in `predict`?

<details><summary>answer</summary>

refitting at prediction time would compute new statistics from the data being predicted: a test-time leak, and also inconsistent behaviour (the model was trained on features scaled with the training statistics). a single test row would even be scaled by its own mean, which makes it zero.

</details>

5. a feature has a few huge outliers. would you rather standardize it, min–max scale it, or do something else?

<details><summary>answer</summary>

min–max is the most fragile (the outliers set the range), and standardization is affected too (the outliers inflate the std). better: find out whether they're errors (fix or drop them), or transform the feature (a log for positive, skewed data), or use a robust scaler based on the median and interquartile range, or clip at sensible percentiles learned on the training data.

</details>

6. what does `clone(make_pipeline(StandardScaler(), model))` return, and why don't the `"name__param"` keys from `get_params` break it?

<details><summary>answer</summary>

a new `Pipeline` with new, unfitted copies of both steps and the same hyperparameters. `clone` passes only the keys without `"__"` to the constructor (here just `steps`), and clones the estimators inside the `steps` list recursively; the `"name__param"` keys are a view into those steps, not constructor arguments.

</details>

## common bugs

- **fitting a transformer on the test set.** calling `fit_transform` on test data, or `scaler.fit(X_test)`. symptom: a test set that's always perfectly centred; single-row predictions that ignore the input.
- **sample instead of population std.** `ddof=1` in the scaler. symptom: standardized training columns with std $\sqrt{(n-1)/n}$ instead of 1, and a small mismatch with scikit-learn.
- **categories in order of appearance.** `pd.unique`-style ordering instead of sorting. symptom: columns permuted between two fits on the same data in a different row order, so models aren't comparable.
- **imputing in place.** writing the statistics into the caller's array. symptom: the "original" data has no missing values anymore, and re-running a cell gives different results.
- **a pipeline that refits in `predict`.** see question 4.
- **mutating the steps list in `set_params`.** replacing a step with `self.steps[i] = ...` changes the list object the caller (or a clone's source) also holds. assign a new list.

## going deeper

- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, sections 3.4 (why ridge and lasso need standardized inputs) and 9.6 (missing data). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Kuhn & Johnson (2019), *Feature Engineering and Selection: A Practical Approach for Predictive Models*, CRC Press: encodings, interactions, missing data. free online. [bookdown.org/max/FES](https://bookdown.org/max/FES/)
- Little & Rubin (2019), *Statistical Analysis with Missing Data*, 3rd ed., Wiley: MCAR/MAR/MNAR and what each allows. [doi:10.1002/9781119482260](https://doi.org/10.1002/9781119482260)
- Buitinck et al. (2013), "API design for machine learning software: experiences from the scikit-learn project": the transformer/pipeline design you just built. [arXiv:1309.0238](https://arxiv.org/abs/1309.0238)
- Micci-Barreca (2001), "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems", *SIGKDD Explorations* 3(1): target encoding, a powerful encoder that leaks unless fitted inside the folds. [doi:10.1145/507533.507538](https://doi.org/10.1145/507533.507538)

## connections

- ← builds on: the `Transformer` base and `clone` (topic 01), the leakage rule (topic 02), rank and least squares (ch 02), conditioning and gradient descent (ch 06).
- → used in: tuning pipelines with `"step__param"` (topic 06), the capstone, every linear model in chapter 08 (standardize before ridge and lasso, polynomial bases), $k$-NN and kernels (ch 09), bag-of-words and TF-IDF features (ch 08 capstone), PCA as a transformer (ch 11), and normalization layers in neural networks (ch 15), which are learned scalers inside the model.
