# splitting & cross-validation

## why this matters

in 2002, Ambroise and McLachlan re-examined published claims that a handful of genes could predict cancer from microarray data with near-zero error. the studies had "cross-validated" their classifiers. but they'd chosen the genes first, using all the samples, and only then cross-validated the classifier on those genes. redone properly, with the gene selection repeated inside each fold, the error rates went from near zero to something much less impressive. and on data with no signal at all, the flawed procedure can *still* report excellent accuracy: you'll reproduce that in this topic's lab.

that mistake has a name, **data leakage**: information from the data you evaluate on seeps into the model before it's evaluated. it's the single most common way machine-learning results turn out to be wrong, and it's rarely as obvious as training on the test set. it hides in preprocessing that was fitted on all the data, in patients whose recordings ended up on both sides of a split, in tomorrow's prices quietly used to predict today's.

topic 01 ended with a rule: anything you choose by looking at the data, you must evaluate on data you didn't look at. this topic builds the machinery that enforces it, `mlfs.model_selection`: a holdout split, $k$-fold cross-validation and its stratified, grouped and time-ordered variants, and `cross_validate`, which fits a fresh clone of the model in every fold. the lab then shows three leaks that turn pure noise into 70–100% accuracy.

## warm-up: predict first

1. you have 1,000 examples. why not train on all of them and report the training score?
2. 5-fold cross-validation gives scores 0.81, 0.79, 0.84, 0.80, 0.83. is the model's "true" accuracy within $0.814 \pm 0.009$ (the mean plus or minus the standard error of the five scores)?
3. you standardize every feature with the mean and standard deviation of the whole dataset, then cross-validate. is that a leak? how big?
4. a dataset has 8 recordings from each of 30 patients. you shuffle the recordings into 5 folds. what could go wrong?
5. why can't you shuffle a time series before splitting it?

## prerequisites check

- [ ] you've built `clone` and know why every fit needs a fresh, unfitted copy (→ topic 01, idea 2)
- [ ] you know the `fit` / `score` protocol and what `classes_` is (→ topic 01, ideas 2–3)
- [ ] you can draw permutations with `as_rng(rng).permutation(n)` (→ ch 00 · topic 05)
- [ ] you know that the variance of a mean of $n$ independent variables is $\sigma^2/n$, and what correlation does to it (→ ch 04 · topic 03)
- [ ] you know what a confidence interval is (→ ch 05 · topic 03)

## core ideas

### idea 1: the holdout split

the score that matters is the risk $R(h)$ on new data, and topic 01 showed that the training score estimates it badly: the model was chosen to make the training score good. the simplest fix is to hide some data. split the examples at random into a **training set**, which the model learns from, and a **test set**, which it never sees until the end. the test score is then an honest estimate of the risk, because the test examples are new draws from the same distribution as far as the model is concerned.

two refinements matter in practice.

**stratification.** with rare classes, a random split can leave the test set with too few, or none, of them. a **stratified** split samples each class separately so that the test set has the same class proportions as the whole. with 70/25/5 rows of three classes and a 20% test set, the ideal counts are $14, 5, 1$. when the ideal counts aren't integers, round them so the total is still exact: give each class the floor of its ideal count, then hand the leftover rows to the classes with the largest remainders.

**three sets, not two.** the moment you use the test score to *decide* something (which model, which hyperparameter, when to stop training), it stops being honest, for exactly the reason the stump's training score wasn't. so the standard workflow uses a **validation set** (or cross-validation, next idea) for every decision, and touches the test set **once**, at the very end. in this chapter's capstone the test set is locked away in the first cell and opened in the last.

> **🔮 predict:** you split 100 examples 75/25 and get a test accuracy of 0.80. roughly how wide is a 95% confidence interval for the true accuracy? (a test accuracy is a mean of 25 coin flips.)

> **▶ build: stage 1.** implement `train_test_split(*arrays, test_size=0.25, shuffle=True, stratify=None, rng=None)`. a float `test_size` is a fraction (round the count *up*), an int is a count. return `[a_train, a_test, b_train, b_test, ...]`, split by the same rows. `shuffle=False` takes the last rows as the test set; `stratify` uses the largest-remainder rule.

### idea 2: k-fold cross-validation

a single holdout split wastes data twice: the model trains on only 75% of it, and the estimate rests on only 25%, so it's noisy (the 🔮 above: about $\pm 0.16$). **$k$-fold cross-validation** reuses every example. cut the data into $k$ folds of (nearly) equal size; for each fold, train on the other $k - 1$ and test on it; report the $k$ scores and their mean:

$$\widehat{\text{CV}} = \frac1k \sum_{f=1}^{k} \text{score}\big(h_{-f},\ \text{fold } f\big),$$

where $h_{-f}$ was fitted without fold $f$. every example is tested exactly once, by a model that never saw it. for $n$ rows and $k$ folds, the first $n \bmod k$ folds get $\lfloor n/k \rfloor + 1$ rows and the rest $\lfloor n/k \rfloor$: with $n = 7$, $k = 3$, the folds are rows $\{0,1,2\}, \{3,4\}, \{5,6\}$.

without shuffling, the folds are contiguous blocks, which is right for data whose order means nothing *and* has been shuffled already, and wrong if the file is sorted by class or by date. with `shuffle=True` the rows are permuted first.

what does $\widehat{\text{CV}}$ estimate? the expected score of a model trained on $\frac{k-1}{k}n$ examples, so it's slightly pessimistic about the model you'll train on all $n$ (a model usually improves with more data). larger $k$ reduces that bias, at the price of $k$ fits; $k = 5$ or $10$ is the usual compromise, and $k = n$ is **leave-one-out**. the $k$ scores also tell you about variability, but be careful with warm-up 2: the folds share most of their training data, so the scores are *correlated*, and the naive standard error $s/\sqrt k$ is too small. there's no unbiased estimator of the variance of $k$-fold CV (Bengio & Grandvalet, 2004). treat the spread of the fold scores as a lower bound on your uncertainty, and use the tools of the capstone (bootstrapping over test examples, repeated CV with different seeds) when it matters.

> **▶ build: stage 2.** implement `KFold(n_splits=5, *, shuffle=False, rng=None)` with `split(X, y=None, groups=None)` yielding `(train_idx, test_idx)` pairs (sorted int arrays), and `get_n_splits()`. `2 <= n_splits <= n`, else `ValueError`.

### idea 3: what is an independent example? strata and groups

cross-validation is honest only if the test fold is *new* in the same sense that deployment data will be new. two common structures break the naive version.

**imbalanced classes.** with a 5% minority class and 5 folds of 20 rows, a random fold can easily get zero minority examples, and its precision or AUC is undefined. **stratified $k$-fold** deals the rows out like cards: sort the rows by class (keeping their order, or a shuffled order, within each class), then send row number $i$ of that ordering to fold $i \bmod k$. every class lands in every fold as evenly as possible (its counts per fold differ by at most one), and so do the fold sizes.

**groups.** in topic 01's language, the examples should be independent draws. 8 recordings of the same patient aren't: they share everything that makes that patient unique. if recordings from one patient land in both train and test, a model can recognize the *patient* instead of learning the *disease*, and the test score measures something nobody asked about (warm-up 4). the unit of splitting has to be the unit of independence: patients, users, sessions, documents, cities. **group $k$-fold** assigns whole groups to folds. to keep the folds about the same size it's greedy: take the groups from largest to smallest, and put each into the fold with the fewest rows so far.

the question to ask is always: *at deployment, what will be new?* if the model will see new patients, the test fold must contain new patients.

> **🔮 predict:** with 30 patients, labels assigned per patient at random, and a 1-nearest-neighbour classifier, what accuracy will shuffled `KFold` report?

> **▶ build: stage 3.** implement `StratifiedKFold` (requires `y`; deal by class) and `GroupKFold` (requires `groups`; greedy largest-first assignment, visiting groups in the order `np.argsort(sizes, kind="stable")[::-1]`, as scikit-learn does). both raise `ValueError` when their requirement is missing or impossible.

### idea 4: time flows one way

time series break independence in a sharper way. consecutive observations are similar (prices, temperatures, sensor readings drift slowly), and at deployment you predict the *future* from the *past*. a shuffled split lets the model train on Tuesday and Thursday and be tested on Wednesday: it can interpolate between neighbours in time, which it will never be able to do in production (warm-up 5).

the honest scheme always tests on data that comes after everything it trained on. **expanding-window** splits cut the last part of the series into $k$ consecutive test blocks of size $\lfloor n/(k+1) \rfloor$, and train on everything before each block:

$$\text{train} = \{0, \dots, s - 1\}, \qquad \text{test} = \{s, \dots, s + m - 1\}, \qquad m = \lfloor n/(k + 1)\rfloor.$$

the first split has the least training data and the last the most, so the scores also show how performance changes as history accumulates. two refinements you'll meet later: a **gap** between train and test, when the label at time $t$ depends on a window that reaches past $t$, and a **sliding** window (fixed training length), when old data stops being relevant.

> **▶ build: stage 4.** implement `TimeSeriesSplit(n_splits=5)`: test blocks of size `n // (n_splits + 1)` covering the last `n_splits * test_size` rows, each with all earlier rows as its training set. `n_splits + 1 > n` raises `ValueError`.

### idea 5: cross_validate, and the leakage rule

a splitter only produces indices. `cross_validate` does the rest: for each `(train, test)` pair, make a fresh model with `clone`, fit it on the training rows, and score it on the test rows, with the model's own `score` or a **scorer**, a function `scorer(fitted_model, X_test, y_test)` (topic 03 builds a library of them). it returns the test scores and, optionally, the training scores (a large gap between the two means overfitting) and the fitted models.

the `clone` is not a detail. if `cross_validate` fitted the model you passed in, fold 2 would start from fold 1's fitted state, and after the loop your model would be fitted on the last fold's training set, which is surprising at best. clone-per-fold also makes the procedure correct for models that keep state between fits.

one convention to watch: an int `cv=5` means `KFold(5)`, **unshuffled**. on data sorted by label (a file with all the negatives first), the contiguous folds then hold one class each: a model trained without any positives can't score them, and a probability scorer like `roc_auc` fails with "a label the estimator never saw". pass `KFold(5, shuffle=True, rng=...)` or `StratifiedKFold` whenever the row order might mean something.

and now the **leakage rule**, the most important sentence in this chapter:

> everything that is learned from data, including preprocessing, feature selection, imputation and hyperparameter tuning, must be learned inside the training fold, and only applied to the test fold.

the cancer studies broke it by selecting genes on all the samples. the effect depends on *what* leaks. a preprocessing step that looks only at the inputs (centring, scaling) and is fitted on all rows leaks a little information about the test inputs' distribution; with plenty of data that's usually tiny (warm-up 3). a step that looks at the **labels**, like choosing the features most correlated with $y$, leaks the answer itself, and can turn pure noise into near-perfect accuracy. the cure is structural: put every learned step inside the object that `cross_validate` clones and fits per fold. in topic 05 that object becomes a `Pipeline`.

> **▶ build: stage 5.** implement `cross_validate(estimator, X, y=None, *, cv=5, scoring=None, groups=None, return_train_score=False, return_estimators=False)` and `cross_val_score`. `cv` may be an int (`KFold(cv)`), a splitter, or a list of `(train, test)` pairs. fit `clone(estimator)` per fold, never the estimator itself, and pass `groups` on to the splitter.

## worked example

cross-validate the mean predictor (topic 01's `DummyRegressor`) on six points, $x = 0, 1, \dots, 5$ with targets $y = (1, 3, 2, 6, 5, 7)$, with 3 unshuffled folds, scoring by mean squared error:

| fold | test rows | training targets | training mean | test targets | test MSE |
|---|---|---|---|---|---|
| 1 | 0, 1 | 2, 6, 5, 7 | 5 | 1, 3 | $\frac{16 + 4}{2} = 10$ |
| 2 | 2, 3 | 1, 3, 5, 7 | 4 | 2, 6 | $\frac{4 + 4}{2} = 4$ |
| 3 | 4, 5 | 1, 3, 2, 6 | 3 | 5, 7 | $\frac{4 + 16}{2} = 10$ |

the CV estimate of the MSE is $(10 + 4 + 10)/3 = 8$. compare the training MSE of the mean predictor fitted on all six points: the mean is 4, and the MSE is $(9 + 1 + 4 + 4 + 1 + 9)/6 \approx 4.67$. the honest estimate is almost twice as large, even for a model with one parameter, because each fold's mean is pulled *away* from its own test points (they're not in its training set).

the tests check these three fold scores (as $-\text{MSE}$, so that higher is better) with a mean predictor of their own, the fold layout $\{0,1,2\}, \{3,4\}, \{5,6\}$ for $n = 7$, $k = 3$, and the stratified counts $14/5/1$.

## on paper

1. **largest remainders.** a stratified split puts 11 of 21 rows (7 per class, classes $0, 1, 2$) in the test set. compute the ideal counts, the floors, and the final counts.

<details><summary>solution</summary>

ideal: $11 \cdot 7/21 = 3.667$ for each class. floors: $3, 3, 3$, total 9, so 2 rows are left. the remainders tie ($0.667$ each), and ties go in sorted class order, so classes 0 and 1 get one more: $4, 4, 3$.

</details>

2. **why the CV estimate is larger.** for the mean predictor, show that leaving out a point $y_i$ and predicting it with the mean of the others gives error $y_i - \bar y_{-i} = \frac{n}{n-1}(y_i - \bar y)$, so leave-one-out MSE is $\left(\frac{n}{n-1}\right)^2$ times the training MSE.

<details><summary>solution</summary>

$\bar y_{-i} = \frac{n\bar y - y_i}{n - 1}$, so $y_i - \bar y_{-i} = \frac{(n - 1)y_i - n\bar y + y_i}{n - 1} = \frac{n}{n - 1}(y_i - \bar y)$. squaring and averaging over $i$ gives the factor $\left(\frac{n}{n-1}\right)^2$. with $n = 6$ that's $1.44$: the held-out residuals are systematically larger, because a point always pulls the mean towards itself when it's included.

</details>

3. **how selection leaks.** with labels independent of $d$ pure-noise features, the sample correlation of one feature with $y$ over $n$ rows is roughly $\mathcal N(0, 1/n)$. explain why the largest of $d$ such correlations grows like $\sqrt{2\log d / n}$, and why selecting the top 20 features on *all* rows makes every test fold easy.

<details><summary>solution</summary>

the maximum of $d$ independent standard normals is about $\sqrt{2\log d}$, so the best correlations are about $\sqrt{2\log d/n}$: at $n = 60$, $d = 2000$, about $0.5$, a strong correlation, entirely due to chance. those correlations were computed on all 60 rows, test rows included, so on the selected features the test rows *also* sit on the "right" side for their label. the classifier then gets them right in every fold. inside the fold, the selection only sees training rows, and on the test rows the chosen features are as random as any other.

</details>

4. **correlated folds.** the five fold scores of a 5-fold CV are each based on 20% of the data, but their training sets overlap by 75%. explain in words why $s/\sqrt5$ (the standard error of their mean, treating them as independent) underestimates the uncertainty of the CV estimate.

<details><summary>solution</summary>

the standard-error formula assumes independent measurements. the five models share most of their training data, so if the data happens to be "lucky" for this model, all five scores are high together: the scores are positively correlated, and the variance of their mean is $\frac{\sigma^2}{k}(1 + (k - 1)\rho)$ with $\rho > 0$, larger than $\sigma^2/k$. the sample variance of the scores can't see the shared part of the luck at all, because it's the same in every fold.

</details>

## experiment

`lab.py` runs your splitters and `cross_val_score` on three datasets where the labels carry no usable signal, so the true accuracy of any model is 50%:

1. feature selection and scaling done before splitting, versus inside the folds.
2. many recordings per patient: `KFold` versus `GroupKFold`.
3. a time series: shuffled `KFold` versus `TimeSeriesSplit`.

## check your understanding

1. you tuned a hyperparameter by comparing test-set scores, and now report the best test score. what's wrong, and what should you have done?

<details><summary>answer</summary>

you used the test set to choose, so the reported score is the best of several noisy estimates and is optimistically biased (topic 01, experiment 3). tune with a validation set or cross-validation on the training data, and evaluate the final choice once on the untouched test set. topic 06 automates this, and its nested CV estimates the whole procedure honestly.

</details>

2. why does stratified $k$-fold deal rows round-robin by class instead of just making sure each fold has *some* of every class?

<details><summary>answer</summary>

round-robin guarantees the tightest possible balance (each class's count per fold differs by at most one) with a two-line algorithm, and it keeps fold sizes balanced too. "some of every class" would still allow folds with very different class proportions, and metrics like precision and AUC are sensitive to the proportions.

</details>

3. a colleague's GroupKFold puts each group into a *random* fold. what goes wrong with a few very large groups?

<details><summary>answer</summary>

fold sizes can be wildly unequal: two big groups in one fold could make it half the dataset, leaving the other folds tiny and that fold's model trained on half the data. the greedy largest-first assignment balances the row counts (scikit-learn now also offers a shuffled variant that splits the group list evenly).

</details>

4. in `TimeSeriesSplit`, why do the early splits typically score worse than the later ones? is that a problem?

<details><summary>answer</summary>

early splits train on less history, so the model is worse; the test blocks may also come from a different regime. it isn't a problem, it's information: the curve of scores shows how much history the model needs and whether the series is drifting. just don't average the scores and call it "the" accuracy without looking.

</details>

5. which of these leak, and how badly? (a) imputing missing values with the column mean of the full dataset; (b) dropping rows whose target is an outlier, decided on the full dataset; (c) removing duplicate rows before splitting; (d) choosing the number of PCA components by looking at the test score.

<details><summary>answer</summary>

(a) leaks a little input information, usually minor. (b) leaks label information, and the test set no longer looks like deployment data (outliers will occur there): potentially serious. (c) is *required* when duplicates exist: otherwise the same row can land in train and test, which is a leak. (d) is tuning on the test set, a real leak of the same kind as question 1.

</details>

6. why does `cross_validate` fit `clone(estimator)` rather than `estimator`?

<details><summary>answer</summary>

so every fold starts from the same unfitted model with the same hyperparameters, no fold sees another fold's fitted state, and the object you passed in is left untouched. it's also what makes the function work for any estimator, including searches and pipelines that contain other estimators.

</details>

## common bugs

- **preprocessing before splitting.** fitting a scaler, imputer, encoder or feature selector on all the data, then cross-validating. symptom: CV scores better than the model ever achieves in production; dramatic when the step uses the labels.
- **fitting the estimator itself.** `estimator.fit(...)` inside the CV loop instead of `clone(estimator).fit(...)`. symptom: the passed-in model ends up fitted on the last fold; warm-starting models leak between folds.
- **splitting rows when the unit is a group.** patients, users or duplicated images on both sides. symptom: near-perfect CV, poor real-world performance; GroupKFold scores far lower than KFold.
- **shuffling time series.** symptom: CV scores far above what walk-forward evaluation gives.
- **off-by-one fold sizes.** giving the extra rows to the *last* folds, or rounding the test size down. symptom: indices that differ from scikit-learn's by one row at the fold boundaries.
- **forgetting `groups` in `cross_validate`.** calling `cv.split(X, y)` without passing `groups` on. symptom: `GroupKFold` raises "needs groups" even though you gave them.

## going deeper

- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, section 7.10, including 7.10.2 "the wrong and right way to do cross-validation" (the feature-selection leak). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Ambroise & McLachlan (2002), "Selection bias in gene extraction on the basis of microarray gene-expression data", *PNAS* 99(10): 6562–6566. the cautionary tale at the top of this lesson. [doi:10.1073/pnas.102102699](https://doi.org/10.1073/pnas.102102699)
- Stone (1974), "Cross-validatory choice and assessment of statistical predictions", *Journal of the Royal Statistical Society B* 36(2): 111–147. an early formal treatment of cross-validation. [doi:10.1111/j.2517-6161.1974.tb00994.x](https://doi.org/10.1111/j.2517-6161.1974.tb00994.x)
- Bengio & Grandvalet (2004), "No unbiased estimator of the variance of K-fold cross-validation", *JMLR* 5: 1089–1105. why fold scores understate uncertainty. [jmlr.org/papers/v5/grandvalet04a](https://www.jmlr.org/papers/v5/grandvalet04a.html)
- Kaufman, Rosset, Perlich & Stitelman (2012), "Leakage in data mining: formulation, detection, and avoidance", *ACM Transactions on Knowledge Discovery from Data* 6(4). a taxonomy of leaks, many from real competitions. [doi:10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579)
- Roberts et al. (2017), "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", *Ecography* 40(8): 913–929. blocked and grouped CV in depth. [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)
- Kapoor & Narayanan (2023), "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* 4(9). a survey of leakage across hundreds of published papers. [doi:10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)

## connections

- ← builds on: `clone` and the estimator protocol (topic 01), the variance of a mean and correlated variables (ch 04), confidence intervals (ch 05), `as_rng` (ch 00).
- → used in: metrics and scorers (topic 03), learning and validation curves and the bias–variance estimator (topic 04), pipelines that keep preprocessing inside the folds (topic 05), grid, random and nested search (topic 06), the capstone, and every evaluation from chapter 08 on (out-of-bag estimates in ch 10 are a free form of cross-validation; early stopping in ch 15 uses a validation split).
