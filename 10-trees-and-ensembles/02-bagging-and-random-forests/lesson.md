# bagging & random forests

## why this matters

in topic 01's lab, two fully grown trees fitted on the same data minus 5% of the rows disagreed on 15% of the test points. a deep tree has low bias (it can follow any boundary) and enormous variance (which boundary it follows depends on accidents of the sample). you can trade variance for bias by pruning, but then you lose the flexibility that made trees attractive.

there's a better trade. if you could grow many trees on independent training sets and average them, the variance would shrink and the bias would stay put. you don't have independent training sets, but you can fake them by resampling the one you have: that's **bagging** (bootstrap aggregating, Breiman, 1996). add one more trick, letting each split consider only a random subset of the features so the trees don't all look alike, and you get the **random forest** (Breiman, 2001): still one of the strongest off-the-shelf models for tabular data, nearly tuning-free, and with a built-in estimate of its own test error that costs nothing.

## warm-up: predict first

1. you draw $n$ rows with replacement from $n$. what fraction of the original rows never gets drawn, for large $n$?
2. you average 100 predictors, each with variance $\sigma^2$, whose pairwise correlation is 0.1. what's the variance of the average: about $\sigma^2/100$, about $\sigma^2/10$, or something else?
3. does a random forest overfit if you keep adding trees, the way a tree overfits if you keep adding depth?
4. you bag decision stumps (depth 1) instead of deep trees. does the average improve on a single stump as much as bagged deep trees improve on a single deep tree?
5. at each split a forest tree may only look at 3 of the 10 features. is each tree better or worse than a normal tree? is the forest?

## prerequisites check

- [ ] your `DecisionTreeClassifier` and `DecisionTreeRegressor` work, including `max_features` and `rng` (→ topic 01)
- [ ] you know the bootstrap and why it approximates resampling from the population (→ ch 05 · topic 03)
- [ ] you can compute the variance of a sum of correlated random variables (→ ch 04 · topic 03)
- [ ] you know the bias–variance decomposition of squared error (→ ch 07 · topic 04)

## core ideas

### idea 1: averaging and the bootstrap

suppose $h_1, \dots, h_B$ are predictions at a point $x$, each with variance $\sigma^2$, and every pair has correlation $\rho$. the variance of their average is

$$\operatorname{Var}\Big(\frac1B\sum_bh_b\Big) = \frac{1}{B^2}\Big(B\sigma^2 + B(B - 1)\rho\sigma^2\Big) = \rho\sigma^2 + \frac{1 - \rho}{B}\sigma^2.$$

(on paper 1.) two lessons in one line. the second term vanishes as $B$ grows: averaging kills the *independent* part of the variance. the first term doesn't: the shared part, $\rho\sigma^2$, is a floor no number of predictors can get under. so you want many predictors that are individually flexible (low bias) and as uncorrelated as possible. and averaging never touches the bias: the average of the $h_b$ has the same expectation as each one.

where do you get many training sets from one? the **bootstrap** (ch 05): draw $n$ rows *with replacement* from your $n$ rows. some rows appear twice or more, and some not at all. the chance that a given row is never drawn is $(1 - 1/n)^n \to e^{-1} \approx 0.368$, so each bootstrap sample contains about 63% of the distinct rows, and leaves out the rest (warm-up 1). **bagging** fits one model per bootstrap sample and averages.

bootstrap samples overlap a lot, so the bagged trees are positively correlated, and $\rho$ is the number to fight. but they differ enough: a deep tree is so sensitive to its data that dropping a third of the rows (and duplicating others) changes most of its small leaves.

> **spec.** `bootstrap_sample(n, rng=None)` returns `(idx, oob)`: `idx` the $n$ row indices drawn with `as_rng(rng).integers(0, n, size=n)`, and `oob` a bool mask $(n,)$, True for the rows that `idx` never contains.

> **▶ build: stage 1.** implement `bootstrap_sample`.

### idea 2: random forests

bagged trees still look alike. if one feature is a strong predictor, nearly every tree splits on it at the root, the trees agree near the top, and their errors are correlated. Breiman's fix: at **every split**, let the tree choose among a random subset of `max_features` features only. now in many trees the strong feature isn't even available at the root, other features get a chance, and the trees become genuinely different.

this makes each tree *worse*: it's forced to make suboptimal splits. it also makes the forest *better*, as long as the loss in individual accuracy is smaller than the gain from lower correlation (warm-up 5). Breiman's recommendations, still the defaults, are $\sqrt d$ features per split for classification and around $d/3$ for regression; scikit-learn's regression default is all $d$ (plain bagging), which is often about as good. your topic-01 tree already does the per-split sampling (`max_features` with `rng`); the forest just passes the setting on and gives every tree the same generator, so the trees draw different subsets.

for classification, the forest averages the trees' **class probabilities** (soft voting) and predicts the most probable class. one detail: a bootstrap sample can miss a rare class entirely, so a tree's `predict_proba` may have fewer columns than the forest has classes. fit the trees on integer class codes $0..k-1$ and use each tree's `classes_` to place its columns into an $(n, k)$ array.

> **spec.** `RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_leaf=1, max_features='sqrt', bootstrap=True, oob_score=False, rng=None)`. `fit` sets `classes_`, draws a bootstrap sample per tree (every row once if `bootstrap=False`), fits a `DecisionTreeClassifier` with the forest's settings and `rng=` the forest's generator on the sampled rows and codes, keeps the trees in `estimators_` and their row indices in `estimators_samples_`, computes OOB estimates if `oob_score` (stage 3; with `bootstrap=False` that's a `ValueError`), and returns `self`. `_tree_proba(tree, X)` returns one tree's probabilities as $(n, k)$; `predict_proba` averages them over the trees.

> **🔮 predict:** with `bootstrap=False` and `max_features=None`, how do the trees of a forest differ from each other?

> **▶ build: stage 2.** implement `RandomForestClassifier.fit`, `_tree_proba` and `predict_proba`.

### idea 3: out-of-bag estimates

every tree leaves out about 37% of the rows. for row $i$, the trees that didn't see it form a small forest (about $0.37B$ trees) that was fitted without row $i$. their averaged prediction for $x_i$ is an honest held-out prediction, and doing this for every row gives the **out-of-bag** (OOB) predictions: a validation set for free, with no data held back and no refitting. the OOB accuracy (or $R^2$) estimates the forest's test performance.

it's slightly pessimistic: each OOB prediction comes from about $0.37B$ trees, not $B$, and from trees that saw about 63% of the distinct rows. with a few dozen trees or more, the difference is small. rows that happened to be in every bootstrap sample (likely only when $B$ is tiny) have no OOB prediction: mark them `nan` and leave them out of the score.

OOB is the same idea as cross-validation (ch 07), with the folds chosen by the bootstrap. it's the reason forests are so convenient: you can tune `max_features` or `min_samples_leaf` on the OOB score without a separate validation split.

> **spec.** `_compute_oob(X, codes)`: for each tree, the rows not in its sample get that tree's $(n, k)$ probabilities added to a running total, and a count. `oob_decision_function_` $(n, k)$ = total / count, `nan` rows where the count is 0. `oob_score_` = accuracy of its argmax against the codes, over rows with a count.

> **▶ build: stage 3.** implement `_compute_oob`.

### idea 4: regression forests, and why more trees can't hurt

the regression forest is the same algorithm with `DecisionTreeRegressor` and a plain average. its OOB prediction for row $i$ is the average of the trees that didn't see it, and `oob_score_` is the $R^2$ of those predictions.

adding trees never makes a forest worse in expectation. for squared error, the forest's test MSE is the average tree's MSE minus the spread of the trees around their average (the "ambiguity" decomposition, on paper 2), and the spread can only help; as $B \to \infty$ the forest converges to a limit (the expectation over the randomness), so its error converges too (Breiman, 2001). in practice, test error falls quickly and then flattens at the $\rho\sigma^2$ floor (warm-up 3). what *can* overfit is each tree: but deep trees are what you want inside a forest, because their high variance is exactly what averaging removes and their low bias is what it keeps (warm-up 4).

the forest's **feature importances** are the mean of its trees' impurity importances, renormalized. they're smoother than one tree's, and just as biased (topic 05).

> **spec.** `RandomForestRegressor(n_estimators=100, *, max_depth=None, min_samples_leaf=1, max_features=1.0, bootstrap=True, oob_score=False, rng=None)`: `fit` like the classifier with `DecisionTreeRegressor`s; with `oob_score`, `oob_prediction_` $(n,)$ (`nan` where no tree left the row out) and `oob_score_` = their $R^2$ over the rows that have one. `predict` averages the trees. both forests get a `feature_importances_` property.

> **🔮 predict:** you bag depth-1 stumps. their bias is high and their variance low. how much does averaging 50 of them reduce the test error?

> **▶ build: stage 4.** implement `RandomForestRegressor.fit` and `predict`, and both forests' `feature_importances_`.

## worked example

six training rows, a forest of three trees. the bootstrap samples come from `bootstrap_sample(6, rng=...)` with seeds 0, 1 and 3:

| tree | rows drawn | out of bag |
|---|---|---|
| A | 5, 3, 3, 1, 1, 0 | 2, 4 |
| B | 2, 3, 4, 5, 0, 0 | 1 |
| C | 4, 0, 1, 1, 1, 4 | 2, 3, 5 |

suppose the trees predict these probabilities of class 1 for the training rows (each tree only matters on its own out-of-bag rows):

| row | true class | A | B | C | OOB trees | OOB $P(1)$ | OOB prediction |
|---|---|---|---|---|---|---|---|
| 0 | 1 | | | | none | nan | (none) |
| 1 | 0 | | 0.2 | | B | 0.2 | 0 ✓ |
| 2 | 1 | 0.9 | | 0.4 | A, C | 0.65 | 1 ✓ |
| 3 | 0 | | | 0.7 | C | 0.7 | 1 ✗ |
| 4 | 1 | 0.6 | | | A | 0.6 | 1 ✓ |
| 5 | 0 | | | 0.1 | C | 0.1 | 0 ✓ |

row 0 is in every sample, so it has no OOB prediction. with 3 trees that's common: a row is in all three samples with probability $(1 - (5/6)^6)^3 \approx 0.29$. with 100 trees it's about $10^{-18}$. the OOB accuracy is $4/5 = 0.8$ over the five rows that have a prediction. a training-set accuracy would have used all three trees on every row, including the trees that memorized it.

`bootstrap_sample(6, rng=0)` must return exactly tree A's sample and out-of-bag rows.

## on paper

1. **the variance of an average.** derive $\operatorname{Var}(\bar h) = \rho\sigma^2 + (1 - \rho)\sigma^2/B$. with $\sigma^2 = 8$ and $\rho = 0.1$, what are the variances for $B = 1, 10, 100$ and the limit?

<details><summary>solution</summary>

$\operatorname{Var}(\sum_bh_b) = \sum_b\operatorname{Var}(h_b) + \sum_{b\ne c}\operatorname{Cov}(h_b, h_c) = B\sigma^2 + B(B - 1)\rho\sigma^2$; divide by $B^2$. numbers: 8, $0.8 + 7.2/10 = 1.52$, $0.8 + 0.072 = 0.872$, and the floor 0.8. ninety extra trees after the first ten buy less than half the remaining variance.

</details>

2. **ambiguity.** for predictions $h_1, \dots, h_B$ at a point with target $y$ and average $\bar h$, show $(\bar h - y)^2 = \frac1B\sum_b(h_b - y)^2 - \frac1B\sum_b(h_b - \bar h)^2$.

<details><summary>solution</summary>

expand $\frac1B\sum_b(h_b - y)^2 = \frac1B\sum_b\big((h_b - \bar h) + (\bar h - y)\big)^2 = \frac1B\sum_b(h_b - \bar h)^2 + (\bar h - y)^2$, since the cross term $\frac2B(\bar h - y)\sum_b(h_b - \bar h) = 0$. so the ensemble's squared error is the average member's error minus their spread (Krogh & Vedelsby, 1995): disagreement between members is pure gain, provided each member stays accurate.

</details>

3. **out of bag.** show that the probability that row $i$ is out of bag for a given tree is $(1 - 1/n)^n$, and that it tends to $1/e$. with $B$ trees, what's the probability that row $i$ is in *every* sample?

<details><summary>solution</summary>

each of the $n$ draws misses row $i$ with probability $1 - 1/n$, independently: $(1 - 1/n)^n$. taking logs, $n\log(1 - 1/n) = -1 - \frac{1}{2n} - \dots \to -1$. being in every one of $B$ independent samples: $(1 - (1 - 1/n)^n)^B \approx 0.632^B$: 0.25 for $B = 3$, $10^{-20}$ for $B = 100$.

</details>

## experiment

`lab.py` investigates three things with your forests:

1. how the variance of the forest's prediction falls as trees are added, for bagging and for a random forest, compared with $\rho\sigma^2 + (1 - \rho)\sigma^2/B$; and how the OOB estimate compares with test performance.
2. how `max_features` trades the accuracy of each tree against the correlation between trees.
3. which trees benefit from bagging: stumps, medium trees or fully grown ones.

## check your understanding

1. why does bagging reduce variance but not bias?

<details><summary>answer</summary>

every bagged tree is fitted by the same procedure on a sample from (approximately) the same distribution, so they share an expected prediction; the average has that same expectation, hence the same bias. its variance is smaller by the formula in idea 1.

</details>

2. if bootstrap samples were completely independent datasets, what would $\rho$ be? why isn't it that for bagging?

<details><summary>answer</summary>

it would be the correlation between trees fitted on independent datasets from the population, which is 0 conditional on nothing shared. bootstrap samples all come from the *same* $n$ rows, so the trees share that dataset's quirks: $\rho > 0$, and the floor $\rho\sigma^2$ is the part of the variance that comes from the training set itself.

</details>

3. a colleague sets `max_features=1`. what happens to the trees, and when might it still work well?

<details><summary>answer</summary>

every split is on a random feature (with only its threshold optimized), so trees are weak, and their splits are much less alike. but their *errors* aren't necessarily less correlated: weak trees tend to miss the same structure, and the lab measures an error correlation back at 0.37, about as high as plain bagging's (0.35). it works best when most features are informative; with many noise features, most splits are wasted on noise, as the lab shows.

</details>

4. why can you tune a forest on its OOB score without a validation set?

<details><summary>answer</summary>

each OOB prediction comes only from trees that never saw that row, so it's a held-out prediction. the OOB score is a (slightly pessimistic) estimate of test performance, like cross-validation with bootstrap folds. the usual caveat applies: after choosing among many settings, the chosen OOB score is optimistic.

</details>

5. why are fully grown trees the right base learner for a forest, when they're the worst single trees?

<details><summary>answer</summary>

averaging only removes variance; it can't remove bias. fully grown trees have the lowest bias and the highest variance, so averaging keeps their strength and removes their weakness. shallow trees bring bias that no amount of averaging fixes.

</details>

6. a forest's test error stopped improving at 200 trees. would a forest of 2,000 trees overfit?

<details><summary>answer</summary>

no: more trees bring the forest closer to its $B \to \infty$ limit, so the error stays flat (up to noise). you pay only in memory and prediction time.

</details>

## common bugs

- **sampling without replacement.** symptom: no out-of-bag rows when you use a permutation; or about half the rows OOB if you subsample. use `integers(0, n, size=n)`.
- **stacking tree probabilities blindly.** symptom: a shape error, or probabilities in the wrong columns, whenever a bootstrap sample misses a class. place columns by `tree.classes_`.
- **the same seed for every tree.** symptom: with `bootstrap=False` and `max_features` set, all trees identical. pass one generator to all trees (or distinct seeds), never the same int.
- **scoring OOB with all trees.** symptom: an OOB score equal to the training accuracy (often 1.0). only trees that didn't see a row may predict it.
- **counting rows with no OOB prediction.** symptom: `nan` scores or wrong accuracies with few trees. leave those rows out.

## going deeper

- Breiman (1996), "Bagging predictors", *Machine Learning* 24(2): 123–140. [doi:10.1007/BF00058655](https://doi.org/10.1007/BF00058655)
- Breiman (2001), "Random forests", *Machine Learning* 45(1): 5–32. strength and correlation, OOB estimates, importances. [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
- Ho (1998), "The random subspace method for constructing decision forests", *IEEE TPAMI* 20(8): 832–844. [doi:10.1109/34.709601](https://doi.org/10.1109/34.709601)
- Krogh & Vedelsby (1995), "Neural network ensembles, cross validation, and active learning", *Advances in Neural Information Processing Systems* 7: 231–238. the ambiguity decomposition.
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, §8.7 (bagging) and chapter 15 (random forests, including the $\rho\sigma^2$ analysis). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Louppe (2014), "Understanding random forests: from theory to practice", PhD thesis, chapter 4 (bias–variance of forests). [arXiv:1407.7502](https://arxiv.org/abs/1407.7502)

## connections

- ← builds on: CART trees with `max_features` (topic 01), the bootstrap (ch 05 · topic 03), variance of sums (ch 04), bias–variance and cross-validation (ch 07 · topics 02 and 04).
- → used in: boosting as the opposite strategy, reducing bias with shallow trees fitted in sequence (topics 03–04); the biased impurity importances and permutation importance computed on OOB-like held-out data (topic 05); isolation forests, random trees for anomaly detection (ch 11 · topic 06); deep ensembles and uncertainty from disagreement (ch 28).
