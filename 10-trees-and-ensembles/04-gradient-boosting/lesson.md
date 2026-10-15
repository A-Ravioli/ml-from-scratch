# gradient boosting

## why this matters

look at the winning solutions of almost any machine-learning competition on tabular data from the last decade (sales forecasts, credit risk, click-through rates, physics event classification) and you'll find the same model: **gradient-boosted trees**, usually XGBoost (Chen & Guestrin, 2016), LightGBM or CatBoost. large benchmark studies still find them ahead of deep networks on typical tables (Grinsztajn et al., 2022). they're also what runs in production for ranking, fraud and pricing at a huge number of companies.

the idea is AdaBoost's, freed from the exponential loss. topic 03 showed that AdaBoost adds trees that fix what the current ensemble gets wrong. Friedman (2001) asked: what does "wrong" mean for *any* differentiable loss? his answer: the negative gradient of the loss with respect to the current predictions. each new tree is a step of gradient descent, not in the space of parameters, but in the space of functions. XGBoost added the second derivative (a Newton step), explicit regularization of the leaves, and the engineering that makes it fast: features binned once into small integers, so that finding a split is a histogram scan instead of a sort.

you'll build all of that in NumPy, and in the capstone race it against scikit-learn's `HistGradientBoostingRegressor`, which is built the same way.

## warm-up: predict first

1. a feature has 20,000 distinct values. you replace it by its bin number among 16 quantile bins. how much test accuracy does a boosted model lose?
2. squared loss, current prediction 4 for a point with target 1. what's the gradient of $\frac12(F - y)^2$ with respect to $F$, and what does a tree fitted to the negative gradient try to predict?
3. a leaf holds rows with gradients summing to $G = 7$ and hessians to $H = 3$, with L2 penalty $\lambda = 1$. what value should the leaf predict?
4. you halve the learning rate. how many more trees do you need for the same training loss: twice as many, more than twice, or fewer?
5. the best split of a node has gain 12.25. the regularization $\gamma$ is 13. does the tree split?

## prerequisites check

- [ ] you can build a tree with an explicit stack and parallel node arrays (→ topic 01)
- [ ] you know AdaBoost as forward stagewise additive modelling (→ topic 03)
- [ ] you can take the gradient and second derivative of the log loss with respect to the logit (→ ch 03, ch 07 · topic 03)
- [ ] you know gradient descent and Newton's method (→ ch 06)

## core ideas

### idea 1: bin the features once

topic 01's split search sorts every feature at every node: $O(n\log n)$ per feature per node, and a lot of Python overhead for small nodes. histogram-based boosting does the expensive part once. before training, each feature is cut into at most `n_bins` (typically 256) bins at its **quantiles**, and every value is replaced by its bin number, a `uint8` code. after that, a node's candidate splits are just the `n_bins - 1` boundaries between bins, and everything the split search needs is a histogram over bins: $O(n)$ to build, $O(\text{n\_bins})$ to scan, no sorting ever again.

two cases. if a feature has at most `n_bins` distinct values, give each value its own bin, with edges at the midpoints between consecutive values (nothing is lost). otherwise use the quantiles at levels $\frac{1}{n_\text{bins}}, \dots, \frac{n_\text{bins} - 1}{n_\text{bins}}$, and remove duplicates (a feature that is 0 for 60% of the rows has many equal quantiles). the code of $x$ is the number of edges strictly below it, `np.searchsorted(edges, x, side='left')`, which gives the property everything else relies on:

$$\text{code}(x) \le b \iff x \le \text{edges}[b].$$

so a split "code $\le b$" *is* a threshold split at `edges[b]`, and new data is binned with the training edges (values beyond the range land in the first or last bin).

what does binning cost in accuracy? a split can now only fall on one of 255 quantile boundaries instead of between any two of 20,000 values. with 256 bins that's almost never visible, and the coarseness even acts as a mild regularizer. with far fewer bins it depends on the data: a smooth effect is captured by a few dozen thresholds spread over hundreds of trees, a sharp one (a price jump at a particular latitude) is not. warm-up 1 and the lab measure it on real data.

> **spec.** `quantile_bin_edges(X, n_bins=256)` returns a list of $d$ sorted arrays (strictly increasing, fewer than `n_bins` entries each) as described; `n_bins` outside $[2, 256]$ is a `ValueError`. `bin_data(X, edges)` returns the codes as a `uint8` array $(n, d)$.

> **▶ build: stage 1.** implement `quantile_bin_edges` and `bin_data`.

### idea 2: second-order splits

**gradient boosting.** you have a model $F(x)$ and a differentiable loss $\ell(y, F)$. at each training point, the **gradient** $g_i = \partial\ell(y_i, F(x_i))/\partial F$ says which way to move the prediction to reduce the loss. gradient descent would update every prediction by $-\eta g_i$, but that only helps at the training points. so fit a regression tree to $-g$: it generalizes "move this way" to nearby points, and add it to $F$ with a small step $\eta$. for squared loss $\frac12(F - y)^2$, $g = F - y$ and the negative gradient is the residual $y - F$: each tree fits what's left (warm-up 2).

**Newton's version.** XGBoost uses the second derivative $h_i = \partial^2\ell/\partial F^2$ too. a tree $f$ assigns a value $w_j$ to each leaf $j$; expanding the loss to second order around the current $F$ and adding penalties on the leaf values and the number of leaves $T$:

$$\sum_i\Big[g_if(x_i) + \tfrac12h_if(x_i)^2\Big] + \tfrac12\lambda\sum_jw_j^2 + \gamma T = \sum_j\Big[G_jw_j + \tfrac12(H_j + \lambda)w_j^2\Big] + \gamma T,$$

where $G_j$ and $H_j$ are the sums of $g$ and $h$ over the rows in leaf $j$. each leaf is a separate one-dimensional quadratic, minimized by

$$w_j^* = -\frac{G_j}{H_j + \lambda},\qquad\text{leaving}\quad -\frac12\sum_j\frac{G_j^2}{H_j + \lambda} + \gamma T$$

(on paper 1). splitting a leaf into left and right children therefore improves the objective by

$$\text{gain} = \frac12\Big[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Big] - \gamma,$$

and a split is only worth making if its gain is positive (warm-up 5). three knobs regularize each tree: $\lambda$ shrinks leaf values toward 0 (and damps leaves with small $H$), $\gamma$ is a minimum gain per split, and `min_child_weight` requires $H_L, H_R \ge$ that value (for squared loss, $h = 1$, so it's a minimum number of rows per child).

with the codes, the split search is the same cumulative-sum trick as topic 01, on histograms instead of sorted rows: $G_\text{hist}[j, b]$ = sum of $g$ over the node's rows whose feature $j$ has code $b$ (one `np.bincount`), the left child of "code $\le b$" gets `cumsum(G_hist, axis=1)[:, b]`, the right child gets the rest, and one vectorized `split_gain` call scores all $d\times(\text{n\_bins} - 1)$ candidates.

> **spec.** `gradient_histograms(codes, g, h, n_bins)` returns `(G, H)`, each $(d, \text{n\_bins})$, built with `np.bincount` (no loop over bins or rows). `split_gain(G_L, H_L, G_R, H_R, reg_lambda=1.0, gamma=0.0)` and `leaf_weight(G, H, reg_lambda=1.0)` are the formulas above, elementwise. `best_hist_split(G_hist, H_hist, *, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0)` returns `(feature, bin, gain)` for the best "code $\le$ bin" split, considering only splits where both children have hessian sums $\ge$ `min_child_weight` (and $> 0$); ties go to the lowest feature, then the lowest bin; `(-1, -1, -inf)` if none is valid.

> **▶ build: stage 2.** implement `gradient_histograms`, `split_gain`, `leaf_weight` and `best_hist_split`.

### idea 3: one tree on histograms

growing a tree is topic 01's explicit-stack loop with three changes: each node carries its histograms, the split is "code $\le b$", and a leaf stores $-G/(H + \lambda)$ (every node does, so a node that turns out not to split is already a leaf). split while the depth is below `max_depth` and the best gain is positive. XGBoost grows trees level by level to a fixed depth (3–8 is typical); LightGBM grows the leaf with the largest gain first. depth-limited growth is the simpler and more common default.

**histogram subtraction** halves the work. the parent's histograms are the sum of its children's, so build the histograms of the *smaller* child from its rows and get the larger child's by subtraction: $G_\text{large} = G_\text{parent} - G_\text{small}$. now each level of the tree touches at most half the rows, not all of them (on paper 4). a tree of depth $D$ costs at most about $n\cdot d\cdot(1 + D/2)$ bincount updates (the root, then at most half the rows per level) plus a scan over bins per node, which is why your NumPy booster can train hundreds of trees on 16,000 rows in seconds.

> **spec.** `HistGradientTree(max_depth=3, min_child_weight=1.0, reg_lambda=1.0, gamma=0.0, n_bins=256)`: `fit(codes, g, h)` grows the tree depth-first with an explicit stack and stores `feature_`, `bin_` (both $-1$ at leaves), `children_left_`, `children_right_` and `value_` (every node's $-G/(H + \lambda)$); returns `self`. `predict(codes)` returns each row's leaf value $(n,)$.

> **🔮 predict:** with $\lambda = 0$, $\gamma = 0$ and every $h_i = 1$, what is this tree, exactly, in terms of topic 01?

> **▶ build: stage 3.** implement `HistGradientTree.fit` and `predict`.

### idea 4: boosting with shrinkage

for regression with squared loss: start from the constant that minimizes the loss, $F_0 = \bar y$; then for $m = 1, \dots, M$: compute $g = F - y$ and $h = 1$, fit a `HistGradientTree`, and update $F \leftarrow F + \eta\,f_m$. the **learning rate** (shrinkage) $\eta$ deliberately takes only a fraction of each Newton step. smaller steps need more trees, but each tree then only corrects a little, the later trees get to work on a smoother residual, and the ensemble generalizes better: in practice $\eta = 0.05$–$0.1$ with early stopping on a validation set beats $\eta = 1$ almost always (warm-up 4, and the lab).

**stochastic gradient boosting** (Friedman, 2002) fits each tree on a random subsample of the rows, drawn without replacement (`subsample` $< 1$, typically 0.5–0.8). it's another regularizer, it decorrelates the trees a little like a forest's bootstrap, and it makes each tree cheaper. the predictions $F$ are still updated for every row.

the model is $F(x) = F_0 + \eta\sum_mf_m(x)$, evaluated on new data by binning it with the *training* edges. `staged_predict` yields $F$ after each tree, which is how you find the best number of trees on a validation set without refitting.

> **spec.** `subsample_rows(n, fraction, rng=None)`: `np.arange(n)` if `fraction >= 1`, else `round(fraction * n)` (at least 1) distinct indices drawn without replacement, sorted. `GradientBoostingRegressor(n_estimators=100, *, learning_rate=0.1, max_depth=3, min_child_weight=1.0, reg_lambda=1.0, gamma=0.0, n_bins=256, subsample=1.0, rng=None)`: `fit` stores `bin_edges_` (from the training data), `init_` $= \bar y$ and the trees in `estimators_`, fitting each tree on `subsample_rows(n, subsample, gen)` with one generator `gen = as_rng(rng)`; returns `self`. `predict` and `staged_predict` bin $X$ with `bin_edges_`.

> **🔮 predict:** with learning rate 1, the test error of a depth-4 booster on California housing is lowest after a couple of dozen trees. with learning rate 0.1, will its best test error be the same, better or worse, and after how many trees?

> **▶ build: stage 4.** implement `subsample_rows` and `GradientBoostingRegressor.fit`, `predict` and `staged_predict`.

### idea 5: classification with the log loss

for binary classification, let $F(x)$ be the **log-odds** of class 1, $p = \sigma(F)$, and use the log loss $\ell = -[t\log p + (1 - t)\log(1 - p)]$ with $t \in \{0, 1\}$. its derivatives with respect to $F$ are beautifully simple (on paper 2):

$$g = p - t,\qquad h = p(1 - p).$$

the gradient is the familiar "predicted minus actual"; the hessian is largest where the model is unsure ($p \approx 0.5$) and vanishes where it's confident. a leaf's Newton step $-\sum g/(\sum h + \lambda)$ is then a regularized version of one IRLS step of logistic regression, restricted to the rows in the leaf. start from the best constant, $F_0 = \log\frac{\bar t}{1 - \bar t}$, the log-odds of the class-1 rate.

compared with AdaBoost's exponential loss, the log loss grows only linearly in the margin for confidently wrong points, so mislabeled examples pull much less (topic 03's lab), and the shrinkage plus $\lambda$ keep $F$ from exploding: gradient-boosted probabilities are usually much better calibrated than AdaBoost's.

> **spec.** `GradientBoostingClassifier` takes the same hyperparameters, is binary (`ValueError` otherwise), sets `classes_` (class 1 = `classes_[1]`), `bin_edges_`, `init_` (the log-odds above, with the rate clipped to $[10^{-12}, 1 - 10^{-12}]$) and `estimators_`; each tree is fitted to $g = p - t$, $h = p(1 - p)$ with $p = \sigma(F)$. `decision_function` returns $F$, `staged_decision_function` yields it after each tree, `predict_proba` returns $[1 - \sigma(F), \sigma(F)]$ (computed without overflow).

> **▶ build: stage 5.** implement `GradientBoostingClassifier.fit`, `decision_function`, `staged_decision_function` and `predict_proba`.

## worked example

six rows, one feature already binned into 3 bins, targets $y = (1, 1, 3, 5, 7, 7)$, squared loss, $\lambda = 1$.

| row | code | $y$ | $g = F_0 - y$ | $h$ |
|---|---|---|---|---|
| 1–2 | 0 | 1, 1 | 3, 3 | 1, 1 |
| 3 | 0 | 3 | 1 | 1 |
| 4 | 1 | 5 | −1 | 1 |
| 5–6 | 2 | 7, 7 | −3, −3 | 1, 1 |

$F_0 = \bar y = 4$. the histograms: $G = (7, -1, -6)$, $H = (3, 1, 2)$; the node totals are $G = 0$, $H = 6$.

- split after bin 0: left $(G_L, H_L) = (7, 3)$, right $(-7, 3)$: gain $= \frac12\big[\frac{49}{4} + \frac{49}{4} - \frac{0}{7}\big] = 12.25$.
- split after bin 1: left $(6, 4)$, right $(-6, 2)$: gain $= \frac12\big[\frac{36}{5} + \frac{36}{3}\big] = 9.6$.

the best split is after bin 0 with gain 12.25. its leaves predict $-7/(3 + 1) = -1.75$ (left) and $+1.75$ (right), so with $\eta = 1$ the new predictions are $2.25$ and $5.75$. the exact residual means are $-2.33$ and $+2.33$: $\lambda$ shrank them by a factor $3/4$. with $\gamma = 13$ the gain is $-0.75$ and the node stays a leaf (with value $-0/(6 + 1) = 0$).

your `gradient_histograms`, `best_hist_split` and `leaf_weight` must reproduce these numbers.

## on paper

1. **the optimal leaf.** for one leaf with sums $G$, $H$, minimize $Gw + \frac12(H + \lambda)w^2$ over $w$, and derive the gain of a split from the minimum values.

<details><summary>solution</summary>

the derivative $G + (H + \lambda)w = 0$ gives $w^* = -G/(H + \lambda)$, and the minimum is $-\frac12G^2/(H + \lambda)$. the objective of a tree is the sum over leaves plus $\gamma T$. replacing one leaf (with $G = G_L + G_R$, $H = H_L + H_R$) by two changes it by $-\frac12\frac{G_L^2}{H_L + \lambda} - \frac12\frac{G_R^2}{H_R + \lambda} + \frac12\frac{G^2}{H + \lambda} + \gamma$; the gain is minus that.

</details>

2. **log loss derivatives.** with $p = \sigma(F)$ and $\ell = -t\log p - (1 - t)\log(1 - p)$, show $\partial\ell/\partial F = p - t$ and $\partial^2\ell/\partial F^2 = p(1 - p)$.

<details><summary>solution</summary>

$\sigma'(F) = \sigma(1 - \sigma)$. so $\partial\ell/\partial F = -t\frac{p(1 - p)}{p} + (1 - t)\frac{p(1 - p)}{1 - p} = -t(1 - p) + (1 - t)p = p - t$. differentiating again, $\partial(p - t)/\partial F = p(1 - p)$.

</details>

3. **CART as a special case.** show that with $h_i = 1$, $\lambda = 0$, $\gamma = 0$ and $g_i = F - y_i$ for a constant $F$, the gain is half the decrease in the sum of squared errors, and the leaf value is the mean residual.

<details><summary>solution</summary>

with $h = 1$, $H$ is the number of rows $n$ and $-G = \sum(y_i - F) = n\bar r$ for residuals $r = y - F$. the leaf value is $-G/n = \bar r$. the term $G^2/H = n\bar r^2$, and the SSE of a node around its mean is $\sum r_i^2 - n\bar r^2$; the $\sum r_i^2$ terms cancel between parent and children, leaving $\text{SSE}_\text{parent} - \text{SSE}_L - \text{SSE}_R = n_L\bar r_L^2 + n_R\bar r_R^2 - n\bar r^2 = 2\cdot$ gain. the tree is CART on the bin codes, which the checks verify against scikit-learn.

</details>

4. **histogram subtraction.** a node with $m$ rows splits into children of sizes $m_1 \le m_2$. how many rows does building both children's histograms touch directly, and with subtraction? over a whole level of a tree, what's the worst case?

<details><summary>solution</summary>

directly: $m_1 + m_2 = m$. with subtraction: $m_1 \le m/2$, plus an $O(d\cdot\text{n\_bins})$ subtraction. a level's nodes partition the rows, so each level touches at most $n/2$ rows instead of $n$, and very unbalanced splits make it far less.

</details>

## experiment

`lab.py` investigates three things with your booster:

1. the learning rate against the number of trees: test error curves for $\eta$ from 1 to 0.03.
2. how many bins you actually need.
3. what the second derivative buys: Newton leaves against plain gradient leaves for classification.

## check your understanding

1. in what sense is gradient boosting "gradient descent in function space"?

<details><summary>answer</summary>

the "parameters" are the predictions $F(x_i)$ themselves. the gradient of the total loss with respect to them is the vector $g$. a tree fitted to $-g$ is a step in (approximately) the steepest-descent direction that can be expressed by a small tree, so it generalizes to new $x$; the learning rate is the step size.

</details>

2. why does every node store $-G/(H + \lambda)$, not only the leaves?

<details><summary>answer</summary>

any node can end up a leaf (when its best gain is $\le 0$, or at max depth), and its value is needed then. storing it everywhere costs nothing and keeps the growing loop simple.

</details>

3. what does `min_child_weight` mean for classification, where $h = p(1 - p)$?

<details><summary>answer</summary>

$\sum h$ is small where the model is confident ($p$ near 0 or 1), so a child needs *more* rows to reach the threshold there. it stops the tree from carving out tiny regions the model already classifies confidently, where Newton steps $-G/H$ would be huge and unreliable.

</details>

4. you use 32 bins instead of 256. which part of the algorithm gets faster, and which doesn't?

<details><summary>answer</summary>

the split scan ($O(d\cdot\text{n\_bins})$ per node) and the histogram memory shrink; building histograms is $O(n\cdot d)$ either way, so on large data the saving is modest. binning itself happens once.

</details>

5. why fit the tree on a *subsample* of rows but update the predictions of *all* rows?

<details><summary>answer</summary>

the subsample only randomizes which rows decide the tree's structure and leaf values (the regularization). the model is the sum of all trees, so its predictions at every row must include every tree; the next round's gradients need those predictions.

</details>

6. gradient boosting with $\eta = 0.1$ is at its best after 600 trees. what are two ways to find that number without touching the test set?

<details><summary>answer</summary>

hold out a validation set and pick the argmin of the validation loss over `staged_predict` (early stopping), or cross-validate the number of trees. with subsampling you can also use the out-of-bag rows of each round.

</details>

## common bugs

- **sign of the gradient.** symptom: the training loss goes *up*. $g = F - y$ (prediction minus target); the leaf is $-G/(H + \lambda)$.
- **binning test data with its own edges.** symptom: test predictions that make no sense. compute edges once on the training data and reuse them.
- **forgetting the learning rate in predict.** symptom: training curves look fine, predictions explode. multiply every tree by $\eta$ in `predict` and `staged_predict` too.
- **`nan` gains from empty children.** symptom: `np.argmax` picks garbage (it treats `nan` as the maximum). mask splits whose children fail `min_child_weight` *before* taking the argmax, and silence the division warnings there.
- **sampling with replacement.** symptom: `subsample=1.0` still gives different trees per seed. use `choice(n, size, replace=False)`, and all rows when the fraction is 1.
- **overflow in the sigmoid.** symptom: warnings and `nan` after many trees. use $\frac12(1 + \tanh(F/2))$.

## going deeper

- Friedman (2001), "Greedy function approximation: a gradient boosting machine", *Annals of Statistics* 29(5): 1189–1232. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
- Friedman (2002), "Stochastic gradient boosting", *Computational Statistics & Data Analysis* 38(4): 367–378. [doi:10.1016/S0167-9473(01)00065-2](https://doi.org/10.1016/S0167-9473(01)00065-2)
- Chen & Guestrin (2016), "XGBoost: a scalable tree boosting system", *KDD*. the second-order objective, the gain, weighted quantile sketches. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
- Ke et al. (2017), "LightGBM: a highly efficient gradient boosting decision tree", *Advances in Neural Information Processing Systems* 30. histograms, histogram subtraction, leaf-wise growth.
- Grinsztajn, Oyallon & Varoquaux (2022), "Why do tree-based models still outperform deep learning on typical tabular data?", *NeurIPS Datasets and Benchmarks*. [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, chapter 10 (boosting and additive trees, shrinkage, subsampling). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)

## connections

- ← builds on: trees with explicit stacks (topic 01), AdaBoost as stagewise additive modelling (topic 03), gradient descent and Newton's method (ch 06), log loss and its derivatives (ch 07 · topic 03), quantiles (ch 05).
- → used in: the capstone's tabular benchmark against scikit-learn's `HistGradientBoostingRegressor`; model interpretation (topic 05); early stopping and learning-rate schedules for neural networks (ch 15); boosting as a baseline throughout the tabular parts of the course.
