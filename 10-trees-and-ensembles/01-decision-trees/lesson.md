# decision trees

## why this matters

a bank wants to approve loans from a table: income, debt, age of the account, number of late payments, region. some columns are in dollars, some are counts, one is a category. there are missing patterns, thresholds that matter ("more than two late payments") and interactions ("high debt is fine if income is high"). a linear model needs you to scale every column, encode the categories, and invent the interaction terms yourself. a **decision tree** needs none of that: it asks one yes/no question about one column at a time, and the questions it learns *are* the thresholds and interactions.

trees on their own are mediocre predictors: a single deep tree overfits, and a small change in the data can change it completely. but they're the building block of the models that still win most competitions on tabular data: random forests (topic 02), AdaBoost (topic 03) and gradient boosting (topic 04). every one of those is "many trees, combined cleverly", and all of them inherit the tree's strengths: no scaling, no encoding of monotone transforms, interactions for free, and fast prediction.

in this topic you build CART (classification and regression trees, Breiman et al., 1984), the algorithm behind scikit-learn's trees: greedy splits chosen by impurity, grown until a stopping rule fires, then pruned back by cost-complexity. your implementation will agree with scikit-learn's to the last digit on the checks, except where two candidate splits tie exactly (scikit-learn then picks among them at random; you take the lowest feature).

## warm-up: predict first

1. a node holds 3 cats and 1 dog. how impure is it, on a scale where a pure node is 0 and a 50/50 node is 0.5?
2. you multiply one feature by 1,000, or take its logarithm. how does the fitted tree change?
3. you grow a tree until every leaf is pure, on data where 20% of the labels are random coin flips. what is its test error: about 20%, about 30%, or about 40%?
4. two classes are separated by the diagonal line $x_1 = x_2$. how many leaves does a tree need to get 99% accuracy?
5. you remove 5% of the training rows and refit a fully grown tree. how much of the tree stays the same?

## prerequisites check

- [ ] you can sort, take cumulative sums and use `np.take_along_axis` on 2-D arrays (→ ch 00 · topic 02)
- [ ] you know entropy in bits, $-\sum_kp_k\log_2p_k$ (→ ch 05 · topic 05)
- [ ] you know the estimator protocol: `fit` returns `self`, `classes_`, `predict_proba`, `score` (→ ch 07 · topic 01)
- [ ] you can explain overfitting with a validation curve (→ ch 07 · topic 04)

## core ideas

### idea 1: impurity, the cost of a node

a tree partitions the feature space into boxes (leaves) and predicts a constant in each: the majority class (or the class fractions) for classification, the mean for regression. a good box is one where that constant fits well: all the labels in it agree. **impurity** measures how badly they disagree.

for classification, with class proportions $p_1, \dots, p_k$ in a node, two impurities are standard:

$$\text{Gini}(p) = 1 - \sum_kp_k^2 = \sum_kp_k(1 - p_k),\qquad H(p) = -\sum_kp_k\log_2p_k.$$

Gini impurity is the probability that two labels drawn at random from the node disagree; entropy is the average number of bits needed to encode a label. both are 0 for a pure node, maximal for a uniform one (0.5 and 1 bit for two classes), and **concave** in $p$, which is what makes splitting always help (on paper 1). warm-up 1: $(3, 1)$ gives $1 - (0.75^2 + 0.25^2) = 0.375$.

for regression, the impurity of a node is the (weighted) variance of its targets, the mean squared error of predicting their mean:

$$\text{MSE} = \frac{1}{W}\sum_iw_i(y_i - \bar y)^2 = \frac{\sum_iw_iy_i^2}{W} - \Big(\frac{\sum_iw_iy_i}{W}\Big)^2,\qquad W = \sum_iw_i.$$

notice what each impurity needs: class impurities need only the (weighted) **class counts** of the node; the variance needs only three **sums**, $(\sum w, \sum wy, \sum wy^2)$. these statistics are *additive*: the counts of a node are the counts of its left child plus the counts of its right child. that's the key to the fast split search in idea 2, so your impurity functions take summed statistics, not raw labels.

two numerical details. the variance formula subtracts two nearly equal numbers when the targets are large or constant, so a constant node can come out as $10^{-17}$ instead of 0, and a tree would happily keep "splitting" it. clip anything below $10^{-12}$ times the mean square to 0. and an empty node (total weight 0) has impurity 0, never `nan`.

> **spec.** `gini(counts)` and `entropy(counts)` take weighted class counts of shape `(..., k)` and return impurities of shape `(...,)` (they reduce the last axis only), 0 where the total is 0; entropy is in bits with $0\log0 = 0$. `squared_error(sums)` takes `(..., 3)` sums $(\sum w, \sum wy, \sum wy^2)$ and returns the weighted variance, 0 for an empty node and clipped to 0 when it's at most $10^{-12}\cdot\sum wy^2/\sum w$.

> **▶ build: stage 1.** implement `gini`, `entropy` and `squared_error`.

### idea 2: the best split, all thresholds at once

a split sends the rows with $x_j \le t$ left and the rest right. CART chooses the feature $j$ and threshold $t$ that reduce impurity the most:

$$\Delta(j, t) = W\,I(\text{node}) - W_L\,I(\text{left}) - W_R\,I(\text{right}),$$

where $W, W_L, W_R$ are the summed sample weights. (dividing by $W$ gives the familiar "impurity minus the weighted average of the children's impurities"; keeping the weights un-normalized makes the decreases of different nodes comparable, which pruning and feature importances need.)

which thresholds are worth trying? only the cuts between consecutive *distinct* sorted values of $x_j$: moving $t$ between two neighbouring values doesn't change which rows go left. CART uses the midpoint. so a feature with $n$ distinct values has $n - 1$ candidates, and the obvious algorithm (for each candidate, sum up both children, compute impurities) costs $O(n^2)$ per feature.

the additive statistics make it $O(n\log n)$. sort the rows by $x_j$. the left child of the cut after sorted position $i$ holds rows $0..i$, so its statistics are a **cumulative sum**; the right child's are the total minus that. one `argsort` and one `cumsum` per feature give you the children's statistics for *every* threshold at once, and your stage-1 impurities, which work on any leading axes, score all of them in one call. do it for all $d$ features in the same arrays: sort along axis 0 of an $(n, d)$ matrix and the cumulative statistics have shape $(n - 1, d, m)$.

then mask out the cuts you can't use: between equal values (`xs[i+1] > xs[i]` must hold), or leaving fewer than `min_samples_leaf` rows on either side. take the best of what's left, breaking ties toward the lowest feature and then the lowest threshold (that's what `np.argmax` gives you if you lay the candidates out feature by feature).

> **spec.** `best_split(X, stats, w, impurity, *, min_samples_leaf=1, features=None)`: `X` $(n, d)$, `stats` $(n, m)$ the per-row statistics (weighted one-hot labels, or $(w, wz, wz^2)$), `w` $(n,)$ weights, `impurity` one of your stage-1 functions, `features` an optional list of column indices to search (default: all). returns `(feature, threshold, decrease)` with the threshold the midpoint of two consecutive distinct values (if rounding puts the midpoint on the upper value, use the lower one), or `(-1, nan, -inf)` when no split is valid. no Python loop over thresholds.

> **🔮 predict:** you replace a feature by its logarithm (or multiply it by 1,000). which of the quantities in $\Delta(j, t)$ change? what happens to the threshold?

> **▶ build: stage 2.** implement `best_split`.

### idea 3: growing the tree

CART is greedy: split the root with the best split, then split each child with *its* best split, and so on, until a stopping rule says "leaf". the rules (all hyperparameters):

- the node is at `max_depth` (the root has depth 0);
- it has fewer than `min_samples_split` rows, or fewer than `2 * min_samples_leaf` (no split could satisfy both children);
- it's pure (impurity $\le$ machine epsilon);
- no valid split exists (every feature is constant on it).

greedy means short-sighted. on XOR-like data the first split can look useless (each half is still 50/50) even though two levels of splits would be perfect; CART still takes the best split it can find, however small its gain, and lets the next level sort it out.

a fitted tree is a handful of parallel arrays, one entry per node, node 0 the root: `feature_` and `threshold_` (the question; feature $-1$ at a leaf), `children_left_` and `children_right_` (node indices, $-1$ at a leaf), and `value_` (the summed statistics of the node, from which a leaf's prediction follows: class counts → fractions, $(w, wz, wz^2)$ → mean). prediction is a walk from the root: all rows at once, one level per step, `node = where(x[feature[node]] <= threshold[node], left[node], right[node])`.

build the tree with an **explicit stack** of nodes still to process, not with recursion. trees can be very deep: on alternating labels along a line, every split can only peel off a single row, and the tree is $n - 1$ levels deep. Python's recursion limit is 1,000, so a recursive builder crashes on 1,500 rows. with a stack (push the right child, then the left), nodes are numbered depth-first, every parent before its children, exactly like scikit-learn.

**random feature subsets.** with `max_features` set (an int, a fraction, `"sqrt"` or `"log2"`), each node only searches a random subset of the features, drawn with the tree's generator. that makes a tree worse on its own and is pointless here, but it's what makes random forests work (topic 02). if none of the drawn features can split a node, search the remaining ones before giving up.

> **spec.** `BaseDecisionTree._grow(X, stats, w, impurity)` fills the node arrays above plus `impurity_`, `n_node_samples_`, `weighted_n_node_samples_` and `n_features_in_`, and returns `self`; `apply(X)` returns each row's leaf index $(n,)$; `get_depth()`, `get_n_leaves()`. `DecisionTreeClassifier.fit(X, y, sample_weight=None)` sets `classes_`, builds `stats` = weights × one-hot labels, grows the tree with `gini` or `entropy` (the `criterion`), prunes if `ccp_alpha > 0` (stage 5) and returns `self`; `predict_proba` returns each row's leaf class fractions. `predict` and `score` come from your `Classifier` base.

> **▶ build: stage 3.** implement `_grow`, `apply`, `get_depth`, `get_n_leaves`, and `DecisionTreeClassifier.fit` and `predict_proba`.

### idea 4: regression trees

nothing about the growing changes for regression: only the statistics and the leaf value. with `stats` $= (w, wz, wz^2)$ and the `squared_error` impurity, each split maximally reduces the weighted sum of squared errors, and each leaf predicts the weighted mean of its targets. the result is a **piecewise-constant** function: a staircase with as many steps as leaves, axis-aligned in every dimension.

one detail matters numerically: build the statistics from $z = y - \bar y_w$ (the targets minus their weighted mean), not from $y$, and add $\bar y_w$ back in `predict`. if the targets are house prices around $10^6$, $\sum wy^2$ is around $10^{12}\cdot n$ and the variance you want, maybe $10^4$, drowns in its rounding error. centering costs nothing: the variance doesn't change when you shift $y$.

> **spec.** `DecisionTreeRegressor.fit(X, y, sample_weight=None)` stores `y_offset_` $= \bar y_w$, grows on $(w, wz, wz^2)$ with `squared_error`, prunes if `ccp_alpha > 0`, returns `self`; `predict(X)` = `y_offset_` + the leaf's $\sum wz / \sum w$.

> **▶ build: stage 4.** implement `DecisionTreeRegressor.fit` and `predict`.

### idea 5: pruning and importances

a fully grown tree memorizes: every leaf is pure, so with noisy labels it fits the noise. you could stop early (`max_depth`, `min_samples_leaf`), but early stopping is short-sighted in the same way as greedy splitting: a split that looks useless may enable great splits below it. CART's answer is to grow the full tree and then **prune** it back.

**minimal cost-complexity pruning** trades training error against size. with $R(t) = \frac{W_t}{W_\text{root}}I(t)$ the weighted impurity of node $t$ and $T$ a tree with leaves $\tilde T$, minimize

$$R_\alpha(T) = \sum_{\ell\in\tilde T}R(\ell) + \alpha\,|\tilde T|.$$

for each internal node $t$, compare keeping its subtree $T_t$ with collapsing it into a leaf. collapsing changes $R_\alpha$ by $R(t) - R(T_t) - \alpha(|\tilde T_t| - 1)$, so it pays off once $\alpha$ reaches the node's **effective alpha**

$$\alpha_\text{eff}(t) = \frac{R(t) - R(T_t)}{|\tilde T_t| - 1},$$

the training-impurity cost per leaf removed. the **weakest link** is the node with the smallest $\alpha_\text{eff}$. pruning it changes $R(T_a)$ and $|\tilde T_a|$ of every ancestor $a$, so recompute and repeat: collapse the weakest link while its $\alpha_\text{eff} \le$ `ccp_alpha`. Breiman et al. showed this sequence of nested subtrees contains the $R_\alpha$-optimal subtree for every $\alpha$; you pick `ccp_alpha` by cross-validation.

**impurity importance** answers "which features does the tree use?": each internal node adds its decrease $W_tI(t) - W_LI(L) - W_RI(R)$ to its feature's total, and the totals are normalized to sum to 1. it's free (computed from the fitted tree) and it's what most libraries report. it is also biased, in a way topic 05 will show you.

> **spec.** `_prune(alpha)` collapses weakest links as above (ties: the lowest node index), removes the nodes below each collapsed node and renumbers the survivors in their original order, and returns `self`. `feature_importances_` is a property returning the normalized importances $(d,)$, all zeros for a single-leaf tree.

> **🔮 predict:** as `ccp_alpha` grows from 0, does the number of leaves fall smoothly, or in jumps? and does the test accuracy of the pruned tree go up, down, or up then down?

> **▶ build: stage 5.** implement `_prune` and `feature_importances_`.

## worked example

six rows, two features, two classes:

| row | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| 1 | 1 | 5 | 0 |
| 2 | 2 | 6 | 0 |
| 3 | 3 | 1 | 1 |
| 4 | 4 | 4 | 0 |
| 5 | 5 | 2 | 1 |
| 6 | 6 | 3 | 1 |

all weights 1, Gini impurity. the root has counts $(3, 3)$: $I = 0.5$, $W\,I = 3$.

**feature $x_1$** (rows already sorted): the labels in order are $0, 0, 1, 0, 1, 1$. the cumulative class counts of the left child after each position are $(1,0), (2,0), (2,1), (3,1), (3,2)$, and the right child gets the rest:

| threshold | left counts | right counts | $W_LI_L$ | $W_RI_R$ | $\Delta$ |
|---|---|---|---|---|---|
| 1.5 | (1, 0) | (2, 3) | 0 | $5\cdot0.48 = 2.4$ | 0.6 |
| 2.5 | (2, 0) | (1, 3) | 0 | $4\cdot0.375 = 1.5$ | **1.5** |
| 3.5 | (2, 1) | (1, 2) | $3\cdot\frac49$ | $3\cdot\frac49$ | 0.333 |
| 4.5 | (3, 1) | (0, 2) | 1.5 | 0 | **1.5** |
| 5.5 | (3, 2) | (0, 1) | 2.4 | 0 | 0.6 |

two thresholds tie at 1.5; the tie goes to the lower one, 2.5.

**feature $x_2$**: sorted by $x_2$ the rows are 3, 5, 6, 4, 1, 2, with labels $1, 1, 1, 0, 0, 0$. the cut at $x_2 \le 3.5$ separates the classes perfectly: $\Delta = 3 - 0 - 0 = 3$, the largest possible.

so the root splits on $x_2$ at 3.5, both children are pure, and the tree is a single split with two leaves. `best_split` must return `(1, 3.5, 3.0)` on all six rows and `(0, 2.5, 1.5)` when given only $x_1$.

## on paper

1. **splitting never hurts.** show that for a concave impurity $I$, $\Delta \ge 0$ for any split. when is it exactly 0?

<details><summary>solution</summary>

the parent's proportions are the weighted average of the children's: $p = \frac{W_L}{W}p_L + \frac{W_R}{W}p_R$. concavity (Jensen) gives $I(p) \ge \frac{W_L}{W}I(p_L) + \frac{W_R}{W}I(p_R)$; multiply by $W$. for the strictly concave Gini and entropy it's 0 only when $p_L = p_R = p$: both children have exactly the parent's class mix. (misclassification error $1 - \max_kp_k$ is concave but not strictly, which is why it's a poor splitting criterion: many useful splits have zero decrease.)

</details>

2. **the regression leaf.** show that the constant $c$ minimizing $\sum_iw_i(y_i - c)^2$ is the weighted mean, and that a split's decrease in weighted SSE is $\frac{W_LW_R}{W}(\bar y_L - \bar y_R)^2$.

<details><summary>solution</summary>

the derivative $-2\sum w_i(y_i - c) = 0$ gives $c = \sum w_iy_i / W$. for the decrease, write $SSE = \sum wy^2 - W\bar y^2$ for each node. the $\sum wy^2$ terms cancel (they're additive), leaving $W_L\bar y_L^2 + W_R\bar y_R^2 - W\bar y^2$ with $\bar y = (W_L\bar y_L + W_R\bar y_R)/W$. expanding: $\frac{W_LW_R}{W}(\bar y_L - \bar y_R)^2$. a split is valuable when it separates the means and both sides are big.

</details>

3. **monotone transforms.** show that replacing $x_j$ by $g(x_j)$ for a strictly increasing $g$ leaves every split's partition (and so the fitted tree's predictions on the training data) unchanged.

<details><summary>solution</summary>

$g$ preserves the order of the values, so sorting by $g(x_j)$ gives the same order, the same cumulative statistics and the same decreases at the same positions. only the threshold's *value* changes: the midpoint of $g(a)$ and $g(b)$ instead of $g$ of the midpoint of $a$ and $b$. so new points between two training values may go a different way, but nothing else changes. this is why trees need no scaling (warm-up 2).

</details>

4. **effective alpha.** a node $t$ has $W_t/W_\text{root} = 0.2$ and Gini 0.3. its subtree has three leaves with weight fractions 0.1, 0.05, 0.05 and Ginis 0.1, 0.2, 0.4. compute $\alpha_\text{eff}(t)$.

<details><summary>solution</summary>

$R(t) = 0.2\cdot0.3 = 0.06$; $R(T_t) = 0.1\cdot0.1 + 0.05\cdot0.2 + 0.05\cdot0.4 = 0.04$. $\alpha_\text{eff} = (0.06 - 0.04)/(3 - 1) = 0.01$. for `ccp_alpha` $\ge 0.01$ (if it's the weakest link) the subtree collapses into one leaf.

</details>

## experiment

`lab.py` investigates three things with your trees:

1. how training and test accuracy move as `max_depth` grows, on data with noisy labels, and what pruning buys.
2. how stable a fully grown tree is when you refit it on slightly different data.
3. how many leaves a tree needs for a boundary that isn't parallel to an axis.

## check your understanding

1. why does CART only try thresholds between consecutive distinct values?

<details><summary>answer</summary>

any threshold between two neighbouring values sends the same rows left, so the impurity decrease is the same; trying one per gap covers everything. the midpoint is a convention that hedges between the two training values for new points.

</details>

2. why are the impurity functions written to take summed statistics instead of labels?

<details><summary>answer</summary>

because the statistics are additive: the left child of every cut is a prefix of the sorted rows, so its statistics are a cumulative sum, and all thresholds can be scored in one vectorized call. with labels you'd recompute each child from scratch.

</details>

3. a fully grown tree gets 100% training accuracy. is that evidence of anything?

<details><summary>answer</summary>

no: it's guaranteed (on distinct inputs) whatever the labels are, even pure noise. only held-out data says anything about the tree (ch 07 · topic 01's memorizer, in tree form).

</details>

4. why grow a full tree and prune it, instead of stopping as soon as no split improves impurity by much?

<details><summary>answer</summary>

the value of a split can be in the splits it enables. on XOR, the first split barely helps but makes two perfect splits possible. early stopping can't see that; pruning evaluates whole subtrees after the fact.

</details>

5. why does a tree need no feature scaling, while $k$-NN does?

<details><summary>answer</summary>

a tree only compares one feature with a threshold at a time; the order of the values is all it uses (on paper 3). $k$-NN adds up differences across features, so their units matter.

</details>

6. Gini and entropy almost always choose the same split. why might you still prefer Gini?

<details><summary>answer</summary>

it's cheaper (no logarithms), and it's what most libraries default to. both are strictly concave and agree near the extremes; they differ slightly in how much they reward pure children. the choice rarely matters as much as depth or pruning.

</details>

7. a regression tree is fitted to $y = x$ on $[0, 1]$ with 4 leaves. what does it predict at $x = 2$?

<details><summary>answer</summary>

the mean of the rightmost leaf, about 0.875. trees can't extrapolate: outside the training range, the prediction is the constant of the outermost box.

</details>

## common bugs

- **cutting between equal values.** symptom: a split that doesn't separate anything, or thresholds equal to a data value. mask positions where `xs[i+1] == xs[i]`.
- **off-by-one cumulative sums.** symptom: decreases that disagree with a brute-force loop by one row. the left child after sorted position $i$ holds rows $0..i$: `cumsum(...)[:-1]`, with the last (empty-right) cut dropped.
- **recursion.** symptom: `RecursionError` on long, thin trees. use an explicit stack.
- **impurity `nan` on empty or pure nodes.** symptom: `nan` decreases, or `argmax` picking garbage. guard the division by the total and the $\log 0$.
- **a node that never becomes a leaf.** symptom: the fit never finishes. every node popped from the stack must either become a leaf or push two strictly smaller children.
- **raw $y$ in the variance sums.** symptom: different trees after adding a constant to $y$. center first.

## going deeper

- Breiman, Friedman, Olshen & Stone (1984), *Classification and Regression Trees*, Wadsworth (now CRC Press). CART, cost-complexity pruning and the weakest-link sequence. [doi:10.1201/9781315139470](https://doi.org/10.1201/9781315139470)
- Quinlan (1986), "Induction of decision trees", *Machine Learning* 1: 81–106. ID3 and information gain. [doi:10.1007/BF00116251](https://doi.org/10.1007/BF00116251)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, §9.2 (tree-based methods, including why trees are unstable). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Louppe (2014), "Understanding random forests: from theory to practice", PhD thesis, University of Liège, chapter 3 (an implementation-level account of CART, including the sorting and cumulative-statistics trick). [arXiv:1407.7502](https://arxiv.org/abs/1407.7502)

## connections

- ← builds on: the estimator protocol and `check_X_y` (ch 07 · topic 01), validation curves and overfitting (ch 07 · topic 04), entropy (ch 05 · topic 05), vectorized sorting and cumulative sums (ch 00 · topic 02), `as_rng` (ch 00 · topic 05).
- → used in: random forests, which average randomized versions of your trees (topic 02); AdaBoost, which reweights the rows your trees see (topic 03); gradient boosting, which grows second-order trees on binned features (topic 04); permutation importance and the bias of impurity importance (topic 05); isolation forests (ch 11 · topic 06).
