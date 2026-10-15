# k-nearest neighbors

## why this matters

every model so far had a fixed form chosen before seeing data: a line, a polynomial of some degree, a stump. pick the wrong form and no amount of data fixes it: a straight line fitted to a sine wave stays wrong at a million points. **nonparametric** methods drop the fixed form. their complexity grows with the data, so with enough of it they can approximate almost any function.

the simplest of them has no training step at all. to predict at a new point, find the training points closest to it and let them vote (classification) or average their targets (regression). that's $k$-nearest neighbours. it's often a strong baseline, it's how retrieval systems and recommendation engines answer "what's similar to this?", and a famous result (Cover & Hart, 1967) says that with enough data, 1-NN makes at most twice the error of the best possible classifier.

it has two problems, and this topic is about both. the first is cost: a naive prediction compares the query with every training point. a **KD-tree** organizes the points so a query can skip most of them, at least in low dimension. the second is that "closest" stops meaning much as the dimension grows: in 1,000 dimensions, random points are all at almost the same distance from each other. that's the **curse of dimensionality**, and it's why the KD-tree's trick stops working in high dimension too.

## warm-up: predict first

1. you scale one feature by 1,000 (metres to millimetres). does the 1-NN prediction change?
2. what's the training accuracy of a 1-NN classifier? what about 15-NN?
3. data is uniform in the 10-dimensional unit cube. to capture 1% of it around a query with a small cube, how long must the cube's edges be?
4. 1,000 points uniform in $[0, 1]^{1000}$, and a query point. roughly how much farther is the farthest point than the nearest one?
5. a KD-tree on 1 million points in 2-D answers a nearest-neighbour query. about how many distances does it compute? and in 50-D?

## prerequisites check

- [ ] you can compute all pairwise distances without loops, and know why $\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2x^\top y$ helps (→ ch 00 · topic 02)
- [ ] you've written a `Classifier` (with `classes_` and `predict_proba`) and a `Regressor` on the `mlfs.base` protocol (→ ch 07 · topic 01)
- [ ] you know what feature scaling does to distances (→ ch 07 · topic 05)
- [ ] you can reason about bias and variance as model complexity changes (→ ch 07 · topic 04)
- [ ] you know the law of large numbers: an average of many independent terms concentrates around its mean (→ ch 04 · topic 05)

## core ideas

### idea 1: nearness needs a metric

"nearest" depends on how you measure distance. the family you'll use most is the **Minkowski** distance,

$$d_p(x, y) = \Big(\sum_{j=1}^{d} |x_j - y_j|^p\Big)^{1/p}, \qquad p \ge 1,$$

with $p = 2$ the Euclidean distance, $p = 1$ the Manhattan distance (sum of absolute differences), and the limit $p \to \infty$ the largest single difference, $\max_j |x_j - y_j|$. for $p \ge 1$ these are **metrics**: $d(x, y) \ge 0$ with equality only when $x = y$, $d(x, y) = d(y, x)$, and the triangle inequality $d(x, z) \le d(x, y) + d(y, z)$. (for $p < 1$ the triangle inequality fails.) bounds like this are what let a search rule out far-away regions without looking inside them, as you'll see in idea 5.

a metric treats every feature the same, so it's at the mercy of their units (warm-up 1). measure one feature in millimetres and the others in metres, and the distance is effectively that feature alone. you saw this in chapter 07: standardize first, inside a pipeline.

the **brute-force** search computes the distance from a query to every training point and keeps the $k$ smallest: $O(nd)$ per query, $O(mnd)$ for $m$ queries. it's exact and, vectorized, fast for a few thousand points: one call to `pairwise_distances` and one sort per row. squared distances have the same order as distances, so sort those and take the square root only of the $k$ you return.

> **▶ build: stage 1.** implement `minkowski_distances(X, Y, p)` (broadcast to an $(n, m, d)$ array of differences; `p = np.inf` is the max) and `brute_kneighbors(X_train, X_query, k)`: squared distances from `pairwise_distances` (yours, chapter 00), a *stable* `argsort` per row so ties go to the smaller index, the first $k$ columns, and their square roots. return `(dist, idx)`, both $(m, k)$, nearest first. $k$ outside $1..n$ raises `ValueError`.

### idea 2: predict by local vote

given the $k$ nearest training points $N_k(x)$, the classifier estimates

$$\hat p(c \mid x) = \frac{\sum_{i \in N_k(x)} w_i\, [y_i = c]}{\sum_{i \in N_k(x)} w_i},$$

and the regressor predicts the weighted mean $\hat f(x) = \sum_{i \in N_k(x)} w_i y_i / \sum_i w_i$. with **uniform** weights $w_i = 1$ these are vote shares and plain averages. with **distance** weights $w_i = 1/d(x, x_i)$, closer neighbours count more. if the query coincides with a training point, $1/0$ is infinite; the convention (scikit-learn's) is to give each exact match weight 1 and every other neighbour weight 0.

$k$ is the complexity knob, running in the *opposite* direction from a polynomial's degree. $k = 1$ follows every training label: zero training error (warm-up 2, as long as no two identical points disagree), a jagged boundary, high variance. $k = n$ predicts the global majority or mean everywhere: maximal bias, no variance. in between there's a sweet spot, chosen by cross-validation like any hyperparameter.

why should this work at all? if the true conditional $p(c \mid x)$ is smooth, points near $x$ have labels drawn from nearly the same distribution as $x$'s own label. as $n \to \infty$ with $k \to \infty$ and $k/n \to 0$, the neighbourhood shrinks around $x$ while the number of votes grows, so $\hat p(c \mid x) \to p(c \mid x)$: $k$-NN is **consistent**, it converges to the Bayes-optimal classifier for any distribution (Stone, 1977). even 1-NN, with its fixed single vote, has asymptotic error at most $2R^*(1 - R^*)$ for two classes, where $R^*$ is the Bayes error (Cover & Hart, 1967). the catch is "as $n \to \infty$", and idea 3 is about how fast $n$ must grow.

the model has no parameters to learn: `fit` just stores the data. everything happens in `predict`. that makes it a **lazy** (or memory-based) learner: cheap to train, expensive to query, and its size grows with the data.

> **▶ build: stage 2.** implement `neighbor_weights(dist, weights)`, then `KNeighborsClassifier` and `KNeighborsRegressor` (both subclass your chapter 07 bases). `fit` runs `check_X_y`, stores `X_` (and `classes_` plus `y_idx_` for the classifier, from `np.unique(..., return_inverse=True)`, or a float `y_` for the regressor), validates `weights`, `algorithm` and `n_neighbors`, and builds `tree_ = KDTree(X_)` when `algorithm="kd_tree"` (you'll write `KDTree` in stage 4; the checks use `"brute"` until stage 5). `kneighbors(X, k=None)` dispatches to `brute_kneighbors` or `tree_.query` and raises `ValueError` if `X` has the wrong number of features. the classifier gets `predict` from `Classifier`.

> **🔮 predict:** on 2-D data with overlapping classes, sketch the decision boundary of 1-NN and of 50-NN. which one has the lower training error? which one would you bet on for new points?

### idea 3: the curse of dimensionality

nearest-neighbour methods assume that a query has neighbours that are actually *near*. in high dimension, that assumption fails in two ways.

**neighbourhoods aren't local.** take $n$ points uniform in the unit cube $[0, 1]^d$. a sub-cube that captures a fraction $r$ of them must have volume $r$, so its edge is

$$e_d(r) = r^{1/d}.$$

in 1-D, 1% of the data lives in an interval of length 0.01. in 10-D, a cube holding 1% of the data has edges of length $0.01^{1/10} \approx 0.63$ (warm-up 3): it spans almost two thirds of every feature's range. the "local" average is an average over most of the space, so the bias that idea 2 hoped would vanish doesn't. to keep the edge at 0.1 in 10-D you'd need a fraction $0.1^{10}$, i.e. $10^{10}$ points per neighbour.

**distances concentrate.** the squared distance between two random points is a sum of $d$ independent terms, $\|x - y\|^2 = \sum_j (x_j - y_j)^2$. by the law of large numbers it grows like $d$ times the mean of one term, while its standard deviation grows only like $\sqrt d$. so relative to their size, all distances become alike. measure it with the **relative contrast** from a query $q$ to the data,

$$\mathrm{RC}(q) = \frac{\max_i \|x_i - q\| - \min_i \|x_i - q\|}{\min_i \|x_i - q\|},$$

which shrinks roughly like $1/\sqrt d$ for data like this (Beyer et al., 1999; Aggarwal et al., 2001). when RC is 0.05, the nearest neighbour is only 5% closer than the *farthest* point (warm-up 4): "nearest" is close to arbitrary.

the saving grace is that real data is rarely uniform in all its dimensions. images, sounds and text have far fewer degrees of freedom than pixels or words: they lie near a low-dimensional surface inside the big space, and what matters for neighbour methods is that **intrinsic** dimension, not the number of columns. the lab measures both on real images.

> **▶ build: stage 3.** implement `relative_contrast(X, q)` (Euclidean distances from `q` to every row) and `edge_length(fraction, d)` (elementwise, so arrays of `d` work).

> **🔮 predict:** MNIST images have 784 pixels. will the relative contrast of a real digit to the other digits look like that of a uniform point in 784 dimensions, or like something much lower-dimensional?

### idea 4: a KD-tree splits space in half, recursively

brute force looks at every point because it knows nothing about where the points are. a **KD-tree** (Bentley, 1975) is a binary tree of nested boxes. the root holds all the points. a node with more than `leaf_size` points picks a dimension, splits at the median value along it, and passes the lower half to its left child and the upper half to its right child. that continues until every leaf has at most `leaf_size` points.

two choices define the variant you'll build:

- **which dimension?** the one where the node's points are most spread out (largest $\max - \min$), so the boxes get split along their longest side and stay roughly cube-shaped.
- **where?** at the median, by *position*: sort the points along that dimension and put the first $\lfloor m/2 \rfloor$ on the left and the rest on the right, with the split value $s$ equal to the value at position $\lfloor m/2 \rfloor$. every left point has $x_{\text{dim}} \le s$ and every right point $x_{\text{dim}} \ge s$, the halves differ in size by at most one, and the depth is $\lceil \log_2(n/\text{leaf size}) \rceil$. splitting by position, not by value, matters when values repeat: a node of identical points still halves, so the recursion ends.

building costs $O(n \log^2 n)$ with a sort at each level (a linear-time median selection brings it to $O(n \log n)$). it's done once, in `fit`.

> **▶ build: stage 4.** implement `KDTree.build(idx)`. it appends a node dict `{"idx", "dim", "split", "left", "right"}` for the rows `idx` to `self.nodes` *before* building its children (so the root is node 0), makes it a leaf (`dim`, `split`, `left`, `right` all `None`) when `len(idx) <= leaf_size`, and otherwise fills in the widest dimension, the median split by position (a stable `argsort`), and the ids returned by building the two halves. it returns the node's id. `__init__` is given and calls `self.build(np.arange(n))`.

### idea 5: search with pruning

to find the nearest neighbour of $q$, descend first into the child on $q$'s side of the split (the **near** child), because that's where close points probably are. after that subtree is done you have a current best distance $r$. now the far child: every point in it lies on the other side of the hyperplane $x_{\text{dim}} = s$, so its distance to $q$ is at least the distance from $q$ to that plane, $|q_{\text{dim}} - s|$. if $|q_{\text{dim}} - s| \ge r$, nothing in the far child can beat what you have, and you skip it entirely. that's **pruning**. the bound is a coordinate projection (a distance is at least its gap along any one axis); ball trees and other metric trees use the triangle inequality for the same purpose.

for $k$ neighbours, keep the $k$ best found so far in a max-heap keyed by distance, so the current $k$-th best (the one to beat) is at the top. Python's `heapq` is a min-heap, so store `(-squared distance, row)`: then `heap[0]` is the worst of the current best $k$. a new candidate goes in while the heap has fewer than $k$ entries, and otherwise replaces the top if it's closer. the far child is skipped only when the heap is full *and* $(q_{\text{dim}} - s)^2 \ge$ the top's squared distance.

in low dimension the pruning is spectacular: for $n$ points in 2-D a query touches a few leaves, $O(\log n)$ work instead of $O(n)$ (warm-up 5). but the hyperplane bound is weak in high dimension: the distance to a single coordinate plane is a small part of a distance summed over $d$ coordinates, and by idea 3 the current best $r$ is barely smaller than the distance to everything else. the rule of thumb is that a KD-tree helps only while $n \gg 2^d$; beyond that it visits most leaves and is slower than brute force because of the bookkeeping. high-dimensional search uses *approximate* methods (locality-sensitive hashing, graph-based indexes) that trade exactness for speed.

> **▶ build: stage 5.** back in the `KDTree` class, implement `search(node_id, q, k, heap)` (at a leaf: compute the squared distances to all its rows at once, add `len(rows)` to `self.n_distances_`, and push or replace on the heap; at an internal node: near child, then far child unless it can be pruned) and `query(X, k=1)`: reset `n_distances_` to 0, run `search(0, q, k, heap)` for each query row with a fresh heap, and turn each heap into sorted `(dist, idx)` rows. then `algorithm="kd_tree"` in your models works.

## worked example

five training points in 2-D with labels:

| point | $x$ | label | distance to $q = (1, 0.5)$ |
|---|---|---|---|
| B | $(1, 0)$ | 0 | $0.5$ |
| A | $(0, 0)$ | 0 | $\sqrt{1.25} \approx 1.118$ |
| C | $(0, 2)$ | 1 | $\sqrt{3.25} \approx 1.803$ |
| D | $(3, 0)$ | 1 | $\sqrt{4.25} \approx 2.062$ |
| E | $(2, 3)$ | 1 | $\sqrt{7.25} \approx 2.693$ |

(listed nearest first; the training order is A, B, C, D, E.)

- **3-NN, uniform:** the neighbours are B, A, C with labels 0, 0, 1, so $\hat p = (2/3, 1/3)$ and the prediction is 0.
- **5-NN, uniform:** all five vote, 2 against 3: $\hat p = (0.4, 0.6)$, prediction 1. adding more distant neighbours flipped the answer.
- **5-NN, distance-weighted:** the weights $1/d$ are $2, 0.894, 0.555, 0.485, 0.371$. class 0 gets $2 + 0.894 = 2.894$, class 1 gets $0.555 + 0.485 + 0.371 = 1.411$, so $\hat p_0 = 2.894/4.306 \approx 0.672$ and the prediction is 0 again: the close points dominate.

the tests check exactly these numbers with your classifier.

## on paper

1. **the edge of a neighbourhood.** with uniform data in $[0, 1]^d$, how large must $n$ be so that the $k = 10$ nearest neighbours of a query fit in a cube of edge 0.1 (on average)? evaluate for $d = 1, 2, 10$.

<details><summary>solution</summary>

a cube of edge 0.1 has volume $0.1^d$ and holds about $n \cdot 0.1^d$ points. you need $n \cdot 0.1^d \ge 10$, so $n \ge 10^{d+1}$: 100 points in 1-D, 1,000 in 2-D, $10^{11}$ in 10-D. the data you need to stay local grows exponentially with the dimension.

</details>

2. **why distances concentrate.** let $x, y$ be independent and uniform on $[0, 1]^d$. compute $\mathbb E\|x - y\|^2$ and $\mathrm{Var}\,\|x - y\|^2$, and show that the ratio of the standard deviation to the mean goes to 0 like $1/\sqrt d$.

<details><summary>solution</summary>

for one coordinate, $u = x_j - y_j$ has $\mathbb E u^2 = \mathrm{Var}(x_j) + \mathrm{Var}(y_j) = 2/12 = 1/6$ and $\mathbb E u^4 = 1/15$ (integrate the triangular density of $u$ on $[-1, 1]$: $\int_{-1}^1 u^4 (1 - |u|)\,du = 2(1/5 - 1/6) = 1/15$), so $\mathrm{Var}(u^2) = 1/15 - 1/36 = 7/180$. the $d$ coordinates are independent, so $\mathbb E\|x - y\|^2 = d/6$ and $\mathrm{Var}\,\|x - y\|^2 = 7d/180$. the ratio is $\sqrt{7d/180}/(d/6) = \sqrt{7/5}/\sqrt d \approx 1.18/\sqrt d$. at $d = 1000$ the squared distances fluctuate by about 4% of their size.

</details>

3. **the pruning bound.** show that if $q_{\text{dim}} < s$ and every point $x$ in the right child has $x_{\text{dim}} \ge s$, then $\|x - q\| \ge s - q_{\text{dim}}$. why does that make the skip rule exact (it never skips the true neighbour)?

<details><summary>solution</summary>

$\|x - q\|^2 = \sum_j (x_j - q_j)^2 \ge (x_{\text{dim}} - q_{\text{dim}})^2 \ge (s - q_{\text{dim}})^2$, since $x_{\text{dim}} - q_{\text{dim}} \ge s - q_{\text{dim}} > 0$. if $(s - q_{\text{dim}})^2 \ge r^2$, where $r$ is the current $k$-th best distance, every point in the right child is at least as far as the $k$-th best, so none of them could enter the top $k$ (ties aside). skipping it can't lose a neighbour; it only saves work.

</details>

4. **the 1-NN error, asymptotically.** two classes with $p(1 \mid x) = \eta(x)$. as $n \to \infty$, the nearest neighbour of $x$ is so close that its label is an independent draw with the same probability $\eta(x)$. show that the 1-NN error at $x$ tends to $2\eta(1 - \eta)$, and that this is at most $2 R^*(x)$ where $R^*(x) = \min(\eta, 1 - \eta)$ is the Bayes error at $x$.

<details><summary>solution</summary>

1-NN errs when the query's label and its neighbour's label differ: $P(\text{differ}) = \eta(1 - \eta) + (1 - \eta)\eta = 2\eta(1 - \eta)$. with $R^* = \min(\eta, 1 - \eta)$, $2\eta(1 - \eta) = 2R^*(1 - R^*) \le 2R^*$. averaging over $x$ (and using Jensen's inequality for the tighter form) gives Cover & Hart's bound $R^* \le R_{1\text{NN}} \le 2R^*(1 - R^*)$.

</details>

## experiment

`lab.py` runs your code on synthetic data and on real handwritten digits:

1. how the relative contrast changes with the dimension for uniform data, and what it is for real 784-pixel MNIST digits.
2. what fraction of the data your KD-tree actually looks at, as the dimension grows from 1 to 20.
3. $k$-NN on MNIST: accuracy as a function of $k$ and the weighting, with no training at all.

## check your understanding

1. why does $k$-NN need feature scaling, while a decision stump (chapter 07) doesn't?

<details><summary>answer</summary>

$k$-NN compares points through a distance that adds up all features, so a feature with a large range dominates. a stump looks at one feature at a time and only at the *order* of its values; any increasing rescaling leaves its splits (and predictions) unchanged.

</details>

2. what happens to the bias and the variance of $k$-NN regression as $k$ grows?

<details><summary>answer</summary>

the variance falls (an average of $k$ noisy targets has variance $\sigma^2/k$) and the bias rises (the neighbours spread farther from $x$, where the true function differs). $k = 1$ is the low-bias, high-variance end; $k = n$ is the constant predictor.

</details>

3. your 1-NN classifier has 100% training accuracy. a colleague says "it's overfitting". is that a useful diagnosis?

<details><summary>answer</summary>

not on its own: 1-NN has zero training error by construction (barring duplicates with different labels), whether it generalizes well or not. the training error carries no information about it. look at cross-validated error, and compare with larger $k$.

</details>

4. distance weighting with $k = n$: is it still the constant predictor?

<details><summary>answer</summary>

no. every point votes, but with weight $1/d$, so the prediction varies with $x$ and leans towards nearby points. it's a smooth interpolator (inverse-distance weighting, used in geostatistics). it can still be pulled by the mass of far-away points, since $1/d$ decays slowly.

</details>

5. why does the KD-tree visit the near child *first*?

<details><summary>answer</summary>

so that the current best distance $r$ becomes small early. the far-child test compares $|q_{\text{dim}} - s|$ with $r$; a small $r$ prunes more. visiting the far side first would still be exact, just slower.

</details>

6. MNIST has 784 features, yet $k$-NN works well on it. doesn't that contradict the curse of dimensionality?

<details><summary>answer</summary>

the curse is about the *intrinsic* dimension of the data. digit images occupy a thin, curved region of the 784-dimensional space (most pixel combinations are not digits), so a digit's nearest neighbours are genuinely closer than the rest. the lab measures this with the relative contrast.

</details>

7. what would make the 1-NN prediction at a training point differ from its own label?

<details><summary>answer</summary>

a duplicate: another training point at distance 0 with a different label. with ties broken by index, the lower index wins. with distance weights, all exact matches vote equally.

</details>

## common bugs

- **unscaled features.** symptom: accuracy near the majority-class rate, and the neighbours of every point agree in one feature and differ wildly in the others. standardize inside a pipeline.
- **sorting instead of selecting, with no stable order.** `np.argpartition` returns the $k$ smallest in arbitrary order. symptom: `dist` isn't ascending, or ties come out in a different order than the oracle. sort the selected $k$ (or the whole row stably).
- **returning squared distances.** symptom: distance weights that are the square of the expected ones, and `dist` that fails a comparison with scikit-learn. take the square root at the end.
- **dividing by zero in distance weights.** symptom: `nan` probabilities for queries that coincide with training points. handle rows with exact matches separately.
- **pruning with the wrong radius.** comparing $|q_{\text{dim}} - s|$ (a distance) with a *squared* distance, or pruning while the heap holds fewer than $k$ entries. symptom: the tree's neighbours occasionally disagree with brute force.
- **splitting by value.** `X[:, dim] < median` puts every copy of a repeated value on one side; with many duplicates a node never shrinks. symptom: `RecursionError`. split by position in the sorted order.

## going deeper

- Cover & Hart (1967), "Nearest neighbor pattern classification", *IEEE Transactions on Information Theory* 13(1): 21–27. the factor-of-two bound. [doi:10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)
- Stone (1977), "Consistent nonparametric regression", *Annals of Statistics* 5(4): 595–620. universal consistency of $k$-NN with $k \to \infty$, $k/n \to 0$. [doi:10.1214/aos/1176343886](https://doi.org/10.1214/aos/1176343886)
- Bentley (1975), "Multidimensional binary search trees used for associative searching", *Communications of the ACM* 18(9): 509–517. the KD-tree. [doi:10.1145/361002.361007](https://doi.org/10.1145/361002.361007)
- Friedman, Bentley & Finkel (1977), "An algorithm for finding best matches in logarithmic expected time", *ACM Transactions on Mathematical Software* 3(3): 209–226. the search with pruning, and its analysis. [doi:10.1145/355744.355745](https://doi.org/10.1145/355744.355745)
- Beyer, Goldstein, Ramakrishnan & Shaft (1999), "When is 'nearest neighbor' meaningful?", *ICDT*. distance concentration. [doi:10.1007/3-540-49257-7_15](https://doi.org/10.1007/3-540-49257-7_15)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., sections 2.3 and 2.5 (nearest neighbours, local methods in high dimensions) and chapter 13. [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)

## connections

- ← builds on: vectorized pairwise distances (ch 00), the `Classifier`/`Regressor` protocol and feature scaling (ch 07), bias–variance (ch 07 · topic 04), the law of large numbers (ch 04).
- → used in: kernel density estimation, which replaces the hard "$k$ nearest" with a smooth kernel (topic 02); kernels as similarity functions (topic 03); the $k$-NN probe for evaluating learned representations (ch 25); retrieval for language models (ch 27); Isomap and $t$-SNE neighbourhood graphs and DBSCAN (ch 11).
