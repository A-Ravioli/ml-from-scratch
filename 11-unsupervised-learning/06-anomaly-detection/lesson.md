# anomaly detection

## why this matters

a card payment from a new country at 3 a.m.; a sensor that drifts before a machine fails; a mislabelled image in a training set; a login pattern no human produces. anomaly detection asks "which of these points don't fit?", usually without labelled examples of what "not fitting" looks like, because anomalies are rare, varied, and often new. it's unsupervised learning with a sharp edge: the output is a ranking of suspicion, and someone acts on the top of the list.

there are two broad ways to define "doesn't fit". **density-based** methods model where normal data live and flag points in low-density regions: a fitted Gaussian (the Mahalanobis distance), the GMM from topic 02, or the distance to the nearest neighbours. **isolation-based** methods skip density estimation entirely and ask how easy a point is to *separate* from the rest by random cuts: anomalies are few and different, so a handful of random splits isolates them, while a normal point deep inside a crowd needs many. that's the **isolation forest** (Liu, Ting & Zhou, 2008), which you'll build tree by tree. the lab probes the knob every deployed detector has, the **contamination rate** (what fraction you call anomalous), and two failure modes: anomalies that hide in a crowd of their own (masking) and normal data with more than one mode.

## warm-up: predict first

1. normal data form two well-separated blobs, and an anomaly sits exactly halfway between them. you fit one Gaussian to everything and score by Mahalanobis distance. where does the anomaly rank?
2. a random cut picks a feature and a threshold uniformly between that feature's min and max. a point sits far outside the bulk of the data on one feature. what's the chance that the very first cut separates it from everyone else?
3. an isolation forest fitted on data where *no* point is special (all identical) gives every point which score?
4. you tell the detector `contamination=0.1` but only 2% of the data are anomalies. does the ranking of points change? what happens to precision and recall?
5. 50 anomalies sit in their own tight little cluster. would you rather build each isolation tree on 16 points or on 4,000?

## prerequisites check

- [ ] you can evaluate a Gaussian log density with a Cholesky factor (→ topic 02, ch 04 · topic 04)
- [ ] you've fitted a `GaussianMixture` and used `score_samples` (→ topic 02)
- [ ] you know ROC curves and AUC (→ ch 07 · topic 03)
- [ ] you know binary search trees, and the harmonic numbers $H_n \approx \ln n + 0.5772$ (→ ch 01)
- [ ] you can sample without replacement with a seeded generator (→ ch 00 · topic 05)

## core ideas

### idea 1: density scores

if you have a model $p(x)$ of normal data, the natural anomaly score is $-\log p(x)$: surprise. for a single Gaussian fitted by maximum likelihood (mean $\mu$, covariance $\Sigma$),

$$-2\log\mathcal N(x \mid \mu, \Sigma) = (x - \mu)^\top\Sigma^{-1}(x - \mu) + \text{const},$$

so ranking by density is ranking by the squared **Mahalanobis distance**: distance measured in units of the data's own spread, along its own correlations. a point 3 units out along a direction where the data vary by 10 is ordinary; 3 units out where they vary by 0.1 is extraordinary. a constant feature makes $\Sigma$ singular; adding a small `reg` to the diagonal keeps the score finite.

one Gaussian is a strong assumption (warm-up 1). with two modes, the fitted mean lies between them, exactly where an anomaly might hide, and the Mahalanobis distance calls that point the most normal of all. a mixture (topic 02's `score_samples`) fixes this when you know roughly how many modes there are. a non-parametric alternative needs no model at all: the **k-nearest-neighbour distance**, the distance from a point to its $k$-th nearest other point. a point in a sparse region is far from its neighbours; and since a ball of radius $r$ around $x$ holds about $n\,p(x)\,r^d$ points, the $k$-NN distance behaves like $(k/(n\,p(x)))^{1/d}$: a density estimate in disguise.

both have costs. density scores depend on the scale of every feature (standardize), the $k$-NN score needs all pairwise distances ($O(n^2)$), and in high dimensions distances concentrate and densities become hard to estimate (the curse of dimensionality).

> **▶ build: stage 1.** implement `gaussian_scores(X_train, X, *, reg=1e-6)` (squared Mahalanobis distance under the MLE Gaussian of `X_train`, via a Cholesky solve) and `knn_distance_scores(X, k=5)` (distance to the $k$-th nearest *other* point, `ValueError` unless $1 \le k < n$).

### idea 2: the yardstick c(n)

isolation forests measure how many random cuts it takes to isolate a point. to turn that depth into a score you need to know what's *typical* for a sample of $n$ points, and that turns out to be a classic number. growing a tree by random cuts until every point is alone is structurally the same as building a binary search tree by inserting keys in random order; the depth at which a point ends up corresponds to an **unsuccessful search** in a random BST, whose average length is

$$c(n) = 2H(n - 1) - \frac{2(n - 1)}{n}, \qquad H(i) \approx \ln i + \gamma, \quad \gamma = 0.5772\ldots \text{(Euler's constant)}$$

with $c(2) = 1$ (two points: one cut) and $c(1) = 0$ (nothing to cut). it grows like $2\ln n$: 10.24 for $n = 256$. (Liu et al. use this approximation of $H$; so does scikit-learn, and so will you.)

$c(n)$ appears twice: as the normalizer of the score, and as a correction at the leaves. trees are cut off at a maximum depth, so a leaf may still hold several training points; a point that lands in a leaf with $m$ points is credited with its depth plus $c(m)$, the number of further cuts it would typically have needed.

> **▶ build: stage 2.** implement `average_path_length(n)`, vectorized over arrays, with the special cases $c(0) = c(1) = 0$ and $c(2) = 1$.

### idea 3: isolation trees

an **isolation tree** is grown on a sample $S$ by recursive random partitioning:

1. if the node is at the maximum depth, holds at most one point, or all its points are identical, it's a leaf. record its depth and the number of points in it.
2. otherwise pick a feature uniformly at random among those that aren't constant in the node (`candidates[rng.integers(len(candidates))]`), and a threshold uniformly between that feature's minimum and maximum in the node (`rng.uniform(lo, hi)`).
3. send points with feature value below the threshold left, the rest right, and recurse.

no impurity, no labels, no optimization: the splits are completely random. why does that find anomalies? a point far from the others on some feature is separated as soon as a threshold falls in the gap between it and the rest, and when the gap is a large fraction of the feature's range, that's likely (warm-up 2, on paper 3). points deep inside a crowd are separated only after many cuts have narrowed the range around them.

**representation.** store the tree as parallel arrays indexed by node: `feature` ($-1$ for a leaf), `threshold`, `left`, `right` (child indices), `depth`, `size` (training points in the node). growing it with an explicit stack of `(node, indices)` avoids recursion limits.

**path lengths.** to evaluate the tree on new points, don't walk point by point in Python; push all points down at once. keep an array `node` (every point starts at the root, 0); at each level, for the points sitting at internal nodes, compare their value on the node's feature with its threshold and move them to the left or right child. after at most (tree depth) rounds every point is at a leaf, and its path length is `depth[leaf] + c(size[leaf])`.

> **🔮 predict:** the three points $0$, $1$ and $10$ on a line, and trees grown without a depth limit. what's the expected path length of each point?

> **▶ build: stage 3.** implement `isolation_tree(X, max_depth, rng=None)` → the dict of node arrays (`feature`, `threshold`, `left`, `right`, `depth`, `size`), and `path_lengths(tree, X)` → $h(x)$ for every row, vectorized over points.

### idea 4: the isolation forest

one random tree is noisy; average many. the **isolation forest** grows `n_estimators` trees, each on its own random subsample of $\psi$ = `max_samples` points (without replacement; all $n$ if $n < \psi$), with depth limit $\lceil\log_2\psi\rceil$, the depth of a balanced tree on $\psi$ points. the score of a point is

$$s(x) = 2^{-\mathbb E[h(x)]/c(\psi)} \in (0, 1],$$

with $\mathbb E[h(x)]$ the average path length over the trees. reading it (on paper 4): $\mathbb E[h] = c(\psi)$, a typical depth, gives exactly $s = 0.5$ (and if no point is distinguishable from any other, every point gets 0.5, warm-up 3); $\mathbb E[h] \to 0$, isolated at once, gives $s \to 1$; normal points, deeper than typical, score below 0.5. (scikit-learn's `score_samples` returns $-s$, "higher is more normal"; the course keeps **higher = more anomalous** throughout.)

two design choices are surprising at first:

- **small subsamples help.** $\psi = 256$ is the default even for millions of points. smaller samples are faster, but they also reduce **masking** (a dense group of anomalies looks normal when its members are all present, because they shield each other from cuts) and **swamping** (normal points near anomalies being flagged). in a small subsample, a group of 50 anomalies is represented by only a few points, which are easy to isolate (warm-up 5).
- **the threshold is separate from the model.** the trees and scores don't depend on the expected anomaly rate at all. `contamination` only sets `threshold_`, the $(1 - \text{contamination})$ quantile of the training scores, which `predict` uses to output 1 (anomaly) or 0. a wrong contamination rate leaves the ranking (and the AUC) untouched and moves the precision–recall trade-off (warm-up 4).

> **🔮 predict:** 5% of the data are scattered anomalies, and you set `contamination` to 1%, 5%, 10% and 20%. how do the AUC, precision and recall change?

> **▶ build: stage 4.** implement `IsolationForest(n_estimators=100, *, max_samples=256, contamination=0.1, rng=None)`: `fit` (`max_samples_`, `estimators_`, `threshold_`; one generator for every subsample and tree), `score_samples` and `predict`.

## worked example

**c(n).** $c(3) = 2(\ln 2 + 0.5772) - \frac{2 \cdot 2}{3} = 2.5407 - 1.3333 = 1.2074$, and $c(256) = 2(\ln 255 + 0.5772) - \frac{2 \cdot 255}{256} = 12.2369 - 1.9922 = 10.2448$.

**one tree on three points.** take $0$, $1$ and $10$ on a line, no depth limit. the first threshold is uniform on $(0, 10)$.

- with probability 0.9 it falls in $(1, 10)$: $10$ is alone at depth 1, and $\{0, 1\}$ needs one more cut, so both reach depth 2.
- with probability 0.1 it falls in $(0, 1)$: $0$ is alone at depth 1, and $\{1, 10\}$ needs one more cut.

so $\mathbb E[h(0)] = 0.9 \cdot 2 + 0.1 \cdot 1 = 1.9$, $\mathbb E[h(1)] = 2$, $\mathbb E[h(10)] = 0.9 \cdot 1 + 0.1 \cdot 2 = 1.1$. with $\psi = 3$ the scores are $s(10) = 2^{-1.1/1.2074} = 0.532$, $s(0) = 2^{-1.9/1.2074} = 0.336$ and $s(1) = 2^{-2/1.2074} = 0.317$: the outlier is the only point above 0.5. the tests check these expectations over 3,000 of your trees.

## on paper

1. **Mahalanobis is density.** show that ranking points by $-\log\mathcal N(x \mid \mu, \Sigma)$ is the same as ranking them by $(x - \mu)^\top\Sigma^{-1}(x - \mu)$, and that the latter is invariant to any invertible linear change of the features (if you refit $\mu$ and $\Sigma$).

<details><summary>solution</summary>

$-\log\mathcal N = \frac12(x - \mu)^\top\Sigma^{-1}(x - \mu) + \frac12\log\det(2\pi\Sigma)$; the second term is the same for every $x$, so the order is the order of the quadratic form. under $x' = Ax + b$, the refitted mean is $A\mu + b$ and covariance $A\Sigma A^\top$, so $(x' - \mu')^\top(A\Sigma A^\top)^{-1}(x' - \mu') = (x - \mu)^\top A^\top A^{-\top}\Sigma^{-1}A^{-1}A(x - \mu) = (x - \mu)^\top\Sigma^{-1}(x - \mu)$. unlike the Euclidean distance, it doesn't care about units or rotations.

</details>

2. **the k-NN distance as a density estimate.** if $p$ is roughly constant near $x$, how many of $n$ points fall within radius $r$ of $x$ in $d$ dimensions? solve for the radius containing $k$ points.

<details><summary>solution</summary>

about $n\,p(x)\,V_dr^d$, with $V_d$ the volume of the unit ball. setting this to $k$: $r_k(x) = \left(\frac{k}{n\,p(x)\,V_d}\right)^{1/d}$. so the $k$-NN distance is a decreasing function of $p(x)$: ranking by it is ranking by an estimate of $-p(x)$. the exponent $1/d$ also shows the curse of dimensionality: in high dimensions large density differences become small distance differences.

</details>

3. **a far point is isolated fast.** $n$ points lie in $[0, 1]$ on one feature and one more point sits at $1 + g$. what's the probability that the first random cut isolates it? for $g = 1$?

<details><summary>solution</summary>

the threshold is uniform on the node's range $[0, 1 + g]$; the far point is alone on the right exactly when the threshold falls in $(1, 1 + g)$, which has probability $g/(1 + g)$: $\frac12$ for $g = 1$, $\frac{9}{10}$ for $g = 9$. a point strictly inside the bulk can never be isolated by one cut (one cut leaves points on both sides of it); only the bulk's extreme point at the other end can, if the threshold falls in the gap next to it, with probability about $1/(n(1 + g))$ for evenly spread points. the further out a point is, the shorter its expected path, which is what the score rewards.

</details>

4. **reading the score.** show that $s = 0.5$ when $\mathbb E[h] = c(\psi)$, find the limits of $s$ as $\mathbb E[h] \to 0$ and as $\mathbb E[h] \to \psi - 1$ (the deepest possible path), and evaluate the latter for $\psi = 256$.

<details><summary>solution</summary>

$2^{-c/c} = 2^{-1} = 0.5$. as $\mathbb E[h] \to 0$, $s \to 2^0 = 1$. at $\mathbb E[h] = \psi - 1 = 255$, $s = 2^{-255/10.24} \approx 3 \times 10^{-8}$: essentially 0. in practice the depth limit $\lceil\log_2\psi\rceil = 8$ plus the leaf correction keeps typical normal scores around 0.4–0.5, so the useful information is in the *ranking*, not the absolute values.

</details>

## experiment

`lab.py` runs your detectors:

1. the contamination rate: what it changes (the threshold) and what it doesn't (the ranking).
2. masking: a tight cluster of anomalies, and the effect of the subsample size `max_samples`.
3. a Gaussian score, a $k$-NN score and an isolation forest on two-mode data with anomalies in between.

## check your understanding

1. why is `contamination` a hyperparameter of the *decision*, not of the model?

<details><summary>answer</summary>

the trees and the scores are built without it; it only chooses where to cut the ranked list (the threshold). that matches how detectors are used: an analyst can review 50 alerts a day, so you flag the top 50, whatever the "true" rate. if the true rate is unknown, report a ranking metric (AUC, precision at k) rather than metrics at an arbitrary threshold.

</details>

2. why do isolation trees use random thresholds instead of optimized splits?

<details><summary>answer</summary>

there's nothing to optimize without labels, and randomness is the point: the expected depth under random cuts is what measures "few and different". optimized splits (like a decision tree's) would need a target. randomness also makes trees cheap (no search over thresholds) and the forest's average stable.

</details>

3. why does a leaf with $m$ points add $c(m)$ to the path length, instead of nothing?

<details><summary>answer</summary>

the depth limit stopped the tree early; the points in that leaf weren't isolated yet. $c(m)$ estimates how many more random cuts they'd have needed on average. without it, every point in a big leaf of normal data would get the same short path as an isolated outlier at that depth.

</details>

4. an isolation forest scores a point 0.48 and another 0.51. how confident are you that the second is anomalous?

<details><summary>answer</summary>

not at all, by those numbers alone. scores near 0.5 mean "about as hard to isolate as a typical point"; the absolute values depend on $\psi$ and the data. look at the ranking within your data, the score distribution (is there a clear tail?), and the variability across forests with different seeds.

</details>

5. why might you prefer an isolation forest over a $k$-NN score on a data set with $10^7$ rows and 50 features?

<details><summary>answer</summary>

the $k$-NN score needs nearest-neighbour searches over all points (quadratic naively, and approximate indexes degrade in high dimensions). the isolation forest's cost is $n_{\text{estimators}} \cdot \psi\log\psi$ to fit, independent of $n$, and $n_{\text{estimators}} \cdot \log\psi$ per scored point. it also needs no distance metric, so it's insensitive to feature scaling (each split looks at one feature at a time).

</details>

6. your anomalies differ from normal data only in a *combination* of features (each feature alone looks normal). which detectors can catch them?

<details><summary>answer</summary>

the Mahalanobis distance can, if the normal data are correlated and the anomaly breaks the correlation: it measures distance along the covariance structure. $k$-NN distance can, if the anomaly lands in an empty region of the joint space. axis-parallel isolation trees struggle: every single split looks at one feature, where the anomaly is in the normal range; they isolate it only by many cuts that narrow both features, so its path isn't much shorter. extended isolation forests use random hyperplanes for exactly this reason.

</details>

## common bugs

- **inverted sign.** returning sklearn-style "higher is normal" scores. symptom: AUC near 0 instead of near 1.
- **counting a point as its own neighbour** in the $k$-NN score. symptom: $k = 1$ gives all zeros.
- **normalizing by $c(n)$ instead of $c(\psi)$.** symptom: identical data don't score 0.5; scores shift with the size of the training set.
- **forgetting the $c(\text{size})$ correction at leaves.** symptom: many ties among normal points; poor ranking when the depth limit is small.
- **re-seeding per tree** (`as_rng(self.rng)` inside the loop with an int seed). symptom: all trees identical; the forest is no better than one tree.
- **choosing the threshold from a range that excludes the extremes in the wrong way**, e.g. `rng.uniform(lo, hi)` on the whole data's range instead of the node's. symptom: many useless splits that send every point to one side; longer paths for everyone.

## going deeper

- Liu, Ting & Zhou (2008), "Isolation forest", *ICDM*. [doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)
- Liu, Ting & Zhou (2012), "Isolation-based anomaly detection", *ACM TKDD* 6(1): the extended journal version, with masking, swamping and subsampling. [doi:10.1145/2133360.2133363](https://doi.org/10.1145/2133360.2133363)
- Hariri, Kind & Brunner (2021), "Extended isolation forest", *IEEE TKDE* 33(4): random hyperplane splits. [arXiv:1811.02141](https://arxiv.org/abs/1811.02141)
- Breunig, Kriegel, Ng & Sander (2000), "LOF: identifying density-based local outliers", *SIGMOD*: a local density score for data with varying density. [doi:10.1145/342009.335388](https://doi.org/10.1145/342009.335388)
- Chandola, Banerjee & Kumar (2009), "Anomaly detection: a survey", *ACM Computing Surveys* 41(3). [doi:10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882)
- Knuth (1998), *The Art of Computer Programming*, vol. 3, section 6.2.2: path lengths in random binary search trees, the origin of $c(n)$.

## connections

- ← builds on: Gaussian densities and `GaussianMixture.score_samples` (topic 02), nearest-neighbour distances (topics 04–05), ROC/AUC (ch 07 · topic 03), seeded subsampling (ch 00).
- → used in: decision trees and random forests, whose growing, depth limits and feature subsampling mirror isolation trees (ch 10), likelihood-based anomaly scores from deep generative models (ch 21), uncertainty and out-of-distribution detection (ch 28 · topic 05), and filtering pretraining data (ch 27 · topic 01).
