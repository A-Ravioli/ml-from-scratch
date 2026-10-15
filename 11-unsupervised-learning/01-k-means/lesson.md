# k-means

## why this matters

you have 60,000 handwritten digits and no labels. are there natural groups in them? how many? which images belong together? nobody can tell you the right answer, because there isn't a label column to check against, and yet this is the everyday situation: customer records, gene-expression profiles, sensor logs and document embeddings mostly arrive unlabelled, and "find the groups" is the first thing anyone asks.

k-means is the oldest serious answer and still the default. it has one idea: a good grouping is one where every point is close to the centre of its group. that idea gives you an objective you can write down, an algorithm (Lloyd's) that's two lines of NumPy per iteration and provably never makes the objective worse, a seeding trick (k-means++) with a guarantee, and a long list of ways to go wrong: the wrong $k$, a bad start, clusters that aren't round. you'll meet all of them. later this chapter builds on it directly: Gaussian mixtures (topic 02) are a soft, probabilistic k-means; spectral clustering (topic 05) runs k-means on a transformed space; and chapter 27's vector index uses your `KMeans` to partition a collection of embeddings into buckets for fast search.

## warm-up: predict first

1. Lloyd's algorithm alternates two steps. can either step ever increase the sum of squared distances from points to their centres?
2. you run k-means with $k = 3$ on data that has 3 obvious, well-separated round blobs. can it return a clustering that puts two blobs in one cluster and splits the third? does it stop there?
3. two moons interlock like a yin-yang. k-means with $k = 2$ splits them along a straight line. is that because it got stuck, or would the true moons have a *higher* k-means objective?
4. the k-means objective (inertia) on 1,000 points with $k = 1, 2, \dots$: what is it at $k = 1{,}000$? can you pick $k$ by minimizing it?
5. k-means++ picks each new starting centre with probability proportional to $D(x)^2$, the squared distance to the nearest centre chosen so far. why squared, and not just $D(x)$, or uniform?

## prerequisites check

- [ ] you can compute all pairwise squared distances between two point sets without an $(n, k, d)$ array (→ ch 00 · topic 02, `pairwise_distances`)
- [ ] you know that the mean minimizes the sum of squared deviations (→ ch 05 · topic 01)
- [ ] you can write an `Estimator` with `fit` returning `self`, fitted attributes ending in `_`, and an `rng` handled by `as_rng` (→ ch 07 · topic 01, ch 00 · topic 05)
- [ ] you know what a local minimum of a non-convex objective is (→ ch 06 · topic 01)

## core ideas

### idea 1: the objective, and Lloyd's two steps

give each of $k$ clusters a centre $\mu_1, \dots, \mu_k \in \mathbb R^d$ and each point $x_i$ a label $z_i \in \{1, \dots, k\}$. k-means looks for the labels and centres that minimize the **inertia** (the within-cluster sum of squares):

$$J(z, \mu) = \sum_{i=1}^n \|x_i - \mu_{z_i}\|^2 .$$

minimizing over both at once is hard (NP-hard in general, even for $k = 2$). but minimizing over each *given the other* is easy:

- **assignment step.** with the centres fixed, each term depends only on its own $z_i$, so set $z_i = \arg\min_j \|x_i - \mu_j\|^2$: every point joins its nearest centre. ties need a rule; take the lowest index, which is what `np.argmin` does.
- **update step.** with the labels fixed, the terms split by cluster, and $\sum_{i: z_i = j}\|x_i - \mu_j\|^2$ is minimized by the cluster's mean (on paper 1): $\mu_j = \frac{1}{n_j}\sum_{i: z_i = j} x_i$.

alternating the two is **Lloyd's algorithm** (Lloyd, 1957, published 1982). each step can only lower $J$ or leave it unchanged (warm-up 1), $J \ge 0$, and there are finitely many labelings, so it stops after finitely many iterations at a point where neither step changes anything (on paper 2). that point is a *local* minimum: a fixed point of the two steps, not necessarily the best clustering.

the cost of one iteration is all in the assignment: $n \times k$ squared distances. compute them with the expansion $\|x - \mu\|^2 = \|x\|^2 - 2x^\top\mu + \|\mu\|^2$ (your `pairwise_distances(..., squared=True)`), never by broadcasting `X[:, None] - centers`, which builds an $(n, k, d)$ array: 65 MB for 4,000 points, 32 centres and 64 features, and gigabytes at chapter 27's scale. (even the $(n, k)$ distance matrix grows with $n$: 100,000 points and 1,000 centres is 800 MB, so at that scale you'd assign points in chunks of a few thousand rows; the algorithm doesn't change.)

one edge case: a cluster can lose all its points (its centre was nobody's nearest). its mean is $0/0$. the simplest safe rule, and the one you'll use, is to leave that centre where it was. (scikit-learn moves it to a far-away point instead; either is fine, `NaN` isn't.)

> **▶ build: stage 1.** implement `assign_to_nearest(X, centers)` → `(labels, d2)` with `d2` the squared distance to the nearest centre, and `update_centers(X, labels, centers)` → the new centres (cluster means; an empty cluster keeps its old centre; don't modify the input).

### idea 2: where to start: k-means++

Lloyd's algorithm only finds the local minimum nearest its start, and bad starts are common. pick the $k$ starting centres uniformly at random from the data, and with 9 well-separated blobs the chance that each blob gets exactly one is $9!/9^9 \approx 0.1\%$. two centres in one blob and none in another is a local minimum Lloyd can't escape: the two centres split their blob, and the orphaned blob's points join some neighbouring cluster whose centre then sits between two blobs (warm-up 2).

**k-means++** (Arthur & Vassilvitskii, 2007) seeds more cleverly. pick the first centre uniformly from the data. then, until you have $k$, pick the next one at random with probability

$$p(x_i) = \frac{D(x_i)^2}{\sum_{l} D(x_l)^2}, \qquad D(x) = \text{distance from } x \text{ to the nearest centre chosen so far}.$$

points far from every existing centre are likely to be chosen, so the centres spread out; points already covered ($D = 0$, including the chosen centres themselves) can't be chosen again. why the square? because $D^2$ is what the objective counts: you're sampling in proportion to each point's current contribution to $J$. and it comes with a guarantee: the seeding alone, before any Lloyd iterations, has expected inertia at most $8(\ln k + 2)$ times the optimum. no such guarantee exists for uniform seeding, or for sampling by $D$.

on the three points $0, 1, 10$ with $k = 2$, you can work out the exact distribution of the chosen pair (on paper 3): the pair $\{0, 10\}$ comes up with probability $0.514$, $\{1, 10\}$ with $0.478$, and $\{0, 1\}$, two centres for the tight pair and none for the outlier, only $0.007$. the tests check your sampler against these numbers.

> **🔮 predict:** on a 5 × 5 grid of well-separated blobs, how often will a *single* k-means++ run find the perfect clustering? and a single run from uniformly random starting points?

> **▶ build: stage 2.** implement `kmeans_plus_plus(X, n_clusters, *, rng=None)`: the first centre at `rng.integers(n)`, each next one by `rng.choice(n, p=D**2 / sum(D**2))`, updating $D$ with a running minimum. return the chosen rows of `X`.

### idea 3: the estimator: restarts and stopping

even k-means++ starts don't always land in the best basin, so the standard practice is **restarts**: run the whole thing `n_init` times from independent starts and keep the run with the lowest inertia. that's cheap insurance, because the runs are independent: if one run finds a good solution with probability $p$, ten runs miss it with probability $(1-p)^{10}$.

**stopping.** iterate until the centres stop moving: stop when the total squared shift $\sum_j\|\mu_j^{\text{new}} - \mu_j^{\text{old}}\|^2$ is at most `tol` times the data's mean per-feature variance (scikit-learn's convention, so `tol` doesn't depend on the units of $X$), or after `max_iters` iterations. count iterations in `n_iter_`; the iteration that moves nothing counts. after the loop, do one more assignment so that `labels_` and `inertia_` belong to the final `cluster_centers_`.

the class follows the estimator protocol of chapter 07, unsupervised flavour: `fit(X, y=None)` ignores `y` and returns `self`; `predict(X)` assigns new points to the nearest fitted centre; `transform(X)` maps each point to its $k$ distances from the centres (a new feature space, useful in its own right); `score(X)` is minus the inertia, so that "higher is better" holds as for every score. for `init`, accept `"k-means++"`, `"random"` ($k$ distinct rows chosen uniformly), or an explicit $(k, d)$ array of starting centres, in which case there's only one possible run.

> **🔮 predict:** two moons, interlocked, $k = 2$. k-means draws a straight boundary through both. is the k-means objective of the true two-moon labeling higher or lower than that of k-means's answer?

> **▶ build: stage 3.** implement `KMeans(n_clusters=8, *, init="k-means++", n_init=10, max_iters=300, tol=1e-4, rng=None)`: `fit` (all runs share one generator from `as_rng(self.rng)`; set `cluster_centers_`, `labels_`, `inertia_`, `n_iter_` from the best run), `predict`, `fit_predict`, `transform` and `score`. the helper `_initial_centers` that picks a run's starting centres is given.

### idea 4: how many clusters? (and how good are they?)

k-means needs $k$, and the objective can't choose it: adding a centre never increases the optimal inertia, and at $k = n$ every point is its own centre and $J = 0$ (warm-up 4). two common ways out:

- **the elbow.** plot the inertia against $k$ and look for the bend after which adding clusters buys little. it's a judgment call, and on real data there's often no clear elbow.
- **the silhouette** (Rousseeuw, 1987). for each point $i$, let $a(i)$ be its mean distance to the *other* points in its cluster, and $b(i)$ the smallest mean distance to the points of any *other* cluster (its nearest neighbouring cluster). then

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1],$$

near 1 when $i$ sits deep inside a cluster far from the others, near 0 on a boundary, negative when it would fit better in the neighbouring cluster. a point alone in its cluster gets $s(i) = 0$ by convention. the **silhouette score** is the mean of $s(i)$; you choose the $k$ that maximizes it. it's defined for $2 \le k \le n - 1$ only.

computed naively the silhouette is a double loop over points. vectorized, it's one $(n, n)$ distance matrix $D$ and one matrix product: with $M$ the $(n, k)$ one-hot label matrix, $(DM)_{ij}$ is the total distance from point $i$ to cluster $j$, and dividing by the cluster sizes (minus one for a point's own cluster) gives every $a(i)$ and every mean distance to other clusters at once.

neither tool knows what you'll *use* the clusters for, and both inherit k-means's taste for round, equal-sized groups. the silhouette in particular rewards well-separated groups, so when structure is nested (pairs of blobs, each pair far from the others) it often prefers the coarse level. the lab shows this.

> **▶ build: stage 4.** implement `silhouette_score(X, labels)`: the mean of $s(i)$ (singletons score 0), labels of any type, a `ValueError` unless $2 \le k \le n - 1$. vectorize it with the one-hot trick.

## worked example

six points on a line, $x = 0, 1, 2, 10, 11, 12$, $k = 2$, starting centres $\mu = (0, 2)$.

**iteration 1.** assignment: $0 \to$ centre 0; $1$ is at distance 1 from both, so the tie goes to centre 0; $2, 10, 11, 12 \to$ centre 1. update: $\mu_0 = (0 + 1)/2 = 0.5$, $\mu_1 = (2 + 10 + 11 + 12)/4 = 8.75$. the centres moved by $0.5^2 + 6.75^2 = 45.8$.

**iteration 2.** assignment: $2$ is now $1.5$ from $\mu_0$ and $6.75$ from $\mu_1$, so it switches to centre 0; the rest stay. update: $\mu = (1, 11)$.

**iteration 3.** assignment changes nothing; the update gives $(1, 11)$ again, a shift of 0, so the loop stops: `n_iter_ = 3`.

the final labels are $(0, 0, 0, 1, 1, 1)$ and the inertia is $1 + 0 + 1 + 1 + 0 + 1 = 4$. the objective went from $J = 0 + 1 + 0 + 64 + 81 + 100 = 246$ under the starting centres, to $0.25 + 0.25 + 45.6 + 1.56 + 5.06 + 10.6 = 63.3$ after the first update, to 4. the tests check all of this.

## on paper

1. **the mean minimizes squared error.** show that $f(\mu) = \sum_{i=1}^m \|x_i - \mu\|^2$ is minimized by $\mu = \bar x$, and that $f(\mu) = f(\bar x) + m\|\mu - \bar x\|^2$.

<details><summary>solution</summary>

write $x_i - \mu = (x_i - \bar x) + (\bar x - \mu)$ and expand: $f(\mu) = \sum\|x_i - \bar x\|^2 + 2(\bar x - \mu)^\top\sum(x_i - \bar x) + m\|\bar x - \mu\|^2$. the middle sum is zero by the definition of $\bar x$. so $f(\mu) = f(\bar x) + m\|\mu - \bar x\|^2 \ge f(\bar x)$, with equality only at $\mu = \bar x$. the identity also tells you how much the update step gains: moving a centre from $\mu$ to its cluster's mean reduces $J$ by exactly $n_j\|\mu - \bar x_j\|^2$.

</details>

2. **Lloyd's algorithm terminates.** show that $J$ never increases, and that the algorithm stops after finitely many iterations (assume ties are broken by a fixed rule).

<details><summary>solution</summary>

the assignment step picks, for each $i$, the $z_i$ minimizing its own term, so $J(z^{\text{new}}, \mu) \le J(z, \mu)$. the update step minimizes each cluster's sum given the labels (problem 1), so $J(z, \mu^{\text{new}}) \le J(z, \mu)$. hence $J$ is non-increasing. the centres after an update are determined by the labeling, and there are at most $k^n$ labelings. if a labeling repeated after $J$ strictly decreased in between, the same labeling would give the same centres and the same $J$: a contradiction. so either $J$ stays constant (then, with a fixed tie rule, labels and centres stop changing) or it visits a new labeling; after finitely many steps it must stop. (the bound $k^n$ is astronomically loose; in practice tens of iterations, though there are constructed examples needing exponentially many.)

</details>

3. **k-means++ by hand.** for the points $0, 1, 10$ and $k = 2$, compute the probability of each pair of starting centres.

<details><summary>solution</summary>

the first centre is each point with probability $1/3$. if it's $0$: $D^2 = (0, 1, 100)$, so the second is $1$ with probability $1/101$ and $10$ with $100/101$. if it's $1$: $D^2 = (1, 0, 81)$, giving $0$ with $1/82$ and $10$ with $81/82$. if it's $10$: $D^2 = (100, 81, 0)$, giving $0$ with $100/181$ and $1$ with $81/181$. so $P(\{0, 10\}) = \frac13(\frac{100}{101} + \frac{100}{181}) = 0.514$, $P(\{1, 10\}) = \frac13(\frac{81}{82} + \frac{81}{181}) = 0.478$, $P(\{0, 1\}) = \frac13(\frac{1}{101} + \frac{1}{82}) = 0.007$. with $D$ instead of $D^2$ the bad pair $\{0, 1\}$ would come up with probability $0.064$, nine times as often.

</details>

4. **why k-means cuts straight.** show that the set of points nearer to $\mu_1$ than to $\mu_2$ is a half-space, and conclude that every k-means cluster is a convex polygon (a Voronoi cell).

<details><summary>solution</summary>

$\|x - \mu_1\|^2 < \|x - \mu_2\|^2 \iff -2x^\top\mu_1 + \|\mu_1\|^2 < -2x^\top\mu_2 + \|\mu_2\|^2 \iff 2x^\top(\mu_2 - \mu_1) < \|\mu_2\|^2 - \|\mu_1\|^2$: the $\|x\|^2$ terms cancel, leaving a linear inequality, a half-space whose boundary is the perpendicular bisector of $\mu_1\mu_2$. cluster $j$ is the intersection of $k - 1$ such half-spaces, so it's convex. a crescent-shaped cluster is not convex, so no choice of centres can produce it.

</details>

## experiment

`lab.py` puts your `KMeans` and `silhouette_score` to work:

1. k-means on two moons and on concentric rings, and the objective of the true labeling versus k-means's answer.
2. on a 5 × 5 grid of blobs: how often a single run finds the right clustering, from uniform starts and from k-means++, and what restarts buy.
3. choosing $k$ with the elbow and the silhouette when the clusters come in pairs.

## check your understanding

1. why does the update step use the mean, and what would you use if the objective were $\sum_i\|x_i - \mu_{z_i}\|_1$ instead?

<details><summary>answer</summary>

the mean minimizes the sum of *squared* Euclidean distances (on paper 1). for the sum of $\ell_1$ distances, the minimizer is the coordinate-wise median, which gives k-medians: more robust to outliers, because a far point pulls a median much less than a mean.

</details>

2. you multiply one feature by 1,000. does the k-means clustering change? what should you do about it?

<details><summary>answer</summary>

almost certainly: the distances are now dominated by that feature, so the clusters become slices along it. k-means is not scale invariant, so standardize features (or choose a scale deliberately) first, exactly as for $k$-NN in chapter 07. the "right" scale is a modelling choice: it says which differences count as large.

</details>

3. Lloyd's algorithm has converged. is the result the global minimum of $J$? how would you gain confidence that it's at least good?

<details><summary>answer</summary>

no, only a fixed point of the two steps. run it from many starts (that's `n_init`) and compare the inertias; if the best value is reached by many runs, that's evidence (not proof) that it's the global minimum. k-means++ seeding also bounds the expected inertia of the start within $O(\log k)$ of the optimum.

</details>

4. why does `KMeans.score` return *minus* the inertia?

<details><summary>answer</summary>

every `score` in the course follows "higher is better" (accuracy, $R^2$, log-likelihood), so that model selection code like `GridSearchCV` can always maximize. inertia is a cost, so its negative is the score. note that comparing scores across different $k$ is meaningless for choosing $k$ (bigger $k$ always wins).

</details>

5. a point is alone in its cluster. what's its silhouette, and why isn't $a(i) = 0$, $s(i) = 1$ the right convention?

<details><summary>answer</summary>

by convention 0. with $a(i) = 0$ the formula would give $s(i) = 1$, the best possible score, rewarding a clustering for isolating points; with $n$ singletons every point would score 1. setting it to 0 keeps "make every point its own cluster" from looking ideal.

</details>

6. k-means++'s guarantee is $8(\ln k + 2)$ times optimal *in expectation*. why might you still want restarts?

<details><summary>answer</summary>

an expectation bound allows individual runs to be bad, and a factor of $8(\ln k + 2)$ (about 34 for $k = 10$) is far from optimal anyway. Lloyd's iterations usually improve the seeding a lot, but each run still ends in one local minimum. independent restarts turn a per-run success probability $p$ into $1 - (1 - p)^{n_{\text{init}}}$.

</details>

7. in the assignment step you compute $\|x\|^2 - 2x^\top\mu + \|\mu\|^2$. what can go wrong numerically, and how does `pairwise_distances` handle it?

<details><summary>answer</summary>

catastrophic cancellation: for a point very close to a centre and far from the origin, the three terms are large and nearly cancel, so the result can come out slightly negative. `pairwise_distances` clips at 0 before any square root. for the argmin, tiny errors only matter between nearly tied centres.

</details>

## common bugs

- **broadcasting a 3-D difference.** `((X[:, None] - C[None]) ** 2).sum(-1)` is correct but allocates $n \cdot k \cdot d$ floats. symptom: fine in tests, out of memory on real data. use the expansion.
- **`NaN` centres.** dividing by a zero count for an empty cluster. symptom: one centre becomes `NaN`, every distance to it is `NaN`, and `argmin` behaves unpredictably from then on.
- **keeping the last restart instead of the best.** symptom: `n_init = 10` is no better than `n_init = 1`.
- **a fresh generator per restart.** `as_rng(self.rng)` inside the restart loop with an int seed gives every run the *same* start. symptom: all restarts identical; `n_init` does nothing.
- **sampling k-means++ by $D$, or uniformly.** symptom: starting centres in the same blob; the distribution test fails.
- **`labels_` from before the last update.** returning the labels computed at the start of the final iteration, then updating the centres. symptom: `predict(X)` disagrees with `labels_` for a few points.

## going deeper

- Lloyd (1982), "Least squares quantization in PCM", *IEEE Transactions on Information Theory* 28(2) (written in 1957). [doi:10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489)
- Arthur & Vassilvitskii (2007), "k-means++: the advantages of careful seeding", *SODA*. the seeding and its $O(\log k)$ guarantee. [dl.acm.org/doi/10.5555/1283383.1283494](https://dl.acm.org/doi/10.5555/1283383.1283494)
- Rousseeuw (1987), "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis", *Journal of Computational and Applied Mathematics* 20. [doi:10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)
- Tibshirani, Walther & Hastie (2001), "Estimating the number of clusters in a data set via the gap statistic", *JRSS B* 63(2): a principled elbow. [doi:10.1111/1467-9868.00293](https://doi.org/10.1111/1467-9868.00293)
- Sculley (2010), "Web-scale k-means clustering", *WWW*: mini-batch k-means, for when the data don't fit in memory. [doi:10.1145/1772690.1772862](https://doi.org/10.1145/1772690.1772862)
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, section 9.1.

## connections

- ← builds on: vectorized distances (ch 00 · topic 02), the mean as a least-squares estimate (ch 05), the estimator protocol (ch 07 · topic 01), local minima of non-convex objectives (ch 06).
- → used in: Gaussian mixtures, initialized from k-means and reducing to it in a limit (topic 02); spectral clustering, which runs k-means on eigenvectors (topic 05); the capstone's MNIST clustering; vector quantization and the inverted-file vector index of chapter 27.
