# manifold learning

## why this matters

PCA finds flat structure: the best plane through a cloud of points. but a lot of data lies near *curved* surfaces. the images of one object under a slowly turning camera trace out a curve in pixel space; a rolled-up sheet of paper (the classic "swiss roll") is two-dimensional, yet no plane through it captures it, because points on neighbouring layers are close in space and far apart along the sheet. **manifold learning** looks for low-dimensional coordinates that respect the *intrinsic* geometry: who is near whom along the surface.

this topic builds three methods that make that idea precise in different ways. **classical MDS** recovers coordinates from a table of distances, and is PCA in disguise. **Isomap** feeds it distances measured *along* the data, by shortest paths through a neighbourhood graph, and unrolls the swiss roll. **t-SNE** gives up on preserving distances at all and preserves *neighbourhoods*, which makes it the most popular tool for looking at high-dimensional data, and also one of the most misread. the lab is about how to read (and misread) a t-SNE plot; the capstone uses your t-SNE to look at MNIST.

## warm-up: predict first

1. someone hands you only the table of road distances between 10 cities. can you draw a map? what can't you recover?
2. two points on neighbouring layers of a swiss roll are 1 cm apart in space. how far apart are they for an ant walking on the sheet? which number should an embedding preserve?
3. t-SNE shows three clusters, one much bigger than the others, and two of them close together. what can you conclude about the original data?
4. t-SNE's output similarity uses a Student-t (Cauchy) kernel $1/(1 + d^2)$ instead of a Gaussian. what does a heavier tail do to points that are moderately far apart?
5. you run t-SNE on pure Gaussian noise (no clusters at all) with a very small perplexity. what do you see?

## prerequisites check

- [ ] you can compute pairwise (squared) distances without loops (→ ch 00 · topic 02)
- [ ] you know entropy and KL divergence, and that KL is asymmetric (→ ch 05 · topic 05)
- [ ] you know momentum and why it helps on badly conditioned problems (→ ch 06 · topic 03)
- [ ] you know PCA via eigendecomposition and the sign convention `flip_signs` (→ topic 03)
- [ ] you can check a gradient with finite differences (→ ch 02 · topic 07)

## core ideas

### idea 1: classical MDS: coordinates from distances

given only the $n \times n$ table of distances $D_{ij} = \|x_i - x_j\|$, can you recover the points? up to a rigid motion (rotation, reflection, translation), yes, because distances are all that survive a rigid motion (warm-up 1). the trick is to turn distances into inner products. expand $D_{ij}^2 = \|x_i\|^2 + \|x_j\|^2 - 2x_i^\top x_j$ and centre: with $J = I - \frac1n\mathbf 1\mathbf 1^\top$,

$$B = -\tfrac12 JD^{(2)}J = X_cX_c^\top,$$

where $D^{(2)}$ holds the *squared* distances and $X_c$ the centred points (on paper 1). in words: subtract each row's mean and each column's mean from $D^{(2)}$, add back the overall mean, multiply by $-\frac12$. $B$ is the Gram matrix of the centred configuration, so its top eigenvectors scaled by the square roots of the eigenvalues give coordinates: $E = V_k\Lambda_k^{1/2}$. this is **classical** (Torgerson) MDS. when the distances are Euclidean, it reproduces PCA's scores exactly (up to signs), computed from the $n \times n$ side, which is kernel PCA with a linear kernel from topic 03.

when the "distances" aren't Euclidean (road distances, dissimilarity scores, geodesics), $B$ can have negative eigenvalues: no configuration in any dimension reproduces them exactly. classical MDS then keeps the top positive part, the best low-rank fit to $B$. the course's sign convention applies to each embedding column (largest-magnitude entry positive).

> **▶ build: stage 1.** implement `MDS(n_components=2, *, dissimilarity="euclidean")` with `fit` (`embedding_`, `eigenvalues_` in descending order; `"precomputed"` means `X` is the distance matrix, which must be square and symmetric) and `fit_transform`.

### idea 2: geodesic distances on a graph

on a curved surface, the straight-line distance between two points can be much shorter than the distance *along* the surface (warm-up 2). you can't measure along the surface directly, but you can approximate it: nearby points are close along the surface too, so connect each point to its $k$ nearest neighbours, weight each edge by its Euclidean length, and measure distances as **shortest paths** through this graph. a path that hops between neighbours stays on the surface.

two details matter. the $k$-nearest-neighbour relation isn't symmetric ($j$ can be among $i$'s neighbours but not vice versa), so make the graph undirected: an edge whenever *either* point lists the other. and for all-pairs shortest paths on a dense matrix, **Floyd–Warshall** is three lines of NumPy: for each vertex $m$ in turn, allow it as an intermediate stop,

$$G_{ij} \leftarrow \min(G_{ij},\ G_{im} + G_{mj}),$$

vectorized over all $(i, j)$ at once with broadcasting. that's $O(n^3)$ work in $n$ NumPy operations: fine for a few thousand points. if the graph has several components, some distances stay $\infty$.

the neighbourhood size is a real choice. too small and the graph falls apart; too large and an edge jumps between layers of the roll (a **short circuit**), after which every geodesic that uses it is wrong and the embedding folds.

> **▶ build: stage 2.** implement `geodesic_distances(X, n_neighbors=10)`: the symmetrized $k$-NN graph (excluding each point itself) with Euclidean edge lengths, then Floyd–Warshall.

### idea 3: Isomap

**Isomap** (Tenenbaum, de Silva & Langford, 2000) is just the composition: geodesic distances, then classical MDS on them. for a surface that's a bent version of a flat region (like the swiss roll, which is an isometric rolling of a rectangle), the geodesics approximate the distances in the flat rectangle, and MDS recovers the rectangle: the roll is unrolled. if the graph is disconnected there's no answer; raise an error that says to increase `n_neighbors`.

> **🔮 predict:** Isomap with 10 neighbours on a densely sampled swiss roll, then on the same roll with fewer points. what happens when the sampling is too sparse for the neighbourhood size?

> **▶ build: stage 3.** implement `Isomap(n_components=2, *, n_neighbors=10)`: `fit` stores `dist_matrix_` and `embedding_` (raise `ValueError` for a disconnected graph), and `fit_transform`.

### idea 4: t-SNE's input affinities

Isomap preserves all distances, so one bad edge or one non-flat region distorts everything. **t-SNE** (van der Maaten & Hinton, 2008) asks for less: points that are neighbours in the data should be neighbours in the plot. it turns distances into probabilities. for each point $i$, a Gaussian centred on it defines how likely $i$ would pick $j$ as its neighbour:

$$p_{j|i} = \frac{\exp(-\beta_i\|x_i - x_j\|^2)}{\sum_{l \ne i}\exp(-\beta_i\|x_i - x_l\|^2)}, \qquad p_{i|i} = 0 .$$

the precision $\beta_i = 1/2\sigma_i^2$ is different for every point, so that dense and sparse regions get the same *effective number of neighbours*. that number is the **perplexity**: $\text{Perp}_i = e^{H_i}$ with $H_i = -\sum_jp_{j|i}\log p_{j|i}$ (natural log). a uniform distribution over $m$ neighbours has perplexity exactly $m$ (on paper 3). you choose the perplexity (typically 5–50), and each $\beta_i$ is found by **bisection**: the entropy decreases as $\beta_i$ grows, so if $H_i$ is too high, raise $\beta_i$; too low, lower it. start at $\beta = 1$, double or halve until the target is bracketed, then halve the bracket; vectorize over all rows at once, and stop when every row is within `tol` or after `max_iters` rounds. subtract each row's largest logit before exponentiating, or far-away rows underflow to $0/0$.

the conditionals aren't symmetric. t-SNE symmetrizes them into one joint distribution over pairs,

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n},$$

which sums to 1 and gives every point at least $\frac{1}{2n}$ of total affinity, so outliers still pull on someone.

> **▶ build: stage 4.** implement `conditional_probabilities(D2, perplexity=30.0, *, tol=1e-5, max_iters=100)` (bisection on every row's $\beta$ at once; `ValueError` unless $1 < \text{perplexity} < n - 1$) and `joint_probabilities(X, perplexity=30.0)`.

### idea 5: output affinities, the KL divergence and its gradient

place the points at $y_1, \dots, y_n$ in 2-D and define their affinities with a **Student-t kernel** with one degree of freedom:

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l}(1 + \|y_k - y_l\|^2)^{-1}}, \qquad q_{ii} = 0 .$$

t-SNE moves the $y$'s to minimize $C = \mathrm{KL}(P\,\|\,Q) = \sum_{i \ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}$. two choices shape everything:

- **the heavy tail** (warm-up 4). in 2-D there isn't enough room to place every point's 30 neighbours at the right distances: volume grows much more slowly in 2 dimensions than in 50 (the *crowding problem*). a Gaussian $q$ would force moderately distant points into a crush in the middle. the Cauchy tail makes "moderately far" in the data map to "much farther" in the plot, which opens up space and separates clusters.
- **the direction of KL.** a pair with large $p_{ij}$ and small $q_{ij}$ (neighbours drawn apart) costs a lot; a pair with small $p_{ij}$ and larger $q_{ij}$ (non-neighbours drawn close) costs little. so t-SNE works hard to keep neighbourhoods together and doesn't care much about large-scale distances.

the gradient has a beautifully simple form (on paper 2):

$$\frac{\partial C}{\partial y_i} = 4\sum_j(p_{ij} - q_{ij})\,\frac{y_i - y_j}{1 + \|y_i - y_j\|^2} .$$

read it as springs: $p_{ij}$ pulls $i$ towards its data neighbours, $q_{ij}$ pushes it away from everyone. vectorized, with $W_{ij} = (p_{ij} - q_{ij})/(1 + \|y_i - y_j\|^2)$, the whole gradient is $4(\operatorname{diag}(W\mathbf 1) - W)Y$: one $(n, n)$ matrix and one product. (at $n = 2{,}000$ every $(n, n)$ operation touches 4 million numbers, so count them: pass $Y$ twice to `pairwise_distances(Y, Y, squared=True)` to skip its symmetrizing pass, which you don't need here.)

> **▶ build: stage 5.** implement `tsne_kl_gradient(Y, P, *, kl=True)` → `(kl, grad)`, vectorized; terms with $p_{ij} = 0$ contribute 0 to the KL. computing the KL costs as much as the gradient, and the optimizer only needs the gradient, so with `kl=False` return `(None, grad)`.

### idea 6: optimizing t-SNE

the objective is non-convex, and the original paper's optimization tricks are part of what makes t-SNE work. the course's version:

- **initialization**: $y_i \sim \mathcal N(0, 10^{-4}I)$, drawn with `rng.standard_normal((n, n_components))`.
- **early exaggeration**: for the first 250 iterations, multiply $P$ by `early_exaggeration` (12). attraction then dominates, and clusters form as tight, well-separated groups that can move past each other before they settle.
- **momentum**: 0.5 for the first 250 iterations, 0.8 after. `update = momentum * update - lr * gains * grad`, then `Y = Y + update`, then re-centre $Y$ (subtract its mean; the objective doesn't care, and it keeps the embedding from drifting).
- **gains** (Jacobs' delta-bar-delta): a per-coordinate step-size multiplier, starting at 1, increased by 0.2 where `np.sign(grad) != np.sign(update)` (the step continues in the same direction as the last one, since the update moves *against* the gradient: speed up) and multiplied by 0.8 where the signs are equal (the last step overshot: slow down), never below 0.01. on the first iteration the update is 0, so every gain grows to 1.2.
- **cost**: exact t-SNE is $O(n^2)$ per iteration in time and memory. a few thousand points are fine on a laptop; for more, Barnes–Hut and FFT-accelerated approximations bring it to $O(n\log n)$ (going deeper).

call `tsne_kl_gradient(..., kl=False)` inside the loop; after `n_iters` iterations, store `embedding_` and `kl_divergence_`, the KL with the *un*-exaggerated $P$.

then, how to read the result. t-SNE preserves who is near whom, at the scale set by the perplexity. it does **not** preserve cluster sizes (every cluster gets a similar effective number of neighbours, whatever its spread), distances between clusters (the KL barely penalizes them), or densities. and with a small perplexity it finds structure in noise (warm-up 5). the lab shows each of these.

> **🔮 predict:** two Gaussian clusters with the same number of points, one ten times wider than the other. in the t-SNE plot, how different do their sizes look?

> **▶ build: stage 6.** implement `TSNE(n_components=2, *, perplexity=30.0, lr=200.0, n_iters=1000, early_exaggeration=12.0, rng=None)` with `fit` and `fit_transform`, exactly as above.

## worked example

**MDS.** three points with pairwise distances 3, 4 and 5 (from $(0,0)$, $(3,0)$, $(0,4)$, but MDS only sees the distances). the squared distances are 9, 16 and 25; double centring gives the Gram matrix $B$ of the centred triangle, whose two nonzero eigenvalues sum to $\operatorname{tr}B = \sum_i\|x_i - \bar x\|^2 = \frac1n\sum_{i<j}D_{ij}^2 = (9 + 16 + 25)/3 = 16.67$. the 2-D embedding is a 3-4-5 triangle again, centred at the origin, in some orientation; the tests check its side lengths and the eigenvalue sum.

**t-SNE with two points.** with $n = 2$, $p_{12} = p_{21} = \frac12$, and whatever $y_1, y_2$ are, $q_{12} = q_{21} = \frac12$ too. so $\mathrm{KL} = 0$ and the gradient is 0: two points carry no information about distance. the tests check this, and check your gradient against finite differences on 24 points.

## on paper

1. **double centring.** show that $-\frac12 JD^{(2)}J = X_cX_c^\top$ when $D^{(2)}_{ij} = \|x_i - x_j\|^2$.

<details><summary>solution</summary>

write $D^{(2)} = a\mathbf 1^\top + \mathbf 1a^\top - 2XX^\top$ with $a_i = \|x_i\|^2$. since $J\mathbf 1 = 0$, $J(a\mathbf 1^\top)J = 0$ and $J(\mathbf 1a^\top)J = 0$. so $-\frac12 JD^{(2)}J = JXX^\top J = (JX)(JX)^\top = X_cX_c^\top$, because $JX$ subtracts the column means. (the same argument with $X$ replaced by $\phi(X)$ is kernel PCA's centring.)

</details>

2. **the t-SNE gradient.** derive $\partial C/\partial y_i$. hint: write $C = \sum p_{ij}\log p_{ij} - \sum p_{ij}\log w_{ij} + \log\sum_{k \ne l}w_{kl}$ with $w_{ij} = (1 + d_{ij}^2)^{-1}$, $d_{ij} = \|y_i - y_j\|$, and use $\sum p_{ij} = 1$.

<details><summary>solution</summary>

only $d_{ij}$ with $i$ or $j$ equal to the moved point depend on $y_i$, and each unordered pair appears twice. $\partial w_{ij}/\partial y_i = -w_{ij}^2\cdot 2(y_i - y_j)$. the middle term gives $-\sum_j 2p_{ij}\frac{1}{w_{ij}}\partial w_{ij}/\partial y_i = \sum_j 4p_{ij}w_{ij}(y_i - y_j)$. the last term gives $\frac{1}{Z}\sum_j 2\,\partial w_{ij}/\partial y_i = -\sum_j 4\frac{w_{ij}}{Z}w_{ij}(y_i - y_j) = -\sum_j 4q_{ij}w_{ij}(y_i - y_j)$. together: $4\sum_j(p_{ij} - q_{ij})w_{ij}(y_i - y_j)$.

</details>

3. **perplexity.** show that a uniform distribution over $m$ points has perplexity $m$, and that the entropy of $p_{\cdot|i}$ decreases as $\beta_i$ increases (so bisection works).

<details><summary>solution</summary>

$H = -\sum_{j=1}^m\frac1m\log\frac1m = \log m$, so $e^H = m$. for the monotonicity: $p_j \propto e^{-\beta d_j}$ is a Gibbs distribution, and $dH/d\beta = -\beta\operatorname{Var}_p(d) \le 0$ (differentiate $H = \beta\mathbb E_p[d] + \log Z$ using $d\log Z/d\beta = -\mathbb E_p[d]$ and $d\mathbb E_p[d]/d\beta = -\operatorname{Var}_p(d)$). so $H$ falls from $\log(n - 1)$ at $\beta = 0$ towards $\log(\text{number of nearest ties})$ as $\beta \to \infty$, and any target in between is hit exactly once.

</details>

4. **why the heavy tail opens gaps.** suppose two points should have $q_{ij}$ equal to some small value $\epsilon$ (relative to the normalization). compare the distance needed with a Gaussian kernel $e^{-d^2}$ and with the Cauchy kernel $1/(1 + d^2)$.

<details><summary>solution</summary>

Gaussian: $e^{-d^2} = \epsilon$ gives $d = \sqrt{\log(1/\epsilon)}$, which grows very slowly: going from $\epsilon = 0.1$ to $\epsilon = 0.001$ moves $d$ only from 1.5 to 2.6, so the map squeezes moderate and large distances together. Cauchy: $1/(1 + d^2) = \epsilon$ gives $d = \sqrt{1/\epsilon - 1} \approx 1/\sqrt\epsilon$, which grows fast: the same change moves $d$ from 3 to 31.6. points with small affinity are placed far apart, which is why t-SNE clusters have clear gaps, and also why the size of those gaps means little.

</details>

## experiment

`lab.py` runs your MDS, Isomap and t-SNE:

1. t-SNE on clustered data at perplexities 2, 10, 30 and 100.
2. what t-SNE distances and sizes say: clusters of different spread, and clusters at different distances.
3. MDS, Isomap and t-SNE on the swiss roll.

## check your understanding

1. classical MDS on Euclidean distances and PCA give the same embedding. why might you still prefer one over the other?

<details><summary>answer</summary>

MDS works from the $n \times n$ distance matrix ($O(n^2)$ memory, $O(n^3)$ eigendecomposition), PCA from the $n \times d$ data ($O(nd\min(n, d))$). PCA wins when $n \gg d$ and gives a map for new points; MDS is what you use when you only *have* distances, or when they aren't Euclidean (then it's no longer PCA).

</details>

2. why does Isomap fail when the manifold has a hole (say, a swiss roll with a rectangle cut out)?

<details><summary>answer</summary>

geodesics must walk around the hole, so distances between points on opposite sides are longer than in the flat parametrization. no flat configuration reproduces them, so MDS distorts the embedding (the region near the hole bulges). Isomap assumes the manifold is isometric to a *convex* region of the plane.

</details>

3. what does the perplexity control, and what are the symptoms of setting it too low or too high?

<details><summary>answer</summary>

the effective number of neighbours each point tries to keep. too low: every point cares about just a few neighbours, so the embedding fragments into many small clumps, even within one real cluster or in pure noise. too high (close to $n$): everything is everyone's neighbour, local structure blurs, clusters merge, and the cost grows. it must be below $n - 1$.

</details>

4. why must the KL be computed with the *un*-exaggerated $P$ for `kl_divergence_`?

<details><summary>answer</summary>

exaggeration is an optimization device, not part of the objective: $12P$ isn't even a probability distribution, and its "KL" isn't comparable across runs. reporting the cost of the actual objective lets you compare runs, seeds and settings.

</details>

5. two t-SNE runs with different seeds give different-looking plots. which differences matter?

<details><summary>answer</summary>

rotations, reflections and translations never matter (the objective is invariant to them). the arrangement of clusters relative to each other can change between runs, because t-SNE barely constrains large distances. what should be stable is the membership: which points share a cluster and who neighbours whom locally. if *that* changes between seeds, the structure is not robust.

</details>

6. is t-SNE a good preprocessing step for clustering?

<details><summary>answer</summary>

with care. t-SNE makes clusters look more separated than they are and can split or invent clusters, so clustering its output can confirm whatever it shows. it also has no `transform` for new points. clustering in a PCA space and using t-SNE only to look is the safer default; the capstone compares the two.

</details>

## common bugs

- **squaring distances twice, or not at all.** MDS needs $D^{(2)}$; feeding $D$ into the double centring gives a distorted embedding. symptom: the triangle test fails while the shapes look "almost" right.
- **an asymmetric $k$-NN graph.** symptom: different geodesic distances from $i$ to $j$ and from $j$ to $i$, and a mismatch with scikit-learn.
- **perplexity in bits.** using $2^H$ with $H$ in nats (or $\log_2$ with $e^H$). symptom: every row's effective number of neighbours is off by a constant factor.
- **forgetting the zero diagonal** in $P$ or $Q$. symptom: KL dominated by self-pairs; points that don't move.
- **leaving early exaggeration on.** symptom: a final embedding with tiny, super-tight clusters and a KL that's far too large.
- **a gradient sign error.** symptom: the KL *increases*; points fly apart. the finite-difference test catches it.

## going deeper

- Torgerson (1952), "Multidimensional scaling: I. Theory and method", *Psychometrika* 17. [doi:10.1007/BF02288916](https://doi.org/10.1007/BF02288916)
- Tenenbaum, de Silva & Langford (2000), "A global geometric framework for nonlinear dimensionality reduction", *Science* 290: Isomap. [doi:10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)
- Roweis & Saul (2000), "Nonlinear dimensionality reduction by locally linear embedding", *Science* 290: LLE, the other classic. [doi:10.1126/science.290.5500.2323](https://doi.org/10.1126/science.290.5500.2323)
- van der Maaten & Hinton (2008), "Visualizing data using t-SNE", *JMLR* 9. [jmlr.org/papers/v9/vandermaaten08a.html](https://jmlr.org/papers/v9/vandermaaten08a.html)
- van der Maaten (2014), "Accelerating t-SNE using tree-based algorithms", *JMLR* 15: Barnes–Hut t-SNE. [jmlr.org/papers/v15/vandermaaten14a.html](https://jmlr.org/papers/v15/vandermaaten14a.html)
- Wattenberg, Viégas & Johnson (2016), "How to use t-SNE effectively", *Distill*. [doi:10.23915/distill.00002](https://doi.org/10.23915/distill.00002)
- McInnes, Healy & Melville (2018), "UMAP: uniform manifold approximation and projection for dimension reduction". [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
- Kobak & Berens (2019), "The art of using t-SNE for single-cell transcriptomics", *Nature Communications* 10: initialization, perplexity and exaggeration in practice. [doi:10.1038/s41467-019-13056-x](https://doi.org/10.1038/s41467-019-13056-x)

## connections

- ← builds on: PCA and kernel centring (topic 03), pairwise distances (ch 00), entropy and KL (ch 05), momentum (ch 06), gradient checking (ch 02).
- → used in: spectral clustering's neighbourhood graphs (topic 05), the capstone's t-SNE map of MNIST, visualizing learned representations of neural networks (ch 15 onward) and of self-supervised embeddings (ch 25), graph shortest paths in graph neural networks (ch 20).
