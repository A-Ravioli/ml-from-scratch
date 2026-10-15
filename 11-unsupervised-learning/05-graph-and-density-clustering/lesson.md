# graph & density clustering

## why this matters

k-means and Gaussian mixtures describe each cluster by a centre (and a shape around it), so they find round or elliptical blobs. topic 01's lab showed the cost: two interlocking moons have a *higher* k-means objective than k-means's own wrong answer. many real groupings aren't blobs. galaxies in a sky survey form filaments; GPS traces form roads; customers form chains of gradually varying behaviour; and real data come with noise points that belong to no cluster at all.

this topic builds three families that define a cluster differently. **spectral clustering** says a cluster is a part of a similarity graph with few edges leaving it, and finds it with eigenvectors. **DBSCAN** says a cluster is a dense region separated by sparse ones, and labels the sparse points as noise. **hierarchical (agglomerative) clustering** builds the whole tree of merges and lets you cut it at any level. and since you'll want to compare them honestly, you start by building the two standard scores for comparing a clustering with ground truth (or with another clustering), the **adjusted Rand index** and **normalized mutual information**, which the capstone uses on MNIST.

## warm-up: predict first

1. a clustering gives the labels $(1, 1, 0, 0)$ and the truth is $(0, 0, 1, 1)$. is that a perfect clustering? what's its accuracy if you compare labels directly?
2. you assign 1,000 points to 10 clusters uniformly at random and compute the (unadjusted) Rand index against the true labels: the fraction of pairs on which the two agree ("same cluster" or "different clusters"). is it close to 0, 0.5, or something else?
3. a similarity graph consists of three pieces with no edges between them. how many eigenvalues of its Laplacian are exactly 0?
4. DBSCAN with a fixed `eps` on two clusters, one dense and one ten times sparser. what can go wrong?
5. single linkage merges the two clusters with the closest pair of points. two big round clusters are joined by a thin line of five points. what happens?

## prerequisites check

- [ ] you've built `KMeans` (→ topic 01)
- [ ] you know eigenvectors of symmetric matrices and the Rayleigh quotient (→ ch 02 · topic 05)
- [ ] you know entropy and mutual information (→ ch 05 · topic 05)
- [ ] you've built a $k$-nearest-neighbour graph (→ topic 04, `geodesic_distances`)
- [ ] you can count pairs: $\binom{m}{2} = m(m-1)/2$ (→ ch 01)

## core ideas

### idea 1: comparing two clusterings

cluster labels are arbitrary names: $(1, 1, 0, 0)$ and $(0, 0, 1, 1)$ describe the same partition (warm-up 1). so you can't compare clusterings label by label; you compare *partitions*. everything starts from the **contingency table** $C_{ij}$: the number of points with the $i$-th true label and the $j$-th predicted label (labels in sorted order). its row sums $a_i$ are the true cluster sizes, its column sums $b_j$ the predicted ones.

**pair counting.** look at all $\binom n2$ pairs of points. a pair is "together" in a clustering if it shares a cluster. the **Rand index** is the fraction of pairs on which the two clusterings agree. its weakness: with many clusters, most pairs are "apart" in both, so even random labelings score high (warm-up 2: about 0.82 for 10 random clusters against 10 true ones). the **adjusted Rand index** (Hubert & Arabie, 1985) subtracts what you'd expect by chance and rescales:

$$\mathrm{ARI} = \frac{\sum_{ij}\binom{C_{ij}}{2} - E}{\tfrac12\left[\sum_i\binom{a_i}{2} + \sum_j\binom{b_j}{2}\right] - E}, \qquad E = \frac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom n2} .$$

the numerator counts pairs together in *both* clusterings, minus the expected number if the predicted labels were randomly permuted (keeping the cluster sizes); the denominator is the same for a perfect match. ARI is 1 for identical partitions, about 0 for random ones, and can be negative. when both clusterings put everything in one cluster, numerator and denominator are both 0; call that 1.

**information.** the contingency table divided by $n$ is a joint distribution of (true label, predicted label). their **mutual information** $I(U; V) = \sum_{ij}p_{ij}\log\frac{p_{ij}}{p_i\,q_j}$ measures how much knowing one tells you about the other. normalized by the average entropy,

$$\mathrm{NMI} = \frac{I(U; V)}{\tfrac12(H(U) + H(V))} \in [0, 1],$$

it's 1 for identical partitions and 0 for independent ones (define it as 1 when both have a single cluster). unlike ARI it isn't adjusted for chance, so random labelings with many clusters score a little above 0.

> **▶ build: stage 1.** implement `contingency_matrix(labels_true, labels_pred)`, `adjusted_rand_score(labels_true, labels_pred)` and `normalized_mutual_info_score(labels_true, labels_pred)`, for labels of any type.

### idea 2: spectral clustering

build a **similarity graph** on the data: weights $w_{ij} \ge 0$, large for similar points, with $w_{ii} = 0$. two common choices: the Gaussian (RBF) affinity $w_{ij} = e^{-\gamma\|x_i - x_j\|^2}$, or a $k$-nearest-neighbour graph ($a_{ij} = 1$ if $j$ is among $i$'s $k$ nearest neighbours, symmetrized as $w = (a + a^\top)/2$). a good cluster is a set of vertices with heavy edges inside and few edges out: a small *cut*.

minimizing a cut directly is combinatorial (and the plain minimum cut just snips off one point). the **graph Laplacian** turns it into linear algebra. with degrees $d_i = \sum_jw_{ij}$ and $D = \operatorname{diag}(d)$, $L = D - W$ satisfies

$$f^\top Lf = \tfrac12\sum_{i,j}w_{ij}(f_i - f_j)^2$$

for any vector $f$ (on paper 1). a vector that's constant on each connected piece of the graph has $f^\top Lf = 0$, so the number of zero eigenvalues of $L$ equals the number of connected components (warm-up 3), and the indicator vectors of the components span its null space. when the clusters are only *nearly* disconnected, the bottom eigenvectors are nearly piecewise constant: they still encode the clusters. finding the $k$ smallest eigenvectors is the continuous relaxation of the **normalized cut** problem (Shi & Malik, 2000), which balances the cut against the cluster sizes.

the algorithm of **Ng, Jordan & Weiss** (2002), the one you build:

1. $W$, then the symmetric normalized Laplacian $L_{\text{sym}} = I - D^{-1/2}WD^{-1/2}$.
2. the eigenvectors of its $k$ *smallest* eigenvalues, the columns of $U$ $(n, k)$. (equivalently, the $k$ largest of $D^{-1/2}WD^{-1/2}$; `np.linalg.eigh` gives them in ascending order.)
3. normalize each row of $U$ to unit length: the embedding. points in the same cluster land near the same point on the unit sphere.
4. k-means (your `KMeans`, with 10 restarts and the estimator's `rng`) on the rows.

the geometry of the original space only enters through $W$, so spectral clustering separates moons and rings as long as the graph connects points *within* each shape more than across. the choice of graph ($\gamma$, or $k$) is the main knob: too local, and the graph falls into extra pieces that steal eigenvectors; too global, and the cut between shapes stops being small.

> **🔮 predict:** spectral clustering on two moons with a 15-nearest-neighbour graph, then with an RBF graph with $\gamma = 1$. which one separates the moons?

> **▶ build: stage 2.** implement `affinity_matrix(X, affinity="rbf", gamma=1.0, n_neighbors=10)` (zero diagonal) and `SpectralClustering(n_clusters=2, *, affinity="rbf", gamma=1.0, n_neighbors=10, rng=None)` with `fit` (`affinity_matrix_`, `embedding_`, `labels_`) and `fit_predict`.

### idea 3: DBSCAN: clusters as dense regions

**DBSCAN** (Ester, Kriegel, Sander & Xu, 1996) defines clusters by density, with two parameters: a radius `eps` and a count `min_samples`.

- a **core point** has at least `min_samples` points within distance `eps`, *counting itself*.
- two points are connected if one is a core point and the other is within `eps` of it. a **cluster** is everything reachable from a core point through chains of core points, plus the non-core points (**border points**) within `eps` of those core points.
- points in no cluster are **noise**, labelled $-1$.

the algorithm is a graph search: for each point in index order, skip it if it's already labelled or isn't core; otherwise start a new cluster there and expand it with a stack (or queue): label each unlabelled neighbour, and push it if it's a core point, so the expansion continues through it. the order matters for one detail only: a border point near two clusters goes to whichever reached it first, so scanning in index order (as scikit-learn does) makes the result reproducible. mark points as labelled when you push them, or a point can be pushed many times, and with a careless loop condition, forever.

DBSCAN finds clusters of any shape, needs no $k$, and handles noise. its weakness is a single global density threshold (warm-up 4): with one dense and one sparse cluster, an `eps` small enough to separate dense neighbours dissolves the sparse cluster into noise, and an `eps` large enough for the sparse one merges dense clusters that are close. a practical way to pick `eps` is the **k-distance plot**: sort every point's distance to its `min_samples`-th nearest neighbour and look for the knee. (HDBSCAN, in going deeper, removes the single threshold.)

> **🔮 predict:** three blobs whose standard deviations are 0.3, 0.8 and 1.6. you try every `eps` from tiny to large. what's the best ARI DBSCAN reaches, and what happens just above the best `eps`?

> **▶ build: stage 3.** implement `DBSCAN(eps=0.5, *, min_samples=5)` with `fit` (`labels_`, `core_sample_indices_`) and `fit_predict`, matching scikit-learn's labels exactly.

### idea 4: agglomerative clustering

start with every point as its own cluster. repeatedly merge the two closest clusters, until one remains. the record of merges is a binary tree (a **dendrogram**), and cutting it where $k$ clusters remain gives a flat clustering for every $k$ at once. what "closest clusters" means is the **linkage**:

| linkage | distance between clusters $A$ and $B$ | tends to produce |
|---|---|---|
| single | $\min_{a \in A, b \in B}\|a - b\|$ | long chains; follows any thin bridge (warm-up 5) |
| complete | $\max_{a, b}\|a - b\|$ | compact clusters of similar diameter |
| average | mean of all $\|a - b\|$ | in between |
| Ward | increase in within-cluster sum of squares from merging | round, similar-sized clusters, like k-means |

you never need to recompute cluster distances from the points. after merging $i$ and $j$, the distance from the new cluster to any other cluster $k$ follows from the old ones by the **Lance–Williams** update:

$$\text{single: } \min(d_{ik}, d_{jk}), \quad \text{complete: } \max(d_{ik}, d_{jk}), \quad \text{average: } \frac{n_id_{ik} + n_jd_{jk}}{n_i + n_j},$$

$$\text{Ward (on squared distances): } \frac{(n_i + n_k)d_{ik} + (n_j + n_k)d_{jk} - n_kd_{ij}}{n_i + n_j + n_k}$$

(on paper 3). so the algorithm keeps an $n \times n$ distance matrix with $\infty$ on the diagonal: find its smallest entry $(i, j)$, record the merge, overwrite row and column $i$ with the updated distances, retire row and column $j$ (set them to $\infty$). each step is one `argmin` over the matrix and one vector update: $O(n^2)$ per merge, $O(n^3)$ in total, fine for a few thousand points.

record the tree like scikit-learn: merge number $t$ (from 0) creates node $n + t$; `children_[t]` holds the two node ids it merged, smaller first; `distances_[t]` its height (for Ward, the square root of the Lance–Williams value, so heights are on the scale of distances, matching scipy). `labels_` comes from the state after $n - k$ merges, numbered by each cluster's smallest point index.

> **▶ build: stage 4.** implement `AgglomerativeClustering(n_clusters=2, *, linkage="single")` with `fit` (`children_`, `distances_`, `labels_`) and `fit_predict`.

## worked example

**ARI.** truth $(0, 0, 0, 1, 1, 1)$, prediction $(0, 0, 1, 1, 2, 2)$. the contingency table is $\begin{pmatrix}2 & 1 & 0\\ 0 & 1 & 2\end{pmatrix}$. pairs together in both: $\binom22 + \binom22 = 2$. pairs together in the truth: $\binom32 + \binom32 = 6$; in the prediction: $\binom22 + \binom22 + \binom22 = 3$; all pairs: $\binom62 = 15$. so $E = 6 \cdot 3/15 = 1.2$ and $\mathrm{ARI} = (2 - 1.2)/(4.5 - 1.2) = 0.242$.

**DBSCAN.** points $0, 1, 2, 3, 10, 11, 12, 30$ on a line, `eps = 1`, `min_samples = 3`. neighbourhood sizes (counting the point itself): 2, 3, 3, 2, 2, 3, 2, 1, so the core points are 1, 2 and 11. scanning in order: point 0 isn't core, skip; point 1 is core and unlabelled, so cluster 0 starts there and reaches 0 and 2, then (through core 2) 3. points 0 and 3 are border points. point 11 starts cluster 1 and picks up 10 and 12. point 30 is noise. labels $(0, 0, 0, 0, 1, 1, 1, -1)$, core indices $(1, 2, 5)$.

**single linkage** on $0, 1, 5, 7, 20$ (indices 0–4): merge $\{0\}, \{1\}$ at height 1 (node 5); $\{5\}, \{7\}$ at 2 (node 6); nodes 5 and 6 at $|1 - 5| = 4$ (node 7); then 20 at $|7 - 20| = 13$. so `children_` $= [(0, 1), (2, 3), (5, 6), (4, 7)]$ and `distances_` $= (1, 2, 4, 13)$. with complete linkage the third merge happens at $|0 - 7| = 7$ and the last at 20. the tests check all three examples.

## on paper

1. **the Laplacian's quadratic form.** show $f^\top(D - W)f = \frac12\sum_{i,j}w_{ij}(f_i - f_j)^2$, and conclude that $L$ is positive semidefinite and that the indicator vector of each connected component is in its null space.

<details><summary>solution</summary>

$f^\top Df = \sum_id_if_i^2 = \sum_{i,j}w_{ij}f_i^2$ and $f^\top Wf = \sum_{i,j}w_{ij}f_if_j$. so $f^\top Lf = \sum_{i,j}w_{ij}(f_i^2 - f_if_j) = \frac12\sum_{i,j}w_{ij}(f_i^2 - 2f_if_j + f_j^2)$, using the symmetry of $w$ to split $f_i^2$ evenly between the two orderings. every term is $\ge 0$, so $L \succeq 0$. for the indicator $f = \mathbf 1_A$ of a component $A$, $f_i \ne f_j$ only for pairs with one end in $A$ and one outside, and those have $w_{ij} = 0$: $f^\top Lf = 0$, and for a PSD matrix that means $Lf = 0$.

</details>

2. **chance agreement.** 1,000 points, true labels in 10 equal groups, predicted labels assigned uniformly at random into 10 groups. estimate the Rand index, and explain why the ARI is about 0.

<details><summary>solution</summary>

a random pair is together in the truth with probability about $0.1$ and in the prediction with probability $0.1$, independently. they agree if both say together ($0.01$) or both say apart ($0.81$): Rand index $\approx 0.82$, though the labels carry no information. the ARI subtracts exactly this expected agreement on the "together" pairs (its $E$ term, here $\approx 0.1 \cdot 0.1 \cdot \binom n2$) and rescales, so its expectation under random labelings is 0.

</details>

3. **Lance–Williams for Ward.** with Ward's distance between clusters $d(A, B) = \frac{2n_An_B}{n_A + n_B}\|\mu_A - \mu_B\|^2$ (twice the increase in the within-cluster sum of squares; for two single points it's $\|a - b\|^2$), verify the update for $d(I \cup J, K)$ when $I$, $J$, $K$ are single points.

<details><summary>solution</summary>

with $n_I = n_J = n_K = 1$, the merged cluster has mean $m = (x_I + x_J)/2$ and $d(I \cup J, K) = \frac{2 \cdot 2 \cdot 1}{3}\|m - x_K\|^2 = \frac43\|m - x_K\|^2$. the right side is $\frac{2d_{IK} + 2d_{JK} - d_{IJ}}{3}$ with $d$ the squared distances. by the parallelogram (median) identity, $\|x_I - x_K\|^2 + \|x_J - x_K\|^2 = 2\|m - x_K\|^2 + \frac12\|x_I - x_J\|^2$, so $2d_{IK} + 2d_{JK} - d_{IJ} = 4\|m - x_K\|^2$, and dividing by 3 gives the left side.

</details>

4. **border points depend on order.** construct points on a line where the DBSCAN label of one point depends on the order in which the points are scanned.

<details><summary>solution</summary>

take `eps = 1`, `min_samples = 4` and the points $-1, -0.5, 0, 1, 2, 2.5, 3$. counting each point itself, the neighbourhoods have sizes 3, 3, 4, 3, 4, 3, 3: only $0$ and $2$ are core points, and they're 2 apart, so they start two different clusters. the point $1$ is within `eps` of both but isn't core itself, so it's a border point of both clusters, and it joins whichever cluster is expanded first: the left one if the points are stored left to right, the right one if they're stored in reverse. border points are the only order-dependent part of DBSCAN, which is why the course (like scikit-learn) fixes the order: scan by index.

</details>

## experiment

`lab.py` puts every clustering method of the chapter side by side:

1. k-means, GMM, spectral clustering, DBSCAN, single and Ward linkage on moons, rings, blobs of different density, and stretched blobs, scored by ARI.
2. DBSCAN's sensitivity to `eps`, and the k-distance plot.
3. single versus Ward linkage when two clusters are joined by a thin bridge.

## check your understanding

1. why is plain accuracy the wrong way to score a clustering, and how could you rescue it?

<details><summary>answer</summary>

cluster labels are arbitrary, so $(1, 1, 0, 0)$ against $(0, 0, 1, 1)$ has accuracy 0 though the partition is perfect. you can rescue it by matching each cluster to a label first: the best one-to-one matching (the Hungarian algorithm), or "purity" (each cluster votes for its majority label, which allows many clusters per label). ARI and NMI avoid the matching by comparing partitions directly.

</details>

2. ARI can be negative. what does that mean?

<details><summary>answer</summary>

the two clusterings agree on fewer "together" pairs than random labelings with the same cluster sizes would: they're systematically anti-aligned, e.g. a clustering that deliberately splits every true cluster across all predicted ones. in practice small negative values are just noise around 0.

</details>

3. spectral clustering's graph accidentally has 3 connected components and you ask for $k = 2$. what goes wrong?

<details><summary>answer</summary>

$L_{\text{sym}}$ has three zero eigenvalues, so the "two smallest" eigenvectors are an arbitrary basis of part of a three-dimensional null space. the embedding may merge two components and ignore the split you wanted, or isolate a tiny component. connect the graph (more neighbours, a wider $\gamma$) or ask for $k = 3$. (scikit-learn warns: "graph is not fully connected".)

</details>

4. why does DBSCAN not need $k$, and what does it need instead?

<details><summary>answer</summary>

clusters emerge as the connected regions of the core-point graph, however many there are. instead you choose what "dense" means: `eps` and `min_samples`, which in effect set a density threshold. that's a scale, and it's often harder to choose than $k$ (the lab's second experiment).

</details>

5. Ward linkage and k-means optimize similar objectives. why can their answers still differ?

<details><summary>answer</summary>

Ward is greedy: each merge minimizes the *increase* in within-cluster sum of squares, and merges are never undone, so the final $k$-cluster solution is the result of a sequence of locally optimal steps, not a minimizer of the $k$-cluster objective. k-means (Lloyd) moves points between clusters freely but gets stuck in local minima of its own. a common trick is to initialize k-means from Ward's clusters.

</details>

6. what does a dendrogram give you that a flat clustering doesn't?

<details><summary>answer</summary>

every clustering from $n$ down to 1 cluster, nested, with the heights at which groups merge. long vertical gaps suggest natural numbers of clusters; the structure shows sub-clusters within clusters (like topic 01's pairs of blobs), which a single $k$ hides.

</details>

## common bugs

- **comparing labels directly instead of partitions.** symptom: a perfect clustering scored as 0% because its labels are permuted.
- **ARI/NMI with labels that aren't $0..k-1$.** indexing a table with raw labels like $-1$ or strings. symptom: index errors, or a silently wrong table. map labels with `np.unique(..., return_inverse=True)`.
- **the largest instead of the smallest eigenvectors of $L$** (or of the wrong matrix). symptom: spectral clustering no better than random.
- **forgetting the zero diagonal of $W$**, or the row normalization. symptom: degraded clusters, especially with uneven degrees.
- **DBSCAN counting neighbours without the point itself.** symptom: different core points from scikit-learn.
- **DBSCAN pushing already-labelled points.** symptom: exponential slowdown or a loop that never ends.
- **merging using stale distances in agglomerative clustering** (forgetting to retire row and column $j$). symptom: the same cluster merged twice, a tree with too many nodes.

## going deeper

- von Luxburg (2007), "A tutorial on spectral clustering", *Statistics and Computing* 17. the clearest introduction to graph Laplacians and cuts. [arXiv:0711.0189](https://arxiv.org/abs/0711.0189)
- Ng, Jordan & Weiss (2002), "On spectral clustering: analysis and an algorithm", *NeurIPS*. [papers.nips.cc/paper/2092](https://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm)
- Shi & Malik (2000), "Normalized cuts and image segmentation", *IEEE TPAMI* 22(8). [doi:10.1109/34.868688](https://doi.org/10.1109/34.868688)
- Ester, Kriegel, Sander & Xu (1996), "A density-based algorithm for discovering clusters in large spatial databases with noise", *KDD*. [aaai.org/Library/KDD/1996/kdd96-037.php](https://www.aaai.org/Library/KDD/1996/kdd96-037.php)
- Campello, Moulavi & Sander (2013), "Density-based clustering based on hierarchical density estimates", *PAKDD*: HDBSCAN. [doi:10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14)
- Hubert & Arabie (1985), "Comparing partitions", *Journal of Classification* 2: the adjusted Rand index. [doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075)
- Vinh, Epps & Bailey (2010), "Information theoretic measures for clusterings comparison", *JMLR* 11: NMI variants and adjusted mutual information. [jmlr.org/papers/v11/vinh10a.html](https://jmlr.org/papers/v11/vinh10a.html)
- Müllner (2011), "Modern hierarchical, agglomerative clustering algorithms". [arXiv:1109.2378](https://arxiv.org/abs/1109.2378)

## connections

- ← builds on: `KMeans` (topic 01), nearest-neighbour graphs (topic 04), eigenvectors (ch 02), mutual information (ch 05).
- → used in: the capstone's evaluation of MNIST clusterings (ARI, NMI); graph Laplacians in graph neural networks (ch 20); hierarchical structure in topic models and in the analysis of learned embeddings (ch 25).
