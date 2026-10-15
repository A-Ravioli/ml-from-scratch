# 11 · unsupervised learning · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by a topic's own tests.

## 1. concept check

1. Lloyd's algorithm alternates two steps. why does neither step increase the inertia, and why does that guarantee termination but not the global minimum?

<details><summary>answer</summary>

the assignment step gives each point its nearest centre, which minimizes its own term; the update step moves each centre to its cluster's mean, which minimizes the cluster's sum of squares. so the inertia never increases; there are finitely many labelings and none can repeat after a strict decrease, so it stops. it stops at a fixed point of the two steps, which can be a poor local minimum (two centres in one blob, none in another). (topic 01, idea 1 and on paper 2)

</details>

2. k-means++ samples each new centre with probability proportional to $D(x)^2$. why squared, and what does it buy you?

<details><summary>answer</summary>

$D(x)^2$ is each point's current contribution to the inertia, so the sampling targets the points the current centres explain worst; points already covered can't be picked again. the seeding alone has expected inertia within $8(\ln k + 2)$ of optimal (Arthur & Vassilvitskii), and in practice far fewer runs end in bad local minima than with uniform seeding. it's not a guarantee for one run, so restarts still matter. (topic 01, idea 2 and lab)

</details>

3. why can't the silhouette (or any internal score) settle the "right" number of clusters once and for all?

<details><summary>answer</summary>

data can have structure at several scales (pairs of blobs, each pair far from the others), and each score prefers some trade-off between compactness and separation. the silhouette rewards the ratio of separation to compactness and picked 2 clusters when there were 4 visible blobs in two pairs; the elbow showed 4. the useful $k$ depends on what the clusters are for. (topic 01, idea 4 and lab)

</details>

4. write the E-step and M-step of EM for a Gaussian mixture.

<details><summary>answer</summary>

E: $r_{ij} = \pi_j\mathcal N(x_i \mid \mu_j, \Sigma_j)/\sum_l\pi_l\mathcal N(x_i \mid \mu_l, \Sigma_l)$, computed in log space with logsumexp. M: $N_j = \sum_ir_{ij}$, $\pi_j = N_j/n$, $\mu_j = \sum_ir_{ij}x_i/N_j$, $\Sigma_j = \sum_ir_{ij}(x_i - \mu_j)(x_i - \mu_j)^\top/N_j$ (plus `reg_covar` on the diagonal). (topic 02, ideas 2–3)

</details>

5. why does EM never decrease the log-likelihood? what does that say, and not say, about the final answer?

<details><summary>answer</summary>

$\log p(x \mid \theta) = \mathcal L(q, \theta) + \mathrm{KL}(q\,\|\,p(z \mid x, \theta))$. the E-step sets $q$ to the posterior, making the KL zero so the bound touches the likelihood; the M-step raises the bound; the likelihood, above the bound, rises at least as much. it says each iteration is no worse than the last; it doesn't say the end point is the maximum (local maxima, saddles, plateaus that fool a `tol` stopping rule). (topic 02, idea 4 and lab)

</details>

6. the likelihood of a GMM can be made infinite. how, and what's the standard guard?

<details><summary>answer</summary>

put a component's mean on one data point and shrink its variance: that point's density grows without bound while the other components keep every other point's density positive. the guard is a floor on the variances (`reg_covar` added to the diagonal); even so, a few duplicated points can capture a component and fool BIC. (topic 02, idea 3, on paper 4 and lab)

</details>

7. PCA's training reconstruction error with $k$ components equals what? why isn't there a similar identity for held-out data?

<details><summary>answer</summary>

$(n - 1)\sum_{j > k}\lambda_j$, the variance in the discarded directions (Eckart–Young, by Pythagoras along the orthogonal components). the components were chosen to fit the training points; a new point has variance in directions none of them span, so its error doesn't vanish even with every component (8.3% of a held-out Olivetti face's variance with all 320). (topic 03, idea 2 and lab)

</details>

8. what does probabilistic PCA add to PCA, and what is its noise variance at the maximum likelihood?

<details><summary>answer</summary>

a Gaussian latent-variable model $x = Wz + \mu + \varepsilon$, so a likelihood (for held-out model selection), a posterior over the latent coordinates, and samples. the MLE noise variance is the average of the discarded eigenvalues of $S = X_c^\top X_c/n$, and $W = U_k(\Lambda_k - \sigma^2I)^{1/2}$. (topic 03, idea 3)

</details>

9. how does kernel PCA embed a *new* point, and what goes wrong if you centre its kernel row with its own mean only?

<details><summary>answer</summary>

compute $k_x = (k(x, x_i))_i$, centre it with the training statistics ($k_x - \bar k_x - \bar K_{\text{col}} + \bar K$), and project: $\tilde k_x^\top\alpha_j/\sqrt{\lambda_j}$. centring differently subtracts the wrong feature-space mean, so new points are shifted relative to the training embedding; with a linear kernel the result no longer equals `PCA.transform`. (topic 03, idea 5)

</details>

10. how do classical MDS and Isomap differ, and why does Isomap unroll the swiss roll?

<details><summary>answer</summary>

classical MDS double-centres the squared distances into a Gram matrix and takes its top eigenvectors; on Euclidean distances it's PCA. Isomap first replaces distances by shortest paths on a $k$-nearest-neighbour graph, approximating distances along the surface; MDS on those recovers the flat sheet the roll was rolled from (as long as the graph has no short circuits between layers). (topic 04, ideas 1–3)

</details>

11. name three things you can't read from a t-SNE plot, and say why for each.

<details><summary>answer</summary>

cluster sizes (each point's bandwidth is tuned to a fixed number of effective neighbours, so tight and wide clusters look alike: spread ratio 9.6 in the data, 1.01 in the plot), distances between clusters (pairs with $p_{ij} \approx 0$ contribute almost nothing to the KL, so far-apart clusters are placed by repulsion alone), and densities. also: structure at a scale below the perplexity can be invented (clumps in pure noise at perplexity 2). (topic 04, idea 6 and lab)

</details>

12. why does spectral clustering find the two moons while k-means can't?

<details><summary>answer</summary>

spectral clustering only sees the similarity graph: with a nearest-neighbour graph, each moon is a well-connected piece with few edges to the other, so the bottom eigenvectors of the normalized Laplacian are nearly constant on each moon, and k-means on those embedded rows separates them easily. k-means in the original space can only draw straight (Voronoi) boundaries. (topic 05, idea 2)

</details>

13. what are core, border and noise points in DBSCAN, and why does a single `eps` struggle with clusters of different density?

<details><summary>answer</summary>

core points have at least `min_samples` points (themselves included) within `eps`; border points are within `eps` of a core point but aren't core; noise is everything else. one `eps` sets one density threshold: small enough for the dense cluster, it dissolves the sparse one into noise; large enough for the sparse one, it merges nearby clusters. in the lab the best `eps` sat right next to the value where everything merged. (topic 05, idea 3 and lab)

</details>

14. ARI versus NMI versus purity: which is adjusted for chance, and which can be gamed by using many clusters?

<details><summary>answer</summary>

ARI is adjusted for chance (expected 0 for random labelings). NMI is normalized but not adjusted, so random labelings with many clusters score a little above 0. purity is neither: one cluster per point gives purity 1, so it's only meaningful at a fixed, small $k$. (topic 05, idea 1; capstone)

</details>

15. why does a smaller `max_samples` often make an isolation forest *better*, and what does `contamination` change?

<details><summary>answer</summary>

small subsamples break up groups of anomalies (masking): a cluster of 100 anomalies among 1,100 points shields its members from isolation in big trees, but a 16-point subsample contains one or two of them, which are isolated at once (AUC 1.000 at 16 against 0.864 with all points). `contamination` doesn't touch the trees or the scores, only the threshold for `predict`: the ranking and AUC stay the same, precision and recall trade off. (topic 06, idea 4 and lab)

</details>

## 2. by hand (no computer)

1. **one Lloyd iteration and one EM iteration.** points $0, 2, 10, 12$ on a line, $k = 2$. (a) starting from centres $(0, 2)$, run Lloyd's algorithm to convergence and give the final inertia. (b) a GMM with means $(1, 11)$, variances 1 and equal weights: give the responsibility of the left component for $x = 2$ (to two significant figures) and argue that the next means are within 0.01 of $(1, 11)$.

<details><summary>solution</summary>

(a) assignment: $0 \to 0$, $2 \to 2$ (its own centre), $10, 12 \to 2$. update: $(0, 8)$. assignment: $0 \to 0$; $2$: distance 2 to 0, 6 to 8, so $\to 0$; $10, 12 \to 8$. update: $(1, 11)$. next assignment changes nothing: converged with inertia $1 + 1 + 1 + 1 = 4$. (b) $r = 1/(1 + e^{((2 - 1)^2 - (2 - 11)^2)/2}) = 1/(1 + e^{-40}) \approx 1.0$ (to two figures, and to 17). every point's responsibilities are essentially one-hot on its pair, so the M-step means are the pair means $(1, 11)$ up to $\sim e^{-40}$.

</details>

2. **PCA by hand.** the points $(2, 0), (-2, 0), (0, 1), (0, -1)$. give the explained variances (ddof = 1), the components (course sign convention), and the squared reconstruction error with one component.

<details><summary>solution</summary>

the mean is 0 and the coordinates are uncorrelated: $\sum x^2 = 8$, $\sum y^2 = 2$, $\sum xy = 0$. variances $8/3$ and $2/3$, components $(1, 0)$ and $(0, 1)$ (largest entries already positive), ratios 0.8 and 0.2. one component keeps the $x$ coordinates; the error is $\sum y^2 = 2 = (n - 1)\lambda_2 = 3 \cdot \frac23$.

</details>

3. **ARI and an isolation tree.** (a) truth $(0, 0, 1, 1)$, prediction $(0, 0, 0, 1)$: compute the ARI. (b) the points $0, 1, 2, 9$ on a line; one isolation tree without a depth limit. what's the probability that the first cut isolates $9$, and what is $\mathbb E[h(9)]$?

<details><summary>solution</summary>

(a) contingency $\begin{pmatrix}2 & 0\\ 1 & 1\end{pmatrix}$: $\sum\binom{C_{ij}}{2} = 1$; rows $\binom22 + \binom22 = 2$; columns $\binom32 + \binom12 = 3$; $\binom42 = 6$. $E = 2 \cdot 3/6 = 1$; $\mathrm{ARI} = (1 - 1)/(\frac12(2 + 3) - 1) = 0$: no better than chance. (b) the threshold is uniform on $(0, 9)$, and $9$ is alone on the right if it falls in $(2, 9)$: probability $7/9$. otherwise (probability $2/9$) the threshold is in $(0, 2)$ and the right part contains $9$ plus one or two of $\{1, 2\}$. if it's in $(1, 2)$ (probability 1/9), the right node is $\{2, 9\}$ and $9$ is isolated at depth 2. if it's in $(0, 1)$ (1/9), the right node is $\{1, 2, 9\}$ with range $(1, 9)$: the next cut isolates 9 with probability $7/8$ (depth 2), else the node $\{2, 9\}$ follows (depth 3). so $\mathbb E[h(9)] = \frac79 \cdot 1 + \frac19 \cdot 2 + \frac19(\frac78 \cdot 2 + \frac18 \cdot 3) = \frac{7}{9} + \frac{2}{9} + \frac{17}{72} = 1.236$.

</details>

## 3. blind rebuild (60 minutes, no notes)

rebuild **EM for a Gaussian mixture** from an empty notebook: copy `02-gaussian-mixtures-and-em/exercise.py` to a scratch name, delete every function and method body you wrote, and implement from memory:

- `gaussian_log_density(X, means, covariances, covariance_type="full")` for full, diagonal and spherical covariances (log space, Cholesky for full);
- `e_step(...)` → responsibilities and the mean log-likelihood, stable for far-away points;
- `m_step(X, resp, covariance_type, reg_covar)`;
- `GaussianMixture.fit`, `predict_proba`, `predict`, `score_samples`, `score`.

then run the topic's own checks on your rebuild (temporarily put it in place of `exercise.py`, or point the test at it):

```bash
pytest 11-unsupervised-learning/02-gaussian-mixtures-and-em/test_exercise.py -k "stage1 or stage2 or stage3 or stage4"
```

you pass if every check is green within the hour. if you finish early, add `sample` and `bic` (stage 5).

## 4. readiness

move on when you score at least 80% on sections 1 and 2 together (12 of 15 concepts and 2 of 3 by-hand problems, roughly) **and** your blind rebuild passes. the same checkpoint works as a placement test: if you pass it cold, skim the labs (the surprises are the point) and go on to chapter 12.
