# 11 · unsupervised learning

> find structure without labels.

most data arrive without labels, and the first question anyone asks of a pile of images, records or embeddings is "what's in here?": are there groups, how many, what are the main directions of variation, which points don't fit? this chapter builds the classical answers from scratch. you'll cluster with k-means and Gaussian mixtures (and meet EM, the algorithm behind hidden Markov models and variational autoencoders), compress with PCA in three guises, draw maps of high-dimensional data with MDS, Isomap and t-SNE, cluster shapes that aren't blobs with spectral clustering, DBSCAN and hierarchies, and hunt anomalies with isolation forests. along the way you'll be surprised by a k-means answer that beats the truth at its own objective, an EM run that "converges" on a saddle point and finds the real answer 500 iterations later, six duplicated points that fool BIC, a t-SNE plot that shows a ten-times-tighter cluster at exactly the same size, a single linkage clustering broken by a bridge of six points, and an anomaly detector that gets better when each tree sees *less* data.

**you will build:** Lloyd's algorithm, k-means++ seeding and the silhouette (`KMeans`, `kmeans_plus_plus`, `silhouette_score`); Gaussian log densities, the E- and M-steps and `GaussianMixture` with three covariance types, sampling and BIC; `PCA` (with whitening), `ProbabilisticPCA` in closed form, kernels and `KernelPCA`; classical `MDS`, geodesic distances and `Isomap`, perplexity-calibrated affinities, the t-SNE gradient and `TSNE`; the contingency table, `adjusted_rand_score` and `normalized_mutual_info_score`, `SpectralClustering`, `DBSCAN` and `AgglomerativeClustering` with four linkages; Gaussian and $k$-NN anomaly scores, isolation trees and `IsolationForest`. module: **`mlfs.unsupervised`** (topics 01–06). chapter 27's vector index partitions embeddings with your `KMeans`.
**prerequisites:** chapter 02 (eigendecompositions, the SVD and Eckart–Young, Cholesky solves), chapter 04 (the multivariate Gaussian), chapter 05 (maximum likelihood, entropy, KL divergence, mutual information), chapter 06 (momentum, local minima), chapter 07 (the `Estimator` / `Transformer` protocol, ROC and AUC), and `pairwise_distances`, `logsumexp` and `as_rng` from chapter 00. no neural networks: everything here is NumPy.
**time · scaffolding:** about 70 hours · L3 (independent): every function comes with its signature and a one-line docstring; the exact conventions (tie-breaking, sign conventions, the order of random draws, stopping rules) are in the lessons. hints have two levels. the capstone gives you the data loading and the experiment cells; you write the evaluation code and the analysis.
**newly unlocked library components:** after this chapter you may use scikit-learn's `KMeans`, `MiniBatchKMeans`, `GaussianMixture`, `PCA`, `IncrementalPCA`, `KernelPCA`, `MDS`, `Isomap`, `TSNE` (including Barnes–Hut), `SpectralClustering`, `DBSCAN`, `AgglomerativeClustering`, `IsolationForest` and the clustering metrics (`adjusted_rand_score`, `normalized_mutual_info_score`, `silhouette_score`) in later chapters' labs and experiments where speed or scale matters, and `scipy.cluster.hierarchy` for dendrograms. inside `mlfs` modules, later chapters use *your* implementations through `mlfs.unsupervised`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md): e.g. chapter 27's inverted-file index partitions embeddings with your `KMeans`. note the sign conventions where yours differ: your `IsolationForest.score_samples` is *higher = more anomalous* (scikit-learn's is the opposite), and your PCA components follow the course's sign rule (largest-magnitude entry positive).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [k-means](01-k-means/) | `assign_to_nearest`, `update_centers`, `kmeans_plus_plus`, `KMeans`, `silhouette_score` → `mlfs.unsupervised` | the true two-moon labeling has a *higher* inertia (252) than k-means's wrong answer (165); on 25 perfectly separated blobs one k-means++ run succeeds only 41% of the time (random starts: never); the silhouette picks 2 clusters when 4 blobs are clearly visible |
| 02 | [gaussian mixtures & EM](02-gaussian-mixtures-and-em/) | `gaussian_log_density`, `e_step`, `m_step`, `GaussianMixture` → `mlfs.unsupervised` | random starts "converge" after 7 iterations at the one-Gaussian saddle and jump to the real solution only after 200–600 more; only full covariances find tilted strips (ARI 0.99 against 0.02); six duplicated points make BIC choose 6 components instead of 4 |
| 03 | [dimensionality reduction](03-dimensionality-reduction/) | `PCA`, `ProbabilisticPCA`, `kernel_matrix`, `KernelPCA` → `mlfs.unsupervised` | 95% explained variance on Olivetti faces captures only 87% of a held-out face, and all 320 components only 92%; held-out reconstruction error can't pick the dimension (it always falls) while PPCA's held-out likelihood peaks at the true 5; the RBF width moves the ring-separating direction from component 3 to component 1 and then destroys it |
| 04 | [manifold learning](04-manifold-learning/) | `MDS`, `geodesic_distances`, `Isomap`, `conditional_probabilities`, `joint_probabilities`, `tsne_kl_gradient`, `TSNE` → `mlfs.unsupervised` | t-SNE finds dozens of "clusters" in pure noise at perplexity 2; a cluster ten times tighter than another is drawn at the same size (ratio 1.01), and a gap four times larger is drawn the same width; t-SNE keeps the swiss roll's neighbourhoods best of all three methods while tearing its global order |
| 05 | [graph & density clustering](05-graph-and-density-clustering/) | `contingency_matrix`, `adjusted_rand_score`, `normalized_mutual_info_score`, `affinity_matrix`, `SpectralClustering`, `DBSCAN`, `AgglomerativeClustering` → `mlfs.unsupervised` | no method wins on all four shapes; DBSCAN's best `eps` on mixed densities sits just below the value where everything merges (ARI 0.93 at eps 0.42, 0.00 at 0.435); six bridge points flip single linkage from perfect to useless while Ward doesn't notice |
| 06 | [anomaly detection](06-anomaly-detection/) | `gaussian_scores`, `knn_distance_scores`, `average_path_length`, `isolation_tree`, `path_lengths`, `IsolationForest` → `mlfs.unsupervised` | a wrong contamination rate leaves the AUC untouched (0.998 everywhere); an isolation forest on clustered anomalies gets *worse* with more data per tree (AUC 1.000 at 16, 0.864 with all 1,100), and the 5-NN score is worse than random (0.099); a single Gaussian ranks anomalies between two modes as the most normal points (AUC 0.012) |

## capstone

*exploring unlabeled MNIST*: cluster and embed 10,000 handwritten digits without labels, then measure how much digit structure you found. you compare k-means on raw pixels, on PCA scores and a diagonal GMM against a random baseline (ARI, NMI and purity with error bars over seeds), look at what each cluster contains, draw PCA and t-SNE maps (your exact t-SNE on 2,500 images), and measure how label-consistent clusters get as you use more of them than there are digits. the reference solution measures ARI 0.36 for k-means on PCA scores (random labels: 0.00; a diagonal GMM: 0.16), a t-SNE map in which 87% of each image's 10 nearest neighbours show the same digit, and cluster-then-label test accuracy rising from 0.59 with 10 clusters to 0.89 with 200. full config: about 4–6 minutes on a laptop; small config (offline): under a minute. see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (Lloyd and EM iterations, PCA of four points, an ARI and an isolation-tree expectation), and a 60-minute blind rebuild of EM for Gaussian mixtures.

## notation

| symbol | meaning |
|---|---|
| $X$, $x_i$, $n$, $d$ | data matrix $(n, d)$, its $i$-th row, number of points, number of features |
| $X_c$, $\bar x$ | centred data and the mean |
| $k$ | number of clusters, components or neighbours (the topic says which) |
| $z_i$, $\mu_j$, $J$ | cluster label of point $i$, centre of cluster $j$, inertia $\sum_i\|x_i - \mu_{z_i}\|^2$ |
| $\pi_j$, $\Sigma_j$, $r_{ij}$, $N_j$ | mixture weights, covariances, responsibilities $p(z_i = j \mid x_i)$, effective counts $\sum_ir_{ij}$ |
| $\mathcal L(q, \theta)$ | the evidence lower bound (ELBO) |
| $S$, $\lambda_j$, $V_k$ | sample covariance, its eigenvalues (descending), the top-$k$ principal directions (rows of `components_`) |
| $W$, $\sigma^2$ | PPCA loading matrix $(d, k)$ and noise variance; in topic 05, $W$ is a graph's weight matrix |
| $K$, $\tilde K$, $\alpha_j$ | kernel (Gram) matrix, its centred version, its unit eigenvectors |
| $D$, $D^{(2)}$, $B$ | distance matrix, squared distances, double-centred Gram matrix $-\frac12JD^{(2)}J$; in topic 05, $D$ is the degree matrix |
| $J$ | the centring matrix $I - \frac1n\mathbf 1\mathbf 1^\top$ (in topic 01, the inertia) |
| $p_{j\mid i}$, $p_{ij}$, $q_{ij}$, $\beta_i$ | t-SNE input conditionals, joint input affinities, output affinities, per-point precision |
| $L$, $L_{\text{sym}}$ | graph Laplacian $D - W$ and its normalized form $I - D^{-1/2}WD^{-1/2}$ |
| $C_{ij}$, $a_i$, $b_j$ | contingency table and its row and column sums |
| $h(x)$, $c(n)$, $\psi$, $s(x)$ | isolation path length, average path length of $n$ points, subsample size, anomaly score |

## how this connects

- builds on: eigenvectors, the SVD and Cholesky (ch 02), the multivariate Gaussian (ch 04), maximum likelihood, entropy, KL and mutual information (ch 05), momentum and non-convex optimization (ch 06), the estimator protocol, scalers and ROC/AUC (ch 07), vectorized distances and stable log-sum-exp (ch 00).
- used later in: hidden Markov models (Baum–Welch is EM) and variational inference for a Bayesian mixture (ch 12), kernels, the Nyström method and Gaussian processes (ch 09), graph Laplacians in graph neural networks (ch 20), variational autoencoders and vector quantization (ch 21), visualizing learned representations (ch 15 onward) and self-supervised embeddings (ch 25), deduplication and the inverted-file vector index for retrieval (ch 27), and out-of-distribution detection (ch 28).

## core references

- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, chapters 9 (mixtures and EM) and 12 (continuous latent variables: PCA, PPCA, kernel PCA).
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, chapter 14 (unsupervised learning). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Murphy (2022), *Probabilistic Machine Learning: An Introduction*, MIT Press, chapters 20–21 (dimensionality reduction, clustering). [probml.github.io/pml-book/book1.html](https://probml.github.io/pml-book/book1.html)
- von Luxburg (2007), "A tutorial on spectral clustering", *Statistics and Computing* 17. [arXiv:0711.0189](https://arxiv.org/abs/0711.0189)
- van der Maaten & Hinton (2008), "Visualizing data using t-SNE", *JMLR* 9. [jmlr.org/papers/v9/vandermaaten08a.html](https://jmlr.org/papers/v9/vandermaaten08a.html)
- Liu, Ting & Zhou (2008), "Isolation forest", *ICDM*. [doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)
