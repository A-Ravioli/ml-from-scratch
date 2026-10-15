# 11 · capstone: exploring unlabeled MNIST

## goal

pretend nobody told you MNIST has ten digits. you have 10,000 small grayscale images and no labels. how much structure can you find, and how would you know? this is the everyday situation of unsupervised learning, and its two hard parts are the ones this capstone is about: choosing a representation and a method without labels to tune against, and evaluating honestly afterwards.

you'll cluster the images with k-means and a Gaussian mixture, on raw pixels and on PCA scores; look at what the clusters contain; draw maps of the data with PCA and your own t-SNE; and only then bring in the labels, purely to *measure*: how well do the clusters match the digits (adjusted Rand index, normalized mutual information, purity), how well do the maps keep same-digit images together, and how label-consistent do clusters become when you allow more of them than there are digits. every number comes with a baseline and an uncertainty over seeds (chapter 07's rule). the capstone uses every topic of the chapter except anomaly detection: k-means (01), Gaussian mixtures (02), PCA (03), t-SNE (04) and the clustering metrics (05).

## data

MNIST through `mlfs.datasets.mnist()` (downloaded once, cached under `~/.cache/mlfs`). the given `load_mnist(n_train, n_test, rng=0)` draws random subsets of the 60,000 training and 10,000 test images and returns them as float pixels in $[0, 1]$, flattened to 784 columns, with their labels. the labels are used only for evaluation. tests use small synthetic data instead (Gaussian clusters in 100 dimensions).

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 1,500 + 300 synthetic images, t-SNE on 300 for 300 iterations, 3 seeds | under a minute |
| **full** (your write-up) | default | 10,000 + 2,000 MNIST images, t-SNE on 2,500 for 1,000 iterations, 5 seeds | about 4–6 minutes with the reference `mlfs.unsupervised` (measured on a busy laptop: experiment A 1.5–3.5 min, most of it k-means on raw pixels; t-SNE 50–75 s; experiment C 45–70 s) |

the notebook picks the config from `datasets.is_offline()`. exact t-SNE costs $O(n^2)$ time and memory per iteration, which is why it runs on a 2,500-image subset.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `purity(labels_true, labels_pred)`: the fraction of points whose cluster's majority label is their own; and `evaluate_clustering(labels_true, labels_pred)` → `{"ari", "nmi", "purity", "n_clusters"}` using your topic 05 metrics.
   - `cluster_pca(X, *, n_components=50, n_clusters=10, method="kmeans", rng=None)`: PCA (or the raw features when `n_components=None`), then `KMeans(n_init=10)` or the best (highest log-likelihood) of three diagonal `GaussianMixture` fits, all drawn from one generator `as_rng(rng)`.
   - `scores_over_seeds(X, y, *, seeds=(0, 1, 2), **kwargs)` → arrays of ARI, NMI and purity, one per seed (`rng=seed`).
   - `knn_label_agreement(Z, labels, k=10)`: the mean fraction of each point's $k$ nearest neighbours in an embedding (not itself) that share its label.
   - `cluster_classifier_accuracy(Z_train, y_train, Z_test, y_test, n_clusters, *, rng=None)`: k-means (`n_init=10`) on the training scores, each cluster named after its most common training label, test points labelled by their nearest centre.
2. the three experiments, run in the full config: (A) clustering pipelines against a random baseline, (B) PCA and t-SNE maps with 10-NN agreement, (C) cluster-then-label accuracy as $k$ grows.
3. the write-up at the end of the notebook, answering its five prompts with numbers and uncertainties.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `cluster_pca` | ARI > 0.9 on ten well-separated synthetic clusters, with k-means and (best of 3 seeds) with the GMM; PCA beats raw features when the signal lives in 8 of 300 dimensions |
| `knn_label_agreement` | exact on a hand example; about $1/C$ for random labels with $C$ classes |
| `cluster_classifier_accuracy` | > 0.95 on ten separated synthetic classes; works with labels of any type |
| your write-up | every number in it reproducible from the notebook, with its uncertainty |

## what the reference solution measures (full config)

all on the same 10,000-image subset (`rng=0`); "±" is the standard deviation over 5 seeds of the clustering (3 for experiment C), which does *not* include the variation you'd see with a different subset of images.

**experiment A** (PCA-50 keeps 82.6% of the pixel variance):

| pipeline | ARI | NMI | purity |
|---|---|---|---|
| random labels (baseline) | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.118 ± 0.001 |
| k-means, raw pixels | 0.355 ± 0.002 | 0.487 ± 0.002 | 0.586 ± 0.002 |
| k-means, PCA-50 | 0.358 ± 0.001 | 0.487 ± 0.004 | 0.587 ± 0.004 |
| k-means, PCA-10 | 0.334 ± 0.001 | 0.465 ± 0.001 | 0.575 ± 0.000 |
| GMM (diagonal), PCA-50 | 0.159 ± 0.012 | 0.423 ± 0.007 | 0.450 ± 0.007 |

k-means finds real structure (far above the baseline), but nowhere near "the ten digits". PCA-50 gives the same clustering as raw pixels at a fraction of the cost; PCA-10 is slightly worse. the diagonal GMM is clearly *worse* than k-means. the cluster means show why k-means stops at ARI 0.36: zeros and ones each get two clusters (upright and slanted), 4, 7 and 9 share clusters, as do 3, 5 and 8, and only 2 and 6 get a clean cluster of their own.

**experiment B** (2,500 images): 10-nearest-neighbour label agreement **0.384** in the first two PCA components, **0.846** in all 50, and **0.865** in the 2-D t-SNE map (chance ≈ 0.10; t-SNE took 47 s, final KL 1.116). k-means with $k = 10$ on the t-SNE map reaches ARI **0.556**, against **0.347 ± 0.018** on the PCA-50 scores of the same images.

**experiment C**: naming clusters by majority vote and classifying held-out test images by nearest centre gives test accuracy **0.585 ± 0.002** with 10 clusters, 0.721 ± 0.011 with 20, 0.815 ± 0.014 with 50, 0.864 ± 0.002 with 100 and **0.885 ± 0.007** with 200.

your numbers should land close to these with your own `mlfs.unsupervised` (within a few hundredths, more for anything involving t-SNE or the GMM, whose local optima depend on the order of random draws).

## stretch goals

- **a better representation.** cluster after standardizing the PCA scores (whitening), or after deskewing the digits; or cluster the t-SNE map *and* report how the result changes across t-SNE seeds and perplexities.
- **a better mixture.** a full-covariance GMM on 10–20 PCA components, with BIC to choose $k$. does BIC agree with "ten digits"?
- **hierarchy.** Ward linkage on 2,000 images (your `AgglomerativeClustering`): at which level of the tree do the digits separate, and which merge first?
- **anomalies.** run your `IsolationForest` on the PCA scores and look at the 30 most anomalous images. what are they?
- **the Hungarian matching.** implement the optimal one-to-one matching between clusters and digits (accuracy after matching) and compare it with purity.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- LeCun, Bottou, Bengio & Haffner (1998), "Gradient-based learning applied to document recognition", *Proceedings of the IEEE* 86(11): the MNIST data set. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
- van der Maaten & Hinton (2008), "Visualizing data using t-SNE", *JMLR* 9: includes t-SNE maps of MNIST. [jmlr.org/papers/v9/vandermaaten08a.html](https://jmlr.org/papers/v9/vandermaaten08a.html)
- Hubert & Arabie (1985), "Comparing partitions", *Journal of Classification* 2. [doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075)
- Vinh, Epps & Bailey (2010), "Information theoretic measures for clusterings comparison", *JMLR* 11. [jmlr.org/papers/v11/vinh10a.html](https://jmlr.org/papers/v11/vinh10a.html)
- Xie, Girshick & Farhadi (2016), "Unsupervised deep embedding for clustering analysis", *ICML*: deep clustering on MNIST, for a sense of what learned representations add. [arXiv:1511.06335](https://arxiv.org/abs/1511.06335)
