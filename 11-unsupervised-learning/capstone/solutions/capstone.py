import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.unsupervised import (  # yours, from this chapter
        PCA,
        TSNE,
        GaussianMixture,
        KMeans,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )
    from mlfs.utils import as_rng, pairwise_distances  # yours, from ch 00

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    N_TRAIN = 1500 if SMALL else 10000  # images clustered
    N_TEST = 300 if SMALL else 2000  # held-out images for the cluster-then-label classifier
    N_TSNE = 300 if SMALL else 2500  # exact t-SNE is O(n^2) per iteration
    TSNE_ITERS = 300 if SMALL else 1000
    SEEDS = 3 if SMALL else 5


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 11 · capstone: exploring unlabeled MNIST (reference solution)

    read [`brief.md`](brief.md) first. you'll cluster and embed handwritten digits *without
    looking at the labels*, and only afterwards use the labels to measure how much of the
    digit structure the unsupervised methods found.

    | stage | you build |
    |---|---|
    | 1 | `purity`, `evaluate_clustering`: score a clustering against the true labels |
    | 2 | `cluster_pca`: PCA, then k-means or a GMM |
    | 3 | `scores_over_seeds`: the same pipeline over several seeds, for error bars |
    | 4 | `knn_label_agreement`: how well an embedding keeps same-digit images together |
    | 5 | `cluster_classifier_accuracy`: label each cluster by majority vote, classify held-out images |

    when they're green, scroll down: the experiments run on your functions, and the write-up
    prompts are at the end.
    """)
    return


@app.function
def load_mnist(n_train, n_test, *, rng=0):
    """given: random subsets of MNIST as float pixels in [0, 1], flattened to (n, 784), with int labels."""
    d = datasets.mnist()
    gen = as_rng(rng)
    tr = gen.choice(len(d["X_train"]), min(n_train, len(d["X_train"])), replace=False)
    te = gen.choice(len(d["X_test"]), min(n_test, len(d["X_test"])), replace=False)
    flat = lambda X: X.reshape(len(X), -1).astype(np.float64) / 255.0  # noqa: E731
    return flat(d["X_train"][tr]), d["y_train"][tr], flat(d["X_test"][te]), d["y_test"][te]


@app.function
def purity(labels_true, labels_pred):
    """stage 1: the fraction of points whose cluster's majority true label is their own label."""
    _, t = np.unique(np.asarray(labels_true), return_inverse=True)
    _, p = np.unique(np.asarray(labels_pred), return_inverse=True)
    C = np.zeros((p.max() + 1, t.max() + 1), dtype=np.int64)
    np.add.at(C, (p, t), 1)
    return float(C.max(axis=1).sum() / len(t))


@app.function
def evaluate_clustering(labels_true, labels_pred):
    """stage 1: {'ari': ..., 'nmi': ..., 'purity': ..., 'n_clusters': ...} for one clustering."""
    return {"ari": float(adjusted_rand_score(labels_true, labels_pred)),
            "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
            "purity": purity(labels_true, labels_pred),
            "n_clusters": int(len(np.unique(labels_pred)))}


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", purity=purity, evaluate_clustering=evaluate_clustering)
    return


@app.function
def cluster_pca(X, *, n_components=50, n_clusters=10, method="kmeans", rng=None):
    """stage 2: labels (n,) from KMeans (n_init=10), or the best of 3 diagonal GaussianMixtures, on PCA scores of X."""
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"method must be 'kmeans' or 'gmm', got {method!r}")
    Z = PCA(n_components).fit_transform(X) if n_components is not None else np.asarray(X, dtype=float)
    rng = as_rng(rng)
    if method == "kmeans":
        return KMeans(n_clusters, n_init=10, rng=rng).fit(Z).labels_
    fits = [GaussianMixture(n_clusters, covariance_type="diag", rng=rng).fit(Z) for _ in range(3)]
    return max(fits, key=lambda g: g.score(Z)).predict(Z)


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", cluster_pca=cluster_pca)
    return


@app.function
def scores_over_seeds(X, y, *, seeds=(0, 1, 2), **kwargs):
    """stage 3: run cluster_pca(X, rng=seed, **kwargs) per seed; {'ari': array, 'nmi': array, 'purity': array}."""
    rows = [evaluate_clustering(y, cluster_pca(X, rng=s, **kwargs)) for s in seeds]
    return {k: np.array([r[k] for r in rows]) for k in ("ari", "nmi", "purity")}


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", scores_over_seeds=scores_over_seeds, evaluate_clustering=evaluate_clustering, cluster_pca=cluster_pca)
    return


@app.function
def knn_label_agreement(Z, labels, k=10):
    """stage 4: the mean, over points, of the fraction of their k nearest neighbours in Z (not themselves) sharing their label."""
    Z = np.asarray(Z, dtype=float)
    labels = np.asarray(labels)
    if not 1 <= k < len(Z):
        raise ValueError(f"k must be in [1, {len(Z) - 1}], got {k}")
    D = pairwise_distances(Z, squared=True)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, k - 1, axis=1)[:, :k]
    return float(np.mean(labels[nn] == labels[:, None]))


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", knn_label_agreement=knn_label_agreement)
    return


@app.function
def cluster_classifier_accuracy(Z_train, y_train, Z_test, y_test, n_clusters, *, rng=None):
    """stage 5: KMeans on Z_train, each cluster -> its majority y_train label, test accuracy of nearest-centre labels."""
    km = KMeans(n_clusters, n_init=10, rng=rng).fit(Z_train)
    classes, y_idx = np.unique(np.asarray(y_train), return_inverse=True)
    counts = np.zeros((n_clusters, len(classes)), dtype=np.int64)
    np.add.at(counts, (km.labels_, y_idx), 1)
    cluster_label = classes[np.argmax(counts, axis=1)]
    return float(np.mean(cluster_label[km.predict(Z_test)] == np.asarray(y_test)))


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", cluster_classifier_accuracy=cluster_classifier_accuracy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data

    a random subset of MNIST's training images to cluster, and of its test images to
    evaluate the cluster-then-label classifier (experiment C). pixels are scaled to
    $[0, 1]$; no other preprocessing. offline (`MLFS_OFFLINE=1`) this is a small synthetic
    stand-in, and the numbers won't resemble the brief's.
    """)
    return


@app.cell
def _():
    X, y, X_test, y_test = load_mnist(N_TRAIN, N_TEST, rng=0)
    _t0 = time.time()
    pca50 = PCA(50).fit(X)
    Z50, Z50_test = pca50.transform(X), pca50.transform(X_test)
    mo.md(f"{len(X):,} training and {len(X_test):,} test images, {X.shape[1]} pixels each. PCA to 50 components: "
          f"**{pca50.explained_variance_ratio_.sum():.1%}** of the pixel variance ({time.time() - _t0:.1f} s).")
    return X, Z50, Z50_test, y, y_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A: clustering, against baselines, with error bars

    four pipelines with $k = 10$ clusters, each over several seeds: k-means on the raw
    pixels, k-means on 50 PCA components, k-means on 10 PCA components, and a diagonal GMM
    on 50 PCA components. the baseline is a random labeling with 10 equally likely clusters.
    """)
    return


@app.cell
def _(X, Z50, y):
    _seeds = tuple(range(SEEDS))
    _t0 = time.time()
    _rng = np.random.default_rng(0)
    _rand = [evaluate_clustering(y, _rng.integers(0, 10, len(y))) for _ in _seeds]
    results_A = {"random labels (baseline)": {k: np.array([r[k] for r in _rand]) for k in ("ari", "nmi", "purity")}}
    results_A["k-means, raw pixels"] = scores_over_seeds(X, y, seeds=_seeds, n_components=None)
    # the PCA scores are computed once (Z50); the first 10 columns of the PCA-50 scores are the PCA-10 scores
    results_A["k-means, PCA-50"] = scores_over_seeds(Z50, y, seeds=_seeds, n_components=None)
    results_A["k-means, PCA-10"] = scores_over_seeds(Z50[:, :10], y, seeds=_seeds, n_components=None)
    results_A["GMM (diag), PCA-50"] = scores_over_seeds(Z50, y, seeds=_seeds, n_components=None, method="gmm")
    _rows = [f"| {name} | " + " | ".join(f"{r[m].mean():.3f} ± {r[m].std(ddof=1):.3f}" for m in ("ari", "nmi", "purity")) + " |"
             for name, r in results_A.items()]
    mo.md(f"mean ± sd over {SEEDS} seeds ({time.time() - _t0:.0f} s):\n\n| pipeline | ARI | NMI | purity |\n|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell
def _(X, Z50, y):
    # what does each k-means cluster look like? its mean image, and its mix of true digits
    _km = KMeans(10, n_init=10, rng=0).fit(Z50)
    _fig, _axes = plt.subplots(2, 10, figsize=(15, 3.6), gridspec_kw=dict(height_ratios=[1.3, 1]))
    for _j in range(10):
        _members = _km.labels_ == _j
        _axes[0, _j].imshow(X[_members].mean(axis=0).reshape(28, 28), cmap="gray")
        _axes[0, _j].axis("off")
        _axes[0, _j].set_title(f"cluster {_j}: {_members.sum()}", fontsize=8)
        _counts = np.bincount(y[_members], minlength=10)
        _axes[1, _j].bar(range(10), _counts / max(_counts.sum(), 1), color="C0")
        _axes[1, _j].set_xticks(range(10))
        _axes[1, _j].tick_params(labelsize=6)
        _axes[1, _j].set_ylim(0, 1)
    _fig.suptitle("k-means on PCA-50: each cluster's mean image (top) and its mix of true digits (bottom)", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B: embeddings

    PCA to 2 components versus t-SNE (your `TSNE`, perplexity 30, run on the 50 PCA
    components of a random subset of the images). we colour the points by their true digit
    *only for the picture*, and measure how often a point's 10 nearest neighbours in each
    embedding share its digit. we also cluster the t-SNE map itself with k-means.
    """)
    return


@app.cell
def _(Z50, y):
    _rng = np.random.default_rng(1)
    sub = _rng.choice(len(Z50), min(N_TSNE, len(Z50)), replace=False)
    _t0 = time.time()
    tsne_map = TSNE(perplexity=30.0, n_iters=TSNE_ITERS, rng=0).fit(Z50[sub])
    tsne_seconds = time.time() - _t0
    _Zs, _ys = Z50[sub], y[sub]
    agreement = {
        "PCA-2": knn_label_agreement(_Zs[:, :2], _ys),
        "PCA-50": knn_label_agreement(_Zs, _ys),
        "t-SNE (2-D)": knn_label_agreement(tsne_map.embedding_, _ys),
    }
    _km_tsne = [evaluate_clustering(_ys, KMeans(10, rng=s).fit(tsne_map.embedding_).labels_) for s in range(SEEDS)]
    _km_pca = [evaluate_clustering(_ys, KMeans(10, rng=s).fit(_Zs).labels_) for s in range(SEEDS)]
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for _ax, (_E, _title) in zip(_axes, [(_Zs[:, :2], "PCA, first 2 components"), (tsne_map.embedding_, f"t-SNE ({len(sub)} images)")]):
        _sc = _ax.scatter(_E[:, 0], _E[:, 1], c=_ys, cmap="tab10", s=4, vmin=-0.5, vmax=9.5)
        _ax.set_title(_title)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.colorbar(_sc, ax=_axes, ticks=range(10), label="true digit (for colour only)")
    _a = lambda rs, m: f"{np.mean([r[m] for r in rs]):.3f} ± {np.std([r[m] for r in rs], ddof=1):.3f}"  # noqa: E731
    mo.vstack([_fig, mo.md(
        "10-NN label agreement: " + ", ".join(f"{k} **{v:.3f}**" for k, v in agreement.items())
        + f" (chance: about 0.10). t-SNE took {tsne_seconds:.0f} s, final KL {tsne_map.kl_divergence_:.3f}. "
        f"k-means (k = 10) on the same {len(sub)} images: on the t-SNE map ARI **{_a(_km_tsne, 'ari')}**, NMI "
        f"{_a(_km_tsne, 'nmi')}; on PCA-50 ARI **{_a(_km_pca, 'ari')}**, NMI {_a(_km_pca, 'nmi')} ({SEEDS} seeds).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C: more clusters than digits

    clusters don't have to correspond one-to-one with digits. we run k-means on the PCA-50
    training scores with $k$ from 10 to 200, name each cluster after its most common
    training digit, and classify the held-out test images by their nearest centre. (this
    uses labels to *name* clusters, so it measures how label-consistent the clusters are,
    not unsupervised performance.)
    """)
    return


@app.cell
def _(Z50, Z50_test, y, y_test):
    _ks = [10, 20, 50, 100, 200] if not SMALL else [10, 20, 50]
    _t0 = time.time()
    acc_C = {k: np.array([cluster_classifier_accuracy(Z50, y, Z50_test, y_test, k, rng=s) for s in range(3)]) for k in _ks}
    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    _ax.errorbar(_ks, [acc_C[k].mean() for k in _ks], yerr=[acc_C[k].std(ddof=1) for k in _ks], fmt="o-", capsize=3)
    _ax.set(xscale="log", xlabel="number of clusters k", ylabel="test accuracy", title="cluster, then label by majority vote")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("test accuracy (mean ± sd over 3 seeds): " + ", ".join(
        f"k = {k}: **{v.mean():.3f} ± {v.std(ddof=1):.3f}**" for k, v in acc_C.items()) + f" ({time.time() - _t0:.0f} s).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer each prompt with numbers from the experiments above (full config: run with
    `MLFS_OFFLINE` unset), with their uncertainty. see [`rubric.md`](rubric.md).

    1. **how much digit structure is there?** report ARI, NMI and purity for your best
       pipeline against the random baseline and k-means on raw pixels. did PCA help? give
       the difference with its uncertainty, and explain what PCA-50 changed and what it
       didn't (what do the 734 discarded dimensions contain, and what did dropping them save?).
    2. **what did the clusters find?** look at the cluster means and label mixes. which
       digits get their own cluster, which share one, and which are split? explain each with
       what you know about the shapes (and about k-means's preference for round, equal-sized
       clusters).
    3. **embeddings.** compare PCA-2 and t-SNE by 10-NN agreement and by eye. k-means on the
       t-SNE map scores well: why is that a dangerous result to report as "the clustering"?
       what did topic 04's lab teach you about reading the distances and sizes in the map?
    4. **more clusters than classes.** why does accuracy keep rising with $k$, and what's the
       catch (what happens as $k$ approaches the number of training points)? what does this
       say about the shape of digit classes in PCA space?
    5. **what you'd do next.** one concrete change (a different representation, method or
       evaluation), the result you'd expect, and why.
    """)
    return


if __name__ == "__main__":
    app.run()
