"""Checks for capstone.py (exploring unlabeled MNIST).

Synthetic data only: "digits" are well-separated Gaussian clusters in 100 dimensions with a
low-dimensional signal, so PCA + k-means can recover them. The reference mlfs.unsupervised is
used under pytest. Clustering calls run under mlfs.testing.bounded.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score as sk_ari
from sklearn.metrics import normalized_mutual_info_score as sk_nmi

from mlfs.testing import bounded
from mlfs.unsupervised import PCA, GaussianMixture, KMeans  # the reference, under pytest
from mlfs.utils import as_rng


def _digits(n_per=60, k=10, d=100, signal=8, sep=10.0, noise=1.0, seed=0):
    """k clusters whose centres differ in a `signal`-dimensional subspace, plus isotropic noise in all d."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(0, 1, (d, signal)))
    centres = rng.normal(0, sep, (k, signal)) @ basis.T
    X = np.concatenate([c + noise * rng.standard_normal((n_per, d)) for c in centres])
    y = np.repeat(np.arange(k), n_per)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _cluster(nb, X, **kw):
    return bounded(nb.cluster_pca, X, budget=5_000_000, what="cluster_pca", **kw)


# ---------------------------------------------------------------- stage 1: scoring a clustering


def test_stage1_purity_hand_example(nb):
    y = [0, 0, 0, 1, 1, 2, 2, 2]
    pred = [5, 5, 7, 7, 7, 9, 9, 9]
    # cluster 5: {0, 0} -> 2; cluster 7: {0, 1, 1} -> 2; cluster 9: {2, 2, 2} -> 3; purity = 7/8
    assert np.isclose(nb.purity(y, pred), 7 / 8), f"purity = sum over clusters of the majority count / n = 7/8, got {nb.purity(y, pred)}"
    assert np.isclose(nb.purity(y, np.arange(8)), 1.0), "one cluster per point is perfectly pure (purity rewards many clusters)"
    assert np.isclose(nb.purity(y, np.zeros(8)), 3 / 8), "one cluster for everything: the majority label's share"


def test_stage1_evaluate(nb):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.integers(0, 6, 300))
    r = nb.evaluate_clustering(y, pred)
    assert set(r) >= {"ari", "nmi", "purity", "n_clusters"}, f"keys ari, nmi, purity, n_clusters; got {sorted(r)}"
    assert np.isclose(r["ari"], sk_ari(y, pred)) and np.isclose(r["nmi"], sk_nmi(y, pred)), (
        "use your adjusted_rand_score and normalized_mutual_info_score from topic 05")
    assert r["n_clusters"] == len(np.unique(pred)) and np.isclose(r["purity"], nb.purity(y, pred))


# ---------------------------------------------------------------- stage 2: PCA + clustering


def test_stage2_recovers_clusters(nb):
    X, y = _digits(seed=2)
    for method, seeds in (("kmeans", (0,)), ("gmm", (0, 1, 2))):
        aris = []
        for s in seeds:  # EM has local optima: a GMM gets three seeds, and one good fit is enough
            labels = bounded(nb.cluster_pca, X, n_components=10, n_clusters=10, method=method, rng=s, budget=5_000_000,
                             what="cluster_pca")
            assert np.shape(labels) == (len(X),), f"{method}: one label per image, got shape {np.shape(labels)}"
            aris.append(sk_ari(y, labels))
        assert max(aris) > 0.9, (
            f"{method}: on 10 well-separated synthetic 'digits', PCA-10 + clustering should reach ARI > 0.9 "
            f"(best of {len(seeds)} seeds), got {max(aris):.3f}")


def test_stage2_follows_the_spec(nb):
    X, _ = _digits(n_per=30, sep=4.0, noise=1.5, seed=10)
    Z = PCA(10).fit_transform(X)
    for s in (0, 1):
        want_km = KMeans(10, n_init=10, rng=s).fit(Z).labels_
        got = _cluster(nb, X, n_components=10, n_clusters=10, method="kmeans", rng=s)
        assert np.array_equal(got, want_km), (
            "method='kmeans' must be KMeans(n_clusters, n_init=10, rng=as_rng(rng)) on the PCA scores; your labels differ "
            "from that construction")
        g = as_rng(s)
        fits = [GaussianMixture(10, covariance_type="diag", rng=g).fit(Z) for _ in range(3)]
        want_gmm = max(fits, key=lambda m: m.score(Z)).predict(Z)
        got = _cluster(nb, X, n_components=10, n_clusters=10, method="gmm", rng=s)
        assert np.array_equal(got, want_gmm), (
            "method='gmm' must fit three diagonal GaussianMixtures, all drawn from ONE generator as_rng(rng), and return the "
            "predictions of the one with the highest mean log-likelihood (score)")


def test_stage2_pca_helps(nb):
    X, y = _digits(n_per=40, sep=3.0, noise=1.6, d=300, seed=3)
    raw = np.median([sk_ari(y, _cluster(nb, X, n_components=None, n_clusters=10, rng=s)) for s in range(3)])
    pca = np.median([sk_ari(y, _cluster(nb, X, n_components=8, n_clusters=10, rng=s)) for s in range(3)])
    assert pca > raw + 0.05, (
        f"when the signal lives in 8 of 300 dimensions, PCA-8 should beat raw features (median ARI {pca:.3f} vs {raw:.3f}). "
        "n_components=None means: cluster the raw features")


def test_stage2_validation(nb):
    X, _ = _digits(n_per=10, seed=4)
    try:
        nb.cluster_pca(X, method="dbscan")
    except ValueError:
        return
    raise AssertionError("an unknown method should raise ValueError")


# ---------------------------------------------------------------- stage 3: error bars over seeds


def test_stage3_scores_over_seeds(nb):
    X, y = _digits(n_per=40, sep=2.5, noise=1.5, seed=5)
    r = bounded(nb.scores_over_seeds, X, y, seeds=(0, 1, 2, 3), n_components=8, n_clusters=10, budget=8_000_000,
                what="scores_over_seeds")
    assert set(r) >= {"ari", "nmi", "purity"}, f"keys ari, nmi, purity; got {sorted(r)}"
    assert all(np.shape(r[k]) == (4,) for k in ("ari", "nmi", "purity")), "one value per seed: arrays of shape (4,)"
    want = [sk_ari(y, _cluster(nb, X, n_components=8, n_clusters=10, rng=s)) for s in (0, 1, 2, 3)]
    assert np.allclose(r["ari"], want), "seed s must be passed as rng=s to cluster_pca, with the other keyword arguments"


# ---------------------------------------------------------------- stage 4: embedding quality


def test_stage4_knn_agreement(nb):
    Z = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [9.0]])
    labels = np.array([0, 0, 1, 1, 1, 0])
    # k = 1 neighbours: 0->1 (same), 1->0 (same), 2->1 (diff), 3->4 (same), 4->3 (same), 5->4 (diff)
    assert np.isclose(nb.knn_label_agreement(Z, labels, k=1), 4 / 6), (
        f"with k=1: 4 of 6 points have a same-label nearest neighbour (not counting themselves), got "
        f"{nb.knn_label_agreement(Z, labels, k=1)}")
    X, y = _digits(seed=6)
    assert nb.knn_label_agreement(X, y) > 0.99, "well-separated clusters: almost every neighbour shares the label"
    rng = np.random.default_rng(7)
    a = nb.knn_label_agreement(rng.normal(0, 1, (500, 5)), rng.integers(0, 10, 500))
    assert abs(a - 0.1) < 0.03, f"random labels with 10 classes agree with about 10% of neighbours, got {a:.3f}"


# ---------------------------------------------------------------- stage 5: cluster, then label


def test_stage5_cluster_classifier(nb):
    X, y = _digits(n_per=80, d=10, seed=8)  # as if already reduced by PCA
    Xtr, ytr, Xte, yte = X[:600], y[:600], X[600:], y[600:]
    acc = bounded(nb.cluster_classifier_accuracy, Xtr, ytr, Xte, yte, 10, rng=0, budget=5_000_000,
                  what="cluster_classifier_accuracy")
    assert acc > 0.95, f"10 clusters on 10 separated classes, named by majority vote: test accuracy > 0.95, got {acc:.3f}"
    Xs, ys = _digits(n_per=60, k=4, sep=1.2, noise=1.0, seed=9)
    for k in (4, 30):
        for s in (0, 1):
            km = KMeans(k, n_init=10, rng=s).fit(Xs[:180])
            names = np.array([np.bincount(ys[:180][km.labels_ == j], minlength=4).argmax() for j in range(k)])
            want = np.mean(names[km.predict(Xs[180:])] == ys[180:])
            got = bounded(nb.cluster_classifier_accuracy, Xs[:180], ys[:180], Xs[180:], ys[180:], k, rng=s,
                          budget=5_000_000, what="cluster_classifier_accuracy")
            assert np.isclose(got, want), (
                f"n_clusters={k}, rng={s}: expected accuracy {want:.4f} from KMeans(n_clusters, n_init=10, rng) on the training "
                f"scores, majority-vote names and nearest-centre test labels; got {got:.4f}. use the n_clusters you're given")
    ytr_str = np.array(["abcdefghij"[v] for v in ytr])
    yte_str = np.array(["abcdefghij"[v] for v in yte])
    acc_s = nb.cluster_classifier_accuracy(Xtr, ytr_str, Xte, yte_str, 10, rng=0)
    assert np.isclose(acc_s, acc), "labels can be any values (strings too): name clusters by their majority label"
