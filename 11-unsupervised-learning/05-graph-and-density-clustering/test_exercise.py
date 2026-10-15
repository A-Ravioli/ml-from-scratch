"""Checks for exercise.py (graph & density clustering).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scikit-learn's
metrics, SpectralClustering, DBSCAN and AgglomerativeClustering, and scipy's linkage, are
oracles only. DBSCAN and the merge loop run under mlfs.testing.bounded.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.cluster import AgglomerativeClustering as SkAgglo
from sklearn.metrics import adjusted_rand_score as sk_ari
from sklearn.metrics import normalized_mutual_info_score as sk_nmi

from mlfs.testing import bounded


def _moons(n=300, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n)
    lab = np.arange(n) % 2
    X = np.where(lab[:, None] == 0, np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
    return X + noise * rng.standard_normal((n, 2)), lab


def _rings(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    lab = np.arange(n) % 2
    r = np.where(lab == 0, 1.0, 0.35)
    return np.c_[r * np.cos(t), r * np.sin(t)] + 0.04 * rng.standard_normal((n, 2)), lab


def _blobs(n_per=40, k=3, d=2, spread=0.5, sep=6.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, (k, d))
    return np.concatenate([c + spread * rng.standard_normal((n_per, d)) for c in centers]), np.repeat(np.arange(k), n_per)


# ---------------------------------------------------------------- stage 1: comparing clusterings


def test_stage1_contingency(nb):
    C = nb.contingency_matrix(["a", "a", "b", "b", "b"], [1, 0, 0, 0, 2])
    assert np.array_equal(C, [[1, 1, 0], [2, 0, 1]]), (
        f"rows = sorted true labels (a, b), columns = sorted predicted labels (0, 1, 2); expected [[1 1 0] [2 0 1]], got {C}")


def test_stage1_hand_example(nb):
    t, p = [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]
    # pairs within: C = [[2,1,0],[0,1,2]] -> sum C(n_ij, 2) = 1 + 1 = 2; rows 3 + 3 = 6; cols 1 + 1 + 1 = 3; C(6, 2) = 15
    want = (2 - 6 * 3 / 15) / (0.5 * (6 + 3) - 6 * 3 / 15)
    got = nb.adjusted_rand_score(t, p)
    assert np.isclose(got, want), f"ARI should be (2 - 1.2) / (4.5 - 1.2) = {want:.4f}, got {got}"
    assert np.isclose(nb.adjusted_rand_score(t, [5, 5, 5, 9, 9, 9]), 1.0), "a relabeled copy of the truth scores 1 (permutation invariant)"


def test_stage1_match_sklearn(nb):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 200)
    for pred in (rng.integers(0, 3, 200), np.where(rng.random(200) < 0.8, y, rng.integers(0, 5, 200)), y):
        assert np.isclose(nb.adjusted_rand_score(y, pred), sk_ari(y, pred)), "ARI should match scikit-learn"
        assert np.isclose(nb.normalized_mutual_info_score(y, pred), sk_nmi(y, pred)), (
            "NMI = I(U; V) / ((H(U) + H(V)) / 2), arithmetic normalization, should match scikit-learn")


def test_stage1_edge_cases(nb):
    assert np.isclose(nb.adjusted_rand_score([0, 0, 0], [1, 1, 1]), 1.0), "one cluster each: identical partitions, ARI 1"
    assert np.isclose(nb.normalized_mutual_info_score([0, 0, 0], [1, 1, 1]), 1.0), "one cluster each: NMI 1"
    assert np.isclose(nb.normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 0, 0]), 0.0), "one side trivial: NMI 0"
    rng = np.random.default_rng(2)
    vals = [nb.adjusted_rand_score(rng.integers(0, 5, 500), rng.integers(0, 5, 500)) for _ in range(20)]
    assert abs(np.mean(vals)) < 0.01, f"ARI of independent random labelings averages ~0, got {np.mean(vals):.3f}"


# ---------------------------------------------------------------- stage 2: spectral clustering


def test_stage2_affinity(nb):
    X, _ = _blobs(n_per=10, seed=3)
    W = nb.affinity_matrix(X, "rbf", gamma=0.5)
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    want = np.exp(-0.5 * D2)
    np.fill_diagonal(want, 0)
    assert np.allclose(W, want), "rbf affinity: exp(-gamma ||x_i - x_j||^2) with a ZERO diagonal (no self-loops)"
    A = nb.affinity_matrix(X, "nearest_neighbors", n_neighbors=3)
    assert np.allclose(A, A.T) and np.allclose(np.diag(A), 0), "the kNN affinity is symmetric with a zero diagonal"
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    nn = np.argsort(D2, 1)[:, :3]
    A0 = np.zeros_like(D2)
    A0[np.repeat(np.arange(len(X)), 3), nn.ravel()] = 1.0
    assert np.allclose(A, (A0 + A0.T) / 2), (
        "the kNN affinity is (A + A^T) / 2 for the 0/1 neighbour matrix A: 1 for mutual neighbours, 0.5 for one-way "
        "(np.maximum(A, A.T) would give one-way pairs the full weight 1)")
    assert np.all((A > 0).sum(1) >= 3), "every point keeps its own 3 nearest neighbours (not counting itself)"


def test_stage2_moons_and_rings(nb):
    for name, (X, y) in [("two moons", _moons(n=400, noise=0.04, seed=4)), ("two rings", _rings(seed=5))]:
        sc = nb.SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=15, rng=0)
        labels = bounded(sc.fit_predict, X, budget=3_000_000, what="SpectralClustering.fit_predict")
        ari = sk_ari(y, labels)
        assert ari > 0.98, (
            f"{name}: spectral clustering on a 15-NN graph should separate them (ARI {ari:.3f}). use the eigenvectors of the "
            "k SMALLEST eigenvalues of L_sym = I - D^-1/2 W D^-1/2 (the LARGEST of D^-1/2 W D^-1/2), normalize rows, then k-means")
        assert sc.embedding_.shape == (len(X), 2) and np.allclose(np.linalg.norm(sc.embedding_, axis=1), 1), (
            "embedding_ holds the row-normalized eigenvectors, (n, k), unit rows")


def test_stage2_rbf_blobs(nb):
    rng = np.random.default_rng(6)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]])
    X = np.concatenate([c + 0.5 * rng.standard_normal((40, 2)) for c in centers])
    y = np.repeat(np.arange(4), 40)
    labels = bounded(nb.SpectralClustering(4, affinity="rbf", gamma=1.0, rng=0).fit_predict, X, budget=3_000_000,
                     what="SpectralClustering.fit_predict")
    assert sk_ari(y, labels) > 0.98, f"rbf spectral clustering should find 4 separated blobs (ARI {sk_ari(y, labels):.3f})"
    assert nb.SpectralClustering().get_params() == dict(n_clusters=2, affinity="rbf", gamma=1.0, n_neighbors=10, rng=None)


# ---------------------------------------------------------------- stage 3: DBSCAN


def _dbscan(nb, X, **kw):
    return bounded(nb.DBSCAN(**kw).fit, X, budget=3_000_000, what="DBSCAN.fit",
                   hint="expanding a cluster must mark every point as it's labelled, so it's never pushed twice.")


def test_stage3_hand_example(nb):
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [30.0]])
    db = _dbscan(nb, X, eps=1.0, min_samples=3)
    assert np.array_equal(db.labels_, [0, 0, 0, 0, 1, 1, 1, -1]), (
        f"eps=1, min_samples=3: points 1, 2 (and 11) are core; 0 and 3 are border points of cluster 0, 10 and 12 of "
        f"cluster 1; 30 is noise. expected [0 0 0 0 1 1 1 -1], got {db.labels_}")
    assert np.array_equal(db.core_sample_indices_, [1, 2, 5]), f"core points are indices [1 2 5], got {db.core_sample_indices_}"


def test_stage3_matches_sklearn(nb):
    rng = np.random.default_rng(7)
    X = np.concatenate([_moons(n=300, seed=7)[0], rng.uniform(-1.5, 2.5, (30, 2))])
    for eps, ms in ((0.15, 5), (0.1, 4), (0.3, 10)):
        db = _dbscan(nb, X, eps=eps, min_samples=ms)
        sk = SkDBSCAN(eps=eps, min_samples=ms).fit(X)
        assert np.array_equal(db.core_sample_indices_, sk.core_sample_indices_), (
            f"eps={eps}, min_samples={ms}: core points are those with >= min_samples points within eps, COUNTING THEMSELVES")
        assert np.array_equal(db.labels_, sk.labels_), (
            f"eps={eps}, min_samples={ms}: labels differ from scikit-learn. scan points in index order, start a new cluster at "
            "each unlabelled core point, and give each border point the first cluster that reaches it")
    assert np.array_equal(nb.DBSCAN(eps=0.15, min_samples=5).fit_predict(X), SkDBSCAN(eps=0.15, min_samples=5).fit(X).labels_)


def test_stage3_shared_border_point(nb):
    # on paper 4: 0 and 2 are the only core points; 1 is a border point of both clusters
    X = np.array([[-1.0], [-0.5], [0.0], [1.0], [2.0], [2.5], [3.0]])
    db = _dbscan(nb, X, eps=1.0, min_samples=4)
    assert np.array_equal(db.labels_, [0, 0, 0, 0, 1, 1, 1]), (
        f"the shared border point (x = 1) belongs to the cluster that reaches it FIRST (cluster 0, started at index 2); "
        f"a later cluster must not relabel it. expected [0 0 0 0 1 1 1], got {db.labels_}")
    rev = _dbscan(nb, X[::-1], eps=1.0, min_samples=4)
    assert np.array_equal(rev.labels_, [0, 0, 0, 0, 1, 1, 1]), "scanning the reversed data, x = 1 joins the right-hand cluster"


def test_stage3_all_noise(nb):
    X = np.arange(10.0)[:, None] * 5
    db = _dbscan(nb, X, eps=1.0, min_samples=2)
    assert np.all(db.labels_ == -1) and len(db.core_sample_indices_) == 0, "points farther apart than eps are all noise"


# ---------------------------------------------------------------- stage 4: agglomerative


def _agglo(nb, X, **kw):
    return bounded(nb.AgglomerativeClustering(**kw).fit, X, budget=3_000_000, what="AgglomerativeClustering.fit")


def test_stage4_hand_example(nb):
    X = np.array([[0.0], [1.0], [5.0], [7.0], [20.0]])
    a = _agglo(nb, X, n_clusters=2, linkage="single")
    assert np.array_equal(a.children_, [[0, 1], [2, 3], [5, 6], [4, 7]]), (
        f"single linkage on the points 0, 1, 5, 7, 20 (indices 0-4) merges indices (0, 1) at height 1 -> node 5, (2, 3) at 2 "
        f"-> node 6, nodes (5, 6) at 4 -> node 7, then (4, 7) at 13. merge t creates node n + t, and each pair is listed "
        f"smaller id first; got {a.children_.tolist()}")
    assert np.allclose(a.distances_, [1, 2, 4, 13]), f"merge heights [1 2 4 13], got {a.distances_}"
    assert np.array_equal(a.labels_, [0, 0, 0, 0, 1]), f"cutting at 2 clusters: (0, 1, 5, 7) and (20), got {a.labels_}"
    c = _agglo(nb, X, n_clusters=2, linkage="complete")
    assert np.allclose(c.distances_, [1, 2, 7, 20]), (
        f"complete linkage: the cluster distance is the LARGEST pairwise distance: heights [1 2 7 20], got {c.distances_}")


def test_stage4_heights_match_scipy(nb):
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, (40, 3))
    for method in ("single", "complete", "average", "ward"):
        a = _agglo(nb, X, n_clusters=3, linkage=method)
        want = scipy_linkage(X, method=method)[:, 2]
        assert a.children_.shape == (39, 2) and a.distances_.shape == (39,), "the full tree has n - 1 merges"
        assert np.allclose(np.sort(a.distances_), np.sort(want)), (
            f"{method}: merge heights differ from scipy's. Lance-Williams updates: single min, complete max, average size-weighted "
            "mean, ward ((n_i + n_k) d_ik + (n_j + n_k) d_jk - n_k d_ij) / (n_i + n_j + n_k) on SQUARED distances, height = sqrt")


def test_stage4_labels_match_sklearn(nb):
    X, _ = _blobs(n_per=25, k=4, spread=1.2, sep=4.0, seed=9)
    for method in ("single", "complete", "average", "ward"):
        for k in (2, 4):
            ours = _agglo(nb, X, n_clusters=k, linkage=method).labels_
            sk = SkAgglo(n_clusters=k, linkage=method).fit(X).labels_
            assert sk_ari(sk, ours) > 0.999, f"{method}, {k} clusters: the cut should match scikit-learn's (ARI {sk_ari(sk, ours):.3f})"
            assert len(np.unique(ours)) == k, f"labels_ must have exactly {k} clusters"
