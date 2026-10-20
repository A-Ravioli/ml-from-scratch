"""Checks for exercise.py (k-means).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Data are small
seeded blobs; scikit-learn is used only as an oracle. Calls that loop run under
mlfs.testing.bounded, so a missing iteration cap fails instead of hanging.
"""
import tracemalloc

import numpy as np
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score as sk_silhouette

from mlfs.testing import bounded

LINE = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])


def _blobs(n_per=60, k=4, d=2, spread=0.5, sep=8.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(k, d))
    X = np.concatenate([c + spread * rng.standard_normal((n_per, d)) for c in centers])
    y = np.repeat(np.arange(k), n_per)
    return X, y, centers


def _grid_blobs(side=3, n_per=40, spread=0.4, seed=0):
    """side x side blobs on a grid with spacing 10: well separated, many clusters."""
    rng = np.random.default_rng(seed)
    centers = np.array([(10.0 * i, 10.0 * j) for i in range(side) for j in range(side)])
    X = np.concatenate([c + spread * rng.standard_normal((n_per, 2)) for c in centers])
    return X, np.repeat(np.arange(side * side), n_per), centers


# ---------------------------------------------------------------- stage 1: one Lloyd step


def test_stage1_assign_worked_example(nb):
    labels, d2 = nb.assign_to_nearest(LINE, np.array([[1.0], [11.0]]))
    assert np.array_equal(np.asarray(labels), [0, 0, 0, 1, 1, 1]), f"nearest of [1, 11]: expected [0 0 0 1 1 1], got {labels}"
    assert np.allclose(d2, [1, 0, 1, 1, 0, 1]), (
        f"the second output is the SQUARED distance to the nearest center: expected [1 0 1 1 0 1], got {d2}")
    labels, _ = nb.assign_to_nearest(LINE, np.array([[0.0], [2.0]]))
    assert labels[1] == 0, "x = 1 is equally far from 0 and 2: a tie goes to the lower index (np.argmin does this)"
    assert np.issubdtype(np.asarray(labels).dtype, np.integer), "labels are integers"


def test_stage1_assign_matches_brute_force(nb):
    rng = np.random.default_rng(1)
    X, C = rng.standard_normal((200, 5)), rng.standard_normal((7, 5))
    labels, d2 = nb.assign_to_nearest(X, C)
    full = ((X[:, None, :] - C[None]) ** 2).sum(-1)
    assert np.array_equal(labels, full.argmin(1)), "each point gets the index of its nearest center"
    assert np.allclose(d2, full.min(1)), "d2[i] = squared distance from X[i] to its nearest center"


def test_stage1_assign_memory(nb):
    rng = np.random.default_rng(2)
    X, C = rng.standard_normal((4000, 64)), rng.standard_normal((32, 64))
    tracemalloc.start()
    nb.assign_to_nearest(X, C)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    blowup = X.shape[0] * C.shape[0] * X.shape[1] * 8
    assert peak < blowup / 4, (
        f"assign_to_nearest peaked at {peak / 1e6:.0f} MB; an (n, k, d) difference array would be {blowup / 1e6:.0f} MB. "
        "use pairwise_distances(X, centers, squared=True) (the ||x||^2 - 2 x.c + ||c||^2 expansion) instead of broadcasting X[:, None] - centers")


def test_stage1_update_centers(nb):
    new = nb.update_centers(LINE, np.array([0, 0, 1, 1, 1, 1]), np.array([[0.0], [2.0]]))
    assert np.allclose(new, [[0.5], [8.75]]), f"cluster means of (0, 1) and (2, 10, 11, 12) are 0.5 and 8.75, got {np.ravel(new)}"
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 3))
    labels = rng.integers(0, 4, 100)
    C = rng.standard_normal((4, 3))
    new = nb.update_centers(X, labels, C)
    want = np.array([X[labels == j].mean(0) for j in range(4)])
    assert np.allclose(new, want), "each new center is the mean of the points assigned to it"


def test_stage1_update_centers_memory(nb):
    rng = np.random.default_rng(16)
    X = rng.standard_normal((20000, 4))
    labels = rng.integers(0, 500, 20000)
    C = rng.standard_normal((500, 4))
    tracemalloc.start()
    new = nb.update_centers(X, labels, C)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    onehot = 500 * 20000 * 8
    assert peak < onehot / 4, (
        f"update_centers peaked at {peak / 1e6:.0f} MB; a (k, n) one-hot matrix would be {onehot / 1e6:.0f} MB. "
        "sum each cluster in O(n d) memory (sort by label + np.add.reduceat, or np.bincount with weights per feature)")
    assert np.allclose(new[labels[0]], X[labels == labels[0]].mean(0)), "each center is its cluster's mean"


def test_stage1_empty_cluster_keeps_center(nb):
    C = np.array([[0.0], [100.0], [11.0]])
    labels, _ = nb.assign_to_nearest(LINE, C)
    new = nb.update_centers(LINE, labels, C)
    assert np.isfinite(new).all(), "an empty cluster must not produce NaN (mean of zero points)"
    assert np.allclose(new[1], [100.0]), f"cluster 1 is empty: it should keep its old center 100, got {new[1]}"
    assert np.allclose(C, [[0.0], [100.0], [11.0]]), "update_centers must not modify the centers array you passed in"


# ---------------------------------------------------------------- stage 2: k-means++


def test_stage2_shape_and_rows(nb):
    X, _, _ = _blobs(seed=4)
    C = nb.kmeans_plus_plus(X, 4, rng=0)
    assert np.shape(C) == (4, 2), f"kmeans_plus_plus(X, 4) returns (4, 2), got {np.shape(C)}"
    rows = [int(np.argmin(((X - c) ** 2).sum(1))) for c in C]
    assert all(np.allclose(X[r], c) for r, c in zip(rows, C)), "every initial center is one of the data points"
    assert len(set(rows)) == 4, "the k centers are k different points"


def test_stage2_d_squared_distribution(nb):
    # X = 0, 1, 10 and k = 2: the first center is uniform, the second is drawn with p ∝ D^2.
    X = np.array([[0.0], [1.0], [10.0]])
    want = {(0, 2): (100 / 101 + 100 / 181) / 3, (1, 2): (81 / 82 + 81 / 181) / 3, (0, 1): (1 / 101 + 1 / 82) / 3}
    rng = np.random.default_rng(5)
    n = 3000
    counts = {key: 0 for key in want}
    for _ in range(n):
        C = np.ravel(nb.kmeans_plus_plus(X, 2, rng=rng))
        key = tuple(sorted(int(np.argmin(np.abs(X[:, 0] - c))) for c in C))
        assert key in counts, f"the two centers must be two different data points, got {C}"
        counts[key] += 1
    for key, p in want.items():
        got = counts[key] / n
        assert abs(got - p) < 0.04, (
            f"P(centers = points {key}) should be {p:.3f}, you sampled it {got:.3f} of the time. "
            "the first center is uniform; the next is drawn with probability proportional to the SQUARED distance "
            "to the nearest chosen center. (sampling by D, unsquared, gives 0.48 / 0.46 / 0.064 instead of "
            "0.514 / 0.478 / 0.007; uniform sampling gives 1/3 each)")


def test_stage2_spreads_over_separated_blobs(nb):
    X, y, _ = _grid_blobs(side=3, seed=6)
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(200):
        C = nb.kmeans_plus_plus(X, 9, rng=rng)
        blob = [y[int(np.argmin(((X - c) ** 2).sum(1)))] for c in C]
        hits += len(set(blob)) == 9
    assert hits >= 160, (
        f"on 9 far-apart blobs, k-means++ should put one center in each blob almost every time; it did in {hits}/200. "
        "(uniform sampling manages it about 1 time in 1,000)")


def test_stage2_rng(nb):
    X, _, _ = _blobs(seed=8)
    np.random.seed(0)
    a = nb.kmeans_plus_plus(X, 3, rng=42)
    np.random.seed(1)
    b = nb.kmeans_plus_plus(X, 3, rng=42)
    assert np.allclose(a, b), "the same rng seed must give the same centers (draw only from as_rng(rng), never np.random)"


# ---------------------------------------------------------------- stage 3: KMeans


def test_stage3_worked_example(nb):
    km = nb.KMeans(2, init=np.array([[0.0], [2.0]]), n_init=1)
    bounded(km.fit, LINE, what="KMeans.fit", hint="Lloyd's loop must stop after max_iters iterations at most.")
    assert np.allclose(np.sort(km.cluster_centers_.ravel()), [1.0, 11.0]), (
        f"from centers (0, 2) Lloyd converges to (1, 11), got {km.cluster_centers_.ravel()}")
    assert np.isclose(km.inertia_, 4.0), f"inertia = sum of squared distances to the nearest center = 4, got {km.inertia_}"
    assert km.n_iter_ == 3, (
        f"the centers go (0, 2) -> (0.5, 8.75) -> (1, 11) -> (1, 11): 3 iterations (the third moves nothing and stops), got n_iter_ = {km.n_iter_}")
    assert np.array_equal(km.labels_, [0, 0, 0, 1, 1, 1]), f"labels_ for the final centers, got {km.labels_}"
    exact = bounded(nb.KMeans(2, init=np.array([[0.0], [2.0]]), tol=0.0).fit, LINE, what="KMeans.fit")
    assert exact.n_iter_ == 3, (
        f"with tol=0, Lloyd stops when the centres don't move at all (shift <= tol, not <): 3 iterations, got {exact.n_iter_}")
    tiny = nb.KMeans(2, init=np.array([[0.0], [0.002]]), n_init=1)
    bounded(tiny.fit, LINE * 1e-3, what="KMeans.fit")
    assert tiny.n_iter_ == 3 and np.allclose(np.sort(tiny.cluster_centers_.ravel()), [0.001, 0.011]), (
        f"the same example in units 1000x smaller must take the same 3 iterations (got {tiny.n_iter_}): the stopping "
        "threshold is tol times the mean per-feature variance of X, so it scales with the data")


def test_stage3_estimator_protocol(nb):
    km = nb.KMeans(3, n_init=2, rng=0)
    assert km.get_params() == dict(n_clusters=3, init="k-means++", n_init=2, max_iters=300, tol=0.0001, rng=0), (
        "KMeans subclasses Estimator and __init__ only stores its arguments")
    X, _, _ = _blobs(seed=9)
    assert bounded(km.fit, X, what="KMeans.fit") is km, "fit returns self"
    D = km.transform(X)
    assert D.shape == (len(X), 3), f"transform gives distances to each center, (n, k); got {D.shape}"
    assert np.allclose(D, np.sqrt(((X[:, None] - km.cluster_centers_[None]) ** 2).sum(-1))), (
        "transform returns plain Euclidean distances (not squared)")
    assert np.array_equal(km.predict(X), km.labels_), "predict on the training data gives labels_"
    assert np.isclose(km.score(X), -km.inertia_), "score(X) = -inertia (higher is better, like every score)"
    assert np.isclose(km.inertia_, (D.min(1) ** 2).sum()), "inertia_ belongs to the final centers and labels"
    assert np.array_equal(nb.KMeans(3, n_init=2, rng=0).fit_predict(X), km.labels_), "fit_predict = fit(X).labels_"


def test_stage3_recovers_blobs(nb):
    X, y, _ = _grid_blobs(side=3, seed=10)
    km = bounded(nb.KMeans(9, rng=0).fit, X, what="KMeans.fit")
    ari = adjusted_rand_score(y, km.labels_)
    assert ari > 0.999, f"9 far-apart blobs should be found exactly (ARI 1), got ARI {ari:.3f}"


def test_stage3_matches_sklearn_inertia(nb):
    X, _, _ = _blobs(n_per=80, k=6, d=3, spread=1.5, sep=4.0, seed=11)
    ours = bounded(nb.KMeans(6, n_init=10, rng=0).fit, X, budget=2_000_000, what="KMeans.fit").inertia_
    sk = SkKMeans(6, n_init=10, random_state=0).fit(X).inertia_
    assert ours <= sk * 1.05, (
        f"on overlapping blobs, the best of 10 k-means++ runs should be as good as scikit-learn's: "
        f"your inertia {ours:.1f}, sklearn {sk:.1f}. keep the run with the LOWEST inertia")


def test_stage3_restarts_help(nb):
    # from uniformly random starts, one run finds the 9 blobs only about 10% of the time; 30 restarts almost always
    X, y, _ = _grid_blobs(side=3, seed=12)
    found = [adjusted_rand_score(y, bounded(nb.KMeans(9, init="random", n_init=30, rng=100 + s).fit, X, budget=3_000_000,
                                            what="KMeans.fit").labels_) > 0.999 for s in range(5)]
    assert sum(found) >= 3, (
        f"with n_init=30 random starts, the best run should find the 9 blobs almost every time; it did for {sum(found)} "
        "of 5 seeds. keep the run with the lowest inertia, and draw every run's start from the same generator")


def test_stage3_max_iters_and_rng(nb):
    X, _, _ = _blobs(seed=13)
    km = bounded(nb.KMeans(4, n_init=1, max_iters=1, rng=0).fit, X, what="KMeans.fit")
    assert km.n_iter_ == 1, f"max_iters=1 allows one Lloyd iteration, got n_iter_ = {km.n_iter_}"
    one = bounded(nb.KMeans(2, init=np.array([[0.0], [2.0]]), max_iters=1).fit, LINE, what="KMeans.fit")
    assert np.allclose(one.cluster_centers_.ravel(), [0.5, 8.75]) and np.array_equal(one.labels_, [0, 0, 0, 1, 1, 1]), (
        "stopped after one iteration, the centres are (0.5, 8.75); labels_ must belong to THOSE centres ([0 0 0 1 1 1]), "
        f"not to the ones before the update. got labels_ {one.labels_}: assign once more after the loop")
    assert np.isclose(one.inertia_, -one.score(LINE)), "inertia_ must be the inertia of the final centres"
    a = nb.KMeans(4, n_init=3, rng=5).fit(X).cluster_centers_
    b = nb.KMeans(4, n_init=3, rng=5).fit(X).cluster_centers_
    assert np.allclose(a, b), "the same rng gives the same fit"


# ---------------------------------------------------------------- stage 4: silhouette


def test_stage4_matches_sklearn(nb):
    X, y, _ = _blobs(n_per=40, k=3, d=4, spread=2.0, sep=3.0, seed=14)
    for labels in (y, np.random.default_rng(15).integers(0, 4, len(X))):
        got, want = nb.silhouette_score(X, labels), sk_silhouette(X, labels)
        assert np.isclose(got, want, atol=1e-9), (
            f"silhouette should be {want:.6f}, got {got:.6f}. a(i) averages over the OTHER points of i's cluster "
            "(divide by n_c - 1); b(i) is the smallest mean distance to another cluster")


def test_stage4_hand_example(nb):
    X = np.array([[0.0], [1.0], [4.0], [5.0]])
    # point 0: a = 1, b = (4 + 5) / 2 = 4.5, s = 3.5 / 4.5; symmetric for the others.
    want = np.mean([3.5 / 4.5, 2.5 / 3.5, 2.5 / 3.5, 3.5 / 4.5])
    assert np.isclose(nb.silhouette_score(X, [0, 0, 1, 1]), want), f"expected {want:.4f}"
    got = nb.silhouette_score(X, [0, 0, 0, 1])
    assert np.isclose(got, sk_silhouette(X, [0, 0, 0, 1])), "a singleton cluster's point scores 0"
    assert np.isclose(nb.silhouette_score(X, ["b", "b", "a", "a"]), want), "labels can be any values"


def test_stage4_rejects_bad_k(nb):
    X = np.arange(5.0)[:, None]
    for labels in ([0] * 5, [0, 1, 2, 3, 4]):
        try:
            nb.silhouette_score(X, labels)
        except ValueError:
            continue
        raise AssertionError(f"silhouette is undefined for {len(set(labels))} clusters of 5 points: raise ValueError")
