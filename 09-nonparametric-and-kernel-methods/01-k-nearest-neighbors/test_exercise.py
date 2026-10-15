"""Checks for exercise.py (k-nearest neighbors).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. scikit-learn is
used only as an oracle. Random data is continuous, so there are no distance ties unless a
test builds them on purpose.
"""
import numpy as np
import pytest

from mlfs.testing import bounded


def _blobs(n, d, seed, n_classes=2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    centers = rng.normal(scale=2.0, size=(n_classes, d))
    return centers[y] + rng.normal(size=(n, d)), y


# ---------------------------------------------------------------- stage 1: distances, brute force


def test_stage1_minkowski_hand_example(nb):
    X = np.array([[0.0, 0.0]])
    Y = np.array([[3.0, 4.0], [1.0, -1.0]])
    for p, want in [(1, [7.0, 2.0]), (2, [5.0, np.sqrt(2)]), (np.inf, [4.0, 1.0])]:
        got = np.asarray(nb.minkowski_distances(X, Y, p))
        assert got.shape == (1, 2), f"X (1, 2) and Y (2, 2) give a (1, 2) distance matrix, got shape {got.shape}"
        assert np.allclose(got, [want]), (
            f"p = {p}: distances from the origin to (3, 4) and (1, -1) are {want}, got {got.ravel().tolist()}. "
            "p = 1 sums |differences|, p = 2 is Euclidean, p = inf takes the largest |difference|")


def test_stage1_minkowski_matches_scipy(nb):
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))
    for p in [1, 1.5, 2, 3]:
        assert np.allclose(nb.minkowski_distances(X, Y, p), cdist(X, Y, "minkowski", p=p)), (
            f"p = {p}: (sum_j |x_j - y_j|^p)^(1/p) for every pair of rows")
    assert np.allclose(nb.minkowski_distances(X, Y, np.inf), cdist(X, Y, "chebyshev")), "p = inf: the max norm"


def test_stage1_brute_kneighbors_matches_sklearn(nb):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(1)
    X, Q = rng.normal(size=(60, 3)), rng.normal(size=(20, 3))
    dist, idx = nb.brute_kneighbors(X, Q, 4)
    ref_d, ref_i = NearestNeighbors(n_neighbors=4, algorithm="brute").fit(X).kneighbors(Q)
    assert np.shape(dist) == (20, 4) and np.shape(idx) == (20, 4), (
        f"20 queries, k = 4: dist and idx are (20, 4), got {np.shape(dist)} and {np.shape(idx)}")
    assert np.array_equal(idx, ref_i), "idx[i] lists the training rows nearest to query i, nearest first"
    assert np.allclose(dist, ref_d, atol=1e-6), "dist holds Euclidean distances (not squared), ascending"


def test_stage1_ties_and_validation(nb):
    X = np.array([[1.0], [-1.0], [2.0], [0.0]])
    dist, idx = nb.brute_kneighbors(X, np.array([[0.0]]), 3)
    assert idx.tolist() == [[3, 0, 1]], (
        f"from 0, row 3 is at distance 0 and rows 0 and 1 tie at distance 1: ties go to the smaller index, so "
        f"[[3, 0, 1]]; got {idx.tolist()}. a stable sort (np.argsort(..., kind='stable')) gives this for free")
    assert np.allclose(dist, [[0.0, 1.0, 1.0]])
    same = np.tile([[1.0, 0.0]], (40, 1))  # 40 identical rows: every distance ties
    _, idx = nb.brute_kneighbors(same, np.array([[0.0, 0.0]]), 25)
    assert idx.tolist() == [list(range(25))], (
        f"with 40 tied distances the 25 neighbours must be rows 0..24 in order (ties go to the smaller index); got "
        f"{idx.tolist()}. NumPy's default sort is not stable on long rows: use kind='stable'")
    with pytest.raises(ValueError):
        nb.brute_kneighbors(X, np.array([[0.0]]), 5)
        pytest.fail("k = 5 with 4 training rows should raise ValueError")


def test_stage1_brute_is_vectorized(nb):
    rng = np.random.default_rng(2)
    X, Q = rng.normal(size=(1500, 5)), rng.normal(size=(1500, 5))
    bounded(nb.brute_kneighbors, X, Q, 3, budget=20_000, what="brute_kneighbors on 1,500 queries",
            hint="compute all the distances at once (pairwise_distances), then sort each row with one "
                 "NumPy call: no Python loop over queries or training rows.")


# ---------------------------------------------------------------- stage 2: models


def test_stage2_neighbor_weights(nb):
    d = np.array([[1.0, 2.0, 4.0], [0.0, 3.0, 0.0]])
    assert np.allclose(nb.neighbor_weights(d, "uniform"), 1.0), "'uniform': every neighbour gets weight 1"
    got = np.asarray(nb.neighbor_weights(d, "distance"))
    assert np.allclose(got[0], [1.0, 0.5, 0.25]), f"'distance': weight 1/d, so [1, 0.5, 0.25]; got {got[0]}"
    assert np.allclose(got[1], [1.0, 0.0, 1.0]), (
        f"a row with exact matches (distance 0) gives weight 1 to each exact match and 0 to the rest (1/0 is "
        f"infinite), so [1, 0, 1]; got {got[1]}")
    with pytest.raises(ValueError):
        nb.neighbor_weights(d, "gaussian")
        pytest.fail("an unknown weights string should raise ValueError")


def test_stage2_worked_example(nb):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [2.0, 3.0]])  # A, B, C, D, E
    y = np.array([0, 0, 1, 1, 1])
    q = np.array([[1.0, 0.5]])
    p3 = nb.KNeighborsClassifier(3).fit(X, y).predict_proba(q)
    assert np.allclose(p3, [[2 / 3, 1 / 3]]), f"3-NN, uniform: neighbours B, A, C vote 0, 0, 1 -> [2/3, 1/3]; got {p3}"
    p5 = nb.KNeighborsClassifier(5).fit(X, y).predict_proba(q)
    assert np.allclose(p5, [[0.4, 0.6]]), f"5-NN, uniform: 2 votes against 3 -> [0.4, 0.6]; got {p5}"
    pd = nb.KNeighborsClassifier(5, weights="distance").fit(X, y)
    got = pd.predict_proba(q)
    assert np.allclose(got, [[0.67225, 0.32775]], atol=1e-4), (
        f"5-NN, distance weights 1/d = 2, 0.894, 0.555, 0.485, 0.371: class 0 gets 2.894 of 4.306 -> 0.672; got {got}")
    assert pd.predict(q)[0] == 0, "the distance-weighted vote goes back to class 0"


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_stage2_classifier_matches_sklearn(nb, weights):
    from sklearn.neighbors import KNeighborsClassifier
    X, y = _blobs(150, 3, seed=3, n_classes=3)
    Q, _ = _blobs(40, 3, seed=4, n_classes=3)
    labels = np.array(["ant", "bee", "cat"])[y]
    ours = nb.KNeighborsClassifier(7, weights=weights).fit(X, labels)
    ref = KNeighborsClassifier(7, weights=weights, algorithm="brute").fit(X, labels)
    assert list(ours.classes_) == ["ant", "bee", "cat"], "classes_ holds the sorted distinct labels"
    assert np.allclose(ours.predict_proba(Q), ref.predict_proba(Q)), (
        f"weights={weights!r}: predict_proba[i, c] is the (weighted) share of query i's 7 neighbours in class c")
    assert np.array_equal(ours.predict(Q), ref.predict(Q)), "predict (inherited) is the most probable class"


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_stage2_regressor_matches_sklearn(nb, weights):
    from sklearn.neighbors import KNeighborsRegressor
    rng = np.random.default_rng(5)
    X, Q = rng.uniform(-2, 2, size=(120, 2)), rng.uniform(-2, 2, size=(30, 2))
    y = np.sin(X[:, 0]) * X[:, 1] + 0.1 * rng.normal(size=120)
    ours = nb.KNeighborsRegressor(5, weights=weights).fit(X, y)
    ref = KNeighborsRegressor(5, weights=weights, algorithm="brute").fit(X, y)
    assert np.allclose(ours.predict(Q), ref.predict(Q)), (
        f"weights={weights!r}: the prediction is the (weighted) mean target of the 5 nearest training rows")
    assert -1 < ours.score(Q, np.sin(Q[:, 0]) * Q[:, 1]) <= 1, "score is R^2 (inherited from Regressor)"


def test_stage2_protocol(nb):
    X, y = _blobs(40, 2, seed=6)
    clf = nb.KNeighborsClassifier(3)
    assert clf.get_params() == {"n_neighbors": 3, "weights": "uniform", "algorithm": "brute"}, (
        "__init__ stores its arguments unchanged under the same names")
    assert clf.fit(X, y) is clf, "fit returns self"
    assert np.array_equal(clf.y_idx_, y), "y_idx_[i] is the index of row i's label in classes_"
    dist, idx = clf.kneighbors(X[:5], k=1)
    assert np.array_equal(idx[:, 0], np.arange(5)), "a training row's nearest neighbour is itself (k=1)"
    assert clf.kneighbors(X[:5])[1].shape == (5, 3), "kneighbors(X) uses k = n_neighbors by default"
    assert nb.KNeighborsClassifier(1).fit(X, y).score(X, y) == 1.0, "1-NN classifies its own training set perfectly"
    reg = nb.KNeighborsRegressor(len(X)).fit(X, y.astype(float))
    assert np.allclose(reg.predict(X[:3]), y.mean()), "with k = n, every prediction is the mean of all targets"
    with pytest.raises(ValueError):
        nb.KNeighborsRegressor(3).fit(X, y.astype(float)).predict(X[:, :1])
        pytest.fail("predicting with a different number of features should raise ValueError")


# ---------------------------------------------------------------- stage 3: the curse of dimensionality


def test_stage3_relative_contrast_hand_example(nb):
    X = np.array([[1.0, 0.0], [0.0, 3.0], [-2.0, 0.0]])
    got = nb.relative_contrast(X, np.zeros(2))
    assert np.isclose(got, 2.0), f"distances 1, 3, 2 from the origin: (3 - 1) / 1 = 2, got {got}"


def test_stage3_contrast_shrinks_with_dimension(nb):
    rng = np.random.default_rng(7)
    med = []
    for d in [2, 20, 200]:
        vals = [nb.relative_contrast(rng.random((500, d)), rng.random(d)) for _ in range(5)]
        med.append(np.median(vals))
    assert med[0] > 5 * med[1] > 0 and med[1] > 2 * med[2] > 0, (
        f"for uniform data the relative contrast should collapse as d grows; medians for d = 2, 20, 200: "
        f"{np.round(med, 3).tolist()}")


def test_stage3_edge_length(nb):
    assert np.isclose(nb.edge_length(0.1, 1), 0.1), "in 1-D, 10% of the data spans 10% of the range"
    assert np.isclose(nb.edge_length(0.01, 2), 0.1), "in 2-D, a 0.1 x 0.1 square holds 1%"
    got = nb.edge_length(0.1, 10)
    assert np.isclose(got, 0.1 ** 0.1), f"in 10-D, 10% of the data needs edge 0.1^(1/10) = 0.794; got {got}"
    assert np.allclose(nb.edge_length(0.01, np.array([1, 2, 3])), [0.01, 0.1, 0.01 ** (1 / 3)]), "works elementwise"


# ---------------------------------------------------------------- stage 4: KD-tree build


def _check_tree(tree, n):
    nodes = tree.nodes
    assert isinstance(nodes, list) and nodes, "self.nodes should be a non-empty list of node dicts"
    assert sorted(nodes[0]["idx"].tolist()) == list(range(n)), "the root (nodes[0]) holds every row 0..n-1"
    leaves = []
    for i, node in enumerate(nodes):
        idx = np.asarray(node["idx"])
        if node["dim"] is None:
            assert node["left"] is None and node["right"] is None, f"node {i}: a leaf has no children"
            assert len(idx) <= tree.leaf_size, f"leaf {i} has {len(idx)} rows > leaf_size {tree.leaf_size}"
            leaves.append(idx)
            continue
        assert len(idx) > tree.leaf_size, f"node {i} has only {len(idx)} rows: it should have been a leaf"
        left, right = np.asarray(nodes[node["left"]]["idx"]), np.asarray(nodes[node["right"]]["idx"])
        assert sorted(np.concatenate([left, right]).tolist()) == sorted(idx.tolist()), (
            f"node {i}: its children split its rows exactly (each row goes to one side)")
        dim, split = node["dim"], node["split"]
        spread = tree.X[idx].max(axis=0) - tree.X[idx].min(axis=0)
        assert spread[dim] == spread.max(), (
            f"node {i} splits dimension {dim}, but its rows are most spread out along dimension {int(np.argmax(spread))}: "
            "every node splits its own widest dimension")
        assert np.all(tree.X[left, dim] <= split) and np.all(tree.X[right, dim] >= split), (
            f"node {i}: rows on the left need X[row, dim] <= split, on the right >= split")
        assert abs(len(left) - len(right)) <= 1, (
            f"node {i}: a median split halves the rows, got {len(left)} and {len(right)}")
    assert sorted(np.concatenate(leaves).tolist()) == list(range(n)), "the leaves partition the rows"
    return leaves


def test_stage4_tree_structure(nb):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 3)) * np.array([1.0, 0.2, 5.0])
    tree = bounded(nb.KDTree, X, leaf_size=10, what="KDTree(X) (the build)",
                   hint="build recurses only while a node has more than leaf_size rows.")
    leaves = _check_tree(tree, 300)
    assert tree.nodes[0]["dim"] == 2, (
        f"the root should split the widest dimension (spread 5x larger in dim 2), got dim {tree.nodes[0]['dim']}")
    assert len(leaves) == 32, f"300 rows halved down to <= 10 per leaf gives 32 leaves, got {len(leaves)}"


def test_stage4_small_and_duplicate_inputs(nb):
    t = nb.KDTree(np.arange(5.0).reshape(-1, 1), leaf_size=16)
    assert len(t.nodes) == 1 and t.nodes[0]["dim"] is None, "5 rows with leaf_size 16 is a single leaf"
    X = np.zeros((40, 2))
    tree = bounded(nb.KDTree, X, leaf_size=4, what="KDTree on 40 identical rows",
                   hint="split by position in the sorted order (at len(idx) // 2), so even identical values halve "
                        "the node and the recursion ends.")
    _check_tree(tree, 40)


# ---------------------------------------------------------------- stage 5: KD-tree query


@pytest.mark.parametrize("d,k,leaf", [(2, 1, 8), (3, 5, 8), (6, 3, 8), (2, 7, 2)])
def test_stage5_query_matches_brute_force(nb, d, k, leaf):
    rng = np.random.default_rng(10 + d + leaf)
    X, Q = rng.normal(size=(400, d)), rng.normal(size=(50, d))
    tree = nb.KDTree(X, leaf_size=leaf)
    dist, idx = bounded(tree.query, Q, k=k, budget=2_000_000, what="KDTree.query",
                        hint="search visits the near child, then the far child only if it could hold something closer.")
    D = np.sqrt(((Q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    ref_i = np.argsort(D, axis=1)[:, :k]
    assert np.shape(idx) == (50, k), f"idx should be (50, {k}), got {np.shape(idx)}"
    assert np.array_equal(idx, ref_i), (
        f"d = {d}, k = {k}: the tree must return exactly the brute-force neighbours, nearest first. if only some "
        "rows are wrong, check the pruning test: skip the far child only when (q[dim] - split)^2 >= the current "
        "k-th best squared distance (and never while the heap holds fewer than k rows)")
    assert np.allclose(dist, np.take_along_axis(D, ref_i, axis=1)), "dist holds Euclidean (not squared) distances"


def test_stage5_query_prunes_in_low_dimension(nb):
    rng = np.random.default_rng(11)
    X = rng.random((4000, 2))
    tree = nb.KDTree(X, leaf_size=16)
    bounded(tree.query, rng.random((100, 2)), k=1, budget=2_000_000, what="KDTree.query")
    frac = tree.n_distances_ / (100 * 4000)
    assert 0 < frac < 0.05, (
        f"in 2-D a KD-tree should compute distances to a few percent of the rows per query at most; it computed "
        f"{frac:.1%} (n_distances_ = {tree.n_distances_} for 100 queries). count the rows of every leaf you scan, and "
        "make sure the far child is skipped when it can't contain anything closer")


def test_stage5_distance_counts(nb):
    rng = np.random.default_rng(14)
    X, Q = rng.normal(size=(50, 3)), rng.normal(size=(7, 3))
    one_leaf = nb.KDTree(X, leaf_size=64)
    one_leaf.query(Q, k=2)
    assert one_leaf.n_distances_ == 7 * 50, (
        f"a tree that is a single leaf computes every distance: 7 queries x 50 rows = 350, but n_distances_ = "
        f"{one_leaf.n_distances_}. add len(rows) for every leaf you scan (not 1 per leaf)")
    one_leaf.query(Q[:3], k=2)
    assert one_leaf.n_distances_ == 3 * 50, (
        f"query resets n_distances_ to 0 at the start: after querying 3 points it should be 150, got {one_leaf.n_distances_}")
    Xh = rng.normal(size=(1000, 16))
    tree = nb.KDTree(Xh, leaf_size=16)
    bounded(tree.query, rng.normal(size=(20, 16)), k=1, budget=3_000_000, what="KDTree.query in 16-D")
    frac = tree.n_distances_ / (20 * 1000)
    assert frac > 0.9, (
        f"in 16 dimensions with 1,000 points the pruning test almost never fires, so the tree computes nearly every "
        f"distance; it reported {frac:.1%}. count every row of every leaf you scan")


def test_stage5_models_with_kd_tree(nb):
    X, y = _blobs(200, 3, seed=12, n_classes=3)
    Q, _ = _blobs(30, 3, seed=13, n_classes=3)
    a = nb.KNeighborsClassifier(5, algorithm="kd_tree").fit(X, y)
    b = nb.KNeighborsClassifier(5).fit(X, y)
    assert np.allclose(a.predict_proba(Q), b.predict_proba(Q)), (
        "algorithm='kd_tree' must give exactly the same predictions as 'brute': fit builds tree_ = KDTree(X_) and "
        "kneighbors calls tree_.query")
    r1 = nb.KNeighborsRegressor(4, weights="distance", algorithm="kd_tree").fit(X, y.astype(float))
    r2 = nb.KNeighborsRegressor(4, weights="distance").fit(X, y.astype(float))
    assert np.allclose(r1.predict(Q), r2.predict(Q)), "the same holds for the regressor"
