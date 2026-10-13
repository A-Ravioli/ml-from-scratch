"""Checks for capstone.py (eigenfaces): functional tests plus a minimum accuracy threshold.

All data here is synthetic and generated deterministically: small "faces" that live near a
low-dimensional subspace, with one cluster per person. Nothing is downloaded.
"""
import numpy as np
import pytest


def _faces(seed=0, n_people=15, per_person=10, side=8, dim=25):
    """Synthetic faces: mean + (identity + per-image variation) in a dim-D subspace + pixel noise."""
    rng = np.random.default_rng(seed)
    d = side * side
    basis, _ = np.linalg.qr(rng.normal(size=(d, dim)))
    mean = rng.uniform(0.3, 0.7, size=d)
    identity = 2.0 * rng.normal(size=(n_people, dim))
    X, y = [], []
    for person in range(n_people):
        for _ in range(per_person):
            coords = identity[person] + 0.6 * rng.normal(size=dim)
            X.append(mean + 0.1 * basis @ coords + 0.02 * rng.normal(size=d))
            y.append(person)
    return np.array(X), np.array(y)


def _split(X, y, n_train=7):
    train = np.concatenate([np.flatnonzero(y == c)[:n_train] for c in np.unique(y)])
    test = np.setdiff1d(np.arange(len(y)), train)
    return X[train], y[train], X[test], y[test]


def _cov_eigs(X):
    w, V = np.linalg.eigh(np.cov(X, rowvar=False))
    return w[::-1], V[:, ::-1]


# ---------------------------------------------------------------- stage 1: PCA


def test_stage1_shapes_mean_and_orthonormal_components(nb):
    X, _ = _faces()
    mean, comps, var = (np.asarray(a) for a in nb.fit_pca(X, 10))
    assert mean.shape == (64,) and comps.shape == (10, 64) and var.shape == (10,), (
        f"fit_pca(X, 10) on X of shape {X.shape} should return mean (64,), components (10, 64) "
        f"(one eigenface per ROW) and explained_variance (10,); got {mean.shape}, {comps.shape}, {var.shape}")
    assert np.allclose(mean, X.mean(axis=0)), "mean should be the per-pixel average over the training images"
    assert np.abs(comps @ comps.T - np.eye(10)).max() < 1e-10, "the components (rows) must be orthonormal"


def test_stage1_matches_covariance_eigendecomposition(nb):
    X, _ = _faces(seed=1)
    _, comps, var = (np.asarray(a) for a in nb.fit_pca(X, 8))
    w, V = _cov_eigs(X)
    assert np.allclose(var, w[:8], rtol=1e-8), (
        f"explained_variance should be the top eigenvalues of the sample covariance, s_i^2 / (n - 1) "
        f"for the singular values s of the CENTERED data; expected {np.round(w[:4], 5).tolist()}..., "
        f"got {np.round(var[:4], 5).tolist()}... (if yours are n-1 times too big, divide by n - 1)")
    P_ours, P_true = comps.T @ comps, V[:, :8] @ V[:, :8].T
    assert np.abs(P_ours - P_true).max() < 1e-8, (
        "your components don't span the top-8 principal subspace. did you center X before the SVD?")


def test_stage1_centers_the_data(nb):
    X, _ = _faces(seed=2)
    X_shifted = X + 100.0  # a huge constant offset must not change the principal directions
    _, c1, v1 = (np.asarray(a) for a in nb.fit_pca(X, 5))
    _, c2, v2 = (np.asarray(a) for a in nb.fit_pca(X_shifted, 5))
    assert np.allclose(v1, v2, rtol=1e-6) and np.abs(c1.T @ c1 - c2.T @ c2).max() < 1e-6, (
        "adding a constant to every pixel changed your PCA. subtract the mean image before the SVD: "
        "otherwise the first 'eigenface' just points at the mean.")


def test_stage1_rejects_bad_k(nb):
    X, _ = _faces()
    for bad in (0, 65):  # k must be between 1 and min(n, d) = min(150, 64)
        with pytest.raises(ValueError):
            nb.fit_pca(X, bad)


# ---------------------------------------------------------------- stage 2: projection and reconstruction


def test_stage2_project_and_reconstruct_shapes(nb):
    X, _ = _faces(seed=3)
    mean, comps, _ = nb.fit_pca(X, 12)
    Z = np.asarray(nb.project(X, mean, comps))
    assert Z.shape == (150, 12), f"project should give one row of 12 coordinates per image, got {Z.shape}"
    assert np.allclose(Z, (X - mean) @ np.asarray(comps).T), "project(X) = (X - mean) @ components.T"
    Xh = np.asarray(nb.reconstruct(Z, mean, comps))
    assert Xh.shape == X.shape, f"reconstruct should return images of shape {X.shape}, got {Xh.shape}"


def test_stage2_full_rank_reconstruction_is_exact(nb):
    X, _ = _faces(seed=4, n_people=5, per_person=6)  # 30 images: centered data has rank 29
    mean, comps, _ = nb.fit_pca(X, 29)
    Xh = np.asarray(nb.reconstruct(nb.project(X, mean, comps), mean, comps))
    assert np.allclose(Xh, X, atol=1e-10), "with every nonzero component kept, reconstruction should be exact"


def test_stage2_error_equals_discarded_variance(nb):
    X, _ = _faces(seed=5)
    w, _ = _cov_eigs(X)
    n = X.shape[0]
    errs = []
    for k in (5, 15, 40):
        mean, comps, _ = nb.fit_pca(X, k)
        Xh = np.asarray(nb.reconstruct(nb.project(X, mean, comps), mean, comps))
        err = np.sum((X - Xh) ** 2)
        want = (n - 1) * np.sum(w[k:])
        assert np.isclose(err, want, rtol=1e-6), (
            f"k={k}: the training reconstruction error ||X - X_hat||_F^2 should equal (n - 1) times the "
            f"sum of the discarded eigenvalues = {want:.6f} (Eckart-Young), got {err:.6f}")
        errs.append(err)
    assert errs[0] > errs[1] > errs[2], "more components must mean a smaller reconstruction error"


# ---------------------------------------------------------------- stage 3: nearest neighbour


def test_stage3_hand_example(nb):
    Z_train = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    y_train = np.array([7, 8, 9])
    pred = np.asarray(nb.nearest_neighbor(Z_train, y_train, np.array([[1.0, 1.0], [9.0, 1.0], [1.0, 8.0]])))
    assert pred.tolist() == [7, 8, 9], f"each test point should take the label of its nearest training point; got {pred.tolist()}"


def test_stage3_matches_brute_force_and_keeps_labels(nb):
    rng = np.random.default_rng(6)
    Z_train, Z_test = rng.normal(size=(40, 5)), rng.normal(size=(25, 5))
    y_train = np.array([f"person{i % 7}" for i in range(40)])
    pred = np.asarray(nb.nearest_neighbor(Z_train, y_train, Z_test))
    d = ((Z_test[:, None, :] - Z_train[None, :, :]) ** 2).sum(-1)
    assert pred.shape == (25,), f"one prediction per test row: expected shape (25,), got {pred.shape}"
    assert (pred == y_train[d.argmin(axis=1)]).all(), (
        "predictions don't match brute-force 1-nearest-neighbour (Euclidean). labels can be any type "
        "(here strings): return y_train[index of the nearest training row].")


# ---------------------------------------------------------------- stage 4: the full classifier


def test_stage4_accuracy_threshold(nb):
    X, y = _faces(seed=7)
    X_train, y_train, X_test, y_test = _split(X, y)
    pred = np.asarray(nb.eigenface_classifier(X_train, y_train, X_test, 20))
    assert pred.shape == y_test.shape, f"expected {y_test.shape} predictions, got {pred.shape}"
    acc = np.mean(pred == y_test)
    assert acc >= 0.9, (
        f"held-out accuracy with k = 20 should be at least 0.90 on this synthetic set (chance is 0.07), "
        f"got {acc:.3f}. fit PCA on the TRAINING images only, project both sets with the training "
        "mean and components, then use 1-nearest-neighbour.")


def test_stage4_uses_training_statistics_only(nb):
    X, y = _faces(seed=8)
    X_train, y_train, X_test, _ = _split(X, y)
    p1 = np.asarray(nb.eigenface_classifier(X_train, y_train, X_test, 15))
    p2 = np.asarray(nb.eigenface_classifier(X_train, y_train, X_test[:5], 15))
    assert (p1[:5] == p2).all(), (
        "predictions for a test image must not depend on which OTHER test images are passed in. "
        "fit everything (mean, components) on X_train only.")


def _numpy_pca_1nn(X_train, y_train, X_test, k):
    """Reference eigenfaces classifier built on NumPy's SVD (independent of your code)."""
    mean = X_train.mean(axis=0)
    Vt = np.linalg.svd(X_train - mean, full_matrices=False)[2][:k]
    Z_tr, Z_te = (X_train - mean) @ Vt.T, (X_test - mean) @ Vt.T
    d = ((Z_te[:, None, :] - Z_tr[None, :, :]) ** 2).sum(-1)
    return y_train[d.argmin(axis=1)]


def test_stage4_no_test_data_in_the_fit(nb):
    X, y = _faces(seed=9)
    X_train, y_train, X_test, _ = _split(X, y)
    outliers = 50.0 * np.random.default_rng(10).normal(size=(5, X.shape[1]))  # huge, weird "test images"
    p1 = np.asarray(nb.eigenface_classifier(X_train, y_train, X_test, 2))
    p2 = np.asarray(nb.eigenface_classifier(X_train, y_train, np.vstack([X_test, outliers]), 2))
    assert p2.shape == (len(X_test) + 5,) and (p1 == p2[:len(X_test)]).all(), (
        "adding a few huge outlier images to X_TEST changed the predictions for the other test images. "
        "that means the test images influenced the fit (data leakage): compute the mean and the "
        "components from X_train alone, then project X_test with them.")


def test_stage4_really_uses_k_components(nb):
    X, y = _faces(seed=11)
    X_train, y_train, X_test, _ = _split(X, y)
    want = _numpy_pca_1nn(X_train, y_train, X_test, 2)
    d_raw = ((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(-1)
    raw = y_train[d_raw.argmin(axis=1)]
    assert (want != raw).any(), "test setup: k = 2 should give different answers from raw pixels here"
    got = np.asarray(nb.eigenface_classifier(X_train, y_train, X_test, 2))
    assert (got == want).all(), (
        f"with k = 2, eigenface_classifier should classify by 1-NN on the 2-D projections; it disagrees "
        f"on {(got != want).sum()} of {len(want)} test images with a reference built on NumPy's SVD"
        + (" and matches 1-NN on raw pixels: did you skip the projection?" if (got == raw).all() else "."))
