"""Checks for 19 · 04 generalization in deep learning.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. label corruption is
checked for its exact count and with chi-squared tests on the new classes and on the chosen
positions; random features for their scale and determinism; the solver against pinv/lstsq on both
sides of the interpolation threshold, rank-deficient designs and ridge closed forms; the curve
against your own pieces and a synthetic problem whose test error must peak at p = n; gradient
descent against the minimum-norm solution and hand-computed steps.
"""
import math

import numpy as np
import pytest
from scipy import stats

from mlfs.testing import bounded

def _run(fn, *args, **kwargs):
    """every call that loops runs under a line budget, so a learner's endless loop fails instead of hanging."""
    return bounded(fn, *args, budget=20_000_000, what=getattr(fn, "__name__", None), **kwargs)



# ---------------------------------------------------------------- stage 1: label noise


def test_stage1_exact_count_and_always_wrong(nb):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, size=37)
    y0 = y.copy()
    for frac, k in ((0.0, 0), (0.3, 11), (0.7, 26), (1.0, 37)):
        out = nb.corrupt_labels(y, frac, 10, rng=1)
        changed = int(np.sum(out != y))
        assert changed == k, (
            f"fraction {frac} of 37 labels: exactly round({frac} * 37) = {k} must change, got {changed}. choose the "
            "positions without replacement, and give each a class different from its own")
        assert out.dtype == y.dtype and out.shape == y.shape, "same shape and dtype as y"
    assert np.array_equal(y, y0), "corrupt_labels must not modify its input"
    a = nb.corrupt_labels(y, 0.5, 10, rng=5)
    b = nb.corrupt_labels(y, 0.5, 10, rng=5)
    c = nb.corrupt_labels(y, 0.5, 10, rng=6)
    assert np.array_equal(a, b) and not np.array_equal(a, c), "the same rng seed gives the same result; another seed differs"


def test_stage1_new_classes_uniform_over_the_others(nb):
    y = np.full(30000, 3)
    out = nb.corrupt_labels(y, 1.0, 10, rng=0)
    counts = np.bincount(out, minlength=10)
    assert counts[3] == 0, "a corrupted label is always a DIFFERENT class"
    others = np.delete(counts, 3)
    chi2 = float(((others - others.mean()) ** 2 / others.mean()).sum())
    assert chi2 < stats.chi2.ppf(0.999, 8), (
        f"the new class is uniform over the other 9: counts {others.tolist()} (chi-squared {chi2:.1f}, 8 dof). an "
        "off-by-one like integers(1, n_classes - 1) never produces one of the classes")


def test_stage1_positions_uniform(nb):
    hits = np.zeros(10)
    y = np.zeros(10, dtype=np.int64)
    for s in range(3000):
        hits += nb.corrupt_labels(y, 0.3, 4, rng=s) != y
    expect = 3000 * 0.3
    chi2 = float(((hits - expect) ** 2 / (expect * 0.7)).sum())
    assert chi2 < stats.chi2.ppf(0.999, 9) + 5, (
        f"every position should be corrupted equally often (900 of 3000 times each); got {hits.astype(int).tolist()}")


# ---------------------------------------------------------------- stage 2: random features


def test_stage2_random_relu_features(nb):
    X = np.random.default_rng(0).normal(size=(50, 40))
    f = nb.RandomReLUFeatures(3000, rng=0)
    assert f.fit(X) is f, "fit returns self"
    W = f.W_
    assert W.shape == (40, 3000), f"W_ is (n_features_in, n_features) = (40, 3000), got {W.shape}"
    assert abs(W.std() - 1 / math.sqrt(40)) < 0.01 * (1 / math.sqrt(40)) * 3 and abs(W.mean()) < 0.003, (
        f"entries of W_ are N(0, 1/d) with d = 40: std {1 / math.sqrt(40):.4f}, got {W.std():.4f}")
    F = f.transform(X)
    assert F.shape == (50, 3000) and np.allclose(F, np.maximum(X @ W, 0) / math.sqrt(3000)), (
        "transform is relu(X @ W_) / sqrt(n_features)")
    g = nb.RandomReLUFeatures(3000, rng=0).fit(X)
    h = nb.RandomReLUFeatures(3000, rng=1).fit(X)
    assert np.array_equal(g.W_, W) and not np.allclose(h.W_, W), "rng fixes the draw: same seed same W_, another seed differs"
    assert nb.RandomReLUFeatures(20, rng=0).fit(X).transform(X[:3]).shape == (3, 20), "n_features sets the output width"


# ---------------------------------------------------------------- stage 3: minimum-norm and ridge solutions


def test_stage3_both_sides_of_the_threshold(nb):
    rng = np.random.default_rng(0)
    Phi = rng.normal(size=(30, 8))
    Y = rng.normal(size=(30, 3))
    B = nb.min_norm_least_squares(Phi, Y)
    assert B.shape == (8, 3) and np.allclose(B, np.linalg.lstsq(Phi, Y, rcond=None)[0]), (
        "with more points than features (p < n) and ridge 0 this is ordinary least squares")
    Phi = rng.normal(size=(10, 40))
    y = rng.normal(size=10)
    b = nb.min_norm_least_squares(Phi, y)
    assert b.shape == (40,) and np.allclose(Phi @ b, y, atol=1e-8), "with p > n the solution interpolates the data"
    P_row = np.linalg.pinv(Phi) @ Phi
    assert np.allclose(P_row @ b, b, atol=1e-8), (
        "of the infinitely many interpolating solutions, return the one with the smallest norm: it has no component "
        "in the null space of Phi (it lies in the row space). np.linalg.pinv or lstsq give it")
    Phi = np.hstack([rng.normal(size=(20, 4))] * 2)  # duplicated columns: rank 4, p = 8 < n = 20
    y = rng.normal(size=20)
    b = nb.min_norm_least_squares(Phi, y)
    assert np.allclose(b, np.linalg.pinv(Phi) @ y, atol=1e-8), (
        "with a rank-deficient design (duplicated columns) the normal equations are singular; the minimum-norm "
        "least-squares solution still exists (pinv)")


def test_stage3_ridge(nb):
    rng = np.random.default_rng(1)
    for n, p in ((30, 8), (8, 30)):
        Phi = rng.normal(size=(n, p))
        Y = rng.normal(size=(n, 2))
        B = nb.min_norm_least_squares(Phi, Y, ridge=0.5)
        want = np.linalg.solve(Phi.T @ Phi + 0.5 * np.eye(p), Phi.T @ Y)
        assert np.allclose(B, want, atol=1e-10), (
            f"ridge = 0.5 with n = {n}, p = {p}: B = (Phi^T Phi + ridge I)^-1 Phi^T Y (the ridge is not multiplied "
            "by n); for p > n the equivalent Phi^T (Phi Phi^T + ridge I)^-1 Y is cheaper")
    Phi = rng.normal(size=(8, 30))
    y = rng.normal(size=8)
    assert np.allclose(nb.min_norm_least_squares(Phi, y, ridge=1e-9), nb.min_norm_least_squares(Phi, y), atol=1e-5), (
        "as ridge -> 0 the ridge solution tends to the minimum-norm one")


# ---------------------------------------------------------------- stage 4: the double-descent curve


def _teacher(n, m, d=15, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(d, 3))
    X = rng.normal(size=(n + m, d)) / math.sqrt(d)
    logits = X @ w * 3 + 0.8 * rng.normal(size=(n + m, 3))
    y = logits.argmax(1)
    return X[:n], np.eye(3)[y[:n]], X[n:], y[n:]


def test_stage4_matches_your_pieces(nb):
    Xtr, Ytr, Xte, yte = _teacher(40, 200)
    out = bounded(nb.double_descent_curve, Xtr, Ytr, Xte, yte, [10, 80], n_seeds=2, what="double_descent_curve")
    assert set(out) >= {"test_error", "test_mse", "train_mse"}, f"return a dict with test_error, test_mse, train_mse; got {set(out)}"
    for k in ("test_error", "test_mse", "train_mse"):
        assert np.asarray(out[k]).shape == (2, 2), f"{k} is (len(widths), n_seeds) = (2, 2), got {np.asarray(out[k]).shape}"
    for i, p in enumerate((10, 80)):
        for s in range(2):
            f = nb.RandomReLUFeatures(p, rng=s).fit(Xtr)
            B = nb.min_norm_least_squares(f.transform(Xtr), Ytr)
            P = f.transform(Xte) @ B
            Ptr = f.transform(Xtr) @ B
            assert math.isclose(out["test_error"][i, s], np.mean(P.argmax(1) != yte)), (
                f"width {p}, seed {s}: test_error is the misclassification rate of argmax over the predicted one-hot "
                "scores, with features from RandomReLUFeatures(p, rng=s)")
            assert math.isclose(out["test_mse"][i, s], np.mean(np.sum((P - np.eye(3)[yte]) ** 2, 1))), (
                "test_mse is the squared error against the one-hot test labels, summed over classes, averaged over points")
            assert math.isclose(out["train_mse"][i, s], np.mean(np.sum((Ptr - Ytr) ** 2, 1)), abs_tol=1e-12), (
                "train_mse is the same quantity on the training set")
    assert out["train_mse"][1].max() < 1e-12, "with 80 features and 40 points the model interpolates: train_mse = 0"
    r = _run(nb.double_descent_curve, Xtr, Ytr, Xte, yte, [80], ridge=1.0, n_seeds=1)
    assert r["train_mse"][0, 0] > 1e-3, "ridge must be passed on to the solver: with ridge = 1 the fit no longer interpolates"


def test_stage4_peak_at_the_interpolation_threshold(nb):
    Xtr, Ytr, Xte, yte = _teacher(60, 400, seed=3)
    widths = [15, 30, 60, 120, 600]
    out = _run(nb.double_descent_curve, Xtr, Ytr, Xte, yte, widths, n_seeds=3)
    mse = np.median(out["test_mse"], axis=1)
    assert mse[2] > 3 * mse[0] and mse[2] > 3 * mse[4] and mse[4] < mse[3], (
        f"test MSE (median over 3 seeds) against width {widths}: {np.round(mse, 2).tolist()}. with ridge 0 it must "
        "spike at p = n = 60 and come down again for p >> n")


# ---------------------------------------------------------------- stage 5: implicit bias of gradient descent


def test_stage5_hand_steps_and_inputs(nb):
    X = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 1.0]])
    y = np.array([1.0, 0.0, 2.0])
    w = np.array([0.5, -0.5])
    w0 = w.copy()
    for _ in range(3):
        w = w - 0.1 * X.T @ (X @ w - y) / 3
    got = _run(nb.gd_least_squares, X, y, lr=0.1, steps=3, w0=w0)
    assert np.allclose(got, w), (
        f"three steps of w <- w - lr X^T (X w - y) / n from w0 give {w}; got {got}. the loss is (1/2n)||Xw - y||^2")
    assert np.array_equal(w0, [0.5, -0.5]), "w0 must not be modified"
    assert np.allclose(nb.gd_least_squares(X, y, lr=0.1, steps=0), 0.0), "w0=None starts at zero; steps=0 returns it"


def test_stage5_converges_to_minimum_norm(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 30))
    y = rng.normal(size=10)
    lr = 1.0 / (np.linalg.eigvalsh(X.T @ X / 10).max())
    w = bounded(nb.gd_least_squares, X, y, lr=lr, steps=5000, what="gd_least_squares")
    w_min = np.linalg.pinv(X) @ y
    assert np.allclose(w, w_min, atol=1e-6), (
        "from zero, gradient descent on an underdetermined least-squares problem converges to the MINIMUM-NORM "
        f"interpolant (pinv): distance {np.linalg.norm(w - w_min):.2e}")
    w0 = rng.normal(size=30)
    w = _run(nb.gd_least_squares, X, y, lr=lr, steps=5000, w0=w0)
    P = np.linalg.pinv(X) @ X
    assert np.allclose(w, w_min + (np.eye(30) - P) @ w0, atol=1e-6), (
        "from w0, gradient descent never changes the component of w0 in the null space of X: it converges to "
        "pinv(X) y + (I - P_row) w0")
