"""Checks for capstone.py (predict house prices): functional tests plus metric thresholds.

All data here is synthetic and generated deterministically: "houses" with a few features on
wildly different scales (like real ones) and a price that is linear in them plus noise.
Nothing is downloaded.
"""
import numpy as np
import pytest

SCALES = np.array([1.0, 50.0, 0.01, 1000.0])
TRUE_W = np.array([0.8, 0.004, 30.0, -0.0002])  # weights in raw units
TRUE_B = 3.0


def _houses(seed=0, n=400):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 4))
    base[:, 1] += 0.5 * base[:, 0]                  # mildly correlated features
    X = base * SCALES + np.array([5.0, 200.0, 0.1, 3000.0])
    y = X @ TRUE_W + TRUE_B + 0.2 * rng.normal(size=n)
    return X, y


def _split(X, y, n_train=300):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _lstsq(X, y):
    A = np.column_stack([X, np.ones(len(y))])
    theta = np.linalg.lstsq(A, y, rcond=None)[0]
    return theta[:-1], theta[-1]


# ---------------------------------------------------------------- stage 1: standardizing


def test_stage1_train_statistics_only(nb):
    X, y = _houses()
    Xtr, _, Xte, _ = _split(X, y)
    Ztr, Zte, mean, std = (np.asarray(a) for a in nb.standardize_features(Xtr, Xte))
    assert mean.shape == (4,) and std.shape == (4,), (
        f"mean and std should have one value per feature, shape (4,); got {mean.shape} and {std.shape}. "
        "use axis=0 (down the columns).")
    assert np.allclose(mean, Xtr.mean(axis=0)) and np.allclose(std, Xtr.std(axis=0)), (
        "mean and std must come from X_train only (the test set mustn't influence them). "
        + ("yours look like statistics of train and test together." if np.allclose(mean, X.mean(axis=0)) else ""))
    assert np.allclose(Zte, (Xte - Xtr.mean(axis=0)) / Xtr.std(axis=0)), (
        "Z_test must be standardized with the TRAINING mean and std, not its own")
    assert np.allclose(Ztr.mean(axis=0), 0, atol=1e-12) and np.allclose(Ztr.std(axis=0), 1), (
        "every column of Z_train should have mean 0 and standard deviation 1")


def test_stage1_rejects_constant_feature(nb):
    X, y = _houses()
    X[:, 2] = 7.0
    with pytest.raises(ValueError):
        nb.standardize_features(X[:300], X[300:])


# ---------------------------------------------------------------- stage 2: model and loss


def test_stage2_predict_and_mse_by_hand(nb):
    X = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
    w, b = np.array([0.5, -1.0]), 2.0
    got = np.asarray(nb.predict(X, w, b))
    assert got.shape == (3,) and np.allclose(got, [0.5, 3.0, 2.5]), (
        f"X w + b: [0.5 - 2 + 2, 0 + 1 + 2, 1.5 - 1 + 2] = [0.5, 3, 2.5], got {got}")
    y = np.array([1.5, 3.0, 0.5])
    got = nb.mse(X, y, w, b)
    assert np.isclose(got, (1 + 0 + 4) / 3), (
        f"residuals [-1, 0, 2] give MSE (1 + 0 + 4) / 3 = 1.6667, got {got}"
        + (": you summed instead of averaging." if np.isclose(got, 5.0) else ""))


def test_stage2_rectangular_shapes(nb):
    X, y = _houses(1, n=50)
    w = np.array([0.3, -0.2, 1.1, 0.01])
    assert np.allclose(nb.predict(X, w, -1.0), X @ w - 1.0), "predict should compute X @ w + b for every row"
    assert np.isclose(nb.mse(X, y, w, -1.0), np.mean((X @ w - 1.0 - y) ** 2))


# ---------------------------------------------------------------- stage 3: the gradient


def test_stage3_matches_finite_differences(nb):
    X, y = _houses(2, n=60)
    Z = (X - X.mean(0)) / X.std(0)
    w, b = np.array([0.4, -1.3, 0.7, 2.1]), 0.9
    gw, gb = nb.mse_grad(Z, y, w, b)
    gw = np.asarray(gw)
    assert gw.shape == (4,), f"grad_w has one partial derivative per weight: shape (4,), got {gw.shape}"
    loss = lambda w_, b_: np.mean((Z @ w_ + b_ - y) ** 2)
    h = 1e-6
    fd_w = np.array([(loss(w + h * e, b) - loss(w - h * e, b)) / (2 * h) for e in np.eye(4)])
    fd_b = (loss(w, b + h) - loss(w, b - h)) / (2 * h)
    assert np.allclose(gw, fd_w, rtol=1e-6, atol=1e-7), (
        f"grad_w differs from finite differences: expected {np.round(fd_w, 5)}, got {np.round(gw, 5)}. "
        + ("it looks like X r instead of X^T r, or a missing factor 2 / n." if not np.allclose(gw, fd_w, rtol=0.1) else ""))
    assert np.isclose(gb, fd_b, rtol=1e-6), f"grad_b should be {fd_b:.6f} (= (2/n) sum r), got {gb:.6f}"


def test_stage3_zero_at_the_optimum(nb):
    X, y = _houses(3)
    Z = (X - X.mean(0)) / X.std(0)
    w, b = _lstsq(Z, y)
    gw, gb = nb.mse_grad(Z, y, w, b)
    assert np.all(np.abs(np.asarray(gw)) < 1e-9) and abs(gb) < 1e-9, (
        "at the least-squares solution every partial derivative should be (numerically) zero")


# ---------------------------------------------------------------- stage 4: gradient descent


def test_stage4_loss_curve(nb):
    X, y = _houses(4)
    Z = (X - X.mean(0)) / X.std(0)
    w, b, losses = nb.fit_gd(Z, y, lr=0.1, n_steps=50)
    losses = np.asarray(losses)
    assert losses.shape == (51,), f"n_steps = 50 should give 51 losses (start + one per step), got {losses.shape}"
    assert np.isclose(losses[0], np.mean(y**2)), (
        f"gradient descent starts at w = 0, b = 0, where the loss is mean(y^2) = {np.mean(y ** 2):.4f}; got {losses[0]:.4f}")
    assert np.all(np.diff(losses) <= 1e-12), "with lr = 0.1 on standardized features the loss should never go up"
    assert np.isclose(losses[-1], np.mean((Z @ np.asarray(w) + b - y) ** 2)), "losses[-1] is the loss at the returned (w, b)"


def test_stage4_converges_to_least_squares(nb):
    X, y = _houses(5)
    Z = (X - X.mean(0)) / X.std(0)
    w, b, losses = nb.fit_gd(Z, y)          # the defaults: lr = 0.1, n_steps = 500
    w_ref, b_ref = _lstsq(Z, y)
    best = np.mean((Z @ w_ref + b_ref - y) ** 2)
    assert np.asarray(losses).shape == (501,), "the default n_steps is 500"
    assert losses[-1] - best < 1e-8, (
        f"with the defaults (lr = 0.1, 500 steps) on standardized features, gradient descent should reach the "
        f"least-squares loss {best:.6f} to within 1e-8; got {losses[-1]:.6f}")
    assert np.allclose(w, w_ref, atol=1e-4) and abs(b - b_ref) < 1e-4, "the weights should match least squares"


def test_stage4_learning_rate_is_used(nb):
    X, y = _houses(6)
    Z = (X - X.mean(0)) / X.std(0)
    _, _, slow = nb.fit_gd(Z, y, lr=0.001, n_steps=20)
    _, _, fast = nb.fit_gd(Z, y, lr=0.1, n_steps=20)
    assert slow[-1] > 2 * fast[-1], (
        f"after 20 steps, lr = 0.001 should be far behind lr = 0.1 (losses {slow[-1]:.3f} vs {fast[-1]:.3f}): "
        "use the lr argument")


# ---------------------------------------------------------------- stage 5: the closed form, and the thresholds


def test_stage5_closed_form_matches_lstsq(nb):
    X, y = _houses(7)
    Z = (X - X.mean(0)) / X.std(0)
    w, b = nb.fit_closed_form(Z, y)
    w_ref, b_ref = _lstsq(Z, y)
    assert np.asarray(w).shape == (4,), f"w should have shape (4,), got {np.asarray(w).shape}"
    assert np.allclose(w, w_ref, atol=1e-9) and np.isclose(b, b_ref), (
        f"the normal equations (A^T A) theta = A^T y with A = [X | 1] should give w = {np.round(w_ref, 5)}, "
        f"b = {b_ref:.5f}; got w = {np.round(np.asarray(w), 5)}, b = {b:.5f}. "
        "did you append the column of ones? (without it, b is forced to 0.)")


def test_stage5_recovers_the_true_model_in_raw_units(nb):
    X, y = _houses(8, n=2000)
    w, b = nb.fit_closed_form(X, y)   # raw, unstandardized features: the closed form doesn't care
    assert np.allclose(w, TRUE_W, rtol=0.1, atol=1e-4) and abs(b - TRUE_B) < 0.5, (
        f"with 2,000 noisy houses the fitted weights should be close to the true ones {TRUE_W}; got {np.asarray(w)}")


def test_stage5_heldout_r2_threshold(nb):
    X, y = _houses(9, n=500)
    Xtr, ytr, Xte, yte = _split(X, y, 400)
    Ztr, Zte, _, _ = nb.standardize_features(Xtr, Xte)
    w, b, _ = nb.fit_gd(Ztr, ytr)
    pred = np.asarray(nb.predict(Zte, w, b))
    r2 = 1 - np.mean((pred - yte) ** 2) / np.var(yte)
    baseline = np.mean((ytr.mean() - yte) ** 2)
    assert r2 >= 0.9, f"the full pipeline (standardize on train, gradient descent, predict) should reach held-out R² >= 0.90, got {r2:.3f}"
    assert np.mean((pred - yte) ** 2) < 0.2 * baseline, "the model should beat 'predict the training mean' by a wide margin"
