"""Checks for exercise.py (the perceptron).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The oracle is
a plain loop written out below (the perceptron is defined by its update rule), plus the
lesson's worked example by hand and Novikoff's bound as a property.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: converges in 2 epochs with w = (2, 1), b = 0
X_WE = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
Y_WE = np.array([1, -1, -1])


def _naive(X, t, max_epochs, shuffle, seed, fit_intercept=True):
    """the textbook loop, storing every iterate: returns final (w, b), averaged (w, b), mistakes."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    ws, bs, mistakes = [], [], []
    for _ in range(max_epochs):
        order = rng.permutation(n) if shuffle else np.arange(n)
        m = 0
        for i in order:
            if t[i] * (X[i] @ w + b) <= 0:
                w = w + t[i] * X[i]
                b = b + t[i] if fit_intercept else b
                m += 1
            ws.append(w)
            bs.append(b)
        mistakes.append(m)
        if m == 0:
            break
    return (w, b), (np.mean(ws, axis=0), float(np.mean(bs))), mistakes


def _separable(seed, n=80, d=3, gap=0.3):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(4 * n, d))
    s = X @ w_true + 0.4
    keep = np.abs(s) > gap
    X, s = X[keep][:n], s[keep][:n]
    return X, np.where(s > 0, 1, -1), w_true, 0.4


def _noisy(seed, n=120, d=4, flip=0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    t = np.where(X @ rng.normal(size=d) > 0, 1, -1)
    t[rng.random(n) < flip] *= -1
    return X, t


# ---------------------------------------------------------------- stage 1: one step


def test_stage1_hand_steps(nb):
    w, b, wrong = nb.perceptron_step(np.zeros(2), 0.0, X_WE[0], 1)
    assert wrong and np.allclose(w, [1.0, 1.0]) and b == 1.0, (
        f"from w = 0 every point scores 0, which counts as a mistake (y * score <= 0): w + y x = (1, 1), b + y = 1. "
        f"got w = {w}, b = {b}, mistake = {wrong}")
    w2, b2, wrong2 = nb.perceptron_step(w, b, X_WE[1], -1)
    assert wrong2 and np.allclose(w2, [2.0, 1.0]) and b2 == 0.0, (
        f"x = (-1, 0) scores -1 + 1 = 0, a mistake on the boundary: w - x = (2, 1), b - 1 = 0. got {w2}, {b2}")
    w3, b3, wrong3 = nb.perceptron_step(w2, b2, X_WE[2], -1)
    assert not wrong3 and np.allclose(w3, w2) and b3 == b2, "x = (0, -1) scores -1 with label -1: correct, no update"


def test_stage1_no_side_effects_and_no_intercept(nb):
    w = np.array([0.5, -0.5])
    w0 = w.copy()
    new_w, new_b, wrong = nb.perceptron_step(w, 2.0, np.array([1.0, 1.0]), -1, fit_intercept=False)
    assert wrong and new_b == 2.0, "with fit_intercept=False a mistake updates w but leaves b alone"
    assert np.allclose(new_w, [-0.5, -1.5]) and np.array_equal(w, w0), (
        "return a NEW w (w + y x): updating the caller's array in place (w += ...) corrupts it")


# ---------------------------------------------------------------- stage 2: Perceptron


def test_stage2_worked_example(nb):
    m = bounded(nb.Perceptron(shuffle=False).fit, X_WE, Y_WE, what="Perceptron.fit",
                hint="each epoch visits every example once; stop after max_epochs epochs")
    assert np.allclose(m.coef_, [2.0, 1.0]) and m.intercept_ == 0.0, (
        f"the lesson's trace ends at w = (2, 1), b = 0; got coef_ {m.coef_}, intercept_ {m.intercept_}")
    assert m.mistakes_ == [2, 0] and m.n_iter_ == 2 and m.n_updates_ == 2, (
        f"two mistakes in epoch 1, none in epoch 2, then stop: mistakes_ [2, 0], n_iter_ 2, n_updates_ 2; got "
        f"{m.mistakes_}, {m.n_iter_}, {m.n_updates_}")
    assert isinstance(m.intercept_, float) and np.asarray(m.coef_).shape == (2,)
    tie = m.predict(np.array([[1.0, -2.0], [1.0, -1.9]]))
    assert list(tie) == [-1, 1], (
        f"(1, -2) lies exactly on the boundary 2 x1 + x2 = 0: a score of 0 is NOT positive, so it gets classes_[0] "
        f"= -1, while (1, -1.9) scores 0.1 and gets +1; got {list(tie)}. use decision_function > 0, not >= 0")


@pytest.mark.parametrize("shuffle", [False, True])
def test_stage2_matches_the_update_rule(nb, shuffle):
    X, t = _noisy(1)
    m = bounded(nb.Perceptron(max_epochs=7, shuffle=shuffle, rng=3).fit, X, t, what="Perceptron.fit")
    (w, b), _, mistakes = _naive(X, t, 7, shuffle, 3)
    assert m.mistakes_ == mistakes, (
        f"mistakes per epoch should be {mistakes}, got {m.mistakes_}. "
        + ("each epoch visits the rows in the order rng_.permutation(n), with rng_ = as_rng(rng) created once in fit"
           if shuffle else "without shuffling, visit rows 0, 1, ..., n-1 every epoch"))
    assert np.allclose(m.coef_, w) and np.isclose(m.intercept_, b), "the final weights should match the update rule"


def test_stage2_separable_data_converge(nb):
    for seed in range(3):
        X, t, _, _ = _separable(seed)
        m = bounded(nb.Perceptron(rng=seed).fit, X, t, what="Perceptron.fit")
        assert m.score(X, t) == 1.0 and m.mistakes_[-1] == 0 and m.n_iter_ < 100, (
            f"on linearly separable data the perceptron must stop at an epoch with no mistakes and classify every "
            f"training point correctly; got training accuracy {m.score(X, t):.3f} after {m.n_iter_} epochs")


def test_stage2_labels_protocol_and_caps(nb):
    X, t, _, _ = _separable(4)
    labels = np.where(t > 0, "spam", "ham")
    m = nb.Perceptron(rng=0).fit(X, labels)
    assert list(m.classes_) == ["ham", "spam"] and set(m.predict(X)) <= {"ham", "spam"}, (
        "any two labels work: classes_ is sorted, classes_[1] ('spam') is the +1 class, predict returns labels")
    assert np.array_equal(m.predict(X), np.where(m.decision_function(X) > 0, "spam", "ham")), (
        "predict is classes_[1] where decision_function > 0, else classes_[0]")
    xor = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    m2 = bounded(nb.Perceptron(max_epochs=25, rng=0).fit, xor, [0, 0, 1, 1], what="Perceptron.fit",
                 hint="XOR is never separated: only max_epochs can stop the loop")
    assert m2.n_iter_ == 25 and len(m2.mistakes_) == 25, "XOR never gets an error-free epoch: it runs all max_epochs"
    assert nb.Perceptron().get_params() == {
        "max_epochs": 100, "averaged": False, "fit_intercept": True, "shuffle": True, "rng": None}
    m3 = nb.Perceptron(fit_intercept=False, rng=0).fit(X, t)
    assert m3.intercept_ == 0.0, "fit_intercept=False keeps the intercept at 0"
    with pytest.raises(ValueError):
        nb.Perceptron().fit(X[:6], [0, 1, 2, 0, 1, 2])
        pytest.fail("three classes should raise ValueError: this perceptron is binary")
    with pytest.raises(ValueError):
        m.decision_function(X[:, :2])
        pytest.fail("a different number of features than in fit should raise ValueError")
    with pytest.raises(NotImplementedError):
        m.predict_proba(X)
        pytest.fail("the perceptron has no probabilities: predict_proba should raise (Classifier's default)")


# ---------------------------------------------------------------- stage 3: averaging


def test_stage3_worked_example(nb):
    m = bounded(nb.Perceptron(shuffle=False, averaged=True).fit, X_WE, Y_WE, what="Perceptron.fit")
    assert np.allclose(m.coef_, [11 / 6, 1.0]) and np.isclose(m.intercept_, 1 / 6), (
        f"over the 6 visits, (w, b) is ((1, 1), 1) once and ((2, 1), 0) five times: the average is "
        f"((11/6, 1), 1/6); got coef_ {np.round(m.coef_, 4)}, intercept_ {m.intercept_:.4f}")


@pytest.mark.parametrize("seed", range(3))
def test_stage3_matches_naive_average(nb, seed):
    X, t = _noisy(10 + seed)
    m = bounded(nb.Perceptron(max_epochs=6, averaged=True, rng=seed).fit, X, t, what="Perceptron.fit")
    _, (wa, ba), _ = _naive(X, t, 6, True, seed)
    assert np.allclose(m.coef_, wa) and np.isclose(m.intercept_, ba), (
        f"averaged=True should give the mean of (w, b) over every visit (after its step, mistake or not) of every "
        f"epoch: expected {np.round(wa, 4)}, {ba:.4f}; got {np.round(m.coef_, 4)}, {m.intercept_:.4f}")
    plain = nb.Perceptron(max_epochs=6, averaged=False, rng=seed).fit(X, t)
    assert plain.mistakes_ == m.mistakes_, "averaging changes the returned weights, not the training run itself"


# ---------------------------------------------------------------- stage 4: the convergence theorem


def test_stage4_hand_values(nb):
    X = np.array([[3.0, 0.0], [0.0, 4.0], [-1.0, 0.0]])
    y = np.array([1, 1, -1])
    w, b = np.array([1.0, 1.0]), -0.5
    got = nb.margin(X, y, w, b)
    assert np.isclose(got, 1.5 / np.sqrt(2)), (
        f"the margin is min_i y_i (w.x_i + b) / ||w|| = min(2.5, 3.5, 1.5) / sqrt(2) = {1.5 / np.sqrt(2):.4f}; got {got}")
    got = nb.mistake_bound(X, y, w, b)
    assert np.isclose(got, 17.0), (
        "with augmented points [x, 1] and weights [w, b] = (1, 1, -0.5): R^2 = max ||[x, 1]||^2 = ||(0, 4, 1)||^2 = 17 "
        f"and gamma = 1.5 / ||(1, 1, -0.5)|| = 1.5 / 1.5 = 1, so the bound is 17; got {got}")
    assert nb.mistake_bound(X, y, np.array([-1.0, 0.0]), 0.0) == np.inf, (
        "a hyperplane that misclassifies a point has gamma <= 0: return np.inf")


def test_stage4_novikoff_holds(nb):
    for seed in range(4):
        X, t, w_true, b_true = _separable(20 + seed, gap=0.2)
        bound = nb.mistake_bound(X, t, w_true, b_true)
        for s in range(3):
            m = bounded(nb.Perceptron(max_epochs=1000, rng=s).fit, X, t, what="Perceptron.fit", budget=5_000_000)
            assert m.mistakes_[-1] == 0 and m.n_updates_ <= bound, (
                f"Novikoff: the perceptron makes at most (R/gamma)^2 = {bound:.0f} updates on separable data, whatever "
                f"the order; got {m.n_updates_}. check mistake_bound's augmentation, and that n_updates_ counts every "
                "update")
