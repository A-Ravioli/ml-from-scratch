"""Checks for exercise.py (softmax regression).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Oracles are
finite differences, scikit-learn's (multinomial) LogisticRegression with C = 1 / (n * alpha),
and a plain minibatch-SGD loop written out below.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: x = (-1, 0, 1) with labels (0, 1, 2), an intercept, Theta = 0
A_WE = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
Y_WE = np.eye(3)


def _blobs(seed, n=150, d=2, k=3, spread=1.0):
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(k) / k
    centers = 2.0 * np.column_stack([np.cos(angles), np.sin(angles), np.zeros((k, max(d - 2, 0)))])[:, :d]
    y = rng.integers(0, k, n)
    X = centers[y] + spread * rng.normal(size=(n, d))
    return (X - X.mean(0)) / X.std(0), y  # standardized, so lr = 1 is a stable step


def _design(X, alpha=0.0):
    A = np.column_stack([np.ones(len(X)), X])
    return A, np.concatenate([[0.0], np.full(X.shape[1], alpha)])


def _onehot(y, k):
    return np.eye(k)[y]


def _num_grad(f, T, h=1e-6):
    G = np.zeros_like(T)
    for idx in np.ndindex(T.shape):
        E = np.zeros_like(T)
        E[idx] = h
        G[idx] = (f(T + E) - f(T - E)) / (2 * h)
    return G


# ---------------------------------------------------------------- stage 1: loss and gradient


def test_stage1_worked_example(nb):
    T0 = np.zeros((2, 3))
    loss = nb.softmax_loss(T0, A_WE, Y_WE, np.zeros(2))
    assert isinstance(loss, float) and np.isclose(loss, np.log(3)), (
        f"at Theta = 0 every class gets probability 1/3, so the loss is log 3 = 1.0986; got {loss}")
    G = np.asarray(nb.softmax_gradient(T0, A_WE, Y_WE, np.zeros(2)))
    assert G.shape == (2, 3) and np.allclose(G, [[0, 0, 0], [1 / 3, 0, -1 / 3]]), (
        f"the gradient is A^T (P - Y) / n with P = 1/3 everywhere: intercept row (0, 0, 0), slope row "
        f"(1/3, 0, -1/3); got {np.round(G, 4).tolist()}")


@pytest.mark.parametrize("seed", range(3))
def test_stage1_gradient_check(nb, seed):
    X, y = _blobs(seed, n=40, k=4)
    A, pen = _design(X, alpha=0.3)
    Y = _onehot(y, 4)
    T = np.random.default_rng(seed + 9).normal(size=(3, 4))
    got = np.asarray(nb.softmax_gradient(T, A, Y, pen))
    want = _num_grad(lambda M: nb.softmax_loss(M, A, Y, pen), T)
    assert got.shape == (3, 4) and np.allclose(got, want, rtol=1e-5, atol=1e-7), (
        "softmax_gradient disagrees with finite differences of your softmax_loss. the data term is "
        "A^T (softmax(A Theta) - Y) / n; the penalty adds penalty[:, None] * Theta (one ridge strength per ROW)")


def test_stage1_invariances_and_stability(nb):
    X, y = _blobs(3, n=30)
    A, _ = _design(X)
    Y = _onehot(y, 3)
    T = np.random.default_rng(0).normal(size=(3, 3))
    shift = np.random.default_rng(1).normal(size=(3, 1))
    assert np.isclose(nb.softmax_loss(T + shift, A, Y, np.zeros(3)), nb.softmax_loss(T, A, Y, np.zeros(3))), (
        "adding the same vector to every class's column adds the same number to every score, which softmax ignores: "
        "the unpenalized loss must not change")
    t = np.random.default_rng(2).normal(size=3)
    two = np.column_stack([np.zeros(3), t])
    logistic = np.mean(np.logaddexp(0, A @ t) - (y == 1) * (A @ t))
    Y2 = _onehot((y == 1).astype(int), 2)
    assert np.isclose(nb.softmax_loss(two, A, Y2, np.zeros(3)), logistic), (
        "with k = 2 and the first column fixed at 0, softmax cross-entropy is exactly the logistic loss")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            big = nb.softmax_loss(1e4 * np.array([[0.0, 0.0], [1.0, -1.0]]), A_WE[:2], np.eye(2)[[0, 0]], np.zeros(2))
        except FloatingPointError as e:
            pytest.fail(f"overflow with scores of +-1e4 ({e}): use log_softmax / softmax from mlfs.utils")
    assert np.isclose(big, (2e4 + np.log(2)) / 2), (
        f"row 1 is wrong by a score difference of 2e4 (cost ~2e4), row 2 is a tie (cost log 2): the mean is "
        f"~1e4; got {big}")


# ---------------------------------------------------------------- stage 2: the estimator


@pytest.mark.parametrize("alpha", [0.01, 0.3])
def test_stage2_matches_sklearn(nb, alpha):
    from sklearn.linear_model import LogisticRegression

    X, y = _blobs(4, n=120, spread=1.5)
    m = bounded(nb.SoftmaxRegression(alpha=alpha, lr=1.0, max_iters=20_000, tol=1e-9).fit, X, y,
                what="SoftmaxRegression.fit", budget=20_000_000)
    ref = LogisticRegression(C=1 / (len(y) * alpha), tol=1e-12, max_iter=100_000).fit(X, y)
    assert np.allclose(m.predict_proba(X), ref.predict_proba(X), atol=1e-6), (
        "predict_proba should match scikit-learn's multinomial logistic regression with C = 1/(n alpha): the "
        "objective is mean cross-entropy + (alpha/2) ||coef_||_F^2, intercept unpenalized")
    assert np.allclose(m.coef_, ref.coef_.T, atol=1e-5), (
        f"coef_ has shape (d, k) = (2, 3): column c holds class c's weights. expected {np.round(ref.coef_.T, 4)}, "
        f"got {np.round(np.asarray(m.coef_), 4)}")
    assert m.converged_ and m.n_iter_ < 20_000


def test_stage2_protocol(nb):
    X, y = _blobs(5)
    labels = np.array(["cat", "dog", "emu"])[y]
    m = nb.SoftmaxRegression(alpha=0.01, lr=1.0, max_iters=500).fit(X, labels)
    assert list(m.classes_) == ["cat", "dog", "emu"], "classes_ are the sorted labels"
    assert np.asarray(m.coef_).shape == (2, 3) and np.asarray(m.intercept_).shape == (3,), (
        f"coef_ is (d, k) and intercept_ (k,); got {np.shape(m.coef_)} and {np.shape(m.intercept_)}")
    S = np.asarray(m.decision_function(X[:5]))
    P = np.asarray(m.predict_proba(X[:5]))
    assert S.shape == (5, 3) and np.allclose(S, X[:5] @ m.coef_ + m.intercept_), "decision_function = X coef_ + intercept_"
    assert np.allclose(P.sum(axis=1), 1) and np.allclose(P, np.exp(S) / np.exp(S).sum(1, keepdims=True)), (
        "predict_proba is the softmax of the scores, rows summing to 1")
    assert set(m.predict(X)) <= {"cat", "dog", "emu"} and m.score(X, labels) > 0.6
    assert nb.SoftmaxRegression().get_params() == {"alpha": 0.0, "lr": 0.1, "max_iters": 1000, "batch_size": None,
                                                   "tol": 1e-06, "fit_intercept": True, "rng": None}
    m0 = nb.SoftmaxRegression(fit_intercept=False, max_iters=50).fit(X, y)
    assert np.allclose(m0.intercept_, 0.0) and np.asarray(m0.intercept_).shape == (3,)
    with pytest.raises(ValueError):
        nb.SoftmaxRegression().fit(X, np.zeros(len(X)))
        pytest.fail("a single class should raise ValueError")
    with pytest.raises(ValueError):
        m.predict(X[:, :1])
        pytest.fail("a different number of features should raise ValueError")


def test_stage2_two_classes_is_logistic(nb):
    from sklearn.linear_model import LogisticRegression

    X, y = _blobs(6, k=2, spread=2.0)
    m = bounded(nb.SoftmaxRegression(alpha=0.2, lr=1.0, max_iters=20_000, tol=1e-9).fit, X, y,
                what="SoftmaxRegression.fit", budget=20_000_000)
    ref = LogisticRegression(C=1 / (len(y) * 0.1), tol=1e-12, max_iter=100_000).fit(X, y)
    assert np.allclose(m.predict_proba(X), ref.predict_proba(X), atol=1e-6), (
        "with 2 classes, softmax regression with penalty alpha is binary logistic regression with alpha / 2 (the "
        "lesson explains why), so the probabilities must agree")
    assert np.allclose(m.coef_[:, 1] - m.coef_[:, 0], ref.coef_[0], atol=1e-5)


# ---------------------------------------------------------------- stage 3: minibatch SGD


def _sgd_oracle(A, Y, pen, lr, batch, epochs, seed):
    gen = np.random.default_rng(seed)
    T = np.zeros((A.shape[1], Y.shape[1]))
    for _ in range(epochs):
        order = gen.permutation(len(A))
        for start in range(0, len(A), batch):
            r = order[start:start + batch]
            P = np.exp(A[r] @ T - (A[r] @ T).max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            T = T - lr * (A[r].T @ (P - Y[r]) / len(r) + pen[:, None] * T)
    return T


def test_stage3_matches_plain_sgd(nb):
    X, y = _blobs(7, n=103)
    A, pen = _design(X, alpha=0.05)
    m = bounded(nb.SoftmaxRegression(alpha=0.05, lr=0.3, batch_size=10, max_iters=4, tol=0.0, rng=11).fit, X, y,
                what="SoftmaxRegression(batch_size=10).fit")
    T = _sgd_oracle(A, _onehot(y, 3), pen, 0.3, 10, 4, 11)
    assert m.n_iter_ == 4 and not m.converged_, f"with tol = 0 it runs all max_iters = 4 epochs; n_iter_ = {m.n_iter_}"
    assert np.allclose(m.intercept_, T[0]) and np.allclose(m.coef_, T[1:]), (
        "after 4 epochs the weights should equal plain minibatch SGD: each epoch, rng_ = as_rng(rng) (made once) "
        "shuffles via minibatch_indices(n, batch_size, shuffle=True, rng=rng_), and each batch takes one step "
        "Theta -= lr * (batch gradient, penalty included). 103 rows in batches of 10: the last batch has 3 rows")


def test_stage3_converges_and_stops(nb):
    X, y = _blobs(8, n=200)
    full = nb.SoftmaxRegression(alpha=0.01, lr=1.0, max_iters=5000, tol=1e-8).fit(X, y)
    sgd = bounded(nb.SoftmaxRegression(alpha=0.01, lr=0.05, batch_size=16, max_iters=60, tol=0.0, rng=0).fit, X, y,
                  what="SoftmaxRegression(batch_size=16).fit", budget=5_000_000)
    A, pen = _design(X, alpha=0.01)
    Y = _onehot(y, 3)
    gap = (nb.softmax_loss(np.vstack([sgd.intercept_, sgd.coef_]), A, Y, pen)
           - nb.softmax_loss(np.vstack([full.intercept_, full.coef_]), A, Y, pen))
    assert gap < 0.02, (
        f"60 epochs of small-step SGD should land near the full-batch optimum: its loss is {gap:.4f} above it")


def test_stage3_stops_on_the_full_gradient(nb):
    X, y = _blobs(9, n=120)
    A, pen = _design(X, alpha=0.05)
    Y = _onehot(y, 3)
    norms = []
    for epochs in (1, 2):
        T = _sgd_oracle(A, Y, pen, 0.3, 8, epochs, 5)
        P = np.exp(A @ T - (A @ T).max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        norms.append(np.linalg.norm(A.T @ (P - Y) / len(A) + pen[:, None] * T))
    tol = norms[1] * 1.0001  # met after epoch 2, not after epoch 1 (the norm is still falling)
    assert norms[0] > tol
    m = bounded(nb.SoftmaxRegression(alpha=0.05, lr=0.3, batch_size=8, max_iters=50, tol=tol, rng=5).fit, X, y,
                what="SoftmaxRegression(batch_size=8).fit")
    assert m.n_iter_ == 2 and m.converged_, (
        f"after each epoch, compare the FULL gradient norm (all rows, penalty included) with tol: here it drops below "
        f"tol after epoch 2, so n_iter_ should be 2 with converged_ True; got n_iter_ {m.n_iter_}, converged_ "
        f"{m.converged_}. (the last minibatch's gradient is noise around the full one: never stop on it)")


# ---------------------------------------------------------------- stage 4: one-vs-rest


def test_stage4_one_vs_rest(nb):
    X, y = _blobs(9, n=120, k=3)
    base = nb.SoftmaxRegression(alpha=0.05, lr=1.0, max_iters=3000, tol=1e-8)
    ovr = bounded(nb.OneVsRest(base).fit, X, y, what="OneVsRest.fit", budget=20_000_000)
    assert not hasattr(base, "coef_"), "OneVsRest must fit CLONES of the estimator, never the estimator itself"
    assert len(ovr.estimators_) == 3 and list(ovr.classes_) == [0, 1, 2]
    cols = []
    for c in range(3):
        b = nb.SoftmaxRegression(alpha=0.05, lr=1.0, max_iters=3000, tol=1e-8).fit(X, (y == c).astype(int))
        cols.append(b.predict_proba(X)[:, 1])
    P = np.column_stack(cols)
    P /= P.sum(axis=1, keepdims=True)
    got = np.asarray(ovr.predict_proba(X))
    assert got.shape == (120, 3) and np.allclose(got, P, atol=1e-8), (
        "column c is the c-th binary model's P(positive), and each row is divided by its sum")
    assert ovr.score(X, y) > 0.7
