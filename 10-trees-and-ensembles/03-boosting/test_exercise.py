"""Checks for exercise.py (boosting).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The reweighting is
checked by hand and by its defining property (the last weak learner looks like a coin flip on
the new weights); the classifier against scikit-learn's SAMME AdaBoost (an oracle only: with
continuous features the weak learners are the same trees, so the estimator weights agree to
rounding) and against the exact identity mean(exp(-y f / 2)) = prod 2 sqrt(err (1 - err)).
Under pytest, mlfs.trees' decision trees are the reference ones. Fits run under bounded.
"""
import numpy as np
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost
from sklearn.tree import DecisionTreeClassifier as SkTree

from mlfs.testing import bounded


def _fit(est, X, y, budget=20_000_000):
    return bounded(est.fit, X, y, budget=budget, what="AdaBoostClassifier.fit",
                   hint="the boosting loop should run at most n_estimators rounds")


def _data(seed, n=300, d=5, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] ** 2 + X[:, 1] + noise * rng.normal(size=n) > 1).astype(int)
    return X, y, rng


# ---------------------------------------------------------------- stage 1: one round


def test_stage1_worked_example(nb):
    y = np.array([1, 1, 1, -1, -1])
    h = np.array([1, 1, -1, -1, 1])
    alpha, w, err = nb.samme_update(y, h, np.full(5, 0.2))
    assert np.isclose(err, 0.4), f"two of five equally weighted points are misclassified: error 0.4, got {err}"
    assert np.isclose(alpha, np.log(1.5)), (
        f"alpha = learning_rate * ln((1 - err) / err) = ln(0.6 / 0.4) = {np.log(1.5):.4f}, got {alpha}")
    want = np.array([1 / 6, 1 / 6, 1 / 4, 1 / 6, 1 / 4])
    assert np.allclose(w, want), (
        f"the misclassified points are multiplied by exp(alpha) = 1.5 and everything is renormalized: expected "
        f"{np.round(want, 4).tolist()}, got {np.round(np.asarray(w), 4).tolist()}")
    alpha2, w2, _ = nb.samme_update(y, h, np.full(5, 0.2), learning_rate=0.5)
    assert np.isclose(alpha2, 0.5 * np.log(1.5)), "learning_rate scales alpha"
    r = np.sqrt(1.5)
    want2 = np.array([1, 1, r, 1, r]) / (3 + 2 * r)
    assert np.allclose(w2, want2), (
        f"with learning_rate=0.5 the SHRUNKEN alpha is used in the update too: misclassified weights are multiplied by "
        f"exp(0.5 ln 1.5) = sqrt(1.5) and renormalized, giving {np.round(want2, 4).tolist()}; got "
        f"{np.round(np.asarray(w2), 4).tolist()}")


def test_stage1_coin_flip_property(nb):
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.choice([-1, 1], size=40)
        h = np.where(rng.random(40) < 0.3, -y, y)
        w0 = rng.uniform(0.5, 2.0, size=40)
        alpha, w, err = nb.samme_update(y, h, w0)
        w = np.asarray(w)
        assert np.isclose(w.sum(), 1.0) and np.all(w > 0), "the new weights are positive and sum to 1"
        assert np.isclose(err, np.sum(w0[h != y]) / w0.sum()), "the error is weighted (and the weights normalized)"
        assert np.isclose(np.sum(w[h != y]), 0.5), (
            f"after the update, the misclassified points must carry exactly half the weight (the weak learner just "
            f"used is a coin flip on the new weights), got {np.sum(w[h != y]):.4f}")


def test_stage1_exponential_loss(nb):
    y = np.array([1.0, -1.0, 1.0])
    f = np.array([0.0, 1.0, 2.0])
    want = np.mean(np.exp([0.0, 1.0, -2.0]))
    assert np.isclose(nb.exponential_loss(y, f), want), f"mean(exp(-y f)) should be {want:.4f}"


# ---------------------------------------------------------------- stage 2: AdaBoost


def test_stage2_matches_sklearn_samme(nb):
    for seed, depth, lr in [(0, 1, 1.0), (1, 1, 1.0), (4, 1, 0.5)]:
        X, y, rng = _data(seed)
        ours = _fit(nb.AdaBoostClassifier(n_estimators=25, max_depth=depth, learning_rate=lr), X, y)
        sk = SkAdaBoost(estimator=SkTree(max_depth=depth), n_estimators=25, learning_rate=lr,
                        random_state=0).fit(X, y)
        a = np.asarray(ours.estimator_weights_)
        # stumps on continuous features don't tie; deeper trees can (the weights take few distinct values), so
        # they're checked through the exact training-bound identity below instead
        assert len(a) == 25 and np.allclose(a, sk.estimator_weights_), (
            f"estimator weights (depth {depth}, learning_rate {lr}) {np.round(a[:4], 4).tolist()}... differ from "
            f"SAMME's {np.round(sk.estimator_weights_[:4], 4).tolist()}...: each round fits a tree with the CURRENT "
            "weights (sample_weight=), then alpha = lr * ln((1 - err) / err), then misclassified weights *= exp(alpha)")
        f = np.asarray(ours.decision_function(X))
        assert np.allclose(f / a.sum(), sk.decision_function(X) / 2), (
            "decision_function = sum_m alpha_m h_m(x) with h = +1 for classes_[1] and -1 for classes_[0]")


def test_stage2_stops_when_no_better_than_chance(nb):
    X = np.ones((40, 2))
    y = np.arange(40) % 2  # constant features: the best tree is a coin flip, weighted error exactly 0.5
    try:
        bounded(nb.AdaBoostClassifier(n_estimators=5).fit, X, y, budget=2_000_000, what="AdaBoostClassifier.fit")
    except ValueError:
        return
    raise AssertionError("if the first weak learner's weighted error is >= 0.5 it can't help (alpha <= 0): raise "
                         "ValueError (later rounds with err >= 0.5 are discarded and stop the loop)")


def test_stage2_labels_and_proba(nb):
    X, y, _ = _data(3)
    labels = np.array(["ham", "spam"])[y]
    ada = nb.AdaBoostClassifier(n_estimators=10)
    assert _fit(ada, X, labels) is ada, "fit must return self"
    assert list(ada.classes_) == ["ham", "spam"], "classes_ = the two sorted labels"
    f = np.asarray(ada.decision_function(X))
    p = np.asarray(ada.predict_proba(X))
    assert p.shape == (len(y), 2) and np.allclose(p.sum(1), 1), "predict_proba: (n, 2), rows sum to 1"
    assert np.allclose(p[:, 1], 1 / (1 + np.exp(-f))), "P(classes_[1]) = sigmoid(decision_function)"
    assert np.array_equal(ada.predict(X), np.where(f > 0, "spam", "ham")), "predict: classes_[1] where f > 0"
    big = nb.AdaBoostClassifier(n_estimators=1).fit(X, labels)
    big.estimator_weights_ = np.array([800.0])  # a huge score must not overflow to nan
    assert np.all(np.isfinite(big.predict_proba(X))), "predict_proba must not overflow for large |f|"
    try:
        nb.AdaBoostClassifier(n_estimators=2).fit(X, np.arange(len(y)) % 3)
    except ValueError:
        pass
    else:
        raise AssertionError("AdaBoostClassifier is binary: three classes must raise ValueError")


def test_stage2_training_bound_is_exact(nb):
    X, y, _ = _data(4)
    ada = _fit(nb.AdaBoostClassifier(n_estimators=30, max_depth=2), X, y)
    err = np.asarray(ada.estimator_errors_)
    assert err.shape == (30,) and np.all((err > 0) & (err < 0.5)), "estimator_errors_: each round's weighted error"
    f = np.asarray(ada.decision_function(X))
    y_pm = 2 * y - 1
    lhs = np.mean(np.exp(-y_pm * f / 2))
    rhs = np.prod(2 * np.sqrt(err * (1 - err)))
    assert np.isclose(lhs, rhs), (
        f"with learning_rate=1, mean(exp(-y f / 2)) must equal prod 2 sqrt(err (1 - err)) exactly (on paper 2): "
        f"{lhs:.6f} vs {rhs:.6f}. check that the weights carried into each round are the previous round's "
        "updated weights")
    assert np.mean(np.sign(f) != y_pm) <= lhs, "the training error is below the exponential loss bound"


def test_stage2_boosting_beats_one_stump(nb):
    gains = []
    for seed in range(4):
        X, y, rng = _data(10 + seed, n=300)
        Xt, yt, _ = _data(100 + seed, n=2000)
        stump = SkTree(max_depth=1).fit(X, y).score(Xt, yt)
        gains.append(_fit(nb.AdaBoostClassifier(n_estimators=40), X, y).score(Xt, yt) - stump)
    assert np.median(gains) > 0.05, (
        f"40 boosted stumps should beat one stump clearly (a curved boundary needs many stumps); test accuracy "
        f"gains: {np.round(gains, 3).tolist()}")


def test_stage2_stops_on_perfect_learner(nb):
    X = np.arange(20.0)[:, None]
    y = (X[:, 0] > 9.5).astype(int)
    ada = _fit(nb.AdaBoostClassifier(n_estimators=10), X, y)
    assert len(ada.estimators_) == 1 and np.allclose(ada.estimator_weights_, [1.0]), (
        "a stump separates this data perfectly (weighted error 0): ln((1 - 0) / 0) is infinite, so keep that one "
        "tree with weight 1.0 and stop (scikit-learn does the same)")
    assert ada.score(X, y) == 1.0


# ---------------------------------------------------------------- stage 3: staged + margins


def test_stage3_staged_decision_function(nb):
    X, y, _ = _data(5)
    ada = _fit(nb.AdaBoostClassifier(n_estimators=12, max_depth=2), X, y)
    Xt = np.random.default_rng(1).normal(size=(50, X.shape[1]))
    staged = [np.asarray(f) for f in ada.staged_decision_function(Xt)]  # no copies: each stage must be a new array
    assert len(staged) == 12, f"one array per round: expected 12, got {len(staged)}"
    assert np.allclose(staged[-1], ada.decision_function(Xt)), "the last stage is the full decision_function"
    steps = np.diff(np.array([np.zeros(50)] + staged), axis=0)
    want = np.array(ada.estimator_weights_)[:, None]
    assert np.allclose(np.abs(steps), want), (
        "stage m adds alpha_m * h_m(x) with h_m in {-1, +1}: each step changes every score by exactly +-alpha_m. "
        "if every yielded array is the same object, copy it (f = f + ..., not f += ...)")


def test_stage3_margins(nb):
    X, y, _ = _data(6)
    labels = np.array(["a", "b"])[y]
    ada = _fit(nb.AdaBoostClassifier(n_estimators=20, max_depth=2), X, labels)
    m = np.asarray(ada.margins(X, labels))
    f = np.asarray(ada.decision_function(X))
    y_pm = np.where(labels == "b", 1, -1)
    assert m.shape == (len(y),) and np.allclose(m, y_pm * f / np.sum(ada.estimator_weights_)), (
        "margins = y f(x) / sum(alpha), with y = +1 for classes_[1] and -1 for classes_[0]")
    assert np.all(np.abs(m) <= 1 + 1e-12), "normalized margins lie in [-1, 1]"
    assert np.array_equal(m > 0, ada.predict(X) == labels), "a positive margin means a correct prediction"
