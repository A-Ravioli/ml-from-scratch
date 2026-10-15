"""Checks for exercise.py (metrics).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Most metrics
are compared with scikit-learn on random data with ties; the lesson's worked example is
checked by hand.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

# the lesson's worked example: 8 emails, 4 spam (label 1), scores from some model
Y_W = np.array([1, 1, 0, 1, 0, 0, 1, 0])
S_W = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1])
P_W = (S_W >= 0.5).astype(int)  # threshold 0.5 -> [1, 1, 1, 1, 1, 0, 0, 0]


def _tied_scores(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    y[:2] = [0, 1]
    s = np.round(rng.normal(size=n) + 0.8 * y, 1)  # rounding creates many ties
    return y, s


# ---------------------------------------------------------------- stage 1: counts


def test_stage1_worked_example(nb):
    C = np.asarray(nb.confusion_matrix(Y_W, P_W))
    assert C.tolist() == [[2, 2], [1, 3]], (
        f"rows are the truth, columns the prediction: [[TN, FP], [FN, TP]] = [[2, 2], [1, 3]]; got {C.tolist()}")
    assert np.isclose(nb.accuracy_score(Y_W, P_W), 0.625), "5 of 8 are right"
    assert np.isclose(nb.precision_score(Y_W, P_W), 0.6), "precision = TP / (TP + FP) = 3/5"
    assert np.isclose(nb.recall_score(Y_W, P_W), 0.75), "recall = TP / (TP + FN) = 3/4"
    assert np.isclose(nb.f1_score(Y_W, P_W), 2 / 3), "F1 = 2TP / (2TP + FP + FN) = 6/9"
    for v in (nb.accuracy_score(Y_W, P_W), nb.f1_score(Y_W, P_W)):
        assert isinstance(v, float), "metrics return Python floats"


@pytest.mark.parametrize("seed", range(3))
def test_stage1_match_sklearn_multiclass(nb, seed):
    import sklearn.metrics as skm
    rng = np.random.default_rng(seed)
    y = rng.choice(["ant", "bee", "cat", "dog"], size=60, p=[0.5, 0.3, 0.15, 0.05])
    p = np.where(rng.random(60) < 0.6, y, rng.choice(["ant", "bee", "cat", "dog"], size=60))
    assert np.array_equal(nb.confusion_matrix(y, p), skm.confusion_matrix(y, p)), "confusion_matrix should match sklearn"
    for avg in ("macro", "micro", "weighted"):
        for name in ("precision_score", "recall_score", "f1_score"):
            got = getattr(nb, name)(y, p, average=avg)
            want = getattr(skm, name)(y, p, average=avg, zero_division=0)
            assert np.isclose(got, want), (
                f"{name}(average={avg!r}) should be {want:.4f}, got {got:.4f}. macro = plain mean over classes, weighted "
                "= weighted by each class's TRUE count, micro = pool TP/FP/FN first; macro F1 averages the per-class F1s")


def test_stage1_labels_pos_label_and_zero_division(nb):
    import sklearn.metrics as skm
    y = np.array([2, 0, 2, 2, 1, 0])
    p = np.array([2, 2, 1, 2, 1, 0])
    Cn = np.asarray(nb.confusion_matrix([0, 0, 1], [0, 2, 1]))
    assert Cn.tolist() == [[1, 0, 1], [0, 1, 0], [0, 0, 0]], (
        "the default labels are the sorted union of y_true AND y_pred: a class that's only ever predicted (2) still "
        f"gets a row and a column; got {Cn.tolist()}")
    C = np.asarray(nb.confusion_matrix(y, p, labels=[2, 0]))
    assert C.tolist() == skm.confusion_matrix(y, p, labels=[2, 0]).tolist(), (
        "labels=[2, 0] sets the row/column order and drops pairs involving other labels")
    yb, pb = np.array(["no", "yes", "yes", "no"]), np.array(["yes", "yes", "no", "no"])
    assert np.isclose(nb.f1_score(yb, pb, pos_label="yes"), 0.5), "pos_label picks the class whose scores are reported"
    assert np.isclose(nb.recall_score(yb, pb, pos_label="no"), 0.5)
    assert nb.precision_score([1, 1, 0], [0, 0, 0]) == 0.0, (
        "the model never predicts the positive class: precision is 0/0, which these metrics report as 0.0")
    with pytest.raises(ValueError):
        nb.f1_score(y, p)
        pytest.fail("average='binary' with three labels should raise ValueError")
    with pytest.raises(ValueError):
        nb.f1_score(y, p, average="mean")
        pytest.fail("an unknown average should raise ValueError")
    with pytest.raises(ValueError):
        nb.accuracy_score([1, 0, 1], [1, 0])
        pytest.fail("arrays of different lengths should raise ValueError")


def test_stage1_confusion_matrix_is_vectorized(nb):
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 5, size=200_000), rng.integers(0, 5, size=200_000)
    C = bounded(nb.confusion_matrix, y, p, budget=50_000, what="confusion_matrix",
                hint="count all pairs at once (np.unique / np.bincount / np.add.at), not with a Python loop over rows.")
    assert np.asarray(C).sum() == 200_000


# ---------------------------------------------------------------- stage 2: ROC


def test_stage2_worked_example(nb):
    fpr, tpr, thr = (np.asarray(a) for a in nb.roc_curve(Y_W, S_W))
    assert fpr.shape == tpr.shape == thr.shape == (9,), (
        f"8 distinct scores give 9 ROC points, (0, 0) first (with threshold inf); got arrays of shapes {fpr.shape}, "
        f"{tpr.shape}, {thr.shape}. did you forget to prepend (0, 0), or keep one point per row instead of per distinct score?")
    assert np.allclose(fpr, [0, 0, 0, 0.25, 0.25, 0.5, 0.75, 0.75, 1.0]) and np.allclose(
        tpr, [0, 0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0, 1.0]), (
        f"expected the 9 ROC points of the worked example, starting at (0, 0); got fpr={fpr.tolist()}, tpr={tpr.tolist()}")
    assert thr[0] == np.inf and np.allclose(thr[1:], S_W), "thresholds[0] = inf, then the distinct scores, decreasing"
    assert np.isclose(nb.roc_auc_score(Y_W, S_W), 0.75), "AUC = 12 of 16 (positive, negative) pairs ranked correctly = 0.75"


def test_stage2_binary_clf_curve_ties(nb):
    fps, tps, thr = (np.asarray(a) for a in nb.binary_clf_curve([1, 0, 1, 0, 1], [0.8, 0.8, 0.5, 0.2, 0.2]))
    assert thr.shape == tps.shape == fps.shape == (3,), (
        f"3 distinct scores give 3 entries each, got shapes {thr.shape}, {tps.shape}, {fps.shape}: tied scores form ONE "
        "step, so keep only the last row of each run of equal sorted scores")
    assert np.allclose(thr, [0.8, 0.5, 0.2]) and tps.tolist() == [1, 2, 3] and fps.tolist() == [1, 1, 2], (
        "tied scores form ONE step: at threshold 0.8 both tied rows count (1 positive, 1 negative). expected "
        f"thresholds [0.8, 0.5, 0.2], tps [1, 2, 3], fps [1, 1, 2]; got {thr.tolist()}, {tps.tolist()}, {fps.tolist()}")


@pytest.mark.parametrize("seed", range(4))
def test_stage2_roc_matches_sklearn(nb, seed):
    from sklearn.metrics import roc_auc_score, roc_curve
    y, s = _tied_scores(80, seed)
    fpr, tpr, thr = (np.asarray(a) for a in nb.roc_curve(y, s))
    rf, rt, rth = roc_curve(y, s, drop_intermediate=False)
    assert fpr.shape == rf.shape, (
        f"expected {len(rf)} ROC points (one per distinct score, plus (0, 0)), got {len(fpr)}")
    assert np.allclose(fpr, rf) and np.allclose(tpr, rt) and np.allclose(thr[1:], rth[1:]), (
        "roc_curve should match sklearn.metrics.roc_curve(..., drop_intermediate=False): one point per DISTINCT score, "
        "ties as a single step")
    got, want = nb.roc_auc_score(y, s), roc_auc_score(y, s)
    assert np.isclose(got, want), (
        f"AUC should be {want:.4f}, got {got:.4f}. use the trapezoid rule, sum(diff(fpr) * (tpr[1:] + tpr[:-1]) / 2): "
        "a rectangle (left or right Riemann) sum gets tied blocks wrong")


@pytest.mark.parametrize("seed", range(4))
def test_stage2_auc_is_the_ranking_probability(nb, seed):
    y, s = _tied_scores(50, seed)
    pos, neg = s[y == 1], s[y == 0]
    mw = (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    got = nb.roc_auc_score(y, s)
    assert np.isclose(got, mw), (
        f"AUC must equal P(score of a random positive > score of a random negative), ties counting 1/2: {mw:.4f}; "
        f"got {got:.4f}. check that ties are one diagonal step (the trapezoid rule handles them)")
    assert np.isclose(nb.roc_auc_score(y, np.exp(3 * s)), got), "AUC only depends on the ranking: a monotone transform can't change it"
    assert np.isclose(nb.roc_auc_score(y, -s), 1 - got), "reversing the ranking gives 1 - AUC"


def test_stage2_labels_and_errors(nb):
    y = np.array(["ham", "spam", "spam", "ham"])
    assert np.isclose(nb.roc_auc_score(y, [0.1, 0.9, 0.3, 0.4]), 0.75), "the larger label ('spam') is the positive class"
    with pytest.raises(ValueError):
        nb.roc_auc_score([1, 1, 1], [0.2, 0.5, 0.9])
        pytest.fail("AUC with only one class is undefined: raise ValueError")


# ---------------------------------------------------------------- stage 3: precision-recall


def test_stage3_worked_example(nb):
    want = (1 + 1 + 0.75 + 4 / 7) / 4
    got = nb.average_precision_score(Y_W, S_W)
    assert np.isclose(got, want), (
        f"AP = mean of the precisions where each positive is reached: (1 + 1 + 3/4 + 4/7) / 4 = {want:.4f}; got {got:.4f}")
    p, r, t = (np.asarray(a) for a in nb.precision_recall_curve(Y_W, S_W))
    assert p[-1] == 1.0 and r[-1] == 0.0 and len(p) == len(r) == len(t) + 1, (
        "the curve ends with the extra point precision = 1, recall = 0, which has no threshold")
    assert np.all(np.diff(t) > 0) and np.all(np.diff(r) <= 0), "thresholds increase and recall decreases along the arrays"


@pytest.mark.parametrize("seed", range(4))
def test_stage3_match_sklearn(nb, seed):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    y, s = _tied_scores(70, seed)
    p, r, t = (np.asarray(a) for a in nb.precision_recall_curve(y, s))
    rp, rr, rt = precision_recall_curve(y, s)
    assert p.shape == rp.shape and np.allclose(p, rp) and np.allclose(r, rr) and np.allclose(t, rt), (
        "precision_recall_curve should match sklearn's (one point per distinct score, reversed, plus (1, 0))")
    got, want = nb.average_precision_score(y, s), average_precision_score(y, s)
    assert np.isclose(got, want), (
        f"average precision should be {want:.4f} (step-wise, no interpolation), got {got:.4f}. a trapezoid area "
        "over the PR curve is a different (optimistic) number")


# ---------------------------------------------------------------- stage 4: log loss and regression metrics


def test_stage4_log_loss(nb):
    from sklearn.metrics import log_loss
    got = nb.log_loss([1, 0], [0.8, 0.3])
    assert np.isclose(got, -(np.log(0.8) + np.log(0.7)) / 2), (
        f"binary y_prob is P(class 1): the true classes get 0.8 and 1 - 0.3 = 0.7, so the loss is "
        f"-(log 0.8 + log 0.7)/2 = 0.2899; got {got}")
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(4), size=50)
    y = rng.integers(0, 4, size=50)
    assert np.isclose(nb.log_loss(y, P), log_loss(y, P, labels=[0, 1, 2, 3])), "multiclass log_loss should match sklearn"
    big = nb.log_loss([1, 0], [0.0, 0.5])
    assert np.isfinite(big) and np.isclose(big, (-np.log(1e-15) - np.log(0.5)) / 2), (
        "a probability of 0 for the true class is clipped to eps = 1e-15, so the loss is large but finite (about 34.5)")
    with pytest.raises(ValueError):
        nb.log_loss([0, 3], np.full((2, 3), 1 / 3))
        pytest.fail("label 3 with only 3 probability columns should raise ValueError")


def test_stage4_regression_metrics(nb):
    import sklearn.metrics as skm
    rng = np.random.default_rng(1)
    y = rng.normal(size=30)
    p = y + rng.normal(scale=0.5, size=30)
    for name in ("mean_squared_error", "mean_absolute_error", "r2_score"):
        got, want = getattr(nb, name)(y, p), getattr(skm, name)(y, p)
        assert np.isclose(got, want), f"{name} should be {want:.5f}, got {got}"
    assert nb.r2_score([3.0, 3.0], [3.0, 3.0]) == 1.0 and nb.r2_score([3.0, 3.0], [2.0, 3.0]) == 0.0, (
        "constant y_true: R^2 is 1.0 for a perfect prediction and 0.0 otherwise")
    assert np.isclose(nb.r2_score([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -3.0), "R^2 can be negative: here 1 - 8/2 = -3"


# ---------------------------------------------------------------- stage 5: calibration


def test_stage5_calibration_hand_example(nb):
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    pt, pp, c = (np.asarray(a) for a in nb.calibration_curve(y, p, n_bins=3))
    assert np.allclose(pt, [0, 0.5, 1]) and np.allclose(pp, [0.2, 0.525, 0.85]) and c.tolist() == [3, 2, 4], (
        f"with 3 bins: {{.1, .2, .3}} -> (0, 0.2), {{.4, .65}} -> (0.5, 0.525), {{.7, .8, .9, 1}} -> (1, 0.85); got "
        f"prob_true {pt.tolist()}, prob_pred {pp.tolist()}, counts {c.tolist()}")
    ece = nb.expected_calibration_error(y, p, n_bins=3)
    assert np.isclose(ece, (3 * 0.2 + 2 * 0.025 + 4 * 0.15) / 9), f"ECE = (3*0.2 + 2*0.025 + 4*0.15)/9 = 0.1389, got {ece}"


@pytest.mark.parametrize("seed", range(3))
def test_stage5_match_sklearn_and_edges(nb, seed):
    from sklearn.calibration import calibration_curve
    rng = np.random.default_rng(seed)
    p = np.round(rng.random(200), 2)  # many values exactly on bin edges (0.1, 0.2, ...)
    y = (rng.random(200) < p).astype(int)
    pt, pp, c = (np.asarray(a) for a in nb.calibration_curve(y, p, n_bins=10))
    rt, rp = calibration_curve(y, p, n_bins=10)
    assert pt.shape == rt.shape and np.allclose(pt, rt) and np.allclose(pp, rp), (
        "calibration_curve should match sklearn's: bin = np.searchsorted(edges[1:-1], p), so 0.1 belongs to the FIRST "
        "bin; only non-empty bins are returned")
    assert c.sum() == 200


@pytest.mark.parametrize("seed", range(3))
def test_stage5_ece_behaviour(nb, seed):
    rng = np.random.default_rng(seed)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p).astype(int)
    good = nb.expected_calibration_error(y, p)
    over = nb.expected_calibration_error(y, np.clip(0.5 + 1.8 * (p - 0.5), 0, 1))  # overconfident
    assert good < 0.02, f"perfectly calibrated probabilities should have ECE close to 0, got {good:.3f}"
    assert over > 0.1, f"overconfident probabilities should have a large ECE, got {over:.3f}"
    P = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]])
    yy = np.array([0, 1, 2, 0, 2])
    want = (2 * 0.05 + 2 * 0.15 + 1 * 0.2) / 5
    got = nb.expected_calibration_error(yy, P, n_bins=4)
    assert np.isclose(got, want), (
        "multiclass ECE bins the top-class confidence (0.7, 0.5, 0.8, 0.6, 0.4) against whether the argmax was right "
        "(1, 0, 1, 0, 1). with 4 bins: {0.4, 0.5} has accuracy 0.5 and confidence 0.45, {0.6, 0.7} has 0.5 and 0.65, "
        f"{{0.8}} has 1 and 0.8, so ECE = (2*0.05 + 2*0.15 + 1*0.2)/5 = {want:.2f}; got {got}")


def test_stage5_reliability_diagram(nb):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    fig, ax = plt.subplots()
    try:
        out = nb.reliability_diagram(y, p, ax=ax)
        assert out is ax, "reliability_diagram should draw on the ax it's given and return it"
        assert len(ax.lines) >= 2, "draw at least the diagonal and the calibration curve"
        assert np.allclose(ax.get_xlim(), (0, 1)) and np.allclose(ax.get_ylim(), (0, 1)), "both axes should run from 0 to 1"
        own = nb.reliability_diagram(y, p)
        assert hasattr(own, "plot"), "with ax=None, make a new figure and return its Axes"
        plt.close(own.figure)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- stage 6: scorers


class _Toy:
    """a fitted 'classifier' with string labels and fixed outputs."""

    classes_ = np.array(["ham", "spam"])

    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.proba, self.proba])

    def predict(self, X):
        return self.classes_[(self.proba >= 0.5).astype(int)]

    def decision_function(self, X):
        return np.log(self.proba / (1 - self.proba))


def test_stage6_make_scorer(nb):
    from sklearn.metrics import log_loss, roc_auc_score
    proba = np.array([0.9, 0.2, 0.6, 0.4, 0.7])
    y = np.array(["spam", "ham", "ham", "spam", "spam"])
    est, X = _Toy(proba), np.zeros((5, 1))
    acc = nb.make_scorer(lambda a, b: np.mean(a == b))(est, X, y)
    assert np.isclose(acc, 0.6), "response='predict' compares y with estimator.predict(X)"
    neg = nb.make_scorer(lambda a, b: 2.0, greater_is_better=False)(est, X, y)
    assert neg == -2.0, "greater_is_better=False negates the metric, so higher is always better"
    ll = nb.make_scorer(nb.log_loss, greater_is_better=False, response="predict_proba")(est, X, y)
    assert np.isclose(ll, -log_loss((y == "spam").astype(int), proba)), (
        "predict_proba scorers turn y into column indices of classes_ (ham -> 0, spam -> 1) and use column 1 for a "
        "binary problem, so log_loss sees int labels")
    auc = nb.make_scorer(nb.roc_auc_score, response="decision_function")(est, X, y)
    assert np.isclose(auc, roc_auc_score(y == "spam", proba)), "response='decision_function' scores with decision_function(X)"
    with pytest.raises(ValueError):
        nb.make_scorer(nb.accuracy_score, response="predict_log_proba")
        pytest.fail("an unknown response should raise ValueError as soon as make_scorer is called")
    with pytest.raises(ValueError):
        nb.make_scorer(nb.log_loss, response="predict_proba")(est, X, np.array(["spam", "eggs", "ham", "ham", "ham"]))
        pytest.fail("a label the estimator never saw ('eggs') can't be scored with predict_proba: raise ValueError")


def test_stage6_get_scorer(nb):
    import sklearn.metrics as skm
    proba = np.array([0.9, 0.2, 0.6, 0.4, 0.7])
    y = np.array(["spam", "ham", "ham", "spam", "spam"])
    est, X = _Toy(proba), np.zeros((5, 1))
    yb = (y == "spam").astype(int)
    assert np.isclose(nb.get_scorer("accuracy")(est, X, y), 0.6)
    assert np.isclose(nb.get_scorer("roc_auc")(est, X, y), skm.roc_auc_score(yb, proba))
    assert np.isclose(nb.get_scorer("neg_log_loss")(est, X, y), -skm.log_loss(yb, proba))

    class Reg:
        def predict(self, X):
            return np.array([1.0, 2.0, 4.0])

    yr = np.array([1.0, 3.0, 2.0])
    pr = Reg().predict(None)
    assert np.isclose(nb.get_scorer("neg_mean_squared_error")(Reg(), None, yr), -skm.mean_squared_error(yr, pr))
    assert np.isclose(nb.get_scorer("neg_mean_absolute_error")(Reg(), None, yr), -skm.mean_absolute_error(yr, pr))
    assert np.isclose(nb.get_scorer("r2")(Reg(), None, yr), skm.r2_score(yr, pr))
    yi = np.array([1, 0, 0, 1, 1])

    class Bin:
        def predict(self, X):
            return np.array([1, 0, 1, 0, 1])

    assert np.isclose(nb.get_scorer("f1")(Bin(), None, yi), skm.f1_score(yi, Bin().predict(None)))
    with pytest.raises(ValueError):
        nb.get_scorer("auc")
        pytest.fail("an unknown scorer name should raise ValueError")
