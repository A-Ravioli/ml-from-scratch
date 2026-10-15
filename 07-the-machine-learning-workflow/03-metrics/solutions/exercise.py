import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # metrics · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. everything here becomes
    `mlfs.metrics`, the scoreboard for the rest of the course.

    | stage | you build |
    |---|---|
    | 1 | `accuracy_score`, `confusion_matrix`, `precision_recall_fscore`, `precision_score`, `recall_score`, `f1_score` → `mlfs.metrics` |
    | 2 | `binary_clf_curve`, `roc_curve`, `roc_auc_score`: ranking quality at every threshold → `mlfs.metrics` |
    | 3 | `precision_recall_curve`, `average_precision_score`: the view that cares about the rare class → `mlfs.metrics` |
    | 4 | `log_loss`, `mean_squared_error`, `mean_absolute_error`, `r2_score` → `mlfs.metrics` |
    | 5 | `calibration_curve`, `expected_calibration_error`, `reliability_diagram`: do the probabilities mean what they say? → `mlfs.metrics` |
    | 6 | `make_scorer`, `get_scorer`: turn any metric into a `scorer(estimator, X, y)` for cross-validation → `mlfs.metrics` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def accuracy_score(y_true, y_pred):
    """stage 1: the fraction of positions where y_pred == y_true, as a float. labels may be
    any type; the two arrays must have the same length (else ValueError)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}")
    return float(np.mean(y_true == y_pred))


@app.function
def confusion_matrix(y_true, y_pred, *, labels=None):
    """stage 1: the (k, k) int matrix C with C[i, j] = #(true label is labels[i] and predicted
    label is labels[j]).

    labels: the row/column order; default = the sorted union of the labels in y_true and
    y_pred (np.unique of both). pairs involving a label not in `labels` are not counted.
    rows are the TRUTH, columns the PREDICTION.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}")
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    k = len(labels)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]

    def position(a):  # index of each entry of a in `labels`, and whether it's there at all
        i = np.minimum(np.searchsorted(sorted_labels, a), k - 1)
        return order[i], sorted_labels[i] == a

    t, t_ok = position(y_true)
    p, p_ok = position(y_pred)
    keep = t_ok & p_ok
    return np.bincount(t[keep] * k + p[keep], minlength=k * k).reshape(k, k)


@app.function
def precision_recall_fscore(y_true, y_pred, *, average="binary", pos_label=1):
    """stage 1: (precision, recall, f1) as floats, with every 0/0 counted as 0.0.

    for one class c: TP = #(pred c and true c), FP = #(pred c, true not c),
    FN = #(pred not c, true c); precision = TP / (TP + FP), recall = TP / (TP + FN),
    F1 = 2 TP / (2 TP + FP + FN) (the harmonic mean of precision and recall).
      "binary":   the scores of class pos_label. more than two distinct labels in
                  y_true and y_pred together raises ValueError (pick an average).
      "macro":    the unweighted mean of the per-class scores over the sorted union of labels.
      "weighted": the per-class scores weighted by support (the number of TRUE rows of each class).
      "micro":    pool TP, FP, FN over all classes first, then apply the formulas.
    any other average raises ValueError.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def div(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    if average == "binary":
        labels = np.unique(np.concatenate([y_true, y_pred]))
        if len(labels) > 2:
            raise ValueError(f"average='binary' needs at most 2 labels, got {len(labels)}: choose macro/micro/weighted")
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return float(div(tp, tp + fp)), float(div(tp, tp + fn)), float(div(2 * tp, 2 * tp + fp + fn))
    if average not in ("macro", "weighted", "micro"):
        raise ValueError(f"average must be 'binary', 'macro', 'micro' or 'weighted', got {average!r}")
    C = confusion_matrix(y_true, y_pred)
    tp = np.diag(C)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp
    if average == "micro":
        TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
        return float(div(TP, TP + FP)), float(div(TP, TP + FN)), float(div(2 * TP, 2 * TP + FP + FN))
    p, r, f = div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)
    if average == "macro":
        return float(p.mean()), float(r.mean()), float(f.mean())
    w = C.sum(axis=1) / C.sum()
    return float(p @ w), float(r @ w), float(f @ w)


@app.function
def precision_score(y_true, y_pred, *, average="binary", pos_label=1):
    """stage 1: precision_recall_fscore(...)[0]."""
    return precision_recall_fscore(y_true, y_pred, average=average, pos_label=pos_label)[0]


@app.function
def recall_score(y_true, y_pred, *, average="binary", pos_label=1):
    """stage 1: precision_recall_fscore(...)[1]."""
    return precision_recall_fscore(y_true, y_pred, average=average, pos_label=pos_label)[1]


@app.function
def f1_score(y_true, y_pred, *, average="binary", pos_label=1):
    """stage 1: precision_recall_fscore(...)[2]."""
    return precision_recall_fscore(y_true, y_pred, average=average, pos_label=pos_label)[2]


@app.cell(hide_code=True)
def _():
    checks.run(1, accuracy_score=accuracy_score, confusion_matrix=confusion_matrix,
               precision_recall_fscore=precision_recall_fscore, precision_score=precision_score,
               recall_score=recall_score, f1_score=f1_score)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def binary_clf_curve(y_true, y_score, *, pos_label=1):
    """stage 2: cumulative counts at every distinct threshold, from the highest score down.

    with the scores sorted in decreasing order and thresholds = the DISTINCT scores
    (decreasing), returns (fps, tps, thresholds): tps[i] (fps[i]) = the number of positives
    (negatives) with score >= thresholds[i]. tied scores form one step. "positive" means
    y_true == pos_label. all three arrays have one entry per distinct score.
    """
    y = np.asarray(y_true) == pos_label
    s = np.asarray(y_score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D arrays of the same length")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # the last row of each run of equal scores
    tps = np.cumsum(y)[last]
    fps = (last + 1) - tps
    return fps, tps, s[last]


@app.function
def roc_curve(y_true, y_score, *, pos_label=1):
    """stage 2: the ROC curve, (fpr, tpr, thresholds), one point per distinct score.

    fpr = fps / #negatives and tpr = tps / #positives from binary_clf_curve, with the point
    (0, 0) prepended and thresholds[0] = np.inf (nothing is predicted positive above every
    score). ties are one step, never a staircase. raises ValueError if y_true has no
    positives or no negatives (the curve is undefined).
    """
    fps, tps, thresholds = binary_clf_curve(y_true, y_score, pos_label=pos_label)
    P, N = tps[-1], fps[-1]
    if P == 0 or N == 0:
        raise ValueError("the ROC curve needs both positive and negative examples")
    return np.r_[0.0, fps / N], np.r_[0.0, tps / P], np.r_[np.inf, thresholds]


@app.function
def roc_auc_score(y_true, y_score):
    """stage 2: the area under the ROC curve by the trapezoid rule, a float.

    y_true must have exactly two distinct labels (else ValueError); the LARGER one is the
    positive class. equals the probability that a random positive scores higher than a random
    negative, counting ties as 1/2.
    """
    classes = np.unique(np.asarray(y_true))
    if len(classes) != 2:
        raise ValueError(f"roc_auc_score needs exactly 2 classes in y_true, got {len(classes)}")
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=classes[1])
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


@app.cell(hide_code=True)
def _():
    checks.run(2, binary_clf_curve=binary_clf_curve, roc_curve=roc_curve, roc_auc_score=roc_auc_score)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def precision_recall_curve(y_true, y_score, *, pos_label=1):
    """stage 3: (precision, recall, thresholds), scikit-learn's layout.

    from binary_clf_curve: precision = tps / (tps + fps) and recall = tps / #positives at each
    distinct threshold. reverse all three so that thresholds INCREASE (and recall decreases),
    then append the point precision = 1, recall = 0 (which has no threshold), so precision and
    recall have one more entry than thresholds. raises ValueError if there are no positives.
    """
    fps, tps, thresholds = binary_clf_curve(y_true, y_score, pos_label=pos_label)
    if tps[-1] == 0:
        raise ValueError("the precision-recall curve needs at least one positive example")
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return np.r_[precision[::-1], 1.0], np.r_[recall[::-1], 0.0], thresholds[::-1]


@app.function
def average_precision_score(y_true, y_score):
    """stage 3: AP = sum_k (R_k - R_{k-1}) P_k over the precision-recall curve (a step
    function, no interpolation), as a float. y_true must have exactly two labels (else
    ValueError); the larger one is positive."""
    classes = np.unique(np.asarray(y_true))
    if len(classes) != 2:
        raise ValueError(f"average_precision_score needs exactly 2 classes in y_true, got {len(classes)}")
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=classes[1])
    return float(max(0.0, -np.sum(np.diff(recall) * precision[:-1])))


@app.cell(hide_code=True)
def _():
    checks.run(3, precision_recall_curve=precision_recall_curve, average_precision_score=average_precision_score)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def log_loss(y_true, y_prob, *, eps=1e-15):
    """stage 4: the mean negative log-likelihood (cross-entropy) in nats, a float.

    y_true: int labels 0..k-1 (bools work too). y_prob: (n,) = P(class 1) for a binary
    problem, or (n, k) with one column per class. the probability given to each row's TRUE
    class is clipped to [eps, 1 - eps] before the log, so a confident mistake costs about
    -log(eps) ~ 34.5 instead of infinity. a label outside 0..k-1 raises ValueError.
    """
    y = np.asarray(y_true).astype(int)
    P = np.asarray(y_prob, dtype=float)
    if P.ndim == 1:
        P = np.column_stack([1.0 - P, P])
    if len(P) != len(y):
        raise ValueError(f"y_true has {len(y)} rows but y_prob has {len(P)}")
    if y.min() < 0 or y.max() >= P.shape[1]:
        raise ValueError(f"labels must be 0..{P.shape[1] - 1} to index the probability columns")
    p = np.clip(P[np.arange(len(y)), y], eps, 1 - eps)
    return float(-np.mean(np.log(p)))


@app.function
def mean_squared_error(y_true, y_pred):
    """stage 4: mean((y_true - y_pred)^2), a float."""
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


@app.function
def mean_absolute_error(y_true, y_pred):
    """stage 4: mean(|y_true - y_pred|), a float."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


@app.function
def r2_score(y_true, y_pred):
    """stage 4: 1 - SS_res / SS_tot (as in Regressor.score). constant y_true: 1.0 if the
    prediction is perfect, else 0.0."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res, ss_tot = float(np.sum((y - p) ** 2)), float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


@app.cell(hide_code=True)
def _():
    checks.run(4, log_loss=log_loss, mean_squared_error=mean_squared_error,
               mean_absolute_error=mean_absolute_error, r2_score=r2_score)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def calibration_curve(y_true, y_prob, *, n_bins=10):
    """stage 5: binned calibration of binary probabilities.

    y_true (n,) of 0/1 (or bool; anything else raises ValueError), y_prob (n,) = P(class 1)
    in [0, 1] (else ValueError). bins: n_bins equal-width bins on [0, 1]; a probability's bin
    is np.searchsorted(edges[1:-1], p) with edges = np.linspace(0, 1, n_bins + 1) (so a value
    exactly on an inner edge goes to the lower bin). for every NON-EMPTY bin, in bin order:
    prob_true = the fraction of positives, prob_pred = the mean predicted probability,
    counts = the number of rows. returns (prob_true, prob_pred, counts).
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_prob, dtype=float)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("calibration_curve needs binary labels 0/1")
    if p.min() < 0 or p.max() > 1:
        raise ValueError("probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.searchsorted(edges[1:-1], p)
    counts = np.bincount(b, minlength=n_bins)
    pos = np.bincount(b, weights=y, minlength=n_bins)
    tot = np.bincount(b, weights=p, minlength=n_bins)
    keep = counts > 0
    return pos[keep] / counts[keep], tot[keep] / counts[keep], counts[keep]


@app.function
def expected_calibration_error(y_true, y_prob, *, n_bins=10):
    """stage 5: ECE = sum over non-empty bins of (count / n) * |accuracy - confidence|.

    binary: y_prob (n,) = P(class 1); "accuracy" is the bin's fraction of positives and
    "confidence" its mean probability (the calibration_curve of y_true and y_prob).
    multiclass: y_prob (n, k); use the top-class confidence max_j P[i, j] and whether the
    top class was right (argmax == y_true, labels 0..k-1), and bin those. returns a float.
    """
    P = np.asarray(y_prob, dtype=float)
    y = np.asarray(y_true)
    if P.ndim == 2:
        correct = (np.argmax(P, axis=1) == y.astype(int)).astype(float)
        prob_true, prob_pred, counts = calibration_curve(correct, P.max(axis=1), n_bins=n_bins)
    else:
        prob_true, prob_pred, counts = calibration_curve(y, P, n_bins=n_bins)
    return float(np.sum(counts * np.abs(prob_true - prob_pred)) / counts.sum())


@app.function
def reliability_diagram(y_true, y_prob, *, n_bins=10, ax=None):
    """stage 5: plot the calibration curve with the diagonal and a count histogram.

    on `ax` (a new figure if None): the line y = x ("perfectly calibrated"), the points
    (prob_pred, prob_true) from calibration_curve joined by a line, and, on a twin y-axis, a
    light histogram of how many predictions fall in each bin. axes limits [0, 1] on both,
    labelled, with the ECE in the title. a 2-D y_prob is plotted as top-class confidence
    versus accuracy (as in expected_calibration_error). returns the main Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    P = np.asarray(y_prob, dtype=float)
    y = np.asarray(y_true)
    if P.ndim == 2:
        y, P = (np.argmax(P, axis=1) == y.astype(int)).astype(float), P.max(axis=1)
    prob_true, prob_pred, counts = calibration_curve(y, P, n_bins=n_bins)
    ece = expected_calibration_error(y, P, n_bins=n_bins)
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfectly calibrated")
    ax.plot(prob_pred, prob_true, "o-", label="model")
    hist = ax.twinx()
    hist.hist(P, bins=np.linspace(0, 1, n_bins + 1), color="gray", alpha=0.2)
    hist.set_ylabel("count")
    ax.set_zorder(hist.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability", ylabel="observed frequency",
           title=f"reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    return ax


@app.cell(hide_code=True)
def _():
    checks.run(5, calibration_curve=calibration_curve, expected_calibration_error=expected_calibration_error,
               reliability_diagram=reliability_diagram)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.function
def make_scorer(metric, *, greater_is_better=True, response="predict"):
    """stage 6: wrap metric(y_true, y_out) as scorer(estimator, X, y) -> float.

    response picks what the fitted estimator is asked for:
      "predict":           y_out = estimator.predict(X)
      "predict_proba":     P = estimator.predict_proba(X); y_out = P[:, 1] if P has 2
                           columns, else P. and y is replaced by its column index in
                           estimator.classes_ (so string labels work with log_loss, and the
                           positive class of a binary problem is classes_[1]); a label that
                           isn't in classes_ raises ValueError.
      "decision_function": y_out = estimator.decision_function(X)
    any other response raises ValueError (when make_scorer is called). the result is
    float(metric(...)), negated if greater_is_better is False, so a scorer is always
    "higher is better".
    """
    if response not in ("predict", "predict_proba", "decision_function"):
        raise ValueError(f"response must be 'predict', 'predict_proba' or 'decision_function', got {response!r}")
    sign = 1.0 if greater_is_better else -1.0

    def scorer(estimator, X, y):
        y = np.asarray(y)
        if response == "predict":
            out = estimator.predict(X)
        elif response == "decision_function":
            out = estimator.decision_function(X)
        else:
            out = np.asarray(estimator.predict_proba(X))
            classes = np.asarray(estimator.classes_)
            idx = np.searchsorted(classes, y)
            if np.any(idx >= len(classes)) or np.any(classes[np.minimum(idx, len(classes) - 1)] != y):
                raise ValueError("y contains a label the estimator never saw in fit (not in classes_)")
            y = idx
            if out.ndim == 2 and out.shape[1] == 2:
                out = out[:, 1]
        return sign * float(metric(y, out))

    return scorer


@app.function
def get_scorer(name):
    """stage 6: a ready-made scorer by name.

      "accuracy"                 accuracy_score on predict
      "f1"                       f1_score (binary, pos_label=1) on predict
      "roc_auc"                  roc_auc_score on predict_proba
      "neg_log_loss"             -log_loss on predict_proba
      "neg_mean_squared_error"   -mean_squared_error on predict
      "neg_mean_absolute_error"  -mean_absolute_error on predict
      "r2"                       r2_score on predict
    any other name raises ValueError listing the valid ones.
    """
    table = {
        "accuracy": (accuracy_score, True, "predict"),
        "f1": (f1_score, True, "predict"),
        "roc_auc": (roc_auc_score, True, "predict_proba"),
        "neg_log_loss": (log_loss, False, "predict_proba"),
        "neg_mean_squared_error": (mean_squared_error, False, "predict"),
        "neg_mean_absolute_error": (mean_absolute_error, False, "predict"),
        "r2": (r2_score, True, "predict"),
    }
    if name not in table:
        raise ValueError(f"unknown scorer {name!r}; valid names: {sorted(table)}")
    metric, greater, response = table[name]
    return make_scorer(metric, greater_is_better=greater, response=response)


@app.cell(hide_code=True)
def _():
    checks.run(6, make_scorer=make_scorer, get_scorer=get_scorer, accuracy_score=accuracy_score,
               log_loss=log_loss, roc_auc_score=roc_auc_score)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
