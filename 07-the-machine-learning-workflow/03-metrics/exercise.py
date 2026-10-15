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
    # metrics · build

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
    raise NotImplementedError("stage1: implement accuracy_score")


@app.function
def confusion_matrix(y_true, y_pred, *, labels=None):
    """stage 1: the (k, k) int matrix C with C[i, j] = #(true label is labels[i] and predicted
    label is labels[j]).

    labels: the row/column order; default = the sorted union of the labels in y_true and
    y_pred (np.unique of both). pairs involving a label not in `labels` are not counted.
    rows are the TRUTH, columns the PREDICTION.
    """
    raise NotImplementedError("stage1: implement confusion_matrix")


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
    raise NotImplementedError("stage1: implement precision_recall_fscore")


@app.function
def precision_score(y_true, y_pred, *, average="binary", pos_label=1):
    """stage 1: precision_recall_fscore(...)[0]."""
    raise NotImplementedError("stage1: implement precision_score")


@app.function
def recall_score(y_true, y_pred, *, average="binary", pos_label=1):
    """stage 1: precision_recall_fscore(...)[1]."""
    raise NotImplementedError("stage1: implement recall_score")


@app.function
def f1_score(y_true, y_pred, *, average="binary", pos_label=1):
    """stage 1: precision_recall_fscore(...)[2]."""
    raise NotImplementedError("stage1: implement f1_score")


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
    raise NotImplementedError("stage2: implement binary_clf_curve")


@app.function
def roc_curve(y_true, y_score, *, pos_label=1):
    """stage 2: the ROC curve, (fpr, tpr, thresholds), one point per distinct score.

    fpr = fps / #negatives and tpr = tps / #positives from binary_clf_curve, with the point
    (0, 0) prepended and thresholds[0] = np.inf (nothing is predicted positive above every
    score). ties are one step, never a staircase. raises ValueError if y_true has no
    positives or no negatives (the curve is undefined).
    """
    raise NotImplementedError("stage2: implement roc_curve")


@app.function
def roc_auc_score(y_true, y_score):
    """stage 2: the area under the ROC curve by the trapezoid rule, a float.

    y_true must have exactly two distinct labels (else ValueError); the LARGER one is the
    positive class. equals the probability that a random positive scores higher than a random
    negative, counting ties as 1/2.
    """
    raise NotImplementedError("stage2: implement roc_auc_score")


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
    raise NotImplementedError("stage3: implement precision_recall_curve")


@app.function
def average_precision_score(y_true, y_score):
    """stage 3: AP = sum_k (R_k - R_{k-1}) P_k over the precision-recall curve (a step
    function, no interpolation), as a float. y_true must have exactly two labels (else
    ValueError); the larger one is positive."""
    raise NotImplementedError("stage3: implement average_precision_score")


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
    raise NotImplementedError("stage4: implement log_loss")


@app.function
def mean_squared_error(y_true, y_pred):
    """stage 4: mean((y_true - y_pred)^2), a float."""
    raise NotImplementedError("stage4: implement mean_squared_error")


@app.function
def mean_absolute_error(y_true, y_pred):
    """stage 4: mean(|y_true - y_pred|), a float."""
    raise NotImplementedError("stage4: implement mean_absolute_error")


@app.function
def r2_score(y_true, y_pred):
    """stage 4: 1 - SS_res / SS_tot (as in Regressor.score). constant y_true: 1.0 if the
    prediction is perfect, else 0.0."""
    raise NotImplementedError("stage4: implement r2_score")


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
    raise NotImplementedError("stage5: implement calibration_curve")


@app.function
def expected_calibration_error(y_true, y_prob, *, n_bins=10):
    """stage 5: ECE = sum over non-empty bins of (count / n) * |accuracy - confidence|.

    binary: y_prob (n,) = P(class 1); "accuracy" is the bin's fraction of positives and
    "confidence" its mean probability (the calibration_curve of y_true and y_prob).
    multiclass: y_prob (n, k); use the top-class confidence max_j P[i, j] and whether the
    top class was right (argmax == y_true, labels 0..k-1), and bin those. returns a float.
    """
    raise NotImplementedError("stage5: implement expected_calibration_error")


@app.function
def reliability_diagram(y_true, y_prob, *, n_bins=10, ax=None):
    """stage 5: plot the calibration curve with the diagonal and a count histogram.

    on `ax` (a new figure if None): the line y = x ("perfectly calibrated"), the points
    (prob_pred, prob_true) from calibration_curve joined by a line, and, on a twin y-axis, a
    light histogram of how many predictions fall in each bin. axes limits [0, 1] on both,
    labelled, with the ECE in the title. a 2-D y_prob is plotted as top-class confidence
    versus accuracy (as in expected_calibration_error). returns the main Axes.
    """
    raise NotImplementedError("stage5: implement reliability_diagram")


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
    raise NotImplementedError("stage6: implement make_scorer")


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
    raise NotImplementedError("stage6: implement get_scorer")


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
