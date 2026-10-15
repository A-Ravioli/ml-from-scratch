import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.base import DummyClassifier, StumpClassifier  # yours, from topic 01
    from mlfs.utils import sigmoid  # yours, from ch 00

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def one_feature_data(n, prevalence, sep, rng):
    """one feature x: N(0, 1) for negatives, N(sep, 1) for positives; label 1 with probability
    `prevalence`. returns (x as a column (n, 1), y)."""
    y = (rng.random(n) < prevalence).astype(int)
    x = rng.normal(size=n) + sep * y
    return x.reshape(-1, 1), y


@app.function
def report(y, pred, score=None):
    """one row of metrics, as a dict, from YOUR mlfs.metrics functions."""
    row = {"accuracy": ex.accuracy_score(y, pred), "precision": ex.precision_score(y, pred),
           "recall": ex.recall_score(y, pred), "F1": ex.f1_score(y, pred)}
    if score is not None:
        row["ROC AUC"] = ex.roc_auc_score(y, score)
        row["avg precision"] = ex.average_precision_score(y, score)
    return row


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # metrics · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: 99% accuracy

    a fraud detector sees one number per transaction. on fraudulent transactions (1% of
    them) it tends to be higher: $x \sim \mathcal N(2.5, 1)$ instead of $\mathcal N(0, 1)$.
    20,000 transactions to train on, 20,000 to test on. three models: the `most_frequent`
    dummy, your `StumpClassifier` (ERM for accuracy), and "flag the top 2% of $x$" (a
    threshold chosen to catch more fraud). the last two columns use $x$ itself as the score.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what accuracy, precision and recall will the dummy and the stump get? will the stump "
        "beat the dummy's accuracy by much?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _Xtr, _ytr = one_feature_data(20_000, 0.01, 2.5, _rng)
    _Xte, _yte = one_feature_data(20_000, 0.01, 2.5, _rng)
    _dummy = DummyClassifier().fit(_Xtr, _ytr)
    _stump = StumpClassifier().fit(_Xtr, _ytr)
    _t2 = np.quantile(_Xtr[:, 0], 0.98)
    _rows = {
        "dummy (most frequent)": report(_yte, _dummy.predict(_Xte), np.zeros(len(_yte))),
        f"stump (x > {_stump.threshold_:.2f})": report(_yte, _stump.predict(_Xte), _Xte[:, 0]),
        f"top 2% (x > {_t2:.2f})": report(_yte, (_Xte[:, 0] > _t2).astype(int), _Xte[:, 0]),
    }
    _cols = list(next(iter(_rows.values())))
    _md = "| model | " + " | ".join(_cols) + " |\n|---|" + "---|" * len(_cols) + "\n"
    _md += "\n".join(f"| {k} | " + " | ".join(f"{v[c]:.3f}" for c in _cols) + " |" for k, v in _rows.items())
    _C = ex.confusion_matrix(_yte, (_Xte[:, 0] > _t2).astype(int))
    mo.vstack([mo.md(_md), mo.md(f"confusion matrix of 'top 2%' (rows = truth 0/1, columns = prediction 0/1): "
                                 f"`{_C.tolist()}`")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the dummy get 99% accuracy, and what do its precision and recall
    say? the stump minimized training error: why did it choose such a high threshold, and
    what did that cost in recall? the "top 2%" rule has *lower* accuracy than the dummy. is it
    a worse model? which numbers in the table would you show a bank, and why does ROC AUC
    look so much better than average precision?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: change the prevalence

    same feature, same separation ($\mathcal N(2.5, 1)$ versus $\mathcal N(0, 1)$), so the
    *model* is identical. only the fraction of positives changes. we draw ROC and
    precision–recall curves with your functions at several prevalences.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "as the positives go from 50% of the data to 0.1%, what happens to ROC AUC? to average "
        "precision?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _lines = []
    for _prev in (0.5, 0.1, 0.01, 0.001):
        _rng = np.random.default_rng(1)
        _X, _y = one_feature_data(200_000, _prev, 2.5, _rng)
        _fpr, _tpr, _ = ex.roc_curve(_y, _X[:, 0])
        _p, _r, _ = ex.precision_recall_curve(_y, _X[:, 0])
        _auc, _ap = ex.roc_auc_score(_y, _X[:, 0]), ex.average_precision_score(_y, _X[:, 0])
        _a1.plot(_fpr, _tpr, label=f"{_prev:.1%} positive (AUC {_auc:.3f})")
        _a2.plot(_r, _p, label=f"{_prev:.1%} positive (AP {_ap:.3f})")
        _lines.append(f"| {_prev:.1%} | {int(_y.sum()):,} | {_auc:.3f} | {_ap:.3f} |")
    _a1.plot([0, 1], [0, 1], "k:", lw=1)
    _a1.set(xlabel="false positive rate", ylabel="true positive rate (recall)", title="ROC", aspect="equal")
    _a2.set(xlabel="recall", ylabel="precision", title="precision–recall", xlim=(0, 1), ylim=(0, 1.02))
    _a1.legend(fontsize=8, loc="lower right")
    _a2.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| prevalence | positives (of 200,000) | ROC AUC | average precision |\n|---|---|---|---|\n"
                            + "\n".join(_lines))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** write the true and false positive rates as conditional probabilities.
    which of them depends on the prevalence? and precision? use your answer to explain why the
    ROC curves lie on top of each other while the PR curves collapse. when would you report
    each one?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: same ranking, different probabilities

    now 30% positives and the classes overlap more ($\mathcal N(1.5, 1)$ versus
    $\mathcal N(0, 1)$). for this setup the true probability of fraud given $x$ is known in
    closed form: $P(y = 1 \mid x) = \sigma(z)$ with $z = 1.5x - 1.125 + \log(0.3/0.7)$. three
    models report probabilities: the **calibrated** one reports $\sigma(z)$, an
    **overconfident** one $\sigma(Tz)$ and an **underconfident** one $\sigma(z/T)$, for the
    temperature $T$ on the slider. all three rank transactions identically.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the three models have the same ranking and the same predictions at 0.5. which of "
        "accuracy, ROC AUC, log loss and ECE will differ between them?"
    )
    p3
    return (p3,)


@app.cell
def _():
    temp = mo.ui.slider(1.0, 5.0, value=3.0, step=0.5, label="temperature T")
    temp
    return (temp,)


@app.cell
def _(p3, temp):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _X, _y = one_feature_data(20_000, 0.3, 1.5, _rng)
    _z = 1.5 * _X[:, 0] - 1.125 + np.log(0.3 / 0.7)
    _models = {"calibrated": sigmoid(_z), f"overconfident (×{temp.value:g})": sigmoid(temp.value * _z),
               f"underconfident (÷{temp.value:g})": sigmoid(_z / temp.value)}
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _rows = []
    for _ax, (_name, _prob) in zip(_axes, _models.items()):
        ex.reliability_diagram(_y, _prob, n_bins=15, ax=_ax)
        _ax.set_title(f"{_name}\n" + _ax.get_title(), fontsize=9)
        _rows.append(f"| {_name} | {ex.accuracy_score(_y, (_prob >= 0.5).astype(int)):.3f} | "
                     f"{ex.roc_auc_score(_y, _prob):.3f} | {ex.log_loss(_y, _prob):.3f} | "
                     f"{ex.expected_calibration_error(_y, _prob, n_bins=15):.3f} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| model | accuracy | ROC AUC | log loss | ECE |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can't accuracy or AUC tell these models apart? which shape does
    overconfidence make on the reliability diagram, and why? log loss punishes the
    overconfident model much more than the underconfident one at the same $T$: look at
    what $-\log p$ does near 0. when would you care about calibration rather than ranking?
    """)
    return


if __name__ == "__main__":
    app.run()
