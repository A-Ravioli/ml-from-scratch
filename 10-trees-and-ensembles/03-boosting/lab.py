import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.metrics import expected_calibration_error, log_loss  # yours, from ch 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # boosting · lab

    every experiment here runs **your** AdaBoost from `exercise.py`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    the data is the "Hastie 10.2" problem: 10 standard normal features, class 1 when
    $\sum_jx_j^2 > 9.34$ (the median of a $\chi^2_{10}$, so the classes are balanced). the
    boundary is a sphere: easy to describe, hard for axis-aligned trees. 600 training points,
    10,000 test points, no label noise. the next cell fits 1,000 rounds of AdaBoost with
    depth-2 trees (about 10 seconds); nothing is shown until you've made your predictions.
    """)
    return


@app.cell
def _():
    def hastie(n, rng, d=10):
        X = rng.normal(size=(n, d))
        return X, (np.sum(X ** 2, axis=1) > 9.34).astype(int)

    _rng = np.random.default_rng(0)
    X_tr, y_tr = hastie(600, _rng)
    X_te, y_te = hastie(10000, _rng)
    ada = ex.AdaBoostClassifier(n_estimators=1000, max_depth=2).fit(X_tr, y_tr)
    F_tr = np.array([f for f in ada.staged_decision_function(X_tr)])  # (rounds, n_train)
    F_te = np.array([f for f in ada.staged_decision_function(X_te)])
    return F_te, F_tr, X_te, X_tr, ada, y_te, y_tr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: after the training error hits zero

    we track the training and test error after every round, the bound
    $\prod_m2\sqrt{\varepsilon_m(1 - \varepsilon_m)}$, and the distribution of the training
    margins $y_if(x_i)/\sum\alpha$ at a few rounds.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the training error reaches 0 at some round. over the remaining rounds (up to 1,000), does the test error "
        "go up, stay flat or go down? what happens to the smallest training margins?"
    )
    p1
    return (p1,)


@app.cell
def _(F_te, F_tr, ada, p1, y_te, y_tr):
    ui.gate(p1)
    _ypm, _ytpm = 2 * y_tr - 1, 2 * y_te - 1
    _tr = np.mean(np.sign(F_tr) != _ypm, axis=1)
    _te = np.mean(np.sign(F_te) != _ytpm, axis=1)
    _bound = np.cumprod(2 * np.sqrt(ada.estimator_errors_ * (1 - ada.estimator_errors_)))
    _zero = int(np.argmax(_tr == 0)) + 1 if np.any(_tr == 0) else None
    _r = np.arange(1, len(_tr) + 1)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.3))
    _axes[0].semilogx(_r, _tr, label="training error")
    _axes[0].semilogx(_r, _te, label="test error")
    _expl = np.array([ex.exponential_loss(_ypm, _f / 2) for _f in F_tr])  # SAMME's f is twice the classic score
    _axes[0].semilogx(_r, _bound, "k:", lw=3, label="bound ∏ 2√(ε(1−ε))")
    _axes[0].semilogx(_r, _expl, "y--", lw=1, label="exponential_loss(y, f / 2)")
    if _zero:
        _axes[0].axvline(_zero, color="gray", ls="--", lw=1, label=f"training error 0 (round {_zero})")
    _axes[0].set(xlabel="round", ylabel="error", ylim=(0, 0.55))
    _axes[0].legend(fontsize=8)
    _alpha_cum = np.cumsum(ada.estimator_weights_)
    _show = [10, 100] + ([_zero] if _zero else []) + [len(_tr)]
    for _k in _show:
        _m = np.sort(_ypm * F_tr[_k - 1] / _alpha_cum[_k - 1])
        _axes[1].plot(_m, np.arange(1, len(_m) + 1) / len(_m), label=f"round {_k}")
    _axes[1].axvline(0, color="k", lw=0.5)
    _axes[1].set(xlabel="normalized margin y f(x) / Σα", ylabel="fraction of training points ≤ margin",
                 title="margin distribution (cumulative)", xlim=(-0.4, 0.4))
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _mins = {k: float(np.min(_ypm * F_tr[k - 1] / _alpha_cum[k - 1])) for k in _show}
    mo.vstack([_fig, mo.md(
        f"training error first 0 at round **{_zero}**, where the test error is **{_te[_zero - 1]:.3f}**; "
        f"at round 1,000 the test error is **{_te[-1]:.3f}** (lowest {_te.min():.3f}). the smallest margin: "
        + ", ".join(f"{_mins[k]:.3f} (round {k})" for k in _show) + ". "
        f"the exponential loss mean exp(−y f / 2) equals the bound to {np.abs(_expl - _bound).max():.1e}, and is "
        f"{_expl[-1]:.4f} at round 1,000. "
        f"the weak learners' weighted errors drift from {ada.estimator_errors_[0]:.3f} (round 1) to "
        f"{np.mean(ada.estimator_errors_[-100:]):.3f} (mean of the last 100 rounds).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why doesn't the test error go up once the training error is zero, even
    though every round makes the model more complex? what happens to the margin curve after
    that round, and why does AdaBoost keep changing anything at all? why do the weak
    learners' errors creep toward 0.5?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: where does the weight go?

    the same training points, but **10% of their labels flipped**. we run 500 rounds with
    `learning_rate` 1 and 0.1, replay the weight updates from `estimator_weights_`, and
    track the share of the total weight on the flipped points, plus the test error (against
    clean test labels).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the flipped points are 10% of the data. after 500 rounds with learning_rate=1, what share of the total "
        "weight do they carry? of the 30 heaviest points, how many are flipped?"
    )
    p2
    return (p2,)


@app.cell
def _(X_te, X_tr, p2, y_te, y_tr):
    ui.gate(p2)
    _flip = np.random.default_rng(5).random(len(y_tr)) < 0.1
    _yn = np.where(_flip, 1 - y_tr, y_tr)
    _ypm = 2 * _yn - 1
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _rows = []
    for _lr, _c in [(1.0, "C0"), (0.1, "C3")]:
        _m = ex.AdaBoostClassifier(n_estimators=500, max_depth=2, learning_rate=_lr).fit(X_tr, _yn)
        _w = np.full(len(_yn), 1.0 / len(_yn))
        _share = []
        for _a, _t in zip(_m.estimator_weights_, _m.estimators_):
            _h = 2 * _t.predict(X_tr) - 1
            _w = _w * np.exp(_a * (_h != _ypm))
            _w = _w / _w.sum()
            _share.append(_w[_flip].sum())
        _te = np.array([np.mean((f > 0) != y_te) for f in _m.staged_decision_function(X_te)])
        _r = np.arange(1, len(_share) + 1)
        _axes[0].semilogx(_r, _share, color=_c, label=f"learning_rate {_lr}")
        _axes[1].semilogx(_r, _te, color=_c, label=f"learning_rate {_lr}")
        _top = np.argsort(_w)[-30:]
        _rows.append(f"| {_lr} | {_share[-1]:.2f} | {int(_flip[_top].sum())} / 30 | {_te.min():.3f} (round "
                     f"{int(np.argmin(_te)) + 1}) | {_te[-1]:.3f} |")
    _axes[0].axhline(_flip.mean(), color="k", ls=":", label=f"their share of the points ({_flip.mean():.2f})")
    _axes[0].set(xlabel="round", ylabel="share of the total weight", title="weight on the flipped points")
    _axes[0].legend(fontsize=8)
    _axes[1].set(xlabel="round", ylabel="test error (clean labels)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| learning_rate | weight share after 500 rounds | flipped among the 30 heaviest | best test error | "
        "test error at 500 |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do mislabeled points accumulate weight? connect it to the exponential
    loss: what does a confidently wrong point cost, compared with the logistic loss? what
    does a smaller learning rate change, and what doesn't it change? how would you detect
    mislabeled points with this?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: better accuracy, worse probabilities?

    back to the clean data of experiment 1. after each round we turn the test scores into
    probabilities $\sigma(f(x))$ and measure the test error, the log loss and the expected
    calibration error (your ch 07 metrics).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the test error keeps improving all the way to round 1,000. does the log loss of sigma(f) improve with it?"
    )
    p3
    return (p3,)


@app.cell
def _(F_te, p3, y_te):
    ui.gate(p3)
    _rounds = np.unique(np.geomspace(1, len(F_te), 40).astype(int))
    _err, _ll, _ece, _size = [], [], [], []
    for _k in _rounds:
        _f = F_te[_k - 1]
        _p = 0.5 * (1 + np.tanh(_f / 2))
        _err.append(np.mean((_f > 0) != y_te))
        _ll.append(log_loss(y_te, _p))
        _ece.append(expected_calibration_error(y_te, _p))
        _size.append(np.mean(np.abs(_f)))
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _axes[0].semilogx(_rounds, _err, label="test error")
    _axes[0].semilogx(_rounds, _ll, label="test log loss")
    _axes[0].semilogx(_rounds, _ece, label="expected calibration error")
    _axes[0].set(xlabel="round")
    _axes[0].legend(fontsize=8)
    _axes[1].semilogx(_rounds, _size, color="C4")
    _axes[1].set(xlabel="round", ylabel="mean |f(x)| on test points", title="how confident the scores get")
    _fig.tight_layout()
    _best = int(np.argmin(_ll))
    mo.vstack([_fig, mo.md(
        f"log loss is lowest at round **{_rounds[_best]}** ({_ll[_best]:.3f}, test error {_err[_best]:.3f}); "
        f"at round {_rounds[-1]} it's **{_ll[-1]:.3f}** with test error **{_err[-1]:.3f}**. "
        f"ECE: {_ece[_best]:.3f} → {_ece[-1]:.3f}; mean |f|: {_size[_best]:.2f} → {_size[-1]:.2f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how can the error rate and the log loss move in opposite directions? what
    is happening to $|f(x)|$, and why does AdaBoost keep inflating it (think of experiment 1's
    margins)? if you needed good probabilities from this model, what would you do?
    """)
    return


if __name__ == "__main__":
    app.run()
