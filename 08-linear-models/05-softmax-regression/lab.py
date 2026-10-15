import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.linear import LogisticRegression  # yours, from topic 04
    from mlfs.metrics import accuracy_score, confusion_matrix  # yours, from ch 07
    from mlfs.stats import bootstrap_ci  # yours, from ch 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    # small CPU config offline (the synthetic stand-in), 20,000 training images otherwise
    SMALL = datasets.is_offline()
    N_TRAIN = 2_000 if SMALL else 20_000


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # softmax regression · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    experiments 1 and 3 use MNIST through `mlfs.datasets` (downloaded once and cached). offline
    (tests, `verify.py`) you get a small synthetic stand-in with the same shapes, whose numbers
    say nothing about real digits.
    """)
    return


@app.cell
def _():
    digits = datasets.mnist()
    _raw_tr = digits["X_train"][:N_TRAIN].reshape(-1, 784) / 255.0
    _mean_image = _raw_tr.mean(axis=0)
    Xtr = _raw_tr - _mean_image  # centered: topic 01's conditioning lesson, applied to pixels
    ytr = digits["y_train"][:N_TRAIN]
    Xte = digits["X_test"].reshape(-1, 784) / 255.0 - _mean_image
    yte = digits["y_test"]
    _L = np.linalg.eigvalsh(np.cov(Xtr.T, bias=True)).max() / 2  # softmax curvature bound (with an intercept)
    mo.md(f"MNIST: {len(Xtr):,} training images, {len(Xte):,} test images, 784 pixels each, scaled to [0, 1] "
          f"and centered with the training mean image" + (" (offline: synthetic stand-in)" if SMALL else "")
          + f". the largest stable gradient-descent step here is about $2/L$ = {2 / max(_L, 0.5):.2f}.")
    return Xte, Xtr, yte, ytr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how far does a linear model get on MNIST?

    one weight per pixel per digit: a $784 \times 10$ matrix and 10 intercepts, 7,850
    parameters in all, on centered pixels (the training mean image subtracted, which keeps
    the problem well conditioned). you fit your `SoftmaxRegression` with minibatch SGD (batches of 64,
    $\eta = 0.5$, a small penalty $\alpha = 10^{-4}$, 5 epochs) and measure test accuracy, with
    a bootstrap confidence interval (your chapter 05 `bootstrap_ci`) and the majority-class
    baseline next to it.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what test accuracy will a linear model get on handwritten digits? which pair of digits "
        "will it confuse most? what will the 10 weight vectors look like as images?"
    )
    p1
    return (p1,)


@app.cell
def _(Xte, Xtr, p1, yte, ytr):
    ui.gate(p1)
    model = ex.SoftmaxRegression(alpha=1e-4, lr=0.5, batch_size=64, max_iters=5, tol=0.0, rng=0).fit(Xtr, ytr)
    _pred = model.predict(Xte)
    _acc = accuracy_score(yte, _pred)
    _lo, _hi = bootstrap_ci(( _pred == yte).astype(float), np.mean, n_boot=2000, rng=0)
    _base = np.max(np.bincount(yte)) / len(yte)
    _C = confusion_matrix(yte, _pred, labels=np.arange(10))
    _off = _C.astype(float).copy()
    np.fill_diagonal(_off, 0)
    _pairs = sorted(((int(_off[i, j]), i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:3]
    _fig, _axes = plt.subplots(2, 5, figsize=(9, 4.6))
    _W = np.asarray(model.coef_)
    _lim = np.abs(_W).max()
    for _c, _ax in enumerate(_axes.ravel()):
        _ax.imshow(_W[:, _c].reshape(28, 28), cmap="RdBu_r", vmin=-_lim, vmax=_lim)
        _ax.set(title=f"digit {_c}", xticks=[], yticks=[])
    _fig.suptitle("weights per class (red: pixel ink raises that digit's score)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"test accuracy **{_acc:.3f}** (95% bootstrap CI {_lo:.3f}–{_hi:.3f}); training accuracy "
        f"{model.score(Xtr, ytr):.3f}; majority-class baseline {_base:.3f}.\n\n"
        "most frequent confusions (true → predicted): "
        + (", ".join(f"{i} → {j} ({n} times)" for n, i, j in _pairs if n > 0) or "none: no test mistakes"))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a linear model scores each digit by a weighted sum of pixel intensities:
    what is it really comparing the image with, and why does that work so well for digits
    that are centered and size-normalized? why do the most confused pairs make sense
    geometrically? what kind of variation in handwriting can a single template per class
    *not* capture (think of two different ways of writing a 4, or a shifted 1)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: one model for all classes, or one per class?

    five classes lined up along the $x_1$ axis, 100 points each, standardized. you fit your
    `SoftmaxRegression` and your `OneVsRest` wrapped around topic 04's `LogisticRegression`
    (both with $\alpha = 10^{-3}$), and look at the accuracy for each class and the
    probability each model gives every class along the line.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "both models are linear. which classes, if any, will one-vs-rest get noticeably worse than "
        "softmax?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _K, _n = 5, 100
    _rng = np.random.default_rng(1)
    _y = np.repeat(np.arange(_K), _n)
    _X = np.column_stack([np.repeat(3.0 * np.arange(_K), _n), np.zeros(_K * _n)]) + _rng.normal(size=(_K * _n, 2))
    _mu, _sd = _X.mean(0), _X.std(0)
    _X = (_X - _mu) / _sd
    _sm = ex.SoftmaxRegression(alpha=1e-3, lr=1.0, max_iters=5000).fit(_X, _y)
    _ovr = ex.OneVsRest(LogisticRegression(alpha=1e-3)).fit(_X, _y)
    _line = np.column_stack([np.linspace(-1.9, 1.9, 300), np.zeros(300)])
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for _ax, _m, _name in ((_axes[0], _sm, "softmax"), (_axes[1], _ovr, "one-vs-rest")):
        _P = _m.predict_proba(_line)
        for _c in range(_K):
            _ax.plot(_line[:, 0], _P[:, _c], label=f"class {_c}")
        _ax.set(title=_name, xlabel="$x_1$ (standardized)")
    _axes[0].set_ylabel("predicted probability")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {_c} | {np.mean(_sm.predict(_X)[_y == _c] == _c):.2f} | {np.mean(_ovr.predict(_X)[_y == _c] == _c):.2f} |"
             for _c in range(_K)]
    mo.vstack([_fig, mo.md("| class | softmax | one-vs-rest |\n|---|---|---|\n" + "\n".join(_rows)
                           + f"\n| all | {_sm.score(_X, _y):.3f} | {_ovr.score(_X, _y):.3f} |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the one-vs-rest model for class 2 must separate class 2 from classes on
    *both* sides of it with a single line. what's the best it can do, and what does its
    probability curve look like? softmax compares scores across classes: why can it give the
    middle class a bump even though each score is linear in $x$? when is one-vs-rest still a
    reasonable choice?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how big should a batch be?

    the same MNIST training set (offline: the stand-in) and the same model, trained for 1, 2
    and 5 epochs with batches of 10, 100 and 1,000 images and with the full batch (plain
    gradient descent, `batch_size=None` with `max_iters` steps, one per epoch). all use
    $\eta = 0.5$ and the same penalty.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "after one epoch (one pass over the data), what test accuracy will each batch size reach? "
        "which is best after 5 epochs?"
    )
    p3
    return (p3,)


@app.cell
def _(Xte, Xtr, p3, yte, ytr):
    ui.gate(p3)
    _rows = []
    for _b in (10, 100, 1000, None):
        _accs = []
        for _e in (1, 2, 5):
            _m = ex.SoftmaxRegression(alpha=1e-4, lr=0.5, batch_size=_b, max_iters=_e, tol=0.0, rng=0).fit(Xtr, ytr)
            _accs.append(_m.score(Xte, yte))
        _steps = _e * (1 if _b is None else -(-len(Xtr) // _b))
        _rows.append(f"| {'full' if _b is None else _b} | " + " | ".join(f"{_a:.3f}" for _a in _accs)
                     + f" | {_steps:,} |")
    mo.md("| batch size | after 1 epoch | after 2 | after 5 | steps in 5 epochs |\n|---|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** every row has seen each image the same number of times. why do small
    batches get so much further per epoch? what do you pay for a smaller batch (noise in each
    step, and time on real hardware, where a batch of 1,000 costs about as much wall-clock time
    as a batch of 10)? why does nobody train large models with the full batch?
    """)
    return


if __name__ == "__main__":
    app.run()
