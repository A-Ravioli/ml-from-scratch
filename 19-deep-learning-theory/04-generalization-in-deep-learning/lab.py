import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import datasets, notebooks, ui
    from mlfs.theory import vc_bound  # yours, from ch 13 · topic 04
    from mlfs.torchnn import MLP, count_parameters, fit, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    N_TRAIN = 1000


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # generalization in deep learning · lab

    every experiment uses **your** `corrupt_labels`, `RandomReLUFeatures`,
    `min_norm_least_squares`, `double_descent_curve` and `gd_least_squares` from `exercise.py`,
    plus your `MLP` and `fit` from chapter 15 and your `vc_bound` from chapter 13. the data is
    1,000 MNIST training digits (average-pooled to 14 × 14, centred and scaled so that an
    average image has norm 1) and 2,000 test digits. the lab takes about a minute on a laptop CPU.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()

    def _prep(x):
        x = torch.tensor(x, dtype=torch.float32)[:, None] / 255.0
        return F.avg_pool2d(x, 2).reshape(len(x), -1).numpy().astype(np.float64)

    _rng = np.random.default_rng(0)
    _idx = _rng.choice(len(_d["X_train"]), size=min(N_TRAIN, len(_d["X_train"])), replace=False)
    _X = _prep(_d["X_train"][_idx])
    _mu = _X.mean(0)
    _scale = np.sqrt(((_X - _mu) ** 2).sum(1).mean())
    X = (_X - _mu) / _scale
    y = _d["y_train"][_idx]
    X_test = (_prep(_d["X_test"][:2000]) - _mu) / _scale
    y_test = _d["y_test"][:2000]
    mo.md(f"{'MNIST' if not datasets.is_offline() else 'offline synthetic stand-in'}: {len(X)} training and "
          f"{len(X_test)} test images, {X.shape[1]} features each")
    return X, X_test, y, y_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: memorizing noise

    an MLP with two hidden layers of 512 units (368,650 parameters, 369 per training image),
    trained with Adam until it gets every training image right (at most 300 epochs), on three
    versions of the training labels: the true labels, half of them replaced by a wrong class,
    and all of them replaced. two seeds each. the test labels are always the true ones.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "can the network reach 100% training accuracy when every label is random? if so, how much longer does it take "
        "than with the true labels, and what test accuracy does each version get?"
    )
    p1
    return (p1,)


@app.cell
def _(X, X_test, p1, y, y_test):
    ui.gate(p1)
    _t0 = time.perf_counter()
    _Xt = torch.tensor(X, dtype=torch.float32)
    _Xtest = torch.tensor(X_test, dtype=torch.float32)
    memo = {}
    for _frac in (0.0, 0.5, 1.0):
        for _seed in (0, 1):
            _labels = ex.corrupt_labels(y, _frac, 10, rng=_seed)
            torch.manual_seed(_seed)
            _model = MLP(X.shape[1], [512, 512], 10)
            _opt = torch.optim.Adam(_model.parameters(), lr=1e-3)
            _yt = torch.tensor(_labels)
            _acc_hist = []

            def _stop(epoch, history, model=_model, yt=_yt, acc_hist=_acc_hist):
                with torch.no_grad():
                    acc_hist.append((model(_Xt).argmax(1) == yt).float().mean().item())
                return acc_hist[-1] == 1.0

            fit(_model, F.cross_entropy, _opt, make_loader(X, _labels, batch_size=100, seed=_seed), epochs=300,
                callback=_stop)
            with torch.no_grad():
                _test = (_model(_Xtest).argmax(1).numpy() == y_test).mean()
            memo[(_frac, _seed)] = dict(epochs=len(_acc_hist), train_acc=_acc_hist[-1], test_acc=float(_test),
                                        hist=_acc_hist)
    n_params = count_parameters(MLP(X.shape[1], [512, 512], 10))
    _fig, _ax = plt.subplots(figsize=(5, 3.3))
    for (_frac, _seed), _r in memo.items():
        _ax.plot(np.arange(1, len(_r["hist"]) + 1), _r["hist"], color={0.0: "C0", 0.5: "C1", 1.0: "C3"}[_frac],
                 label=f"{int(_frac * 100)}% random labels" if _seed == 0 else None)
    _ax.set(xlabel="epoch", ylabel="training accuracy", xscale="log")
    _ax.legend(fontsize=8)
    _rows = "\n".join(f"| {int(f * 100)}% | {', '.join(str(memo[(f, s)]['epochs']) for s in (0, 1))} | "
                      f"{', '.join(f'{memo[(f, s)]['train_acc']:.3f}' for s in (0, 1))} | "
                      f"{', '.join(f'{memo[(f, s)]['test_acc']:.3f}' for s in (0, 1))} |" for f in (0.0, 0.5, 1.0))
    _vc = vc_bound(len(X), n_params, 0.05, emp_risk=0.0)
    mo.vstack([_fig, mo.md(
        "| labels randomized | epochs to 100% train accuracy (2 seeds) | final train accuracy | test accuracy |\n"
        f"|---|---|---|---|\n{_rows}\n\nthe VC bound of chapter 13 for {n_params:,} parameters (taking the VC "
        f"dimension to be about the parameter count) and n = {len(X)}: test error at most **{_vc:.1f}** with 95% "
        f"confidence. {time.perf_counter() - _t0:.0f} s")])
    return (memo,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the same network, with the same optimizer and no change of
    hyperparameters, fits real labels and pure noise. so the network's *capacity*, the size of
    the set of functions it can fit, is enough to memorize anything, and any bound that depends
    only on the capacity (VC dimension, Rademacher complexity of the whole class) has to allow
    for noise memorization: why does that make it vacuous? the network that fitted true labels
    generalizes; what is different about the *solution* it found, if not the class it was
    chosen from? (with half the labels wrong it still gets far more test digits right than half:
    what does that tell you about how it fitted the wrong half?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: double descent

    a two-layer network whose first layer is frozen at random (your `RandomReLUFeatures`) and
    whose second layer is the minimum-norm least-squares fit to one-hot labels (your
    `min_norm_least_squares`): the "model" has $p$ trainable parameters per class. sweep $p$
    from 100 to 8,000 across the interpolation threshold $p = n = 1{,}000$, three random feature
    draws each, with your `double_descent_curve`. the controls add label noise or a ridge.
    """)
    return


@app.cell
def _():
    noise = mo.ui.dropdown(options={"0%": 0.0, "20%": 0.2}, value="0%", label="training labels randomized")
    ridge = mo.ui.dropdown(options={"0 (minimum norm)": 0.0, "0.001": 1e-3, "0.1": 0.1}, value="0 (minimum norm)",
                           label="ridge")
    p2 = ui.predict(
        "as the number of random features grows past the number of training points (1,000), what happens to the test "
        "error? where is it worst?"
    )
    mo.vstack([p2, mo.hstack([noise, ridge])])
    return noise, p2, ridge


@app.cell
def _(X, X_test, noise, p2, ridge, y, y_test):
    ui.gate(p2)
    _t0 = time.perf_counter()
    widths = [100, 300, 600, 800, 950, 1000, 1050, 1200, 1500, 2500, 4000, 8000]
    _labels = ex.corrupt_labels(y, noise.value, 10, rng=0)
    curve = ex.double_descent_curve(X, np.eye(10)[_labels], X_test, y_test, widths, ridge=ridge.value, n_seeds=3)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for _ax, _key, _lab in ((_axes[0], "test_error", "test error"), (_axes[1], "test_mse", "test MSE (log scale)")):
        _v = curve[_key]
        _ax.plot(widths, np.median(_v, 1), "o-", ms=3, label="median of 3 draws")
        _ax.fill_between(widths, _v.min(1), _v.max(1), alpha=0.25, label="min to max")
        _ax.axvline(len(X), color="gray", ls=":")
        _ax.set(xscale="log", xlabel="random features p", ylabel=_lab)
    _axes[1].set_yscale("log")
    _axes[0].legend(fontsize=8)
    _rows = "\n".join(f"| {w} | {np.median(curve['train_mse'][i]):.3f} | {np.median(curve['test_error'][i]):.3f} | "
                      f"{np.median(curve['test_mse'][i]):.2f} |" for i, w in enumerate(widths))
    mo.vstack([_fig, mo.md(f"noise {noise.value:.0%}, ridge {ridge.value}\n\n| p | train MSE | test error | test MSE |\n"
                           f"|---|---|---|---|\n{_rows}\n\n{time.perf_counter() - _t0:.0f} s")])
    return (curve,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the test error rises as $p$ approaches $n$, peaks at the interpolation
    threshold, and falls again: past the threshold, *more* parameters mean *better*
    generalization, though every model there fits the training data exactly. near $p = n$ there
    is exactly one way to interpolate, and it needs huge coefficients (look at the smallest
    singular value of a square random matrix); far past it there are infinitely many, and the
    minimum-norm one is smooth. what does the ridge do to the peak, and why? what does label
    noise do?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: which interpolant does gradient descent pick?

    100 training digits, 2,000 random ReLU features (here without the $1/\sqrt p$, so that the
    weights are of order one), and a binary task (even digit or odd). with
    more parameters than points there are infinitely many weight vectors that fit the training
    labels exactly. run your `gd_least_squares` to (near) zero training error from two starts:
    $w_0 = 0$, and $w_0$ drawn from $\mathcal N(0, \sigma^2 I)$ with the $\sigma$ on the slider.
    compare both with the minimum-norm solution from your `min_norm_least_squares`.
    """)
    return


@app.cell
def _():
    sigma = mo.ui.slider(0.0, 1.0, step=0.1, value=0.5, label="initialization scale σ")
    p3 = ui.predict(
        "both runs reach (almost) zero training error. does the starting point change the test error? which of the "
        "infinitely many interpolating solutions does gradient descent from zero converge to?"
    )
    mo.vstack([p3, sigma])
    return p3, sigma


@app.cell
def _(X, X_test, p3, sigma, y, y_test):
    ui.gate(p3)
    _n, _p = 100, 2000
    _feats = ex.RandomReLUFeatures(_p, rng=0).fit(X[:_n])
    _F, _Ft = _feats.transform(X[:_n]) * np.sqrt(_p), _feats.transform(X_test) * np.sqrt(_p)
    _yb = np.where(y[:_n] % 2 == 0, 1.0, -1.0)
    _ybt = np.where(y_test % 2 == 0, 1.0, -1.0)
    _lr = 1.0 / np.linalg.eigvalsh(_F @ _F.T / _n).max()
    _w_min = ex.min_norm_least_squares(_F, _yb)
    _rows, implicit = [], {}
    for _name, _w0 in (("w0 = 0", None), (f"w0 ~ N(0, {sigma.value:.1f}²)", np.random.default_rng(1).normal(0, sigma.value, _p))):
        _w = ex.gd_least_squares(_F, _yb, lr=_lr, steps=20000, w0=_w0)
        implicit[_name] = dict(train_mse=float(np.mean((_F @ _w - _yb) ** 2)),
                               test_err=float(np.mean(np.sign(_Ft @ _w) != _ybt)),
                               dist=float(np.linalg.norm(_w - _w_min)), norm=float(np.linalg.norm(_w)))
    implicit["minimum norm (pinv)"] = dict(train_mse=float(np.mean((_F @ _w_min - _yb) ** 2)),
                                           test_err=float(np.mean(np.sign(_Ft @ _w_min) != _ybt)), dist=0.0,
                                           norm=float(np.linalg.norm(_w_min)))
    mo.md("| solution | train MSE | test error | distance to minimum-norm | weight norm |\n|---|---|---|---|---|\n"
          + "\n".join(f"| {k} | {v['train_mse']:.1e} | {v['test_err']:.3f} | {v['dist']:.3f} | {v['norm']:.2f} |"
                      for k, v in implicit.items()))
    return (implicit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** from zero, gradient descent lands on the minimum-norm interpolant, though
    nothing in the loss mentions a norm. every gradient $X^\top(Xw - y)$ is a combination of
    the rows of $X$: what does that say about the part of $w$ that is orthogonal to all of them?
    from a random start that part never changes, so the solution carries the initialization's
    random component forever. this is the simplest example of an **implicit bias**: the
    algorithm, not the loss, chooses among the solutions. what would you expect for logistic
    loss on separable data (Soudry et al., 2018)?
    """)
    return


if __name__ == "__main__":
    app.run()
