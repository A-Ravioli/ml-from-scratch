import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.nn import Linear, ReLU, Sequential, cross_entropy, fit, predict  # yours, from ch 14 · topic 04
    from mlfs.optim import Adam, AdamW  # yours, from ch 06 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    # a deliberately SMALL training set (1,000 images), so that the network can overfit it
    if datasets.is_offline():  # the stand-in has no blank border; give it one, like MNIST's
        _X = _X.reshape(-1, 28, 28).copy()
        _X[:, :3], _X[:, -3:], _X[:, :, :3], _X[:, :, -3:] = 0, 0, 0, 0
        _X = _X.reshape(-1, 784)
    X_tr, y_tr = _X[:1000], _d["y_train"][:1000]
    X_val, y_val = _X[1000:][-2000:], _d["y_train"][1000:][-2000:]  # never overlaps the training images
    EPOCHS = 30


@app.function
def make_net(*, p=0.0, rng=0):
    """784 -> 256 -> 256 -> 10 ReLU network, with Dropout(p) after each hidden ReLU if p > 0."""
    rng = np.random.default_rng(rng)
    layers = [Linear(784, 256, rng=rng), ReLU()]
    if p:
        layers.append(ex.Dropout(p, rng=rng))
    layers += [Linear(256, 256, rng=rng), ReLU()]
    if p:
        layers.append(ex.Dropout(p, rng=rng))
    layers.append(Linear(256, 10, rng=rng))
    return Sequential(*layers)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # regularization · lab

    every experiment here uses **your** `Dropout`, `label_smoothing_cross_entropy` and
    `EarlyStopping` from `exercise.py`, with `mlfs.optim`'s `Adam` and `AdamW`. the setting is
    built to overfit: a $784 \to 256 \to 256 \to 10$ network (about 270,000 parameters)
    trained for 30 epochs on only **1,000** MNIST digits, scored on 2,000 held-out ones
    (offline: a synthetic stand-in with the same shapes, 1,000 held out, and a blanked
    border). for each experiment: write down a prediction, lock it in, run it, then explain
    what happened. compare your explanations with `solutions/lab_notes.md` afterwards. the
    whole lab takes about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: Adam + L2 versus AdamW

    `Adam(weight_decay=λ)` adds $\lambda w$ to the gradient (an L2 penalty, like your
    `l2_penalty`); `AdamW(weight_decay=λ)` shrinks the weights directly, $w \leftarrow w -
    \eta\lambda w$, outside Adam's normalization. we train with each over a range of λ and
    look at the first layer's weights in two groups: those attached to **active** pixels and
    those attached to **dead** pixels: on MNIST, the 175 border pixels that are 0 in every
    one of the 1,000 training images, so their weights get no gradient from the loss at all.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with Adam + L2 at a tiny λ = 1e-4, what happens to the dead-pixel weights after 30 epochs, and to the "
        "active-pixel weights? and with AdamW?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _active = X_tr.std(axis=0) > 0
    _rows, _res = [], {}
    for _name, _cls, _lams in [("Adam + L2", Adam, [0.0, 1e-4, 1e-3, 1e-2, 1e-1]),
                               ("AdamW", AdamW, [0.0, 1e-2, 1e-1, 1.0, 10.0])]:
        _res[_name] = []
        for _lam in _lams:
            _net = make_net(rng=0)
            fit(_net, cross_entropy, _cls(_net.parameters(), lr=1e-3, weight_decay=_lam), X_tr, y_tr, epochs=EPOCHS,
                batch_size=64, rng=0)
            _W = _net[0].weight.data
            _acc = float(np.mean(predict(_net, X_val).argmax(1) == y_val))
            _ra, _rd = np.sqrt(np.mean(_W[:, _active] ** 2)), np.sqrt(np.mean(_W[:, ~_active] ** 2))
            _res[_name].append((_lam, _acc, _ra, _rd))
            _rows.append(f"| {_name} | {_lam:g} | {_acc:.3f} | {_ra:.4f} | {_rd:.2e} |")
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for _ax, _name in zip(_axes, _res):
        _lam = [r[0] for r in _res[_name][1:]]
        _ax.loglog(_lam, [r[2] for r in _res[_name][1:]], "o-", label="active-pixel weights")
        _ax.loglog(_lam, [max(r[3], 1e-16) for r in _res[_name][1:]], "s--", label="dead-pixel weights")
        _ax.axhline(_res[_name][0][2], color="C0", ls=":", lw=1)
        _ax.axhline(_res[_name][0][3], color="C1", ls=":", lw=1, label="λ = 0 (both groups)")
        _ax.set(xlabel="weight_decay λ", title=_name)
    _axes[0].set_ylabel("RMS of first-layer weights")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"{int((~_active).sum())} dead pixels.\n\n| optimizer | λ | validation accuracy | RMS, active-pixel "
                           "weights | RMS, dead-pixel weights |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with Adam + L2, the update for a dead-pixel weight is
    $-\eta\,\hat m/\sqrt{\hat v}$ where the only gradient is $\lambda w$. what does that
    ratio come to, whatever λ is (so how fast does the weight shrink per step)? and for an
    active weight whose loss gradient is much larger than $\lambda w$? so which weights does
    an L2 penalty under Adam actually regularize? why does AdamW treat both groups the same
    way (shrinking each weight by the same *fraction* per step)? does the choice matter for
    validation accuracy here?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: dropout on a small dataset

    the same network with your `Dropout` after each hidden ReLU, $p \in \{0, 0.2, 0.5\}$,
    trained with Adam ($10^{-3}$) for 30 epochs, 2 seeds each. we record the training and
    validation loss after every epoch.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "every network reaches 100% training accuracy. which p gives the best validation accuracy, and does dropout "
        "change the training loss, the validation loss, or both?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _curves, _rows = {}, []
    for _p in [0.0, 0.2, 0.5]:
        _hs, _accs, _tr = [], [], []
        for _seed in range(2):
            _net = make_net(p=_p, rng=_seed)
            _h = fit(_net, cross_entropy, Adam(_net.parameters(), lr=1e-3), X_tr, y_tr, epochs=EPOCHS, batch_size=64,
                     X_val=X_val, y_val=y_val, rng=_seed)
            _hs.append(_h)
            _accs.append(float(np.mean(predict(_net, X_val).argmax(1) == y_val)))
            _tr.append(float(np.mean(predict(_net, X_tr).argmax(1) == y_tr)))
        _curves[_p] = (np.mean([h["train_loss"] for h in _hs], 0), np.mean([h["val_loss"] for h in _hs], 0))
        _rows.append(f"| {_p} | {np.mean(_tr):.3f} | {np.mean(_accs):.3f} | {_curves[_p][0][-1]:.4f} | "
                     f"{_curves[_p][1].min():.3f} (epoch {int(_curves[_p][1].argmin()) + 1}) | {_curves[_p][1][-1]:.3f} |")
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ep = np.arange(1, EPOCHS + 1)
    for _i, (_p, (_t, _v)) in enumerate(_curves.items()):
        _ax.semilogy(_ep, _t, ":", color=f"C{_i}", label=f"p = {_p}: train")
        _ax.semilogy(_ep, _v, "-", color=f"C{_i}", label=f"p = {_p}: validation")
    _ax.set(xlabel="epoch", ylabel="cross-entropy (mean of 2 seeds)")
    _ax.legend(fontsize=8, ncol=2)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| dropout p | training accuracy | validation accuracy | final training loss | best "
                           "validation loss | final validation loss |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the validation loss of the network without dropout *rise* after
    its minimum while its validation accuracy hardly changes (what happens to the confidence
    of its wrong answers)? what does dropout do to the training loss, and why? is the
    improvement in validation accuracy large compared with the seed-to-seed differences you
    can see? (topic 06 returns to that question.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: label smoothing, and when to stop

    the network without dropout, trained for 30 epochs with plain cross-entropy and with your
    `label_smoothing_cross_entropy` (s = 0.1 and 0.3). after training we measure the
    validation accuracy, the plain cross-entropy (NLL) on the validation set, the average
    confidence on the examples it gets **wrong**, and the expected calibration error (ECE: the
    gap between confidence and accuracy, averaged over 10 confidence bins). then we replay
    the validation curves through two `EarlyStopping(patience=5)` monitors, one watching the
    validation loss (mode 'min') and one the validation accuracy (mode 'max').
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "label smoothing makes the model less confident. will it be better calibrated (lower ECE)? and will early "
        "stopping on the validation loss stop at the same epoch as early stopping on the validation accuracy?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rows, _acc_curves = [], {}
    for _s in [0.0, 0.1, 0.3]:
        _net = make_net(rng=0)
        _accs = []
        _loss = cross_entropy if _s == 0 else (lambda z, t, s=_s: ex.label_smoothing_cross_entropy(z, t, smoothing=s))
        _h = fit(_net, _loss, Adam(_net.parameters(), lr=1e-3), X_tr, y_tr, epochs=EPOCHS, batch_size=64, X_val=X_val,
                 y_val=y_val, rng=0,
                 callback=lambda e, h, net=_net, accs=_accs: accs.append(np.mean(predict(net, X_val).argmax(1) == y_val)))
        _z = predict(_net, X_val)
        _pr = np.exp(_z - _z.max(1, keepdims=True))
        _pr /= _pr.sum(1, keepdims=True)
        _pred, _conf = _pr.argmax(1), _pr.max(1)
        _nll = float(np.mean(-np.log(_pr[np.arange(len(y_val)), y_val] + 1e-300)))
        _ece = sum(np.mean(_m) * abs(_conf[_m].mean() - np.mean(_pred[_m] == y_val[_m]))
                   for _lo in np.linspace(0, 0.9, 10) if (_m := (_conf > _lo) & (_conf <= _lo + 0.1)).any())
        _stops = []
        for _mon, _vals in [(ex.EarlyStopping(5), _h["val_loss"]), (ex.EarlyStopping(5, mode="max"), _accs)]:
            _stop = next((i for i, v in enumerate(_vals) if _mon.step(v)), None)
            _stops.append("never" if _stop is None else f"epoch {_stop + 1} (best {_mon.best_step + 1})")
        _acc_curves[_s] = _accs
        _wrong = _pred != y_val
        _cw = f"{_conf[_wrong].mean():.3f}" if _wrong.any() else "none wrong"
        _rows.append(f"| {_s} | {_accs[-1]:.3f} | {_nll:.3f} | {_cw} | {_ece:.3f} | "
                     f"{_stops[0]} | {_stops[1]} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    for _s, _a in _acc_curves.items():
        _ax.plot(np.arange(1, EPOCHS + 1), _a, label=f"smoothing {_s}")
    _ax.set(xlabel="epoch", ylabel="validation accuracy")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| smoothing | validation accuracy | NLL | confidence when wrong | ECE | stop on "
                           "val loss | stop on val accuracy |\n|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** label smoothing's target puts $1 - s + s/k$ on the true class. what is
    the *most* confident prediction it rewards, for $s = 0.1$ and $s = 0.3$? use that to
    explain the ECE column: is the smoothed model over- or under-confident? why do the two
    early-stopping monitors disagree about when to stop the plain model, and which would
    you trust if what you care about is accuracy?
    """)
    return


if __name__ == "__main__":
    app.run()
