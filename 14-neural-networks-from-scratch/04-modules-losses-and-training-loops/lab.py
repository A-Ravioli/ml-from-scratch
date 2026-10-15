import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.optim import SGD, Adam  # yours, from ch 06 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    X_all = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    y_all = _d["y_train"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # modules, losses & training loops · lab

    every experiment here runs **your** `mlfs.nn` from `exercise.py`, on MNIST digits
    (flattened to 784 pixels in $[0, 1]$; offline, a synthetic stand-in with the same
    shapes). for each one: write down a prediction, lock it in, run it, then explain what
    happened. compare your explanations with `solutions/lab_notes.md` afterwards. the whole
    lab takes about half a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: overfit one batch

    a $784 \to 128 \to 10$ ReLU network (about 100,000 parameters) trains on **one** batch of
    32 images for 200 Adam steps (learning rate $10^{-3}$), once with the true labels and once
    with labels drawn at random. we run a correct training loop and four broken ones:

    - **no zero_grad**: the loop never calls `optimizer.zero_grad()`;
    - **no activation**: the `ReLU` was left out (two stacked `Linear` layers);
    - **lr too small**: learning rate $10^{-6}$;
    - **last layer only**: the optimizer was given `model[2].parameters()` instead of `model.parameters()`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with random labels, can the correct network reach 100% accuracy on its batch? which of the four bugs "
        "will make the one-batch test fail (final loss far from 0)?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _idx = _rng.choice(len(X_all), 32, replace=False)
    _xb = X_all[_idx]
    _labels = {"true labels": y_all[_idx], "random labels": _rng.integers(0, 10, 32)}
    _bugs = ["correct", "no zero_grad", "no activation", "lr too small", "last layer only"]
    _curves, _rows = {}, []
    for _bug in _bugs:
        _cells = []
        for _lname, _yb in _labels.items():
            _layers = [ex.Linear(784, 128, rng=1), ex.ReLU(), ex.Linear(128, 10, rng=2)]
            if _bug == "no activation":
                _layers = [_layers[0], _layers[2]]
            _model = ex.Sequential(*_layers)
            _params = _model[-1].parameters() if _bug == "last layer only" else _model.parameters()
            _opt = Adam(_params, lr=1e-6 if _bug == "lr too small" else 1e-3)
            _losses = []
            for _ in range(200):
                if _bug != "no zero_grad":
                    _opt.zero_grad()
                _loss = ex.cross_entropy(_model(_xb), _yb)
                _loss.backward()
                _opt.step()
                _losses.append(_loss.item())
            _acc = np.mean(ex.predict(_model, _xb).argmax(1) == _yb)
            _cells.append(f"{_losses[-1]:.3f} / {_acc:.0%}")
            if _lname == "random labels":
                _curves[_bug] = _losses
        _rows.append(f"| {_bug} | " + " | ".join(_cells) + " |")
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _bug, _c in _curves.items():
        _ax.semilogy(_c, label=_bug)
    _ax.set(xlabel="Adam step", ylabel="loss on the batch (log scale)", title="one batch of 32, random labels")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| training loop | true labels: final loss / accuracy | random labels: final loss / accuracy |\n"
                           "|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how can 100,000 parameters fit 32 random labels perfectly, and why is
    that *good* news for a sanity check? which bugs did the test catch, and which slipped
    through? for each one that slipped through, explain why it can still memorize 32
    examples, and say what check would catch it instead.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the loss before training

    the same architecture, untrained, evaluated with your `evaluate` on 1,000 images. we
    multiply every initial weight and bias by a factor $s$ and record the loss and the
    network's average confidence (its largest softmax probability).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what's the cross-entropy of the untrained network with the default init (s = 1)? and with s = 10?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _X, _y = X_all[:1000], y_all[:1000]
    _scales = [0.1, 0.3, 1.0, 3.0, 10.0]
    _losses, _rows = [], []
    for _s in _scales:
        _model = ex.Sequential(ex.Linear(784, 128, rng=1), ex.ReLU(), ex.Linear(128, 10, rng=2))
        for _p in _model.parameters():
            _p.data *= _s
        _loss = ex.evaluate(_model, ex.cross_entropy, _X, _y)
        _z = ex.predict(_model, _X)
        _p_ = np.exp(_z - _z.max(1, keepdims=True))
        _conf = np.mean(_p_.max(1) / _p_.sum(1))
        _losses.append(_loss)
        _rows.append(f"| {_s} | {_loss:.3f} | {_conf:.2f} | {np.std(_z):.2f} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.loglog(_scales, _losses, "o-", label="untrained loss")
    _ax.axhline(np.log(10), color="gray", ls=":", label="log 10 = 2.303 (uniform guess)")
    _ax.set(xlabel="initial weight scale s", ylabel="cross-entropy")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| scale s | loss | mean max probability | std of the logits |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the loss at $s = 1$ almost exactly $\log 10$? why does it grow so
    fast with $s$ (by how much does the std of the logits grow when *both* layers are scaled
    by $s$)? what does a large initial loss tell you about the network's predictions, and
    why is starting there a bad idea for training?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: too large a learning rate

    your `fit` trains the default network with plain SGD (batch size 32, 3 epochs) on 1,000
    images at learning rates from $10^{-3}$ to $3$. we record the final training loss and the
    fraction of **dead** hidden units: ReLUs that output 0 for every training image.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the best learning rate is somewhere in the middle. what happens to the training loss at the largest "
        "learning rates: does it blow up to infinity or nan, or something else?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X, _y = X_all[:1000], y_all[:1000]
    _lrs = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
    _final, _rows = [], []
    for _lr in _lrs:
        _model = ex.Sequential(ex.Linear(784, 128, rng=1), ex.ReLU(), ex.Linear(128, 10, rng=2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _h = ex.fit(_model, ex.cross_entropy, SGD(_model.parameters(), lr=_lr), _X, _y, epochs=3,
                        batch_size=32, rng=0)
        _hidden = ex.predict(_model[:2], _X)
        _dead = np.mean(np.all(_hidden == 0, axis=0))
        _final.append(_h["train_loss"][-1])
        _rows.append(f"| {_lr:g} | {_h['train_loss'][0]:.3f} | {_h['train_loss'][-1]:.3f} | {_dead:.0%} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.semilogx(_lrs, _final, "o-")
    _ax.axhline(np.log(10), color="gray", ls=":", label="log 10")
    _ax.set(xlabel="learning rate", ylabel="training loss after 3 epochs")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| learning rate | loss, epoch 1 | loss, epoch 3 | dead hidden units |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the loss at the largest learning rates settle near $\log 10$
    instead of exploding? connect it to the dead-unit column: what does one huge update do to
    a ReLU's bias, and why can a dead ReLU never come back (what's its gradient)? what does
    the network predict once most hidden units are dead? where is the best learning rate
    relative to the one where things break?
    """)
    return


if __name__ == "__main__":
    app.run()
