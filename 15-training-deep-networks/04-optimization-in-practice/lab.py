import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.nn import Linear, ReLU, Sequential, cross_entropy, fit, predict  # yours, from ch 14 · topic 04
    from mlfs.optim import SGD, Adam, minibatch_indices  # yours, from ch 06 · topic 04
    from mlfs.train import LayerNorm, kaiming_normal_  # yours, from topics 01 and 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    X_all, y_all = (_X - _X.mean()) / _X.std(), _d["y_train"]
    X_val, y_val = X_all[-1000:], y_all[-1000:]


@app.function
def make_net(*, depth=8, width=128, norm=False, rng=0):
    """depth x [Linear (kaiming, zero bias) -> optional LayerNorm -> ReLU] -> Linear(width, 10)."""
    rng = np.random.default_rng(rng)
    layers, n_in = [], 784
    for _ in range(depth):
        lin = Linear(n_in, width, rng=rng)
        kaiming_normal_(lin.weight, rng=rng)
        lin.bias.data[...] = 0.0
        layers += [lin] + ([LayerNorm(width)] if norm else []) + [ReLU()]
        n_in = width
    layers.append(Linear(n_in, 10, rng=rng))
    return Sequential(*layers)


@app.function
def train_scheduled(net, opt, X, y, *, epochs, batch_size, warmup_steps=0, rng=0):
    """minibatch training with a CosineSchedule stepped once per optimizer step; returns validation accuracy."""
    steps = epochs * int(np.ceil(len(X) / batch_size))
    sched = ex.CosineSchedule(opt, total_steps=steps, warmup_steps=warmup_steps)
    rng = np.random.default_rng(rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # diverging runs overflow
        for _ in range(epochs):
            for idx in minibatch_indices(len(X), batch_size, rng=rng):
                opt.zero_grad()
                cross_entropy(net(X[idx]), y[idx]).backward()
                opt.step()
                sched.step()
        out = np.nan_to_num(predict(net, X_val))
    return float(np.mean(out.argmax(1) == y_val))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # optimization in practice · lab

    every experiment here uses **your** `lr_range_test`, `suggest_lr` and `CosineSchedule`
    from `exercise.py`, on MNIST digits (standardized pixels; offline, a synthetic stand-in
    with the same shapes), scored on 1,000 held-out images. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about a minute and a half.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the learning-rate range test

    an 8-layer ReLU network (128 wide, kaiming init) and Adam. the range test sweeps the
    learning rate from $10^{-6}$ to 1 over 150 minibatches of 64 and stops when the loss
    blows up; `suggest_lr` reads a learning rate off the curve. then we check the suggestion
    by actually training (2 epochs, cosine schedule, 4,000 images; offline 1,000) at the
    suggested learning rate, 10 times smaller and 10 times larger.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "Adam's default learning rate is 1e-3. will the range test suggest something larger or smaller for this "
        "8-layer network? and will training at 10x the suggestion be better or worse than at the suggestion?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _n = min(4000, len(X_all) - 1000)
    _Xtr, _ytr = X_all[:_n], y_all[:_n]
    _net = make_net(rng=0)
    _lrs, _losses = ex.lr_range_test(_net, cross_entropy, Adam(_net.parameters()), _Xtr, _ytr, min_lr=1e-6,
                                     max_lr=1.0, n_steps=150, batch_size=64, rng=0)
    _lr = ex.suggest_lr(_lrs, _losses)
    _rows = []
    for _name, _l in [("suggestion / 10", _lr / 10), ("suggestion", _lr), ("suggestion × 10", _lr * 10)]:
        _accs = []
        for _s in range(2):
            _m = make_net(rng=_s)
            _accs.append(train_scheduled(_m, Adam(_m.parameters(), lr=_l), _Xtr, _ytr, epochs=2, batch_size=64, rng=_s))
        _rows.append(f"| {_name} | {_l:.2e} | {np.mean(_accs):.3f} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.semilogx(_lrs, _losses, lw=1, label="loss on each minibatch")
    _ax.axvline(_lr, color="C2", ls="--", label=f"suggest_lr = {_lr:.1e}")
    _ax.axvline(_lr * 10, color="C3", ls=":", label="the minimum of the smoothed loss")
    _ax.set(xlabel="learning rate (Adam)", ylabel="training loss", ylim=(0, max(4.0, min(_losses) * 4.5)))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"the sweep stopped after {len(_lrs)} of 150 steps, at lr {_lrs[-1]:.2e}.\n\n"
                           "| learning rate | value | validation accuracy (2 epochs, mean of 2 seeds) |\n|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the loss stay flat for the first few orders of magnitude, fall,
    and then explode? why does `suggest_lr` back off by a factor of 10 from the minimum of the
    curve, instead of using the minimum itself (what is the loss at the minimum actually
    measuring, after only ~100 steps)? the range test took a fraction of a second; what did
    it save you?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: when warmup matters

    the 8-layer network, plain and with `LayerNorm` after every hidden `Linear`, trained with
    Adam for 2 epochs on a cosine schedule, **with no warmup** and with **linear warmup over
    the first 40% of the steps** (50 of 126; offline 12 of 32), at peak learning rates from
    $10^{-3}$ to $3 \times 10^{-2}$, 2 seeds each.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "Adam already adapts its step size per parameter. will warmup change the final accuracy at the default "
        "lr 1e-3? at 3e-2? does the layer-norm network need warmup more or less than the plain one?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _n = min(4000, len(X_all) - 1000)
    _Xtr, _ytr = X_all[:_n], y_all[:_n]
    _peak = [1e-3, 3e-3, 1e-2, 3e-2]
    _steps = 2 * int(np.ceil(_n / 64))
    _res, _rows = {}, []
    for _norm in [False, True]:
        for _warm in [0, int(0.4 * _steps)]:
            _accs = []
            for _lr in _peak:
                _a = []
                for _s in range(2):
                    _m = make_net(norm=_norm, rng=_s)
                    _a.append(train_scheduled(_m, Adam(_m.parameters(), lr=_lr), _Xtr, _ytr, epochs=2, batch_size=64,
                                              warmup_steps=_warm, rng=_s))
                _accs.append(np.mean(_a))
            _key = f"{'layer norm' if _norm else 'plain'}, {f'{_warm} warmup steps' if _warm else 'no warmup'}"
            _res[_key] = _accs
            _rows.append(f"| {_key} | " + " | ".join(f"{a:.2f}" for a in _accs) + " |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    for _i, (_k, _a) in enumerate(_res.items()):
        _ax.semilogx(_peak, _a, "o-" if "no warmup" in _k else "s--", color=f"C{_i // 2}", label=_k)
    _ax.set(xlabel="peak learning rate (Adam)", ylabel="validation accuracy", ylim=(0, 1))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| network, schedule | " + " | ".join(f"lr {l:g}" for l in _peak) + " |\n|---|"
                           + "---|" * len(_peak) + "\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at the first step Adam's update for every weight is about $\pm\eta$,
    whatever the gradient's size (why? what are $\hat m$ and $\hat v$ after one step?). with
    $\eta = 0.03$ and a weight of typical size 0.1, what does that first step do? why does
    warmup make no difference at $10^{-3}$? the layer-norm network is the most fragile
    without warmup and the most robust with it: why might normalization make the *early*
    steps more dangerous? (the same mechanism is why transformers are almost always trained
    with warmup.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: batch size and the learning rate

    a $784 \to 256 \to 256 \to 10$ network, SGD with momentum 0.9, 2 epochs on 8,000 images
    (offline 1,000), at batch sizes from 8 to 2,048. once with a **fixed** learning rate of
    0.01, and once with the **linear scaling rule**: $\eta = 0.01 \times B/32$. 2 seeds each.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with a fixed learning rate, what happens to the accuracy after 2 epochs as the batch grows from 8 to 2048? does "
        "scaling the learning rate with the batch size fix it all the way to 2048?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _n = min(8000, len(X_all) - 1000)
    _Xtr, _ytr = X_all[:_n], y_all[:_n]
    _sizes = [8, 32, 128, 512, 2048]
    _res, _rows = {}, []
    for _rule in ["fixed lr 0.01", "linear scaling: 0.01 × B / 32"]:
        _accs = []
        for _bs in _sizes:
            _lr = 0.01 if _rule.startswith("fixed") else 0.01 * _bs / 32
            _a = []
            for _s in range(2):
                _r = np.random.default_rng(_s)
                _m = Sequential(Linear(784, 256, rng=_r), ReLU(), Linear(256, 256, rng=_r), ReLU(), Linear(256, 10, rng=_r))
                kaiming_normal_(_m[0].weight, rng=_r)
                kaiming_normal_(_m[2].weight, rng=_r)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit(_m, cross_entropy, SGD(_m.parameters(), lr=_lr, momentum=0.9), _Xtr, _ytr, epochs=2,
                        batch_size=_bs, rng=_s)
                    _a.append(float(np.mean(np.nan_to_num(predict(_m, X_val)).argmax(1) == y_val)))
            _accs.append(np.mean(_a))
        _res[_rule] = _accs
        _rows.append(f"| {_rule} | " + " | ".join(f"{a:.3f}" for a in _accs) + " |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    for _k, _a in _res.items():
        _ax.semilogx(_sizes, _a, "o-", label=_k, base=2)
    _ax.set(xlabel="batch size B", ylabel="validation accuracy after 2 epochs", ylim=(0, 1))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _steps = " · ".join(f"B={b}: {2 * int(np.ceil(_n / b))}" for b in _sizes)
    mo.vstack([_fig, mo.md(f"optimizer steps in 2 epochs: {_steps}.\n\n| learning rate | "
                           + " | ".join(f"B = {b}" for b in _sizes) + " |\n|---|" + "---|" * len(_sizes) + "\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with a fixed learning rate and a fixed number of epochs, why do large
    batches do worse (count the optimizer steps)? the linear scaling rule says: multiply the
    batch by $k$, multiply the learning rate by $k$. why does that keep the *expected* change
    of the weights per epoch about the same? what breaks at $B = 2048$, where the rule asks
    for $\eta = 0.64$ with momentum 0.9? what would you add to the large-batch run (hint:
    experiment 2)?
    """)
    return


if __name__ == "__main__":
    app.run()
