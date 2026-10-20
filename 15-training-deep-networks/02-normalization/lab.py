import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.nn import Linear, ReLU, Sequential, cross_entropy, fit, predict  # yours, from ch 14 · topic 04
    from mlfs.optim import SGD  # yours, from ch 06 · topic 04
    from mlfs.tensor import no_grad  # yours, from ch 14 · topic 03
    from mlfs.train import kaiming_normal_  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    _mu, _sd = _X.mean(), _X.std()
    X_all, y_all = (_X - _mu) / _sd, _d["y_train"]
    X_test = (_d["X_test"].reshape(len(_d["X_test"]), -1) / 255.0 - _mu) / _sd
    y_test = _d["y_test"]
    N_TRAIN = min(4000, len(X_all) - 1000)
    X_tr, y_tr = X_all[:N_TRAIN], y_all[:N_TRAIN]
    X_val, y_val = X_all[-1000:], y_all[-1000:]


@app.function
def make_net(norm, *, depth=8, width=128, rng=0):
    """depth x [Linear (kaiming, zero bias) -> optional norm -> ReLU] -> Linear(width, 10). norm: 'none' | 'batch' | 'layer'."""
    rng = np.random.default_rng(rng)
    layers, n_in = [], 784
    for _ in range(depth):
        lin = Linear(n_in, width, rng=rng)
        kaiming_normal_(lin.weight, rng=rng)
        lin.bias.data[...] = 0.0
        layers.append(lin)
        if norm == "batch":
            layers.append(ex.BatchNorm1d(width))
        elif norm == "layer":
            layers.append(ex.LayerNorm(width))
        layers.append(ReLU())
        n_in = width
    layers.append(Linear(n_in, 10, rng=rng))
    return Sequential(*layers)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # normalization · lab

    every experiment here runs **your** `BatchNorm1d` and `LayerNorm` from `exercise.py`
    (and your kaiming init from topic 01), inside 8-layer ReLU networks, 128 units wide, on
    MNIST digits (standardized pixels; offline, a synthetic stand-in with the same shapes).
    for each one: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes
    about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how large can the learning rate be?

    the same 8-layer network, with no normalization, with `BatchNorm1d` after every hidden
    `Linear`, and with `LayerNorm` there instead, trained with plain SGD (no momentum,
    batch size 64) for 2 epochs on 4,000 images (offline: 1,000) at learning rates from
    0.01 to 3. we report validation accuracy on 1,000 held-out images, averaged over 2
    seeds.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the plain network diverges somewhere between lr 0.3 and 1. how much further can the batch-norm network go? "
        "and the layer-norm network?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _lrs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    _res = {}
    for _norm in ["none", "batch", "layer"]:
        _res[_norm] = []
        for _lr in _lrs:
            _accs = []
            for _seed in range(2):
                _net = make_net(_norm, rng=_seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # the diverging runs overflow
                    fit(_net, cross_entropy, SGD(_net.parameters(), lr=_lr), X_tr, y_tr, epochs=2, batch_size=64,
                        rng=_seed)
                    _p = np.nan_to_num(predict(_net, X_val))
                _accs.append(float(np.mean(_p.argmax(1) == y_val)))
            _res[_norm].append(np.mean(_accs))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    for _norm, _a in _res.items():
        _ax.semilogx(_lrs, _a, "o-", label=f"norm = {_norm}")
    _ax.axhline(0.1, color="gray", ls=":", label="chance")
    _ax.set(xlabel="learning rate (SGD)", ylabel="validation accuracy after 2 epochs", ylim=(0, 1))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {n} | " + " | ".join(f"{a:.2f}" for a in v) + " |" for n, v in _res.items())
    mo.vstack([_fig, mo.md("| normalization | " + " | ".join(f"lr {l:g}" for l in _lrs) + " |\n|---|"
                           + "---|" * len(_lrs) + "\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can the batch-norm network take a learning rate several times larger
    before it breaks? (think about what happens to the next layer's input if one step makes a
    weight matrix 10 times larger, with and without a normalization after it.) why is
    batch norm *worse* than no normalization at the smallest learning rate? layer norm
    normalizes too, so why doesn't it buy the same tolerance here (what does it normalize
    over, and what can a single large unit do to that)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: forgetting `model.eval()`

    we train one batch-norm network (lr 0.1, 2 epochs) and score it on 2,000 test images:
    first properly, in eval mode (running statistics), then in **training mode**, as if
    someone forgot `model.eval()`, with test batches of different sizes, and finally with
    test batches that happen to be **sorted by class** (as a test loader without shuffling
    might give you).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the network scores about 86% in eval mode. what does it score in training mode with batches of 64, and with "
        "batches of 64 that each contain a single class?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _net = make_net("batch", rng=0)
    fit(_net, cross_entropy, SGD(_net.parameters(), lr=0.1), X_tr, y_tr, epochs=2, batch_size=64, rng=0)
    _Xt, _yt = X_test[:2000], y_test[:2000]
    _rows = [f"| eval mode (running statistics) | any | {np.mean(predict(_net, _Xt).argmax(1) == _yt):.3f} |"]

    def _train_mode_acc(order, bs):
        _m = copy.deepcopy(_net)  # training mode updates the running stats: work on a copy
        _m.train()
        _pred = np.full(len(_yt), -1)
        with no_grad():
            for _s in range(0, len(order), bs):
                _idx = order[_s:_s + bs]
                if len(_idx) > 1:  # a batch of one can't be normalized in training mode
                    _pred[_idx] = _m(_Xt[_idx]).data.argmax(1)
        return float(np.mean(_pred == _yt))

    _rng = np.random.default_rng(0)
    _sizes = [1000, 64, 8, 2]
    _rand = [_train_mode_acc(_rng.permutation(len(_yt)), b) for b in _sizes]
    _rows += [f"| training mode, shuffled batches | {b} | {a:.3f} |" for b, a in zip(_sizes, _rand)]
    _sorted = np.argsort(_yt, kind="stable")
    _srt = [_train_mode_acc(_sorted, b) for b in (1000, 64)]
    _rows += [f"| training mode, batches sorted by class | {b} | {a:.3f} |" for b, a in zip((1000, 64), _srt)]
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.semilogx(_sizes, _rand, "o-", label="training mode, shuffled")
    _ax.semilogx([1000, 64], _srt, "s--", label="training mode, sorted by class")
    _ax.axhline(np.mean(predict(_net, _Xt).argmax(1) == _yt), color="C2", label="eval mode")
    _ax.set(xlabel="test batch size", ylabel="test accuracy", ylim=(0, 1))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| how it's scored | test batch size | accuracy |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in training mode, an example's output depends on the other examples in
    its batch. why does that barely matter for batches of 1,000 and matter enormously for
    batches of 2? what exactly does batch norm remove from a batch in which every image is
    a 3 (think about which features distinguish 3s from other digits)? why is this a
    *silent* bug, and which line of code prevents it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: tiny batches

    the three networks once more, trained for 1 epoch with a batch size of **4** (learning
    rate 0.01) and, for comparison, of 64 (learning rate 0.1), 2 seeds each. choose the
    batch size below and compare.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with batches of 4, each batch-norm statistic is estimated from 4 numbers. which network does best now: "
        "none, batch norm or layer norm?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    small_batch = {}
    for _bs, _lr in [(4, 0.01), (64, 0.1)]:
        for _norm in ["none", "batch", "layer"]:
            _accs, _losses = [], []
            for _seed in range(2):
                _net = make_net(_norm, rng=_seed + 1)
                _h = fit(_net, cross_entropy, SGD(_net.parameters(), lr=_lr), X_tr, y_tr, epochs=1, batch_size=_bs,
                         rng=_seed)
                _accs.append(float(np.mean(predict(_net, X_val).argmax(1) == y_val)))
                _losses.append(_h["train_loss"][-1])
            small_batch[(_bs, _norm)] = (np.mean(_accs), np.mean(_losses))
    return (small_batch,)


@app.cell
def _():
    bs_pick = mo.ui.radio(options=["4", "64"], value="4", label="batch size")
    bs_pick
    return (bs_pick,)


@app.cell
def _(bs_pick, small_batch):
    _bs = int(bs_pick.value)
    _norms = ["none", "batch", "layer"]
    _fig, _ax = plt.subplots(figsize=(5, 3.2))
    _ax.bar(_norms, [small_batch[(_bs, n)][0] for n in _norms], color=["C0", "C1", "C2"])
    _ax.set(ylabel="validation accuracy (mean of 2 seeds)", ylim=(0, 1), title=f"batch size {_bs}, 1 epoch")
    _fig.tight_layout()
    _rows = "\n".join(f"| {b} | {n} | {small_batch[(b, n)][0]:.3f} | {small_batch[(b, n)][1]:.3f} |"
                      for b in (4, 64) for n in _norms)
    mo.vstack([_fig, mo.md("| batch size | normalization | validation accuracy | final training loss |\n"
                           "|---|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does batch norm fall behind with batches of 4, while layer norm
    doesn't care? (two separate problems: the noise in a 4-sample mean and variance during
    training, and the mismatch between those noisy statistics and the running averages used
    at evaluation.) which normalization would you choose for a model trained one sequence at
    a time, like a language model, and why do transformers use layer norm (chapter 18)?
    """)
    return


if __name__ == "__main__":
    app.run()
