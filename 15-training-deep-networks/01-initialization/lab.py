import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.nn import cross_entropy, fit, predict  # yours, from ch 14 · topic 04
    from mlfs.optim import SGD  # yours, from ch 06 · topic 04
    from mlfs.tensor import Tensor, no_grad  # yours, from ch 14 · topic 03

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    X_all = (_X - _X.mean()) / _X.std()  # standardized pixels: mean 0, variance 1 overall
    y_all = _d["y_train"]

    CONFIGS = {  # (activation, init) for the 50-layer experiments
        "tanh · N(0, 0.01²)": ("tanh", 0.01),
        "tanh · xavier": ("tanh", "xavier"),
        "relu · xavier": ("relu", "xavier"),
        "relu · kaiming": ("relu", "kaiming"),
        "relu · torch default": ("relu", "default"),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # initialization · lab

    every experiment here runs **your** initializers, `deep_mlp` and `activation_stats` from
    `exercise.py`, on MNIST digits (standardized pixels; offline, a synthetic stand-in with
    the same shapes). for each one: write down a prediction, lock it in, run it, then explain
    what happened. compare your explanations with `solutions/lab_notes.md` afterwards. the
    whole lab takes about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: fifty layers of signal

    five 50-layer networks, 256 units wide ($784 \to 256 \times 50 \to 10$), differ only in
    their activation and their initialization. we push 1,000 images through each one and
    record, after every activation, the standard deviation of the activations, and then the
    norm of each layer's weight gradient for a cross-entropy loss on 256 of them.

    - **tanh · N(0, 0.01²)**: small random weights, the "safe" choice of early neural-net folklore;
    - **tanh · xavier** and **relu · xavier**: Glorot's variance $2/(n_\text{in} + n_\text{out})$, gain 1;
    - **relu · kaiming**: He's variance $2/n_\text{in}$;
    - **relu · torch default**: `Linear`'s own $U(-1/\sqrt{n}, 1/\sqrt{n})$ weights *and biases*, what you used in chapter 14.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which of the five networks still has activations with std near 1 at layer 50? for the others, roughly "
        "what is the std at layer 50: 0.1? 1e-5? 1e-40?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _Xb, _yb = X_all[:1000], y_all[:1000]
    signal = {}
    _grads, _rows = {}, []
    for _name, (_act, _init) in CONFIGS.items():
        _net = ex.deep_mlp([784] + [256] * 50 + [10], activation=_act, init=_init, rng=0)
        _stats = ex.activation_stats(_net, _Xb)
        _post = [s for s in _stats if "ReLU" in s["name"] or "Tanh" in s["name"]]
        signal[_name] = (_net, _post)
        for _p in _net.parameters():
            _p.grad = None
        cross_entropy(_net(_Xb[:256]), _yb[:256]).backward()
        _grads[_name] = [float(np.linalg.norm(_net[2 * i].weight.grad)) for i in range(51)]
        _s = [_post[i]["std"] for i in (0, 9, 49)]
        _rows.append(f"| {_name} | {_s[0]:.3g} | {_s[1]:.3g} | {_s[2]:.3g} | {_grads[_name][0]:.3g} | "
                     f"{_grads[_name][50]:.3g} |")
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _name, (_net, _post) in signal.items():
        _axes[0].semilogy(np.arange(1, 51), [max(s["std"], 1e-300) for s in _post], label=_name)
        _axes[1].semilogy(np.arange(1, 52), np.maximum(_grads[_name], 1e-300), label=_name)
    _axes[0].set(xlabel="layer", ylabel="std of the activations", title="forward: the signal")
    _axes[1].set(xlabel="layer (1 = closest to the input)", ylabel="‖∂L/∂W‖", title="backward: the gradients")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| network | std, layer 1 | std, layer 10 | std, layer 50 | ‖∂L/∂W‖, layer 1 | "
                           "‖∂L/∂W‖, layer 51 |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (signal,)


@app.cell
def _():
    pick = mo.ui.dropdown(options=list(CONFIGS), value="relu · kaiming", label="network")
    pick
    return (pick,)


@app.cell
def _(pick, signal):
    # histograms of one network's activations at four depths (turn the dropdown to compare)
    _net, _ = signal[pick.value]
    _h = Tensor(X_all[:1000])
    _keep = {}
    with no_grad():
        for _i in range(len(_net) - 1):
            _h = _net[_i](_h)
            if _i % 2 == 1 and (_i // 2 + 1) in (1, 10, 25, 50):
                _keep[_i // 2 + 1] = _h.data.ravel()
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for _ax, (_layer, _a) in zip(_axes, _keep.items()):
        _ax.hist(_a, bins=60, color="C0")
        _ax.set(title=f"layer {_layer}: std {_a.std():.2g}", xlabel="activation")
        _ax.set_yticks([])
    _fig.suptitle(pick.value)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** use the variance-propagation law $\operatorname{Var}(z_{l+1}) =
    n\operatorname{Var}(W)\,E[h_l^2]$ to predict the factor by which each network's signal
    changes per layer, and check it against the plot (relu · xavier: its std should shrink by $2^{-49/2}$ from layer 1 to layer 50; does it?). why
    does tanh · xavier decay slowly instead of geometrically? the torch default's std stops
    falling near 0.02 instead of vanishing: where does that floor come from (look at what
    `init='default'` keeps that the others zero)? how do the gradient norms mirror the
    forward signal?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: can the network still tell two images apart?

    a network whose activations keep a healthy scale might still be useless, if every input
    ends up at the *same* point. for the same 50-layer networks we measure the average
    **cosine similarity** between the activation vectors of 500 different images, after
    layers 1, 10, 25 and 50. two unrelated images in pixel space have a cosine similarity of
    about 0.3.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "relu · kaiming keeps the activations' scale constant through 50 layers. does it also keep different images "
        "apart (cosine similarity staying near the input's 0.3), or do their representations drift together?"
    )
    p2
    return (p2,)


@app.cell
def _(p2, signal):
    ui.gate(p2)

    def _mean_cos(h):
        _u = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-300)
        _c = _u @ _u.T
        return float(_c[np.triu_indices(len(h), 1)].mean())

    _Xb = X_all[:500]
    _layers = (1, 5, 10, 25, 50)
    _rows, _curves = [], {}
    for _name in ["tanh · xavier", "relu · xavier", "relu · kaiming", "relu · torch default"]:
        _net, _ = signal[_name]
        _h, _cos = Tensor(_Xb), []
        with no_grad():
            for _i in range(len(_net) - 1):
                _h = _net[_i](_h)
                if _i % 2 == 1 and (_i // 2 + 1) in _layers:
                    _cos.append(_mean_cos(_h.data))
        _curves[_name] = _cos
        _rows.append(f"| {_name} | " + " | ".join(f"{c:.3f}" for c in _cos) + " |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    for _name, _c in _curves.items():
        _hollow = _name == "relu · xavier"  # it sits exactly under relu · kaiming
        _ax.plot(_layers, _c, "s--" if _hollow else "o-", ms=10 if _hollow else 5, mfc="none" if _hollow else None,
                 label=_name)
    _ax.axhline(_mean_cos(_Xb), color="gray", ls=":", label=f"pixels: {_mean_cos(_Xb):.2f}")
    _ax.set(xlabel="layer", ylabel="mean cosine similarity between images", ylim=(0, 1.02))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| network | " + " | ".join(f"layer {l}" for l in _layers) + " |\n|---|"
                           + "---|" * len(_layers) + "\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do relu · xavier and relu · kaiming give *identical* cosine
    similarities (what does rescaling every weight do to a cosine, given that ReLU is
    positively homogeneous)? why does a ReLU layer push two vectors closer together (think
    about what it does to two vectors with negative entries in different places)? what does
    the torch default's similarity of 1.000 say about its layer-50 output? what does this
    imply about how deep a plain ReLU network can usefully be, even with a perfect variance
    scale?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: does it train?

    20-layer networks, 128 units wide, trained with your `fit` and SGD (learning rate 0.01,
    momentum 0.9, batch size 64) for 3 epochs on 5,000 training images (offline: 1,000),
    then scored on 1,000 held-out images. each configuration runs with 3 seeds.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "rank these by validation accuracy after 3 epochs: relu · torch default, relu · xavier, relu · kaiming, "
        "tanh · xavier. will any of them stay at chance (about 10%)?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _n_tr = min(5000, len(X_all) - 1000)
    _Xtr, _ytr, _Xv, _yv = X_all[:_n_tr], y_all[:_n_tr], X_all[-1000:], y_all[-1000:]
    _rows, _curves = [], {}
    for _name, (_act, _init) in [("relu · torch default", ("relu", "default")), ("relu · xavier", ("relu", "xavier")),
                                 ("relu · kaiming", ("relu", "kaiming")), ("tanh · xavier", ("tanh", "xavier"))]:
        _accs, _losses = [], []
        for _seed in range(3):
            _net = ex.deep_mlp([784] + [128] * 20 + [10], activation=_act, init=_init, rng=_seed)
            _h = fit(_net, cross_entropy, SGD(_net.parameters(), lr=0.01, momentum=0.9), _Xtr, _ytr, epochs=3,
                     batch_size=64, rng=_seed)
            _accs.append(float(np.mean(predict(_net, _Xv).argmax(1) == _yv)))
            _losses.append(_h["train_loss"])
        _curves[_name] = np.mean(_losses, axis=0)
        _rows.append(f"| {_name} | " + " · ".join(f"{a:.3f}" for a in _accs) + f" | {np.mean(_accs):.3f} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    for _name, _c in _curves.items():
        _ax.plot(np.arange(1, 4), _c, "o-", label=_name)
    _ax.axhline(np.log(10), color="gray", ls=":", label="log 10")
    _ax.set(xlabel="epoch", ylabel="training loss (mean of 3 seeds)", xticks=[1, 2, 3])
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| network (20 layers) | validation accuracy, 3 seeds | mean |\n|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** connect each result to experiments 1 and 2. why can't SGD rescue the
    torch default even though its last layer gets a reasonable gradient? relu · xavier at
    20 layers starts with a signal shrunk by $2^{-20}$: why does it learn *something*, and
    why so unevenly across seeds? tanh · xavier wasn't variance-preserving in experiment 1,
    yet it trains well here: what does experiment 2 suggest is going on? which init would
    you pick for a 20-layer ReLU network, and what would you still worry about?
    """)
    return


if __name__ == "__main__":
    app.run()
