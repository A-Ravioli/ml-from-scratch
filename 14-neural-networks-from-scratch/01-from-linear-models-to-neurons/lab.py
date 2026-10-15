import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    X_XOR = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y_XOR = np.array([0.0, 1.0, 1.0, 0.0])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # from linear models to neurons · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about fifteen seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: depth without nonlinearity

    two parts. first, random networks of 1 to 10 linear layers (width 16, no activation)
    are compared with the single layer your `collapse_linear` makes from them. second, your
    `train_two_layer` fits a 2D "ring" dataset (class 1 inside a circle, class 0 around it)
    with 16 hidden units, once with `tanh` and once with the identity as activation.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how close is a 10-layer linear network to its collapsed single layer? on the ring data, what accuracy "
        "will the network without an activation reach, and what will its decision boundary look like?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _grid = _rng.normal(size=(500, 2))
    _gaps = []
    for _depth in range(1, 11):
        _widths = [2] + [16] * (_depth - 1) + [1]
        _params = [(_rng.normal(size=(o, i)) / np.sqrt(i), _rng.normal(size=o)) for i, o in zip(_widths[:-1], _widths[1:])]
        _W, _b = ex.collapse_linear(_params)
        _gaps.append(np.max(np.abs(ex.mlp_forward(_grid, _params, lambda z: z) - (_grid @ _W.T + _b))))
    _X = _rng.uniform(-1.5, 1.5, size=(300, 2))
    _y = (np.sum(_X ** 2, axis=1) < 0.8).astype(float)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _xx, _yy = np.meshgrid(np.linspace(-1.5, 1.5, 150), np.linspace(-1.5, 1.5, 150))
    _G = np.c_[_xx.ravel(), _yy.ravel()]
    _rows = []
    for _ax, _act in zip(_axes, ["tanh", "identity"]):
        _f = np.tanh if _act == "tanh" else (lambda z: z)
        _p, _losses = ex.train_two_layer(_X, _y, 16, lr=0.5, n_steps=3000, activation=_act, rng=1)
        _acc = np.mean((ex.mlp_forward(_X, _p, _f)[:, 0] > 0) == _y)
        _Z = ex.sigmoid(ex.mlp_forward(_G, _p, _f)[:, 0]).reshape(_xx.shape)
        _ax.contourf(_xx, _yy, _Z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.6)
        _ax.contour(_xx, _yy, _Z, levels=[0.5], colors="k")
        _ax.scatter(_X[:, 0], _X[:, 1], c=_y, cmap="RdBu_r", s=10, edgecolors="k", linewidths=0.3)
        _ax.set(title=f"activation = {_act}: accuracy {_acc:.0%}", aspect="equal")
        _rows.append(f"| {_act} | {_losses[-1]:.3f} | {_acc:.1%} |")
    _fig.tight_layout()
    mo.vstack([mo.md("largest |deep - collapsed| over 500 points, depth 1 to 10: "
                     + ", ".join(f"{g:.1e}" for g in _gaps)),
               _fig,
               mo.md("| activation | final loss | training accuracy |\n|---|---|---|\n" + "\n".join(_rows)
                     + f"\n\nalways guessing the majority class scores {max(_y.mean(), 1 - _y.mean()):.0%}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the gap between a 10-layer linear network and its collapsed layer
    at the level of rounding error? what's the best a linear decision boundary can do on
    the ring, and how does the identity network's accuracy compare with always guessing the
    majority class? what does that say about adding depth without nonlinearity?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: big enough isn't enough

    your stage 3 network shows that **2** hidden units can compute XOR exactly. here your
    `train_two_layer` (tanh, learning rate 1, 1,500 steps of full-batch gradient descent from
    a standard normal init) tries to learn XOR with 1, 2, 3, 4 and 8 hidden units, from 50
    different random initializations each. a run counts as solved if all four corners end up
    on the right side of 0.5.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what fraction of the 50 runs will solve XOR with 2 hidden units? with 8? and with 1?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _widths = [1, 2, 3, 4, 8]
    _rates, _rows = [], []
    for _h in _widths:
        _ok, _finals = 0, []
        for _seed in range(50):
            _p, _l = ex.train_two_layer(X_XOR, Y_XOR, _h, lr=1.0, n_steps=1500, rng=_seed)
            _pred = ex.sigmoid(ex.mlp_forward(X_XOR, _p, np.tanh))[:, 0]
            _ok += int(np.all((_pred > 0.5) == Y_XOR))
            _finals.append(_l[-1])
        _rates.append(_ok / 50)
        _rows.append(f"| {_h} | {_ok}/50 | {np.median(_finals):.3f} | {np.max(_finals):.3f} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.bar([str(h) for h in _widths], _rates)
    _ax.set(xlabel="hidden units", ylabel="fraction of runs that solve XOR", ylim=(0, 1.05))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| hidden units | solved | median final loss | worst final loss |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can't 1 hidden unit ever solve XOR (what does its decision boundary
    look like)? 2 units are enough in principle: why does gradient descent find a solution
    only some of the time? what changes with 8 units, even though 6 of them are "unnecessary"?
    (look at the worst final loss: what does a stuck run converge to?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what the hidden layer does

    where do the four XOR corners go in **hidden space**? we plot the hidden activations
    $h = \text{act}(W_1x + b_1)$ of the four points for your hand-wired ReLU network from
    stage 3 and for a trained 2-unit tanh network, with the output layer's decision line
    $W_2h + b_2 = 0$ (relu network: $= 0.5$).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the four corners aren't linearly separable in the input plane. where do they land in hidden space, and "
        "how does the output layer, which is linear, separate them?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _hand = ex.xor_params()
    _trained = None
    for _seed in range(50):
        _p, _l = ex.train_two_layer(X_XOR, Y_XOR, 2, lr=1.0, n_steps=1500, rng=_seed)
        if np.all((ex.sigmoid(ex.mlp_forward(X_XOR, _p, np.tanh))[:, 0] > 0.5) == Y_XOR):
            _trained = _p
            break
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4))
    _cols = ["C0" if t == 0 else "C3" for t in Y_XOR]
    _axes[0].scatter(X_XOR[:, 0], X_XOR[:, 1], c=_cols, s=120)
    for (_a, _b), _t in zip(X_XOR, Y_XOR):
        _axes[0].annotate(f"({_a:.0f},{_b:.0f}) → {_t:.0f}", (_a, _b), textcoords="offset points", xytext=(6, 6))
    _axes[0].set(title="input space", xlim=(-0.4, 1.6), ylim=(-0.4, 1.6), aspect="equal")
    for _ax, (_name, _params, _act, _thr) in zip(_axes[1:], [("hand-wired relu", _hand, ex.relu, 0.5),
                                                                ("trained tanh (2 units)", _trained, np.tanh, 0.0)]):
        (_W1, _b1), (_W2, _b2) = _params
        _H = _act(ex.linear_layer(X_XOR, _W1, _b1))
        _ax.scatter(_H[:, 0], _H[:, 1], c=_cols, s=120)
        _groups = {}
        for (_a, _b), _hpt in zip(X_XOR, _H):  # corners that land on the same hidden point share a label
            _groups.setdefault(tuple(np.round(_hpt, 2)), []).append(f"({_a:.0f},{_b:.0f})")
        for _hpt, _names in _groups.items():
            _ax.annotate(" & ".join(_names), _hpt, textcoords="offset points", xytext=(6, 6))
        _lo, _hi = _H.min() - 0.5, _H.max() + 0.5
        _t = np.linspace(_lo, _hi, 50)
        if abs(_W2[0, 1]) > 1e-12:
            _ax.plot(_t, (_thr - _b2[0] - _W2[0, 0] * _t) / _W2[0, 1], "k--", label="output decision line")
        _ax.set(title=f"hidden space: {_name}", xlim=(_lo, _hi), ylim=(_lo, _hi), aspect="equal",
                xlabel="h1", ylabel="h2")
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in the hand-wired network, which two corners land on the same hidden point,
    and why does that make the problem linear? what's the analogous move in the trained
    network? in one sentence: what is a hidden layer *for*?
    """)
    return


if __name__ == "__main__":
    app.run()
