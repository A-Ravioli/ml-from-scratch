import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import evaluate, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.cifar10()  # real CIFAR-10, or a small synthetic stand-in when offline
    _n = min(5000, len(_d["X_train"]))
    _mean = _d["X_train"][:_n].reshape(-1, 3).mean(0) / 255
    _std = _d["X_train"][:_n].reshape(-1, 3).std(0) / 255
    def _prep(X):  # standardize, NCHW, and average 2x2 blocks: 16x16 images, so four ResNets train on a CPU
        x = ((X / 255.0 - _mean) / _std).transpose(0, 3, 1, 2)
        return x.reshape(len(x), 3, 16, 2, 16, 2).mean(axis=(3, 5)).astype(np.float32)

    X_train, y_train = _prep(_d["X_train"][:_n]), _d["y_train"][:_n]
    X_test, y_test = _prep(_d["X_test"][:2000]), _d["y_test"][:2000]
    DEPTHS = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}
    WIDTH = 8
    N2 = 3000  # training images for experiment 2


@app.function
def make_net(depth, residual, seed=0):
    """your ResNet with ResNet-18 or ResNet-34 block counts, base width WIDTH, CIFAR stem."""
    torch.manual_seed(seed)
    return ex.ResNet(DEPTHS[depth], 10, base_width=WIDTH, residual=residual)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # residual networks · lab

    every experiment here uses **your** `ResNet` and `BasicBlock` from `exercise.py`, in
    up to four versions: 18 or 34 layers (block counts $(2, 2, 2, 2)$ or $(3, 4, 6, 3)$),
    residual or plain (`residual=False`: the same layers without the additions), all with
    batch norm and base width {WIDTH} so they train on a CPU. data: 5,000 CIFAR-10 training
    and 2,000 test images, downsampled to $16 \times 16$ (offline, a synthetic stand-in). for each experiment: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: gradients at initialization

    one batch of 256 images, one backward pass, no training: the gradient norm of each
    block's first convolution, from the input end (block 1) to the output end, for the
    four networks. all have batch norm and kaiming initialization.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "all four networks have batch norm, which keeps every layer's activations at unit scale. in the plain "
        "34-layer net, will the gradient at the first blocks be about the same as at the last blocks, much smaller, "
        "or much larger? and in the residual one?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _x, _y = torch.tensor(X_train[:256]), torch.tensor(y_train[:256])
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _rows = []
    for _depth in (18, 34):
        for _res in (True, False):
            _m = make_net(_depth, _res)
            F.cross_entropy(_m(_x), _y).backward()
            _blocks = [b for b in _m.modules() if isinstance(b, ex.BasicBlock)]
            _g = np.array([b.conv1.weight.grad.norm().item() for b in _blocks])
            _ax.semilogy(np.arange(1, len(_g) + 1), _g, "o-" if _res else "s--", ms=4,
                         label=f"{'residual' if _res else 'plain'}-{_depth}")
            _rows.append(f"| {'residual' if _res else 'plain'}-{_depth} | {_g[0]:.2e} | {_g[-1]:.2e} | "
                         f"{_g[0] / _g[-1]:.3f} |")
    _ax.set(xlabel="block (input → output)", ylabel="‖∂L/∂W‖ of the block's first conv")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| network | first block | last block | first / last |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** batch norm fixes the *forward* scale of every layer. why doesn't that
    fix the scale of the *gradients* in the plain network (the backward pass multiplies 33
    layer Jacobians, and each batch-norm layer divides by a batch standard deviation that
    depends on the input)? in which direction did they go wrong? write the gradient of a
    residual block's input, $\partial (x + F(x))/\partial x = I + \partial F/\partial x$:
    why does the product over 16 blocks stay much closer to the identity?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the degradation problem

    three of the networks trained for 4 epochs on the first 3,000 training images (SGD,
    learning rate 0.05, momentum 0.9, batch 64, the same data order). we report the mean training loss over the last epoch
    (in training mode, as the optimizer saw it), the training accuracy, and the test
    accuracy. the 34-layer networks
    contain every layer of the 18-layer ones and more, so they can represent anything the
    18-layer ones can.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "rank the three by final TRAINING loss: plain-18, plain-34, residual-34. does the deeper plain network fit "
        "the training data better or worse than the shallower one?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    trained, _curves, _rows = {}, {}, []
    for _depth, _res in ((18, False), (34, False), (34, True)):
        _name = f"{'residual' if _res else 'plain'}-{_depth}"
        _m = make_net(_depth, _res)
        _opt = torch.optim.SGD(_m.parameters(), lr=0.05, momentum=0.9)
        _loader = make_loader(X_train[:N2], y_train[:N2], batch_size=64, seed=0)
        _t0, _losses = time.perf_counter(), []
        for _epoch in range(4):
            _m.train()
            for _xb, _yb in _loader:
                _opt.zero_grad()
                _loss = F.cross_entropy(_m(_xb), _yb)
                _loss.backward()
                _opt.step()
                _losses.append(_loss.item())
        _test = evaluate(_m, make_loader(X_test, y_test, batch_size=500, shuffle=False))["accuracy"]
        _train = evaluate(_m, make_loader(X_train[:N2], y_train[:N2], batch_size=500, shuffle=False))
        trained[_name] = _m
        _curves[_name] = np.convolve(_losses, np.ones(20) / 20, mode="valid")
        _last = float(np.mean(_losses[-len(_loader):]))
        _rows.append(f"| {_name} | {sum(p.numel() for p in _m.parameters()):,} | {_last:.3f} | "
                     f"{_train['accuracy']:.3f} | {_test:.3f} | {time.perf_counter() - _t0:.0f} s |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    for _name, _c in _curves.items():
        _ax.plot(_c, "-" if _name.startswith("residual") else "--", label=_name)
    _ax.set(xlabel="step", ylabel="training loss (moving average of 20)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| network | parameters | training loss (last epoch) | training accuracy | test accuracy "
                           "| time |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (trained,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the plain-34 network could copy plain-18 and make its extra layers
    identities. why doesn't the optimizer find that solution? what is the default
    behaviour of a residual block whose weights are small, and why does that make extra
    blocks harmless? is the plain-34 problem overfitting (compare the training and test
    columns)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: delete a block

    take the trained 34-layer networks from experiment 2 and remove one block at a time
    (replace it with the identity), then measure test accuracy. only blocks whose input and
    output shapes match can be removed: the 13 of the 16 that don't downsample.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "removing one of 16 blocks from a trained network: how much test accuracy does the residual-34 lose on "
        "average? and the plain-34?"
    )
    p3
    return (p3,)


@app.cell
def _(p3, trained):
    ui.gate(p3)
    _test = make_loader(X_test, y_test, batch_size=500, shuffle=False)
    _res = {}
    for _name in ("residual-34", "plain-34"):
        _m = trained[_name]
        _base = evaluate(_m, _test)["accuracy"]
        _accs = []
        for _s, _stage in enumerate(_m.layers):
            for _b in range(len(_stage)):
                if _b == 0 and _s > 0:
                    continue  # a downsampling block can't be replaced by the identity
                _lesioned = copy.deepcopy(_m)
                _lesioned.layers[_s][_b] = nn.Identity()
                _accs.append(evaluate(_lesioned, _test)["accuracy"])
        _res[_name] = (_base, np.array(_accs))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _name, (_base, _a) in _res.items():
        _ax.plot(np.arange(1, len(_a) + 1), _a, "o-" if _name.startswith("residual") else "s--", label=_name)
        _ax.axhline(_base, color="C0" if _name.startswith("residual") else "C1", lw=0.8, ls=":")
    _ax.axhline(0.1, color="gray", lw=0.8, label="chance")
    _ax.set(xlabel="removed block (non-downsampling, input → output)", ylabel="test accuracy")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| network | intact | mean after removing one block | worst | best |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {k} | {b:.3f} | {a.mean():.3f} | {a.min():.3f} | {a.max():.3f} |"
                                       for k, (b, a) in _res.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in the plain network every layer's output is the only input of the
    next, so what does deleting one do to everything after it? in the residual network,
    unroll $x_{l+1} = x_l + F_l(x_l)$ over several blocks: how many paths from input to
    output are there, and how many of them does one deletion remove? what does this
    suggest about how a deep ResNet actually computes?
    """)
    return


if __name__ == "__main__":
    app.run()
