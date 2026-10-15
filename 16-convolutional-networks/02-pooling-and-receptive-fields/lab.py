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
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _n = min(12000, len(_d["X_train"]))
    X_train = ((_d["X_train"][:_n, None] / 255.0 - 0.13) / 0.31).astype(np.float32)
    y_train = _d["y_train"][:_n]
    X_test = ((_d["X_test"][:5000, None] / 255.0 - 0.13) / 0.31).astype(np.float32)
    y_test = _d["y_test"][:5000]
    BACKGROUND = float((0 - 0.13) / 0.31)  # the value of a blank pixel after standardizing


@app.function
def conv_stack(depth, *, width=16, activation="relu", seed=0):
    """depth 3x3 'same' convolutions (1 -> width -> ... -> width), kaiming-normal init, no bias."""
    torch.manual_seed(seed)
    layers, c_in = [], 1
    for _ in range(depth):
        conv = nn.Conv2d(c_in, width, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu" if activation == "relu" else "linear")
        layers.append(conv)
        if activation == "relu":
            layers.append(nn.ReLU())
        c_in = width
    return nn.Sequential(*layers)


@app.function
def small_cnn(downsample, seed=0):
    """four 3x3 conv layers with a downsampling step ('max', 'avg' or 'stride') after each of the first three, global average pooling, linear."""
    torch.manual_seed(seed)

    def down(c):
        if downsample == "max":
            return [nn.MaxPool2d(2)]
        if downsample == "avg":
            return [nn.AvgPool2d(2)]
        return [nn.Conv2d(c, c, 3, stride=2, padding=1), nn.ReLU()]

    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), *down(16),
                         nn.Conv2d(16, 64, 3, padding=1), nn.ReLU(), *down(64),
                         nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), *down(64),
                         nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


@app.function
def shift_images(x, dy, dx, fill):
    """translate a batch (N, C, H, W) by (dy, dx) pixels, filling the uncovered border with `fill`."""
    out = np.full_like(x, fill)
    h, w = x.shape[-2:]
    out[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = x[..., max(-dy, 0):h + min(-dy, 0),
                                                                      max(-dx, 0):w + min(-dx, 0)]
    return out


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # pooling & receptive fields · lab

    every experiment here uses **your** `receptive_field`, `rf_layers`,
    `effective_receptive_field`, `erf_radius` and `max_pool2d` from `exercise.py`, with
    small PyTorch networks (you may use `nn.Conv2d` and the pooling layers now: you've built
    them) on MNIST digits (offline, a synthetic stand-in). for each experiment: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes.
    """)
    return


@app.cell
def _():
    _x = np.random.default_rng(0).normal(size=(2, 3, 9, 9))
    _ok = np.allclose(ex.max_pool2d(_x, 3, 2), F.max_pool2d(torch.tensor(_x), 3, 2).numpy())
    mo.md(f"sanity check: your `max_pool2d` matches `torch.nn.functional.max_pool2d`: **{_ok}**")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how far does a deep network actually look?

    a stack of $L$ randomly initialized $3 \times 3$ convolutions (16 channels, kaiming
    init) on a $129 \times 129$ input. its theoretical receptive field, from your
    `receptive_field(rf_layers(model))`, is $2L + 1$ pixels wide: a radius of $L$. the
    **effective** receptive field is your gradient map; we report the radius of the square
    that holds 90% of it (`erf_radius(erf, 0.9)`). switch the activation to see whether the
    ReLUs matter.
    """)
    return


@app.cell
def _():
    activation = mo.ui.dropdown(options=["relu", "linear"], value="relu", label="activation between convs")
    p1 = ui.predict(
        "a 32-layer stack of 3x3 convs has a theoretical receptive field of 65x65 (radius 32). what radius holds 90% "
        "of its effective receptive field: about 30? 15? 8?"
    )
    mo.vstack([p1, activation])
    return activation, p1


@app.cell
def _(activation, p1):
    ui.gate(p1)
    _depths = [2, 4, 8, 16, 32]
    _rows, _radii, _maps = [], [], {}
    for _L in _depths:
        _m = conv_stack(_L, activation=activation.value)
        _rf, _jump = ex.receptive_field(ex.rf_layers(_m))
        _erf = ex.effective_receptive_field(_m, 129, n_samples=8)
        _r90 = ex.erf_radius(_erf, 0.9)
        _radii.append(_r90)
        _maps[_L] = _erf
        _rows.append(f"| {_L} | {_rf} | {(_rf - 1) // 2} | {_r90} | {_r90 / ((_rf - 1) // 2):.2f} |")
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _axes[0].plot(_depths, [_L for _L in _depths], "o-", label="theoretical radius L")
    _axes[0].plot(_depths, _radii, "s-", label="90% ERF radius")
    _axes[0].plot(_depths, _radii[0] * np.sqrt(np.array(_depths) / _depths[0]), "k:", label="∝ √L")
    _axes[0].set(xlabel="depth L", ylabel="radius (pixels)", xscale="log", yscale="log")
    _axes[0].set_xticks(_depths, [str(_L) for _L in _depths])
    _axes[0].set_yticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    _axes[0].minorticks_off()
    _axes[0].legend(fontsize=8)
    for _ax, _L in zip(_axes[1:], (4, 32)):
        _c = 64
        _ax.imshow(_maps[_L][_c - 40:_c + 41, _c - 40:_c + 41], cmap="magma")
        _ax.add_patch(plt.Rectangle((40 - _L - 0.5, 40 - _L - 0.5), 2 * _L + 1, 2 * _L + 1, fill=False, ec="c",
                                    lw=1.2, ls="--"))
        _ax.set(title=f"ERF, L = {_L} (dashed: theoretical field)", xticks=[], yticks=[])
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| depth L | theoretical rf | theoretical radius | 90% ERF radius | ratio |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** count the paths from an input pixel to the centre output through $L$
    layers of $3 \times 3$ kernels: why are there far more paths through the middle of the
    field than through its edge? what familiar distribution does the number of paths
    approach, and how does its width grow with $L$? what does that mean for a network that
    needs to relate two objects 50 pixels apart?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: is a CNN translation-invariant?

    three small CNNs (four $3 \times 3$ conv layers with a downsampling step after each of
    the first three, then global average pooling and a linear layer), identical except for
    how they downsample: **max** pooling, **average** pooling, or a **stride-2**
    convolution. each is trained for one epoch on 12,000 digits (Adam, learning rate
    $10^{-3}$, batch 32). then we shift every test digit by $s$ pixels
    (diagonally, filling with background) and measure the accuracy, and how many
    predictions change when a digit moves by a single pixel.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "convolutions are translation-equivariant and global average pooling is translation-invariant. if you move a "
        "digit by 1 pixel, what fraction of the predictions change? and by 4 pixels, how much accuracy is lost?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    torch.manual_seed(0)
    _loader = make_loader(X_train, y_train, batch_size=32, seed=0)
    _shifts = [0, 1, 2, 3, 4, 6]
    _acc, _flip, _rows = {}, {}, []
    for _kind in ("max", "avg", "stride"):
        _m = small_cnn(_kind)
        _opt = torch.optim.Adam(_m.parameters(), lr=1e-3)
        _t0 = time.perf_counter()
        _m.train()
        for _xb, _yb in _loader:
            _opt.zero_grad()
            F.cross_entropy(_m(_xb), _yb).backward()
            _opt.step()
        _m.eval()
        _base_pred = None
        _acc[_kind] = []
        for _s in _shifts:
            _xs = shift_images(X_test, _s, _s, BACKGROUND)
            with torch.no_grad():
                _pred = _m(torch.tensor(_xs)).argmax(1).numpy()
            _acc[_kind].append(float(np.mean(_pred == y_test)))
            if _s == 0:
                _base_pred = _pred
            if _s == 1:
                _flip[_kind] = float(np.mean(_pred != _base_pred))
        _rows.append(f"| {_kind} | " + " | ".join(f"{a:.3f}" for a in _acc[_kind])
                     + f" | {100 * _flip[_kind]:.1f}% | {time.perf_counter() - _t0:.0f} s |")
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    for _kind, _a in _acc.items():
        _ax.plot(_shifts, _a, "o-", label=f"{_kind} downsampling")
    _ax.set(xlabel="diagonal shift (pixels)", ylabel="test accuracy")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| downsampling | " + " | ".join(f"shift {s}" for s in _shifts)
                           + " | predictions changed by a 1-px shift | time |\n|---|" + "---|" * (len(_shifts) + 2)
                           + "\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the convolutions are equivariant and the final pooling is invariant,
    so where does the sensitivity to small shifts come from (think about what a stride-2
    step does to a signal shifted by one pixel)? why does the accuracy fall off for larger
    shifts too (what does the network see near the image border, and what did training
    show it)? which downsampling is most robust, and why might that be?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: three ways to see farther

    three 8-layer stacks of $3 \times 3$ convolutions (16 channels, ReLU, kaiming init),
    differing only in how they grow the receptive field: **plain** (stride 1 everywhere),
    **strided** (stride 2 in layers 1 and 3, like a network that downsamples early), and
    **dilated** (dilations $1, 2, 4, 8, 1, 2, 4, 8$ with matching padding). for each, the
    theoretical receptive field from your `receptive_field(rf_layers(model))` and the
    effective one from your `effective_receptive_field` on a $129 \times 129$ input.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the strided and dilated stacks have theoretical fields 3-4x wider than the plain one's. which of the three "
        "uses the largest FRACTION of its theoretical field? (in experiment 1, a 32-layer stack used 41%.)"
    )
    p3
    return (p3,)


@app.function
def growth_stack(kind, *, width=16, seed=0):
    """8 conv layers: 'plain', 'strided' (stride 2 in layers 1 and 3) or 'dilated' (1, 2, 4, 8, 1, 2, 4, 8)."""
    torch.manual_seed(seed)
    layers, c_in = [], 1
    for i in range(8):
        stride = 2 if kind == "strided" and i in (0, 2) else 1
        dil = (1, 2, 4, 8)[i % 4] if kind == "dilated" else 1
        conv = nn.Conv2d(c_in, width, 3, stride=stride, padding=dil, dilation=dil, bias=False)
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        layers += [conv, nn.ReLU()]
        c_in = width
    return nn.Sequential(*layers)


@app.cell
def _(p3):
    ui.gate(p3)
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.8))
    _rows = []
    for _ax, _kind in zip(_axes, ("plain", "strided", "dilated")):
        _m = growth_stack(_kind)
        _rf, _jump = ex.receptive_field(ex.rf_layers(_m))
        _erf = ex.effective_receptive_field(_m, 129, n_samples=8)
        _r50, _r90 = ex.erf_radius(_erf, 0.5), ex.erf_radius(_erf, 0.9)
        _rows.append(f"| {_kind} | {_rf} | {_jump} | {(_rf - 1) // 2} | {_r50} | {_r90} | {_r90 / ((_rf - 1) // 2):.2f} |")
        _ax.imshow(np.sqrt(_erf[64 - 40:64 + 41, 64 - 40:64 + 41]), cmap="magma")
        _h = (_rf - 1) // 2
        _ax.add_patch(plt.Rectangle((40 - _h - 0.5, 40 - _h - 0.5), 2 * _h + 1, 2 * _h + 1, fill=False, ec="c",
                                    lw=1.2, ls="--"))
        _ax.set(title=f"{_kind}: rf {_rf}, 90% ERF radius {_r90}", xticks=[], yticks=[])
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("(square root of the ERF shown, so faint regions are visible; dashed: the theoretical "
                           "field, clipped to the 81×81 crop)\n\n"
                           "| stack | theoretical rf | jump | theoretical radius | 50% ERF radius | 90% ERF radius | "
                           "90% radius / theoretical |\n|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** all three stacks use about the same fraction of their theoretical
    field, while experiment 1's deeper stacks used less and less. so what sets that
    fraction: how the field is grown, or the number of layers the paths pass through? why
    do strides grow the field so cheaply, and what do they cost (look at the jump: how
    many output positions are there)? the dilated stack mixes rates $1, 2, 4, 8$: what
    would happen to the pixels between the taps if every layer used dilation 8?
    """)
    return


if __name__ == "__main__":
    app.run()
