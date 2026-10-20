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
    from mlfs.generative import bits_per_dim  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()
    torch.set_num_threads(min(4, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)
    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline

    def _binarize(X, seed):
        """14 x 14 binarized digits: average 2 x 2 blocks, then draw each pixel ~ Bernoulli(intensity)."""
        x = torch.tensor(X / 255.0, dtype=torch.float32)[:, None]
        x = F.avg_pool2d(x, 2)
        return torch.bernoulli(x, generator=torch.Generator().manual_seed(seed))

    N_TRAIN = min(10_000, len(_d["X_train"]))
    B_train = _binarize(_d["X_train"][:N_TRAIN], 0)
    B_test = _binarize(_d["X_test"][:2000], 1)
    D = B_train[0].numel()  # 196


@app.function
def train_made(x_train, order, *, epochs=8, hidden=500, seed=0):
    """your MADE on binary vectors whose dimensions are permuted by `order` (a permutation of 0..D-1)."""
    torch.manual_seed(seed)
    model = ex.MADE(x_train.shape[1], [hidden], seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    xp = x_train[:, order]
    for _ in range(epochs):
        perm = torch.randperm(len(xp), generator=g)
        for i in range(0, len(xp), 100):
            opt.zero_grad()
            ex.made_nll(model, xp[perm[i:i + 100]]).mean().backward()
            opt.step()
    return model


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # autoregressive models · lab

    every experiment uses **your** `MADE`, `made_nll`, `PixelCNN` and `pixelcnn_nll`. data:
    MNIST digits shrunk to $14 \times 14$ and **binarized** (each pixel drawn as a coin flip
    with its intensity as the probability, with a fixed seed), {N_TRAIN:,} for training and
    2,000 for testing. {'**offline: synthetic stand-in data, so the numbers say nothing about MNIST.**' if OFFLINE else ''}
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: scoring is one pass, sampling is many

    a small PixelCNN (32 channels, 5 layers, 2 levels) is trained for two epochs on the
    binarized digits, then timed: one forward pass (which scores the likelihood of a batch)
    and one full sample, at image sizes from $7 \times 7$ to $28 \times 28$ (the network
    is convolutional, so it runs at any size).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "sampling one 14 x 14 image takes t seconds. how long does a 28 x 28 image take (2t, 4t, 16t, ...)? and how "
        "many times longer than scoring one 28 x 28 image is sampling one?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    torch.manual_seed(0)
    pcnn = ex.PixelCNN(1, n_levels=2, hidden_channels=32, n_layers=5)
    _opt = torch.optim.Adam(pcnn.parameters(), lr=2e-3)
    _g = torch.Generator().manual_seed(0)
    _t0 = time.perf_counter()
    for _ in range(2):
        _perm = torch.randperm(len(B_train), generator=_g)
        for _i in range(0, len(B_train), 64):
            _opt.zero_grad()
            ex.pixelcnn_nll(pcnn, B_train[_perm[_i:_i + 64]]).mean().backward()
            _opt.step()
    _train_s = time.perf_counter() - _t0
    with torch.no_grad():
        pcnn_nll = float(ex.pixelcnn_nll(pcnn, B_test).mean())
    _rows = []
    for _s in (7, 14, 21, 28):
        _x = torch.zeros(1, 1, _s, _s)
        with torch.no_grad():
            _t0 = time.perf_counter()
            for _ in range(5):
                pcnn(_x)
            _fwd = (time.perf_counter() - _t0) / 5
        _t0 = time.perf_counter()
        pcnn.sample(1, _s, generator=torch.Generator().manual_seed(1))
        _smp = time.perf_counter() - _t0
        _rows.append((_s, _fwd, _smp))
    _samples = pcnn.sample(16, 14, generator=torch.Generator().manual_seed(2))
    _fig, _axes = plt.subplots(2, 8, figsize=(8, 2.2))
    for _a, _im in zip(_axes.flat, _samples):
        _a.imshow(_im[0], cmap="gray_r", vmin=0, vmax=1)
        _a.axis("off")
    _fig.suptitle("samples from your PixelCNN (14 x 14, binary)", fontsize=9)
    _fig.tight_layout()
    _base = _rows[1][2]
    _table = "\n".join(f"| {s} × {s} | {s * s} | {f * 1e3:.2f} ms | {t:.2f} s | {t / _base:.1f} | {t / f:.0f} |"
                       for s, f, t in _rows)
    mo.vstack([_fig, mo.md(
        f"trained in {_train_s:.0f} s; test NLL **{pcnn_nll:.1f} nats** per image "
        f"(**{bits_per_dim(pcnn_nll, D):.3f} bits/dim**, exact)\n\n"
        "| image | pixels | one forward pass (score) | one sample | sample time / 14×14's | sample / score |\n"
        "|---|---|---|---|---|---|\n" + _table)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how many forward passes does one sample take, and how does the cost of
    each pass grow with the image size? put the two together and predict the exponent:
    does the measured scaling match it at the larger sizes (and what makes the small sizes
    deviate)? scoring needs one pass at any size: why can't sampling be parallelized the
    same way?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does the order matter?

    the chain rule holds for any ordering of the 196 pixels, so the same model family can
    factorize the digits in raster order, in reversed order, or in a random order. your
    MADE (one hidden layer of 500 units, 8 epochs of Adam) is trained on each, with two
    seeds, and evaluated by exact test NLL.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "raster order, reversed order, random order: will the three reach about the same test NLL? if not, which is "
        "worst, and by how much (in nats per image)?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _orders = {
        "raster (row by row)": torch.arange(D),
        "reversed raster": torch.arange(D).flip(0),
        "random permutation": torch.randperm(D, generator=torch.Generator().manual_seed(0)),
    }
    _x_tr, _x_te = B_train.flatten(1), B_test.flatten(1)
    _rows = []
    _t0 = time.perf_counter()
    for _name, _order in _orders.items():
        _vals = []
        for _seed in (0, 1):
            _m = train_made(_x_tr, _order, seed=_seed)
            with torch.no_grad():
                _vals.append(float(ex.made_nll(_m, _x_te[:, _order]).mean()))
        _rows.append(f"| {_name} | {np.mean(_vals):.1f} | {abs(_vals[0] - _vals[1]) / 2:.1f} | "
                     f"{bits_per_dim(np.mean(_vals), D):.3f} |")
    mo.md(f"test NLL of your MADE (mean of 2 seeds; ± is half their difference), {time.perf_counter() - _t0:.0f} s\n\n"
          "| order | test NLL (nats / image) | ± | bits/dim |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** all three orders define the same set of possible distributions, so the
    differences come from *learning*. which order won, and is the gap bigger than the
    seed noise? in raster order, what does a pixel in the middle of a digit get to
    condition on, and what stays hidden from it (everything below)? in a random order,
    what does a pixel late in the order know about its neighbourhood, and what does an
    early one know? a MADE is fully connected, with no built-in notion of which pixels
    are neighbours: which kind of conditional is easier for it to learn?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the blind spot

    the gradient of the logits at the centre pixel of a $15 \times 15$ image with respect
    to every input pixel, for a PixelCNN with $3 \times 3$ type-B layers after the type-A
    first layer. the network is untrained: which pixels *can* influence the centre is a
    property of the masks alone. the slider sets the number of layers.
    """)
    return


@app.cell
def _():
    n_layers = mo.ui.slider(1, 8, value=6, label="PixelCNN layers")
    n_layers
    return (n_layers,)


@app.cell
def _():
    p3 = ui.predict(
        "the centre pixel may use every pixel above it and to its left. with 6 layers, does it actually see all of "
        "the rows above? which pixels, if any, are missing?"
    )
    p3
    return (p3,)


@app.cell
def _(n_layers, p3):
    ui.gate(p3)
    torch.manual_seed(0)
    _m = ex.PixelCNN(1, n_levels=2, hidden_channels=16, n_layers=n_layers.value)
    _first = [mod for mod in _m.modules() if isinstance(mod, ex.MaskedConv2d)][0]
    _S, _c = 15, 7
    _seen = torch.zeros(_S, _S)
    for _rep in range(4):  # several random inputs, so a ReLU that happens to be off doesn't hide a connection
        _x = torch.rand(1, 1, _S, _S, requires_grad=True)
        _m(_x)[0, :, :, _c, _c].sum().backward()
        _seen += _x.grad[0, 0].abs()
    _vis = (_seen > 0).float()
    _allowed = torch.zeros(_S, _S)
    _allowed.view(-1)[: _c * _S + _c] = 1
    _img = _vis * 2 + (_allowed - _vis).clamp(min=0)  # 2 = seen, 1 = allowed but blind, 0 = future
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _ax.imshow(_img, cmap="viridis", vmin=0, vmax=2)
    _ax.plot(_c, _c, "rx", ms=10)
    _ax.set_title(f"{n_layers.value} layers, first layer {_first.kernel_size[0]}×{_first.kernel_size[0]}: "
                  "yellow = seen, teal = allowed but blind", fontsize=8)
    _ax.set_xticks([])
    _ax.set_yticks([])
    _fig.tight_layout()
    _blind = int(((_allowed - _vis) > 0).sum())
    mo.hstack([_fig, mo.md(f"pixels the centre may use: **{int(_allowed.sum())}**  \n"
                           f"pixels it actually sees: **{int(_vis.sum())}**  \n"
                           f"blind (allowed, unseen): **{_blind}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** trace how a type-B $3 \times 3$ layer moves information: by one pixel
    up, left or diagonally up-left, and up-right only by moving a row up at the same time.
    why can't any stack of these layers reach the pixels up and to the right beyond the
    diagonal? which digits have strokes that run through the blind spot? (the reference's
    $7 \times 7$ first layer hides part of the problem near the centre: where?)
    """)
    return


if __name__ == "__main__":
    app.run()
