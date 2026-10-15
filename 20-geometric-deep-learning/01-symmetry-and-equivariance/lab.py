import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    torch.set_num_threads(4)
    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    N_TRAIN = min(6000, len(_d["X_train"]))
    N_TEST = min(2000, len(_d["X_test"]))
    # pixels standardized; the border stays at one constant value, so a circular shift is an ordinary shift
    X_TRAIN = torch.tensor((_d["X_train"][:N_TRAIN, None] / 255.0 - 0.13) / 0.31, dtype=torch.float32)
    Y_TRAIN = torch.tensor(_d["y_train"][:N_TRAIN])
    X_TEST = torch.tensor((_d["X_test"][:N_TEST, None] / 255.0 - 0.13) / 0.31, dtype=torch.float32)
    Y_TEST = torch.tensor(_d["y_test"][:N_TEST])
    OFFLINE = datasets.is_offline()


@app.class_definition
class CircularCNN(torch.nn.Module):
    """four circular 3x3 convolutions with ReLU (16, 32, 32, 64 channels; the middle two with `stride`), a global
    max pool and a linear head. blur=True low-pass filters (a 3x3 binomial blur) before each subsampling (Zhang, 2019).
    """

    def __init__(self, stride=2, n_out=10, blur=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.ws, self.bs, n_in = torch.nn.ParameterList(), torch.nn.ParameterList(), 1
        for n in (16, 32, 32, 64):
            self.ws.append(torch.nn.Parameter(torch.randn(n, n_in, 3, 3, generator=g) * (2 / (9 * n_in)) ** 0.5))
            self.bs.append(torch.nn.Parameter(torch.zeros(n)))
            n_in = n
        torch.manual_seed(seed)
        self.head = torch.nn.Linear(n_in, n_out)
        self.strides, self.blur = (1, stride, stride, 1), blur
        k = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("kernel", k[:, None] * k[None, :] / 16)

    def forward(self, x):
        h = x
        for w, b, s in zip(self.ws, self.bs, self.strides):
            if self.blur and s > 1:
                h = F.relu(ex.circular_conv2d(h, w, b))
                kern = self.kernel.to(h.dtype).expand(h.shape[1], 1, 3, 3)
                h = F.conv2d(F.pad(h, (1, 1, 1, 1), mode="circular"), kern, groups=h.shape[1])[..., ::s, ::s]
            else:
                h = F.relu(ex.circular_conv2d(h, w, b, stride=s))
        return self.head(h.amax(dim=(-2, -1)))


@app.function
def train(model, X, y, *, epochs=2, lr=3e-3, batch_size=128, seed=0, augment=None):
    """Adam on cross-entropy; `augment(xb) -> xb` is applied to each minibatch if given."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx] if augment is None else augment(X[idx])
            loss = F.cross_entropy(model(xb), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # symmetry & equivariance · lab

    every experiment here uses **your** `translate`, `circular_conv2d`, `equivariance_error`,
    `c4_rotations` and `group_average`. the networks are small CNNs built from your circular
    convolution, trained on {N_TRAIN:,} MNIST digits {"(offline: a synthetic stand-in with the same shapes)" if OFFLINE else ""}.
    for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards. the whole lab takes about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: is a CNN translation-invariant?

    a randomly initialized CNN: four circular 3×3 convolutions with ReLU, the middle two with
    stride $s$, then a global max pool and a linear layer. "convolutions are equivariant and
    pooling is invariant", so the output should not care where the digit sits. we shift 200
    digits by 0 to 8 pixels and measure the largest change in the output, relative to the
    output's typical size.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with stride s = 2 in two of the layers, how big is the relative change in the output for a 1-pixel "
        "shift? for a 4-pixel shift? (0 = perfectly invariant)"
    )
    p1
    return (p1,)


@app.cell
def _():
    stride_pick = mo.ui.dropdown({"1": 1, "2": 2, "3": 3}, value="2", label="stride of the two strided layers")
    stride_pick
    return (stride_pick,)


@app.cell
def _(CircularCNN, p1, stride_pick):
    ui.gate(p1)
    _x = X_TEST[:200].double()
    _shifts = list(range(9))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.5))
    _rows = []
    for _s in sorted({1, stride_pick.value, 2}):
        _net = CircularCNN(stride=_s).double().eval()
        _scale = float(_net(_x).detach().abs().mean())
        _errs = [ex.equivariance_error(_net, _x, lambda t, k=k: ex.translate(t, (0, k)), None) / _scale
                 for k in _shifts]
        _ax.plot(_shifts, _errs, "o-", label=f"stride {_s} (total {_s * _s})")
        _rows.append(f"| {_s} | " + " | ".join(f"{e:.2g}" for e in _errs) + " |")
    _ax.set(xlabel="horizontal shift (pixels)", ylabel="max |f(shifted) - f| / mean |f|",
            title="shift invariance of a random circular CNN")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| stride | " + " | ".join(f"{k} px" for k in _shifts) + " |\n|---|" + "---|" * 9 + "\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which shifts leave the output exactly unchanged, and why exactly those? what
    is a stride-2 layer doing to a 1-pixel shift of its input? the convolution and the pooling
    are both "symmetric": which step breaks the symmetry?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does it matter once the network is trained?

    the same architecture (stride 2), now trained for 2 epochs. we shift every test digit by 1
    to 4 pixels and count how often the **predicted class** changes. a second network blurs
    before every subsampling step (Zhang, 2019); a third is the stride-2 network trained with
    random shifts as data augmentation.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the trained stride-2 CNN is about 90% accurate. what fraction of test digits change their predicted class when "
        "shifted by 1 pixel? by 4 pixels? does blurring before subsampling fix it?"
    )
    p2
    return (p2,)


@app.cell
def _(CircularCNN, p2, train):
    ui.gate(p2)
    _g = torch.Generator().manual_seed(1)

    def _augment(xb):
        _k = torch.randint(-3, 4, (2,), generator=_g)
        return ex.translate(xb, (int(_k[0]), int(_k[1])))

    _models = {
        "stride 2": train(CircularCNN(stride=2), X_TRAIN, Y_TRAIN),
        "stride 2 + blur": train(CircularCNN(stride=2, blur=True), X_TRAIN, Y_TRAIN),
        "stride 2 + shift augmentation": train(CircularCNN(stride=2), X_TRAIN, Y_TRAIN, augment=_augment),
    }
    _rows = []
    with torch.no_grad():
        for _name, _m in _models.items():
            _p0 = _m(X_TEST).argmax(1)
            _acc = float((_p0 == Y_TEST).float().mean())
            _flips = [float((_m(ex.translate(X_TEST, (0, k))).argmax(1) != _p0).float().mean()) for k in (1, 2, 3, 4)]
            _rows.append(f"| {_name} | {_acc:.3f} | " + " | ".join(f"{f:.1%}" for f in _flips) + " |")
    exp2_models = _models
    mo.md("| model | test accuracy | flips at 1 px | 2 px | 3 px | 4 px |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return (exp2_models,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does a 4-pixel shift never change a prediction, while 1 and 3 pixels do?
    does blurring make the network exactly invariant, or only less sensitive? what does
    augmentation buy compared with blurring, and what does neither of them give you that a
    stride-1 network has by construction?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: invariance by averaging, and what it costs

    `group_average` makes any network exactly invariant to quarter turns. we train the stride-2
    CNN *inside* a C4 average (every forward pass averages the logits over the four rotations),
    then test it on upright digits, on digits rotated by 90°, and on digits rotated by 45°
    (bilinear interpolation, which is not in C4).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the C4-averaged network is exactly invariant to 90° turns. will its accuracy on upright digits be higher, "
        "the same, or lower than the plain network's? and at 45°?"
    )
    p3
    return (p3,)


@app.cell
def _(CircularCNN, exp2_models, p3, train):
    ui.gate(p3)
    _rots = ex.c4_rotations()

    class _C4(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = CircularCNN(stride=2, seed=0)

        def forward(self, x):
            return ex.group_average(self.net, x, _rots)

    _c4 = train(_C4(), X_TRAIN, Y_TRAIN)

    def _rotate(x, deg):
        _t = torch.tensor(np.deg2rad(deg), dtype=torch.float32)
        _A = torch.stack([torch.stack([torch.cos(_t), -torch.sin(_t), torch.zeros(())]),
                          torch.stack([torch.sin(_t), torch.cos(_t), torch.zeros(())])])
        _grid = F.affine_grid(_A.expand(len(x), 2, 3), list(x.shape), align_corners=False)
        return F.grid_sample(x, _grid, align_corners=False)

    _rows, _conf = [], {}
    with torch.no_grad():
        for _name, _m in {"plain (stride 2)": exp2_models["stride 2"], "C4-averaged": _c4}.items():
            _accs = []
            for _x in (X_TEST, _rots[1](X_TEST), _rotate(X_TEST, 45)):
                _accs.append(float((_m(_x).argmax(1) == Y_TEST).float().mean()))
            _e90 = ex.equivariance_error(_m, X_TEST[:200], _rots[1], None)
            _rows.append(f"| {_name} | " + " | ".join(f"{a:.3f}" for a in _accs) + f" | {_e90:.1e} |")
            _pred = _m(X_TEST).argmax(1)
            _cm = np.zeros((10, 10), dtype=int)
            np.add.at(_cm, (Y_TEST.numpy(), _pred.numpy()), 1)
            _conf[_name] = _cm
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for _ax, (_name, _cm) in zip(_axes, _conf.items()):
        _norm = _cm / np.maximum(_cm.sum(1, keepdims=True), 1)
        np.fill_diagonal(_norm, 0)
        _im = _ax.imshow(_norm, cmap="magma", vmin=0, vmax=max(0.05, _norm.max()))
        _ax.set(title=f"{_name}: errors", xlabel="predicted", ylabel="true", xticks=range(10), yticks=range(10))
        _fig.colorbar(_im, ax=_ax, fraction=0.046)
    _fig.tight_layout()
    mo.vstack([mo.md("| model | upright | rotated 90° | rotated 45° | max output change at 90° |\n|---|---|---|---|---|\n"
                     + "\n".join(_rows)), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the averaged network can't tell a digit from its rotated copy. which pairs of
    digits does that hurt, and does the confusion matrix agree? why is the 45° accuracy not
    rescued by the C4 average? when is building in a symmetry the right call, and when is the
    symmetry not actually a symmetry of the task?
    """)
    return


if __name__ == "__main__":
    app.run()
