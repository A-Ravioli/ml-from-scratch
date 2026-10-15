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
    from mlfs.torchnn import MLP, count_parameters, evaluate, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


@app.function
def vgg16(num_classes=1000):
    """the original VGG-16 (configuration D) for 224x224 images: 13 convs in 5 blocks, then 3 fully connected layers."""
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    layers, c_in = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2))
        else:
            layers += [nn.Conv2d(c_in, v, 3, padding=1), nn.ReLU()]
            c_in = v
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Linear(4096, 4096),
                         nn.ReLU(), nn.Linear(4096, num_classes))


@app.function
def train_classifier(model, X, y, *, epochs, lr, optimizer="adam", batch_size=64, seed=0):
    """a plain training loop (Adam or SGD with momentum 0.9) on in-memory arrays; returns the per-epoch training losses."""
    torch.manual_seed(seed)
    opt = (torch.optim.Adam(model.parameters(), lr=lr) if optimizer == "adam"
           else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))
    loader = make_loader(X, y, batch_size=batch_size, seed=seed)
    losses = []
    for _ in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(xb), n + len(xb)
        losses.append(total / n)
    return losses


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # classic architectures · lab

    every experiment here uses **your** `LeNet`, `MiniVGG` and `profile_model` from
    `exercise.py`, with your chapter 15 `MLP`, `make_loader` and `evaluate`, on MNIST and
    CIFAR-10 (offline, synthetic stand-ins with the same shapes). everything runs on the
    CPU. for each experiment: write down a prediction, lock it in, run it, then explain what
    happened. compare your explanations with `solutions/lab_notes.md` afterwards. the
    whole lab takes two to three minutes on an idle laptop.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: where do the parameters live, and where does the time go?

    your `profile_model` on three networks: LeNet-5 on a $28 \times 28$ digit, your
    `MiniVGG` on a $32 \times 32$ CIFAR image, and the original VGG-16 (Simonyan &
    Zisserman, 2015: 13 convolutions and three fully connected layers of 4,096 units) on a
    $224 \times 224$ photo. for each network we split the parameters and the
    multiply–accumulates between convolutional and fully connected layers.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "VGG-16 has 138 million parameters and does about 15 billion multiply-accumulates per image. what fraction of "
        "the parameters sit in its three fully connected layers? and what fraction of the compute?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _nets = {"LeNet-5": (ex.LeNet(), (1, 28, 28)), "MiniVGG (32, 64, 128)": (ex.MiniVGG(), (3, 32, 32)),
             "VGG-16": (vgg16(), (3, 224, 224))}
    _rows, _profiles = [], {}
    for _name, (_m, _shape) in _nets.items():
        _prof = ex.profile_model(_m, _shape)
        _profiles[_name] = _prof
        _conv = [r for r in _prof if len(r["out_shape"]) == 3]
        _fc = [r for r in _prof if len(r["out_shape"]) == 1]
        _p_tot, _m_tot = sum(r["params"] for r in _prof), sum(r["macs"] for r in _prof)
        _rows.append(f"| {_name} | {_p_tot:,} | {sum(r['params'] for r in _fc) / _p_tot:.1%} | {_m_tot / 1e6:,.1f} M | "
                     f"{sum(r['macs'] for r in _conv) / _m_tot:.1%} |")
    _vgg = _profiles["VGG-16"]
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 3.6))
    _x = np.arange(len(_vgg))
    _labels = [("conv" if len(r["out_shape"]) == 3 else "fc") + str(i + 1) for i, r in enumerate(_vgg)]
    _axes[0].bar(_x, [r["params"] / 1e6 for r in _vgg], color=["C0" if len(r["out_shape"]) == 3 else "C3" for r in _vgg])
    _axes[0].set(ylabel="parameters (millions)", title="VGG-16: parameters per layer")
    _axes[1].bar(_x, [r["macs"] / 1e9 for r in _vgg], color=["C0" if len(r["out_shape"]) == 3 else "C3" for r in _vgg])
    _axes[1].set(ylabel="multiply-accumulates (billions)", title="VGG-16: compute per layer")
    for _a in _axes:
        _a.set_xticks(_x, _labels, rotation=90, fontsize=7)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| network | conv + linear parameters | in fully connected layers | MACs per image | in conv layers |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does a convolution with few weights cost so much compute, and a
    fully connected layer with many weights so little (count how many times each weight is
    used per image)? VGG-16's first fully connected layer takes a $7 \times 7 \times 512$
    map: what happens to its size if the input is $448 \times 448$, and what does your
    MiniVGG do instead that makes the question disappear? which of the two budgets,
    parameters or MACs, matters more for training speed, and which for the size of the
    file you ship?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what does a convolutional network assume?

    your `LeNet` (61,706 parameters) against your chapter 15 `MLP` with one hidden layer
    of 77 units (61,225 parameters), each trained for 2 epochs on 20,000 MNIST digits
    (Adam, learning rate $10^{-3}$). then the same two models trained and tested on
    **permuted** digits: the 784 pixels of every image shuffled by one fixed random
    permutation, the same for all images. the permuted images contain exactly the same
    information; a fixed permutation is invertible.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "on the original digits, which model wins and by how much? on the permuted digits, how much does each "
        "model lose?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _d = datasets.mnist()
    _n = min(20000, len(_d["X_train"]))
    _Xtr = ((_d["X_train"][:_n] / 255.0 - 0.13) / 0.31).astype(np.float32)
    _Xte = ((_d["X_test"][:5000] / 255.0 - 0.13) / 0.31).astype(np.float32)
    _ytr, _yte = _d["y_train"][:_n], _d["y_test"][:5000]
    _perm = np.random.default_rng(0).permutation(28 * 28)
    _rows = []
    for _pixels in ("original", "permuted"):
        def _view(X):
            flat = X.reshape(len(X), -1)
            return (flat[:, _perm] if _pixels == "permuted" else flat).reshape(len(X), 1, 28, 28)
        _a, _b = _view(_Xtr), _view(_Xte)
        for _name in ("LeNet", "MLP (77 hidden)"):
            torch.manual_seed(0)
            _m = ex.LeNet() if _name == "LeNet" else nn.Sequential(nn.Flatten(), MLP(784, [77], 10))
            _t0 = time.perf_counter()
            train_classifier(_m, _a, _ytr, epochs=2, lr=1e-3)
            _acc = evaluate(_m, make_loader(_b, _yte, batch_size=1000, shuffle=False))["accuracy"]
            _rows.append(f"| {_pixels} | {_name} | {count_parameters(_m):,} | **{_acc:.4f}** | "
                         f"{time.perf_counter() - _t0:.0f} s |")
    _fig, _axes = plt.subplots(1, 2, figsize=(5, 2.6))
    _axes[0].imshow(_Xte[0], cmap="gray")
    _axes[1].imshow(_Xte[0].reshape(-1)[_perm].reshape(28, 28), cmap="gray")
    for _ax, _t in zip(_axes, ("original", "permuted")):
        _ax.set(title=_t, xticks=[], yticks=[])
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| pixels | model | parameters | test accuracy | time |\n|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the MLP's accuracy (almost) unchanged by the permutation? think
    about what permuting the inputs does to its first weight matrix. what does LeNet assume
    about pixels that the permutation breaks, and why does that assumption help so much on
    the original digits? what would the result be for a convolutional network on data
    with no spatial structure at all, say a table of patient records?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: depth vs accuracy on CIFAR-10

    your `MiniVGG` with 1, 2 or 4 blocks (widths $8$, $8\text{–}16$,
    $8\text{–}16\text{–}32\text{–}64$: 2, 4 and 8 convolutional layers), with and without
    batch norm, each trained for 2 epochs on 6,000 CIFAR-10 images with SGD (learning rate
    0.05, momentum 0.9, batch 64) and PyTorch's default initialization, and tested on
    2,000.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with batch norm, does accuracy keep rising from 2 to 8 conv layers in this short budget? and without batch "
        "norm, does the deepest network do better or worse than the shallowest?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _d = datasets.cifar10()
    _n = min(6000, len(_d["X_train"]))
    _mean = _d["X_train"][:_n].reshape(-1, 3).mean(0) / 255
    _std = _d["X_train"][:_n].reshape(-1, 3).std(0) / 255
    _prep = lambda X: ((X / 255.0 - _mean) / _std).transpose(0, 3, 1, 2).astype(np.float32)  # noqa: E731
    _Xtr, _ytr = _prep(_d["X_train"][:_n]), _d["y_train"][:_n]
    _Xte, _yte = _prep(_d["X_test"][:2000]), _d["y_test"][:2000]
    _widths = [(8,), (8, 16), (8, 16, 32, 64)]
    _acc = {True: [], False: []}
    _rows = []
    for _w in _widths:
        _cells = []
        for _bn in (True, False):
            torch.manual_seed(0)
            _m = ex.MiniVGG(widths=_w, batch_norm=_bn)
            _t0 = time.perf_counter()
            _losses = train_classifier(_m, _Xtr, _ytr, epochs=2, lr=0.05, optimizer="sgd")
            _a = evaluate(_m, make_loader(_Xte, _yte, batch_size=500, shuffle=False))["accuracy"]
            _acc[_bn].append(_a)
            _cells.append(f"{_a:.3f} (loss {_losses[-1]:.2f}, {time.perf_counter() - _t0:.0f} s)")
        _rows.append(f"| {2 * len(_w)} | {_w} | " + " | ".join(_cells) + " |")
    _fig, _ax = plt.subplots(figsize=(5.5, 3.4))
    _depth = [2 * len(_w) for _w in _widths]
    _ax.plot(_depth, _acc[True], "o-", label="with batch norm")
    _ax.plot(_depth, _acc[False], "s-", label="without batch norm")
    _ax.axhline(0.1, color="gray", ls=":", label="chance")
    _ax.set(xlabel="conv layers", ylabel="test accuracy", xticks=_depth)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| conv layers | widths | with batch norm: accuracy (final train loss, time) | "
                           "without batch norm |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with batch norm, what does each extra block buy (receptive field, how
    much of the image the final features summarize, capacity)? without it, look at the
    final training losses: is the deep network failing to *generalize*, or failing to
    *train*? connect it to chapter 15's initialization lab: what happens to the signal's
    scale through 8 layers with PyTorch's default init? the original VGG authors trained
    their deepest nets by initializing them from shallower ones. what would you do today?
    """)
    return


if __name__ == "__main__":
    app.run()
