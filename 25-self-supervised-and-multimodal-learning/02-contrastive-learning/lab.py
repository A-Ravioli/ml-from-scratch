import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.ssl import extract_features, knn_eval, linear_probe  # yours, from topic 01
    from mlfs.torchnn import make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(min(4, torch.get_num_threads()))  # never raise the thread count


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # contrastive learning · lab

    every experiment here runs **your** `info_nce_loss`, `SimCLRAugment` and
    `train_simclr`, and scores encoders with your topic 01 harness. for each experiment:
    write down a prediction, lock it in, run it, then explain what happened. compare with
    `solutions/lab_notes.md` afterwards. the whole lab takes about two and a half minutes on
    a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what does a loss value mean?

    no training yet. we feed your `info_nce_loss` two kinds of synthetic embeddings in
    $d = 128$ dimensions, for several batch sizes $B$ and temperatures $T$:

    - **random**: $z_1$ and $z_2$ independent and uniform on the sphere, an untrained
      encoder's best impression of knowing nothing;
    - **perfect positives**: $z_2 = z_1$ (every pair agrees exactly), with the $z_1$
      uniform on the sphere, so negatives are as spread out as random points can be.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with B = 256 and T = 0.1, what loss do random embeddings give? and perfect positives: is their loss close "
        "to 0 at every temperature?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _g = torch.Generator().manual_seed(0)
    _Bs, _Ts = (16, 64, 256, 1024), (0.05, 0.1, 0.5, 1.0)
    _rand = np.zeros((len(_Bs), len(_Ts)))
    _perf = np.zeros_like(_rand)
    for _i, _B in enumerate(_Bs):
        _z1, _z2 = torch.randn(_B, 128, generator=_g), torch.randn(_B, 128, generator=_g)
        for _j, _T in enumerate(_Ts):
            _rand[_i, _j] = ex.info_nce_loss(_z1, _z2, temperature=_T).item()
            _perf[_i, _j] = ex.info_nce_loss(_z1, _z1.clone(), temperature=_T).item()
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for _j, _T in enumerate(_Ts):
        _axes[0].plot(_Bs, _rand[:, _j], marker="o", label=f"T = {_T}")
        _axes[1].plot(_Bs, _perf[:, _j], marker="o", label=f"T = {_T}")
    _axes[0].plot(_Bs, [math.log(2 * b - 1) for b in _Bs], "k--", lw=1, label="log(2B − 1)")
    _axes[0].set(title="random embeddings", xlabel="batch size B", ylabel="InfoNCE loss", xscale="log")
    _axes[1].set(title="perfect positives (z2 = z1)", xlabel="batch size B", xscale="log")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {b} | {math.log(2 * b - 1):.2f} | " + " | ".join(f"{r:.2f} / {p:.2f}" for r, p in zip(_rand[i], _perf[i]))
                      + " |" for i, b in enumerate(_Bs))
    mo.vstack([_fig, mo.md("| B | log(2B−1) | " + " | ".join(f"T={t}: random / perfect" for t in _Ts)
                           + " |\n|---|---|" + "---|" * len(_Ts) + "\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at $T \ge 0.5$ the random-embedding loss sits on $\log(2B - 1)$: why
    that number? at small temperatures it's *above* it: random cosines in 128 dimensions
    are spread by about $1/\sqrt{128} \approx 0.09$, so what happens to that spread after
    dividing by $T = 0.05$, and why does a spread in the negatives' logits raise the loss?
    why can't perfect positives reach a loss of 0 at $T = 1$, and what does that mean for
    the gradient late in training at a high temperature? a loss of 4 is "nearly random"
    at one setting and "already learning" at another: what must you report alongside an
    InfoNCE loss so someone else can read it?
    """)
    return


@app.class_definition
class SmallCNN(nn.Module):
    """three conv-BN-ReLU-pool blocks and global average pooling: 28x28 grayscale -> 64 features."""

    def __init__(self):
        super().__init__()
        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]

        self.net = nn.Sequential(*block(1, 16), *block(16, 32), *block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


@app.cell
def _():
    _d = datasets.mnist()
    X_ssl = _d["X_train"][:10_000, None].astype(np.float32) / 255  # labels never used for training
    X_lab, y_lab = X_ssl[:5000], _d["y_train"][:5000]  # labelled set for the kNN / probe evaluation
    X_te, y_te = _d["X_test"][:2000, None].astype(np.float32) / 255, _d["y_test"][:2000]
    return X_lab, X_ssl, X_te, y_lab, y_te


@app.function
def score(encoder, X_lab, y_lab, X_te, y_te):
    """(kNN accuracy, linear-probe accuracy) of an encoder, with your topic 01 harness."""
    f, l = extract_features(encoder, make_loader(X_lab, y_lab, batch_size=500, shuffle=False))
    ft, lt = extract_features(encoder, make_loader(X_te, y_te, batch_size=500, shuffle=False))
    return knn_eval(f, l, ft, lt), linear_probe(f, l, ft, lt, max_iters=300)


@app.function
def align_uniform(encoder, head, X, augment):
    """Wang & Isola's alignment (mean squared distance of positive pairs) and uniformity (log mean
    exp(-2 * squared distance) over all pairs) of the normalised embeddings z, on 1,000 images."""
    encoder.eval()
    head.eval()
    with torch.no_grad():
        x = torch.as_tensor(X[:1000])
        z1 = F.normalize(head(encoder(augment(x))), dim=1)
        z2 = F.normalize(head(encoder(augment(x))), dim=1)
        align = (z1 - z2).pow(2).sum(1).mean().item()
        d2 = torch.cdist(z1, z1).pow(2)
        off = ~torch.eye(len(z1), dtype=torch.bool)
        uniform = torch.log(torch.exp(-2 * d2[off]).mean()).item()
    return align, uniform


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the temperature in training

    SimCLR on 10,000 unlabelled MNIST digits: a small CNN (64 features), your
    `train_simclr` for 2 epochs (Adam, lr $10^{-3}$, strength 0.5), at three temperatures
    and 2 seeds each. after training we measure kNN and linear-probe accuracy (5,000
    labelled images, 2,000 test images), and two properties of the normalised embeddings
    $z$ (Wang & Isola, 2020): **alignment**, the mean squared distance between the two
    views of an image (lower = more invariant), and **uniformity**, $\log$ of the mean of
    $e^{-2\|z_i - z_j\|^2}$ over pairs of different images (lower = more spread over the
    sphere). you can change the batch size; the default is 128.
    """)
    return


@app.cell
def _():
    batch = mo.ui.dropdown(options={"32": 32, "128": 128, "512": 512}, value="128", label="batch size")
    batch
    return (batch,)


@app.cell
def _():
    p2 = ui.predict(
        "which temperature (0.05, 0.2 or 1.0) gives the best kNN accuracy, and by how much? which one gives the most "
        "uniform embeddings?"
    )
    p2
    return (p2,)


@app.cell
def _(X_lab, X_ssl, X_te, batch, p2, y_lab, y_te):
    ui.gate(p2)
    _t0 = time.time()
    torch.manual_seed(0)
    _rk, _rp = score(SmallCNN(), X_lab, y_lab, X_te, y_te)
    _rows = [f"| random init (no training) | – | {_rk:.3f} | {_rp:.3f} | – | – |"]
    _aug = ex.SimCLRAugment(28, strength=0.5)
    _curves = {}
    for _T in (0.05, 0.2, 1.0):
        _res = []
        for _seed in range(2):
            torch.manual_seed(_seed)
            _enc = SmallCNN()
            _head, _h = ex.train_simclr(_enc, 64, X_ssl, epochs=2, batch_size=batch.value, temperature=_T, lr=1e-3,
                                        strength=0.5, hidden_dim=128, proj_dim=64, seed=_seed)
            torch.manual_seed(100)
            _res.append((*score(_enc, X_lab, y_lab, X_te, y_te), *align_uniform(_enc, _head, X_ssl, _aug),
                         _h["loss"][-1]))
            _curves.setdefault(_T, _h["loss"])
        _r = np.array(_res)
        _rows.append(f"| T = {_T} | {_r[:, 4].mean():.2f} | " + " | ".join(
            f"{_r[:, k].mean():.3f} ({_r[:, k].min():.3f}–{_r[:, k].max():.3f})" for k in range(4)) + " |")
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    for _T, _c in _curves.items():
        _ax.plot(_c, label=f"T = {_T} (seed 0)")
    _ax.set(xlabel="step", ylabel="InfoNCE loss", title=f"batch size {batch.value}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| encoder | final loss | kNN | linear probe | alignment ↓ | uniformity ↓ |\n|---|---|---|---|---|---|\n"
        + "\n".join(_rows) + f"\n\nmean over 2 seeds (min–max in brackets). ({time.time() - _t0:.0f} s)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the loss curves can't be compared across temperatures (experiment 1
    told you why): what *can*? relate the uniformity column to the temperature, using the
    gradient of the loss with respect to a negative's similarity (on paper, problem 2). is
    the gap between temperatures bigger than the spread between seeds? and before you
    trust any of it: how much better than a *random* network is SimCLR after two epochs,
    and why is a random CNN already that good on MNIST?
    """)
    return


if __name__ == "__main__":
    app.run()
