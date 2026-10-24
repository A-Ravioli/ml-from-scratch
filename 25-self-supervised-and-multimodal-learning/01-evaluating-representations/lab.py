import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(min(4, torch.get_num_threads()))  # never raise the thread count


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # evaluating representations · lab

    every experiment here runs **your** `extract_features`, `linear_probe`, `knn_eval` and
    `fine_tune`. the setting is a small **transfer** problem built from MNIST: a CNN is
    trained (with labels) on the digits 0–4 only, and we ask how good its features are for
    a task it has never seen, telling apart the digits 5–9. for each experiment: write
    down a prediction, lock it in, run it, then explain what happened. compare with
    `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes on a laptop
    CPU.
    """)
    return


@app.class_definition
class SmallCNN(nn.Module):
    """three conv-BN-ReLU blocks and global average pooling: 28x28 images -> 64 features."""

    def __init__(self, n_classes=5):
        super().__init__()
        def block(c_in, c_out, pool):
            layers = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]
            return layers + [nn.MaxPool2d(2)] if pool else layers

        self.features = nn.Sequential(*block(1, 16, True), *block(16, 32, True), *block(32, 64, False),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc(self.features(x))


@app.cell
def _():
    _d = datasets.mnist()
    _Xtr = (_d["X_train"].astype(np.float32) / 255.0)[:, None]
    _Xte = (_d["X_test"].astype(np.float32) / 255.0)[:, None]
    _ytr, _yte = _d["y_train"], _d["y_test"]
    _src = _ytr < 5
    # source task: digits 0-4 (labels used to pretrain); target task: digits 5-9, relabelled 0-4
    X_src, y_src = _Xtr[_src][:15_000], _ytr[_src][:15_000]
    X_pool, y_pool = _Xtr[~_src], _ytr[~_src] - 5  # the target task's labelled pool
    X_test, y_test = _Xte[_yte >= 5][:2000], _yte[_yte >= 5][:2000] - 5

    torch.manual_seed(0)
    _t0 = time.time()
    pretrained = SmallCNN()
    _opt = torch.optim.Adam(pretrained.parameters(), lr=2e-3)
    pretrained.train()
    for _x, _y in make_loader(X_src, y_src, batch_size=128, seed=0):
        _loss = nn.functional.cross_entropy(pretrained(_x), _y)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    pretrained.eval()
    pretrain_seconds = time.time() - _t0
    mo.md(f"pretrained a `SmallCNN` on {len(X_src):,} images of the digits 0–4 for one epoch "
          f"({pretrain_seconds:.0f} s). target pool: {len(X_pool):,} images of 5–9; target test set: "
          f"{len(X_test):,}.")
    return X_pool, X_test, pretrained, y_pool, y_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how good is a network that has never been trained?

    three representations of the digits 5–9, each evaluated with your linear probe and
    your kNN (k = 20, T = 0.07) using 1,000 labelled examples (200 per class):

    - **raw pixels**: the 784 pixel values;
    - **random CNN**: a `SmallCNN` straight from its initialisation, never trained;
    - **pretrained CNN**: the `SmallCNN` above, trained on the digits 0–4 (a different task).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "rank the three representations by linear-probe accuracy on the digits 5-9. roughly what accuracy do you "
        "expect from the random, never-trained CNN (chance is 20%)?"
    )
    p1
    return (p1,)


@app.function
def balanced_subset(labels, per_class, seed):
    """indices of `per_class` random examples of each class (classes 0..k-1)."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in np.unique(labels)])


@app.function
def features_of(encoder, X, y):
    """your extract_features over a non-shuffled loader."""
    return ex.extract_features(encoder, make_loader(X, y, batch_size=512, shuffle=False))


@app.cell
def _(X_pool, X_test, p1, pretrained, y_pool, y_test):
    ui.gate(p1)
    _t0 = time.time()
    _idx = balanced_subset(y_pool, 200, seed=0)
    torch.manual_seed(1)
    _random = SmallCNN()
    _rows = []
    for _name, _get in [
        ("raw pixels", lambda X, y: (X.reshape(len(X), -1), y)),
        ("random CNN", lambda X, y: features_of(_random.features, X, y)),
        ("pretrained CNN (digits 0–4)", lambda X, y: features_of(pretrained.features, X, y)),
    ]:
        _ftr, _ltr = _get(X_pool[_idx], y_pool[_idx])
        _fte, _lte = _get(X_test, y_test)
        _probe = ex.linear_probe(_ftr, _ltr, _fte, _lte)
        _knn = ex.knn_eval(_ftr, _ltr, _fte, _lte, k=20, temperature=0.07)
        _rows.append(f"| {_name} | {_ftr.shape[1]} | {_probe:.3f} | {_knn:.3f} |")
    _se = np.sqrt(0.9 * 0.1 / len(X_test))
    mo.md("| representation | dim | linear probe | kNN (k=20) |\n|---|---|---|---|\n" + "\n".join(_rows)
          + f"\n\n{len(X_test):,} test images: a binomial standard error of about ±{_se:.3f} at 90% accuracy. "
          f"({time.time() - _t0:.0f} s)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a network that has never seen a gradient lets a linear classifier
    do far better than chance. what do random 3×3 filters, ReLUs and pooling already
    compute that helps? now look at kNN: raw pixels win it outright, and the network
    trained on 0–4 does no better than the random one. MNIST digits of the same class
    really are close in pixel space, so what is kNN measuring that the probe isn't? and
    what might training a 64-dimensional feature space to separate *five* classes do to
    the neighbourhoods of five *other* classes? (keep in mind that a benchmark where raw
    pixels are this strong is a poor judge of representations: the capstone uses CIFAR-10.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: probe or fine-tune?

    now the label budget varies: 5, 20, 100 or 500 labelled examples per class of the
    target task. four ways to use them, each over 3 random label subsets:

    - **linear probe** and **kNN** on the frozen pretrained features;
    - **fine-tune**: your `fine_tune` on a copy of the pretrained network, end to end
      (Adam, lr $10^{-3}$, 120 steps of batch 32 whatever the budget);
    - **from scratch**: the same, starting from a random `SmallCNN`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with only 5 labels per class, which method wins, and by roughly how much? at 500 per class, does "
        "fine-tuning beat the linear probe?"
    )
    p2
    return (p2,)


@app.cell
def _(X_pool, X_test, p2, pretrained, y_pool, y_test):
    ui.gate(p2)
    import copy

    _budgets = (5, 20, 100, 500)
    _steps, _bs = 120, 32
    _ftest, _ = features_of(pretrained.features, X_test, y_test)
    _Xte_t = torch.as_tensor(X_test)
    results = {m: np.zeros((len(_budgets), 3)) for m in ("linear probe", "kNN", "fine-tune", "from scratch")}
    _t0 = time.time()
    for _i, _b in enumerate(_budgets):
        for _s in range(3):
            _idx = balanced_subset(y_pool, _b, seed=100 + _s)
            _ftr, _ltr = features_of(pretrained.features, X_pool[_idx], y_pool[_idx])
            results["linear probe"][_i, _s] = ex.linear_probe(_ftr, _ltr, _ftest, y_test)
            results["kNN"][_i, _s] = ex.knn_eval(_ftr, _ltr, _ftest, y_test, k=min(20, len(_idx)))
            _epochs = int(np.ceil(_steps / np.ceil(len(_idx) / _bs)))
            for _m in ("fine-tune", "from scratch"):
                torch.manual_seed(_s)
                _enc = copy.deepcopy(pretrained.features) if _m == "fine-tune" else SmallCNN().features
                _loader = make_loader(X_pool[_idx], y_pool[_idx], batch_size=_bs, seed=_s)
                _model = ex.fine_tune(_enc, 64, 5, _loader, epochs=_epochs, lr=1e-3, seed=_s)
                with torch.no_grad():
                    _pred = torch.cat([_model(_Xte_t[j:j + 1000]).argmax(1) for j in range(0, len(_Xte_t), 1000)])
                results[_m][_i, _s] = float((_pred.numpy() == y_test).mean())
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    for _m, _r in results.items():
        _ax.errorbar(_budgets, _r.mean(1), yerr=_r.std(1, ddof=1), marker="o", capsize=3, label=_m)
    _ax.set(xscale="log", xlabel="labelled examples per class (target task: digits 5–9)", ylabel="test accuracy",
            title="mean ± std over 3 label subsets", xticks=_budgets, xticklabels=[str(b) for b in _budgets])
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _table = "| labels per class | " + " | ".join(results) + " |\n|---|" + "---|" * len(results) + "\n"
    for _i, _b in enumerate(_budgets):
        _table += f"| {_b} | " + " | ".join(f"{r[_i].mean():.3f} ± {r[_i].std(ddof=1):.3f}" for r in results.values()) + " |\n"
    mo.vstack([_fig, mo.md(_table + f"\n({time.time() - _t0:.0f} s)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compare the spread over label subsets with the gaps between methods:
    at which budgets can you actually tell the methods apart with 3 subsets? fine-tuning
    updates every weight from as few as 25 labels: why doesn't it simply overfit and lose
    to the probe (what does starting from pretrained weights, and a fixed small number of
    steps, protect you from)? where does "from scratch" catch up, and what does that say
    about *when* pretraining pays? (keep this picture in mind: the chapter capstone
    repeats it on CIFAR-10 with a network pretrained *without any labels*.)
    """)
    return


if __name__ == "__main__":
    app.run()
