import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.cnn import ResNet, profile_model  # yours, from topics 03 and 04
    from mlfs.torchnn import evaluate, make_loader  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.cifar10()  # real CIFAR-10, or a small synthetic stand-in when offline
    _mean = _d["X_train"][:10000].reshape(-1, 3).mean(0) / 255
    _std = _d["X_train"][:10000].reshape(-1, 3).std(0) / 255

    def _prep(X):  # standardize, NCHW, and average 2x2 blocks: 16x16 images, so everything trains on a CPU
        x = ((X / 255.0 - _mean) / _std).transpose(0, 3, 1, 2)
        return x.reshape(len(x), 3, 16, 2, 16, 2).mean(axis=(3, 5)).astype(np.float32)

    _src_tr = np.flatnonzero(_d["y_train"] < 5)[:10000]
    _src_te = np.flatnonzero(_d["y_test"] < 5)[:2000]
    _tgt_tr = np.flatnonzero(_d["y_train"] >= 5)
    _tgt_te = np.flatnonzero(_d["y_test"] >= 5)[:2000]
    # source task: airplane, automobile, bird, cat, deer. target task: dog, frog, horse, ship, truck
    Xs, ys = _prep(_d["X_train"][_src_tr]), _d["y_train"][_src_tr]
    Xs_test, ys_test = _prep(_d["X_test"][_src_te]), _d["y_test"][_src_te]
    Xt_pool, yt_pool = _prep(_d["X_train"][_tgt_tr]), _d["y_train"][_tgt_tr] - 5
    Xt_test, yt_test = _prep(_d["X_test"][_tgt_te]), _d["y_test"][_tgt_te] - 5


@app.function
def small_resnet(num_classes=5, seed=0):
    """your ResNet with one block per stage and base width 16 (about 308k parameters)."""
    torch.manual_seed(seed)
    return ResNet((1, 1, 1, 1), num_classes, base_width=16)


@app.function
def few_shot_subset(n_per_class, seed):
    """n_per_class labelled target images per class, drawn from the target training pool."""
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(yt_pool == c) for c in range(5)]  # (offline, the stand-in has fewer than 500 per class)
    idx = np.concatenate([rng.choice(p, min(n_per_class, len(p)), replace=False) for p in pools])
    return Xt_pool[idx], yt_pool[idx]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # modern convnets & transfer · lab

    every experiment here uses **your** `DepthwiseSeparableConv`, `ConvNeXtBlock`,
    `replace_head`, `freeze_backbone` and `fine_tune` from `exercise.py`, with your
    `ResNet` (topic 04) and `profile_model` (topic 03). transfer happens *within* CIFAR-10:
    a network pretrained on five classes (airplane, automobile, bird, cat, deer) is
    adapted to the other five (dog, frog, horse, ship, truck), with images downsampled to
    $16 \times 16$ so that everything trains on a CPU. offline, synthetic stand-ins are
    used. everything runs on the CPU. for each experiment: write down a prediction,
    lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes two to three minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: fewer multiplications, faster network?

    three ways to process a $C \times 32 \times 32$ feature map with a $3 \times 3$ (or
    larger) spatial filter: a standard $3 \times 3$ convolution, your
    `DepthwiseSeparableConv` (same input and output channels), and your `ConvNeXtBlock`
    ($7 \times 7$ depthwise plus a 4× MLP). parameters and multiply–accumulates come from
    your `profile_model`; the time is one forward pass on a batch of 32 on the CPU, best
    of 5.
    """)
    return


@app.cell
def _():
    channels = mo.ui.dropdown(options=["64", "128", "256"], value="128", label="channels C")
    p1 = ui.predict(
        "at C = 128, the depthwise separable conv does about 8x fewer multiply-accumulates than the standard conv. "
        "how much faster will it run?"
    )
    mo.vstack([p1, channels])
    return channels, p1


@app.cell
def _(channels, p1):
    ui.gate(p1)
    _c = int(channels.value)

    def _best(m, x, reps=5):
        with torch.no_grad():
            m(x)
            best = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                m(x)
                best = min(best, time.perf_counter() - t0)
        return best

    _x = torch.randn(32, _c, 32, 32)
    _blocks = {"standard 3x3 conv": nn.Sequential(nn.Conv2d(_c, _c, 3, padding=1, bias=False), nn.BatchNorm2d(_c),
                                                  nn.ReLU()),
               "depthwise separable": ex.DepthwiseSeparableConv(_c, _c),
               "ConvNeXt block": ex.ConvNeXtBlock(_c)}
    _rows, _base = [], None
    for _name, _m in _blocks.items():
        _m.eval()
        _prof = profile_model(_m, (_c, 32, 32))
        _params = sum(p.numel() for p in _m.parameters())
        _macs = sum(r["macs"] for r in _prof)
        _t = _best(_m, _x)
        if _base is None:
            _base = (_macs, _t)
        _rows.append(f"| {_name} | {_params:,} | {_macs / 1e6:.1f} M | {_base[0] / _macs:.1f}× | {_t * 1e3:.1f} ms | "
                     f"{_base[1] / _t:.1f}× |")
    mo.md("| block | parameters | MACs per image | fewer MACs than standard | time (batch 32) | faster than "
          "standard |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** count the arithmetic per byte moved: a standard conv reads each input
    value once and uses it $9 C$ times; a depthwise conv uses it only 9 times. which kind
    of operation is limited by arithmetic, and which by memory bandwidth? why do MobileNets
    still win on phones, and why did ConvNeXt's authors accept the cost of $7 \times 7$
    depthwise kernels (look at its MACs and receptive field)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: linear probe, fine-tune, or start from scratch?

    first, a small `ResNet` (one block per stage, base width 16) is pretrained on 10,000
    source images (2 epochs, Adam). then, for each budget of $n$ labelled target images
    per class, four ways to get a target classifier, each with 60 optimizer steps of your
    `fine_tune` (batch 32, AdamW, learning rate $10^{-3}$):

    - **linear probe**: `mode="linear"`, the pretrained backbone frozen;
    - **fine-tune**: `mode="full"`, the backbone at $0.1 \times$ the head's learning rate;
    - **fine-tune, one lr**: `mode="full"` with the same learning rate everywhere;
    - **from scratch**: the same network, randomly initialized, trained like "fine-tune,
      one lr".

    test accuracy on 2,000 target images (± about 0.011), for one draw of the labelled subset.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 5 labelled images per class, which of the three wins? with 500 per class? is there a crossover?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _t0 = time.perf_counter()
    pretrained = small_resnet(num_classes=5)
    _opt = torch.optim.Adam(pretrained.parameters(), lr=2e-3)
    _loader = make_loader(Xs, ys, batch_size=128, seed=0)
    for _epoch in range(2):
        pretrained.train()
        for _xb, _yb in _loader:
            _opt.zero_grad()
            F.cross_entropy(pretrained(_xb), _yb).backward()
            _opt.step()
    source_acc = evaluate(pretrained, make_loader(Xs_test, ys_test, batch_size=500, shuffle=False))["accuracy"]
    _pre_s = time.perf_counter() - _t0
    _budgets, _modes = [5, 50, 500], ["linear probe", "fine-tune", "fine-tune, one lr", "from scratch"]
    _acc = {m: [] for m in _modes}
    _t0 = time.perf_counter()
    for _n in _budgets:
        _per = {m: [] for m in _modes}
        for _seed in (0,):
            _X, _y = few_shot_subset(_n, _seed)
            _epochs = max(1, math.ceil(60 / math.ceil(len(_X) / 32)))
            for _mode in _modes:
                _m = small_resnet(seed=_seed) if _mode == "from scratch" else copy.deepcopy(pretrained)
                torch.manual_seed(_seed)
                ex.fine_tune(_m, make_loader(_X, _y, batch_size=32, seed=_seed), 5,
                             mode="linear" if _mode == "linear probe" else "full", epochs=_epochs, lr=1e-3,
                             backbone_lr_mult=0.1 if _mode == "fine-tune" else 1.0)
                _per[_mode].append(evaluate(_m, make_loader(Xt_test, yt_test, batch_size=500, shuffle=False))["accuracy"])
        for _mode in _modes:
            _acc[_mode].append(float(np.mean(_per[_mode])))
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    for _mode, _style in zip(_modes, ("o-", "s-", "d-.", "^--")):
        _ax.semilogx(_budgets, _acc[_mode], _style, label=_mode)
    _ax.axhline(0.2, color="gray", lw=0.8, ls=":", label="chance")
    _ax.set(xlabel="labelled target images per class", ylabel="target test accuracy")
    _ax.set_xticks(_budgets, [str(b) for b in _budgets])
    _ax.minorticks_off()
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"pretraining: {_pre_s:.0f} s, source-task test accuracy **{source_acc:.3f}**; transfer "
                           f"runs: {time.perf_counter() - _t0:.0f} s.\n\n"
                           "| images per class | " + " | ".join(_modes) + " |\n|---|---|---|---|---|\n"
                           + "\n".join(f"| {n} | " + " | ".join(f"{_acc[m][i]:.3f}" for m in _modes) + " |"
                                       for i, n in enumerate(_budgets)))])
    return pretrained, source_acc


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with 5 images per class, how many parameters does each method fit, and
    how many numbers of supervision does it get? why can fine-tuning *hurt* with very little
    data, even with a small backbone learning rate? where does the advantage of
    pretraining go as the budget grows, and what would you expect at 5,000 per class?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what does adapting a backbone cost the old task?

    the target task now comes from a different *domain*: the same dog/frog/horse/ship/truck
    images, but dim and low-contrast (as if taken at dusk: each standardized pixel $x$
    becomes $0.5x - 1$). three ways to adapt the pretrained network to 100 of them per
    class, all on exactly the same batches (10 epochs, batch 16, AdamW at $10^{-3}$): your
    `fine_tune(mode="linear")`; the obvious hand-written linear probe (your
    `freeze_backbone`, then train the head with the model in `train()` mode as usual,
    otherwise identical); and your `fine_tune(mode="full")`. afterwards we put the
    *original* source head back on each backbone and test it on the original source task,
    which none of them trained on.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for each of the three, will the backbone (with its old head) still classify the original source task as "
        "well as before? the two probes never update a single backbone weight."
    )
    p3
    return (p3,)


@app.cell
def _(p3, pretrained, source_acc):
    ui.gate(p3)
    _X, _y = few_shot_subset(100, 0)
    _X = 0.5 * _X - 1.0  # the dusk domain
    _Xt_dusk = 0.5 * Xt_test - 1.0
    _src_head = copy.deepcopy(pretrained.fc)
    _rows = []
    for _name in ("your fine_tune(mode='linear')", "hand-written probe in train() mode", "your fine_tune(mode='full')"):
        _m = copy.deepcopy(pretrained)
        _loader = make_loader(_X, _y, batch_size=16, seed=0)  # identical batches for all three
        torch.manual_seed(0)  # identical new heads
        if _name.startswith("your"):
            ex.fine_tune(_m, _loader, 5, mode=_name.split("'")[1], epochs=10, lr=1e-3)
        else:
            ex.replace_head(_m, 5)
            ex.freeze_backbone(_m)
            _opt = torch.optim.AdamW(list(_m.fc.parameters()), lr=1e-3, weight_decay=0.0)
            for _epoch in range(10):
                _m.train()  # the only difference from your linear probe
                for _xb, _yb in _loader:
                    _opt.zero_grad()
                    F.cross_entropy(_m(_xb), _yb).backward()
                    _opt.step()
        _tgt = evaluate(_m, make_loader(_Xt_dusk, yt_test, batch_size=500, shuffle=False))["accuracy"]
        _same = all(torch.equal(a, b) for (k, a), b in zip(_m.named_parameters(), pretrained.parameters())
                    if not k.startswith("fc."))
        _back = copy.deepcopy(_m)
        _back.fc = _src_head
        _src = evaluate(_back, make_loader(Xs_test, ys_test, batch_size=500, shuffle=False))["accuracy"]
        _rows.append(f"| {_name} | {_same} | {_tgt:.3f} | {source_acc:.3f} → **{_src:.3f}** |")
    mo.md("| method | backbone parameters unchanged? | target (dusk) accuracy | source accuracy before → after |\n"
          "|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the hand-written probe changed no parameter and saw exactly the same
    batches as yours, yet its accuracies differ on *both* tasks: what did it change?
    (think about what batch norm does in training mode, besides normalizing, and what the
    statistics of dusk images look like.) why did that help the target task and hurt the
    source task? how does full fine-tuning's damage compare? when would you choose a
    linear probe in eval mode despite its lower target accuracy (think about serving
    several tasks from one backbone)?
    """)
    return


if __name__ == "__main__":
    app.run()
