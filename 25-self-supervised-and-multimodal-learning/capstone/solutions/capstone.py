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

    from mlfs import checks, datasets
    from mlfs.cnn import ResNet  # yours, from ch 16
    from mlfs.ssl import (ProjectionHead, SimCLRAugment, extract_features, info_nce_loss, knn_eval,  # yours, this chapter
                          linear_probe)
    from mlfs.torchnn import get_device, make_loader  # yours, from ch 15
    from mlfs.train import CosineSchedule  # yours, from ch 15

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    DEVICE = torch.device("cpu") if SMALL else get_device()
    CFG = dict(
        width=8 if SMALL else 32,             # ResNet-18 layout; base width 32 = half the standard width
        pretrain_epochs=1 if SMALL else 12,
        pretrain_batch=128 if SMALL else 256,
        pretrain_lr=1e-3,
        temperature=0.2,
        fractions=(0.01, 0.1, 1.0),
        subset_seeds=(0, 1) if SMALL else (0, 1, 2),
        steps={0.01: 10, 0.1: 20, 1.0: 30} if SMALL else {0.01: 150, 0.1: 300, 1.0: 1000},
        finetune_lr=0.01,
        scratch_lr=0.05,
    )
    if SMALL:
        torch.set_num_threads(min(2, torch.get_num_threads()))  # never raise the thread count


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 25 · capstone: label efficiency (reference solution)

    read [`brief.md`](brief.md) first. you'll pretrain a ResNet on CIFAR-10 with SimCLR,
    using no labels at all, then measure what that pretraining is worth when you have 1%,
    10% or 100% of the labels, against a network trained from scratch on the same labels.

    | stage | you build |
    |---|---|
    | 1 | `label_subset`: class-balanced, nested label subsets |
    | 2 | `pretrain_simclr`: SimCLR with a warmup + cosine schedule, on uint8 or float images |
    | 3 | `train_classifier`, `accuracy`: supervised training for a fixed number of steps, with augmentation |
    | 4 | `summarize`: mean, spread and count per (method, fraction) |

    when they're green, scroll down: the study runs on your functions, and the write-up
    prompts are at the end. with `MLFS_OFFLINE=1` everything runs on a tiny synthetic
    stand-in in about three minutes on a CPU (its numbers only check the plumbing); the
    full config needs a GPU (MPS or CUDA) and one to one and a half hours.
    """)
    return


@app.function
def label_subset(labels, fraction, *, seed=0):
    """stage 1: sorted int64 indices; per class (in sorted order) the first max(1, round(fraction * n_c)) of a permutation from default_rng(seed)."""
    labels = np.asarray(labels)
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = max(1, math.floor(fraction * len(idx) + 0.5))
        picked.append(rng.permutation(idx)[:k])
    return np.sort(np.concatenate(picked)).astype(np.int64)


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", label_subset=label_subset)
    return


@app.function
def to_float(xb, device):
    """a batch of uint8 (0..255) or float (0..1) images -> float32 on device."""
    xb = xb.to(device)
    return xb.float() / 255.0 if xb.dtype == torch.uint8 else xb.float()


@app.function
def pretrain_simclr(encoder, feature_dim, images, *, epochs=1, batch_size=256, lr=1e-3, temperature=0.2, strength=0.5,
                    warmup_epochs=0, weight_decay=1e-6, seed=0, device="cpu"):
    """stage 2: SimCLR (AdamW, linear warmup then cosine to 0, stepped every batch); trains encoder in place; returns (head, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(np.asarray(images))
    n = len(X)
    bs = min(batch_size, n)
    steps_per_epoch = n // bs
    head = ProjectionHead(feature_dim)
    encoder.to(device)
    head.to(device)
    augment = SimCLRAugment(X.shape[-1], strength=strength)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)
    sched = CosineSchedule(opt, epochs * steps_per_epoch, warmup_steps=warmup_epochs * steps_per_epoch)
    history = {"loss": [], "lr": []}
    for _ in range(epochs):
        encoder.train()
        head.train()
        perm = rng.permutation(n)
        for start in range(0, steps_per_epoch * bs, bs):
            xb = to_float(X[perm[start:start + bs]], device)
            loss = info_nce_loss(head(encoder(augment(xb))), head(encoder(augment(xb))), temperature=temperature)
            history["lr"].append(opt.param_groups[0]["lr"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            history["loss"].append(loss.item())
    return head, history


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", pretrain_simclr=pretrain_simclr)
    return


@app.function
def flip_crop(x):
    """per image: horizontal flip with probability 1/2, then a random crop after zero-padding by 4 pixels."""
    B, C, H, W = x.shape
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (B,), device=x.device)
    ox = torch.randint(0, 9, (B,), device=x.device)
    rows = (oy[:, None] + torch.arange(H, device=x.device))[:, :, None]  # (B, H, 1)
    cols = (ox[:, None] + torch.arange(W, device=x.device))[:, None, :]  # (B, 1, W)
    b = torch.arange(B, device=x.device)[:, None, None]
    return xp.permute(0, 2, 3, 1)[b, rows, cols].permute(0, 3, 1, 2)


@app.function
def train_classifier(encoder, feature_dim, n_classes, images, labels, *, steps, batch_size=128, lr=0.05,
                     weight_decay=5e-4, augment=True, seed=0, device="cpu"):
    """stage 3: nn.Sequential(encoder, Linear) trained for exactly `steps` SGD steps (momentum 0.9, cosine to 0); returned in eval mode."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(np.asarray(images))
    y = torch.as_tensor(np.asarray(labels), dtype=torch.int64)
    n = len(X)
    bs = min(batch_size, n)
    model = nn.Sequential(encoder, nn.Linear(feature_dim, n_classes)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = CosineSchedule(opt, steps)
    model.train()
    done = 0
    while done < steps:
        perm = rng.permutation(n)
        for start in range(0, (n // bs) * bs, bs):
            if done == steps:
                break
            idx = perm[start:start + bs]
            xb = to_float(X[idx], device)
            if augment:
                xb = flip_crop(xb)
            loss = F.cross_entropy(model(xb), y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            done += 1
    model.eval()
    return model


@app.function
def accuracy(model, images, labels, *, batch_size=512, device="cpu"):
    """stage 3: fraction of correct argmax predictions, in eval mode under no_grad (the model's modes restored)."""
    modes = {m: m.training for m in model.modules()}
    model.eval()
    X = torch.as_tensor(np.asarray(images))
    y = np.asarray(labels)
    correct = 0
    try:
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                pred = model(to_float(X[start:start + batch_size], device)).argmax(1).cpu().numpy()
                correct += int((pred == y[start:start + batch_size]).sum())
    finally:
        for m, mode in modes.items():
            m.training = mode
    return correct / len(X)


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", train_classifier=train_classifier, accuracy=accuracy)
    return


@app.function
def summarize(records):
    """stage 4: records [{'method', 'fraction', 'accuracy'}] -> {(method, fraction): {'mean', 'std' (ddof=1, nan if n=1), 'n'}}."""
    groups = {}
    for r in records:
        groups.setdefault((r["method"], r["fraction"]), []).append(float(r["accuracy"]))
    out = {}
    for key, accs in groups.items():
        a = np.array(accs)
        out[key] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
                    "n": len(a)}
    return out


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", summarize=summarize)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## the study

    config: **{"small (offline, synthetic data)" if SMALL else "full (CIFAR-10)"}** on **{DEVICE}**:
    `{CFG}`
    """)
    return


@app.function
def make_encoder(seed):
    """ResNet-18 layout at CFG width, CIFAR stem; the encoder is its forward_features (feature_dim = 8 x width)."""
    torch.manual_seed(seed)
    net = ResNet((2, 2, 2, 2), 10, base_width=CFG["width"])
    net.fc = nn.Identity()  # forward(x) now returns the pooled features
    return net, net.feature_dim


@app.cell
def _():
    _d = datasets.cifar10()
    X_train = np.ascontiguousarray(_d["X_train"].transpose(0, 3, 1, 2))  # uint8 NCHW
    X_test = np.ascontiguousarray(_d["X_test"].transpose(0, 3, 1, 2))
    y_train, y_test = _d["y_train"], _d["y_test"]
    mo.md(f"{len(X_train):,} training and {len(X_test):,} test images, {X_train.shape[1:]} uint8.")
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train):
    _t0 = time.time()
    encoder, feature_dim = make_encoder(0)
    _head, pre_hist = pretrain_simclr(encoder, feature_dim, X_train, epochs=CFG["pretrain_epochs"],
                                      batch_size=CFG["pretrain_batch"], lr=CFG["pretrain_lr"],
                                      temperature=CFG["temperature"], warmup_epochs=1 if CFG["pretrain_epochs"] > 2 else 0,
                                      seed=0, device=DEVICE)
    pretrain_minutes = (time.time() - _t0) / 60
    encoder = encoder.to("cpu").eval()
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(pre_hist["loss"])
    _ax.axhline(np.log(2 * CFG["pretrain_batch"] - 1), color="gray", ls="--", lw=1, label="chance, log(2B − 1)")
    _ax.set(xlabel="step", ylabel="InfoNCE loss", title=f"SimCLR pretraining ({pretrain_minutes:.1f} min)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return encoder, feature_dim, pretrain_minutes


@app.function
def frozen_features(enc, X, y):
    """features of a frozen encoder (your extract_features), computed on DEVICE."""
    enc = enc.to(DEVICE)
    out = extract_features(enc, make_loader(X, y, batch_size=512, shuffle=False), device=DEVICE)
    enc.to("cpu")
    return out


@app.cell
def _(X_test, X_train, encoder, feature_dim, y_test, y_train):
    _t0 = time.time()
    records = []
    _rand, _ = make_encoder(1)
    _feat = {"SimCLR": (frozen_features(encoder, X_train, y_train)[0], frozen_features(encoder, X_test, y_test)[0]),
             "random init": (frozen_features(_rand, X_train, y_train)[0], frozen_features(_rand, X_test, y_test)[0])}
    for _frac in CFG["fractions"]:
        for _s in (CFG["subset_seeds"] if _frac < 1 else CFG["subset_seeds"][:1]):
            _idx = label_subset(y_train, _frac, seed=_s)
            for _name, (_ftr, _fte) in _feat.items():
                records.append({"method": f"linear probe ({_name})", "fraction": _frac, "accuracy": linear_probe(
                    _ftr[_idx], y_train[_idx], _fte, y_test, max_iters=300 if SMALL else 1000)})
                records.append({"method": f"kNN ({_name})", "fraction": _frac, "accuracy": knn_eval(
                    _ftr[_idx], y_train[_idx], _fte, y_test, k=20)})
            for _name, _start, _lr in (("fine-tune SimCLR", encoder, CFG["finetune_lr"]),
                                       ("from scratch", make_encoder(100 + _s)[0], CFG["scratch_lr"])):
                _model = train_classifier(copy.deepcopy(_start), feature_dim, 10, X_train[_idx], y_train[_idx],
                                          steps=CFG["steps"][_frac], lr=_lr, seed=_s, device=DEVICE)
                records.append({"method": _name, "fraction": _frac,
                                "accuracy": accuracy(_model, X_test, y_test, device=DEVICE)})
    study_minutes = (time.time() - _t0) / 60
    return records, study_minutes


@app.cell
def _(pretrain_minutes, records, study_minutes):
    summary = summarize(records)
    _methods = sorted({m for m, _ in summary})
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    for _m in _methods:
        _fr = [f for f in CFG["fractions"] if (_m, f) in summary]
        _mean = np.array([summary[(_m, f)]["mean"] for f in _fr])
        _std = np.nan_to_num(np.array([summary[(_m, f)]["std"] for f in _fr]))
        _ax.errorbar(_fr, _mean, yerr=_std, marker="o", capsize=3, label=_m,
                     ls="--" if "random" in _m else "-", alpha=0.6 if "random" in _m else 1.0)
    _ax.set(xscale="log", xlabel="fraction of CIFAR-10 training labels", ylabel="test accuracy",
            xticks=CFG["fractions"], xticklabels=[f"{f:.0%}" for f in CFG["fractions"]],
            title="label efficiency (error bars: std over label subsets)")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _rows = "\n".join(
        f"| {m} | " + " | ".join(
            (f"{summary[(m, f)]['mean']:.4f} ± {summary[(m, f)]['std']:.4f} (n={summary[(m, f)]['n']})"
             if summary[(m, f)]["n"] > 1 else f"{summary[(m, f)]['mean']:.4f} (n=1)") if (m, f) in summary else "–"
            for f in CFG["fractions"]) + " |" for m in _methods)
    mo.vstack([_fig, mo.md("| method | " + " | ".join(f"{f:.0%} labels" for f in CFG["fractions"]) + " |\n|---|"
                           + "---|" * len(CFG["fractions"]) + "\n" + _rows
                           + f"\n\npretraining {pretrain_minutes:.1f} min, evaluation {study_minutes:.1f} min.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer in this cell (replace the prompts with your text). every number needs its
    uncertainty and the number of runs behind it.

    1. **the headline.** at 1%, 10% and 100% of the labels, how much does SimCLR
       pretraining buy over training from scratch, with fine-tuning and with a linear
       probe? where does the crossover appear to be, if any, and what about the budget
       (pretraining epochs, supervised steps, untuned learning rates) could move it?
    2. **what's measured.** which differences are larger than the spread over label
       subsets? which sources of variation does your study *not* measure (pretraining
       seed, fine-tuning seed, hyperparameters), and how might they change your
       conclusions?
    3. **probe versus fine-tune.** how do the frozen-feature protocols compare with
       fine-tuning at each fraction in your results, and why? (at 1%, fine-tuning has
       few labels to adapt millions of weights; at 100%, frozen features can't adapt at
       all.) relate it to topic 01's lab.
    4. **baselines.** what do the random-init rows tell you about how much of the
       "SimCLR" number is the architecture rather than the pretraining?
    5. **an ablation.** state a hypothesis about one choice (temperature, augmentation
       strength, pretraining length, SimSiam instead of SimCLR, …), *then* run it, and
       report what happened with uncertainty.
    """)
    return


if __name__ == "__main__":
    app.run()
