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

    from mlfs import datasets, notebooks, ui
    from mlfs.cnn import ResNet  # yours, from ch 16 · topic 04
    from mlfs.torchnn import count_parameters, get_device  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    DEVICE = get_device()
    OFFLINE = datasets.is_offline()
    STEPS = 150 if OFFLINE else 300  # the same number of optimizer steps for every model and every data size
    SIZES = (500, 1900) if OFFLINE else (1000, 5000)
    BIG = 1900 if OFFLINE else 20000  # the optional third size, behind a run button


@app.function
def make_resnet():
    """a narrow ResNet-18 (your ch 16 ResNet, base width 16): about 0.7M parameters."""
    return ResNet((2, 2, 2, 2), 10, base_width=16)


@app.function
def make_vit(**kw):
    """a small ViT on 4 x 4 patches (64 tokens + [CLS]), 4 layers of width 128: about 0.8M parameters."""
    return ex.ViT(32, 4, 3, 10, d_model=128, n_layers=4, n_heads=4, **kw)


@app.function
def augment(x, generator):
    """random crops of 32 from 4-pixel zero padding, and random horizontal flips, on a whole batch."""
    B = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4))
    i, j = torch.randint(0, 9, (2,), generator=generator).tolist()  # one offset per batch keeps it cheap
    out = padded[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(B, generator=generator) < 0.5
    return torch.where(flip.to(x.device)[:, None, None, None], out.flip(-1), out)


@app.function
def train_eval(model, X, y, X_test, y_test, *, steps=STEPS, batch_size=128, lr=1e-3, seed=0):
    """AdamW (weight decay 0.05) with 50 warmup steps and cosine decay; returns test accuracy."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = torch.randint(0, len(X), (batch_size,), generator=g)
        xb = augment(X[idx].to(DEVICE), g)
        loss = F.cross_entropy(model.train()(xb), y[idx].to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(X_test), 500):
            correct += (model(X_test[start:start + 500].to(DEVICE)).argmax(-1).cpu() == y_test[start:start + 500]).sum().item()
    return correct / len(X_test)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # vision transformers · lab

    the experiments train **your** `ViT` (with your `PatchEmbedding`, and your
    `TransformerBlock` inside) against your ResNet from chapter 16 on CIFAR-10, and look inside
    the trained ViT with your `attention_distance`. for each one: write down a prediction, lock it
    in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the lab trains five small models on CIFAR-10 (about 4 minutes on a
    busy laptop GPU, `mps`; longer on a CPU), plus two optional ones behind a button; offline it uses a
    small synthetic stand-in.
    """)
    return


@app.cell
def _():
    _d = datasets.cifar10()
    _mean = torch.tensor([0.491, 0.482, 0.447])[:, None, None]
    _std = torch.tensor([0.247, 0.243, 0.262])[:, None, None]

    def prep(a):
        return (torch.tensor(a, dtype=torch.float32).permute(0, 3, 1, 2) / 255 - _mean) / _std

    _perm = np.random.default_rng(0).permutation(len(_d["X_train"]))
    X_train, y_train = prep(_d["X_train"][_perm]), torch.tensor(_d["y_train"][_perm])
    X_test, y_test = prep(_d["X_test"][:2000]), torch.tensor(_d["y_test"][:2000])
    mo.md(f"{'CIFAR-10' if not OFFLINE else 'offline stand-in'}: {len(X_train):,} training images available, "
          f"{len(X_test):,} test images · device **{DEVICE}** · ResNet {count_parameters(make_resnet()):,} parameters, "
          f"ViT {count_parameters(make_vit()):,}")
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: ViT against ResNet, by amount of data

    a ResNet and a ViT of about the same size, trained with the same recipe (AdamW, warmup and
    cosine, random crops and flips) for the same 300 steps of 128 images, on the first 1,000 or
    5,000 training images. accuracy on 2,000 test images. a third size, 20,000 images, takes
    several more minutes: it's behind a button below, with the reference run's numbers shown
    until you press it.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how far behind the ResNet is the ViT with 1,000 images? with 5,000? does the gap shrink, grow or stay the same?"
    )
    p1
    return (p1,)


@app.cell
def _(X_test, X_train, p1, y_test, y_train):
    ui.gate(p1)
    _t0 = time.perf_counter()
    acc = {"ResNet": [], "ViT": []}
    vit_big = None
    for _n in SIZES:
        acc["ResNet"].append(train_eval(make_resnet(), X_train[:_n], y_train[:_n], X_test, y_test))
        torch.manual_seed(0)
        _vit = make_vit()
        acc["ViT"].append(train_eval(_vit, X_train[:_n], y_train[:_n], X_test, y_test))
        vit_big = _vit
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    for _k, _v in acc.items():
        _ax.plot(SIZES, _v, "o-", label=_k)
    _ax.set(xscale="log", xlabel="training images", ylabel="test accuracy", xticks=SIZES, xticklabels=[str(s) for s in SIZES])
    _ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| training images | ResNet | ViT |\n|---|---|---|\n"
                           + "\n".join(f"| {n:,} | {a:.3f} | {b:.3f} |" for n, a, b in zip(SIZES, acc["ResNet"], acc["ViT"]))
                           + f"\n\n{time.perf_counter() - _t0:.0f} s")])
    return (vit_big,)


@app.cell
def _():
    big_button = mo.ui.run_button(label=f"also train both models on {BIG:,} images (several minutes)")
    big_button
    return (big_button,)


@app.cell
def _(X_test, X_train, big_button, y_test, y_train):
    if big_button.value:
        _t0 = time.perf_counter()
        _r = train_eval(make_resnet(), X_train[:BIG], y_train[:BIG], X_test, y_test)
        torch.manual_seed(0)
        _v = train_eval(make_vit(), X_train[:BIG], y_train[:BIG], X_test, y_test)
        _out = mo.md(f"{BIG:,} images: ResNet **{_r:.3f}**, ViT **{_v:.3f}** ({time.perf_counter() - _t0:.0f} s)")
    else:
        _out = mo.md(f"reference run (`mps`, 300 steps), 20,000 images: ResNet **0.505**, ViT **0.409**. "
                     "press the button to run it yourself.")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a convolution builds in two assumptions that a ViT has to learn from data:
    what are they? why does that cost the most when data is scarce? what would you expect with
    10× or 1,000× more images (the ViT paper's answer is in the lesson)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does the ViT use its positions?

    the same ViT on 5,000 images, with its position embeddings removed (fixed at zero), so
    it sees the image as an unordered *set* of patches.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "without any position information, the ViT can't tell where a patch came from. how much accuracy does it lose?"
    )
    p2
    return (p2,)


@app.cell
def _(X_test, X_train, p2, y_test, y_train):
    ui.gate(p2)
    torch.manual_seed(0)
    _nopos = make_vit()
    with torch.no_grad():
        _nopos.pos_emb.zero_()
    _nopos.pos_emb.requires_grad_(False)
    _n = SIZES[-1]
    nopos_acc = train_eval(_nopos, X_train[:_n], y_train[:_n], X_test, y_test)
    mo.md(f"ViT without positions, {_n:,} images: **{nopos_acc:.3f}** (compare with experiment 1's last row)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a bag of 64 patches with no arrangement still classifies most images. what in
    a CIFAR-10 patch (colour, texture, a wheel, an eye) says "truck" or "frog" without knowing where
    it is? which classes would you expect to suffer most from losing the arrangement?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how far does each head look?

    your `attention_distance` on 256 test images, for the trained ViT from experiment 1 (the
    5,000-image subset) and for a freshly initialized one: the average distance, in pixels, between a
    patch and the patches it attends to, per layer and head. uniform attention over the image
    would give 16.5 pixels.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "a ViT can attend anywhere from its first layer. after training, will the first layer's heads look locally "
        "(like a convolution) or globally?"
    )
    p3
    return (p3,)


@app.cell
def _(X_test, p3, vit_big):
    ui.gate(p3)
    _x = X_test[:256].to(DEVICE)
    _trained = ex.attention_distance(vit_big, _x)
    torch.manual_seed(1)
    _fresh = ex.attention_distance(make_vit().to(DEVICE), _x)
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    for _l in range(_trained.shape[0]):
        _ax.scatter([_l + 1] * _trained.shape[1], _trained[_l], color="C0", label="trained" if _l == 0 else None)
        _ax.scatter([_l + 1.15] * _fresh.shape[1], _fresh[_l], color="gray", marker="x", label="untrained" if _l == 0 else None)
    _ax.set(xlabel="layer", ylabel="mean attention distance (pixels)", ylim=(0, None))
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("trained, per layer (heads sorted): " + " · ".join(
        f"L{l + 1}: " + ", ".join(f"{v:.1f}" for v in sorted(_trained[l].tolist())) for l in range(_trained.shape[0])))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does an untrained ViT attend uniformly (about 16.5 pixels everywhere)?
    compare the trained heads with that number and with a 3×3 convolution's view (neighbours 4
    pixels away). has this ViT discovered locality? relate your answer to experiment 1: what does a
    model that still attends globally after 300 steps have to learn that the ResNet got for free?
    (Dosovitskiy et al. found strongly local early heads, but only after training on hundreds of
    millions of images.)
    """)
    return


if __name__ == "__main__":
    app.run()
