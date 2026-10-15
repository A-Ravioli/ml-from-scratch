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
    from mlfs.transformer import GPT, generate_cached  # yours, from ch 18 · 04 and 06

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    HAS_MPS = torch.backends.mps.is_available()
    GPU = "cuda" if torch.cuda.is_available() else ("mps" if HAS_MPS else None)  # the accelerator, if any
    DEVICES = ["cpu"] + ([GPU] if GPU else [])
    _d = datasets.mnist()
    N_TRAIN = min(20_000, len(_d["X_train"]))
    XTR = torch.tensor(_d["X_train"][:N_TRAIN].reshape(N_TRAIN, -1) / 255.0, dtype=torch.float32)
    YTR = torch.tensor(_d["y_train"][:N_TRAIN])
    XTE = torch.tensor(_d["X_test"].reshape(len(_d["X_test"]), -1) / 255.0, dtype=torch.float32)
    YTE = torch.tensor(_d["y_test"])


@app.function
def mlp(hidden, seed=0):
    """784 -> hidden... -> 10 with ReLUs."""
    torch.manual_seed(seed)
    sizes = [784, *hidden, 10]
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


@app.function
def train(model, X, Y, *, epochs=3, lr=2e-3, seed=0, loss_fn=None):
    """Adam on minibatches of 128; loss_fn(logits, idx) overrides cross-entropy (idx = the batch's row indices)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), 128):
            idx = perm[s:s + 128]
            out = model(X[idx])
            loss = F.cross_entropy(out, Y[idx]) if loss_fn is None else loss_fn(out, idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@app.function
@torch.no_grad()
def accuracy(model, X=None, Y=None):
    X, Y = (XTE, YTE) if X is None else (X, Y)
    return (model(X).argmax(1) == Y).float().mean().item()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # efficient inference · lab

    the experiments use **your** `fake_quantize`, `quantize_linears`, `magnitude_prune`,
    `distillation_loss` and `decode_tokens_per_second`, on MLPs trained on MNIST ({N_TRAIN:,}
    training images here; a synthetic stand-in offline) and on your chapter 18 GPT. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare with `solutions/lab_notes.md` afterwards. about a minute.
    """)
    return


@app.cell
def _():
    base = train(mlp([256, 256]), XTR, YTR)
    base_acc = accuracy(base)
    mo.md(f"the float32 MLP (784 → 256 → 256 → 10, 3 epochs of Adam): test accuracy **{100 * base_acc:.2f}%**.")
    return base, base_acc


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: accuracy against bits

    every linear layer's weight replaced by your `fake_quantize` at 8 down to 2 bits, with
    one scale per tensor, per output channel, or per group of 32; then the real int8 model
    from your `quantize_linears`: its accuracy, its bytes, and its speed.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("at 4 bits, how much accuracy does each scheme lose? at what bit width does per-channel "
                    "quantization fall apart?")
    p1
    return (p1,)


@app.cell
def _(base, base_acc, p1):
    ui.gate(p1)
    bits_list = [8, 6, 5, 4, 3, 2]
    schemes = {"per tensor": dict(axis=None), "per channel": dict(axis=0), "per group of 32": dict(group_size=32)}
    curves = {}
    for _name, _kw in schemes.items():
        _accs = []
        for _b in bits_list:
            _m = copy.deepcopy(base)
            with torch.no_grad():
                for _l in _m.modules():
                    if isinstance(_l, nn.Linear):
                        _l.weight.copy_(ex.fake_quantize(_l.weight, _b, **_kw))
            _accs.append(accuracy(_m))
        curves[_name] = _accs
    _fig, _ax = plt.subplots(figsize=(5.5, 3.2))
    for _name, _accs in curves.items():
        _ax.plot(bits_list, [100 * a for a in _accs], "o-", label=_name)
    _ax.axhline(100 * base_acc, color="k", ls=":", label="float32")
    _ax.set(xlabel="bits per weight", ylabel="test accuracy (%)", ylim=(0, 100))
    _ax.invert_xaxis()
    _ax.legend(fontsize=8)
    _q = ex.quantize_linears(base)
    _fb = sum(p.numel() * p.element_size() for p in base.parameters())
    _qb = sum(b.numel() * b.element_size() for b in _q.buffers())

    def _t(m):
        with torch.no_grad():
            m(XTE[:512])
            _ts = []
            for _ in range(20):
                _t0 = time.perf_counter()
                m(XTE[:512])
                _ts.append(time.perf_counter() - _t0)
        return min(_ts)

    _tab = "\n".join(f"| {n} | " + " | ".join(f"{100 * a:.1f}" for a in accs) + " |" for n, accs in curves.items())
    mo.vstack([_fig, mo.md("| scheme | " + " | ".join(f"{b} bits" for b in bits_list) + " |\n|---|"
                           + "---|" * len(bits_list) + "\n" + _tab +
                           f"\n\nyour `quantize_linears` (int8, per channel): accuracy **{100 * accuracy(_q):.2f}%** "
                           f"(float {100 * base_acc:.2f}%), **{_qb / 1e3:.0f} KB** of weights against {_fb / 1e3:.0f} KB, "
                           f"forward on 512 images {1e3 * _t(_q):.2f} ms against {1e3 * _t(base):.2f} ms.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is per-tensor quantization the first to break, and which weights does
    it destroy? (compare a few rows' `abs().max()`.) the int8 model is about four times
    smaller: why isn't it faster, and what would make it faster?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: pruning

    your `magnitude_prune` at sparsities from 0 to 98%, globally and per layer, on copies of
    the MLP (no fine-tuning afterwards). then the speed question: the same matmul with a 90%
    sparse weight, dense versus in PyTorch's sparse CSR format.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("how much of the MLP can you zero out before accuracy drops by more than 1 point? is global or "
                    "per-layer pruning better?")
    p2
    return (p2,)


@app.cell
def _(base, p2):
    ui.gate(p2)
    sparsities = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    prune = {s: [] for s in ("global", "layer")}
    for _scope in prune:
        for _sp in sparsities:
            _m = copy.deepcopy(base)
            ex.magnitude_prune(_m, _sp, scope=_scope)
            prune[_scope].append(accuracy(_m))
    _m90 = copy.deepcopy(base)
    _masks = ex.magnitude_prune(_m90, 0.9)
    _dens = {k: f"{100 * v.float().mean().item():.0f}%" for k, v in _masks.items()}
    _w = _m90[2].weight.detach()
    _x = torch.randn(256, 2048)
    _ws = _w.to_sparse_csr()

    def _best(fn):
        fn()
        _ts = []
        for _ in range(20):
            _t0 = time.perf_counter()
            fn()
            _ts.append(time.perf_counter() - _t0)
        return min(_ts)

    _td, _tsp = _best(lambda: _w @ _x), _best(lambda: _ws @ _x)
    _fig, _ax = plt.subplots(figsize=(5.5, 3.2))
    for _scope, _accs in prune.items():
        _ax.plot([100 * s for s in sparsities], [100 * a for a in _accs], "o-", label=_scope)
    _ax.set(xlabel="sparsity (% of weights zeroed)", ylabel="test accuracy (%)", ylim=(0, 100))
    _ax.legend(fontsize=8)
    _tab = "\n".join(f"| {s} | " + " | ".join(f"{100 * a:.1f}" for a in accs) + " |" for s, accs in prune.items())
    mo.vstack([_fig, mo.md("| scope | " + " | ".join(f"{int(100 * s)}%" for s in sparsities) + " |\n|---|"
                           + "---|" * len(sparsities) + "\n" + _tab +
                           f"\n\nat 90% global sparsity the layers keep {_dens} of their weights. a (256 × 256) weight "
                           f"with 90% zeros times a (256 × 2048) input: dense **{1e3 * _td:.3f} ms**, sparse CSR "
                           f"**{1e3 * _tsp:.3f} ms**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where does accuracy collapse, and why does the global scope hold up longer
    (look at how it distributes the zeros across the three layers)? at 90% sparsity a dense
    matmul does 10× more multiplications than necessary: why does the sparse format not win
    by 10×, or at all?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: distillation

    a small student (784 → 32 → 10) with only **500 labelled images**, three ways: trained on
    the labels; trained with your `distillation_loss` against the teacher (the MLP above) on
    the same 500 images; and with the teacher also labelling 5,000 *unlabelled* images (a
    transfer set), on which the student learns only from the soft targets. three seeds each.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("how many accuracy points does distillation on the same 500 images add? and with the unlabelled "
                    "transfer set?")
    p3
    return (p3,)


@app.cell
def _(base, p3):
    ui.gate(p3)
    _n_lab, _n_tr = min(500, N_TRAIN // 4), min(5000, N_TRAIN)
    with torch.no_grad():
        _t_logits = base(XTR[:_n_tr])
    dist = {"labels only": [], "distilled (same 500)": [], "distilled + transfer set": []}
    for _seed in (0, 1, 2):
        _perm = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(_seed))
        _lab = _perm[:_n_lab]
        _xl, _yl = XTR[_lab], YTR[_lab]
        dist["labels only"].append(accuracy(train(mlp([32], seed=_seed), _xl, _yl, epochs=30, seed=_seed)))
        with torch.no_grad():
            _tl = base(_xl)
        _kd = lambda out, idx: ex.distillation_loss(out, _tl[idx], _yl[idx], temperature=4.0, alpha=0.9)
        dist["distilled (same 500)"].append(accuracy(train(mlp([32], seed=_seed), _xl, _yl, epochs=30, seed=_seed,
                                                           loss_fn=_kd)))
        _xt = torch.cat([_xl, XTR[:_n_tr]])
        _tt = torch.cat([_tl, _t_logits])
        _ylab = torch.cat([_yl, torch.full((_n_tr,), -1)])

        def _kd2(out, idx):
            _soft = ex.distillation_loss(out, _tt[idx], _ylab[idx].clamp(min=0), temperature=4.0, alpha=1.0)
            _has = _ylab[idx] >= 0
            _hard = F.cross_entropy(out[_has], _ylab[idx][_has]) if _has.any() else 0.0
            return 0.9 * _soft + 0.1 * _hard
        dist["distilled + transfer set"].append(accuracy(train(mlp([32], seed=_seed), _xt, _ylab, epochs=6,
                                                               seed=_seed, loss_fn=_kd2)))
    mo.md(f"teacher: {100 * accuracy(base):.2f}% (784 → 256 → 256 → 10, all {N_TRAIN:,} labels).\n\n"
          "| student (784 → 32 → 10) | mean test accuracy | range over 3 seeds |\n|---|---|---|\n"
          + "\n".join(f"| {k} | {100 * np.mean(v):.2f}% | {100 * min(v):.2f}–{100 * max(v):.2f}% |"
                      for k, v in dist.items()))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with the same 500 images, what extra information do the soft targets
    carry? why does the transfer set help so much more, even though those images have no
    labels? is the difference between the first two rows larger than the seed-to-seed spread?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 4: batching

    chapter 18's model M (random weights: speed doesn't depend on what it knows) generating
    64 tokens with your KV cache for batches of 1 to 64 sequences, on your fastest device.
    next to it, your `decode_tokens_per_second` with this machine's bandwidth (a 64 MiB copy)
    and peak (a large matmul), measured here.
    """)
    return


@app.cell
def _():
    p4 = ui.predict("from batch 1 to batch 64, how much do tokens per second grow? does your roofline model predict "
                    "the batch-1 speed?")
    p4
    return (p4,)


@app.cell
def _(p4):
    ui.gate(p4)
    _dev = DEVICES[-1]
    _sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(_dev, lambda: None)

    def _best(fn, n=3):
        fn()
        _sync()
        _ts = []
        for _ in range(n):
            _t0 = time.perf_counter()
            fn()
            _sync()
            _ts.append(time.perf_counter() - _t0)
        return min(_ts)

    _a = torch.randn(16 * 2**20, device=_dev)
    _bw = 2 * _a.numel() * 4 / _best(lambda: _a.clone())
    _s = 1024 if _dev == "cpu" else 2048
    _mm = torch.randn(_s, _s, device=_dev)
    _peak = 2 * _s**3 / _best(lambda: _mm @ _mm)
    torch.manual_seed(0)
    _m = GPT(65, block_size=128, n_layers=4, n_heads=6, d_model=192).to(_dev).eval()
    _n = sum(p.numel() for p in _m.parameters())
    batch_rows = []
    for _B in (1, 4, 16, 64):
        _prompt = torch.randint(0, 65, (_B, 8), device=_dev)
        _t = _best(lambda: generate_cached(_m, _prompt, 64, temperature=0.0))
        batch_rows.append((_B, _B * 64 / _t, ex.decode_tokens_per_second(_n, _B, bytes_per_param=4, bandwidth=_bw,
                                                                          peak_flops=_peak)))
    _fig, _ax = plt.subplots(figsize=(5.5, 3.2))
    _ax.loglog([r[0] for r in batch_rows], [r[1] for r in batch_rows], "o-", label="measured")
    _ax.loglog([r[0] for r in batch_rows], [r[2] for r in batch_rows], "s--", label="your roofline model")
    _ax.set(xlabel="batch size", ylabel="tokens per second", title=f"model M decoding on {_dev}")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(f"measured here: bandwidth {_bw / 1e9:.0f} GB/s, peak {_peak / 1e9:.0f} GFLOP/s; model M "
                           f"{_n:,} parameters ({4 * _n / 1e6:.1f} MB in float32).\n\n| batch | measured tokens/s | "
                           "roofline tokens/s | measured / roofline |\n|---|---|---|---|\n"
                           + "\n".join(f"| {b} | {m:,.0f} | {r:,.0f} | {m / r:.3f} |" for b, m, r in batch_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** throughput grows almost in proportion to the batch, as the lesson says,
    yet the roofline model overestimates the batch-1 speed by a large factor. which regime of
    topic 01 is a 7 MB model in, and what would you have to measure to model it? for a 7B model
    on the same machine, which regime would decoding be in?
    """)
    return


if __name__ == "__main__":
    app.run()
