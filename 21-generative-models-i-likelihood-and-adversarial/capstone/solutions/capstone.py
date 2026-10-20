import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, datasets
    from mlfs.generative import (  # yours, from this chapter
        PixelCNN,
        RealNVP,
        VAE,
        DCGANDiscriminator,
        DCGANGenerator,
        cached_feature_net,
        fid_from_features,
        flow_bits_per_dim,
        gan_step,
        iwae_nll,
        logit_transform,
        dequantize,
        pixelcnn_nll,
        precision_recall_knn,
    )
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.torchnn import count_parameters, get_device  # yours, from ch 15 · topic 07

    FAMILIES = ("autoregressive", "vae", "gan", "flow")
    # full config: about an hour on an Apple M2 GPU (see brief.md); small config: offline, tests, verify
    FULL = {
        "autoregressive": dict(hidden_channels=64, n_layers=7, epochs=8, batch_size=128, lr=1e-3),
        "vae": dict(latent_dim=32, hidden_dims=(512, 256), epochs=40, batch_size=128, lr=1e-3, iwae_k=500),
        "gan": dict(latent_dim=64, base_channels=64, epochs=15, batch_size=128, lr=2e-4),
        "flow": dict(n_layers=8, hidden_dim=1024, epochs=40, batch_size=256, lr=5e-4, alpha=0.05),
    }
    SMALL = {
        "autoregressive": dict(hidden_channels=16, n_layers=3, epochs=2, batch_size=64, lr=2e-3),
        "vae": dict(latent_dim=8, hidden_dims=(128,), epochs=4, batch_size=64, lr=2e-3, iwae_k=20),
        "gan": dict(latent_dim=16, base_channels=16, epochs=3, batch_size=64, lr=5e-4),
        "flow": dict(n_layers=4, hidden_dim=128, epochs=4, batch_size=64, lr=1e-3, alpha=0.05),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 21 · capstone: four families, one dataset (reference solution)

    read [`brief.md`](brief.md) first. you'll train one model from each family of the
    chapter (an autoregressive PixelCNN, a VAE, a DCGAN and a RealNVP flow) on the same
    binarized MNIST, and compare them on likelihood, sample quality, diversity and cost,
    with honest uncertainty. the checks in `test_capstone.py` pin down the interface; the
    design is yours.

    | stage | you build |
    |---|---|
    | 1 | `binarize`: the shared dataset |
    | 2 | `build_family`, `train_family`: one model per family, one training function |
    | 3 | `nll_bits_per_dim`: exact or bounded test likelihoods on a common scale |
    | 4 | `sample_binary`: samples in the data's own space, from every family |
    | 5 | `evaluate_samples`, `bpd_ci`: FID, precision/recall, label entropy, bootstrap intervals |
    """)
    return


@app.function
def binarize(X, *, seed=0):
    """stage 1: uint8 images (N, H, W) -> float32 tensor (N, 1, H, W) of 0/1, each pixel ~ Bernoulli(X / 255), drawn
    with torch.bernoulli(..., generator=torch.Generator().manual_seed(seed))."""
    p = torch.tensor(np.asarray(X), dtype=torch.float32)[:, None] / 255.0
    return torch.bernoulli(p, generator=torch.Generator().manual_seed(seed))


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", binarize=binarize)
    return


@app.function
def build_family(family, config, *, img_size=28, seed=0):
    """stage 2: torch.manual_seed(seed), then 'autoregressive': PixelCNN(1, n_levels=2, ...); 'vae': VAE(img_size**2,
    latent_dim, hidden_dims, likelihood='bernoulli'); 'gan': nn.ModuleDict(G=DCGANGenerator, D=DCGANDiscriminator) for
    1 channel; 'flow': RealNVP(img_size**2, n_layers, hidden_dim); ValueError otherwise."""
    torch.manual_seed(seed)
    if family == "autoregressive":
        return PixelCNN(1, n_levels=2, hidden_channels=config["hidden_channels"], n_layers=config["n_layers"])
    if family == "vae":
        return VAE(img_size * img_size, config["latent_dim"], hidden_dims=tuple(config["hidden_dims"]),
                   likelihood="bernoulli")
    if family == "gan":
        return nn.ModuleDict({
            "G": DCGANGenerator(config["latent_dim"], 1, base_channels=config["base_channels"], img_size=img_size),
            "D": DCGANDiscriminator(1, base_channels=config["base_channels"], img_size=img_size)})
    if family == "flow":
        return RealNVP(img_size * img_size, n_layers=config["n_layers"], hidden_dim=config["hidden_dim"])
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


@app.function
def flow_inputs(x, alpha, generator=None):
    """binary images -> (y flattened logit-space inputs, log_det of the logit transform), dequantized with 2 levels."""
    y, ld = logit_transform(dequantize(x.long(), n_levels=2, generator=generator), alpha=alpha)
    return y.flatten(1), ld


@app.function
def train_family(family, model, X, config, *, device="cpu", seed=0):
    """stage 2: train `model` on binary images X (N, 1, H, W) for config['epochs'] epochs of shuffled minibatches
    (torch.Generator seeded with `seed`), Adam(config['lr']); returns dict(seconds, losses = per-epoch mean loss)."""
    model.to(device).train()
    g = torch.Generator().manual_seed(seed)
    bs, n = config["batch_size"], len(X)
    d = X[0].numel()
    if family == "gan":
        opt_g = torch.optim.Adam(model["G"].parameters(), lr=config["lr"], betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(model["D"].parameters(), lr=config["lr"], betas=(0.5, 0.999))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    losses, t0 = [], time.perf_counter()
    for _ in range(config["epochs"]):
        perm, total, count = torch.randperm(n, generator=g), 0.0, 0
        for i in range(0, n - bs + 1, bs):
            xb = X[perm[i:i + bs]].to(device)
            if family == "gan":
                loss, _ = gan_step(model["G"], model["D"], opt_g, opt_d, xb * 2 - 1, latent_dim=config["latent_dim"],
                                   generator=g)
            else:
                if family == "autoregressive":
                    loss_t = pixelcnn_nll(model, xb).mean() / d
                elif family == "vae":
                    loss_t = model.loss(xb.flatten(1))[0] / d
                else:
                    y, ld = flow_inputs(xb.cpu(), config["alpha"], g)
                    loss_t = -(model.log_prob(y.to(device)) + ld.to(device)).mean() / d
                opt.zero_grad()
                loss_t.backward()
                if family == "flow":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                opt.step()
                loss = loss_t.item()
            total, count = total + loss, count + 1
        losses.append(total / max(count, 1))
    if device != "cpu" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    model.eval()
    return {"seconds": time.perf_counter() - t0, "losses": losses}


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", build_family=build_family, train_family=train_family, binarize=binarize)
    return


@app.function
def nll_bits_per_dim(family, model, X, config, *, device="cpu", generator=None, batch_size=500):
    """stage 3: per-image test bits/dim, float64 ndarray (N,): 'autoregressive' exact (pixelcnn_nll), 'vae' the
    importance-weighted bound (iwae_nll, k = config['iwae_k']), 'flow' the dequantized bound (flow_bits_per_dim,
    n_levels=2, alpha); None for 'gan' (no likelihood). eval mode, no gradients, in batches."""
    if family == "gan":
        return None
    model.to(device).eval()
    d = X[0].numel()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            if family == "autoregressive":
                nll = pixelcnn_nll(model, xb.to(device)).cpu()
            elif family == "vae":
                nll = iwae_nll(model, xb.flatten(1).to(device), k=config["iwae_k"], generator=generator).cpu()
            else:
                bpd = flow_bits_per_dim(model.cpu(), xb.long().flatten(1), n_levels=2, alpha=config["alpha"],
                                        generator=generator)
                out.append(bpd.double())
                continue
            out.append(nll.double() / (d * math.log(2)))
    model.to(device)
    return torch.cat(out).numpy()


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", nll_bits_per_dim=nll_bits_per_dim, build_family=build_family,
               train_family=train_family, binarize=binarize)
    return


@app.function
def sample_binary(family, model, n, config, *, img_size=28, device="cpu", generator=None):
    """stage 4: n binary samples, float32 CPU tensor (n, 1, img_size, img_size) of 0/1: PixelCNN samples directly; VAE
    and GAN outputs treated as pixel probabilities and drawn with torch.bernoulli(generator); flow samples mapped back
    through the inverse logit transform and quantized (pixel = 1 where the dequantized value >= 1/2)."""
    model.to(device).eval()
    with torch.no_grad():
        if family == "autoregressive":
            return model.sample(n, img_size, generator=generator, device=device).cpu()
        if family == "vae":
            p = model.sample(n, generator=generator).cpu().view(n, 1, img_size, img_size)
        elif family == "gan":
            z = torch.randn((n, config["latent_dim"]), generator=generator).to(device)
            p = ((model["G"](z) + 1) / 2).clamp(0, 1).cpu()
        elif family == "flow":
            y = model.sample(n, generator=generator).cpu()
            u = (torch.sigmoid(y) - config["alpha"]) / (1 - 2 * config["alpha"])
            return (u >= 0.5).float().view(n, 1, img_size, img_size)
        else:
            raise ValueError(f"unknown family {family!r}")
        return torch.bernoulli(p, generator=generator)


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", sample_binary=sample_binary, build_family=build_family,
               train_family=train_family, binarize=binarize)
    return


@app.function
def evaluate_samples(samples, real_feats, featnet, *, k=3):
    """stage 5: dict(fid, precision, recall, class_entropy): featnet.embed(samples) against real_feats with
    fid_from_features and precision_recall_knn(real_feats, fake, k=k); class_entropy = entropy (nats) of the
    histogram of featnet's predicted labels (argmax of featnet(samples))."""
    fake = featnet.embed(samples)
    p, r = precision_recall_knn(real_feats, fake, k=k)
    with torch.no_grad():
        labels = featnet(samples.to(next(featnet.parameters()).device)).argmax(1).cpu().numpy()
    hist = np.bincount(labels) / len(labels)
    hist = hist[hist > 0]
    return {"fid": fid_from_features(real_feats, fake), "precision": p, "recall": r,
            "class_entropy": float(-(hist * np.log(hist)).sum())}


@app.function
def bpd_ci(bpd, *, n_boot=2000, alpha=0.05, seed=0):
    """stage 5: (mean, lo, hi): the mean of per-image bits/dim and its percentile bootstrap interval (bootstrap_ci)."""
    bpd = np.asarray(bpd, dtype=np.float64)
    lo, hi = bootstrap_ci(bpd, np.mean, n_boot=n_boot, alpha=alpha, rng=seed)
    return float(bpd.mean()), float(lo), float(hi)


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", evaluate_samples=evaluate_samples, bpd_ci=bpd_ci)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the experiment

    binarized MNIST: the 60,000 training digits binarized with seed 0, the 10,000 test
    digits with seed 1. the feature network for FID and precision/recall is topic 01's
    `FeatureNet`, trained on the *binarized* training digits (so the features see the same
    kind of images the models produce). each family is built, trained, timed, scored on
    the test set (bits/dim with a bootstrap interval), sampled (2,000 images, twice, timed) and
    evaluated. set `MLFS_CAPSTONE_FULL=0` to run the small config on real data while you
    develop.
    """)
    return


@app.cell
def _():
    OFFLINE = datasets.is_offline()
    RUN_FULL = not OFFLINE and os.environ.get("MLFS_CAPSTONE_FULL", "1") != "0"
    CONFIG = FULL if RUN_FULL else SMALL
    DEVICE = get_device() if RUN_FULL else torch.device("cpu")
    _d = datasets.mnist()
    _n_train = len(_d["X_train"]) if RUN_FULL else min(2000, len(_d["X_train"]))
    X_train = binarize(_d["X_train"][:_n_train], seed=0)
    y_train = _d["y_train"][:_n_train]
    X_test = binarize(_d["X_test"] if RUN_FULL else _d["X_test"][:400], seed=1)
    N_SAMPLES = 2000 if RUN_FULL else 300
    featnet = cached_feature_net(X_train, y_train, name="mnist-binary", epochs=2, seed=0, device=DEVICE).to("cpu")
    real_feats = featnet.embed(X_test)
    _acc = float((featnet(X_test[:2000]).argmax(1).numpy() == (_d["y_test"] if RUN_FULL else _d["y_test"][:400])[:2000]).mean())
    mo.md(f"{'**full config**' if RUN_FULL else '**small config**'} · {len(X_train):,} training and {len(X_test):,} test "
          f"images · device **{DEVICE}** · feature network accuracy on binarized test digits **{_acc:.3f}**")
    return CONFIG, DEVICE, N_SAMPLES, X_test, X_train, featnet, real_feats


@app.cell
def _(CONFIG, DEVICE, N_SAMPLES, X_test, X_train, featnet, real_feats):
    results = {}
    for _fam in FAMILIES:
        _cfg = CONFIG[_fam]
        _m = build_family(_fam, _cfg, seed=0)
        _train = train_family(_fam, _m, X_train, _cfg, device=DEVICE, seed=0)
        _t0 = time.perf_counter()
        _bpd = nll_bits_per_dim(_fam, _m, X_test, _cfg, device=DEVICE, generator=torch.Generator().manual_seed(1))
        _t_nll = time.perf_counter() - _t0
        sample_binary(_fam, _m, 4, _cfg, device=DEVICE, generator=torch.Generator().manual_seed(9))  # warm-up
        _t0 = time.perf_counter()
        _s = sample_binary(_fam, _m, N_SAMPLES, _cfg, device=DEVICE, generator=torch.Generator().manual_seed(2))
        _t_sample = time.perf_counter() - _t0
        _ev = evaluate_samples(_s, real_feats, featnet)
        _s2 = sample_binary(_fam, _m, N_SAMPLES, _cfg, device=DEVICE, generator=torch.Generator().manual_seed(3))
        _ev2 = evaluate_samples(_s2, real_feats, featnet)
        print(f"{_fam}: trained in {_train['seconds'] / 60:.1f} min, sampled in {_t_sample:.0f} s", flush=True)
        results[_fam] = dict(model=_m, train=_train, bpd=_bpd, t_nll=_t_nll, t_sample=_t_sample, samples=_s,
                             eval=_ev, eval2=_ev2, n_params=count_parameters(_m))
    _ref = evaluate_samples(X_train[:N_SAMPLES], real_feats, featnet)
    _rows = []
    for _fam, _r in results.items():
        _b = "—" if _r["bpd"] is None else "{:.4f} [{:.4f}, {:.4f}]".format(*bpd_ci(_r["bpd"]))
        _e, _e2 = _r["eval"], _r["eval2"]
        _rows.append(f"| {_fam} | {_b} | {_e['fid']:.1f} / {_e2['fid']:.1f} | {_e['precision']:.3f} | {_e['recall']:.3f} | "
                     f"{_e['class_entropy']:.3f} | {_r['n_params']:,} | {_r['train']['seconds'] / 60:.1f} | "
                     f"{1e3 * _r['t_sample'] / N_SAMPLES:.2f} |")
    _rows.append(f"| real training images | — | {_ref['fid']:.1f} | {_ref['precision']:.3f} | {_ref['recall']:.3f} | "
                 f"{_ref['class_entropy']:.3f} | | | |")
    _fig, _axes = plt.subplots(len(results), 1, figsize=(9, 1.3 * len(results)))
    for _ax, (_fam, _r) in zip(np.atleast_1d(_axes), results.items()):
        _k = min(16, len(_r["samples"]))
        _grid = _r["samples"][:_k, 0].permute(1, 0, 2).reshape(_r["samples"].shape[-2], -1)
        _ax.imshow(_grid, cmap="gray_r", vmin=0, vmax=1)
        _ax.set_ylabel(_fam, fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"bits/dim: mean over test images with 95% bootstrap CI (the VAE and flow values are upper bounds) · FID of "
        f"{N_SAMPLES:,} samples against the test set, two independent sample draws · precision/recall ($k = 3$) and label "
        "entropy (uniform over 10 classes: 2.303) from the first draw\n\n"
        "| family | test bits/dim | FID (draw 1 / 2) | precision | recall | label entropy | parameters | training "
        "(min) | sampling (ms / image) |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (results,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)):

    1. **the table**, with what each column can and cannot tell you: which likelihoods are
       exact and which are bounds, why the GAN has none, what the FID's two draws say about
       its noise, and why course FIDs aren't comparable with papers'.
    2. **ranking the families**: on likelihood, on sample quality, on diversity, on cost.
       do the rankings agree? where they disagree, explain the mechanism (mode-covering
       maximum likelihood vs adversarial training, bounds vs exact likelihoods, sequential
       vs one-pass sampling).
    3. **look at the samples**: one failure per family that the numbers miss, or that a
       number reveals before your eyes do.
    4. **what you'd change** with ten times the compute, for each family, and which
       conclusions you expect to survive.
    """)
    return


if __name__ == "__main__":
    app.run()
