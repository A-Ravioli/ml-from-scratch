import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import checks, datasets
    from mlfs.diffusion import (  # yours, from this chapter
        DiffusionSchedule,
        TimeUNet,
        alpha_bar_at,
        cfg_model,
        ddim_sample,
        ddpm_loss,
        drop_labels,
        flow_matching_loss,
        sample_flow,
    )
    from mlfs.generative import cached_feature_net, fid_from_features  # yours, from ch 21 · topic 01
    from mlfs.torchnn import count_parameters, ema_update, get_device  # yours, from ch 15 · topic 07

    OBJECTIVES = ("ddpm", "flow")
    N_CLASSES = 10
    # full config: about an hour on an Apple M2 GPU shared with other jobs (see brief.md); small: offline, tests, verify
    FULL = dict(base_channels=32, channel_mults=(1, 2, 2), dropout=0.1, n_iters=2500, batch_size=128, lr=5e-4,
                warmup=100, ema_decay=0.999, p_uncond=0.1, hflip=True, n_train_steps=1000, clip=True)
    SMALL = dict(base_channels=8, channel_mults=(1, 2), dropout=0.0, n_iters=40, batch_size=32, lr=1e-3,
                 warmup=5, ema_decay=0.9, p_uncond=0.1, hflip=True, n_train_steps=100, clip=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 22 · capstone: a class-conditional generator for CIFAR-10 (reference solution)

    read [`brief.md`](brief.md) first. you'll train the same class-conditional U-Net twice
    on CIFAR-10, once as a DDPM noise predictor and once as a flow-matching velocity field,
    with the same data, budget and label dropout, and compare them under classifier-free
    guidance at equal numbers of network calls. the checks in `test_capstone.py` pin down the
    interface; the design is yours.

    | stage | you build |
    |---|---|
    | 1 | `prepare_cifar`, `to_unit`, `random_hflip`: the data in model space and back |
    | 2 | `build_model`, `make_schedule`: one network for both objectives |
    | 3 | `train_generator`: label dropout, warmup, EMA; the same budget for both objectives |
    | 4 | `ClipPrediction`, `generate`: classifier-free guidance at a given number of network calls |
    | 5 | `evaluate`: course FID and class accuracy of the samples |
    """)
    return


@app.function
def prepare_cifar(X, y):
    """stage 1: uint8 images (N, 32, 32, 3) NHWC -> (float32 tensor (N, 3, 32, 32) in [-1, 1] = X / 127.5 - 1, int64
    labels (N,))."""
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1
    return x.contiguous(), torch.as_tensor(np.asarray(y), dtype=torch.int64)


@app.function
def to_unit(x):
    """stage 1: model-space images in [-1, 1] -> [0, 1], clamped, float32 on the CPU (for plots and FeatureNet)."""
    return ((x.detach().float().cpu() + 1) / 2).clamp(0, 1)


@app.function
def random_hflip(x, *, generator=None):
    """stage 1: flip each image (B, C, H, W) left-right where torch.rand(B, generator) < 0.5; returns a new tensor."""
    mask = (torch.rand(x.shape[0], generator=generator) < 0.5).to(x.device)
    return torch.where(mask.view(-1, 1, 1, 1), x.flip(-1), x)


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", prepare_cifar=prepare_cifar, to_unit=to_unit, random_hflip=random_hflip)
    return


@app.function
def build_model(config, *, seed=0):
    """stage 2: torch.manual_seed(seed), then TimeUNet(3, base_channels, channel_mults, n_classes=10, dropout) from
    config."""
    torch.manual_seed(seed)
    return TimeUNet(3, base_channels=config["base_channels"], channel_mults=tuple(config["channel_mults"]),
                    n_classes=N_CLASSES, dropout=config["dropout"])


@app.function
def make_schedule(config):
    """stage 2: the DDPM schedule: DiffusionSchedule(config['n_train_steps']) with the linear betas rescaled to keep the
    default endpoints' total noise: beta_start = 1e-4 * 1000 / n, beta_end = 0.02 * 1000 / n."""
    n = config["n_train_steps"]
    return DiffusionSchedule(n, beta_start=1e-4 * 1000 / n, beta_end=0.02 * 1000 / n)


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", build_model=build_model, make_schedule=make_schedule)
    return


@app.function
def train_generator(objective, model, X, y, config, *, device="cpu", seed=0):
    """stage 3: objective 'ddpm' (ddpm_loss on make_schedule(config)) or 'flow' (flow_matching_loss), ValueError
    otherwise. Adam(config['lr']) with a linear warmup from lr / warmup over config['warmup'] steps; all randomness
    from g = torch.Generator().manual_seed(seed): per step idx = randint(N, (batch_size,), g), random_hflip(..., g) if
    config['hflip'], labels drop_labels(y[idx], p_uncond, 10, g), then the loss (generator=g). an EMA copy of the model
    (copy.deepcopy before training) is updated after optimizer step k (k = 0, 1, ...) with ema_update(ema, model,
    min(ema_decay, (1 + k) / (10 + k))), a warmed-up decay so the EMA forgets the initialization. returns
    dict(ema = the EMA model in eval mode, losses, lrs (per step), seconds)."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    model.to(device).train()
    ema = copy.deepcopy(model).eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.int64)
    schedule = make_schedule(config) if objective == "ddpm" else None
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    losses, lrs = [], []
    t0 = time.perf_counter()
    for step in range(config["n_iters"]):
        lr = config["lr"] * min(1.0, (step + 1) / max(config["warmup"], 1))
        for group in opt.param_groups:
            group["lr"] = lr
        idx = torch.randint(len(X), (config["batch_size"],), generator=g)
        xb = X[idx]
        if config["hflip"]:
            xb = random_hflip(xb, generator=g)
        labels = drop_labels(y[idx], config["p_uncond"], N_CLASSES, generator=g)
        xb, labels = xb.to(device), labels.to(device)
        if objective == "ddpm":
            loss = ddpm_loss(model, schedule, xb, y=labels, generator=g)
        else:
            loss = flow_matching_loss(model, xb, y=labels, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema_update(ema, model, min(config["ema_decay"], (1 + step) / (10 + step)))
        losses.append(loss.item())
        lrs.append(lr)
    return dict(ema=ema.eval(), losses=losses, lrs=lrs, seconds=time.perf_counter() - t0)


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", train_generator=train_generator, build_model=build_model,
               random_hflip=random_hflip)
    return


@app.class_definition
class ClipPrediction(torch.nn.Module):
    """a model wrapper whose implied clean-image prediction is clipped to [-1, 1] (Ho et al.'s sampler does this)."""

    def __init__(self, model, objective, schedule=None):
        """stage 4: keep the model as a submodule, the objective and (for 'ddpm') the schedule."""
        super().__init__()
        self.model, self.objective, self.schedule = model, objective, schedule

    def forward(self, x, t, y=None):
        """stage 4: 'ddpm': x0 = clamp((x - sqrt(1 - ab) eps) / sqrt(ab), -1, 1) with ab = alpha_bar_at(schedule, t),
        returns (x - sqrt(ab) x0) / sqrt(1 - ab); 'flow': x1 = clamp(x + (1 - t) v, -1, 1), returns (x1 - x) / (1 - t)."""
        out = self.model(x, t, y)
        shape = (-1,) + (1,) * (x.dim() - 1)
        if self.objective == "ddpm":
            ab = alpha_bar_at(self.schedule, t).view(shape)
            x0 = ((x - (1 - ab).sqrt() * out) / ab.sqrt()).clamp(-1, 1)
            return (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        tt = t.view(shape)
        x1 = (x + (1 - tt) * out).clamp(-1, 1)
        return (x1 - x) / (1 - tt)


@app.function
def generate(objective, model, labels, config, *, guidance_scale=1.0, n_steps=50, device="cpu", generator=None,
             batch_size=500):
    """stage 4: images in [0, 1] (CPU float32, (N, 3, H, W)) for the given labels (N,), in chunks of batch_size, from
    cfg_model(model, guidance_scale, 10), wrapped in ClipPrediction if config['clip']: 'ddpm' with ddim_sample(eta=0,
    n_steps, schedule make_schedule(config)), 'flow' with sample_flow(method='euler', n_steps): n_steps network calls
    per chunk either way (each on 2 x chunk rows, for guidance). image size config.get('img_size', 32)."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    size = config.get("img_size", 32)
    schedule = make_schedule(config) if objective == "ddpm" else None
    guided = cfg_model(model, guidance_scale, N_CLASSES)
    if config.get("clip", False):
        guided = ClipPrediction(guided, objective, schedule)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    out = []
    for i in range(0, len(labels), batch_size):
        yb = labels[i:i + batch_size].to(device)
        shape = (len(yb), 3, size, size)
        if objective == "ddpm":
            x = ddim_sample(guided, schedule, shape, n_steps=n_steps, y=yb, generator=generator, device=device)
        else:
            x = sample_flow(guided, shape, n_steps=n_steps, method="euler", y=yb, generator=generator, device=device)
        out.append(to_unit(x))
    return torch.cat(out)


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", generate=generate, ClipPrediction=ClipPrediction,
               train_generator=train_generator, build_model=build_model, make_schedule=make_schedule)
    return


@app.function
def evaluate(images, labels, featnet, real_feats):
    """stage 5: dict(fid = fid_from_features(real_feats, featnet.embed(images)), accuracy = fraction of images whose
    featnet(images).argmax(1) equals their label (eval mode, no gradients))."""
    feats = featnet.embed(images)
    with torch.no_grad():
        was = featnet.training
        featnet.eval()
        preds = torch.cat([featnet(images[i:i + 500]).argmax(1).cpu() for i in range(0, len(images), 500)])
        featnet.train(was)
    acc = float((preds == torch.as_tensor(labels)).float().mean())
    return dict(fid=float(fid_from_features(real_feats, feats)), accuracy=acc)


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", evaluate=evaluate)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the experiment

    CIFAR-10's 50,000 training images in $[-1, 1]$; the feature network for FID and class
    accuracy is chapter 21's `FeatureNet`, trained once on the training images
    (`cached_feature_net(..., name="cifar10", epochs=5)`), so **these FIDs are course-internal**:
    comparable within this table, not with published CIFAR-10 FIDs (which use an Inception
    network). both objectives train the same `TimeUNet` from the same initialization with the
    same budget; then every (objective, guidance scale, number of network calls) cell draws
    1,000 samples (100 per class). the main setting is drawn twice to show the sampling noise,
    and once each with the raw (non-EMA) weights and without clipping the predicted image.
    set `MLFS_CAPSTONE_FULL=0` to run the small config on real data while you develop.
    """)
    return


@app.cell
def _():
    OFFLINE = datasets.is_offline()
    RUN_FULL = not OFFLINE and os.environ.get("MLFS_CAPSTONE_FULL", "1") != "0"
    CONFIG = FULL if RUN_FULL else SMALL
    DEVICE = get_device() if RUN_FULL else torch.device("cpu")
    _d = datasets.cifar10()
    _n_train = len(_d["X_train"]) if RUN_FULL else min(2000, len(_d["X_train"]))
    X_train, y_train = prepare_cifar(_d["X_train"][:_n_train], _d["y_train"][:_n_train])
    X_test, y_test = prepare_cifar(_d["X_test"], _d["y_test"])
    N_PER_CLASS = 100 if RUN_FULL else 6
    _t0 = time.perf_counter()
    featnet = cached_feature_net(to_unit(X_train), y_train, name="cifar10", epochs=5 if RUN_FULL else 1, seed=0,
                                 device=DEVICE).to("cpu")
    real_feats = featnet.embed(to_unit(X_test))
    _acc = evaluate(to_unit(X_test[:2000]), y_test[:2000], featnet, real_feats)["accuracy"]
    _ref = evaluate(to_unit(X_train[:N_PER_CLASS * 10]), y_train[:N_PER_CLASS * 10], featnet, real_feats)
    mo.md(f"{'**full config**' if RUN_FULL else '**small config**'} · {len(X_train):,} training images · device "
          f"**{DEVICE}** · feature network: test accuracy **{_acc:.3f}** ({time.perf_counter() - _t0:.0f} s incl. "
          f"training if not cached) · {N_PER_CLASS * 10:,} real training images score FID **{_ref['fid']:.2f}** and "
          f"accuracy {_ref['accuracy']:.3f} (the floor and ceiling at this sample size)")
    return CONFIG, DEVICE, N_PER_CLASS, RUN_FULL, X_train, featnet, real_feats, y_train


@app.cell
def _(CONFIG, DEVICE, X_train, y_train):
    runs = {}
    for _obj in OBJECTIVES:
        _m = build_model(CONFIG, seed=0)
        runs[_obj] = train_generator(_obj, _m, X_train, y_train, CONFIG, device=DEVICE, seed=0)
        runs[_obj]["raw"] = _m.eval()  # the trained weights themselves, for the EMA ablation
        print(f"{_obj}: {CONFIG['n_iters']} steps in {runs[_obj]['seconds'] / 60:.1f} min, final loss "
              f"{np.mean(runs[_obj]['losses'][-100:]):.4f}", flush=True)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    for _obj, _r in runs.items():
        _l = np.array(_r["losses"])
        _k = max(1, len(_l) // 50)
        _ax.plot(np.convolve(_l, np.ones(_k) / _k, mode="valid"), label=_obj)
    _ax.set(xlabel="step", ylabel="training loss (smoothed)", yscale="log")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"parameters per model: {count_parameters(runs['ddpm']['ema']):,} · training: "
                           + ", ".join(f"{o} {r['seconds'] / 60:.1f} min" for o, r in runs.items()))])
    return (runs,)


@app.cell
def _(CONFIG, DEVICE, N_PER_CLASS, RUN_FULL, featnet, real_feats, runs):
    labels = torch.arange(N_CLASSES).repeat_interleave(N_PER_CLASS)
    # two sweeps through the same main setting (w = 2, 25 calls): network calls at w = 2, guidance at 25 calls
    _calls, _scales, _main = ((5, 10, 25), (1.0, 2.0, 4.0), (2.0, 25)) if RUN_FULL else ((4, 8), (1.0, 3.0), (3.0, 8))
    _grid = sorted({(_main[0], _n) for _n in _calls} | {(_w, _main[1]) for _w in _scales}, key=lambda k: (k[1], k[0]))
    results, _rows, samples = {}, [], {}
    for _obj in OBJECTIVES:
        for _w, _n in _grid:
            _t0 = time.perf_counter()
            _img = generate(_obj, runs[_obj]["ema"], labels, CONFIG, guidance_scale=_w, n_steps=_n, device=DEVICE,
                            generator=torch.Generator().manual_seed(1))
            _secs = time.perf_counter() - _t0
            results[(_obj, _w, _n)] = _e = evaluate(_img, labels, featnet, real_feats)
            samples[(_obj, _w, _n)] = _img
            _rows.append(f"| {_obj} | {_w:g} | {_n} | {_e['fid']:.2f} | {_e['accuracy']:.3f} | {_secs:.0f} s |")
        # a second sample draw of the main setting: the sampling noise of the table
        _img = generate(_obj, runs[_obj]["ema"], labels, CONFIG, guidance_scale=_main[0], n_steps=_main[1], device=DEVICE,
                        generator=torch.Generator().manual_seed(2))
        results[(_obj, "draw2")] = _e = evaluate(_img, labels, featnet, real_feats)
        _rows.append(f"| {_obj} (second draw) | {_main[0]:g} | {_main[1]} | {_e['fid']:.2f} | {_e['accuracy']:.3f} | |")
        # ablations of the main setting: the raw (non-EMA) weights, and no clipping of the predicted clean image
        for _tag, _model, _cfg in (("raw weights", runs[_obj]["raw"], CONFIG),
                                   ("no clipping", runs[_obj]["ema"], {**CONFIG, "clip": False})):
            _img = generate(_obj, _model, labels, _cfg, guidance_scale=_main[0], n_steps=_main[1], device=DEVICE,
                            generator=torch.Generator().manual_seed(1))
            results[(_obj, _tag)] = _e = evaluate(_img, labels, featnet, real_feats)
            _rows.append(f"| {_obj} ({_tag}) | {_main[0]:g} | {_main[1]} | {_e['fid']:.2f} | {_e['accuracy']:.3f} | |")
    mo.md(f"{len(labels):,} samples per cell ({N_PER_CLASS} per class) · FID against the 10,000 test images in the course "
          "feature space · accuracy = the feature network's agreement with the requested label\n\n"
          "| objective | guidance w | network calls | FID | class accuracy | sampling time |\n|---|---|---|---|---|---|\n"
          + "\n".join(_rows))
    return labels, results, samples


@app.cell
def _(results, samples):
    _n = max({k[2] for k in samples}, key=lambda n: sum(1 for k in samples if k[2] == n))  # the guidance sweep's calls
    _ws = sorted(k[1] for k in samples if k[2] == _n and k[0] == "ddpm")
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 4.6))
    for _r, _obj in enumerate(OBJECTIVES):
        for _c, _w in enumerate((_ws[0], _ws[-1])):
            _img = samples[(_obj, _w, _n)]
            _per = len(_img) // N_CLASSES
            _rows_img = [torch.cat([_img[k * _per + j] for k in range(N_CLASSES)], dim=2) for j in range(min(4, _per))]
            _axes[_r, _c].imshow(torch.cat(_rows_img, dim=1).permute(1, 2, 0).numpy())
            _axes[_r, _c].set_axis_off()
            _axes[_r, _c].set_title(f"{_obj}, w = {_w:g}, {_n} calls (columns: classes 0-9)", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)):

    1. **the table**: FID and class accuracy for both objectives across guidance scales and
       numbers of network calls, with the sampling noise (the second draw), the real-data
       reference row and the hardware. say what the course FID can and cannot be compared
       with.
    2. **DDPM against flow matching at equal compute**: which wins at few calls, which at many,
       and why (the curvature of each ODE, topic 04's DDIM-as-Euler and topic 05's straight
       paths); is the difference larger than the sampling noise?
    3. **the guidance tradeoff**: how FID and accuracy move with $w$, where the best FID is,
       and what you see in the samples at large $w$.
    4. **what you'd change** with ten times the compute, and which conclusions you expect to
       survive a second training seed.
    """)
    return


if __name__ == "__main__":
    app.run()
