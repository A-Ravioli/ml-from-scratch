import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import contextlib
    import os
    import time
    import types

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, datasets
    from mlfs.systems import GradScaler  # yours, from topic 03
    from mlfs.text import CharTokenizer  # yours, from ch 17 · 02
    from mlfs.torchnn import get_device  # yours, from ch 15 · 07
    from mlfs.train import CosineSchedule  # yours, from ch 15 · 04
    from mlfs.transformer import GPT, estimate_loss, get_batch, train_lm  # yours, from ch 18 · 04

    plt.rcParams["figure.autolayout"] = True

    # small CPU config (offline, tests, verify, or no GPU) vs full config (tiny Shakespeare on a GPU: cuda or mps).
    # set MLFS_CAPSTONE=small to force the small config. on a CPU the full config would take hours (and float16
    # autocast on a CPU is many times slower than float32), so a CPU-only machine runs the small one on the real data.
    OFFLINE = datasets.is_offline()
    FULL = (not OFFLINE and str(get_device()) != "cpu" and os.environ.get("MLFS_CAPSTONE", "full") != "small")
    if FULL:
        DEVICE = str(get_device())
        CONFIG = dict(model=dict(d_model=192, n_layers=4, n_heads=6, dropout=0.1), block_size=128, batch_size=64,
                      base_steps=2000, base_lr=1e-3, warmup=100, eval_interval=250, eval_batches=20,
                      fast_lr=3e-3, fast_steps=[600, 800, 1000], seeds=[0, 1, 2], time_steps=20, time_repeats=5,
                      compile_backend="inductor", amp=torch.float16, final_eval_batches=50)
    else:
        DEVICE = "cpu"
        CONFIG = dict(model=dict(d_model=32, n_layers=1, n_heads=2, dropout=0.0), block_size=32, batch_size=8,
                      base_steps=60, base_lr=3e-3, warmup=10, eval_interval=30, eval_batches=4,
                      fast_lr=6e-3, fast_steps=[30, 45], seeds=[0, 1], time_steps=3, time_repeats=3,
                      compile_backend=None, amp=None, final_eval_batches=4)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 26 · capstone: make your GPT 5× faster to train

    read [`brief.md`](brief.md) first. chapter 18 trained a GPT with your `train_lm`; here you
    make the same training reach the same validation loss in a fifth of the time (or find out
    exactly why you can't), changing one thing at a time and measuring each change with
    uncertainty. the five functions are the tools; the experiment and the write-up below them
    are the capstone.

    | stage | you build |
    |---|---|
    | 1 | `fused_attention`: swap your attention's score-softmax-weights chain for one fused kernel |
    | 2 | `time_steps`: honest per-step timing (warmup, device sync, repeated windows) |
    | 3 | `make_train_step`: one training step with optional autocast, loss scaling and compilation |
    | 4 | `train_fast`: `train_lm` rebuilt for speed, every optimization behind a switch |
    | 5 | `speedup_ci`: a speedup with a bootstrap interval |

    when the checks are green, scroll down: profile, ablate, race, write up.
    """)
    return


@app.function
def sync_for(device):
    """a function that waits for `device`'s queued work: cuda and mps run asynchronously, the CPU doesn't (given)."""
    device = str(device)
    if device.startswith("cuda"):
        return torch.cuda.synchronize
    if device.startswith("mps"):
        return torch.mps.synchronize
    return lambda: None


@app.function
def fused_attention(model):
    """stage 1: patch every block's self-attention to use F.scaled_dot_product_attention (causal); returns model."""

    raise NotImplementedError("stage1: implement fused_attention")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", fused_attention=fused_attention)
    return


@app.function
def time_steps(step_fn, n_steps, *, warmup=3, repeats=3, sync=None):
    """stage 2: (repeats,) array of mean seconds per step over n_steps consecutive calls, after warmup calls; synced."""
    raise NotImplementedError("stage2: implement time_steps")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", time_steps=time_steps)
    return


@app.function
def make_train_step(model, optimizer, *, amp_dtype=None, scaler=None, grad_clip=1.0, compile_backend=None):
    """stage 3: step(x, y) -> detached loss: forward (compiled, autocast), (scaled) backward, unscale, clip, step."""
    raise NotImplementedError("stage3: implement make_train_step")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", make_train_step=make_train_step)
    return


@app.function
def train_fast(model, train_data, block_size, *, val_data=None, batch_size=64, max_steps=1000, lr=1e-3,
               warmup_steps=100, weight_decay=0.1, grad_clip=1.0, eval_batches=20, device="cpu", seed=0,
               fused_attn=False, amp_dtype=None, compile_backend=None, fused_optimizer=False, scaler=None):
    """stage 4: train_lm's recipe (same batches, schedule, AdamW) with data on device, no mid-run evals, and switches; scaler for fp16 (default: a new mlfs GradScaler)."""
    raise NotImplementedError("stage4: implement train_fast")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", train_fast=train_fast)
    return


@app.function
def speedup_ci(base_times, new_times, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 5: (median(base) / median(new), lo, hi), percentile bootstrap resampling each sample independently."""
    raise NotImplementedError("stage5: implement speedup_ci")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", speedup_ci=speedup_ci)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## the experiment

    config: **{"full" if FULL else "small"}** on **{DEVICE}** (offline: {OFFLINE}). the full config
    takes the better part of an hour on an Apple-silicon laptop GPU (`brief.md` gives the measured
    runtime); the small one, used by the tests, `verify.py` and CPU-only machines, a minute or two. every timing below is from this machine, now: on a
    shared machine, rerun the notebook and compare the spreads before trusting a difference.
    """)
    return


@app.cell
def _():
    _text = datasets.tiny_shakespeare()
    tok = CharTokenizer().train(_text)
    ids = np.asarray(tok.encode(_text), dtype=np.int64)
    _a, _b = int(0.8 * len(ids)), int(0.9 * len(ids))
    train_ids, val_ids = ids[:_a], ids[_a:_b]
    V, T, B = tok.vocab_size, CONFIG["block_size"], CONFIG["batch_size"]
    sync = sync_for(DEVICE)
    # validation split in two: the recipe search picks its winner on the first half, the result is reported on the
    # second, so the reported loss isn't the one the search optimized against
    val_sel, val_rep = val_ids[:len(val_ids) // 2], val_ids[len(val_ids) // 2:]

    def new_model(seed=0, dropout=None):
        torch.manual_seed(seed)
        _cfg = dict(CONFIG["model"], **({} if dropout is None else {"dropout": dropout}))
        return GPT(V, block_size=T, **_cfg)

    mo.md(f"tiny Shakespeare: {len(ids):,} characters, vocabulary {V}; training on the first {len(train_ids):,}, "
          f"validating on the next {len(val_ids):,}. model: {CONFIG['model']}, {sum(p.numel() for p in new_model().parameters()):,} "
          f"parameters, context {T}, batch {B}.")
    return B, T, new_model, sync, train_ids, val_ids, val_rep, val_sel


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1. profile the baseline

    where does a step of chapter 18's trainer spend its time? first the CPU profiler over a few
    steps on the CPU (where it measures compute), then, on the training device, each piece of the
    step timed on its own with synchronization.
    """)
    return


@app.cell
def _(B, T, new_model, sync, train_ids):
    from torch.profiler import ProfilerActivity, profile

    _m = new_model().train()
    _opt = torch.optim.AdamW(_m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    _g = torch.Generator().manual_seed(0)
    _cpu_b = max(4, B // 8)

    def _cpu_step():
        _x, _y = get_batch(train_ids, T, _cpu_b, generator=_g)
        _, _loss = _m(_x, _y)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()

    _cpu_step()
    with profile(activities=[ProfilerActivity.CPU]) as _prof:
        for _ in range(2):
            _cpu_step()
    _rows = sorted(((e.key, e.self_cpu_time_total, e.count) for e in _prof.key_averages()), key=lambda r: -r[1])
    _tot = sum(r[1] for r in _rows)
    _top = "\n".join(f"| `{k}` | {c} | {100 * t / _tot:.1f}% |" for k, t, c in _rows[:10])
    _matmul = 100 * sum(t for k, t, c in _rows if k in ("aten::mm", "aten::bmm", "aten::addmm")) / _tot
    # on the training device: forward, backward and optimizer timed separately
    _md = new_model().to(DEVICE).train()
    _od = torch.optim.AdamW(_md.parameters(), lr=1e-3, betas=(0.9, 0.95))
    _x, _y = get_batch(train_ids, T, B, generator=_g, device=DEVICE)
    _parts = {
        "get_batch (CPU sampling + copy)": lambda: get_batch(train_ids, T, B, generator=_g, device=DEVICE),
        "forward": lambda: _md(_x, _y),
        "forward + backward": lambda: _md(_x, _y)[1].backward(),
        "optimizer step": lambda: _od.step(),
    }
    _times = {k: np.median(time_steps(f, CONFIG["time_steps"], warmup=2, repeats=3, sync=sync)) for k, f in _parts.items()}
    _dev = "\n".join(f"| {k} | {1e3 * v:.1f} ms |" for k, v in _times.items())
    mo.md(f"**CPU profile** (batch {_cpu_b}, 2 steps): matrix products take **{_matmul:.0f}%** of the self time.\n\n"
          "| op | calls | share |\n|---|---|---|\n" + _top +
          f"\n\n**on {DEVICE}** (batch {B}), median of 3 windows:\n\n| piece | time |\n|---|---|\n" + _dev)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2. the baseline run

    chapter 18's `train_lm`, unchanged, with chapter 18's capstone settings for this model:
    AdamW at the base learning rate, warmup then cosine, clipping, and its evaluation every
    `eval_interval` steps. its final validation loss is the **target**.
    """)
    return


@app.cell
def _(T, new_model, sync, train_ids, val_ids, val_rep, val_sel):
    _m = new_model(0)
    sync()
    _t0 = time.perf_counter()
    base_hist = train_lm(_m, train_ids, T, val_data=val_ids, batch_size=CONFIG["batch_size"],
                         max_steps=CONFIG["base_steps"], lr=CONFIG["base_lr"], warmup_steps=CONFIG["warmup"],
                         eval_interval=CONFIG["eval_interval"], eval_batches=CONFIG["eval_batches"], device=DEVICE)
    sync()
    base_wall = time.perf_counter() - _t0
    target = estimate_loss(_m, val_rep, T, batch_size=CONFIG["batch_size"], n_batches=CONFIG["final_eval_batches"],
                           device=DEVICE)
    target_sel = estimate_loss(_m, val_sel, T, batch_size=CONFIG["batch_size"],
                               n_batches=CONFIG["final_eval_batches"], device=DEVICE)
    mo.md(f"baseline: **{base_wall:.1f} s** for {CONFIG['base_steps']} steps; final validation loss (estimate_loss, "
          f"{CONFIG['final_eval_batches']} batches) **{target:.4f}** nats/char on the reporting half of the validation "
          f"block: this is the target ({target_sel:.4f} on the selection half, which the recipe search uses).")
    return base_wall, target, target_sel


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3. ablation: what each optimization buys per step

    step time (median of repeated windows, after warmup and compilation) for the baseline step,
    for each optimization added on its own, for all of them together, and for all of them minus
    one. speedups are relative to the baseline step, with 95% bootstrap intervals over windows.
    """)
    return


@app.cell
def _(B, T, new_model, sync, train_ids):
    def _measure(on_device=False, fused_attn=False, fused_optimizer=False, amp=None, compiled=False, torch_scaler=False):
        _m = new_model().to(DEVICE).train()
        if fused_attn:
            fused_attention(_m)
        _opt = torch.optim.AdamW(_m.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                 **({"fused": True} if fused_optimizer else {}))
        _scaler = None
        if amp == torch.float16:
            _scaler = torch.amp.GradScaler(DEVICE) if torch_scaler else GradScaler()
        _step = make_train_step(_m, _opt, amp_dtype=amp, scaler=_scaler,
                                compile_backend=CONFIG["compile_backend"] if compiled else None)
        _g = torch.Generator().manual_seed(0)
        _data = torch.as_tensor(train_ids).to(DEVICE)
        _offs = torch.arange(T, device=DEVICE)

        def _one():
            if on_device:
                _i = torch.randint(0, len(_data) - T, (B,), generator=_g).to(DEVICE)[:, None] + _offs
                _step(_data[_i], _data[_i + 1])
            else:
                _step(*get_batch(train_ids, T, B, generator=_g, device=DEVICE))
        return time_steps(_one, CONFIG["time_steps"], warmup=5, repeats=CONFIG["time_repeats"], sync=sync)

    OPTS = {"on-device batches": dict(on_device=True), "fused attention": dict(fused_attn=True),
            "fused AdamW": dict(fused_optimizer=True)}
    if CONFIG["amp"] is not None:
        OPTS["fp16 + your GradScaler"] = dict(amp=CONFIG["amp"])
    if CONFIG["compile_backend"] is not None:
        OPTS["torch.compile"] = dict(compiled=True)
    _all = {k: v for d in OPTS.values() for k, v in d.items()}
    ablation = {"baseline step": _measure()}
    for _name, _kw in OPTS.items():
        ablation[f"+ {_name} only"] = _measure(**_kw)
    if CONFIG["amp"] is not None:
        ablation["+ fp16 + torch.amp.GradScaler only"] = _measure(amp=CONFIG["amp"], torch_scaler=True)
    ablation["all"] = _measure(**_all)
    for _name, _kw in OPTS.items():
        if _name in ("fp16 + your GradScaler", "torch.compile"):  # the others were within noise of "all"
            ablation[f"all − {_name}"] = _measure(**{k: v for k, v in _all.items() if k not in _kw})
    candidates = {"all": dict(_all)}
    if CONFIG["amp"] is not None:
        ablation["all, with torch.amp.GradScaler"] = _measure(**_all, torch_scaler=True)
        candidates["all, with torch.amp.GradScaler"] = dict(_all, torch_scaler=True)
        candidates["all − fp16 + your GradScaler"] = {k: v for k, v in _all.items() if k != "amp"}
    best_config = min(candidates, key=lambda k: np.median(ablation[k]))
    race_kwargs = candidates[best_config]
    _base = ablation["baseline step"]
    _rows = []
    for _name, _t in ablation.items():
        _s, _lo, _hi = speedup_ci(_base, _t, rng=0)
        _rows.append(f"| {_name} | {1e3 * np.median(_t):.1f} ({1e3 * _t.min():.1f}–{1e3 * _t.max():.1f}) | "
                     f"{_s:.2f}× [{_lo:.2f}, {_hi:.2f}] |")
    mo.md("| configuration | ms per step, median (range) | speedup vs baseline step [95% CI] |\n|---|---|---|\n"
          + "\n".join(_rows) + f"\n\nfastest full configuration, used for the race: **{best_config}**.")
    return ablation, race_kwargs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4. the race: time to the target loss

    `train_fast` with the fastest configuration from the ablation, first with the baseline's
    exact recipe (same steps, learning rate and dropout: the pure systems speedup, the headline),
    then a small recipe search: 3× the learning rate, no dropout (a short run underfits, so
    regularization only slows it down) and shorter schedules. the search picks its winner on one
    half of the validation block and reports on the other, reruns the winner with more seeds,
    and the baseline is rerun at the end, so every speedup is a range over two baseline runs.
    """)
    return


@app.cell
def _(T, base_wall, new_model, race_kwargs, sync, target, target_sel, train_ids, val_ids, val_rep, val_sel):
    _fast = dict(fused_attn=race_kwargs.get("fused_attn", False), amp_dtype=race_kwargs.get("amp"),
                 compile_backend=CONFIG["compile_backend"] if race_kwargs.get("compiled") else None,
                 fused_optimizer=race_kwargs.get("fused_optimizer", False))

    def _run(steps, lr, seed=0, dropout=None):
        _m = new_model(seed, dropout)
        _h = train_fast(_m, train_ids, T, val_data=val_sel, batch_size=CONFIG["batch_size"], max_steps=steps, lr=lr,
                        warmup_steps=CONFIG["warmup"], eval_batches=CONFIG["final_eval_batches"], device=DEVICE,
                        seed=seed, scaler=torch.amp.GradScaler(DEVICE) if race_kwargs.get("torch_scaler") else None,
                        **_fast)
        _h["val_rep"] = estimate_loss(_m, val_rep, T, batch_size=CONFIG["batch_size"],
                                      n_batches=CONFIG["final_eval_batches"], device=DEVICE)
        return _h

    _drop = CONFIG["model"]["dropout"]
    systems = _run(CONFIG["base_steps"], CONFIG["base_lr"])  # the same recipe, only faster steps
    races = {("same recipe (systems only)", CONFIG["base_steps"], CONFIG["base_lr"], _drop): systems}
    for _steps in CONFIG["fast_steps"]:  # the recipe search: 3x the learning rate, no dropout, shorter schedules
        races[("lr 3×, no dropout", _steps, CONFIG["fast_lr"], 0.0)] = _run(_steps, CONFIG["fast_lr"], dropout=0.0)
    search_cost = sum(h["wall"] for k, h in races.items() if k[0] != "same recipe (systems only)")
    _ok = [k for k, h in races.items() if k[0] != "same recipe (systems only)" and h["val_loss"] <= target_sel]
    winner = min(_ok, key=lambda k: races[k]["wall"]) if _ok else None  # chosen on the selection half only
    seed_runs = []
    if winner is not None:
        seed_runs = [races[winner]] + [_run(winner[1], winner[2], seed=s, dropout=winner[3]) for s in CONFIG["seeds"][1:]]
    _m2 = new_model(0)
    sync()
    _t0 = time.perf_counter()
    train_lm(_m2, train_ids, T, val_data=val_ids, batch_size=CONFIG["batch_size"], max_steps=CONFIG["base_steps"],
             lr=CONFIG["base_lr"], warmup_steps=CONFIG["warmup"], eval_interval=CONFIG["eval_interval"],
             eval_batches=CONFIG["eval_batches"], device=DEVICE)
    sync()
    base_walls = [base_wall, time.perf_counter() - _t0]
    _lo_b, _hi_b = min(base_walls), max(base_walls)

    def _range(wall_lo, wall_hi=None):
        return f"{_lo_b / (wall_hi or wall_lo):.2f}–{_hi_b / wall_lo:.2f}×"

    _rows = [f"| {k[0]} | {k[1]} | {k[2]:g} | {k[3]:g} | {h['wall']:.1f} s | {h['val_loss']:.4f} | {h['val_rep']:.4f} | "
             f"{'yes' if h['val_rep'] <= target else 'no'} | {_range(h['wall'])} | {h['skipped']} |"
             for k, h in races.items()]
    _md = (f"baselines: **{base_walls[0]:.1f} s** and, rerun after the race, **{base_walls[1]:.1f} s**; target "
           f"**{target:.4f}** (reporting half). every speedup below is a range over the two baseline runs.\n\n"
           "| run | steps | lr | dropout | wall | val loss (selection half) | val loss (reporting half) | reaches target "
           "| speedup | skipped steps |\n|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows))
    _md += (f"\n\n**headline, same recipe:** the systems optimizations alone run chapter 18's exact training "
            f"{_range(systems['wall'])} faster.")
    if seed_runs:
        _w = np.array([h["wall"] for h in seed_runs])
        _v = np.array([h["val_rep"] for h in seed_runs])
        _md += (f"\n\n**recipe factor:** the search picked {winner[1]} steps at lr {winner[2]:g}, dropout {winner[3]:g} "
                f"(the shortest run reaching the target on the selection half). over seeds {CONFIG['seeds']}: wall "
                f"{_w.min():.1f}–{_w.max():.1f} s, reporting-half loss {_v.min():.4f}–{_v.max():.4f} "
                f"({'all' if (_v <= target).all() else 'not all'} at or below the target). that's "
                f"{systems['wall'] / _w.max():.2f}–{systems['wall'] / _w.min():.2f}× on top of the systems speedup, and "
                f"**{_range(_w.min(), _w.max())} in total** against the baselines. the search itself cost "
                f"{search_cost:.0f} s of training, which this speedup does not charge.")
    else:
        _md += "\n\nno run of the recipe search reached the target on the selection half."
    mo.md(_md)
    return races, winner


@app.cell
def _(ablation, races, target):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    _names = list(ablation)
    _a1.barh(range(len(_names)), [1e3 * np.median(ablation[n]) for n in _names],
             xerr=[[1e3 * (np.median(ablation[n]) - ablation[n].min()) for n in _names],
                   [1e3 * (ablation[n].max() - np.median(ablation[n])) for n in _names]], color="C0")
    _a1.set_yticks(range(len(_names)), _names, fontsize=7)
    _a1.invert_yaxis()
    _a1.set(xlabel="ms per step (median, range)", title="ablation")
    for (_label, _steps, _lr, _dr), _h in races.items():
        _x = np.linspace(0, _h["wall"], len(_h["loss"]))
        _sm = np.convolve(_h["loss"], np.ones(25) / 25, mode="valid") if len(_h["loss"]) > 50 else _h["loss"]
        _a2.plot(_x[:len(_sm)], _sm, label=f"{_label}, {_steps} steps", lw=1)
    _a2.axhline(target, color="k", ls=":", lw=1, label="target (a validation loss)")
    _a2.set(xlabel="wall-clock seconds", ylabel="training loss (smoothed)", title="loss vs time")
    _a2.legend(fontsize=6)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace each prompt with your answer, using the numbers above (and your reruns).

    1. **the profile.** where did the baseline step's time go, on the CPU and on your training
       device? which ops were memory-bound by topic 01's analysis, and which optimization targets
       each?
    2. **the ablation.** which optimizations helped on their own, which only in combination, and
       which hurt? explain one interaction (for example, why fp16 can be slower alone but faster
       with compilation) with a measurement, not a guess.
    3. **the race.** what speedup in time to the target loss did you reach, with what spread
       over seeds? how much of it is systems (same steps, faster steps) and how much is
       optimization (fewer steps)? is the comparison fair to the baseline (its evaluations, its
       learning rate)?
    4. **the hardware.** what limits further speedup on your machine? estimate your final MFU
       with topic 01's tools and compare it with the roofline. what would change on an NVIDIA GPU
       with tensor cores and FlashAttention?
    5. **if not 5×:** what's the largest remaining cost, and what would it take to remove it?
    """)
    return


if __name__ == "__main__":
    app.run()
