import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.text import CharTokenizer  # yours, from ch 17 · 02
    from mlfs.transformer import GPT, get_batch  # yours, from ch 18 · 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    HAS_MPS = torch.backends.mps.is_available()
    GPU = "cuda" if torch.cuda.is_available() else ("mps" if HAS_MPS else None)  # the accelerator, if any
    DEVICES = ["cpu"] + ([GPU] if GPU else [])
    _text = datasets.tiny_shakespeare()[:200_000]
    TOK = CharTokenizer().train(_text)
    DATA = np.asarray(TOK.encode(_text), dtype=np.int64)


@app.function
def small_gpt(seed=0):
    """a small GPT (2 layers of width 64, context 64) for CPU experiments."""
    torch.manual_seed(seed)
    return GPT(TOK.vocab_size, block_size=64, n_layers=2, n_heads=4, d_model=64)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # mixed precision · lab

    the experiments use **your** `round_to_format`, `GradScaler` and `amp_step` from
    `exercise.py`, with your chapter 18 GPT on tiny Shakespeare (a synthetic stand-in
    offline). for each one: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards. about a minute on a laptop.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: where gradients live

    one float32 forward and backward pass of a GPT that has trained for 30 steps. we collect
    every parameter gradient, and every gradient flowing back through the residual stream
    between blocks (the activation gradients a float16 backward pass would actually store),
    and ask your `round_to_format` what float16 would do to each: keep it, keep it as a
    low-precision subnormal, or flush it to zero; then the same after scaling by $2^{16}$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("what fraction of the activation gradients would float16 flush to zero? and with a loss scale of 65,536?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _m = small_gpt()
    _opt = torch.optim.AdamW(_m.parameters(), lr=3e-3)
    _g = torch.Generator().manual_seed(0)
    for _ in range(30):
        _x, _y = get_batch(DATA, 64, 32, generator=_g)
        _, _loss = _m(_x, _y)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    _acts = []
    _hooks = [b.register_full_backward_hook(lambda mod, gi, go: _acts.append(go[0].detach().flatten()))
              for b in _m.blocks]
    _x, _y = get_batch(DATA, 64, 32, generator=_g)
    _opt.zero_grad()
    _m(_x, _y)[1].backward()
    for _h in _hooks:
        _h.remove()
    groups = {"parameter gradients": torch.cat([p.grad.flatten() for p in _m.parameters()]).double().numpy(),
              "activation gradients": torch.cat(_acts).double().numpy()}
    _f16 = ex.float_format(5, 10)
    _rows = []
    for _name, _gv in groups.items():
        _gv = _gv[_gv != 0]
        for _s in (1.0, 2.0**16):
            _r = ex.round_to_format(_gv * _s, 5, 10)
            _zero = np.mean(_r == 0)
            _sub = np.mean((_r != 0) & (np.abs(_r) < _f16["min_normal"]))
            _rows.append(f"| {_name} | {_s:,.0f} | {100 * _zero:.2f}% | {100 * _sub:.2f}% | "
                         f"{100 * np.mean(np.isinf(_r)):.2f}% |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    for _name, _gv in groups.items():
        _ax.hist(np.log2(np.abs(_gv[_gv != 0])), bins=120, alpha=0.6, density=True, label=_name)
    for _v, _lab in ((np.log2(_f16["min_subnormal"]), "fp16 smallest subnormal"),
                     (np.log2(_f16["min_normal"]), "fp16 smallest normal"), (np.log2(_f16["max"]), "fp16 max")):
        _ax.axvline(_v, color="k", ls=":", lw=1)
        _ax.text(_v, _ax.get_ylim()[1] * 0.9, " " + _lab, fontsize=6, rotation=90, va="top")
    _ax.set(xlabel="log2 |gradient|", ylabel="density", title="gradient magnitudes vs float16's range")
    _ax.legend(fontsize=7, loc="upper left")
    mo.vstack([_fig, mo.md("| gradients | loss scale | flushed to 0 | subnormal (imprecise) | overflow |\n|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are activation gradients so much smaller than parameter gradients?
    (think about what the mean over a batch of $B \times T$ positions does to $\partial L / \partial h$
    for one position.) how far does the scale of 65,536 shift the histogram, and is there
    room for more before anything overflows?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: float16 overflow, seen live

    the same GPT trained for 160 steps with your `amp_step` three ways: float32, bfloat16
    without a scaler, and float16 with your `GradScaler` started at a deliberately absurd
    scale of $2^{32}$ and a `growth_interval` of 40 (instead of 2,000) so you can watch it
    grow. skipped steps are marked.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("starting at 2^32, how many steps does float16 training skip before the first real update? "
                    "where does the scale settle?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    runs = {}
    for _name, _dt, _scaler in (("float32", None, None), ("bfloat16", torch.bfloat16, None),
                                ("float16 + GradScaler", torch.float16,
                                 ex.GradScaler(init_scale=2.0**32, growth_interval=40))):
        _m = small_gpt()
        _opt = torch.optim.AdamW(_m.parameters(), lr=3e-3)
        _g = torch.Generator().manual_seed(1)
        _losses, _scales, _skipped = [], [], []
        _lossfn = lambda out, t: torch.nn.functional.cross_entropy(out[0].flatten(0, 1).float(), t.flatten())
        for _step in range(160):
            _x, _y = get_batch(DATA, 64, 16, generator=_g)
            if _dt is None:
                _opt.zero_grad()
                _l = _lossfn(_m(_x), _y)
                _l.backward()
                _opt.step()
                _lv, _ok = _l.item(), True
            else:
                _lv, _ok = ex.amp_step(_m, _lossfn, _opt, _x, _y, dtype=_dt, scaler=_scaler)
            _losses.append(_lv)
            _scales.append(_scaler.get_scale() if _scaler else np.nan)
            _skipped.append(not _ok)
        runs[_name] = dict(loss=np.array(_losses), scale=np.array(_scales), skipped=np.array(_skipped))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    for _name, _r in runs.items():
        _a1.plot(np.convolve(_r["loss"], np.ones(8) / 8, mode="valid"), label=_name)
    _a1.set(xlabel="step", ylabel="training loss (8-step mean)", title="three precisions")
    _a1.legend(fontsize=8)
    _r = runs["float16 + GradScaler"]
    _a2.semilogy(_r["scale"], "C2-", label="loss scale after update()")
    _a2.plot(np.flatnonzero(_r["skipped"]), _r["scale"][_r["skipped"]], "rx", label="skipped step")
    _a2.set(xlabel="step", ylabel="scale", title="your GradScaler, float16")
    _a2.legend(fontsize=8)
    _first = int(np.argmin(_r["skipped"]))
    mo.vstack([_fig, mo.md(
        f"float16: **{int(_r['skipped'].sum())}** skipped steps, the first real update at step **{_first}**; final scale "
        f"**2^{int(np.log2(_r['scale'][-1]))}**. final 20-step mean losses: "
        + ", ".join(f"{k} {v['loss'][-20:].mean():.3f}" for k, v in runs.items()))])
    return (runs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does every one of the first steps overflow, and why does the count of
    skipped steps come out to what it does? a few more overflows happen shortly after the first
    real update, then none for the rest of the run, even though the scale keeps growing every
    40 clean steps: what does that say about how the gradients' size changes as training
    proceeds, and how far could the scale grow before overflowing again (experiment 1)? do the
    three loss curves differ by more than the noise of one seed?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: is 16-bit faster here?

    square matmuls in float32, float16 and bfloat16, and a full training step of a chapter-18
    model M GPT (batch 32) under autocast in each dtype, on your CPU and (if present) your GPU.
    best of 5.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("on your fastest device, how much faster is a float16 matmul than a float32 one? and bfloat16?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _best(fn, sync, n=5):
        fn()
        fn()
        if sync:
            sync()
        _ts = []
        for _ in range(n):
            _t0 = time.perf_counter()
            fn()
            if sync:
                sync()
            _ts.append(time.perf_counter() - _t0)
        return min(_ts)

    _rows = []
    for _d in DEVICES:
        _sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(_d)
        _size = 1024 if _d == "cpu" else 2048
        _res = {}
        for _dt in (torch.float32, torch.float16, torch.bfloat16):
            _a = torch.randn(_size, _size, device=_d).to(_dt)
            _res[str(_dt).split(".")[1]] = 2 * _size**3 / _best(lambda: _a @ _a, _sync) / 1e9
        _steps = {}
        if _d != "cpu":
            torch.manual_seed(0)
            _m = GPT(TOK.vocab_size, block_size=128, n_layers=4, n_heads=6, d_model=192).to(_d)
            _opt = torch.optim.AdamW(_m.parameters(), lr=1e-3)
            _g = torch.Generator().manual_seed(0)
            _lossfn = lambda out, t: torch.nn.functional.cross_entropy(out[0].flatten(0, 1).float(), t.flatten())
            for _dt in (None, torch.float16, torch.bfloat16):
                _scaler = ex.GradScaler() if _dt == torch.float16 else None

                def _step():
                    _x, _y = get_batch(DATA, 128, 32, generator=_g, device=_d)
                    if _dt is None:
                        _opt.zero_grad()
                        _lossfn(_m(_x), _y).backward()
                        _opt.step()
                    else:
                        ex.amp_step(_m, _lossfn, _opt, _x, _y, dtype=_dt, scaler=_scaler)
                _steps[str(_dt).split(".")[-1] if _dt else "float32"] = _best(_step, _sync, n=4)
        _rows.append(f"| {_d} | {_size}³ | " + " | ".join(f"{_res[k]:.0f}" for k in ("float32", "float16", "bfloat16"))
                     + " | " + (" | ".join(f"{1e3 * _steps[k]:.0f}" for k in ("float32", "float16", "bfloat16"))
                                if _steps else "– | – | –") + " |")
    mo.md("| device | matmul | fp32 GFLOP/s | fp16 GFLOP/s | bf16 GFLOP/s | GPT step fp32 (ms) | fp16 + scaler (ms) "
          "| bf16 (ms) |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
          + "\n\n(the GPT step is timed on the GPU only; on the CPU it takes too long for this lab.)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compare your numbers with the 8–16× that tensor cores give 16-bit matmuls
    on NVIDIA GPUs. what does your hardware have, or lack? if a 16-bit format is *slower*,
    what extra work is autocast doing? given these measurements, should the capstone use float16
    on your machine, and what would you need to measure to decide?
    """)
    return


if __name__ == "__main__":
    app.run()
