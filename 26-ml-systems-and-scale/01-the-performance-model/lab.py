import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import notebooks, ui
    from mlfs.transformer import GPT, get_batch  # yours, from ch 18 · 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    HAS_MPS = torch.backends.mps.is_available()
    GPU = "cuda" if torch.cuda.is_available() else ("mps" if HAS_MPS else None)  # the accelerator, if any
    DEVICES = ["cpu"] + ([GPU] if GPU else [])


@app.function
def sync_for(device):
    """the function that waits for `device` to finish its queued work (None on the CPU, which runs synchronously)."""
    device = str(device)
    if device.startswith("cuda"):
        return torch.cuda.synchronize
    if device.startswith("mps"):
        return torch.mps.synchronize
    return None


@app.function
def measure_machine(device):
    """measured bandwidth (B/s, best of 5 large copies) and peak FLOP/s (best float32 matmul over three sizes) on device."""
    sync = sync_for(device)
    n = 16 * 2**20  # 16M floats = 64 MiB per tensor, far bigger than any cache
    x, y = torch.randn(n, device=device), torch.empty(n, device=device)
    t_copy = ex.benchmark(lambda: y.copy_(x), warmup=2, repeats=5, sync=sync)["best"]
    best = (0.0, None)
    for size in ((512, 1024, 2048) if device == "cpu" else (1024, 2048, 4096)):
        a, b = torch.randn(size, size, device=device), torch.randn(size, size, device=device)
        t_mm = ex.benchmark(lambda: a @ b, warmup=2, repeats=5, sync=sync)["best"]
        best = max(best, (ex.matmul_flops(size, size, size) / t_mm, size))
    return {"bandwidth": 2 * n * 4 / t_copy, "peak": best[0], "mm_size": best[1]}


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # the performance model · lab

    the experiments use **your** `benchmark`, `roofline`, `gpt_block_ops`, `mfu` and
    `profile_ops` from `exercise.py`, and your `GPT` from chapter 18. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards.

    every number here is measured on **your machine, now**. they will differ from the lab notes,
    and from one run to the next if other programs are busy; the ratios and the shapes of the
    plots are what to compare. devices found: **{", ".join(DEVICES)}**. the lab takes 1–2
    minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: your machine's roofline

    bandwidth from copying a 64 MiB tensor (read once, write once: $2 \times 64$ MiB moved),
    peak from the fastest of three large float32 matmuls, both with your `benchmark` (best of 5). then a sweep:
    square matmuls from $n = 16$ to $n = 2048$ (intensity $n/6$) and elementwise adds of
    several sizes (intensity $1/12$: one FLOP per three 4-byte elements), each plotted at its
    achieved FLOP/s against your `roofline`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how close to the roofline will a 64×64 matmul get, as a percentage of what the roofline allows? and a "
        "2048×2048 one?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    machine = {d: measure_machine(d) for d in DEVICES}
    _rows = [f"| {d} | {m['bandwidth'] / 1e9:.1f} GB/s | {m['peak'] / 1e9:.0f} GFLOP/s ({m['mm_size']}³) | "
             f"{ex.ridge_point(m['peak'], m['bandwidth']):.1f} FLOP/B |" for d, m in machine.items()]
    mo.md("| device | measured bandwidth | measured peak (float32 matmul) | ridge point |\n|---|---|---|---|\n"
          + "\n".join(_rows))
    return (machine,)


@app.cell
def _(machine):
    sweep = {}
    for _d in DEVICES:
        _sync, _pts = sync_for(_d), []
        _P, _W = machine[_d]["peak"], machine[_d]["bandwidth"]
        for _n in (16, 32, 64, 128, 256, 512, 1024, 2048):
            _a, _b = torch.randn(_n, _n, device=_d), torch.randn(_n, _n, device=_d)
            _t = ex.benchmark(lambda: _a @ _b, warmup=2, repeats=5, sync=_sync)["best"]
            _I = ex.arithmetic_intensity(ex.matmul_flops(_n, _n, _n), ex.matmul_bytes(_n, _n, _n))
            _pts.append(dict(kind="matmul", label=str(_n), I=_I, got=ex.matmul_flops(_n, _n, _n) / _t,
                             roof=ex.roofline(_I, _P, _W)))
        for _k in (12, 16, 20, 24):
            _n = 2**_k
            _a, _b, _out = torch.randn(_n, device=_d), torch.randn(_n, device=_d), torch.empty(_n, device=_d)
            _t = ex.benchmark(lambda: torch.add(_a, _b, out=_out), warmup=2, repeats=5, sync=_sync)["best"]
            _I = ex.arithmetic_intensity(_n, 3 * 4 * _n)
            _pts.append(dict(kind="add", label=f"2^{_k}", I=_I, got=_n / _t, roof=ex.roofline(_I, _P, _W)))
        sweep[_d] = _pts
    _fig, _axes = plt.subplots(1, len(DEVICES), figsize=(5.2 * len(DEVICES), 3.6), squeeze=False)
    for _ax, _d in zip(_axes[0], DEVICES):
        _I = np.logspace(-2, 3, 200)
        _ax.plot(_I, ex.roofline(_I, machine[_d]["peak"], machine[_d]["bandwidth"]), "k-", lw=1.5, label="your roofline")
        for _kind, _m in (("matmul", "o"), ("add", "s")):
            _p = [r for r in sweep[_d] if r["kind"] == _kind]
            _ax.plot([r["I"] for r in _p], [r["got"] for r in _p], _m, label=_kind)
            if _kind == "matmul":  # the add points share one intensity: their sizes are in the table instead
                for _r in _p:
                    _ax.annotate(_r["label"], (_r["I"], _r["got"]), fontsize=6, xytext=(3, -8), textcoords="offset points")
        _ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (FLOP/byte)", ylabel="achieved FLOP/s",
                title=f"{_d}: ridge at {ex.ridge_point(machine[_d]['peak'], machine[_d]['bandwidth']):.0f} FLOP/B")
        _ax.legend(fontsize=7, loc="lower right")
    _lines = [f"| {_d} | {r['kind']} {r['label']} | {r['I']:.3f} | {r['got'] / 1e9:.2f} | {100 * r['got'] / r['roof']:.1f}% |"
              for _d in DEVICES for r in sweep[_d]]
    mo.vstack([_fig, mo.accordion({"the numbers": mo.md(
        "| device | op, size | intensity | GFLOP/s | % of roofline |\n|---|---|---|---|---|\n" + "\n".join(_lines))})])
    return (sweep,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which points sit on the roofline, and which fall far below it? the small
    matmuls have high enough intensity to be compute-bound, yet run at a tiny fraction of
    peak: which of the lesson's three regimes are they in? what would you have to do to move a
    64×64 matmul up to the roof? if you have a GPU, compare its bandwidth with the
    CPU's: they share the same memory chips, so why might they differ? some small adds may sit
    *above* the line: where did their data come from, if not DRAM?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a GPT block, op by op

    every op of chapter 18's model M block ($d = 192$, 6 heads, $d_\text{ff} = 768$, $T = 128$)
    on a batch of windows, timed on its own with your `benchmark` and compared with the time
    your roofline predicts for it from `gpt_block_ops`: $\text{FLOPs} / \text{roofline}(I)$.
    the bars compare each op's share of the block's FLOPs with its share of the measured time.
    """)
    return


@app.cell
def _():
    device2 = mo.ui.dropdown(DEVICES, value=DEVICES[-1], label="device")
    p2 = ui.predict("the softmax and the elementwise ops (layernorms, GELU, residual adds) do about 1.3% of the block's "
                    "FLOPs. what share of its time will they take?")
    mo.vstack([p2, device2])
    return device2, p2


@app.cell
def _(device2, machine, p2):
    ui.gate(p2)
    _d = device2.value
    B, T, d, H, dff = (16 if _d == "cpu" else 64), 128, 192, 6, 768
    _sync, _g = sync_for(_d), torch.Generator().manual_seed(0)
    _x = torch.randn(B, T, d, generator=_g).to(_d)
    _w = {k: (torch.randn(o, i, generator=_g) * 0.02).to(_d) for k, (i, o) in
          {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d), "fc1": (d, dff), "fc2": (dff, d)}.items()}
    _q = torch.randn(B, H, T, d // H, generator=_g).to(_d)
    _s = torch.randn(B, H, T, T, generator=_g).to(_d)
    _h = torch.randn(B, T, dff, generator=_g).to(_d)
    _fns = {
        "ln1": lambda: F.layer_norm(_x, (d,)), "q_proj": lambda: F.linear(_x, _w["q"]),
        "k_proj": lambda: F.linear(_x, _w["k"]), "v_proj": lambda: F.linear(_x, _w["v"]),
        "scores": lambda: _q @ _q.transpose(-2, -1), "softmax": lambda: torch.softmax(_s, dim=-1),
        "attn_v": lambda: _s @ _q, "out_proj": lambda: F.linear(_x, _w["o"]), "residual1": lambda: _x + _x,
        "ln2": lambda: F.layer_norm(_x, (d,)), "fc1": lambda: F.linear(_x, _w["fc1"]),
        "gelu": lambda: F.gelu(_h), "fc2": lambda: F.linear(_h, _w["fc2"]), "residual2": lambda: _x + _x,
    }
    _P, _W = machine[_d]["peak"], machine[_d]["bandwidth"]
    block = []
    for _op in ex.gpt_block_ops(B, T, d, H, d_ff=dff):
        _t = ex.benchmark(_fns[_op["name"]], warmup=3, repeats=7, sync=_sync)["best"]
        _I = ex.arithmetic_intensity(_op["flops"], _op["bytes"])
        block.append(dict(name=_op["name"], flops=_op["flops"], I=_I, kind=ex.bound(_I, _P, _W),
                          pred=_op["flops"] / ex.roofline(_I, _P, _W), meas=_t))
    _F = sum(r["flops"] for r in block)
    _Tm = sum(r["meas"] for r in block)
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    _xs = np.arange(len(block))
    _ax.bar(_xs - 0.2, [100 * r["flops"] / _F for r in block], 0.4, label="share of FLOPs")
    _ax.bar(_xs + 0.2, [100 * r["meas"] / _Tm for r in block], 0.4, label="share of measured time")
    _ax.set_xticks(_xs, [r["name"] for r in block], rotation=45, ha="right", fontsize=8)
    _ax.set(ylabel="%", title=f"model M block, batch {B}, on {_d}")
    _ax.legend(fontsize=8)
    _small = [r for r in block if r["name"] in ("ln1", "ln2", "softmax", "gelu", "residual1", "residual2")]
    _rows = "\n".join(f"| {r['name']} | {r['I']:.2f} | {r['kind']} | {1e3 * r['pred']:.3f} | {1e3 * r['meas']:.3f} | "
                      f"{r['meas'] / r['pred']:.1f}× |" for r in block)
    mo.vstack([_fig, mo.md(
        f"softmax + elementwise ops: **{100 * sum(r['flops'] for r in _small) / _F:.1f}%** of the FLOPs, "
        f"**{100 * sum(r['meas'] for r in _small) / _Tm:.0f}%** of the measured time.\n\n"
        "| op | intensity | roofline says | predicted ms | measured ms | measured / predicted |\n"
        "|---|---|---|---|---|---|\n" + _rows)])
    return (block,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which ops does the roofline call memory-bound, and does their share of the
    time match their share of the bytes rather than of the FLOPs? which ops run closest to their
    prediction, and which are furthest from it (the last column)? the attention-score matmul is
    a matmul: why is it on the memory side? what does this table say a fused attention kernel
    (ch 18 · 06, topic 02) would save?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: profiling a training step

    a full training step (forward, backward, AdamW) of your GPT, profiled op by op with your
    `profile_ops` on the CPU, where the profiler's times are real compute times. then the same
    model's step time on each device, measured with `benchmark`, turned into MFU with your
    `transformer_train_flops` and the peak you measured in experiment 1.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("what MFU will chapter 18's model M reach on your fastest device: above 30%, 10–30%, or below 10%?")
    p3
    return (p3,)


@app.cell
def _(machine, p3):
    ui.gate(p3)
    _data = np.random.default_rng(0).integers(0, 65, 50_000)

    def _make(device, batch):
        torch.manual_seed(0)
        _m = GPT(65, block_size=128, n_layers=4, n_heads=6, d_model=192).to(device).train()
        _opt = torch.optim.AdamW(_m.parameters(), lr=1e-3)
        _gen = torch.Generator().manual_seed(0)

        def _step():
            _xb, _yb = get_batch(_data, 128, batch, generator=_gen, device=device)
            _, _loss = _m(_xb, _yb)
            _opt.zero_grad(set_to_none=True)
            _loss.backward()
            _opt.step()
        return _m, _step

    _m, _step = _make("cpu", 8)
    rows3 = ex.profile_ops(_step, n_steps=2, warmup=1)
    _total = sum(r[1] for r in rows3)
    _mm = sum(r[1] for r in rows3 if r[0] in ("aten::mm", "aten::bmm", "aten::addmm", "aten::matmul"))
    _top = "\n".join(f"| `{r[0]}` | {r[2]} | {100 * r[1] / _total:.1f}% |" for r in rows3[:12])
    _n_params = sum(p.numel() for p in _m.parameters())
    _mfu_rows = []
    for _d in DEVICES:
        _batch = 16 if _d == "cpu" else 64
        _, _st = _make(_d, _batch)
        _t = ex.benchmark(_st, warmup=3, repeats=5, sync=sync_for(_d))
        _flops = ex.transformer_train_flops(_n_params, _batch * 128)
        _mfu_rows.append(f"| {_d} | {_batch} | {1e3 * _t['best']:.0f} ms ({1e3 * min(_t['times']):.0f}–"
                         f"{1e3 * max(_t['times']):.0f}) | {_flops / _t['best'] / 1e9:.0f} GFLOP/s | "
                         f"**{100 * ex.mfu(_flops, _t['best'], machine[_d]['peak']):.1f}%** |")
    mo.md(f"model M: {_n_params:,} parameters. CPU profile of 2 steps at batch 8; matrix products "
          f"(`mm`, `bmm`, `addmm`) take **{100 * _mm / _total:.0f}%** of the self time.\n\n"
          "| op | calls | share of self time |\n|---|---|---|\n" + _top +
          "\n\n| device | batch | step time, best (range of 5) | achieved (6N per token) | MFU |\n|---|---|---|---|---|\n"
          + "\n".join(_mfu_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what fraction of the CPU step goes to matrix products, and what are the
    biggest non-matmul ops? (look for the ops your attention uses to build and apply its mask.)
    the MFU is far below the ~50% of large training runs: using experiments 1 and 2, list the
    reasons, roughly in order of size. which topic of this chapter attacks each one?
    """)
    return


if __name__ == "__main__":
    app.run()
