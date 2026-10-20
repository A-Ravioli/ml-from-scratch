import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import os
    import pickle
    import subprocess
    import sys
    import tempfile
    import textwrap
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui
    from mlfs.transformer import GPT  # yours, from ch 18 · 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))

    WORKER = textwrap.dedent(r"""
        import pickle, sys
        import torch, torch.distributed as dist
        from torch import nn
        import torch.nn.functional as F
        torch.set_num_threads(1)
        nb_path, rank, world, store, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
        from mlfs import notebooks
        nb = notebooks.load(nb_path)
        dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=world)
        try:
            torch.manual_seed(0)
            m = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 10))
            opt = nb.ZeroAdamW(m.parameters(), rank=rank, world_size=world, lr=1e-3)
            g = torch.Generator().manual_seed(0)
            for step in range(20):
                x, y = torch.randn(64, 256, generator=g), torch.randint(0, 10, (64,), generator=g)
                b = 64 // world
                opt.zero_grad()
                F.cross_entropy(m(x[rank * b:(rank + 1) * b]), y[rank * b:(rank + 1) * b]).backward()
                opt.step()
            state = (opt.exp_avg.numel() + opt.exp_avg_sq.numel()) * 4
            with open(out, "wb") as f:
                pickle.dump({"params": torch.cat([p.detach().flatten() for p in m.parameters()]), "state": state}, f)
        finally:
            dist.destroy_process_group()
    """)


@app.function
def model_m(block_size=128, seed=0):
    """chapter 18's model M: width 192, 4 layers, 6 heads, context 128, 65 characters."""
    torch.manual_seed(seed)
    return GPT(65, block_size=block_size, n_layers=4, n_heads=6, d_model=192)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # model parallelism & memory · lab

    the experiments use **your** `training_memory`, `saved_tensor_bytes`, `checkpoint`,
    `gpipe_schedule` and `ZeroAdamW`, with chapter 18's model M (1.8M parameters). for each
    one: write down a prediction, lock it in, run it, then explain what happened. compare with
    `solutions/lab_notes.md` afterwards. about a minute; experiment 3 starts real processes
    that load `exercise.py` from disk, so save it first.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: memory, predicted and measured

    your `training_memory` for float32 AdamW (`bytes_per_param=4`) against the bytes PyTorch
    actually holds after one AdamW step: parameters, gradients, and every tensor in the
    optimizer's state. then the activations: your `saved_tensor_bytes` around a forward pass
    (with the loss), excluding the parameters, as the batch and the context grow.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("at batch 64 and context 128, how many times larger than the 29 MB of weights + gradients + AdamW "
                    "state will the saved activations be?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _m = model_m()
    _n = sum(p.numel() for p in _m.parameters())
    _opt = torch.optim.AdamW(_m.parameters(), lr=1e-3)
    _x = torch.randint(0, 65, (2, 128))
    _m(_x, _x)[1].backward()
    _opt.step()
    _meas = {"params": sum(p.numel() * p.element_size() for p in _m.parameters()),
             "grads": sum(p.grad.numel() * p.grad.element_size() for p in _m.parameters()),
             "optimizer": sum(t.numel() * t.element_size() for s in _opt.state.values() for t in s.values()
                              if torch.is_tensor(t))}
    _pred = ex.training_memory(_n, bytes_per_param=4)
    _mem_rows = "\n".join(f"| {k} | {_pred[k] / 1e6:.2f} MB | {_meas[k] / 1e6:.2f} MB |" for k in _meas)
    acts = {}
    _params = list(_m.parameters())
    for _T in (32, 64, 128):
        for _B in (8, 16, 32, 64):
            _xb = torch.randint(0, 65, (_B, _T))
            _, acts[(_B, _T)] = ex.saved_tensor_bytes(lambda a: _m(a, a)[1], _xb, exclude=_params)
    _fig, _ax = plt.subplots(figsize=(5.5, 3.2))
    for _T in (32, 64, 128):
        _ax.plot([b * _T for b in (8, 16, 32, 64)], [acts[(b, _T)] / 1e6 for b in (8, 16, 32, 64)], "o-",
                 label=f"context {_T}")
    _ax.axhline(sum(_pred[k] for k in ("params", "grads", "optimizer")) / 1e6, color="k", ls=":",
                label="weights + grads + AdamW")
    _ax.set(xlabel="tokens per batch (B × T)", ylabel="saved activations (MB)", xscale="log", yscale="log")
    _ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _ax.yaxis.set_minor_formatter(plt.NullFormatter())
    _ax.set_yticks([20, 50, 100, 200, 500], ["20", "50", "100", "200", "500"])
    _ax.set_xticks([256, 1024, 4096], ["256", "1,024", "4,096"])
    _ax.legend(fontsize=7)
    _per_tok = {T: acts[(64, T)] / (64 * T) for T in (32, 64, 128)}
    mo.vstack([mo.md(f"model M: {_n:,} parameters.\n\n| part | your prediction | measured |\n|---|---|---|\n" + _mem_rows),
               _fig,
               mo.md(f"batch 64, context 128: **{acts[(64, 128)] / 1e6:.0f} MB** of saved activations, "
                     f"**{acts[(64, 128)] / _pred['total']:.0f}×** the weights, gradients and AdamW state. bytes per "
                     "token: " + ", ".join(f"T={T}: {v / 1e3:.1f} KB" for T, v in _per_tok.items()))])
    return (acts,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** does the state accounting match exactly, and if not, what's the small
    difference? the activations grow linearly with the batch, but the bytes *per token* grow
    with the context: which tensors in your chapter 18 attention scale with $T^2$, and how many
    of them are saved per layer? what would a fused attention kernel (topic 02, the capstone)
    save here?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: activation checkpointing

    every block of model M wrapped in your `checkpoint`, so only the block inputs are saved.
    saved activations measured as above, and a forward + backward step timed (best of 5) with
    and without, at batch 16 and context 128 on the CPU.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("checkpointing every block: by what factor do the saved activations fall? how much slower is a "
                    "forward + backward step?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _wrap(m):
        for _blk in m.blocks:
            _orig = _blk.forward
            _blk.forward = (lambda o: (lambda x, *a, **k: ex.checkpoint(lambda t: o(t, *a, **k), x)))(_orig)
        return m

    _plain, _ck = model_m(), _wrap(model_m())
    _x = torch.randint(0, 65, (16, 128), generator=torch.Generator().manual_seed(0))
    _res = {}
    for _name, _m in (("ordinary", _plain), ("checkpointed", _ck)):
        _params = list(_m.parameters())
        _, _saved = ex.saved_tensor_bytes(lambda a: _m(a, a)[1], _x, exclude=_params)

        def _step():
            _m.zero_grad()
            _m(_x, _x)[1].backward()
        _step()
        _ts = []
        for _ in range(5):
            _t0 = time.perf_counter()
            _step()
            _ts.append(time.perf_counter() - _t0)
        _res[_name] = (_saved, min(_ts), [p.grad.clone() for p in _params])
    _same = all(torch.allclose(a, b, atol=1e-5) for a, b in zip(_res["ordinary"][2], _res["checkpointed"][2]))
    mo.md("| model | saved activations | forward + backward (best of 5) |\n|---|---|---|\n"
          + "\n".join(f"| {k} | {v[0] / 1e6:.1f} MB | {1e3 * v[1]:.0f} ms |" for k, v in _res.items())
          + f"\n\nmemory **{_res['ordinary'][0] / _res['checkpointed'][0]:.1f}×** smaller, time "
          f"**{_res['checkpointed'][1] / _res['ordinary'][1]:.2f}×**; identical gradients: **{_same}**")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what is left in the checkpointed model's saved activations (count the
    block inputs: $4 \times B \times T \times d$ floats, plus what?). compare the time overhead
    with on paper 3's 4/3. if it's different, which part of the step does the recomputation not
    repeat?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: ZeRO-1 on real processes, and the pipeline bubble

    an MLP (0.4M parameters) trained for 20 steps with your `ZeroAdamW` on 1, 2 and 4 gloo
    processes, each rank taking its slice of every batch of 64. then your `gpipe_schedule`
    drawn for 4 stages, and the bubble for more micro-batches.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("with 4 ranks, how much optimizer state does each rank hold compared with 1 rank? will the final "
                    "weights be identical, close, or different?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    zero = {}
    for _w in (1, 2, 4):
        with tempfile.TemporaryDirectory() as _d:
            _outs = [os.path.join(_d, f"r{r}.pkl") for r in range(_w)]
            _procs = [subprocess.Popen([sys.executable, "-c", WORKER, ex.__file__, str(r), str(_w),
                                        os.path.join(_d, "store"), _outs[r]], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True,
                                       env={**os.environ, "MLFS_ROOT": str(notebooks.repo_root())}) for r in range(_w)]
            try:
                _errs = [p.communicate(timeout=120)[1] for p in _procs]
            except subprocess.TimeoutExpired:
                for p in _procs:
                    p.kill()
                raise
            for p, e in zip(_procs, _errs):
                if p.returncode:
                    raise RuntimeError(e[-2000:])
            zero[_w] = [pickle.load(open(o, "rb")) for o in _outs]
    _ref = zero[1][0]["params"]
    _rows = "\n".join(
        f"| {w} | {zero[w][0]['state'] / 1e6:.2f} MB | {ex.training_memory(_ref.numel(), bytes_per_param=4, zero_stage=1, world_size=w)['optimizer'] / 1e6:.2f} MB | "
        f"{max((r['params'] - zero[w][0]['params']).abs().max().item() for r in zero[w]):.1e} | "
        f"{(zero[w][0]['params'] - _ref).abs().max().item():.1e} |" for w in zero)
    _grid = ex.gpipe_schedule(4, 8)
    _img = np.array([[0 if e is None else (1 if e[0] == "F" else 2) for e in row] for row in _grid]).T
    _fig, _ax = plt.subplots(figsize=(7, 1.8))
    _ax.imshow(_img, cmap=plt.matplotlib.colors.ListedColormap(["#eeeeee", "#4c8bd6", "#e8923a"]), aspect="auto")
    for _t, _row in enumerate(_grid):
        for _s, _e in enumerate(_row):
            if _e is not None:
                _ax.text(_t, _s, str(_e[1]), ha="center", va="center", fontsize=6, color="white")
    _ax.set_yticks(range(4))
    _ax.set_xticks(range(len(_grid)))
    _ax.tick_params(labelsize=6)
    _ax.set(xlabel="tick", ylabel="stage", title="GPipe, 4 stages, 8 micro-batches (blue forward, orange backward)")
    _bub = " · ".join(f"m={m}: {100 * ex.bubble_fraction(4, m):.0f}%" for m in (1, 4, 8, 16, 64))
    mo.vstack([mo.md("| ranks | AdamW state per rank (measured) | your training_memory, ZeRO-1 | max difference "
                     "between ranks | max difference from 1 rank |\n|---|---|---|---|---|\n" + _rows),
               _fig, mo.md(f"bubble with 4 stages: {_bub}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the ranks agree exactly with each other: why? the multi-rank runs differ
    slightly from the 1-rank run: where does the difference come from, given that the
    mathematics is identical? (think about the order of additions in the all-reduce and in a
    mean over 64 examples versus two means over 32.) in the schedule, where is the bubble, and
    why does it shrink as $m$ grows?
    """)
    return


if __name__ == "__main__":
    app.run()
