import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    HAS_MPS = torch.backends.mps.is_available()
    HAS_CUDA = torch.cuda.is_available()


@app.function
def best_time(fn, sync=None, repeats=7, warmup=3):
    """the fastest of `repeats` timings of fn() in seconds, after warmup calls, synchronizing the device."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
    return min(ts)


@app.function
def eager_softmax(x):
    """eager PyTorch's softmax written out as its five ops: each one a separate kernel and a trip to memory."""
    m = x.amax(dim=-1, keepdim=True)
    e = (x - m).exp()
    return e / e.sum(dim=-1, keepdim=True)


@app.function
def eager_softmax_interp(x, out, n_rows, n_cols):
    """the same five ops as five interpreter kernels; intermediates go to Buffers (DRAM) between them."""
    t, e = ex.Buffer.empty(n_rows * n_cols), ex.Buffer.empty(n_rows * n_cols)
    m, s = ex.Buffer.empty(n_rows), ex.Buffer.empty(n_rows)
    cols = np.arange(n_cols)

    def rowmax(pid):
        ex.store(m, [pid[0]], ex.load(x, pid[0] * n_cols + cols).max())

    def sub(pid):
        ex.store(t, pid[0] * n_cols + cols, ex.load(x, pid[0] * n_cols + cols) - ex.load(m, [pid[0]]))

    def exp(pid):
        ex.store(e, pid[0] * n_cols + cols, np.exp(ex.load(t, pid[0] * n_cols + cols)))

    def rowsum(pid):
        ex.store(s, [pid[0]], ex.load(e, pid[0] * n_cols + cols).sum())

    def div(pid):
        ex.store(out, pid[0] * n_cols + cols, ex.load(e, pid[0] * n_cols + cols) / ex.load(s, [pid[0]]))

    for k in (rowmax, sub, exp, rowsum, div):
        ex.launch(k, (n_rows,))
    return [x, t, e, m, s, out]


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # writing kernels · lab

    the experiments use **your** kernels from `exercise.py`: the interpreter versions for
    traffic counts, and your Metal softmax for the race on a real GPU. for each experiment:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards.

    Apple GPU found: **{HAS_MPS}**. {"" if HAS_MPS else "without one, the Metal parts are skipped: read the lab notes for what they show."}
    {"an NVIDIA GPU was found: experiment 1 times the real thing on it too, and that's where Triton itself runs (`pip install triton`; the lesson's Triton kernels are the same programs as your interpreter's)." if HAS_CUDA else ""}
    timings are from **your machine, now**: the ratios matter, not the absolute numbers. about
    a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what fusion saves

    a $64 \times 300$ matrix softmaxed by your fused kernel and by eager PyTorch's five ops
    written as five interpreter kernels, with every element loaded and stored counted. then
    the real thing: `eager_softmax` (five PyTorch ops) against `torch.softmax` (one fused
    kernel) on 16M floats, on the CPU and the GPU.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("the eager version moves 4× the bytes. how much slower will it be on the CPU? on the GPU?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _xs = _rng.normal(size=(64, 300)).astype(np.float32)
    _x, _out = ex.Buffer(_xs), ex.Buffer.empty(64 * 300)
    ex.softmax(_x, _out, 64, 300)
    _fused = _x.loads + _out.stores
    _bufs = eager_softmax_interp(ex.Buffer(_xs), ex.Buffer.empty(64 * 300), 64, 300)
    _eager = sum(b.loads + b.stores for b in _bufs)
    _devs = ["cpu"] + (["cuda"] if HAS_CUDA else []) + (["mps"] if HAS_MPS else [])
    _rows = []
    for _d in _devs:
        _sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(_d)
        _xt = torch.randn(4096, 4096, device=_d)
        _te = best_time(lambda: eager_softmax(_xt), _sync)
        _tf = best_time(lambda: torch.softmax(_xt, -1), _sync)
        _rows.append(f"| {_d} | {1e3 * _te:.2f} ms | {1e3 * _tf:.2f} ms | **{_te / _tf:.1f}×** | "
                     f"{2 * _xt.numel() * 4 / _tf / 1e9:.0f} GB/s |")
    mo.md(f"interpreter traffic for 64 × 300: fused **{_fused:,}** elements, eager **{_eager:,}** "
          f"(**{_eager / _fused:.2f}×**).\n\n| device | eager (5 ops) | torch.softmax (fused) | eager / fused | "
          "fused, effective bandwidth (2 passes) |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the traffic ratio is a hair above 4: where do the extra elements come
    from? does the timing ratio match the traffic ratio on each device? compare the fused
    kernel's effective bandwidth with what you measured in topic 01's lab: is it memory-bound?
    if one device's ratio is far from the traffic ratio, what other cost is there per op?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: tiling, counted and timed

    your tiled `matmul` on a $128 \times 128 \times 128$ problem with tiles from 4 to 64,
    counting loads. then (on an Apple GPU) two Metal kernels written for this lab: a naive one,
    one thread per output element reading its row and column straight from DRAM, and a tiled
    one, $16 \times 16$ threadgroups staging tiles in threadgroup memory, exactly the structure
    of your interpreter kernel. both race PyTorch's own matmul.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("on the GPU, how much faster will the 16×16 tiled Metal matmul be than the naive one? and how close "
                    "will it get to torch's matmul?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _n = 128
    _A, _B = _rng.normal(size=(_n, _n)).astype(np.float32), _rng.normal(size=(_n, _n)).astype(np.float32)
    traffic = []
    for _b in (4, 8, 16, 32, 64):
        _a, _bb, _c = ex.Buffer(_A), ex.Buffer(_B), ex.Buffer.empty(_n * _n)
        ex.matmul(_a, _bb, _c, _n, _n, _n, block_m=_b, block_n=_b, block_k=16)
        _ok = np.allclose(_c.data.reshape(_n, _n), _A @ _B, atol=1e-3)
        traffic.append((_b, _a.loads + _bb.loads, _ok))
    _naive = 2 * _n**3
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _ax.plot([t[0] for t in traffic], [t[1] for t in traffic], "o-", label="your tiled kernel")
    _ax.axhline(_naive, color="C3", ls="--", label="one element at a time")
    _ax.axhline(2 * _n * _n, color="k", ls=":", label="each input once")
    _ax.set(xscale="log", yscale="log", xlabel="tile side b", ylabel="elements loaded", title="128³ matmul")
    _ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
    _ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _ax.legend(fontsize=7)
    _tab = "\n".join(f"| {b} | {l:,} | {_naive / l:.0f}× | {ok} |" for b, l, ok in traffic)
    mo.vstack([_fig, mo.md("| tile | loads | fewer than naive | correct |\n|---|---|---|---|\n" + _tab)])
    return


@app.cell
def _(p2):
    ui.gate(p2)
    METAL_MATMUL = r"""
    #include <metal_stdlib>
    using namespace metal;
    #define TS 16

    kernel void matmul_naive(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                             device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
                             constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]]) {
        uint col = gid.x, row = gid.y;
        if (row >= M || col >= N) return;
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }

    kernel void matmul_tiled(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                             device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
                             constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],
                             uint2 tid [[thread_position_in_threadgroup]],
                             uint2 tg [[threadgroup_position_in_grid]]) {
        threadgroup float As[TS][TS];
        threadgroup float Bs[TS][TS];
        uint row = tg.y * TS + tid.y, col = tg.x * TS + tid.x;
        float acc = 0.0f;
        for (uint k0 = 0; k0 < K; k0 += TS) {
            As[tid.y][tid.x] = (row < M && k0 + tid.x < K) ? A[row * K + k0 + tid.x] : 0.0f;
            Bs[tid.y][tid.x] = (k0 + tid.y < K && col < N) ? B[(k0 + tid.y) * N + col] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < TS; ++k) acc += As[tid.y][k] * Bs[k][tid.x];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row < M && col < N) C[row * N + col] = acc;
    }
    """
    if not HAS_MPS:
        _out = mo.callout(mo.md("no Apple GPU: the Metal race is skipped. the lab notes show what it measured."), kind="info")
    else:
        _lib = ex.compile_metal(METAL_MATMUL)
        _rows = []
        for _n in (256, 1024, 2048):
            _a, _b = torch.randn(_n, _n, device="mps"), torch.randn(_n, _n, device="mps")
            _c = torch.empty(_n, _n, device="mps")
            _ref = _a @ _b
            _runs = {
                "naive Metal": lambda: _lib.matmul_naive(_a, _b, _c, _n, _n, _n, threads=(_n, _n), group_size=(16, 16)),
                "tiled Metal (16×16)": lambda: _lib.matmul_tiled(_a, _b, _c, _n, _n, _n, threads=(_n, _n),
                                                                 group_size=(16, 16)),
                "torch (a @ b)": lambda: torch.matmul(_a, _b, out=_c),
            }
            for _name, _fn in _runs.items():
                _t = best_time(_fn, torch.mps.synchronize, repeats=5)
                _err = (_c - _ref).abs().max().item() / _ref.abs().max().item()
                _rows.append(f"| {_n} | {_name} | {1e3 * _t:.2f} ms | {2 * _n**3 / _t / 1e9:.0f} | {_err:.0e} |")
        _out = mo.md("| n | kernel | time | GFLOP/s | max rel. error |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** by what factor does each doubling of the tile side cut the loads? on the
    GPU, does the tiled kernel's speedup over the naive one match the 16× traffic reduction?
    (hint: the naive kernel's neighbouring threads read the same row of $A$, and the GPU's
    caches notice.) what does PyTorch's matmul do that a 16×16 tiled kernel doesn't?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: your Metal softmax against PyTorch

    16M floats arranged as rows of 128 to 16,384 elements, softmaxed by your `metal_softmax`,
    eager PyTorch's five ops, and `torch.softmax`. (without an Apple GPU this compares the CPU
    versions of the last two only.)
    """)
    return


@app.cell
def _():
    p3 = ui.predict("for rows of 1,024, will your Metal kernel beat eager PyTorch's five ops? torch.softmax?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _d = "mps" if HAS_MPS else "cpu"
    _sync = torch.mps.synchronize if HAS_MPS else None
    race = []
    for _cols in (128, 512, 1024, 4096, 16384):
        _x = torch.randn(2**24 // _cols, _cols, device=_d)
        _r = {"eager (5 ops)": best_time(lambda: eager_softmax(_x), _sync),
              "torch.softmax": best_time(lambda: torch.softmax(_x, -1), _sync)}
        if HAS_MPS:
            _r["your Metal kernel"] = best_time(lambda: ex.metal_softmax(_x), _sync)
            _ok = torch.allclose(ex.metal_softmax(_x), torch.softmax(_x, -1), atol=1e-5)
            assert _ok, "your Metal softmax disagrees with torch.softmax"
        race.append((_cols, _r))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    for _name in race[0][1]:
        _ax.plot([c for c, _ in race], [2 * 2**24 * 4 / r[_name] / 1e9 for _, r in race], "o-", label=_name)
    _ax.set(xscale="log", xlabel="row length", ylabel="effective GB/s (2 passes)", title=f"16M floats on {_d}")
    _ax.set_xticks([c for c, _ in race], [str(c) for c, _ in race])
    _ax.legend(fontsize=8)
    _hdr = list(race[0][1])
    _tab = "\n".join(f"| {c} | " + " | ".join(f"{1e3 * r[k]:.2f}" for k in _hdr) + " |" for c, r in race)
    mo.vstack([_fig, mo.md("| row length | " + " | ".join(f"{k} (ms)" for k in _hdr) + " |\n|---|"
                           + "---|" * len(_hdr) + "\n" + _tab)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does your kernel do against each opponent, and does it change with row
    length? your kernel reads each row three times (max, sum, write): why doesn't that cost
    three passes of DRAM traffic for short rows, and what happens when rows get long? if
    `torch.softmax` wins, list what its authors could do that your kernel doesn't (look at the
    reductions).
    """)
    return


if __name__ == "__main__":
    app.run()
