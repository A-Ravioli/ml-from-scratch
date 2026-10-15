import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import gc
    import time
    import tracemalloc

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.autodiff import jacobian  # yours: forward mode with dual numbers, from ch 03 · topic 04
    from mlfs.tensor import Tensor  # yours, from topic 03

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # backprop in depth · lab

    every experiment here runs **your** code: `jacobian_reverse`, `graph_nbytes` and
    `checkpoint_sequential` from `exercise.py`, your `Tensor` engine, and your forward-mode
    `jacobian` from chapter 03. for each one: write down a prediction, lock it in, run it,
    then explain what happened. compare your explanations with `solutions/lab_notes.md`
    afterwards. the whole lab takes about ten seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: tall and wide Jacobians

    $f(x) = \tanh(A\tanh(x))$, with $A$ of shape $m \times n$, maps $\mathbb R^n \to \mathbb R^m$. we compute its full
    $m \times n$ Jacobian two ways: forward mode (your `Dual`-based `jacobian`: one pass per
    input) and reverse mode (your `jacobian_reverse`: one backward pass per output), for
    shapes from $n = 400, m = 1$ (a loss of 400 parameters) to $n = 1, m = 400$, with
    $n \cdot m = 400$ throughout.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which mode wins at each end, and by how much? where do they cross over? (your Dual works on NumPy "
        "object arrays, so one forward-mode pass is much slower than one NumPy pass. does that change the answer?)"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _shapes = [(400, 1), (40, 10), (20, 20), (10, 40), (1, 400)]
    _rows, _tf, _tr = [], [], []
    for _n, _m in _shapes:
        _A = _rng.normal(size=(_m, _n)) / np.sqrt(_n)
        _x = _rng.normal(size=_n)
        _t0 = time.perf_counter()
        _Jr = ex.jacobian_reverse(lambda t, A=_A: (t.tanh() @ A.T).tanh(), _x)
        _t1 = time.perf_counter()
        _Jf = jacobian(lambda v, A=_A: np.tanh(A @ np.tanh(v)), _x)
        _t2 = time.perf_counter()
        _tr.append(_t1 - _t0)
        _tf.append(_t2 - _t1)
        _rows.append(f"| {_n} × {_m} | {_n} | {_m} | {(_t2 - _t1) * 1e3:.1f} ms | {(_t1 - _t0) * 1e3:.1f} ms | "
                     f"{np.abs(_Jr - _Jf).max():.1e} |")
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _labels = [f"n={n}\nm={m}" for n, m in _shapes]
    _ax.semilogy(range(5), np.array(_tf) * 1e3, "o-", label="forward mode (n passes)")
    _ax.semilogy(range(5), np.array(_tr) * 1e3, "s-", label="reverse mode (m passes)")
    _ax.set_xticks(range(5), _labels)
    _ax.set(ylabel="time for the full Jacobian (ms)", title="n · m = 400 entries in every case")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| shape (inputs × outputs) | forward passes | backward passes | forward mode | reverse mode | "
                           "largest difference |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does reverse mode win by so much at $n = 400, m = 1$, and what does
    that case correspond to in deep learning? why does the gap close (and reverse mode lose,
    relative to its best case) as $m$ grows? one forward-mode pass with object arrays of
    `Dual`s costs far more than one NumPy pass: where in the table can you see that constant,
    and why doesn't it change which mode is right for a scalar loss?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: memory for compute

    a 16-layer tanh network of width 128 on a batch of 2,048: each layer is
    `h = (h @ W.T).tanh()`. we run forward + backward without checkpointing, and with your
    `checkpoint_sequential` at 1 to 16 segments, and measure (with `tracemalloc`) the memory
    held after the forward pass and the peak during forward + backward, in units of one
    activation ($2048 \times 128$ float64 = 2 MB); plus the number of layer evaluations and the
    time (best of 3).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which number of segments gives the lowest peak memory? is checkpointing with 1 segment better or worse "
        "than not checkpointing? how much slower is checkpointed training?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _L, _d, _n = 16, 128, 2048
    _Ws = [Tensor(_rng.normal(size=(_d, _d)) / np.sqrt(_d), requires_grad=True) for _ in range(_L)]
    _calls = [0]

    def _layer(W):
        def f(h):
            _calls[0] += 1
            return (h @ W.T).tanh()
        return f

    _fs = [_layer(W) for W in _Ws]
    _X = _rng.normal(size=(_n, _d))
    _act = _n * _d * 8

    def _plain(x):
        for f in _fs:
            x = f(x)
        return x

    def _measure(build):
        for W in _Ws:
            W.grad = None
        _calls[0] = 0
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        out = build()
        held = tracemalloc.get_traced_memory()[0] - base
        graph = ex.graph_nbytes(out)
        out.sum().backward()
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        calls = _calls[0]
        best = np.inf
        for _ in range(3):
            gc.collect()
            t0 = time.perf_counter()
            build().sum().backward()
            best = min(best, time.perf_counter() - t0)
        return held / _act, graph / _act, peak / _act, calls, best

    _results = {"none": _measure(lambda: _plain(Tensor(_X)))}
    for _s in [1, 2, 4, 6, 8, 16]:
        _results[_s] = _measure(lambda s=_s: ex.checkpoint_sequential(_fs, Tensor(_X), s))
    _rows = [f"| {k} | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} | {r[3]} | {r[4] * 1e3:.0f} ms |" for k, r in _results.items()]
    _segs = [s for s in _results if s != "none"]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(_segs, [_results[s][2] for s in _segs], "o-", label="peak, forward + backward")
    _axes[0].plot(_segs, [_results[s][0] for s in _segs], "s-", label="held after forward")
    _axes[0].axhline(_results["none"][2], color="C0", ls=":", label="peak without checkpointing")
    _axes[0].axhline(_results["none"][0], color="C1", ls=":", label="held without checkpointing")
    _s = np.linspace(1, 16, 100)
    _axes[0].plot(_s, _s + 3 * _L / _s, "k--", lw=1, label="model s + 3L/s")
    _axes[0].set(xlabel="segments s", ylabel="memory (activations of 2 MB)")
    _axes[0].legend(fontsize=8)
    _axes[1].bar(["none"] + [str(s) for s in _segs], [_results[k][4] * 1e3 for k in _results])
    _axes[1].set(xlabel="segments", ylabel="forward + backward time (ms)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| segments | held after forward | graph_nbytes | peak | layer evaluations | time |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows)
                           + "\n\n(memory in units of one 2 MB activation)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the peak with 1 segment no better than without checkpointing,
    even though it holds almost nothing after the forward pass? why does the peak fall and
    then rise again as $s$ grows, and how well does the model $s + 3L/s$ (from on paper 2)
    predict where the minimum is? why does every checkpointed run evaluate exactly 32 layers,
    and how does that match the measured times?
    """)
    return


if __name__ == "__main__":
    app.run()
