import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.autodiff import grad as forward_grad  # yours: forward mode with dual numbers, from ch 03 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # scalar reverse-mode autodiff · lab

    every experiment here runs **your** `Value` from `exercise.py`, and in experiment 1 also
    your forward-mode `grad` from chapter 03. for each one: write down a prediction, lock it
    in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes about ten seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: one gradient, two modes

    the function

    $$f(x) = \sum_{i=1}^{n}\Big[\tanh(x_ix_{i+1}) + \tfrac12x_i^2\Big]\qquad(x_{n+1} = x_1)$$

    is written once, in plain Python with `*`, `+`, `**` and `.tanh()`, so it runs on your
    `Value`s *and* on your chapter 03 `Dual`s. we compute its gradient at a random point with
    reverse mode (`value_and_grad`: one forward pass that records the graph, one backward
    pass) and with forward mode (`grad`: one pass per input, each seeding one dual part), for
    $n$ from 10 to 300 inputs (best of 3 runs each).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how does the time of each mode grow with n? at n = 300, which is faster, and by roughly how much?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    def _f(v):
        n = len(v)
        s = 0.0
        for i in range(n):
            s = s + (v[i] * v[(i + 1) % n]).tanh() + 0.5 * v[i] ** 2
        return s

    _ns = [10, 25, 50, 100, 200, 300]
    _tr, _tf, _rows = [], [], []
    for _n in _ns:
        _x = np.random.default_rng(0).normal(size=_n)
        _rev, _fwd = [], []
        for _ in range(3):  # best of 3 runs, to reduce timing noise
            _t0 = time.perf_counter()
            _val, _g = ex.value_and_grad(_f, _x)
            _t1 = time.perf_counter()
            _g2 = forward_grad(_f, _x)
            _t2 = time.perf_counter()
            _rev.append(_t1 - _t0)
            _fwd.append(_t2 - _t1)
        _tr.append(min(_rev))
        _tf.append(min(_fwd))
        _rows.append(f"| {_n} | {_tr[-1] * 1e3:.2f} ms | {_tf[-1] * 1e3:.1f} ms | {_tf[-1] / _tr[-1]:.0f}× | "
                     f"{np.abs(_g - _g2).max():.1e} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.loglog(_ns, np.array(_tf) * 1e3, "o-", label="forward mode (n passes)")
    _ax.loglog(_ns, np.array(_tr) * 1e3, "s-", label="reverse mode (1 forward + 1 backward)")
    _ref = np.array(_ns, dtype=float)
    _ax.loglog(_ref, _tf[-1] * 1e3 * (_ref / _ref[-1]) ** 2, "k:", lw=1, label="slope 2")
    _ax.loglog(_ref, _tr[-1] * 1e3 * (_ref / _ref[-1]), "k--", lw=1, label="slope 1")
    _ax.set(xlabel="number of inputs n", ylabel="time for the gradient (ms)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| n | reverse mode | forward mode | forward / reverse | largest difference |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does forward mode's time grow like $n^2$ and reverse mode's like $n$
    (how many passes does each make, and how much does each pass cost)? both give the same
    gradient to $10^{-16}$: when would you still prefer forward mode? why is reverse mode
    the only practical choice for a network with a million weights and one loss?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the price of thinking in scalars

    one dense layer of a small MNIST network: 784 inputs, 32 tanh units, loss = the sum of
    the outputs, for **one** image. with your `Value`, every weight and pixel is a node, and
    the layer is a double loop. we time one forward + backward pass, and compare it with the
    same computation in NumPy (forward: `tanh(W @ x)`; backward, written by hand:
    `outer(1 - h**2, x)` and `W.T @ (1 - h**2)`).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "how many nodes does your Value graph have for this one layer and one image, and how many times slower "
        "than NumPy is it?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _W = _rng.normal(size=(32, 784)) / 28.0
    _x = _rng.random(784)
    _t0 = time.perf_counter()
    _xs = [ex.Value(v) for v in _x]
    _Ws = [[ex.Value(w) for w in row] for row in _W]
    _out = 0.0
    for _j in range(32):
        _z = 0.0
        for _i in range(784):
            _z = _z + _Ws[_j][_i] * _xs[_i]
        _out = _out + _z.tanh()
    _out.backward()
    _tv = time.perf_counter() - _t0
    _gW = np.array([[w.grad for w in row] for row in _Ws])

    def _numpy_layer():
        h = np.tanh(_W @ _x)
        gz = 1.0 - h ** 2
        return np.outer(gz, _x), _W.T @ gz

    _gW_np, _ = _numpy_layer()
    _reps = 2000
    _t0 = time.perf_counter()
    for _ in range(_reps):
        _numpy_layer()
    _tn = (time.perf_counter() - _t0) / _reps
    _nodes = ex.graph_size(_out)
    _epoch_h = 60_000 * _tv / 3600
    mo.md(f"""
    | | your `Value` engine | NumPy |
    |---|---|---|
    | forward + backward, one image | {_tv * 1e3:.0f} ms | {_tn * 1e6:.1f} µs |
    | graph nodes | {_nodes:,} | 0 (three arrays) |
    | slowdown | **{_tv / _tn:,.0f}×** | 1× |
    | largest gradient difference | {np.abs(_gW - _gW_np).max():.1e} | |

    one epoch of MNIST (60,000 images) through this single layer would take your scalar
    engine about **{_epoch_h:.0f} hours**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where does the time go in the scalar engine (count the Python objects and
    closures created per multiply-add)? NumPy does the same arithmetic: why is it thousands of
    times faster? what would you change about the *unit* of the graph to keep reverse mode
    but lose the overhead? (that's topic 03.)
    """)
    return


if __name__ == "__main__":
    app.run()
