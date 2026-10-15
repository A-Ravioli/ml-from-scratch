import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import contextlib
    import io
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.optim import SGD  # yours, from ch 06 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # a tensor autograd engine · lab

    every experiment here runs **your** `Tensor` from `exercise.py` (and your `SGD` from
    chapter 06, which works on it unchanged). for each one: write down a prediction, lock it
    in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes about ten seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a wrong gradient that still trains

    a linear regression, $\hat y = Xw + b$, on $n$ points with 5 features; the true bias is
    3. the bias $b$ has shape $(1,)$ and is broadcast over the $n$ rows, so the add's VJP
    must *sum* the gradient over them. we train with three versions of that add, built with
    your `Tensor.from_op` and `unbroadcast`:

    - **correct**: `unbroadcast(g, b.shape)` (a sum over the rows);
    - **mean bug**: `g.mean(axis=0)` (an average, $n$ times too small);
    - **first-row bug**: `g[:1]` (only the first example's gradient).

    300 steps of SGD with learning rate 0.05 on the mean squared error, then your
    `gradcheck` on each version.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with the two buggy gradients, does the training loss still go down? does the bias get near 3? "
        "which versions will gradcheck flag?"
    )
    p1
    return (p1,)


@app.cell
def _():
    n1 = mo.ui.slider(20, 1000, value=200, step=20, label="number of training points n")
    n1
    return (n1,)


@app.cell
def _(n1, p1):
    ui.gate(p1)

    def _make_add(kind):
        def add(a, b):
            def vjp(g):
                if kind == "correct":
                    gb = ex.unbroadcast(g, b.shape)
                elif kind == "mean bug":
                    gb = g.mean(axis=0, keepdims=True).reshape(b.shape)
                else:
                    gb = g[:1].reshape(b.shape)
                return ex.unbroadcast(g, a.shape), gb
            return ex.Tensor.from_op(a.data + b.data, (a, b), vjp)
        return add

    _rng = np.random.default_rng(0)
    _n, _d = n1.value, 5
    _X = _rng.normal(size=(_n, _d))
    _y = _X @ _rng.normal(size=_d) + 3.0 + 0.1 * _rng.normal(size=_n)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _rows = []
    for _kind in ["correct", "mean bug", "first-row bug"]:
        _add = _make_add(_kind)
        _w = ex.Tensor(np.zeros(_d), requires_grad=True)
        _b = ex.Tensor(np.zeros(1), requires_grad=True)
        _opt = SGD([_w, _b], lr=0.05)
        _losses = []
        for _ in range(300):
            _opt.zero_grad()
            _loss = ((_add(ex.Tensor(_X) @ _w, _b) - _y) ** 2).mean()
            _loss.backward()
            _opt.step()
            _losses.append(_loss.item())
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            _ok = ex.gradcheck(lambda w, b, add=_add: add(ex.Tensor(_X[:8]) @ w, b),
                               [ex.Tensor(_rng.normal(size=_d), requires_grad=True),
                                ex.Tensor(_rng.normal(size=1), requires_grad=True)])
        _ax.semilogy(_losses, label=_kind)
        _rows.append(f"| {_kind} | {_losses[0]:.2f} | {_losses[-1]:.3f} | {_b.data[0]:.2f} | "
                     f"{'✅ passes' if _ok else '❌ fails'} |")
    _ax.set(xlabel="SGD step", ylabel="training MSE (log scale)", title=f"linear regression, n = {_n}")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| bias gradient | first loss | loss after 300 steps | learned bias (true 3) | gradcheck |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the loss decrease even with a wrong gradient? (think about the
    direction of the buggy bias gradient, not its size.) why is the mean bug's bias so far
    from 3 after 300 steps, and how does that change when you move the slider to larger
    $n$? in a big network, how would you ever notice a bug like this without `gradcheck`?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the bug gradcheck can't see

    the same data, now with a linear layer in torch's layout: `pred = X @ W.T + b` with $W$
    of shape $(1, 5)$, so `pred` has shape $(n, 1)$, while the targets `y` have shape
    $(n,)$. we train with `loss = ((pred - y) ** 2).mean()`, exactly as written, and compare
    it with the same model trained on `pred.reshape(-1)`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what shape is pred - y? what will the model with the shape bug learn: something close to the right "
        "answer, something worse, or nothing at all? will gradcheck on its loss pass?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _n, _d = 200, 5
    _X = _rng.normal(size=(_n, _d))
    _y = _X @ _rng.normal(size=_d) + 3.0 + 0.3 * _rng.normal(size=_n)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _rows = []
    for _ax, (_name, _fix) in zip(_axes, [("pred.reshape(-1) - y", True), ("pred - y (shape bug)", False)]):
        _W = ex.Tensor(np.zeros((1, _d)), requires_grad=True)
        _b = ex.Tensor(np.zeros(1), requires_grad=True)
        _opt = SGD([_W, _b], lr=0.05)

        def _loss_fn(W, b, X=_X, y=_y, fix=_fix):
            pred = ex.Tensor(X) @ W.T + b
            diff = (pred.reshape(-1) if fix else pred) - y
            return (diff ** 2).mean(), diff.shape

        for _ in range(300):
            _opt.zero_grad()
            _loss, _shape = _loss_fn(_W, _b)
            _loss.backward()
            _opt.step()
        _pred = (_X @ _W.data.T + _b.data).ravel()
        _r2 = 1 - np.sum((_pred - _y) ** 2) / np.sum((_y - _y.mean()) ** 2)
        _ok = ex.gradcheck(lambda W, b, f=_loss_fn: f(W, b, _X[:10], _y[:10])[0],
                           [ex.Tensor(_rng.normal(size=(1, _d)), requires_grad=True),
                            ex.Tensor(_rng.normal(size=1), requires_grad=True)])
        _ax.scatter(_y, _pred, s=8, alpha=0.6)
        _lim = [_y.min(), _y.max()]
        _ax.plot(_lim, _lim, "k--", lw=1)
        _ax.set(xlabel="true y", title=_name)
        _rows.append(f"| `{_name}` | {_shape} | {_r2:.3f} | {np.abs(_W.data).max():.2g} | {_b.data[0]:.2f} | "
                     f"{'✅ passes' if _ok else '❌ fails'} |")
    _axes[0].set_ylabel("prediction")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"mean of y: {_y.mean():.2f}\n\n"
                           "| loss uses | shape of the difference | train $R^2$ | largest weight | bias | gradcheck |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what function of $W$ and $b$ does the buggy loss actually compute? (write
    it as a double sum over $i$ and $j$.) why is its minimizer $W = 0$, $b = \bar y$? why does
    `gradcheck` pass, and what does that tell you about what a gradient check can and can't
    verify? what one-line assertion in your loss function would have caught this?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what does a gradient cost?

    a 4-layer tanh network, all layers of the same width, on a batch of 256. we time a
    forward pass (under your `no_grad`) and a forward plus backward pass (best of 5), and
    estimate what the same gradient would cost by finite differences: $2n$ forward passes for
    $n$ weights.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "as the width grows from 16 to 1024 (1,000 to 4 million weights), how many times longer than a forward "
        "pass does forward + backward take? does that ratio grow with the number of weights?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _best(f, k=5):
        out = []
        for _ in range(k):
            t0 = time.perf_counter()
            f()
            out.append(time.perf_counter() - t0)
        return min(out)

    _rng = np.random.default_rng(2)
    _widths, _ratios, _rows = [16, 64, 256, 1024], [], []
    for _w in _widths:
        _X = _rng.normal(size=(256, _w))
        _Ws = [ex.Tensor(_rng.normal(size=(_w, _w)) / np.sqrt(_w), requires_grad=True) for _ in range(4)]

        def _forward(Ws=_Ws, X=_X):
            h = ex.Tensor(X)
            for W in Ws:
                h = (h @ W.T).tanh()
            return h.sum()

        def _forward_only(fwd=_forward):
            with ex.no_grad():
                fwd()

        def _forward_backward(fwd=_forward, Ws=_Ws):
            for W in Ws:
                W.grad = None
            fwd().backward()

        _tf, _tb = _best(_forward_only), _best(_forward_backward)
        _n = sum(W.size for W in _Ws)
        _ratios.append(_tb / _tf)
        _fd = 2 * _n * _tf
        _fd_txt = f"{_fd:.2g} s" if _fd < 3600 else f"{_fd / 3600:.1f} hours"
        _rows.append(f"| {_w} | {_n:,} | {_tf * 1e3:.2f} ms | {_tb * 1e3:.2f} ms | {_tb / _tf:.1f} | {_fd_txt} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.semilogx(_widths, _ratios, "o-")
    _ax.axhline(3, color="gray", ls=":", label="theory: about 3 (on paper 4)")
    _ax.set(xlabel="layer width", ylabel="(forward + backward) / forward", ylim=(0, max(5, max(_ratios) + 1)))
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| width | weights | forward | forward + backward | ratio | finite differences (estimate) |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the ratio stay a small constant while the number of weights grows
    4,000-fold? count the matrix products in the forward pass and in the backward pass. why is
    the ratio *below* 3 for small widths (what dominates the time there)? what would finite
    differences cost for the capstone's MNIST network, at a few milliseconds per forward
    pass and about 200,000 weights?
    """)
    return


if __name__ == "__main__":
    app.run()
