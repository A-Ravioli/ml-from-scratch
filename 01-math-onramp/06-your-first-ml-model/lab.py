import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # your first ML model · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    the data: 60 houses with sizes between 50 and 250 m² and prices (in \$100k) that grow
    roughly linearly with size, plus noise.
    """)
    return


@app.cell
def _():
    _rng = np.random.default_rng(0)
    size = _rng.uniform(50, 250, 60)                           # square meters
    price = 0.02 * size + 0.5 + 0.3 * _rng.normal(size=60)     # $100k
    z, size_mean, size_std = ex.standardize(size)
    w_best, b_best = ex.closed_form_fit(size, price)
    best_loss = ex.mse_loss(size, price, w_best, b_best)

    def max_stable_lr(x):
        """2 / (the steepest curvature of the MSE in (w, b)). the curvatures are the eigenvalues
        of the 2x2 matrix of second derivatives H (chapter 02); for a 2x2 matrix they solve
        lam^2 - (trace) lam + (det) = 0, so the larger one is (T + sqrt(T^2 - 4D)) / 2."""
        H = 2 * np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])
        T = H[0, 0] + H[1, 1]
        D = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        return 2 / ((T + np.sqrt(T * T - 4 * D)) / 2)

    return b_best, best_loss, max_stable_lr, price, size, size_mean, size_std, w_best, z


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: learning rates

    gradient descent on the **standardized** sizes $z$ (mean 0, standard deviation 1), for
    50 steps from $(0, 0)$, with your `gradient_descent`. the slider picks a learning rate;
    four fixed ones are drawn for reference.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which of the learning rates 0.01, 0.3, 0.9 and 1.05 will converge, and how will "
        "each loss curve look? what's the largest learning rate that works?"
    )
    p1
    return (p1,)


@app.cell
def _():
    lr = mo.ui.slider(steps=[0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.01, 1.05, 1.2],
                      value=0.5, label="learning rate", show_value=True)
    lr
    return (lr,)


@app.cell
def _(best_loss, lr, p1, price, z):
    ui.gate(p1)
    _w_star, _ = ex.closed_form_fit(z, price)
    _fig, (_ax, _axw) = plt.subplots(1, 2, figsize=(12, 4))
    _rows = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _eta, _style in ((0.01, ":"), (0.3, "--"), (0.9, "-."), (1.05, "-"), (lr.value, "o-")):
            _w, _b, _losses = ex.gradient_descent(z, price, lr=_eta, n_steps=50)
            _label = f"lr = {_eta:g}" + (" (slider)" if _style == "o-" else "")
            _ax.semilogy(np.asarray(_losses) - best_loss + 1e-16, _style, ms=3, label=_label)
            _rows.append(f"| {_label} | {_losses[-1]:.4g} |")
            # the slope w after every step (one step at a time, so we can watch it move)
            _ws, _wb = [0.0], (0.0, 0.0)
            for _ in range(20):
                _wb = ex.gradient_descent(z, price, lr=_eta, n_steps=1, w0=_wb[0], b0=_wb[1])[:2]
                _ws.append(_wb[0])
            _axw.plot(_ws, _style, ms=3, label=_label)
    _ax.set(xlabel="step", ylabel="loss − best possible loss", ylim=(1e-16, 1e4))
    _ax.legend(fontsize=8)
    _axw.axhline(_w_star, color="k", lw=0.8, ls="--", label="best w")
    _axw.set(xlabel="step", ylabel="slope w", title="the slope w, first 20 steps",
             ylim=(-0.5 * abs(_w_star), 2.5 * abs(_w_star)))
    _axw.legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"best possible loss (closed form): **{best_loss:.4f}**\n\n"
                           "| learning rate | loss after 50 steps |\n|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for standardized inputs the curvature in each direction is 2. use the
    lesson's factor $(1 - \eta a)$ to explain each curve (the right panel shows the slope $w$
    itself, step by step): which converge smoothly, which
    zigzag, which explode, and why the boundary is exactly 1. which learning rate is fastest,
    and why isn't it the largest one that converges?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the same data, in square meters

    now the raw sizes, in m². from the data we compute the largest learning rate gradient
    descent can survive (2 divided by the steepest curvature), and train with 90% and 110%
    of it for 1,000 steps. for comparison: the standardized version from experiment 1, with
    learning rate 0.3 for 30 steps, converted back with your `unstandardize_params`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the raw run uses the largest learning rate that's safe, for 1,000 steps. will it "
        "find the best line? how will it compare with the standardized run's 30 steps?"
    )
    p2
    return (p2,)


@app.cell
def _(best_loss, b_best, max_stable_lr, p2, price, size, size_mean, size_std, w_best, z):
    ui.gate(p2)
    _limit = max_stable_lr(size)
    _rows = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _name, _eta in (("raw, 0.9 × limit", 0.9 * _limit), ("raw, 1.1 × limit", 1.1 * _limit)):
            _w, _b, _losses = ex.gradient_descent(size, price, lr=_eta, n_steps=1000)
            _rows.append(f"| {_name} (lr = {_eta:.2e}) | 1000 | {_w:.5g} | {_b:.4g} | {_losses[-1]:.3g} |")
    _wz, _bz, _lz = ex.gradient_descent(z, price, lr=0.3, n_steps=30)
    _w, _b = ex.unstandardize_params(_wz, _bz, size_mean, size_std)
    _rows.append(f"| standardized (lr = 0.3), converted back | 30 | {_w:.5f} | {_b:.4f} | "
                 f"{ex.mse_loss(size, price, _w, _b):.4g} |")
    _rows.append(f"| closed form | – | {w_best:.5f} | {b_best:.4f} | {best_loss:.4g} |")
    mo.md(f"largest stable learning rate on the raw sizes: **{_limit:.2e}**\n\n"
          "| run | steps | w | b | final loss |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the safe learning rate so tiny on raw sizes? at that learning
    rate, how far does the intercept $b$ move per step? (its curvature is 2.) which of
    $w$ and $b$ did the raw run get right, and which is still wrong? why does
    standardizing fix both problems at once, without changing the model?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the shape of the valley

    the loss $L(w, b)$ as a contour map (like a hiking map: each line is one loss level),
    with gradient descent's path drawn on it. left: raw sizes, 300 steps at 90% of the safe
    learning rate. right: standardized sizes, 30 steps at learning rate 0.3.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what shape will the contour lines have on each side? what path will gradient "
        "descent trace on each?"
    )
    p3
    return (p3,)


@app.cell
def _(max_stable_lr, p3, price, size, z):
    ui.gate(p3)

    def _path(x, y, eta, n):
        _pts = [(0.0, 0.0)]
        for _ in range(n):
            _w, _b, _ = ex.gradient_descent(x, y, lr=eta, n_steps=1, w0=_pts[-1][0], b0=_pts[-1][1])
            _pts.append((_w, _b))
        return np.array(_pts)

    def _contours(ax, x, y, ws, bs, path, title):
        _W, _B = np.meshgrid(ws, bs)
        _L = np.mean((_W[..., None] * x + _B[..., None] - y) ** 2, axis=-1)
        ax.contour(_W, _B, np.log10(_L), levels=25, cmap="viridis", linewidths=0.8)
        ax.plot(path[:, 0], path[:, 1], "o-", color="C3", ms=2, lw=1, label="gradient descent")
        ax.plot(*path[-1], "*", color="k", ms=10, label="where it stopped")
        ax.set(xlabel="w", ylabel="b", title=title)
        ax.legend(fontsize=7)
        return ax

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _contours(_axes[0], size, price, np.linspace(-0.005, 0.04, 200), np.linspace(-1.0, 2.0, 200),
              _path(size, price, 0.9 * max_stable_lr(size), 300), "raw sizes (m²)")
    # equal spans around the optimum and equal axis scales, so the shape of the bowl is honest
    _wz, _bz = ex.closed_form_fit(z, price)
    _half = 1.15 * max(abs(_wz), abs(_bz))
    _contours(_axes[1], z, price, np.linspace(_wz - _half, _wz + _half, 200),
              np.linspace(_bz - _half, _bz + _half, 200), _path(z, price, 0.3, 30),
              "standardized sizes (equal axis scales)").set_aspect("equal")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the gradient is always perpendicular to the contour lines. use that to
    explain the raw path: why does it shoot almost straight to the valley floor, and then
    hardly move along it? why is the standardized bowl round? (the axes on the left are
    stretched to make the valley visible: in equal units it would be thousands of times
    longer than it is wide.)
    """)
    return


if __name__ == "__main__":
    app.run()
