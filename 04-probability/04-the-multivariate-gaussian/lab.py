import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def corr_from_cov(C):
    sd = np.sqrt(np.diag(C))
    return C / np.outer(sd, sd)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the multivariate gaussian · lab

    every experiment here runs **your** `MultivariateNormal` from `exercise.py`. for each one:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: conditioning is slicing

    a 2-D Gaussian with standard deviations 1 (horizontal) and 2 (vertical) and correlation
    $\rho$. the vertical line is the slice $x_1 = v$. on the right: the joint density along
    that line, rescaled to integrate to 1, against your `condition([0], [v])`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with ρ = 0.8, how does the conditional variance of x2 change as you move the slice from "
        "v = 0 to v = 2.5? how does the conditional mean move?"
    )
    p1
    return (p1,)


@app.cell
def _():
    rho = mo.ui.slider(-0.95, 0.95, value=0.8, step=0.05, label="correlation ρ", show_value=True)
    v_slice = mo.ui.slider(-3.0, 3.0, value=1.5, step=0.25, label="slice x1 = v", show_value=True)
    mo.hstack([rho, v_slice])
    return rho, v_slice


@app.cell
def _(p1, rho, v_slice):
    ui.gate(p1)
    _S = np.array([[1.0, 2.0 * rho.value], [2.0 * rho.value, 4.0]])
    _g = ex.MultivariateNormal([0.0, 0.0], _S)
    _t1, _t2 = np.linspace(-3.5, 3.5, 141), np.linspace(-7, 7, 141)
    _xx, _yy = np.meshgrid(_t1, _t2)
    _dens = np.asarray(_g.pdf(np.column_stack([_xx.ravel(), _yy.ravel()]))).reshape(_xx.shape)
    _v = v_slice.value
    _slice = np.asarray(_g.pdf(np.column_stack([np.full_like(_t2, _v), _t2])))
    _slice = _slice / (_slice.sum() * (_t2[1] - _t2[0]))
    _c = _g.condition([0], [_v])
    _cond = np.asarray(_c.pdf(_t2[:, None]))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    _a1.contour(_xx, _yy, _dens, levels=8, cmap="Blues")
    _a1.axvline(_v, color="C3")
    _a1.set(xlabel="x1", ylabel="x2", title="joint density and the slice")
    _a2.plot(_t2, _slice, lw=4, alpha=0.4, label="slice of the joint, renormalized")
    _a2.plot(_t2, _cond, "k--", label="your condition([0], [v]).pdf")
    _a2.set(xlabel="x2", title="p(x2 | x1 = v)")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _rows = []
    for _vv in (-2.0, 0.0, 1.0, 2.5):
        _cc = _g.condition([0], [_vv])
        _rows.append(f"| {_vv:+.1f} | {float(_cc.mean[0]):+.3f} | {float(_cc.cov[0, 0]):.3f} |")
    mo.vstack([_fig, mo.md(
        f"prior: x2 ~ N(0, 4). given x1 = {_v:+.2f}: mean **{float(_c.mean[0]):+.3f}**, variance "
        f"**{float(_c.cov[0, 0]):.3f}**.\n\n| v | conditional mean | conditional variance |\n|---|---|---|\n"
        + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the conditional variance the same for every slice? write it as a
    function of $\rho$ and explain what happens at $\rho = \pm 0.95$ and $\rho = 0$. what line
    does the conditional mean trace out as $v$ moves, and how is it related to the tilt of the
    ellipse?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a chain, and what the precision matrix knows

    five variables in a chain: $x_1 \sim \mathcal{N}(0, 1)$ and each next one is a noisy copy
    of the previous, $x_{k+1} = 0.9\,x_k + \sqrt{0.19}\,\varepsilon_k$. the covariance is
    $\Sigma_{ij} = 0.9^{|i - j|}$: every pair is correlated. we condition on the middle
    variable $x_3$ with your `condition`, and look at the correlations that remain.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "after observing x3, what is the correlation between x1 and x5? between x1 and x2? "
        "and if you observe x1 instead, between x3 and x5?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _k = np.arange(5)
    _S = 0.9 ** np.abs(_k[:, None] - _k[None, :])
    _g = ex.MultivariateNormal(np.zeros(5), _S)
    _c3 = _g.condition([2], [1.0])            # remaining: x1, x2, x4, x5
    _c1 = _g.condition([0], [1.0])            # remaining: x2, x3, x4, x5
    _R3, _R1 = corr_from_cov(np.asarray(_c3.cov)), corr_from_cov(np.asarray(_c1.cov))
    _P = np.linalg.inv(_S)
    _Pm = np.linalg.inv(np.asarray(_g.marginal([0, 1, 3, 4]).cov))

    def _table(M, labels):
        _head = "| | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n"
        return _head + "\n".join(f"| **{labels[i]}** | " + " | ".join(f"{M[i, j]:+.3f}" for j in range(len(labels))) + " |"
                                 for i in range(len(labels)))
    _x = ["x1", "x2", "x3", "x4", "x5"]
    mo.md(f"""
**correlation of (x1, x2, x4, x5) given x3** (your `condition`):

{_table(_R3, ["x1", "x2", "x4", "x5"])}

**correlation of (x2, x3, x4, x5) given x1**:

{_table(_R1, ["x2", "x3", "x4", "x5"])}

**the precision matrix Σ⁻¹ of all five**:

{_table(_P, _x)}

**the precision matrix of the marginal of (x1, x2, x4, x5)** (x3 integrated out):

{_table(_Pm, ["x1", "x2", "x4", "x5"])}
""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does observing $x_3$ cut every correlation *across* it to exactly zero,
    but observing $x_1$ cut nothing among the rest? the precision matrix has zeros exactly
    where... what? why does integrating out $x_3$ fill those zeros back in, while conditioning
    on it doesn't?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: where do high-dimensional samples live?

    5,000 samples from $\mathcal{N}(0, I_d)$, drawn with your `sample`. on the left, their
    distances from the mean. on the right, their log densities (your `logpdf`), with the log
    density at the mean marked.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "in d = 1000 dimensions: how far from the mean is a typical sample? what fraction of samples "
        "are within distance 25 of the mean? how does a typical sample's density compare with the "
        "density at the mean?"
    )
    p3
    return (p3,)


@app.cell
def _():
    dim = mo.ui.dropdown(options=["1", "2", "10", "100", "1000"], value="1000", label="dimension d")
    dim
    return (dim,)


@app.cell
def _(dim, p3):
    ui.gate(p3)
    _d = int(dim.value)
    _g = ex.MultivariateNormal(np.zeros(_d), np.eye(_d))
    _x = np.asarray(_g.sample(5000, rng=0))
    _r = np.linalg.norm(_x, axis=1)
    _lp = np.asarray(_g.logpdf(_x))
    _lp0 = _g.logpdf(np.zeros(_d))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    _a1.hist(_r, bins=60, alpha=0.7)
    _a1.axvline(np.sqrt(_d), color="C3", ls="--", label="√d")
    _a1.set(xlabel="distance from the mean ‖x‖", ylabel="count", xlim=(0, None))
    _a1.legend(fontsize=8)
    _a2.hist(_lp, bins=60, alpha=0.7)
    _a2.axvline(_lp0, color="C3", ls="--", label="log density at the mean")
    _a2.set(xlabel="log density of each sample", ylabel="count")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"d = {_d}: distances {_r.min():.2f} … {_r.max():.2f} (mean {_r.mean():.2f}, √d = {np.sqrt(_d):.2f}). "
        f"fraction within distance 25 of the mean: **{np.mean(_r < 25):.4f}**. log density at the mean "
        f"{_lp0:.1f}; median log density of a sample {np.median(_lp):.1f}, a ratio of "
        f"**e^{_lp0 - np.median(_lp):.0f}** in density.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the density is highest at the mean, yet no sample comes anywhere near it.
    reconcile the two. (how much *volume* is there at radius $r$ in $d$ dimensions?) what does
    this mean for a generative model that outputs the "most likely" image, or for
    interpolating between two samples along a straight line?
    """)
    return


if __name__ == "__main__":
    app.run()
