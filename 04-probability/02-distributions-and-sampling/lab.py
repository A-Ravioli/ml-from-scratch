import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.class_definition
class CountingGenerator(np.random.Generator):
    """A Generator that counts how many uniforms .random() hands out."""

    def __init__(self, seed):
        super().__init__(np.random.PCG64(seed))
        self.n_uniforms = 0

    def random(self, *args, **kwargs):
        out = super().random(*args, **kwargs)
        self.n_uniforms += int(np.size(out))
        return out


@app.function
def broken_normal(loc=0.0, scale=1.0, size=None, *, rng=None):
    """Box–Muller with a classic bug: the 2 in sqrt(-2 log u1) is missing (for comparison only)."""
    rng = np.random.default_rng(rng)
    n = int(np.prod(size))
    u1, u2 = 1.0 - rng.random(n), rng.random(n)
    return (loc + scale * np.sqrt(-np.log(u1)) * np.cos(2 * np.pi * u2)).reshape(size)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # distributions & sampling · lab

    every experiment here runs **your** samplers from `exercise.py`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: QQ plots against NumPy

    a **QQ plot** puts the quantiles of one sample (the 1st, 2nd, …, 99th percentiles) against
    the quantiles of another. if the two samples come from the same distribution, the points
    lie on the diagonal. here every one of your samplers faces NumPy's built-in sampler for the
    same distribution, 50,000 draws each. the last panel is a deliberately broken normal
    sampler, Box–Muller without the 2.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "will your samplers' QQ plots be straight diagonal lines? what will the broken normal's "
        "plot look like, and would you notice it in a histogram?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _n = 50_000
    _np = np.random.default_rng(100)
    _cases = [
        ("normal(1, 2)", lambda r: ex.normal(1.0, 2.0, _n, rng=r), _np.normal(1.0, 2.0, _n)),
        ("exponential(3)", lambda r: ex.exponential(3.0, _n, rng=r), _np.exponential(1 / 3.0, _n)),
        ("gamma(0.5)", lambda r: ex.gamma(0.5, 1.0, _n, rng=r), _np.gamma(0.5, 1.0, _n)),
        ("gamma(4, 2)", lambda r: ex.gamma(4.0, 2.0, _n, rng=r), _np.gamma(4.0, 2.0, _n)),
        ("beta(2, 5)", lambda r: ex.beta(2.0, 5.0, _n, rng=r), _np.beta(2.0, 5.0, _n)),
        ("poisson(4)", lambda r: ex.poisson(4.0, _n, rng=r), _np.poisson(4.0, _n)),
        ("poisson(300)", lambda r: ex.poisson(300.0, _n, rng=r), _np.poisson(300.0, _n)),
        ("BROKEN normal(0, 1)", lambda r: broken_normal(0.0, 1.0, _n, rng=r), _np.normal(0.0, 1.0, _n)),
    ]
    _q = np.linspace(0.5, 99.5, 100)
    _fig, _axes = plt.subplots(2, 4, figsize=(12, 6))
    _rows = []
    for (_label, _mine, _theirs), _ax in zip(_cases, _axes.ravel()):
        _x = np.asarray(_mine(np.random.default_rng(7)), dtype=float)
        _a, _b = np.percentile(_x, _q), np.percentile(_theirs, _q)
        _lo, _hi = min(_a.min(), _b.min()), max(_a.max(), _b.max())
        _ax.plot([_lo, _hi], [_lo, _hi], color="gray", lw=0.8)
        _ax.plot(_b, _a, ".", ms=3, color="C3" if "BROKEN" in _label else "C0")
        _ax.set_title(_label, fontsize=9)
        _ax.set_xlabel("NumPy quantiles", fontsize=8)
        _ax.set_ylabel("your quantiles", fontsize=8)
        _rows.append(f"| {_label} | {_x.mean():.4f} | {np.mean(_theirs):.4f} | {_x.std():.4f} | {np.std(_theirs):.4f} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| sampler | your mean | NumPy mean | your std | NumPy std |\n|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the broken sampler's QQ plot is still a straight line. why straight, and
    what does its slope tell you? (what distribution *does* it sample?) why are the Poisson
    plots staircases? and why would a histogram of the broken sampler look perfectly
    plausible on its own?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: rejection sampling in high dimensions

    target: the standard normal in $d$ dimensions, $\mathcal{N}(0, I_d)$. proposal: a normal
    that's a little wider in every direction, $\mathcal{N}(0, \sigma^2 I_d)$. the ratio
    $p(x)/q(x)$ is largest at the origin, where it equals $\sigma^d$, so the tightest valid
    envelope is $M = \sigma^d$. we draw 200 samples with your `rejection_sample` and count the
    proposals it needed.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with σ = 1.1, what fraction of proposals is accepted in d = 1? d = 10? d = 50? d = 100?"
    )
    p2
    return (p2,)


@app.cell
def _():
    sigma = mo.ui.slider(1.02, 1.5, value=1.1, step=0.02, label="proposal width σ", show_value=True)
    sigma
    return (sigma,)


@app.function
def high_dim_acceptance(d, sigma, n_samples, seed):
    """run your rejection_sample on N(0, I_d) with a N(0, sigma^2 I_d) proposal.

    returns (acceptance rate, mean |x|^2 / d of the accepted draws). the acceptance rate is the
    average acceptance probability over every proposal made, which doesn't depend on how your
    sampler sizes its batches.
    """
    probs = []

    def log_p(x):
        return -0.5 * np.sum(x**2, axis=1) - 0.5 * d * math.log(2 * math.pi)

    def log_q(x):
        return -0.5 * np.sum(x**2, axis=1) / sigma**2 - d * math.log(sigma) - 0.5 * d * math.log(2 * math.pi)

    def sample_q(n, r):
        x = ex.normal(0.0, sigma, (n, d), rng=r)
        probs.append(np.exp(log_p(x) - d * math.log(sigma) - log_q(x)))
        return x

    x = ex.rejection_sample(log_p, sample_q, log_q, d * math.log(sigma), n_samples, rng=seed)
    return float(np.mean(np.concatenate(probs))), float(np.mean(np.sum(x**2, axis=1)) / d)


@app.cell
def _(p2, sigma):
    ui.gate(p2)
    _s = sigma.value
    _dims = [1, 2, 5, 10, 20, 30, 40, 50]
    _rows, _measured = [], []
    for _d in _dims:
        _theory = _s ** (-_d)
        if 200 / _theory > 3_000_000:  # too expensive to run: report theory only
            _rows.append(f"| {_d} | {_theory:.2e} | (skipped: ~{200 / _theory:,.0f} proposals) | |")
            _measured.append(np.nan)
            continue
        _rate, _r2 = high_dim_acceptance(_d, _s, 200, seed=_d)
        _measured.append(_rate)
        _rows.append(f"| {_d} | {_theory:.2e} | {_rate:.2e} | {_r2:.3f} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _grid = np.arange(1, 101)
    _ax.semilogy(_grid, _s ** (-_grid.astype(float)), label="theory 1/M = σ^(-d)")
    _ax.semilogy(_dims, _measured, "o", label="measured (your rejection_sample)")
    _ax.set(xlabel="dimension d", ylabel="acceptance rate")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"σ = {_s:.2f}. at d = 100 the acceptance rate would be **{_s ** -100:.1e}**: about "
        f"{_s ** 100:,.0f} proposals per accepted sample.\n\n"
        "| d | theory 1/M | measured | mean of ‖x‖²/d (should be ≈ 1) |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the proposal is only 10% wider than the target in each direction. why
    does that cost grow *exponentially* in $d$? (where, in $d$ dimensions, is the envelope
    $Mq$ much bigger than $p$?) what does this suggest about sampling posteriors with
    thousands of parameters?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the cost of exactness

    every sampler you wrote is exact, but they pay for exactness in very different
    currencies. a counting generator records how many uniforms each of your samplers consumes
    to produce 20,000 draws.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "uniforms per draw for: normal, exponential, gamma(3), gamma(0.5), beta(2, 5), "
        "poisson(5), poisson(30), poisson(31). which is the most expensive?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _n = 20_000
    _cases = [
        ("normal", lambda r: ex.normal(0.0, 1.0, _n, rng=r)),
        ("exponential", lambda r: ex.exponential(1.0, _n, rng=r)),
        ("bernoulli(0.3)", lambda r: ex.bernoulli(0.3, _n, rng=r)),
        ("gamma(3)", lambda r: ex.gamma(3.0, 1.0, _n, rng=r)),
        ("gamma(0.5)", lambda r: ex.gamma(0.5, 1.0, _n, rng=r)),
        ("beta(2, 5)", lambda r: ex.beta(2.0, 5.0, _n, rng=r)),
        ("poisson(5)", lambda r: ex.poisson(5.0, _n, rng=r)),
        ("poisson(30)", lambda r: ex.poisson(30.0, _n, rng=r)),
        ("poisson(31)", lambda r: ex.poisson(31.0, _n, rng=r)),
        ("poisson(1000)", lambda r: ex.poisson(1000.0, _n, rng=r)),
    ]
    _rows = []
    for _label, _f in _cases:
        _g = CountingGenerator(3)
        _f(_g)
        _rows.append(f"| {_label} | {_g.n_uniforms / _n:.2f} |")
    mo.md("| sampler | uniforms per draw |\n|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** account for each number: why exactly 1 for the normal (Box–Muller uses two
    uniforms…)? why a little over 2 for gamma(3)? why does the Poisson cost jump between
    λ = 30 and λ = 31? when would you prefer Knuth's method anyway?
    """)
    return


if __name__ == "__main__":
    app.run()
