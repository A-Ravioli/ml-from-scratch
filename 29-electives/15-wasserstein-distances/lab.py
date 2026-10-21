import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # wasserstein distances · lab

    every experiment here runs **your** code: your 1-D closed form, cost matrices, LP and
    transportation simplex. for each one: write down a prediction, lock it in, run it, then
    explain what happened. compare your explanations with `solutions/lab_notes.md`
    afterwards. the lab takes about 5 seconds on one CPU core. all data are synthetic.

    since you've now built an exact OT solver, this lab also uses SciPy's
    `linear_sum_assignment` (the Hungarian algorithm, for uniform equal-size problems) where
    it needs speed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: Wasserstein vs KL and JS when supports don't overlap

    two bumps on a grid of 1,201 points over $[-6, 6]$: $p$ is a triangular bump of half-width
    $w$ centred at 0, and $q_\theta$ is the same bump centred at $\theta$. as $\theta$ moves
    from 0 to 5 we compute $\mathrm{KL}(p\,\|\,q_\theta)$, the Jensen–Shannon divergence
    $\mathrm{JS}(p, q_\theta)$ (natural log) and $W_1(p, q_\theta)$ with your
    `wasserstein_1d` (grid points as locations, bump values as weights). then we try to *fit*
    $\theta$: start at $\theta = 3$ and run 40 steps of gradient descent (step 0.25) on each
    divergence, with central finite differences for the gradient.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with bumps of half-width 0.5, how do KL, JS and W1 change as the shift grows from 1 to 4? which of the three "
        "can gradient descent use to move theta from 3 back to 0?"
    )
    p1
    return (p1,)


@app.cell
def _():
    width = mo.ui.slider(0.1, 2.0, value=0.5, step=0.1, label="bump half-width w")
    width
    return (width,)


@app.cell
def _(p1, width):
    ui.gate(p1)
    grid = np.linspace(-6, 6, 1201)

    def bump(c):
        h = np.clip(1 - np.abs(grid - c) / width.value, 0, None)
        return h / h.sum()

    def kl(p, q):
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p / q), 0.0)
        return float(np.sum(terms))

    def js(p, q):
        m = (p + q) / 2
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    def w1(p, q):
        keep_p, keep_q = p > 0, q > 0
        return ex.wasserstein_1d(grid[keep_p], grid[keep_q], 1, u_weights=p[keep_p], v_weights=q[keep_q])

    divs = {"KL(p ‖ q)": kl, "JS(p, q)": js, "W1(p, q)": w1}
    p0 = bump(0.0)
    thetas = np.linspace(0, 5, 101)
    curves = {k: np.array([f(p0, bump(t)) for t in thetas]) for k, f in divs.items()}

    def descend(f, theta=3.0, lr=0.25, steps=40, h=0.01):
        path = [theta]
        for _ in range(steps):
            grad = (f(p0, bump(theta + h)) - f(p0, bump(theta - h))) / (2 * h)
            if not np.isfinite(grad):
                break
            theta -= lr * grad
            path.append(theta)
        return np.array(path)

    paths = {k: descend(f) for k, f in divs.items()}
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for _k, _c in curves.items():
        _axes[0].plot(thetas, np.where(np.isfinite(_c), _c, np.nan), label=_k)
    _axes[0].set(xlabel=r"shift $\theta$", ylabel="divergence", ylim=(0, 5.5),
                 title=f"half-width w = {width.value:g}")
    _axes[0].text(2.5, 4.6, "KL = ∞ for every θ > 0\n(not drawn)", ha="center", fontsize=8, color="C0")
    _axes[0].legend(fontsize=8, loc="upper left")
    _axes[1].plot(grid, p0, label="p")
    _axes[1].plot(grid, bump(2.0), label=r"$q_\theta$, $\theta = 2$")
    _axes[1].set(xlim=(-3, 4), xlabel="x", title="the two bumps")
    _axes[1].legend(fontsize=8)
    for _k, _pth in paths.items():
        _axes[2].plot(_pth, "o-", ms=3, label=_k)
    _axes[2].set(xlabel="gradient step", ylabel=r"$\theta$", title=r"fitting $\theta$ from 3")
    _axes[2].legend(fontsize=8)
    _fig.tight_layout()
    _at = {t: int(np.argmin(np.abs(thetas - t))) for t in (0.5, 1, 2, 4)}
    _rows = [f"| {k} | " + " | ".join(f"{c[i]:.3f}" if np.isfinite(c[i]) else "∞" for i in _at.values())
             + f" | {paths[k][-1]:.3f} ({len(paths[k]) - 1} steps"
             + (", gradient not finite" if len(paths[k]) == 1 else "") + ") |" for k, c in curves.items()]
    mo.vstack([_fig, mo.md("| divergence | θ = 0.5 | θ = 1 | θ = 2 | θ = 4 | θ after descent |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is KL infinite as soon as $\theta > 0$ (look at where $q_\theta$ is
    zero), and why is JS stuck at exactly $\log 2 \approx 0.693$ once the bumps stop
    overlapping? what does that do to the finite-difference gradient? why does $W_1$ grow
    *linearly* in $\theta$, whatever the width? move the slider: what changes for JS when the
    bumps are wide, and what doesn't change for $W_1$? this is the argument of Arjovsky et
    al. (2017) for training GANs with a Wasserstein critic: real images and early generator
    samples live on thin, non-overlapping sets.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what exactness costs

    uniform point clouds of $n$ points in the unit square (and $n$ more as targets), squared
    Euclidean cost. we solve the same problem with your `transport_simplex`, your
    `kantorovich_lp` (HiGHS on the $n^2$-variable LP) and SciPy's Hungarian algorithm, check
    that they agree, and time them (best of 2; your simplex only up to $n = 64$). for scale:
    your `wasserstein_1d` on two 1-D samples of $10^5$ points each.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "each time n doubles, by what factor does the time to solve the n x n LP grow? how large an n can your "
        "simplex handle in a second?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def best_time(fn, reps=2):
        out, best = None, np.inf
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        return out, best

    _g = np.random.default_rng(0)
    timing = {"your simplex": {}, "your LP (HiGHS)": {}, "Hungarian (SciPy)": {}}
    _rows = []
    for _n in (8, 16, 32, 64, 128, 256):
        _X, _Y = _g.random((_n, 2)), _g.random((_n, 2))
        _C = ex.cost_matrix(_X, _Y, 2)
        _a = np.full(_n, 1 / _n)
        (_, _c_lp), timing["your LP (HiGHS)"][_n] = best_time(lambda: ex.kantorovich_lp(_a, _a, _C))
        (_r, _c), timing["Hungarian (SciPy)"][_n] = best_time(lambda: linear_sum_assignment(_C), reps=3)
        _c_h = _C[_r, _c].mean()
        _c_s = np.nan
        if _n <= 64:
            (_P, _f, _gg), timing["your simplex"][_n] = best_time(lambda: ex.transport_simplex(_a, _a, _C), reps=1)
            _c_s = float(np.sum(_P * _C))
        _rows.append(f"| {_n} | {_n * _n:,} | {_c_lp:.5f} | {_c_h:.5f} | "
                     + ("–" if np.isnan(_c_s) else f"{_c_s:.5f}") + " | "
                     + " | ".join(f"{timing[k][_n] * 1000:.1f}" if _n in timing[k] else "–" for k in timing) + " |")
    _u, _v = _g.normal(size=100_000), _g.normal(size=100_000) + 0.3
    _w, t_1d = best_time(lambda: ex.wasserstein_1d(_u, _v, 2))
    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _k, _d in timing.items():
        _ax.loglog(list(_d), [1000 * v for v in _d.values()], "o-", label=_k)
    _ax.set(xlabel="n (points per side)", ylabel="time (ms)", title="exact OT on n × n problems")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| n | LP variables | LP cost | Hungarian cost | simplex cost | simplex ms | LP ms | Hungarian ms |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\n1-D closed form on 10⁵ + 10⁵ points: W2 = {_w:.4f} in {t_1d * 1000:.0f} ms.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** fit a slope to each curve on the log–log plot. why does the LP get so much
    more expensive than the $n^2$ size of its input (what does each simplex or interior-point
    iteration cost, and how many are there)? why is the Hungarian algorithm faster on the
    same problem (what structure does it exploit that a general LP solver doesn't)? and why is
    the 1-D case essentially free? a minibatch of 256 images per training step is the scale
    topic 17 needs; topic 16 makes it cheap by giving up exactness.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how fast do empirical distances converge?

    draw two *independent* samples of $n$ points from the **same** distribution
    $\mathcal N(0, I_d)$ and compute the exact $W_2$ between them (Hungarian algorithm on your
    `cost_matrix`). the true distance is 0, so whatever you measure is pure sampling error.
    5 repetitions per point; the bars are ± one standard deviation. for comparison, the
    dashed line is the *true* $W_2$ between $\mathcal N(0, I_d)$ and $\mathcal N(0.5\cdot\mathbf
    1/\sqrt d, I_d)$, a mean shift of length 0.5, from your `gaussian_w2` (0.5 in every $d$).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "two samples of 800 points from the same 10-dimensional Gaussian: is their W2 closer to 0.05, 0.5 or 3? "
        "when n grows 32-fold, how much does it shrink in 1, 2 and 10 dimensions?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _g = np.random.default_rng(1)
    ns = [25, 50, 100, 200, 400, 800]
    dims = [1, 2, 5, 10]
    floor = {}
    for _d in dims:
        for _n in ns:
            _vals = []
            for _ in range(5):
                _X, _Y = _g.normal(size=(_n, _d)), _g.normal(size=(_n, _d))
                _C = ex.cost_matrix(_X, _Y, 2)
                _r, _c = linear_sum_assignment(_C)
                _vals.append(np.sqrt(_C[_r, _c].mean()))
            floor[_d, _n] = (np.mean(_vals), np.std(_vals))
    _fig, _ax = plt.subplots(figsize=(6.5, 4.2))
    _rows = []
    for _i, _d in enumerate(dims):
        _m = np.array([floor[_d, _n][0] for _n in ns])
        _s = np.array([floor[_d, _n][1] for _n in ns])
        _ax.errorbar(ns, _m, yerr=_s, fmt="o-", color=f"C{_i}", label=f"d = {_d}", capsize=2)
        _true = ex.gaussian_w2(np.zeros(_d), np.eye(_d), np.full(_d, 0.5 / np.sqrt(_d)), np.eye(_d))
        _slope = np.polyfit(np.log(ns), np.log(_m), 1)[0]
        _rows.append(f"| {_d} | {_m[0]:.3f} | {_m[-1]:.3f} | {_m[0] / _m[-1]:.1f}× | {_slope:.2f} | {_true:.3f} |")
    _ax.axhline(_true, color="k", ls="--", lw=0.8, label="true W2 of a 0.5 mean shift (any d)")
    _ax.set(xscale="log", yscale="log", xlabel="n (points per sample)",
            ylabel=r"$W_2$ between two samples of $\mathcal{N}(0, I_d)$", title="sampling error of empirical W2")
    _ax.minorticks_off()
    _ax.set_xticks(ns, [str(v) for v in ns])
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| d | W2 at n = 25 | W2 at n = 800 | shrinks by | log–log slope | true W2 of a 0.5 shift |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the empirical $W_p$ converges like $n^{-1/d}$ in dimension $d > 2p$
    (Fournier & Guillin, 2015): compare with your slopes. why must it be slow? (how far is a
    typical point of one sample from its nearest point of the other, when $n$ points fill a
    $d$-dimensional ball?) why is the 1-D curve so much better? what does this mean for using
    a minibatch $W_2$ between 256 generated and 256 real images as a quality score, and for
    any claim that one model is "closer" to the data than another? this floor comes back in
    topic 17, where the project measures sample quality with exactly this distance.
    """)
    return


if __name__ == "__main__":
    app.run()
