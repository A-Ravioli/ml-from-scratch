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
    # variational inference · lab

    every experiment here runs **your** `reparam_vi` and `cavi_gmm` from `exercise.py`. for
    each one: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how wide is the mean-field posterior?

    the target is a 2-D Gaussian with unit variances and correlation $\rho$ (the slider). we
    fit a **diagonal** Gaussian $q$ with your `reparam_vi` (3,000 Adam steps, learning rate
    0.01) and compare its
    standard deviations with the target's true marginal standard deviations (both 1), for
    several $\rho$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("at rho = 0.9, what standard deviation does mean-field VI report for x1 (the truth is 1)?")
    rho1 = mo.ui.slider(0.0, 0.98, value=0.9, step=0.02, label="ρ")
    mo.vstack([p1, rho1])
    return p1, rho1


@app.function
def fit_correlated(rho, family="diagonal", seed=0):
    P = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))
    q, hist = ex.reparam_vi(lambda z: -0.5 * z @ P @ z, 2, grad_log_prob=lambda z: -P @ z, family=family,
                            lr=0.01, n_iters=3000, rng=seed)
    return q, hist


@app.cell
def _(p1, rho1):
    ui.gate(p1)
    _rhos = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]
    _rows = []
    _sds = []
    for _r in _rhos:
        _q, _ = fit_correlated(_r)
        _sd = np.sqrt(np.diag(_q.cov))
        _sds.append(_sd.mean())
        _rows.append(f"| {_r} | {_sd[0]:.3f} | {_sd[1]:.3f} | {np.sqrt(1 - _r**2):.3f} | 1 |")
    _q, _ = fit_correlated(rho1.value)
    _qf, _ = fit_correlated(rho1.value, family="full")
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _t = np.linspace(0, 2 * np.pi, 200)
    _circle = np.stack([np.cos(_t), np.sin(_t)])
    for _C, _style, _lab in [(np.array([[1, rho1.value], [rho1.value, 1]]), "k-", "target"),
                             (_q.cov, "C3--", "mean-field q"), (_qf.cov, "C0:", "full-covariance q")]:
        _L = np.linalg.cholesky(_C + 1e-12 * np.eye(2))
        for _k in (1, 2):
            _e = _k * _L @ _circle
            _axes[0].plot(_e[0], _e[1], _style, lw=1.2, label=_lab if _k == 1 else None)
    _axes[0].set(aspect="equal", xlim=(-3, 3), ylim=(-3, 3), title=f"1 and 2 sd contours, ρ = {rho1.value}",
                 xlabel="x1", ylabel="x2")
    _axes[0].legend(fontsize=8)
    _rr = np.linspace(0, 0.98, 100)
    _axes[1].plot(_rr, np.sqrt(1 - _rr**2), "k:", label="√(1 − ρ²)")
    _axes[1].plot(_rhos, _sds, "o", color="C3", label="mean-field sd (fitted)")
    _axes[1].axhline(1, color="0.6", lw=1, label="true marginal sd")
    _axes[1].set(xlabel="ρ", ylabel="standard deviation of x1")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| ρ | q sd of x1 | q sd of x2 | √(1 − ρ²) | true sd |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** mean-field VI minimizes $\mathrm{KL}(q \,\|\, p) = \mathbb E_q[\log q - \log p]$.
    where does that expectation punish $q$: where $q$ puts mass and $p$ doesn't, or the reverse?
    why does a product of independent Gaussians then shrink to fit *inside* the narrow ellipse,
    and why is the answer $\sqrt{1 - \rho^2} = 1/\sqrt{\Lambda_{ii}}$ (the *conditional* sd, not the
    marginal)? the mean is still exactly right. which downstream decisions would this
    overconfidence break?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: which way round is the KL?

    a bimodal 1-D target: $0.4\,\mathcal N(-d/2, 1) + 0.6\,\mathcal N(d/2, 1)$, with the
    separation $d$ on the slider. we fit a single Gaussian two ways: with your `reparam_vi`
    (which minimizes the *reverse* KL, $\mathrm{KL}(q\|p)$; 3,000 steps), from six random
    seeds, also reporting each fit's $-\mathrm{KL}(q\|p)$; and by
    **moment matching**, which minimizes the *forward* KL $\mathrm{KL}(p\|q)$ (for a Gaussian
    family the answer is simply the target's mean and variance).
    """)
    return


@app.cell
def _():
    p2 = ui.predict("with the modes 7 apart, does VI's Gaussian sit between the modes, cover both, or pick one? "
                    "and the moment-matched one?")
    d2 = mo.ui.slider(0.0, 9.0, value=7.0, step=0.5, label="separation d")
    mo.vstack([p2, d2])
    return d2, p2


@app.cell
def _(d2, p2):
    ui.gate(p2)
    _d = d2.value
    _logz = 0.5 * np.log(2 * np.pi)  # the unnormalized target below integrates to sqrt(2 pi)
    _w = np.array([0.4, 0.6])
    _mu = np.array([-_d / 2, _d / 2])

    def _logp(z):
        a = np.log(_w) - 0.5 * (z[0] - _mu) ** 2
        return float(np.logaddexp(a[0], a[1]))

    def _grad(z):
        a = np.log(_w) - 0.5 * (z[0] - _mu) ** 2
        r = np.exp(a - np.logaddexp(a[0], a[1]))
        return np.array([np.sum(r * (_mu - z[0]))])

    _fits = []
    for _s in range(6):
        _q, _h = ex.reparam_vi(_logp, 1, grad_log_prob=_grad, lr=0.05, n_iters=3000, rng=_s)
        _fits.append((float(_q.mean[0]), float(np.sqrt(_q.cov[0, 0])), float(np.mean(_h[-300:])) - _logz))
    _mm_mean = float(_w @ _mu)
    _mm_sd = float(np.sqrt(_w @ (1 + _mu**2) - _mm_mean**2))
    _x = np.linspace(-9, 9, 600)
    _p = sum(w * np.exp(-0.5 * (_x - m) ** 2) / np.sqrt(2 * np.pi) for w, m in zip(_w, _mu))
    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _ax.fill_between(_x, _p, color="0.85", label="target")
    for _k, (_m, _s, _) in enumerate(_fits):
        _ax.plot(_x, np.exp(-0.5 * ((_x - _m) / _s) ** 2) / (_s * np.sqrt(2 * np.pi)), "C3", lw=1,
                 alpha=0.8, label="VI, reverse KL (6 seeds)" if _k == 0 else None)
    _ax.plot(_x, np.exp(-0.5 * ((_x - _mm_mean) / _mm_sd) ** 2) / (_mm_sd * np.sqrt(2 * np.pi)), "C0--",
             label="moment matching, forward KL")
    _ax.set(xlabel="z", ylim=(0, 0.45))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| VI seed {k} | {m:+.2f} | {s:.2f} | {e:.3f} |" for k, (m, s, e) in enumerate(_fits))
    mo.vstack([_fig, mo.md("| fit | mean | sd | ELBO − log Z = −KL(q ‖ p) |\n|---|---|---|---|\n" + _rows
                           + f"\n| moment matching | {_mm_mean:+.2f} | {_mm_sd:.2f} | |\n\n"
                           f"a Gaussian sitting exactly on the larger mode scores about log 0.6 = {np.log(0.6):.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the reverse KL is an expectation under $q$; the forward KL is an expectation
    under $p$. which one becomes huge if $q$ puts mass in the valley between the modes, and
    which one becomes huge if $q$ misses a mode entirely? that explains "mode-seeking" versus
    "mass-covering". some seeds may still be stuck straddling the modes: compare their
    $-\mathrm{KL}$ with the single-mode fits. why does $q$, which starts at mean 0 with sd 1
    (in the valley), have trouble leaving? which behaviour would you want for a posterior over
    parameters, and which for a generative model of images? (move the slider to small $d$:
    when do the two fits agree?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: CAVI's many answers

    360 points from a 1-D mixture of 6 unit-variance Gaussians at $-10, -6, -2, 2, 6, 10$ (60
    each). we run your `cavi_gmm` with $K = 6$ from 30 random starts and look at the final
    ELBOs and the fitted means.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("out of 30 random starts, how many find the six true clusters?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _true = np.array([-10.0, -6.0, -2.0, 2.0, 6.0, 10.0])
    _x = np.concatenate([_rng.normal(m, 1.0, 60) for m in _true])
    _runs = [ex.cavi_gmm(_x, 6, rng=s) for s in range(30)]
    _final = np.array([r["elbo"][-1] for r in _runs])
    _found = np.array([np.allclose(np.sort(r["m"]), _true, atol=0.5) for r in _runs])
    _best, _worst = int(np.argmax(_final)), int(np.argmin(_final))
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    _axes[0].hist(_final, bins=15, color="C7")
    _axes[0].set(xlabel="final ELBO", ylabel="random starts")
    _grid = np.linspace(-14, 14, 500)
    for _ax, _i, _t in [(_axes[1], _best, "best start"), (_axes[2], _worst, "worst start")]:
        _ax.hist(_x, bins=50, density=True, color="0.85")
        _r = _runs[_i]
        _wts = _r["phi"].mean(0)
        for _k in range(6):
            _ax.plot(_grid, _wts[_k] * np.exp(-0.5 * (_grid - _r["m"][_k]) ** 2) / np.sqrt(2 * np.pi), lw=1.2)
        _ax.set(title=f"{_t}: ELBO {_final[_i]:.1f}", xlabel="x")
    _fig.tight_layout()
    _e = np.asarray(_runs[_best]["elbo"])
    mo.vstack([_fig, mo.md(
        f"**{int(_found.sum())} of 30** starts found all six clusters (means within 0.5). final ELBOs range from "
        f"{_final.min():.1f} to {_final.max():.1f}. best fit's means: {np.round(np.sort(_runs[_best]['m']), 2).tolist()}, "
        f"posterior sds of the means: {np.round(np.sqrt(_runs[_best]['s2']), 3).tolist()}; its ELBO rose in "
        f"{len(_e)} iterations and never decreased (largest drop {min(0.0, float(np.min(np.diff(_e)))):.1e}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each CAVI update can only increase the ELBO. why can it still get stuck with
    two components sharing one cluster and another component straddling two clusters? what
    would it have to do to escape, and why can't a coordinate-wise update do that? how does
    the ELBO tell you which run to trust? the posterior sds of the means are about $1/\sqrt{60}$:
    is that the full story of your uncertainty about this clustering?
    """)
    return


if __name__ == "__main__":
    app.run()
