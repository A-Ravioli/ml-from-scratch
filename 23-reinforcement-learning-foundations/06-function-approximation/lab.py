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
    # function approximation · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes under a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: driving up the mountain

    your `linear_sarsa` with your `TileCoder` (8 tilings of $8\times8$ tiles over position
    and velocity) learns `MountainCar` (time limit 200 steps) for 150 episodes, with
    $\varepsilon = 0$ (no random exploration at all) and step sizes $\alpha = 0.1, 0.3, 0.5,
    0.9$ (divided by the 8 active features), 2 runs each. we plot the steps per episode
    (200 = didn't reach the top), and the learned cost-to-go $-\max_aQ(x, v)$ for
    $\alpha = 0.5$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with epsilon = 0 the agent never explores at random. does it still learn to reach the top? which alpha "
        "learns fastest, and what will the cost-to-go surface look like?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _tc = ex.TileCoder([-1.2, -0.07], [0.5, 0.07], n_tilings=8, tiles_per_dim=8)
    _curves, _w = {}, None
    for _alpha in [0.1, 0.3, 0.5, 0.9]:
        _runs = [ex.linear_sarsa(ex.MountainCar(), _tc, 150, alpha=_alpha, rng=s) for s in range(2)]
        _curves[_alpha] = -np.mean([r[1] for r in _runs], axis=0)
        if _alpha == 0.5:
            _w = _runs[0][0]
    _xs, _vs = np.linspace(-1.2, 0.5, 60), np.linspace(-0.07, 0.07, 60)
    _cost = np.array([[-_w[_tc(np.array([x, v]))].sum(axis=0).max() for x in _xs] for v in _vs])
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _k = np.ones(5) / 5
    for _alpha, _c in _curves.items():
        _axes[0].plot(np.convolve(_c, _k, mode="valid"), label=f"α = {_alpha}")
    _axes[0].set(xlabel="episode", ylabel="steps per episode (2 runs, smoothed)", ylim=(80, 205))
    _axes[0].legend(fontsize=8)
    _im = _axes[1].imshow(_cost, origin="lower", aspect="auto", extent=[-1.2, 0.5, -0.07, 0.07], cmap="magma")
    _fig.colorbar(_im, ax=_axes[1], label="−max_a Q (steps to go)")
    _axes[1].set(xlabel="position", ylabel="velocity", title="learned cost-to-go, α = 0.5, after 150 episodes")
    _fig.tight_layout()
    _rows = [f"| {a} | {c[:50].mean():.0f} | {c[-20:].mean():.0f} |" for a, c in _curves.items()]
    mo.vstack([_fig, mo.md("| α | mean steps, episodes 1–50 | mean steps, last 20 |\n|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with no random actions, what makes the agent explore? (what does $Q = 0$
    mean when every step costs $-1$?) where is the cost-to-go highest, and why there? why are
    the corners of the map (fast cars far from the top) still at 0 after 150 episodes? why is
    every curve flat at 200 for the first episodes, and why does $\alpha = 0.1$ barely move?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: coarse, fine, or both?

    a supervised warm-up that isolates what tile coding does. we fit $f(x) = \sin(2\pi x)$ on
    $[0, 1]$ from **40 noisy samples** with a linear model on tile features, trained by the
    same kind of update `linear_sarsa` uses ($w_{\text{active}} \mathrel{+}= \frac{\alpha}{n}
    (y - \hat y)$, 50 passes). three coders with different shapes:

    - 1 tiling of 8 tiles (coarse);
    - 1 tiling of 64 tiles (fine);
    - 8 tilings of 8 tiles (coarse tiles, offset: 64 features, like the fine one).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which coder gives the lowest error on a fine test grid? the fine single grid has as many features as "
        "the 8 x 8 coder: will it do as well?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _xtr = _rng.random(40)
    _ytr = np.sin(2 * np.pi * _xtr) + 0.1 * _rng.normal(size=40)
    _xte = np.linspace(0, 1, 400)
    _coders = {"1 × 8 tiles": ex.TileCoder(0.0, 1.0, n_tilings=1, tiles_per_dim=8),
               "1 × 64 tiles": ex.TileCoder(0.0, 1.0, n_tilings=1, tiles_per_dim=64),
               "8 tilings × 8 tiles": ex.TileCoder(0.0, 1.0, n_tilings=8, tiles_per_dim=8)}
    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _ax.plot(_xte, np.sin(2 * np.pi * _xte), "k--", lw=1, label="sin(2πx)")
    _ax.plot(_xtr, _ytr, "k.", ms=5, label="40 samples")
    _rows = []
    for _name, _tc in _coders.items():
        _w = np.zeros(_tc.n_features)
        _feats = [_tc(np.array([x])) for x in _xtr]
        for _ in range(50):
            for _i in _rng.permutation(40):
                _idx = _feats[_i]
                _w[_idx] += 0.5 / len(_idx) * (_ytr[_i] - _w[_idx].sum())
        _pred = np.array([_w[_tc(np.array([x]))].sum() for x in _xte])
        _ax.plot(_xte, _pred, label=_name)
        _rows.append(f"| {_name} | {_tc.n_features} | {np.sqrt(np.mean((_pred - np.sin(2 * np.pi * _xte)) ** 2)):.3f} | "
                     f"{np.mean([np.abs(_w[_tc(np.array([x]))]).sum() == 0 for x in _xte]):.0%} |")
    _ax.set(xlabel="x", ylabel="f(x)", ylim=(-1.6, 1.6))
    _ax.legend(fontsize=8, ncol=2)
    mo.vstack([_fig, mo.md("| coder | features | test RMS error | test points with no trained feature |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what limits each coder: resolution (how finely it can bend) or
    generalization (how far one sample's update reaches)? why does the 8 × 8 coder get both,
    with the same number of features as the fine grid? how would you expect the three to
    compare with 4,000 samples instead of 40?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the deadly triad

    Baird's counterexample (Sutton & Barto, Example 11.1). seven states, all rewards 0, so
    $w = 0$ (every value 0) is a perfect solution. the values are linear in 8 weights:
    $v(s) = 2w_s + w_8$ for states 1–6 and $v(7) = w_7 + 2w_8$. the target policy always
    moves to state 7; the behaviour policy visits all seven states equally. your
    `expected_td` runs the *expected* semi-gradient TD(0) update (no sampling noise at all)
    from $w_0 = (1, 1, 1, 1, 1, 1, 10, 1)$, $\alpha = 0.01$, in three settings: off-policy
    (uniform state weighting), on-policy (all weight on the states the target policy visits),
    and off-policy with *tabular* features (one weight per state).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the problem is deterministic, the update has no noise, and the right answer (w = 0) is representable. what "
        "happens to |w| in each of the three settings? does it depend on gamma?"
    )
    p3
    return (p3,)


@app.cell
def _():
    gamma3 = mo.ui.slider(0.80, 0.999, value=0.99, step=0.001, label="γ")
    gamma3
    return (gamma3,)


@app.cell
def _(gamma3, p3):
    ui.gate(p3)
    _Phi = np.zeros((7, 8))
    for _i in range(6):
        _Phi[_i, _i], _Phi[_i, 7] = 2.0, 1.0
    _Phi[6, 6], _Phi[6, 7] = 1.0, 2.0
    _P = np.zeros((7, 7))
    _P[:, 6] = 1.0
    _r = np.zeros(7)
    _w0 = np.array([1, 1, 1, 1, 1, 1, 10, 1.0])
    _g = gamma3.value
    _settings = {
        "off-policy, linear (the triad)": (_Phi, np.full(7, 1 / 7), _w0),
        "on-policy, linear": (_Phi, np.eye(7)[6], _w0),
        "off-policy, tabular": (np.eye(7), np.full(7, 1 / 7), _Phi @ _w0),
    }
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _rows = []
    for _name, (_F, _d, _start) in _settings.items():
        _W = ex.expected_td(_F, _P, _r, _d, _g, alpha=0.01, n_iters=5000, w0=_start)
        _ax.semilogy(np.linalg.norm(_W, axis=1), label=_name)
        _A, _ = ex.td_matrices(_F, _P, _r, _d, _g)
        _rows.append(f"| {_name} | {np.linalg.norm(_W[0]):.1f} | {np.linalg.norm(_W[-1]):.3g} | "
                     f"{np.linalg.eigvals(_A).real.min():+.4f} |")
    _ax.set(xlabel="expected update", ylabel="‖w‖", title=f"Baird's counterexample, γ = {_g:.3f}")
    _ax.legend(fontsize=8)
    _gs = np.linspace(0.8, 0.999, 400)
    _mins = [np.linalg.eigvals(ex.td_matrices(_Phi, _P, _r, np.full(7, 1 / 7), g)[0]).real.min() for g in _gs]
    _crit = _gs[int(np.argmax(np.array(_mins) < -1e-9))]
    mo.vstack([_fig, mo.md(
        "| setting | ‖w‖ at start | after 5,000 updates | smallest real part of eig(A) |\n|---|---|---|---|\n"
        + "\n".join(_rows) + f"\n\noff-policy linear TD starts to diverge for γ above about **{_crit:.3f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the expected update is $w \leftarrow w + \alpha(b - Aw)$: what property of
    $A$ decides whether it converges, and which setting breaks it? name the three ingredients
    of the "deadly triad" and show that removing any one of them (here: off-policy weighting
    or function approximation) restores convergence. (the tabular run converges too, but
    slowly: its slowest direction shrinks by only $\alpha(1 - \gamma)/7$ per update. which
    number in the table tells you so?) why does a *smaller* $\gamma$ save it?
    what does this predict for DQN (ch 24), which has all three?
    """)
    return


if __name__ == "__main__":
    app.run()
