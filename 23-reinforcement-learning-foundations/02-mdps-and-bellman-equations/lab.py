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
    # MDPs & bellman equations · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how far does the agent see?

    a $6\times6$ `GridWorld` (goal bottom-right, $-1$ per step) and the uniformly random
    policy. we compute $V^\pi$ with your `policy_evaluation` for four discount factors and
    draw it as a heat map, one panel per $\gamma$, each with its own colour scale.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what is V in the far corner (0, 0) for gamma = 0.5, 0.9, 0.99 and 1? for which gammas does the map "
        "still show where the goal is?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _g = ex.GridWorld(6, 6)
    _pi = np.full((36, 4), 0.25)
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.4))
    _rows = []
    for _ax, _gamma in zip(_axes, [0.5, 0.9, 0.99, 1.0]):
        _V = ex.policy_evaluation(_g.P, _g.R, _pi, gamma=_gamma, max_iters=100_000)
        _im = _ax.imshow(_V.reshape(6, 6), cmap="viridis")
        _ax.set(title=f"γ = {_gamma}", xticks=[], yticks=[])
        _fig.colorbar(_im, ax=_ax, fraction=0.046)
        _bound = "—" if _gamma == 1 else f"{-1 / (1 - _gamma):.1f}"
        _rows.append(f"| {_gamma} | {_V[0]:.2f} | {_V[34]:.2f} | {_V[0] - _V[34]:.3f} | {_bound} |")
    _fig.suptitle("V of the random policy on a 6×6 grid (goal bottom-right)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| γ | V(far corner) | V(next to goal) | difference | −1/(1−γ) |\n|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the far corner's value sit right at $-1/(1 - \gamma)$ for small
    $\gamma$? use the effective horizon $1/(1 - \gamma)$ to explain which states can "see"
    the goal. with $\gamma = 1$, why is the far corner worth about $-170$ on a grid where the
    goal is only 10 steps away? try the slider below for intermediate values.
    """)
    return


@app.cell
def _():
    gamma_s = mo.ui.slider(0.0, 0.999, value=0.95, step=0.001, label="γ")
    gamma_s
    return (gamma_s,)


@app.cell
def _(gamma_s, p1):
    ui.gate(p1)
    _g = ex.GridWorld(6, 6)
    _V = ex.policy_evaluation(_g.P, _g.R, np.full((36, 4), 0.25), gamma=gamma_s.value, max_iters=100_000)
    _fig, _ax = plt.subplots(figsize=(4, 3.4))
    _im = _ax.imshow(_V.reshape(6, 6), cmap="viridis")
    _fig.colorbar(_im, ax=_ax)
    _ax.set(title=f"γ = {gamma_s.value:.3f}, horizon ≈ {1 / (1 - gamma_s.value):.0f} steps", xticks=[], yticks=[])
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: pay the agent to walk

    a $5\times5$ grid with $\gamma = 0.95$ and three policies: **direct** (go right along the
    top row, then down: 8 steps), **wall** (always "up": it never leaves the top-left
    corner) and **random**. we evaluate them with `step_reward = -1` (the usual cost per
    step) and with `step_reward = +1` (every step pays). reaching the goal is worth 0 either way.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with -1 per step, direct is clearly best. which policy is best with +1 per step? "
        "(isn't adding a constant to every reward supposed to leave the best policy unchanged?)"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _direct = np.zeros((25, 4))
    for _s in range(25):
        _direct[_s, 1 if _s % 5 < 4 else 2] = 1.0
    _wall = np.zeros((25, 4))
    _wall[:, 0] = 1.0
    _pols = {"direct": _direct, "wall": _wall, "random": np.full((25, 4), 0.25)}
    _rows = []
    for _sr in [-1.0, 1.0]:
        _g = ex.GridWorld(5, 5, step_reward=_sr)
        _vals = {k: ex.policy_evaluation(_g.P, _g.R, p, gamma=0.95)[_g.start_state] for k, p in _pols.items()}
        _best = max(_vals, key=_vals.get)
        _rows.append(f"| {_sr:+.0f} | " + " | ".join(f"{v:.3f}" for v in _vals.values()) + f" | **{_best}** |")
    mo.md("| reward per step | V(start): direct | wall | random | best |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the second row is exactly the first with the sign flipped. why? on paper
    4 you showed that a constant shift changes nothing in a task that never ends: which
    assumption fails here, and which state's value didn't shift? what does this tell you
    about designing rewards ("+1 for staying alive" in a game that you want the agent to
    finish)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: rollouts versus the Bellman equation

    the random policy on the default $4\times4$ grid, $\gamma = 1$. `policy_evaluation`
    gives the exact expected return $V^\pi(\text{start})$. we also estimate it the
    model-free way: the average `return` of 2,000 seeded `rollout`s, once with the
    gridworld's default `max_steps=100` and once with no time limit (`max_steps=None`).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "do the two rollout averages agree with V(start) to within their standard errors? if not, which one is off, "
        "and in which direction?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _g = ex.GridWorld()
    _V = ex.policy_evaluation(_g.P, _g.R, np.full((16, 4), 0.25), gamma=1.0)
    _rng = np.random.default_rng(0)
    _pol = lambda obs: int(_rng.integers(4))  # noqa: E731
    _res = {}
    for _ms in [100, None]:
        _out = [ex.rollout(ex.GridWorld(max_steps=_ms), _pol, seed=i) for i in range(2000)]
        _rets = np.array([o["return"] for o in _out])
        _res[_ms] = (_rets, np.mean([o["truncated"] for o in _out]))
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _bins = np.arange(-300, 1, 10)
    _ax.hist(_res[None][0], bins=_bins, alpha=0.5, label="no time limit")
    _ax.hist(_res[100][0], bins=_bins, alpha=0.5, label="max_steps = 100")
    _ax.axvline(_V[0], color="k", ls="--", label=f"V(start) = {_V[0]:.1f}")
    _ax.set(xlabel="return of one episode", ylabel="episodes")
    _ax.legend(fontsize=8)
    _rows = [f"| {'100' if k else 'none'} | {r.mean():.2f} ± {r.std(ddof=1) / np.sqrt(len(r)):.2f} | {tr:.1%} |"
             for k, (r, tr) in _res.items()]
    mo.vstack([_fig, mo.md(f"V(start) from `policy_evaluation`: **{_V[0]:.2f}**\n\n"
                           "| time limit | mean return ± s.e. | episodes truncated |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which estimate is biased, by how much, and why? which episodes does the
    time limit change, and what return do they get instead of their true one? why is the
    distribution of returns so skewed? an agent that learns from truncated episodes and
    treats them as terminated would learn which wrong values?
    """)
    return


if __name__ == "__main__":
    app.run()
