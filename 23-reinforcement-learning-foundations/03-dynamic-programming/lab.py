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
    # dynamic programming · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    experiments 1 and 2 use a random **continuing** MDP: 50 states, 4 actions, no terminal
    states, sparse random transitions (each row a Dirichlet(0.1) draw) and standard-normal
    rewards, so only the discount keeps the values finite.
    """)
    return


@app.cell
def _():
    _rng = np.random.default_rng(0)
    P50 = _rng.dirichlet(np.ones(50) * 0.1, size=(50, 4))
    R50 = _rng.normal(size=(50, 4))
    return P50, R50


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: iterations versus $\gamma$

    for $\gamma = 0.5, 0.9, 0.99, 0.999$ we run your `modified_policy_iteration` with
    $m = 1$ (value iteration), $m = 10$ and $m = 100$, and your `policy_iteration`, counting
    how many policies it evaluates (we wrap the `policy_evaluation` it calls with a counter).
    `tol = 1e-8` everywhere.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how does value iteration's iteration count grow as gamma goes from 0.9 to 0.999? and policy iteration's "
        "number of evaluations?"
    )
    p1
    return (p1,)


@app.cell
def _(P50, R50, p1):
    ui.gate(p1)
    _gammas = [0.5, 0.9, 0.99, 0.999]
    _calls = [0]
    _orig = ex.policy_evaluation

    def _counted(*args, **kwargs):
        _calls[0] += 1
        return _orig(*args, **kwargs)

    _table = {1: [], 10: [], 100: [], "PI": []}
    ex.policy_evaluation = _counted  # count the evaluations inside your policy_iteration
    try:
        for _g in _gammas:
            for _m in [1, 10, 100]:
                _table[_m].append(ex.modified_policy_iteration(P50, R50, gamma=_g, m=_m, max_iters=100_000)[2])
            _calls[0] = 0
            ex.policy_iteration(P50, R50, gamma=_g)
            _table["PI"].append(_calls[0])
    finally:
        ex.policy_evaluation = _orig
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _h = 1 / (1 - np.array(_gammas))
    for _k, _lab in [(1, "value iteration (m = 1)"), (10, "MPI, m = 10"), (100, "MPI, m = 100"),
                     ("PI", "policy iteration (evaluations)")]:
        _ax.loglog(_h, _table[_k], "o-", label=_lab)
    _ax.loglog(_h, _h * np.log(1e8), "k:", label="ln(1/tol) / (1 − γ)")
    _ax.set(xlabel="effective horizon 1 / (1 − γ)", ylabel="iterations (greedy steps)")
    _ax.legend(fontsize=8)
    _rows = [f"| {g} | " + " | ".join(str(_table[k][i]) for k in [1, 10, 100, "PI"]) + " |"
             for i, g in enumerate(_gammas)]
    mo.vstack([_fig, mo.md("| γ | VI | MPI m = 10 | MPI m = 100 | PI evaluations |\n|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does value iteration's count scale like $1/(1 - \gamma)$, and why is
    the slope on the log-log plot exactly 1? why does policy iteration's count barely move?
    MPI with $m = 10$ needs about 10 times fewer greedy steps than value iteration: does it
    do 10 times less *work*? what does each policy-iteration evaluation cost at
    $\gamma = 0.999$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the policy converges first

    the same MDP with $\gamma = 0.99$. we run value iteration by hand with your
    `bellman_optimality`, one backup at a time, and after each backup record the error
    $\|V_k - V^*\|_\infty$ and whether the greedy policy for $V_k$ is optimal (the same as
    `policy_iteration`'s answer).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "value iteration needs about 1,800 backups to reach tol = 1e-8 here. after how many backups is its greedy "
        "policy optimal for good?"
    )
    p2
    return (p2,)


@app.cell
def _(P50, R50, p2):
    ui.gate(p2)
    _gamma = 0.99
    _pi_star, _ = ex.policy_iteration(P50, R50, gamma=_gamma)
    _, _vstar = ex.value_iteration(P50, R50, gamma=_gamma, tol=1e-12, max_iters=100_000)
    _opt = np.argmax(_pi_star, axis=1)
    _V = np.zeros(50)
    _errs, _ok = [], []
    for _ in range(2500):
        _V = ex.bellman_optimality(P50, R50, _V, _gamma)
        _errs.append(np.max(np.abs(_V - _vstar)))
        _ok.append(np.array_equal(np.argmax(ex.greedy_policy(R50 + _gamma * P50 @ _V), axis=1), _opt))
    _ok = np.array(_ok)
    _last_bad = int(np.flatnonzero(~_ok)[-1]) + 1 if (~_ok).any() else 0
    _conv = int(np.argmax(np.array(_errs) < 1e-6)) + 1
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.semilogy(np.arange(1, 2501), _errs, label="‖V_k − V*‖∞")
    _ax.axvline(_last_bad + 1, color="g", ls="--", label=f"greedy policy optimal from backup {_last_bad + 1}")
    _ax.set(xlabel="backup k", ylabel="error", xscale="log")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"greedy policy optimal from backup **{_last_bad + 1}** on; the error at that point is "
        f"**{_errs[_last_bad]:.2f}** (the optimal values range over [{_vstar.min():.1f}, {_vstar.max():.1f}]). "
        f"the error first drops below 1e-6 at backup **{_conv}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how can the greedy policy be optimal while the values are still off by
    that much? what has to be right for the argmax to be right? if you only need the policy,
    what does this suggest about how long to run value iteration, and why is it still hard to
    *know* when to stop?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: near and small, or far and big?

    a chain of 11 states. states 0 and 10 are terminal; moving into state 0 pays $+1$,
    moving into state 10 pays $+10$, every other step pays 0; two actions, left and right,
    both deterministic. we solve it with your `value_iteration` for a range of $\gamma$ and
    show which way the optimal policy goes from each state.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "from state 2, below what gamma does the optimal policy go left to the +1? from state 4? from state 5? "
        "(on paper 4 has the answer for state 2)"
    )
    p3
    return (p3,)


@app.cell
def _():
    n_chain = 11
    P_chain = np.zeros((n_chain, 2, n_chain))
    R_chain = np.zeros((n_chain, 2))
    for _s in range(n_chain):
        if _s in (0, n_chain - 1):
            P_chain[_s, :, _s] = 1.0
            continue
        P_chain[_s, 0, _s - 1] = 1.0
        P_chain[_s, 1, _s + 1] = 1.0
    R_chain[1, 0] = 1.0
    R_chain[n_chain - 2, 1] = 10.0
    gamma3 = mo.ui.slider(0.3, 0.99, value=0.7, step=0.01, label="γ")
    gamma3
    return P_chain, R_chain, gamma3, n_chain


@app.cell
def _(P_chain, R_chain, gamma3, n_chain, p3):
    ui.gate(p3)
    _gs = np.linspace(0.3, 0.99, 139)
    _grid = np.array([np.argmax(ex.value_iteration(P_chain, R_chain, gamma=g, tol=1e-12)[0], axis=1) for g in _gs])
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    _axes[0].imshow(_grid[:, 1:-1].T, aspect="auto", origin="lower", cmap="coolwarm",
                    extent=[_gs[0], _gs[-1], 0.5, n_chain - 1.5])
    _axes[0].set(xlabel="γ", ylabel="state", title="optimal action (blue: left to +1, red: right to +10)")
    _pi, _V = ex.value_iteration(P_chain, R_chain, gamma=gamma3.value, tol=1e-12)
    _axes[1].bar(range(n_chain), _V, color=["C0" if a == 0 else "C3" for a in np.argmax(_pi, 1)])
    _axes[1].set(xlabel="state", ylabel="V*", title=f"γ = {gamma3.value:.2f}")
    _fig.tight_layout()
    _rows = []
    for _s in range(1, 6):
        _right = _grid[:, _s] == 1
        _at = "always right" if _right.all() else ("never" if not _right.any() else f"{_gs[np.argmax(_right)]:.3f}")
        _rows.append(f"| {_s} | {_at} |")
    mo.vstack([_fig, mo.md("| state | goes right (to +10) from γ ≈ |\n|---|---|\n" + "\n".join(_rows)
                           + f"\n\nfor state 2 the theory says 0.1^(1/6) = {0.1 ** (1 / 6):.3f} (the grid step is 0.005).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** derive the switching $\gamma$ for a general state $s$ and check it
    against the picture. is there a $\gamma$ for which *every* state goes left? what does it
    mean that the "optimal" behaviour depends on $\gamma$, if $\gamma$ is something you chose
    rather than part of the world?
    """)
    return


if __name__ == "__main__":
    app.run()
