import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui
    from mlfs.deeprl import CartPole  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # small networks: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # actor-critic · lab

    every experiment here runs **your** `compute_gae` and `train_a2c`. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about a
    minute and a half.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what λ buys, and what it costs

    GAE's returns $\hat V_t = \hat A_t + V(s_t)$ are the targets the critic regresses onto,
    and their errors are the errors of the advantages. we can measure those errors exactly on
    a problem whose true values we know: the 19-state random walk (start in the middle,
    step left or right with probability ½, $-1$ for falling off the left end and $+1$ for the
    right, $\gamma = 1$; the true values are $-0.9, -0.8, \dots, 0.9$). we run 64 walks in
    parallel for 2,000 steps (with automatic resets), feed the rollout to your
    `compute_gae` with three critics, and measure, for every state, the squared bias and the
    variance of the targets around the true value:

    - a **perfect** critic, $V = V^\pi$;
    - a **shrunk** critic, $V = 0.5\,V^\pi$ (right shape, wrong scale, as early in training);
    - a **noisy** critic, $V = V^\pi + \varepsilon(s)$ with a fixed random error of size 0.3 per state.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with a perfect critic, which lambda gives the lowest-error targets? with the shrunk critic? is there a "
        "lambda that is good for all three?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _n, _N, _T = 19, 64, 2000
    _true = np.linspace(-0.9, 0.9, _n)
    _s = np.full(_N, _n // 2)
    _S, _R, _D = np.zeros((_T, _N), int), np.zeros((_T, _N)), np.zeros((_T, _N))
    for _t in range(_T):
        _S[_t] = _s
        _s = _s + _rng.choice([-1, 1], size=_N)
        _R[_t] = np.where(_s < 0, -1.0, np.where(_s >= _n, 1.0, 0.0))
        _D[_t] = (_s < 0) | (_s >= _n)
        _s = np.where(_D[_t] > 0, _n // 2, _s)
    _critics = {"perfect": _true, "shrunk (0.5 V)": 0.5 * _true,
                "noisy (± 0.3)": _true + 0.3 * np.random.default_rng(1).standard_normal(_n)}
    _lams = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    _rows = []
    for _ax, (_name, _V) in zip(_axes, _critics.items()):
        _b2, _var = [], []
        for _lam in _lams:
            _vals = torch.as_tensor(_V[_S], dtype=torch.float32)
            _last = torch.as_tensor(_V[_s], dtype=torch.float32)
            _, _ret = ex.compute_gae(torch.as_tensor(_R, dtype=torch.float32), _vals, torch.as_tensor(_D), _last,
                                     gamma=1.0, lam=_lam)
            _ret = _ret.numpy()[: _T - 200]  # drop the tail, whose targets lean on last_value
            _st = _S[: _T - 200]
            _bias2 = np.mean([(_ret[_st == k].mean() - _true[k]) ** 2 for k in range(_n)])
            _v = np.mean([_ret[_st == k].var() for k in range(_n)])
            _b2.append(_bias2)
            _var.append(_v)
        _b2, _var = np.array(_b2), np.array(_var)
        _ax.plot(_lams, _b2, "o-", label="bias²")
        _ax.plot(_lams, _var, "s-", label="variance")
        _ax.plot(_lams, _b2 + _var, "k^--", label="MSE")
        _ax.set(title=f"critic: {_name}", xlabel="λ", yscale="log", ylim=(1e-4, 1.5))
        _ax.legend(fontsize=8)
        _best = int(np.argmin(_b2 + _var))
        _rows.append(f"| {_name} | {_b2[0]:.3f} / {_var[0]:.3f} | {_b2[-1]:.3f} / {_var[-1]:.3f} | {_lams[_best]} "
                     f"({(_b2 + _var)[_best]:.3f}) |")
    _axes[0].set_ylabel("error of the targets\n(mean over states)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| critic | λ = 0: bias² / variance | λ = 1: bias² / variance | best λ (MSE) |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does λ = 1 have the same error whatever the critic, and λ = 0 inherit
    the critic's error? why does the *variance* grow with λ even with a perfect critic, and
    why does it dominate everything at λ = 1 on this task? what does a one-step target do to
    a critic's per-state errors (what does it average over)? what does this say about
    choosing λ when the critic is still learning?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: sweeping λ in A2C

    your `train_a2c` on CartPole for 100,000 steps (8 environments, 32-step rollouts, Adam
    lr $10^{-3}$, $\gamma = 0.99$), for five values of λ, four seeds each. we report the
    return of the last 20 episodes and the mean return over training (a proxy for learning
    speed).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which lambda learns CartPole best in 100,000 steps: 0 (one-step TD), 1 (Monte Carlo), or something between? "
        "how big are the differences compared with the seed-to-seed spread?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _lams = [0.0, 0.5, 0.9, 0.97, 1.0]
    _res = {}
    for _lam in _lams:
        _res[_lam] = [ex.train_a2c(CartPole, 100_000, n_envs=8, n_steps=32, lam=_lam, lr=1e-3, seed=s)[1]
                      for s in range(4)]
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _rows = []
    for _lam, _hs in _res.items():
        _grid = np.linspace(2_000, 100_000, 50)
        _curves = []
        for _h in _hs:
            _ends = np.asarray(_h["episode_ends"])
            _r = np.convolve(_h["episode_returns"], np.ones(10) / 10, mode="full")[: len(_ends)]
            _idx = np.clip(np.searchsorted(_ends, _grid, side="right") - 1, 0, None)
            _curves.append(_r[_idx])
        _axes[0].plot(_grid, np.mean(_curves, axis=0), label=f"λ = {_lam}")
        _final = np.array([np.mean(h["episode_returns"][-20:]) for h in _hs])
        _avg = np.array([np.mean(h["episode_returns"]) for h in _hs])
        _axes[1].scatter([_lam] * len(_final), _final, color="k", s=12)
        _rows.append(f"| {_lam} | {_final.mean():.0f} ± {_final.std(ddof=1) / 2:.0f} | "
                     f"{', '.join(f'{x:.0f}' for x in _final)} | {_avg.mean():.0f} ± {_avg.std(ddof=1) / 2:.0f} |")
    _axes[0].set(xlabel="environment steps", ylabel="return (10-episode average, mean of 4 seeds)", ylim=(0, 510))
    _axes[0].legend(fontsize=8)
    _axes[1].set(xlabel="λ", ylabel="final return (last 20 episodes), per seed", ylim=(0, 510))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| λ | final return (mean ± s.e.) | per seed | mean over training (± s.e.) |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is λ = 0 so much worse than everything else here, in light of
    experiment 1 (what is the critic like for most of training)? why isn't λ = 1 clearly the
    best, given that its targets are unbiased? with four seeds, which differences can you
    actually claim? what would you need to rank 0.9 against 0.97 with confidence?
    """)
    return


if __name__ == "__main__":
    app.run()
