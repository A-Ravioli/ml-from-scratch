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
    # policy gradients · lab

    every experiment here runs **your** code from `exercise.py` on your `CartPole`. for each
    one: write down a prediction, lock it in, run it, then explain what happened. compare
    your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about
    a minute and a half.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how noisy is one episode's gradient?

    we freeze a policy (REINFORCE with a value baseline, trained for 100 episodes, so it
    balances for a while but is far from perfect), fit its value network to the frozen
    policy's returns on 300 episodes, and collect 600 fresh episodes. from each episode we
    compute a policy-gradient estimate three ways, all with your `reinforce_loss` (summed
    over the episode's steps):

    - **total return**: every step weighted by $G_0$;
    - **reward-to-go**: step $t$ weighted by $G_t$;
    - **reward-to-go − V**: weighted by $G_t - V(s_t)$.

    all three have the same expectation, the true gradient. we estimate it as the mean of
    the reward-to-go − V estimates over the last 300 episodes, and compare each
    single-episode estimate from the first 300 with it.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what's the typical cosine similarity between a single episode's gradient and the true gradient, without and "
        "with the baseline? how much does the baseline reduce the variance: 10%, 2x, 10x?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _policy, _h = ex.train_reinforce(CartPole(), 100, lr=1e-2, batch_episodes=4, baseline="value", seed=0)
    _value = _h["value_net"]
    _params = list(_policy.parameters())
    torch.manual_seed(1)
    _env = CartPole()
    _env.reset(seed=1)

    def _episode():
        _obs, _ = _env.reset()
        _os, _as, _rs = [], [], []
        while True:
            _a = _policy.act(_obs)
            _o2, _r, _te, _tr, _ = _env.step(_a)
            _os.append(_obs)
            _as.append(_a)
            _rs.append(_r)
            _obs = _o2
            if _te or _tr:
                return np.array(_os), np.array(_as), np.array(_rs)

    # first, fit the value network to this frozen policy's returns (300 episodes, 300 Adam steps)
    _fit = [_episode() for _ in range(300)]
    _X = torch.as_tensor(np.concatenate([o for o, _, _ in _fit]), dtype=torch.float32)
    _Y = torch.as_tensor(np.concatenate([ex.discounted_returns(r, 0.99) for _, _, r in _fit]), dtype=torch.float32)
    _vopt = torch.optim.Adam(_value.parameters(), lr=1e-2)
    for _ in range(300):
        _vloss = ((_value(_X).squeeze(-1) - _Y) ** 2).mean()
        _vopt.zero_grad()
        _vloss.backward()
        _vopt.step()

    _estimates = {"total return": [], "reward-to-go": [], "reward-to-go − V": []}
    _lengths = []
    for _ep in range(600):
        _os, _as, _rs = _episode()
        _lengths.append(len(_rs))
        _o = torch.as_tensor(_os, dtype=torch.float32)
        _G = torch.as_tensor(ex.discounted_returns(_rs, 0.99), dtype=torch.float32)
        with torch.no_grad():
            _v = _value(_o).squeeze(-1)
        _weights = {"total return": _G[0].expand(len(_rs)), "reward-to-go": _G, "reward-to-go − V": _G - _v}
        _lp = _policy(_o).log_prob(torch.as_tensor(_as))
        for _k, _w in _weights.items():
            _loss = ex.reinforce_loss(_lp, _w) * len(_rs)  # the episode's sum, not its mean
            _g = torch.autograd.grad(-_loss, _params, retain_graph=True)
            _estimates[_k].append(torch.cat([x.flatten() for x in _g]).numpy())
    _E = {k: np.array(v) for k, v in _estimates.items()}
    _true = _E["reward-to-go − V"][300:].mean(axis=0)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _rows = []
    for _k, _e in _E.items():
        _cos = _e[:300] @ _true / (np.linalg.norm(_e[:300], axis=1) * np.linalg.norm(_true))
        _ax.hist(_cos, bins=30, range=(-1, 1), alpha=0.5, label=_k)
        _var = np.trace(np.cov(_e.T))
        _rows.append(f"| {_k} | {np.mean(_cos):.3f} | {_var:.3g} | {_var / np.sum(_true**2):.0f} |")
    _ax.set(xlabel="cosine(single-episode gradient, true gradient)", ylabel="episodes",
            title=f"frozen policy, mean episode length {np.mean(_lengths):.0f}, {len(_true)} parameters")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| estimator | mean cosine | total variance | variance / ‖true gradient‖² |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the cosines pile up near $+1$ and $-1$: a single episode tells you the
    right *axis* in a 226-dimensional space, but not always the right *sign*. what single
    direction could dominate every episode's gradient on CartPole? without a baseline, every
    weight $G_t$ is positive: so what decides an episode's sign, and why do the common,
    short episodes push the wrong way (giving a negative *mean* cosine, although the
    estimator is unbiased)? how does $G_t - V(s_t)$ fix the sign? the last column is roughly
    the number of episodes you must average before the noise is as large as the signal: use
    it to explain why policy-gradient methods average over many episodes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does the baseline speed up learning?

    REINFORCE on CartPole for 600 episodes (batches of 4 episodes, Adam, lr $3\cdot10^{-3}$,
    $\gamma = 0.99$, one hidden layer of 32), with and without the value baseline, 5 seeds
    each. the shaded bands are ± one standard error over seeds.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "after 600 episodes, how much better (if at all) is REINFORCE with the value baseline? does the gap open "
        "early or late?"
    )
    p2
    return (p2,)


@app.cell
def _():
    n_seeds = mo.ui.slider(2, 8, value=5, label="seeds per variant")
    n_seeds
    return (n_seeds,)


@app.cell
def _(n_seeds, p2):
    ui.gate(p2)
    _curves = {}
    for _b in (None, "value"):
        _curves[_b] = np.array([ex.train_reinforce(CartPole(), 600, lr=3e-3, batch_episodes=4, baseline=_b,
                                                   seed=s)[1]["episode_returns"] for s in range(n_seeds.value)])
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    _k = np.ones(20) / 20
    _rows = []
    for _b, _c in _curves.items():
        _sm = np.array([np.convolve(c, _k, mode="valid") for c in _c])
        _m, _se = _sm.mean(0), _sm.std(0, ddof=1) / np.sqrt(len(_sm))
        _x = np.arange(len(_m)) + 20
        _label = "value baseline" if _b else "no baseline"
        _ax.plot(_x, _m, label=_label)
        _ax.fill_between(_x, _m - _se, _m + _se, alpha=0.25)
        _last = _c[:, -100:].mean(1)
        _area = _c.mean(1)
        _rows.append(f"| {_label} | {_c[:, 100:200].mean():.0f} | {_last.mean():.0f} ± "
                     f"{_last.std(ddof=1) / np.sqrt(len(_last)):.0f} | {_area.mean():.0f} ± "
                     f"{_area.std(ddof=1) / np.sqrt(len(_area)):.0f} |")
    _ax.set(xlabel="episode", ylabel="return (20-episode average)", ylim=(0, 510))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| variant | mean return, episodes 100–200 | last 100 episodes (mean ± s.e.) | "
                           "mean over all 600 (± s.e.) |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** is the improvement what experiment 1's variance reduction would lead you
    to expect? why might the baseline matter little in the first hundred episodes (what is
    the value network worth before it has been trained)? what would you change to make both
    variants learn faster, and what would that cost?
    """)
    return


if __name__ == "__main__":
    app.run()
