import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # small networks: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # deep Q-networks · lab

    every experiment here runs **your** `train_dqn` on **your** `CartPole`. for each one:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about two
    minutes on a laptop CPU (12 training runs of 15,000 steps, plus 3 for double DQN).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: take the plumbing away

    four agents, three seeds each, 15,000 environment steps on CartPole with the same
    hyperparameters (Adam, lr $5\cdot10^{-4}$, batch 64, ε from 1 to 0.05 over 5,000 steps,
    $\gamma = 0.99$, Huber loss):

    - **DQN**: replay buffer of 20,000 transitions, hard target update every 100 gradient steps;
    - **no target network**: the target is copied every gradient step (`target_update=1`),
      so the bootstrap uses the network being trained;
    - **no replay**: a buffer of only 64 transitions, so each minibatch is (a resampling
      of) the last 64 steps: nearly online, highly correlated updates;
    - **neither**.

    after training we run the greedy policy for 5 episodes per seed, and record the
    network's estimate $\max_aQ(s_0, a)$ at the start states.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "rank the four agents by their final greedy return. which ingredient matters more on CartPole, replay or "
        "the target network? what happens to the Q-values without a target network?"
    )
    p1
    return (p1,)


@app.cell
def _():
    n_steps = mo.ui.slider(5_000, 30_000, value=15_000, step=5_000, label="training steps per run")
    n_steps
    return (n_steps,)


@app.function
def greedy_eval(q_net, n_episodes=5, seed=1000):
    """greedy returns over n_episodes, plus max_a Q(s_0, a) at each start state."""
    returns, q0 = [], []
    for k in range(n_episodes):
        env = ex.CartPole()
        obs, _ = env.reset(seed=seed + k)
        total, first = 0.0, True
        while True:
            with torch.no_grad():
                q = q_net(torch.as_tensor(obs, dtype=torch.float32)[None])[0]
            if first:
                q0.append(float(q.max()))
                first = False
            obs, r, term, trunc, _ = env.step(int(q.argmax()))
            total += r
            if term or trunc:
                break
        returns.append(total)
    return float(np.mean(returns)), float(np.mean(q0))


@app.function
def smooth_curve(history, n_steps, grid):
    """episode returns as a step function of env steps, evaluated on grid (a moving average of 10 episodes)."""
    ends = np.asarray(history["episode_ends"])
    rets = np.convolve(history["episode_returns"], np.ones(10) / 10, mode="full")[: len(ends)]
    idx = np.searchsorted(ends, grid, side="right") - 1
    return np.where(idx >= 0, rets[np.clip(idx, 0, None)], np.nan)


@app.cell
def _(n_steps, p1):
    ui.gate(p1)
    base = dict(lr=5e-4, batch_size=64, learning_starts=1000, eps_decay_steps=5000, gamma=0.99, hidden_sizes=(64, 64))
    variants = {
        "DQN": dict(buffer_size=20_000, target_update=100),
        "no target network": dict(buffer_size=20_000, target_update=1),
        "no replay": dict(buffer_size=64, target_update=100),
        "neither": dict(buffer_size=64, target_update=1),
    }
    runs = {}
    for _name, _kw in variants.items():
        runs[_name] = [ex.train_dqn(ex.CartPole(), n_steps.value, seed=s, **base, **_kw) for s in range(3)]
    _grid = np.linspace(500, n_steps.value, 60)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    _rows = []
    for (_name, _rs), _ls in zip(runs.items(), ["-", "--", "-", ":"]):
        _curves = np.array([smooth_curve(h, n_steps.value, _grid) for _, h in _rs])
        _ax.plot(_grid, np.nanmean(_curves, axis=0), ls=_ls, lw=2 if _ls == ":" else 1.5, label=_name)
        _ev = [greedy_eval(q) for q, _ in _rs]
        _g = np.array([e[0] for e in _ev])
        _q = np.array([e[1] for e in _ev])
        _rows.append(f"| {_name} | {_g.mean():.0f} ± {_g.std(ddof=1) / np.sqrt(len(_g)):.0f} | "
                     f"{', '.join(f'{x:.0f}' for x in _g)} | {', '.join(f'{x:.3g}' for x in _q)} |")
    _ax.set(xlabel="environment steps", ylabel="training return (ε-greedy, 10-episode average)",
            title="CartPole, mean of 3 seeds", ylim=(0, 510))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| agent | greedy return (mean ± s.e., 3 seeds) | per seed | max Q(s0, ·) per seed |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows)
                           + "\n\nfor reference: no return is worth more than $1/(1-\\gamma) = 100$.")])
    return base, runs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does removing the target network do so much more damage than removing
    replay here? what feedback loop produces Q-values that large (compare with the bound of
    100 from on paper 1), and why doesn't a small learning rate save it? replay barely matters
    on CartPole at this budget: what is it about this task (and this short run) that makes
    correlated updates tolerable, and when would you expect that to change?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how good are the value estimates?

    a Q-network's estimates are only useful if they track the returns the policy really
    gets. we take the three DQN runs from experiment 1, train three **double DQN** runs with
    the same settings, and roll out each greedy policy for 5 episodes. at every visited
    state we compare the network's $\max_aQ(s, a)$ with the discounted return actually
    obtained from that state on ($\gamma = 0.99$), over the first 300 steps of each
    episode (so the cut-off at 500 barely affects the true return).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "do DQN's value estimates sit above, below or on the true discounted returns? does double DQN change that?"
    )
    p2
    return (p2,)


@app.cell
def _(base, n_steps, p2, runs):
    ui.gate(p2)

    def _estimates(q_net, n_episodes=5, seed=2000, gamma=0.99):
        est, true = [], []
        for k in range(n_episodes):
            env = ex.CartPole()
            obs, _ = env.reset(seed=seed + k)
            qs, rs = [], []
            while True:
                with torch.no_grad():
                    q = q_net(torch.as_tensor(obs, dtype=torch.float32)[None])[0]
                qs.append(float(q.max()))
                obs, r, term, trunc, _ = env.step(int(q.argmax()))
                rs.append(r)
                if term or trunc:
                    break
            G, acc = np.zeros(len(rs)), 0.0
            for t in reversed(range(len(rs))):
                acc = rs[t] + gamma * acc
                G[t] = acc
            if trunc:  # bootstrap the cut-off tail with the true value of balancing forever
                G += gamma ** np.arange(len(rs), 0, -1) * 100.0
            est += qs[:300]
            true += G[:300].tolist()
        return np.array(est), np.array(true)

    _double = [ex.train_dqn(ex.CartPole(), n_steps.value, seed=s, double=True, buffer_size=20_000, target_update=100,
                            **base) for s in range(3)]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _rows = []
    for _ax, (_name, _rs) in zip(_axes, [("DQN", runs["DQN"]), ("double DQN", _double)]):
        for _s, (_q, _) in enumerate(_rs):
            _e, _t = _estimates(_q)
            _ax.scatter(_t, _e, s=3, alpha=0.4, label=f"seed {_s}")
            _rows.append(f"| {_name} | {_s} | {_t.mean():.1f} | {_e.mean():.1f} | {np.mean(_e - _t):+.1f} | "
                         f"{_e.max():.1f} |")
        _ax.plot([0, 110], [0, 110], "k--", lw=1)
        _ax.axhline(100, color="r", lw=0.8, ls=":")
        _ax.set(title=_name, xlabel="true discounted return from s", xlim=(0, 105))
        _ax.legend(fontsize=7, markerscale=3)
    _axes[0].set_ylabel("estimated max_a Q(s, a)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| agent | seed | mean true return | mean estimate | mean (estimate − true) | max estimate |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where do the estimates sit relative to the diagonal, and does that depend
    on how good the policy is? any estimate above the red line at 100 is impossible: where
    can such values come from? did double DQN remove the bias you found, and if not, why
    might maximization bias be small on a two-action task with deterministic dynamics? what
    else limits how high the values can be after 15,000 steps (on paper 4)?
    """)
    return


if __name__ == "__main__":
    app.run()
