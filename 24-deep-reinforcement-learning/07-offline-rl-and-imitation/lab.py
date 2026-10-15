import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui
    from mlfs.deeprl import CartPole, CategoricalPolicy, ReplayBuffer  # yours, from topics 01-02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # small networks: one thread is fastest


@app.function
def expert(obs):
    """a hand-tuned linear controller that balances CartPole for the full 500 steps."""
    return int(0.1 * obs[0] + 0.5 * obs[1] + 5.0 * obs[2] + 1.0 * obs[3] > 0)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # offline RL & imitation · lab

    every experiment here runs **your** `behavior_cloning`, `dagger` and `train_offline_q`
    on your `CartPole`. the expert is a hand-written linear controller,
    $a = [0.1x + 0.5\dot x + 5\theta + \dot\theta > 0]$, which balances for the full 500 steps
    from every start. for each experiment: write down a prediction, lock it in, run it, then
    explain what happened. compare with `solutions/lab_notes.md` afterwards. the lab takes
    about a minute and a half.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: errors compound

    behavior cloning from 1, 2 or 3 of the expert's episodes (500, 1,000 or 1,500 labelled
    states; a one-hidden-layer policy, 30 epochs), and DAgger starting from one expert episode
    and adding one learner episode per iteration for 3 more iterations (so at most 2,000
    states). each policy (5 seeds) runs 20 test episodes, sampling its actions. we plot the
    fraction of episodes still balancing at each step.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "a clone trained on ONE expert episode agrees with the expert on almost every training state. how long does "
        "it balance? does doubling the demonstrations double the time? where does DAgger end up?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _lengths = {}
    for _n in (1, 2, 3):
        _all = []
        for _seed in range(5):
            _, _obs = ex.run_policy(CartPole(), expert, _n, seed=_seed)
            torch.manual_seed(_seed)
            _pi = CategoricalPolicy(4, 2, hidden_sizes=(32,))
            ex.behavior_cloning(_pi, np.array(_obs), [expert(o) for o in _obs], epochs=30, seed=_seed)
            _all += ex.run_policy(CartPole(), _pi.act, 20, seed=100 + _seed)[0]
        _lengths[f"BC, {_n} episode{'s' if _n > 1 else ''}"] = np.array(_all)
    _all = []
    for _seed in range(5):
        torch.manual_seed(_seed)
        _pi = CategoricalPolicy(4, 2, hidden_sizes=(32,))
        _pi, _ = ex.dagger(CartPole(), expert, _pi, n_iterations=4, episodes_per_iteration=1, epochs=30, seed=_seed)
        _all += ex.run_policy(CartPole(), _pi.act, 20, seed=100 + _seed)[0]
    _lengths["DAgger (1 expert + 3 learner episodes)"] = np.array(_all)
    _fig, _ax = plt.subplots(figsize=(7.5, 4))
    _t = np.arange(0, 500)
    _rows = []
    for (_name, _L), _ls in zip(_lengths.items(), ["-", "-", "--", ":"]):
        _ax.plot(_t, [(_L > s).mean() for s in _t], ls=_ls, lw=2, label=_name)
        _rows.append(f"| {_name} | {_L.mean():.0f} | {np.median(_L):.0f} | {(_L >= 500).mean():.2f} |")
    _ax.set(xlabel="step (episodes are cut off at 500)", ylabel="fraction of episodes still balancing", ylim=(0, 1.05))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| policy | mean return | median | fraction reaching 500 |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what states does a single expert episode contain, and which states does
    the clone reach after its first few mistakes? why does the survival curve keep falling
    over time instead of failing early or not at all? why does a little more data help so
    much here, and why does DAgger reach the expert's level with fewer *expert* episodes
    than cloning needs?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: offline RL from a noisy controller

    a dataset of 20,000 transitions collected by the expert with 30% random actions
    (ε-greedy around the controller, which is robust enough to balance most of the time anyway). three
    learners that never touch the environment: your `train_offline_q` (double DQN, 10,000
    updates) without and with the CQL penalty (α = 1), and behavior cloning of the dataset's
    actions. each is then evaluated greedily for 10 episodes, two seeds per method. we also
    record the largest Q-value the network predicts on the dataset's states; the true values
    can't exceed $1/(1 - \gamma) = 100$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "rank offline DQN, CQL and behavior cloning of the noisy data. how large do offline DQN's Q-values get?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _buf = ReplayBuffer(20_000, (4,), seed=0)
    _env = CartPole()
    _o, _ = _env.reset(seed=0)
    _behaviour = []
    _ret = 0.0
    for _ in range(20_000):
        _a = expert(_o) if _rng.random() > 0.3 else int(_rng.integers(2))
        _o2, _r, _te, _tr, _ = _env.step(_a)
        _buf.add(_o, _a, _r, _o2, _te)
        _ret += _r
        if _te or _tr:
            _behaviour.append(_ret)
            _ret = 0.0
            _o, _ = _env.reset()
        else:
            _o = _o2
    _rows = [f"| behaviour policy (the data) | {np.mean(_behaviour):.0f} | — |"]
    for _name, _alpha in [("offline double DQN", 0.0), ("CQL (α = 1)", 1.0)]:
        _scores, _qmax = [], []
        for _seed in range(2):
            _q, _ = ex.train_offline_q(_buf, 4, 2, 10_000, lr=1e-3, cql_alpha=_alpha, seed=_seed)
            _greedy = lambda o, q=_q: int(q(torch.as_tensor(o, dtype=torch.float32)[None]).argmax())
            _scores += ex.run_policy(CartPole(), _greedy, 10, seed=7 + _seed)[0]
            with torch.no_grad():
                _qmax.append(_q(torch.as_tensor(_buf.obs)).max().item())
        _rows.append(f"| {_name} | {np.mean(_scores):.0f} | {', '.join(f'{x:.3g}' for x in _qmax)} |")
    _scores = []
    for _seed in range(2):
        torch.manual_seed(_seed)
        _pi = CategoricalPolicy(4, 2, hidden_sizes=(32,))
        ex.behavior_cloning(_pi, _buf.obs, _buf.actions, epochs=10, seed=_seed)
        _greedy = lambda o, pi=_pi: int(pi(torch.as_tensor(o, dtype=torch.float32)[None]).probs.argmax())
        _scores += ex.run_policy(CartPole(), _greedy, 10, seed=7 + _seed)[0]
    _rows.append(f"| behavior cloning of the data (greedy) | {np.mean(_scores):.0f} | — |")
    mo.md("| learner | mean return (greedy; 2 seeds × 10 episodes) | largest predicted Q on the data, per seed |\n"
          "|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the dataset has both actions in plenty of states, yet offline DQN's values
    explode: which states and actions does the target's max query that the data covers
    poorly, and why does nothing correct the error offline? why does CQL keep the values
    near 100 and recover a good policy? greedy behavior cloning of noisy data does well
    here: why does taking the argmax remove the noise, and when would cloning the data be
    a bad idea?
    """)
    return


if __name__ == "__main__":
    app.run()
