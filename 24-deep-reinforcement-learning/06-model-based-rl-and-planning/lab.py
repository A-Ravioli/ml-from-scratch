import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui
    from mlfs.rl import GridWorld  # yours, from ch 23 · topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # small networks: one thread is fastest


@app.function
def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k / n."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


@app.function
def play(game, agent_a, agent_b, a_first):
    """one greedy game between two searchers; +1 if a wins, -1 if b wins, 0 for a draw."""
    s = game.initial_state()
    a_player = 1 if a_first else -1
    while not game.is_terminal(s):
        agent = agent_a if game.current_player(s) == a_player else agent_b
        s = game.next_state(s, int(np.argmax(agent.search(s))))
    return int(game.terminal_value(s) * game.current_player(s) * a_player)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # model-based RL & planning · lab

    every experiment here runs **your** `dyna_q`, `ConnectFour`, `MCTS` and AlphaZero pieces.
    for each one: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes
    about two and a half minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the Dyna maze

    Sutton & Barto's maze (Example 8.1): a 6 × 9 grid with walls, start on the left, goal in
    the top-right corner, reward $+1$ for reaching it and 0 otherwise, $\gamma = 0.95$. your
    `dyna_q` with α = 0.1, ε = 0.1 and 0, 5 or 50 planning updates per real step, for 50
    episodes, averaged over 10 seeds. the shortest path is 14 steps.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how many episodes does each agent (0, 5, 50 planning steps) need before its episodes are close to the "
        "shortest path? does planning change the length of the FIRST episode?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    class Maze(GridWorld):
        def __init__(self):
            super().__init__(6, 9, start=(2, 0), goal=(0, 8), walls=[(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7),
                                                                     (2, 7)], step_reward=0.0, max_steps=10_000)

        def reset(self, *, seed=None):
            self.lengths = getattr(self, "lengths", [])
            return super().reset(seed=seed)

        def step(self, action):
            s2, r, term, trunc, info = super().step(action)
            if term or trunc:
                self.lengths.append(self.t)
            return s2, float(term), term, trunc, info

    _curves = {}
    for _n in (0, 5, 50):
        _runs = []
        for _seed in range(10):
            _env = Maze()
            ex.dyna_q(_env, 50, n_planning=_n, alpha=0.1, gamma=0.95, epsilon=0.1, rng=_seed)
            _runs.append(_env.lengths)
        _curves[_n] = np.mean(_runs, axis=0)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _n, _c in _curves.items():
        _ax.plot(np.arange(2, 51), _c[1:], label=f"{_n} planning steps")
    _ax.axhline(14, color="k", ls=":", lw=1)
    _ax.set(xlabel="episode", ylabel="steps per episode (10 seeds)", yscale="log", title="episodes 2-50")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {n} | {c[0]:.0f} | {c[1]:.0f} | {c[4]:.0f} | {np.argmax(c < 20) + 1 if np.any(c < 20) else '> 50'} |"
             for n, c in _curves.items()]
    mo.vstack([_fig, mo.md("| planning steps | episode 1 | episode 2 | episode 5 | first episode under 20 steps |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** until the goal is first reached every Q-value is 0, so all three agents
    do the same random walk in the first episode: why do their first-episode lengths still
    differ so much (how variable is the length of a random walk)? why does Q-learning
    (0 planning steps) need many episodes to shorten its path, and what do the planning
    updates do after the goal has been found once? what does the agent pay for 50 planning
    updates per step, if not real experience?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: planning budget versus strength

    your `MCTS` with random rollouts plays full-size Connect Four against an opponent that
    uses 100 simulations per move ($c_{\text{puct}} = 1.5$ for both, the most visited move
    is played). we vary our player's budget, 10 to 400 simulations per move, and play 16
    games per budget, alternating who moves first. the intervals are 95% Wilson intervals
    for the win rate.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "against a 100-simulation opponent, what win rate do you expect with 10, 30, 100 and 400 simulations? "
        "is doubling the budget worth a constant amount of strength?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _g = ex.ConnectFour()
    _budgets = [10, 30, 100, 400]
    _rates = []
    _rows = []
    for _b in _budgets:
        _res = np.array([play(_g, ex.MCTS(_g, n_simulations=_b, c_puct=1.5, rng=1000 + k),
                              ex.MCTS(_g, n_simulations=100, c_puct=1.5, rng=2000 + k), k % 2 == 0)
                         for k in range(16)])
        _w, _d = int(np.sum(_res > 0)), int(np.sum(_res == 0))
        _lo, _hi = wilson(_w, len(_res))
        _rates.append((_w / len(_res), _lo, _hi))
        _rows.append(f"| {_b} | {_w} / {_d} / {len(_res) - _w - _d} | {_w / len(_res):.2f} [{_lo:.2f}, {_hi:.2f}] |")
    _r = np.array(_rates)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.errorbar(_budgets, _r[:, 0], yerr=[_r[:, 0] - _r[:, 1], _r[:, 2] - _r[:, 0]], fmt="o-", capsize=4)
    _ax.axvline(100, color="k", ls=":", lw=1)
    _ax.set(xscale="log", xlabel="our simulations per move", ylabel="win rate vs 100 simulations", ylim=(0, 1))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| our simulations | wins / draws / losses | win rate [95% CI] |\n|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does strength grow with the budget: linearly, or roughly per
    doubling? why should the equal-budget match come out near 50%, and does yours? with 16
    games per point, which differences can you actually claim (look at the interval widths),
    and how many games would you need to tell 55% from 50%?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: AlphaZero-lite

    on a smaller 5 × 5 board (still four in a row), `train_alphazero` alternates 10
    self-play games per iteration (your `MCTS` with your `PolicyValueNet` as the evaluator,
    50 simulations per move) with a few epochs of training on the last 200 games, for 25
    iterations (about 30 seconds). we do this for **three training seeds**: a single
    AlphaZero run is as noisy as a single RL learning curve. each network, untrained and
    trained, then sits inside a 50-simulation search and plays 20 games against a
    50-simulation search with random rollouts (draws count half).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the untrained network's search versus random rollouts: what score? after 30 seconds of self-play training? "
        "how much do the three training seeds disagree?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _g = ex.ConnectFour(5, 5)

    def _score(net):
        res = np.array([play(_g, ex.MCTS(_g, n_simulations=50, c_puct=1.5, evaluate=net.evaluate, rng=k),
                             ex.MCTS(_g, n_simulations=50, c_puct=1.5, rng=100 + k), k % 2 == 0) for k in range(20)])
        return (np.sum(res > 0) + 0.5 * np.sum(res == 0)) / len(res)

    _rows, _before, _after = [], [], []
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    for _seed in range(3):
        torch.manual_seed(_seed)
        _b = _score(ex.PolicyValueNet(_g))
        _net, _hist = ex.train_alphazero(_g, 25, games_per_iteration=10, n_simulations=50, lr=2e-3, epochs=6,
                                        seed=_seed)
        _a = _score(_net)
        _before.append(_b)
        _after.append(_a)
        _ax.plot(np.arange(1, len(_hist["loss"]) + 1), _hist["loss"], "o-", ms=3, label=f"seed {_seed}")
        _rows.append(f"| {_seed} | {_b:.2f} | {_a:.2f} |")
    _rows.append(f"| mean | {np.mean(_before):.2f} | {np.mean(_after):.2f} |")
    _ax.set(xlabel="iteration", ylabel="training loss (last minibatch)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| training seed | untrained network: score vs rollouts | after 25 iterations |\n"
                           "|---|---|---|\n" + "\n".join(_rows)
                           + "\n\n20 games per entry: a score's standard error is about 0.1.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does a search guided by an *untrained* network fare against random
    rollouts, and why does it vary so much with the network's random initialization? did
    self-play training help reliably, i.e. by more than the spread between seeds? the loss
    falls, so the network is learning *something*: why doesn't that translate into
    strength here (what are the training targets, and how good can they be when they come
    from a weak search playing itself)? what would you change first to make AlphaZero-lite
    stronger than rollouts, and how would you show that it worked?
    """)
    return


if __name__ == "__main__":
    app.run()
