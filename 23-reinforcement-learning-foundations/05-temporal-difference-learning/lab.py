import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.rl import Discrete, Env, GridWorld, policy_evaluation  # yours, from topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # temporal-difference learning · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about half a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: cliff walking

    Sutton & Barto's Example 6.6. your `sarsa`, `q_learning` and `expected_sarsa` each learn
    the cliff for 500 episodes with $\varepsilon = 0.1$ and $\alpha = 0.5$, 10 seeded runs
    each. we plot the return of every episode (the ε-greedy behaviour's return, averaged
    over runs and smoothed over 10 episodes), and draw the *greedy* path each method's final
    $Q$ prescribes in run 0.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which method earns more reward per episode while learning, SARSA or Q-learning? which one learns the "
        "optimal (shortest) path?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _methods = {"SARSA": ex.sarsa, "Q-learning": ex.q_learning, "expected SARSA": ex.expected_sarsa}
    _curves, _Qs = {}, {}
    for _name, _fn in _methods.items():
        _runs = [_fn(ex.CliffWalking(), 500, alpha=0.5, epsilon=0.1, rng=s) for s in range(10)]
        _curves[_name] = np.mean([r[1] for r in _runs], axis=0)
        _Qs[_name] = _runs[0][0]
    _g = GridWorld(4, 12, start=(3, 0), goal=(3, 11))

    def _path(Q):
        s, xs = 36, [36]
        for _ in range(60):
            s = int(np.argmax(_g.P[s, int(np.argmax(Q[s]))]))
            xs.append(s)
            if s == 47 or 37 <= s <= 46:
                break
        return xs

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    _k = np.ones(10) / 10
    for _name, _c in _curves.items():
        _axes[0].plot(np.convolve(_c, _k, mode="valid"), label=_name)
    _axes[0].set(xlabel="episode", ylabel="return during learning (10 runs, smoothed)", ylim=(-120, 0))
    _axes[0].legend(fontsize=8)
    _axes[1].add_patch(plt.Rectangle((0.5, -0.5), 10, 1, color="0.3"))
    _axes[1].text(5.5, 0, "cliff", color="w", ha="center", va="center")
    for _i, (_name, _Q) in enumerate(_Qs.items()):
        _xs = _path(_Q)
        _axes[1].plot([s % 12 + 0.08 * (_i - 1) for s in _xs], [3 - s // 12 + 0.08 * (_i - 1) for s in _xs], "o-",
                      ms=3, label=f"{_name} ({len(_xs) - 1} steps)")
    _axes[1].set(xlim=(-0.5, 11.5), ylim=(-0.5, 3.5), xticks=[], yticks=[], title="greedy paths after 500 episodes",
                 aspect="equal")
    _axes[1].legend(fontsize=7, loc="upper center")
    _fig.tight_layout()
    _rows = [f"| {n} | {c[-100:].mean():.1f} |" for n, c in _curves.items()]
    mo.vstack([_fig, mo.md("| method | average return, last 100 episodes (ε-greedy behaviour) |\n|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** Q-learning's greedy path is the shortest one. why does it earn *less*
    while learning? what is each method's $Q$ the value *of*? what would happen to the two
    curves if $\varepsilon$ decayed to 0? why does expected SARSA do at least as well as
    SARSA here?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how much λ?

    the 19-state random walk (Sutton & Barto's Example 7.1): states 1–19 in a row, start in
    the middle, step left or right with probability 1/2; falling off the left end pays $-1$,
    the right end $+1$. the true values are $-0.9, -0.8, \dots, 0.9$. your
    `td_lambda_prediction` learns from 10 episodes; we measure the RMS error over the 19
    states, averaged over 25 runs, for a grid of step sizes $\alpha$ and trace decays
    $\lambda$ ($\lambda = 0$ is TD(0), $\lambda = 1$ behaves like Monte Carlo).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which lambda gives the lowest error after 10 episodes (at its best alpha)? what happens with lambda = 1 "
        "and a large alpha?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    class RandomWalk(Env):
        def __init__(self, n=19):
            self.n = n
            self.observation_space, self.action_space = Discrete(n + 2), Discrete(1)

        def reset(self, *, seed=None):
            super().reset(seed=seed)
            self.s = (self.n + 1) // 2
            return self.s, {}

        def step(self, action):
            self.s += 1 if self.np_random.random() < 0.5 else -1
            if self.s == 0:
                return 0, -1.0, True, False, {}
            if self.s == self.n + 1:
                return self.s, 1.0, True, False, {}
            return self.s, 0.0, False, False, {}

    _true = np.linspace(-1, 1, 21)[1:-1]
    _alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    _lams = [0.0, 0.4, 0.8, 0.9, 0.95, 1.0]
    _err = {}
    with np.errstate(all="ignore"):  # large alpha with large lambda diverges: that's part of the result
        for _lam in _lams:
            _err[_lam] = []
            for _a in _alphas:
                _e = [np.sqrt(np.mean((ex.td_lambda_prediction(RandomWalk(), lambda s: 0, 10, lam=_lam, alpha=_a,
                                                               gamma=1.0, rng=r)[1:-1] - _true) ** 2))
                      for r in range(25)]
                _err[_lam].append(np.mean(_e) if np.all(np.isfinite(_e)) else np.inf)
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    for _lam in _lams:
        _ax.plot(_alphas, np.minimum(_err[_lam], 10), "o-", label=f"λ = {_lam}")
    _ax.set(xlabel="α", ylabel="RMS error after 10 episodes (25 runs)", ylim=(0.1, 0.6))
    _ax.legend(fontsize=8)
    _best = {lam: (min(v), _alphas[int(np.argmin(v))]) for lam, v in _err.items()}
    _rows = [f"| {lam} | {b:.3f} | {a} | {'diverged' if not np.isfinite(_err[lam][-1]) or _err[lam][-1] > 10 else f'{_err[lam][-1]:.2f}'} |"
             for lam, (b, a) in _best.items()]
    mo.vstack([_fig, mo.md("| λ | best RMS error | at α | error at α = 1 |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does an intermediate $\lambda$ beat both TD(0) and Monte Carlo after
    only 10 episodes? why does every curve rise again for large $\alpha$, and why do the
    high-$\lambda$ curves blow up altogether (how large can an accumulating trace get when a
    random walk revisits a state many times)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: a time limit is not a cliff

    the random policy on a $5\times5$ `GridWorld` with $\gamma = 0.95$ and a short time limit,
    `max_steps = 15`. your `td0_prediction` learns $V$ from 5,000 episodes, twice: once on the
    environment as it is, and once through a wrapper that reports every time-out as
    `terminated=True` (a single `done` flag, the way older code did it). we compare both with
    the exact $V^\pi$ of the same gridworld *without* a time limit, from `policy_evaluation`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "which of the two TD estimates matches the exact values (with no time limit)? how far off is the other, "
        "and in which direction?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    class TimeoutIsDeath(Env):
        """reports truncation as termination: the bug"""

        def __init__(self, env):
            self.env = env
            self.observation_space, self.action_space = env.observation_space, env.action_space

        def reset(self, *, seed=None):
            return self.env.reset(seed=seed)

        def step(self, action):
            obs, r, terminated, truncated, info = self.env.step(action)
            return obs, r, terminated or truncated, False, info

    _exact = policy_evaluation(GridWorld(5, 5).P, GridWorld(5, 5).R, np.full((25, 4), 0.25), gamma=0.95)
    _est = {}
    for _name, _env in [("bootstrap on truncation", GridWorld(5, 5, max_steps=15)),
                        ("time-out = terminal", TimeoutIsDeath(GridWorld(5, 5, max_steps=15)))]:
        _prng = np.random.default_rng(0)
        _est[_name] = ex.td0_prediction(_env, lambda s: int(_prng.integers(4)), 5000, alpha=0.05, gamma=0.95, rng=0)
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for _ax, (_name, _V) in zip(_axes, [("exact, no time limit", _exact)] + list(_est.items())):
        _im = _ax.imshow(_V.reshape(5, 5), vmin=-19, vmax=0, cmap="viridis")
        _ax.set(title=_name, xticks=[], yticks=[])
        for _s in range(25):
            _ax.text(_s % 5, _s // 5, f"{_V[_s]:.1f}", ha="center", va="center", fontsize=7,
                     color="w" if _V[_s] < -9 else "k")
    _fig.colorbar(_im, ax=_axes, fraction=0.02)
    _rows = [f"| {n} | {V[0]:.2f} | {np.mean(np.abs(V - _exact)):.2f} |" for n, V in _est.items()]
    mo.vstack([_fig, mo.md(f"exact V(start) = **{_exact[0]:.2f}**\n\n| TD(0) on | V(start) | mean abs error |\n"
                           "|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does bootstrapping on truncation recover the values of the
    time-unlimited problem, even though no episode lasts more than 15 steps? what does the
    buggy version learn instead (what is it the exact value *of*)? why is the bias biggest far
    from the goal? ch 24's DQN and GAE carry exactly this `terminated`/`truncated` distinction.
    """)
    return


if __name__ == "__main__":
    app.run()
