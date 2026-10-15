import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from functools import lru_cache

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.rl import Discrete, Env  # yours, from topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # monte carlo methods · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the whole lab takes about half a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the exact answer, for reference

    blackjack with an infinite deck is small enough to solve exactly: the next card is 1–9
    with probability $1/13$ each and 10 with probability $4/13$, so the dealer's final total
    given the face-up card, and the value of hitting or sticking in every state, follow from
    a short recursion. the cell below does that (it's DP from topic 03, written for the card
    game instead of with a `P` array). `exact_value(policy)` returns the exact expected
    winnings per game of any policy, given as an array of 360 actions.
    """)
    return


@app.cell
def _():
    CARD_P = np.array([0.0] + [1 / 13] * 9 + [4 / 13])  # CARD_P[c] for card value c = 1..10

    @lru_cache(None)
    def _dealer(hard, ace):
        # distribution of the dealer's final total over (17, 18, 19, 20, 21, bust)
        total = hard + 10 if ace and hard + 10 <= 21 else hard
        out = np.zeros(6)
        if total > 21:
            out[5] = 1.0
        elif total >= 17:
            out[total - 17] = 1.0
        else:
            for c in range(1, 11):
                out = out + CARD_P[c] * _dealer(hard + c, ace or c == 1)
        return out

    def stick_value(total, shown):
        out = sum(CARD_P[c] * _dealer(shown + c, shown == 1 or c == 1) for c in range(1, 11))
        win = out[5] + sum(out[i] for i in range(5) if 17 + i < total)
        lose = sum(out[i] for i in range(5) if 17 + i > total)
        return win - lose

    def exact_values(policy=None):
        """value function over (hard total, holds an ace, dealer card); policy None = optimal."""
        memo = {}

        def val(hard, ace, shown):
            key = (hard, ace, shown)
            if key not in memo:
                usable = ace and hard + 10 <= 21
                total = hard + 10 if usable else hard
                stick = stick_value(total, shown)
                hit = sum(-CARD_P[c] if hard + c > 21 else CARD_P[c] * val(hard + c, ace or c == 1, shown)
                          for c in range(1, 11))
                if policy is None:
                    memo[key] = max(stick, hit)
                else:
                    memo[key] = hit if policy[ex.Blackjack.encode(total, shown, usable)] == 1 else stick
            return memo[key]

        return val

    def exact_value(policy=None):
        val = exact_values(policy)
        return sum(CARD_P[a] * CARD_P[b] * CARD_P[d] * val(a + b, a == 1 or b == 1, d)
                   for a in range(1, 11) for b in range(1, 11) for d in range(1, 11))

    _opt = exact_values()
    optimal_policy = np.zeros(360, dtype=int)
    for _o in range(360):
        _ps, _d, _u = ex.Blackjack.decode(_o)
        _hard = _ps - 10 if _u else _ps
        _hit = sum(-CARD_P[c] if _hard + c > 21 else CARD_P[c] * _opt(_hard + c, _u or c == 1, _d) for c in range(1, 11))
        optimal_policy[_o] = int(_hit > stick_value(_ps, _d))
    def action_gap(obs):
        """exact |Q*(s, hit) - Q*(s, stick)| for a decision state."""
        ps, d, u = ex.Blackjack.decode(obs)
        hard = ps - 10 if u else ps
        hit = sum(-CARD_P[c] if hard + c > 21 else CARD_P[c] * _opt(hard + c, u or c == 1, d) for c in range(1, 11))
        return abs(hit - stick_value(ps, d))

    return action_gap, exact_value, optimal_policy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: Monte Carlo control on blackjack

    your `mc_control` plays 500,000 games with $\varepsilon = 0.1$. we compare the greedy
    policy for its $Q$ with the exact optimal policy, on the 200 states where a decision
    matters (sums 12–21), and compute the exact expected winnings per game of: the optimal
    policy, your learned greedy policy, and the "stick on 20 or 21" policy. as a check of the
    DP against your environment, we also average 100,000 of your `Blackjack` games played
    with the optimal policy.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "in how many of the 200 decision states will the learned greedy policy differ from the optimal one? "
        "how much worse will it be, in expected winnings per game?"
    )
    p1
    return (p1,)


@app.cell
def _(exact_value, optimal_policy, p1):
    ui.gate(p1)
    _Q, _rets = ex.mc_control(ex.Blackjack(), 500_000, epsilon=0.1, rng=0)
    learned = np.argmax(_Q, axis=1)
    _stick20 = np.array([0 if ex.Blackjack.decode(o)[0] >= 20 else 1 for o in range(360)])
    _dec = [o for o in range(360) if ex.Blackjack.decode(o)[0] >= 12]
    _diff = [o for o in _dec if learned[o] != optimal_policy[o]]
    _env = ex.Blackjack()
    _env.reset(seed=123)
    _wins = []
    for _ in range(100_000):
        _obs, _ = _env.reset()
        _done = False
        while not _done:
            _obs, _r, _term, _trunc, _ = _env.step(int(optimal_policy[_obs]))
            _done = _term or _trunc
        _wins.append(_r)
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for _row, _u in enumerate([True, False]):
        for _col, (_name, _pol) in enumerate([("optimal (exact DP)", optimal_policy), ("learned by your mc_control", learned)]):
            _grid = np.array([[_pol[ex.Blackjack.encode(s, d, _u)] for d in range(1, 11)] for s in range(12, 22)])
            _ax = _axes[_row, _col]
            _ax.imshow(_grid, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1, extent=[0.5, 10.5, 11.5, 21.5],
                       aspect="auto")
            _ax.set(title=f"{_name}, {'usable ace' if _u else 'no usable ace'}", xlabel="dealer shows",
                    ylabel="player sum", xticks=range(1, 11), yticks=range(12, 22))
    _fig.suptitle("red = hit, green = stick")
    _fig.tight_layout()
    _w = np.array(_wins)
    mo.vstack([_fig, mo.md(
        f"decision states where learned ≠ optimal: **{len(_diff)}** of 200. exact expected winnings per game: "
        f"optimal **{exact_value(optimal_policy):.4f}**, your learned greedy policy **{exact_value(learned):.4f}**, "
        f"stick on 20+ **{exact_value(_stick20):.4f}**. average return during training (the ε-greedy behaviour, last "
        f"100,000 games): **{np.mean(_rets[-100_000:]):.4f}**. 100,000 games of the optimal policy in your env: "
        f"**{_w.mean():.4f} ± {_w.std() / np.sqrt(len(_w)):.4f}**.")])
    return (learned,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where are the disagreements (which sums, which dealer cards), and why
    there? why does a policy that's wrong in that many states lose so little? why is the
    training return worse than the learned greedy policy's value? compare the two maps with
    Sutton & Barto's Figure 5.2: where does a usable ace make you hit more boldly, and why?
    """)
    return


@app.cell
def _(action_gap, learned, optimal_policy):
    # how close were the disagreements? the exact gap |Q*(hit) - Q*(stick)| in each
    _dis = [o for o in range(360) if ex.Blackjack.decode(o)[0] >= 12 and learned[o] != optimal_policy[o]]
    _all = [action_gap(o) for o in range(360) if ex.Blackjack.decode(o)[0] >= 12]
    _rows = []
    for _o in _dis:
        _ps, _d, _u = ex.Blackjack.decode(_o)
        _rows.append(f"| {_ps} | {_d} | {'yes' if _u else 'no'} | {'hit' if optimal_policy[_o] else 'stick'} "
                     f"| {action_gap(_o):.4f} |")
    mo.md("the states where your greedy policy differs from the optimum, with the exact value gap between the two "
          f"actions there (median gap over all 200 decision states: {np.median(_all):.3f}):\n\n"
          "| sum | dealer | usable ace | optimal | gap |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: ordinary versus weighted importance sampling

    Sutton & Barto's Example 5.4: estimate the value of one state, a sum of 13 with a usable
    ace (an ace and a 2) against a dealer's 2, under the **target** policy "stick only on 20
    or 21", from episodes of a **uniformly random** behaviour policy. the target hits in that
    state, so we track your estimate of $Q(s, \text{hit})$; its exact value is $-0.27726$.
    every episode starts in that state (a small subclass of your `Blackjack` overrides
    `reset`). we run 100 independent estimates for each of $n = 1, 3, 10, \dots, 1000$
    episodes and plot the mean squared error.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which estimator has the lower error after 10 episodes? after 1,000? by how much?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    class FixedStart(ex.Blackjack):
        def reset(self, *, seed=None):
            Env.reset(self, seed=seed)
            self.player = [1, 2]
            self.dealer = [2, self.draw()]
            return self.observe(), {}

    _s0 = ex.Blackjack.encode(13, 2, True)
    _target = np.zeros((360, 2))
    for _o in range(360):
        _target[_o] = [1.0, 0.0] if ex.Blackjack.decode(_o)[0] >= 20 else [0.0, 1.0]
    _behavior = np.full((360, 2), 0.5)
    _true = -0.27726
    _ns = [1, 3, 10, 30, 100, 300, 1000]
    _mse = {}
    for _w in (False, True):
        _mse[_w] = [np.mean([(ex.off_policy_mc_prediction(FixedStart(), _target, _behavior, _n, weighted=_w,
                                                          rng=1000 * _i + _r)[_s0, 1] - _true) ** 2
                             for _r in range(100)]) for _i, _n in enumerate(_ns)]
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.loglog(_ns, _mse[False], "o-", label="ordinary IS")
    _ax.loglog(_ns, _mse[True], "s-", label="weighted IS")
    _ax.set(xlabel="episodes", ylabel="mean squared error (100 runs)")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "| episodes | " + " | ".join(str(n) for n in _ns) + " |\n|---|" + "---|" * len(_ns) + "\n"
        "| ordinary | " + " | ".join(f"{v:.3f}" for v in _mse[False]) + " |\n"
        "| weighted | " + " | ".join(f"{v:.3f}" for v in _mse[True]) + " |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is ordinary IS so bad with few episodes (what values can $\rho G$
    take here)? why does weighted IS start with a moderate error even though it's "the good
    one"? why do the two curves meet for large $n$, and what does their slope tell you?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: right on average, and useless

    the one-state MDP from on paper 3: "left" loops back with probability 0.9 (reward 0) or
    ends with $+1$; "right" ends with 0. the target always goes left, so
    $Q^\pi(s, \text{left}) = 1$. the behaviour is uniform. we run your
    `off_policy_mc_prediction` 200 times with 1,000 episodes each, ordinary and weighted, and
    look at the spread of the 200 estimates.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "every term of ordinary IS has the right expectation. what will the median of the 200 estimates be? the largest?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    class OneState(Env):
        def __init__(self):
            self.observation_space, self.action_space = Discrete(2), Discrete(2)

        def reset(self, *, seed=None):
            super().reset(seed=seed)
            return 0, {}

        def step(self, action):
            if action == 1:
                return 1, 0.0, True, False, {}
            if self.np_random.random() < 0.1:
                return 1, 1.0, True, False, {}
            return 0, 0.0, False, False, {}

    _tp = np.array([[1.0, 0.0], [1.0, 0.0]])
    _bp = np.full((2, 2), 0.5)
    _est = {w: np.array([ex.off_policy_mc_prediction(OneState(), _tp, _bp, 1000, weighted=w, rng=r)[0, 0]
                         for r in range(200)]) for w in (False, True)}
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.hist(_est[False], bins=np.linspace(0, 6, 61), alpha=0.7, label="ordinary IS")
    _ax.axvline(1.0, color="k", ls="--", label="true value 1")
    _ax.axvline(np.median(_est[False]), color="C3", ls=":", label="median (ordinary)")
    _ax.set(xlabel="estimate after 1,000 episodes", ylabel="runs (of 200)")
    _ax.legend(fontsize=8)
    _o = _est[False]
    mo.vstack([_fig, mo.md(
        f"ordinary IS: mean **{_o.mean():.3f}**, median **{np.median(_o):.3f}**, largest **{_o.max():.2f}**, "
        f"{np.mean(_o < 1):.0%} of runs below the true value, sample std {_o.std():.2f}. weighted IS: every run gives "
        f"**{_est[True].min():.3f}** to **{_est[True].max():.3f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how can an estimator built from terms that are right on average be below the truth in most runs? where is
    the missing mass? on paper 3 says the variance is infinite: what does that mean for the
    standard error you'd report, and for how the sample mean behaves as you add episodes?
    why is weighted IS exact here (and would it still be if the returns varied)?
    """)
    return


if __name__ == "__main__":
    app.run()
