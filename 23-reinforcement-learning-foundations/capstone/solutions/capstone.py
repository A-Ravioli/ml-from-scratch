import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.rl import (  # yours, from this chapter
        Discrete,
        Env,
        linear_sarsa,
        mc_control,
        q_learning,
        rollout,
        sarsa,
        value_iteration,
    )

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    SEEDS = 3 if SMALL else 10
    CHECKPOINTS = [25, 100, 400] if SMALL else [25, 50, 100, 200, 400, 800, 1600]
    EVAL_EPISODES = 200 if SMALL else 2000


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 23 · capstone: design your own environment (reference solution)

    read [`brief.md`](brief.md) first. you'll build a small game as a gym-style environment
    with an exact tabular model, a checker that catches the bugs every new environment has, an
    evaluation harness, and features for an approximate learner. then you solve the game
    exactly (value iteration), with tabular learners and with a linear learner, compare them,
    and design a game of your own.

    | stage | you build |
    |---|---|
    | 1 | `Catch`: a falling-ball game with wind, as an `Env` with `P` and `R` |
    | 2 | `check_env`: test any environment against the protocol and its own model |
    | 3 | `evaluate_policy`, `q_policy`: estimate a policy's return with an error bar |
    | 4 | `CatchFeatures`: a featurizer for `linear_sarsa` that generalizes across positions |

    when they're green, scroll down: the experiments run on your functions, and the write-up
    prompts are at the end.
    """)
    return


@app.class_definition
class Catch(Env):
    """a ball falls from the top row; move the paddle on the bottom row to catch it.

    obs = (ball_row * cols + ball_col) * cols + paddle; actions 0 left, 1 stay, 2 right.
    """

    def __init__(self, rows=6, cols=5, *, wind=0.2):
        """stage 1: store rows, cols, wind; build n_states, n_actions, spaces, terminal (S,), P (S, A, S'), R (S, A)."""
        self.rows, self.cols, self.wind = int(rows), int(cols), float(wind)
        S, A = self.rows * self.cols * self.cols, 3
        self.n_states, self.n_actions = S, A
        self.observation_space, self.action_space = Discrete(S), Discrete(A)
        self.terminal = np.array([self.decode(s)[0] == self.rows - 1 for s in range(S)])
        self.P = np.zeros((S, A, S))
        self.R = np.zeros((S, A))
        drifts = [(-1, self.wind / 2), (0, 1 - self.wind), (1, self.wind / 2)]
        for s in range(S):
            r, c, p = self.decode(s)
            if self.terminal[s]:
                self.P[s, :, s] = 1.0
                continue
            for a in range(A):
                p2 = min(max(p + a - 1, 0), self.cols - 1)
                for dc, prob in drifts:
                    c2 = min(max(c + dc, 0), self.cols - 1)
                    self.P[s, a, self.encode(r + 1, c2, p2)] += prob
                    if r + 1 == self.rows - 1:
                        self.R[s, a] += prob * (1.0 if c2 == p2 else -1.0)

    def encode(self, ball_row, ball_col, paddle):
        """stage 1: (ball_row, ball_col, paddle) -> obs int."""
        return (int(ball_row) * self.cols + int(ball_col)) * self.cols + int(paddle)

    def decode(self, obs):
        """stage 1: obs -> (ball_row, ball_col, paddle) ints."""
        rest, paddle = divmod(int(obs), self.cols)
        ball_row, ball_col = divmod(rest, self.cols)
        return ball_row, ball_col, paddle

    def reset(self, *, seed=None):
        """stage 1: ball at row 0 in a column drawn with self.np_random.integers(cols); paddle at cols // 2."""
        super().reset(seed=seed)
        self.ball = [0, int(self.np_random.integers(self.cols))]
        self.paddle = self.cols // 2
        return self.encode(self.ball[0], self.ball[1], self.paddle), {}

    def step(self, action):
        """stage 1: move the paddle, drop the ball a row with wind (u = self.np_random.random()); +1/-1 on landing."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}: 0, 1 or 2")
        if self.ball[0] >= self.rows - 1:
            return self.encode(*self.ball, self.paddle), 0.0, True, False, {}
        self.paddle = min(max(self.paddle + action - 1, 0), self.cols - 1)
        u = self.np_random.random()
        dc = -1 if u < self.wind / 2 else (1 if u < self.wind else 0)
        self.ball = [self.ball[0] + 1, min(max(self.ball[1] + dc, 0), self.cols - 1)]
        terminated = self.ball[0] == self.rows - 1
        reward = (1.0 if self.ball[1] == self.paddle else -1.0) if terminated else 0.0
        return self.encode(*self.ball, self.paddle), reward, terminated, False, {}


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", Catch=Catch)
    return


@app.function
def check_env(env, *, n_steps=20000, seed=0):
    """stage 2: run env with random actions and return a list of problems (strings); [] means it passed."""
    problems = []

    def report(msg):
        if msg not in problems:
            problems.append(msg)

    rng = np.random.default_rng(seed)
    out = env.reset(seed=seed)
    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)):
        return ["reset(seed=...) must return a tuple (obs, info) with info a dict"]
    obs = out[0]
    if not env.observation_space.contains(obs):
        report(f"reset returned obs {obs!r}, which is not in observation_space")

    tabular = all(hasattr(env, k) for k in ("P", "R", "n_states", "n_actions"))
    if tabular:
        P, R = np.asarray(env.P), np.asarray(env.R)
        S, A = env.n_states, env.n_actions
        if P.shape != (S, A, S) or R.shape != (S, A):
            report(f"P should have shape {(S, A, S)} and R {(S, A)}; got {P.shape} and {R.shape}")
            tabular = False
        elif np.any(P < 0) or not np.allclose(P.sum(axis=2), 1.0):
            report("every P[s, a, :] must be a probability distribution (nonnegative, summing to 1)")
        if tabular and hasattr(env, "terminal"):
            for s in np.flatnonzero(env.terminal):
                if not (np.allclose(P[s, :, s], 1.0) and np.allclose(R[s], 0.0)):
                    report(f"terminal state {s} must be absorbing with zero reward")
                    break

    counts, rewards = {}, {}
    steps = 0
    while steps < n_steps:
        a = env.action_space.sample(rng)
        res = env.step(a)
        steps += 1
        if not (isinstance(res, tuple) and len(res) == 5):
            return problems + ["step must return a 5-tuple (obs, reward, terminated, truncated, info)"]
        obs2, r, term, trunc, info = res
        if not env.observation_space.contains(obs2):
            report(f"step returned obs {obs2!r}, which is not in observation_space")
        if not isinstance(r, float):
            report(f"rewards must be Python floats; got {type(r).__name__}")
        if type(term) is not bool or type(trunc) is not bool:
            report(f"terminated and truncated must be Python bools; got {type(term)} and {type(trunc)}")
        if not isinstance(info, dict):
            report("info must be a dict")
        if tabular and env.observation_space.contains(obs) and env.observation_space.contains(obs2):
            key = (int(obs), int(a))
            counts.setdefault(key, np.zeros(env.n_states))[int(obs2)] += 1
            rewards.setdefault(key, []).append(float(r))
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = obs2

    if tabular and rewards:
        scale = np.std(np.concatenate([np.asarray(v) for v in rewards.values()]))  # typical reward spread
        bad_p, bad_r = [], []
        for (s, a), c in counts.items():
            n = c.sum()
            if n < 30:  # too few samples to judge
                continue
            p = P[s, a]
            pc = np.clip(p, 0.0, 1.0)  # a broken P may hold negative entries
            if np.any((c > 0) & (pc == 0)) or np.any(np.abs(c / n - pc) > 6 * np.sqrt(pc * (1 - pc) / n) + 2.0 / n):
                bad_p.append(f"(s={s}, a={a}, {int(n)} samples)")
            rs = np.array(rewards[(s, a)])
            if abs(rs.mean() - R[s, a]) > 6 * max(rs.std(), scale) / np.sqrt(n) + 1e-9:
                bad_r.append(f"(s={s}, a={a}: mean {rs.mean():.3f}, R {R[s, a]:.3f})")
        if bad_p:
            report(f"step's next states disagree with P for {len(bad_p)} state-action pairs, e.g. "
                   + ", ".join(bad_p[:3]))
        if bad_r:
            report(f"step's rewards disagree with R for {len(bad_r)} state-action pairs, e.g. " + ", ".join(bad_r[:3]))

    def trajectory():  # 100 steps with a fixed action sequence, starting from reset(seed=seed + 1)
        acts = np.random.default_rng(seed + 2)
        o, _ = env.reset(seed=seed + 1)
        seq = [np.ravel(o).tolist()]
        for _ in range(100):
            o, r, te, tr, _ = env.step(env.action_space.sample(acts))
            seq.append((np.ravel(o).tolist(), float(r)))
            if te or tr:
                o, _ = env.reset()
                seq.append(np.ravel(o).tolist())
        return seq

    if trajectory() != trajectory():
        report("reset(seed=s) must make the episode reproducible: draw every random number from self.np_random")
    return problems


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", check_env=check_env, Catch=Catch)
    return


@app.function
def evaluate_policy(env, policy, n_episodes, *, seed=0):
    """stage 3: mean undiscounted return of policy (callable obs -> action) over n_episodes, and its standard error."""
    returns = [rollout(env, policy, max_steps=10_000, seed=seed if k == 0 else None)["return"]
               for k in range(n_episodes)]
    returns = np.array(returns, dtype=float)
    se = returns.std(ddof=1) / np.sqrt(n_episodes) if n_episodes > 1 else float("nan")
    return float(returns.mean()), float(se)


@app.function
def q_policy(Q):
    """stage 3: the greedy policy for a Q table, as a callable obs -> int (ties to the lowest action)."""
    Q = np.array(Q)
    return lambda obs: int(np.argmax(Q[obs]))


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", evaluate_policy=evaluate_policy, q_policy=q_policy, Catch=Catch)
    return


@app.class_definition
class CatchFeatures:
    """features for Catch that depend on the ball's row and its offset from the paddle, not on absolute position."""

    def __init__(self, env):
        """stage 4: store env; n_features = rows * (2 cols - 1) + (2 cols - 1)."""
        self.env = env
        self.n_offsets = 2 * env.cols - 1
        self.n_features = env.rows * self.n_offsets + self.n_offsets

    def __call__(self, obs):
        """stage 4: two active indices: (ball_row, ball_col - paddle) and, after those, (ball_col - paddle) alone."""
        r, c, p = self.env.decode(obs)
        dx = c - p + self.env.cols - 1
        return np.array([r * self.n_offsets + dx, self.env.rows * self.n_offsets + dx])


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", CatchFeatures=CatchFeatures, Catch=Catch)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A: the exact answer

    value iteration on `Catch`'s model gives $V^*$ and the optimal policy. the start state is
    random (the ball's column), so the optimal expected return is the average of $V^*$ over
    the starting columns. we check it against `evaluate_policy` on the real environment.
    """)
    return


@app.cell
def _():
    game = Catch()
    pi_star, V_star = value_iteration(game.P, game.R, gamma=1.0)
    starts = [game.encode(0, c, game.cols // 2) for c in range(game.cols)]
    v_opt = float(np.mean(V_star[starts]))
    _m, _se = evaluate_policy(Catch(), q_policy(pi_star), EVAL_EPISODES, seed=1)
    _stay = evaluate_policy(Catch(), lambda obs: 1, EVAL_EPISODES, seed=1)
    mo.md(f"optimal expected return (exact, from value iteration): **{v_opt:.4f}**. measured with "
          f"`evaluate_policy` over {EVAL_EPISODES} episodes: **{_m:.3f} ± {_se:.3f}**. baseline 'never move': "
          f"**{_stay[0]:.3f} ± {_stay[1]:.3f}**.")
    return game, starts, v_opt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B: tabular versus approximate learners

    Q-learning, SARSA and Monte Carlo control on the table, and `linear_sarsa` with
    `CatchFeatures` (ε = 0.1 for all), each trained for an increasing number of episodes,
    over several seeds. after training we evaluate the *greedy* policy with
    `evaluate_policy`, and report the mean and spread over seeds.
    """)
    return


@app.cell
def _(v_opt):
    _feat = CatchFeatures(Catch())

    def _linear(n, s):
        w, _ = linear_sarsa(Catch(), _feat, n, alpha=0.5, epsilon=0.1, rng=s)
        return lambda obs: int(np.argmax(w[_feat(obs)].sum(axis=0)))

    learners = {
        "Q-learning": lambda n, s: q_policy(q_learning(Catch(), n, alpha=0.5, epsilon=0.1, rng=s)[0]),
        "SARSA": lambda n, s: q_policy(sarsa(Catch(), n, alpha=0.5, epsilon=0.1, rng=s)[0]),
        "MC control": lambda n, s: q_policy(mc_control(Catch(), n, epsilon=0.1, rng=s)[0]),
        "linear SARSA + CatchFeatures": _linear,
    }
    results = {}
    for _name, _make in learners.items():
        results[_name] = np.array([[evaluate_policy(Catch(), _make(n, s), EVAL_EPISODES // 4, seed=100 + s)[0]
                                    for s in range(SEEDS)] for n in CHECKPOINTS])
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    for _name, _r in results.items():
        _m, _s = _r.mean(axis=1), _r.std(axis=1, ddof=1) / np.sqrt(SEEDS)
        _ax.errorbar(CHECKPOINTS, _m, yerr=_s, marker="o", capsize=3, label=_name)
    _ax.axhline(v_opt, color="k", ls="--", label=f"optimal ({v_opt:.3f})")
    _ax.set(xscale="log", xlabel="training episodes", ylabel="greedy policy's return (mean ± s.e. over seeds)")
    _ax.legend(fontsize=8)
    _rows = [f"| {n} | " + " | ".join(f"{r[i].mean():.3f}" for r in results.values()) + " |"
             for i, n in enumerate(CHECKPOINTS)]
    mo.vstack([_fig, mo.md("| episodes | " + " | ".join(results) + " |\n|---|" + "---|" * len(results) + "\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C: your own game

    replace `MyGame` below with your own design (see the brief for ideas and requirements).
    the example is a variant of Catch in which the falling object is a fruit (+1 if caught)
    or a bomb (−1 if caught, 0 if dodged), visible to the agent. run your `check_env` on it,
    solve it exactly if it's tabular, and compare at least one tabular and one approximate
    learner.
    """)
    return


@app.class_definition
class MyGame(Catch):
    """an example design: Catch where the falling object is a fruit or a bomb (the last obs bit)."""

    def __init__(self, rows=6, cols=5, *, wind=0.2, p_bomb=0.3):
        super().__init__(rows, cols, wind=wind)
        self.p_bomb = float(p_bomb)
        S0 = self.n_states
        P0, R0, T0 = self.P, self.R, self.terminal
        S = 2 * S0
        self.n_states = S
        self.observation_space = Discrete(S)
        self.terminal = np.concatenate([T0, T0])
        self.P = np.zeros((S, 3, S))
        self.P[:S0, :, :S0] = P0
        self.P[S0:, :, S0:] = P0
        landing = np.array([self.decode(x)[0] == rows - 2 for x in range(S0)])
        p_catch = (R0 + 1) / 2  # on a landing step, R0 = P(catch) - P(miss)
        self.R = np.concatenate([R0, np.where(landing[:, None], -p_catch, 0.0)])

    def reset(self, *, seed=None):
        obs, info = super().reset(seed=seed)
        self.bomb = bool(self.np_random.random() < self.p_bomb)
        return obs + self.bomb * (self.n_states // 2), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if self.bomb:
            r = -1.0 if r > 0 else 0.0
        return obs + self.bomb * (self.n_states // 2), r, term, trunc, info


@app.cell
def _():
    my = MyGame()
    _problems = check_env(MyGame())
    _pi, _V = value_iteration(my.P, my.R, gamma=1.0)
    _h = my.n_states // 2
    _starts = [my.encode(0, c, my.cols // 2) + b * _h for c in range(my.cols) for b in (0, 1)]
    _weights = [(1 - my.p_bomb) if b == 0 else my.p_bomb for c in range(my.cols) for b in (0, 1)]
    _vopt = float(np.average(_V[_starts], weights=_weights))
    _Q, _ = q_learning(MyGame(), 400 if SMALL else 3000, alpha=0.5, epsilon=0.1, rng=0)
    _m, _se = evaluate_policy(MyGame(), q_policy(_Q), EVAL_EPISODES, seed=5)
    mo.md(f"`check_env(MyGame())`: **{_problems or 'no problems'}**. optimal expected return: **{_vopt:.3f}**. "
          f"Q-learning's greedy policy: **{_m:.3f} ± {_se:.3f}**.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer each prompt with numbers from the experiments above (full config: run with
    `MLFS_OFFLINE` unset). see [`rubric.md`](rubric.md).

    1. **the environment.** describe your game: states, actions, dynamics, rewards, episode
       end, and why it's a reasonable test (what makes it hard? what's the optimal value?).
       what did `check_env` catch while you built it?
    2. **the exact solution.** what does value iteration's policy do, and how does
       `evaluate_policy` confirm it? how big an error bar did you need, and why?
    3. **tabular learners.** rank Q-learning, SARSA and MC control by sample efficiency and by
       final performance, with uncertainty over seeds. explain the ranking with the chapter's
       ideas (bootstrapping, on- vs off-policy, variance of returns).
    4. **approximation.** what did your features generalize across, and what did that buy
       (or cost) compared with the table? is the approximate learner's final policy optimal?
       if not, which states does the representation confuse?
    5. **what you'd do next.** one concrete change to the game or the learners, with the
       result you'd expect and why.
    """)
    return


if __name__ == "__main__":
    app.run()
