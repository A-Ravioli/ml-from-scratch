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
    # 23 · capstone: design your own environment

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
        raise NotImplementedError("stage1: implement Catch.__init__")

    def encode(self, ball_row, ball_col, paddle):
        """stage 1: (ball_row, ball_col, paddle) -> obs int."""
        raise NotImplementedError("stage1: implement Catch.encode")

    def decode(self, obs):
        """stage 1: obs -> (ball_row, ball_col, paddle) ints."""
        raise NotImplementedError("stage1: implement Catch.decode")

    def reset(self, *, seed=None):
        """stage 1: ball at row 0 in a column drawn with self.np_random.integers(cols); paddle at cols // 2."""
        raise NotImplementedError("stage1: implement Catch.reset")

    def step(self, action):
        """stage 1: move the paddle, drop the ball a row with wind (u = self.np_random.random()); +1/-1 on landing."""
        raise NotImplementedError("stage1: implement Catch.step")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", Catch=Catch)
    return


@app.function
def check_env(env, *, n_steps=20000, seed=0):
    """stage 2: run env with random actions and return a list of problems (strings); [] means it passed."""
    raise NotImplementedError("stage2: implement check_env")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", check_env=check_env, Catch=Catch)
    return


@app.function
def evaluate_policy(env, policy, n_episodes, *, seed=0):
    """stage 3: mean undiscounted return of policy (callable obs -> action) over n_episodes, and its standard error."""
    raise NotImplementedError("stage3: implement evaluate_policy")


@app.function
def q_policy(Q):
    """stage 3: the greedy policy for a Q table, as a callable obs -> int (ties to the lowest action)."""
    raise NotImplementedError("stage3: implement q_policy")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", evaluate_policy=evaluate_policy, q_policy=q_policy, Catch=Catch)
    return


@app.class_definition
class CatchFeatures:
    """features for Catch that depend on the ball's row and its offset from the paddle, not on absolute position."""

    def __init__(self, env):
        """stage 4: store env; n_features = rows * (2 cols - 1) + (2 cols - 1)."""
        raise NotImplementedError("stage4: implement CatchFeatures.__init__")

    def __call__(self, obs):
        """stage 4: two active indices: (ball_row, ball_col - paddle) and, after those, (ball_col - paddle) alone."""
        raise NotImplementedError("stage4: implement CatchFeatures.__call__")


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
    over several seeds. after training, evaluate the *greedy* policy with `evaluate_policy`,
    and report the mean and spread over seeds. you write this harness.
    """)
    return


@app.cell
def _(v_opt):
    # your harness. for each learner (q_learning, sarsa, mc_control on the table; linear_sarsa with
    # CatchFeatures), each number of training episodes n in CHECKPOINTS and each of SEEDS seeds: train,
    # evaluate the greedy policy with evaluate_policy (q_policy for the tables), and plot the mean and
    # standard error over seeds against n, with v_opt as a reference line. print a table too.
    mo.md(f"experiment B isn't written yet. the optimum to compare with is {v_opt:.3f}.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C: your own game

    replace `MyGame` below with your own design (see the brief for ideas and requirements).
    run your `check_env` on it, solve it exactly if it's tabular, and compare at least one
    tabular and one approximate learner.
    """)
    return


@app.class_definition
class MyGame(Catch):
    """your game goes here: replace this class (it's plain Catch until you do)."""


@app.cell
def _():
    # run check_env(MyGame()) until it reports no problems; solve your game exactly if it's tabular
    # (value_iteration on its P and R); then train at least one tabular and one approximate learner
    # and compare them with the optimum using evaluate_policy.
    _problems = check_env(MyGame(), n_steps=5000)
    mo.md(f"`check_env(MyGame())`: **{_problems or 'no problems'}**")
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
