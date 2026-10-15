import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.rl import GridWorld, epsilon_greedy  # yours, from topics 02 and 04
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # temporal-difference learning · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. TD methods learn from every single step,
    bootstrapping from their own estimates: the idea behind almost every RL algorithm in
    use today.

    | stage | you build |
    |---|---|
    | 1 | `CliffWalking`, `td0_prediction`: the cliff, and TD(0) policy evaluation → `mlfs.rl` |
    | 2 | `sarsa`, `q_learning`: on- and off-policy TD control → `mlfs.rl` |
    | 3 | `expected_sarsa`: average over the next action instead of sampling it → `mlfs.rl` |
    | 4 | `n_step_sarsa`: bootstrap after n steps instead of one → `mlfs.rl` |
    | 5 | `td_lambda_prediction`: all n at once, with eligibility traces |

    as in topic 04, every learner seeds its environment once from `rng` (first episode:
    `env.reset(seed=int(rng.integers(2**31)))`) and picks actions with your `epsilon_greedy`.
    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class CliffWalking(GridWorld):
    """Sutton & Barto's 4x12 cliff: start (3, 0), goal (3, 11), cliff (3, 1)..(3, 10)."""

    def __init__(self):
        """stage 1: a GridWorld(4, 12, start=(3, 0), goal=(3, 11), max_steps=1000) whose moves into the cliff cost -100 and return to start."""
        raise NotImplementedError("stage1: implement CliffWalking.__init__")


@app.function
def td0_prediction(env, policy, n_episodes, *, alpha=0.1, gamma=0.99, rng=None):
    """stage 1: V (S,) for policy (callable obs -> action) by TD(0); bootstrap unless terminated."""
    raise NotImplementedError("stage1: implement td0_prediction")


@app.cell(hide_code=True)
def _():
    checks.run(1, CliffWalking=CliffWalking, td0_prediction=td0_prediction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sarsa(env, n_episodes, *, alpha=0.5, gamma=1.0, epsilon=0.1, rng=None):
    """stage 2: on-policy TD control, target r + gamma Q(s', a') with a' the next epsilon-greedy action; returns (Q, returns)."""
    raise NotImplementedError("stage2: implement sarsa")


@app.function
def q_learning(env, n_episodes, *, alpha=0.5, gamma=1.0, epsilon=0.1, rng=None):
    """stage 2: off-policy TD control, target r + gamma max_a Q(s', a); returns (Q, returns)."""
    raise NotImplementedError("stage2: implement q_learning")


@app.cell(hide_code=True)
def _():
    checks.run(2, sarsa=sarsa, q_learning=q_learning, CliffWalking=CliffWalking)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def expected_sarsa(env, n_episodes, *, alpha=0.5, gamma=1.0, epsilon=0.1, rng=None):
    """stage 3: target r + gamma sum_a pi(a|s') Q(s', a) under the epsilon-greedy policy; returns (Q, returns)."""
    raise NotImplementedError("stage3: implement expected_sarsa")


@app.cell(hide_code=True)
def _():
    checks.run(3, expected_sarsa=expected_sarsa, sarsa=sarsa, q_learning=q_learning, CliffWalking=CliffWalking)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def n_step_sarsa(env, n_episodes, n=4, *, alpha=0.5, gamma=1.0, epsilon=0.1, rng=None):
    """stage 4: SARSA with n-step returns (bootstrap from Q(S_{t+n}, A_{t+n}) unless the episode terminated first); returns (Q, returns)."""
    raise NotImplementedError("stage4: implement n_step_sarsa")


@app.cell(hide_code=True)
def _():
    checks.run(4, n_step_sarsa=n_step_sarsa, sarsa=sarsa, CliffWalking=CliffWalking)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def td_lambda_prediction(env, policy, n_episodes, *, lam=0.9, alpha=0.1, gamma=0.99, rng=None):
    """stage 5: V (S,) by TD(lambda) with accumulating eligibility traces (reset every episode)."""
    raise NotImplementedError("stage5: implement td_lambda_prediction")


@app.cell(hide_code=True)
def _():
    checks.run(5, td_lambda_prediction=td_lambda_prediction, td0_prediction=td0_prediction)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `CliffWalking` and the TD family are part of `mlfs.rl`. open
    the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
