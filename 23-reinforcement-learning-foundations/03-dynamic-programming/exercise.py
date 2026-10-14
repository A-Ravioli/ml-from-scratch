import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.rl import policy_evaluation, q_from_v  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # dynamic programming · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. with a model of the MDP (`P` and
    `R` as arrays) you can compute the optimal policy exactly; these solvers are the ground
    truth that every learning algorithm in the rest of the chapter is checked against.

    | stage | you build |
    |---|---|
    | 1 | `greedy_policy`, `bellman_optimality`: act greedily on Q, and the Bellman optimality backup → `mlfs.rl` |
    | 2 | `policy_iteration`: evaluate, improve, repeat → `mlfs.rl` |
    | 3 | `value_iteration`: iterate the optimality backup → `mlfs.rl` |
    | 4 | `modified_policy_iteration`: the family that contains both, with an iteration count |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def greedy_policy(Q):
    """stage 1: Q (S, A) -> deterministic one-hot policy (S, A), ties to the lowest action."""
    raise NotImplementedError("stage1: implement greedy_policy")


@app.function
def bellman_optimality(P, R, V, gamma=0.99):
    """stage 1: the optimality backup (T V)(s) = max_a [R(s, a) + gamma sum_s' P(s'|s, a) V(s')], shape (S,)."""
    raise NotImplementedError("stage1: implement bellman_optimality")


@app.cell(hide_code=True)
def _():
    checks.run(1, greedy_policy=greedy_policy, bellman_optimality=bellman_optimality)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def policy_iteration(P, R, gamma=0.99, *, tol=1e-08):
    """stage 2: alternate policy_evaluation and greedy improvement until stable; returns (policy (S, A) one-hot, V (S,))."""
    raise NotImplementedError("stage2: implement policy_iteration")


@app.cell(hide_code=True)
def _():
    checks.run(2, policy_iteration=policy_iteration, greedy_policy=greedy_policy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def value_iteration(P, R, gamma=0.99, *, tol=1e-08, max_iters=10000):
    """stage 3: iterate V <- max_a (R + gamma P V) until the largest change is below tol; returns (greedy policy, V)."""
    raise NotImplementedError("stage3: implement value_iteration")


@app.cell(hide_code=True)
def _():
    checks.run(3, value_iteration=value_iteration, bellman_optimality=bellman_optimality, greedy_policy=greedy_policy)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def modified_policy_iteration(P, R, gamma=0.99, *, m=5, tol=1e-08, max_iters=10000):
    """stage 4: greedy step, then m evaluation sweeps, until V changes by < tol; returns (policy, V, n_iters)."""
    raise NotImplementedError("stage4: implement modified_policy_iteration")


@app.cell(hide_code=True)
def _():
    checks.run(4, modified_policy_iteration=modified_policy_iteration, value_iteration=value_iteration)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `greedy_policy`, `policy_iteration` and `value_iteration` are
    part of `mlfs.rl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
