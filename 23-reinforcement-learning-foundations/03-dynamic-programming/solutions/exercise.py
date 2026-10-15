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
    # dynamic programming · build (reference solution)

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
    Q = np.asarray(Q)
    policy = np.zeros(Q.shape)
    policy[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1.0
    return policy


@app.function
def bellman_optimality(P, R, V, gamma=0.99):
    """stage 1: the optimality backup (T V)(s) = max_a [R(s, a) + gamma sum_s' P(s'|s, a) V(s')], shape (S,)."""
    return np.max(q_from_v(P, R, V, gamma), axis=1)


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
    S, A = R.shape
    V = policy_evaluation(P, R, np.full((S, A), 1.0 / A), gamma, tol=tol)
    policy = greedy_policy(q_from_v(P, R, V, gamma))
    for _ in range(1000):
        V = policy_evaluation(P, R, policy, gamma, tol=tol)
        Q = q_from_v(P, R, V, gamma)
        current = np.argmax(policy, axis=1)
        better = Q.max(axis=1) > Q[np.arange(S), current] + tol
        if not better.any():
            break
        policy = policy.copy()
        policy[better] = greedy_policy(Q)[better]
    return policy, V


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
    V = np.zeros(R.shape[0])
    for _ in range(max_iters):
        V_new = bellman_optimality(P, R, V, gamma)
        done = np.max(np.abs(V_new - V)) < tol
        V = V_new
        if done:
            break
    return greedy_policy(q_from_v(P, R, V, gamma)), V


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
    V = np.zeros(R.shape[0])
    n = 0
    while n < max_iters:
        n += 1
        policy = greedy_policy(q_from_v(P, R, V, gamma))
        P_pi = np.einsum("sa,sat->st", policy, P)
        R_pi = np.sum(policy * R, axis=1)
        V_old = V
        for _ in range(m):
            V = R_pi + gamma * P_pi @ V
        if np.max(np.abs(V - V_old)) < tol:
            break
    return greedy_policy(q_from_v(P, R, V, gamma)), V, n


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
