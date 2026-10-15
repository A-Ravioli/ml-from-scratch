import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.sampling import normal  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # markov chains & stochastic processes · build (reference solution)

    work through the stages in order. a Markov chain on states $0, \dots, S-1$ is described by
    a **transition matrix** `P` of shape `(S, S)`: row `i` is the distribution of the next
    state when the current state is `i`. distributions over states are ROW vectors, and one
    step multiplies on the right: `p @ P`. `simulate_markov_chain`, `stationary_distribution`
    and `brownian_motion` become part of `mlfs.sampling`.

    | stage | you build |
    |---|---|
    | 1 | `evolve`: push a distribution forward through the chain |
    | 2 | `simulate_markov_chain`: one random path |
    | 3 | `stationary_distribution`: the distribution that doesn't change |
    | 4 | `total_variation`, `spectral_gap`, `mixing_time`: how fast the chain forgets where it started |
    | 5 | `brownian_motion`: the continuous-time limit of a random walk |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def evolve(p0, P, n_steps: int) -> np.ndarray:
    """stage 1: the distributions p_0, p_1, ..., p_n of the chain's state, starting from p0 (S,).

    p_{t+1} = p_t @ P (a row vector times the matrix). returns (n_steps + 1, S), row t = p_t.
    """
    P = np.asarray(P, dtype=float)
    out = np.empty((n_steps + 1, P.shape[0]))
    out[0] = np.asarray(p0, dtype=float)
    for t in range(n_steps):
        out[t + 1] = out[t] @ P
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, evolve=evolve)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def simulate_markov_chain(P, x0: int, n_steps: int, *, rng=None) -> np.ndarray:
    """stage 2: one path of the chain: int states (n_steps + 1,), starting at x0.

    at each step, draw the next state from row P[current] by the discrete inverse-CDF method
    (cumulative sums of the row, one rng.random() per step). raises ValueError unless P is
    square with non-negative entries and rows summing to 1 (within 1e-8).
    """
    P = np.asarray(P, dtype=float)
    S = P.shape[0]
    if P.ndim != 2 or P.shape != (S, S) or np.any(P < 0) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("P must be a square matrix with non-negative rows summing to 1")
    rng = np.random.default_rng(rng)
    cdf = np.cumsum(P, axis=1)
    u = rng.random(n_steps)
    path = np.empty(n_steps + 1, dtype=np.int64)
    path[0] = x0
    for t in range(n_steps):
        row = cdf[path[t]]
        path[t + 1] = np.searchsorted(row, u[t] * row[-1], side="right")
    return path


@app.cell(hide_code=True)
def _():
    checks.run(2, simulate_markov_chain=simulate_markov_chain)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def stationary_distribution(P) -> np.ndarray:
    """stage 3: the stationary distribution pi (S,) with pi @ P == pi and sum(pi) == 1.

    solve the linear system (P^T - I) pi = 0 together with the extra equation sum(pi) = 1:
    stack them into an (S + 1, S) system and solve it by least squares (np.linalg.lstsq).
    assumes the chain is irreducible (so pi is unique). works for periodic chains too.
    """
    P = np.asarray(P, dtype=float)
    S = P.shape[0]
    A = np.vstack([P.T - np.eye(S), np.ones((1, S))])
    b = np.zeros(S + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)  # remove -1e-17s
    return pi / pi.sum()


@app.cell(hide_code=True)
def _():
    checks.run(3, stationary_distribution=stationary_distribution)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def total_variation(p, q) -> float:
    """stage 4: the total variation distance between distributions p, q (S,): half the L1 distance."""
    return 0.5 * float(np.sum(np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float))))


@app.function
def spectral_gap(P) -> float:
    """stage 4: 1 - |lambda_2|, where |lambda_2| is the second-largest eigenvalue MODULUS of P.

    (the largest is always 1.) eigenvalues can be complex: compare np.abs of np.linalg.eigvals.
    """
    mods = np.sort(np.abs(np.linalg.eigvals(np.asarray(P, dtype=float))))[::-1]
    return float(1.0 - mods[1])


@app.function
def mixing_time(P, eps: float = 0.25, max_steps: int = 10_000):
    """stage 4: the first t with max over starting states x of TV(P^t[x], pi) <= eps.

    P^t[x] (row x of the t-th power) is the distribution after t steps from x. evolve all
    starting states at once by repeatedly multiplying a matrix that starts as the identity.
    returns an int, or None if the distance is still above eps after max_steps steps.
    """
    P = np.asarray(P, dtype=float)
    pi = stationary_distribution(P)
    D = np.eye(P.shape[0])
    for t in range(max_steps + 1):
        if 0.5 * np.max(np.sum(np.abs(D - pi), axis=1)) <= eps:
            return t
        D = D @ P
    return None


@app.cell(hide_code=True)
def _():
    checks.run(4, total_variation=total_variation, spectral_gap=spectral_gap, mixing_time=mixing_time)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def brownian_motion(n_paths: int, n_steps: int, *, dt: float = 0.01, rng=None) -> np.ndarray:
    """stage 5: n_paths Brownian paths on the time grid 0, dt, 2 dt, ..., n_steps dt.

    each path starts at 0 and adds independent N(0, dt) increments (standard deviation
    sqrt(dt)), drawn with YOUR mlfs.sampling.normal. returns (n_paths, n_steps + 1).
    """
    inc = normal(0.0, math.sqrt(dt), (n_paths, n_steps), rng=rng)
    return np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)


@app.cell(hide_code=True)
def _():
    checks.run(5, brownian_motion=brownian_motion)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your chain tools are part of `mlfs.sampling` (try
    `from mlfs.sampling import stationary_distribution`). then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
