import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.autodiff import grad  # yours, from chapter 03
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hamiltonian monte carlo · build (reference solution)

    work through the stages in order. HMC treats the parameters $q$ as the position of a
    particle on the surface $U(q) = -\log\pi(q)$, gives it a random momentum $p$, and lets
    physics carry it far across the distribution before a Metropolis accept/reject step.
    `leapfrog` and `hmc` become part of `mlfs.infer`.

    | stage | you build |
    |---|---|
    | 1 | `hamiltonian`, `leapfrog` → `mlfs.infer`: the energy, and the integrator that (almost) conserves it |
    | 2 | `hmc` → `mlfs.infer`: momentum refresh, a trajectory, and an accept/reject on the energy error |
    | 3 | `hmc` without a gradient: your forward-mode `mlfs.autodiff.grad` fills it in |
    | 4 | `no_u_turn`, `u_turn_steps`: the idea behind NUTS, how long a trajectory should be |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def hamiltonian(q, p, log_prob, inv_mass=None):
    """stage 1: total energy H(q, p) = -log_prob(q) + 1/2 sum(inv_mass * p^2) (inv_mass None = identity)."""
    p = np.asarray(p, dtype=float)
    m = 1.0 if inv_mass is None else np.asarray(inv_mass, dtype=float)
    return float(-log_prob(np.asarray(q, dtype=float)) + 0.5 * np.sum(m * p * p))


@app.function
def leapfrog(q, p, grad_log_prob, step_size, n_steps, inv_mass=None):
    """stage 1: n_steps leapfrog steps (half momentum step, full position steps, half momentum step); returns new (q, p)."""
    q = np.array(q, dtype=float)
    p = np.array(p, dtype=float)
    m = 1.0 if inv_mass is None else np.asarray(inv_mass, dtype=float)
    if n_steps == 0:
        return q, p
    p = p + 0.5 * step_size * grad_log_prob(q)
    for i in range(n_steps):
        q = q + step_size * m * p
        g = grad_log_prob(q)
        p = p + (step_size if i < n_steps - 1 else 0.5 * step_size) * g
    return q, p


@app.cell(hide_code=True)
def _():
    checks.run(1, hamiltonian=hamiltonian, leapfrog=leapfrog)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def hmc(log_prob, x0, n_samples, *, grad_log_prob=None, step_size=0.1, n_leapfrog=20, burn_in=0, inv_mass=None, rng=None):
    """stage 2: Hamiltonian Monte Carlo; returns (samples (n_samples, d) after burn_in, acceptance rate over all iterations)."""
    rng = as_rng(rng)
    if grad_log_prob is None:
        grad_log_prob = lambda x: grad(log_prob, x)
    q = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    d = q.size
    m = np.ones(d) if inv_mass is None else np.asarray(inv_mass, dtype=float)
    total = burn_in + n_samples
    out = np.empty((n_samples, d))
    accepted = 0
    for it in range(total):
        p = rng.standard_normal(d) / np.sqrt(m)  # p ~ N(0, M) with M = diag(1 / inv_mass)
        h0 = hamiltonian(q, p, log_prob, m)
        q_new, p_new = leapfrog(q, p, grad_log_prob, step_size, n_leapfrog, m)
        with np.errstate(over="ignore", invalid="ignore"):
            h1 = hamiltonian(q_new, p_new, log_prob, m) if np.all(np.isfinite(q_new)) else math.inf
        u = rng.random()
        if np.isfinite(h1) and math.log(max(u, 1e-300)) < h0 - h1:
            q = q_new
            accepted += 1
        if it >= burn_in:
            out[it - burn_in] = q
    return out, accepted / max(total, 1)


@app.cell(hide_code=True)
def _():
    checks.run(2, hmc=hmc)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *stage 3 is a branch of `hmc` above: when `grad_log_prob is None`, use your
    `mlfs.autodiff.grad(log_prob, x)`.*
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, hmc=hmc)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def no_u_turn(q_minus, q_plus, p_minus, p_plus):
    """stage 4: True while the trajectory from q_minus to q_plus is still stretching at both ends (Hoffman & Gelman's test)."""
    dq = np.asarray(q_plus, dtype=float) - np.asarray(q_minus, dtype=float)
    return bool(dq @ np.asarray(p_minus, dtype=float) >= 0 and dq @ np.asarray(p_plus, dtype=float) >= 0)


@app.function
def u_turn_steps(q0, p0, grad_log_prob, step_size, max_steps=1000):
    """stage 4: leapfrog one step at a time from (q0, p0); the first step count at which (q - q0) . p < 0, else max_steps."""
    q = np.asarray(q0, dtype=float)
    p = np.asarray(p0, dtype=float)
    q_start = q.copy()
    for s in range(1, max_steps + 1):
        q, p = leapfrog(q, p, grad_log_prob, step_size, 1)
        if (q - q_start) @ p < 0:
            return s
    return max_steps


@app.cell(hide_code=True)
def _():
    checks.run(4, no_u_turn=no_u_turn, u_turn_steps=u_turn_steps)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `leapfrog` and `hmc` are part of `mlfs.infer`. then open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
