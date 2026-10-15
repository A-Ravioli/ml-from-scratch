import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.optim import (  # yours, from this chapter
        SGD,
        Adagrad,
        Adam,
        AdamW,
        Param,
        RMSProp,
        minibatch_indices,
        minimize,
        perturbed_gradient_descent,
        rastrigin,
        rastrigin_grad,
        rosenbrock,
        rosenbrock_grad,
    )
    from mlfs.utils import as_rng  # yours, from ch 00

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    MAX_ITERS = 4_000 if SMALL else 20_000
    EPOCHS = 20 if SMALL else 25
    # adaptive learning rates tuned per config: the small run takes ~20x fewer steps
    LR_ADAM, LR_RMS = (0.05, 0.03) if SMALL else (0.01, 0.01)
    SEEDS = 2 if SMALL else 5
    N_STARTS = 8 if SMALL else 40


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 06 · capstone: an optimizer benchmark suite

    read [`brief.md`](brief.md) first. you'll build the harness that runs every optimizer in
    your `mlfs.optim` on the same problems, counts what each one really costs, and turns the
    runs into a convergence report; then use it on standard test functions and on a
    logistic regression with real data.

    | stage | you build |
    |---|---|
    | 1 | `logistic_objective`: the regularized logistic loss, its gradient and Hessian |
    | 2 | `run_deterministic`: run any `minimize` method with counted calls, recording cost and progress |
    | 3 | `run_stochastic`: minibatch training with any `Optimizer`, recording progress per epoch |
    | 4 | `cost_to_reach`, `rank_methods`: when did each run get close enough, and who got there first |

    when they're green, scroll down: the experiments run on your functions, and the
    write-up prompts are at the end.
    """)
    return


@app.function
def logistic_objective(X, y, alpha):
    """stage 1: the ridge-regularized logistic regression objective, as three closures.

    X (n, d) features (include a column of ones yourself if you want an intercept), y (n,)
    labels in {0, 1}, alpha >= 0. returns (f, grad, hess) with, for w (d,) and z = X w,
        f(w)    = mean(log(1 + e^z) - y z) + alpha/2 ||w||^2       (a float; use np.logaddexp)
        grad(w) = X^T (sigmoid(z) - y) / n + alpha w                (d,)
        hess(w) = X^T diag(s (1 - s)) X / n + alpha I, s = sigmoid(z)   (d, d)
    all three must be finite for any w, even when |z| is huge (compute the sigmoid as
    (1 + tanh(z/2)) / 2, which never overflows).
    """
    raise NotImplementedError("stage1: implement logistic_objective")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", logistic_objective=logistic_objective)
    return


@app.function
def run_deterministic(method, f, grad, x0, *, hess=None, max_iters=1000, tol=1e-06, **options):
    """stage 2: run mlfs.optim.minimize(f, grad, x0, method=method, ...) and measure it.

    wrap f, grad (and hess, if given) in counters, pass a callback that records the work done
    so far after every step, time the run with time.perf_counter, and forward hess,
    max_iters, tol and **options to minimize. work is measured in gradient-equivalents:
        work = (calls of f) + (calls of grad) + d * (calls of hess),  d = x0.size
    (for most ML losses one f or one grad costs about the same, a pass over the data, and a
    Hessian about d of them). returns a dict with keys
        "method" (str), "x", "fun", "converged", "n_iters" (from the OptimResult),
        "history"  (list: f at x0 and after every step, the OptimResult's history),
        "costs"    (list, same length as history: 0.0 for x0, then the work done when each
                    later iterate was reached),
        "n_f", "n_grad", "n_hess" (total calls, ints), "seconds" (float).
    """
    raise NotImplementedError("stage2: implement run_deterministic")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", run_deterministic=run_deterministic, logistic_objective=logistic_objective)
    return


@app.function
def run_stochastic(make_optimizer, loss_grad, full_loss, w0, n, *, epochs=10, batch_size=32, rng=None):
    """stage 3: minibatch training with any Optimizer from mlfs.optim.

    rng = as_rng(rng). wrap a copy of w0 in a Param p and build opt = make_optimizer([p]).
    each epoch: for idx in minibatch_indices(n, batch_size, rng=rng): opt.zero_grad();
    p.grad = loss_grad(p.data, idx) (the gradient of the minibatch loss); opt.step().
    after each epoch record full_loss(p.data); stop early if it isn't finite. returns a dict
        "method" (the optimizer's class name), "x" (a copy of the final weights),
        "fun" (the last recorded loss), "history" (full loss at w0, then after each epoch),
        "costs" (gradient-equivalents: 0.0, 1.0, 2.0, ... one full pass over the data per epoch),
        "n_steps" (optimizer steps taken), "seconds".
    """
    raise NotImplementedError("stage3: implement run_stochastic")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", run_stochastic=run_stochastic, logistic_objective=logistic_objective)
    return


@app.function
def cost_to_reach(record, f_star, rel_tol):
    """stage 4: the first cost at which a run got within rel_tol of the optimum.

    with f0 = record["history"][0], return the smallest record["costs"][k] such that
    history[k] - f_star <= rel_tol * (f0 - f_star); np.inf if no iterate qualifies. a float.
    """
    raise NotImplementedError("stage4: implement cost_to_reach")


@app.function
def rank_methods(records, f_star, rel_tol):
    """stage 4: [(record["method"], cost_to_reach(record, f_star, rel_tol)), ...] sorted by
    cost, cheapest first (np.inf last; ties keep the input order)."""
    raise NotImplementedError("stage4: implement rank_methods")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", cost_to_reach=cost_to_reach, rank_methods=rank_methods)
    return


@app.function
def report(records, f_star, rel_tols=(1e-2, 1e-4, 1e-8)):
    """(given) a markdown table: per run, the cost to reach each relative tolerance, plus
    iterations, wall-clock time and the final suboptimality."""
    head = "| method | " + " | ".join(f"cost to {t:g}" for t in rel_tols) + " | iterations / steps | seconds | final f − f* |"
    rows = [head, "|" + "---|" * (len(rel_tols) + 4)]
    for r in records:
        cells = [f"{cost_to_reach(r, f_star, t):.0f}" for t in rel_tols]
        its = r.get("n_iters", r.get("n_steps"))
        rows.append(f"| {r['method']} | " + " | ".join(cells) + f" | {its} | {r['seconds']:.2f} | {r['fun'] - f_star:.2e} |")
    return "\n".join(rows)


@app.function
def plot_records(records, f_star, ax, title, label_key="method"):
    """(given) suboptimality against cost, on log axes (minibatch runs dashed)."""
    for r in records:
        gap = np.maximum(np.asarray(r["history"], dtype=float) - f_star, 1e-16)
        ax.loglog(np.maximum(r["costs"], 1.0), gap, label=r[label_key], lw=1.2,
                  ls="--" if "n_steps" in r else "-")  # dashed: minibatch runs
    ax.set(xlabel="cost (gradient-equivalents)", ylabel="f − f*", title=title)
    ax.legend(fontsize=7)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A (given): the deterministic methods on test functions

    every deterministic method in `mlfs.optim`, through your `run_deterministic`, on three
    smooth problems: an ill-conditioned quadratic ($\kappa = 1{,}000$, 50-D), Rosenbrock in 2-D,
    and Rosenbrock in 10-D (from $(-1.2, 1, -1.2, 1, \dots)$). the step sizes for the
    first-order methods are fixed by hand (the quadratic's $L$ is known; for Rosenbrock a
    step that's stable along the whole path); BFGS, L-BFGS and Newton need none.
    """)
    return


@app.cell
def _():
    _rng = np.random.default_rng(0)
    _Q, _ = np.linalg.qr(_rng.normal(size=(50, 50)))
    _eig = np.geomspace(1.0, 1000.0, 50)
    _A = _Q @ np.diag(_eig) @ _Q.T
    _b = _rng.normal(size=50)
    _x_star = np.linalg.solve(_A, _b)
    _quad = (lambda x: float(0.5 * x @ _A @ x - _b @ x), lambda x: _A @ x - _b, lambda x: _A,
             np.zeros(50), float(0.5 * _x_star @ _A @ _x_star - _b @ _x_star), 1e-3)
    _problems = {
        "quadratic, κ = 1,000": _quad,
        "Rosenbrock 2-D": (rosenbrock, rosenbrock_grad, None, np.array([-1.2, 1.0]), 0.0, 2e-4),
        "Rosenbrock 10-D": (rosenbrock, rosenbrock_grad, None, np.tile([-1.2, 1.0], 5), 0.0, 2e-4),
    }

    def _rosen_hess(x):
        n = x.size
        H = np.zeros((n, n))
        for i in range(n - 1):
            H[i, i] += 2 - 400 * (x[i + 1] - 3 * x[i] ** 2)
            H[i, i + 1] += -400 * x[i]
            H[i + 1, i] += -400 * x[i]
            H[i + 1, i + 1] += 200
        return H

    _figA, _axesA = plt.subplots(1, 3, figsize=(14, 4))
    _tables = []
    for _ax, (_name, (_f, _g, _h, _x0, _fs, _lr)) in zip(_axesA, _problems.items()):
        _h = _h if _h is not None else _rosen_hess
        _runs = [
            run_deterministic("gd", _f, _g, _x0, lr=1.0, line_search=True, max_iters=MAX_ITERS, tol=1e-8),
            run_deterministic("heavy_ball", _f, _g, _x0, lr=_lr, momentum=0.9, max_iters=MAX_ITERS, tol=1e-8),
            run_deterministic("nesterov", _f, _g, _x0, lr=_lr, max_iters=MAX_ITERS, tol=1e-8),
            run_deterministic("newton", _f, _g, _x0, hess=_h, max_iters=500, tol=1e-8),
            run_deterministic("bfgs", _f, _g, _x0, max_iters=MAX_ITERS, tol=1e-8),
            run_deterministic("lbfgs", _f, _g, _x0, max_iters=MAX_ITERS, tol=1e-8),
        ]
        plot_records(_runs, _fs, _ax, _name)
        _tables.append(f"**{_name}**\n\n" + report(_runs, _fs))
    _figA.tight_layout()
    mo.vstack([_figA, mo.md("\n\n".join(_tables))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B (given): a nonconvex test function

    5-D Rastrigin from random starts in $[-3, 3]^5$ (small config: 8 starts; full: 40):
    L-BFGS, gradient descent with a line search, and perturbed gradient descent with large
    kicks. a run "finds" the global minimum if it ends within 0.1 of the origin.
    """)
    return


@app.cell
def _():
    _starts = np.random.default_rng(1).uniform(-3, 3, size=(N_STARTS, 5))
    _out = {"lbfgs": [], "gd": [], "perturbed gd (radius 0.8)": []}
    for _i, _s in enumerate(_starts):
        _out["lbfgs"].append(run_deterministic("lbfgs", rastrigin, rastrigin_grad, _s, max_iters=500)["x"])
        _out["gd"].append(run_deterministic("gd", rastrigin, rastrigin_grad, _s, lr=1.0, line_search=True,
                                            max_iters=500)["x"])
        _out["perturbed gd (radius 0.8)"].append(perturbed_gradient_descent(
            rastrigin, rastrigin_grad, _s, lr=0.002, noise_radius=0.8, grad_threshold=0.05, max_iters=1500, rng=_i).x)
    _rows = []
    for _k, _xs in _out.items():
        _xs = np.array(_xs)
        _vals = np.array([rastrigin(x) for x in _xs])
        _rows.append(f"| {_k} | {np.mean(np.linalg.norm(_xs, axis=1) < 0.1):.0%} | {np.median(_vals):.2f} | {_vals.min():.2f} |")
    mo.md("| method | found the global minimum | median final f | best final f |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C (given): logistic regression on California housing

    predict whether a district's median house value is above the median, from its 8 raw
    features plus an intercept ($n = 20{,}433$; small config: the 1,000-row synthetic stand-in),
    with a ridge penalty $\alpha = 10^{-3}$. two versions of the same problem: **raw** features
    (scales from about 1 to about 2,000) and **standardized** ones (each feature with mean 0
    and standard deviation 1). $f^*$ comes from a long Newton run. the deterministic methods
    use your `run_deterministic`; the minibatch methods (batch 64, several seeds) your
    `run_stochastic`, with gradients of the minibatch loss. AdamW gets the ridge penalty as
    decoupled weight decay instead of in its gradient.
    """)
    return


@app.cell
def _():
    _X, _y_val, _names = datasets.california_housing()
    _y = (_y_val > np.median(_y_val)).astype(float)
    _alpha = 1e-3
    _versions = {"raw": _X, "standardized": (_X - _X.mean(0)) / _X.std(0)}
    _figC, _axesC = plt.subplots(1, 2, figsize=(13, 4.5))
    _texts = []
    for _ax, (_vname, _Xv) in zip(_axesC, _versions.items()):
        _Xb = np.hstack([np.ones((len(_Xv), 1)), _Xv])
        _f, _g, _h = logistic_objective(_Xb, _y, _alpha)
        _d = _Xb.shape[1]
        _ref = minimize(_f, _g, np.zeros(_d), method="newton", hess=_h, max_iters=200, tol=1e-10)
        _fs = _ref.fun
        _H = _h(_ref.x)
        _ev = np.linalg.eigvalsh(_H)
        _L = float(_ev[-1])
        _runs = [
            run_deterministic("gd", _f, _g, np.zeros(_d), lr=1.0, line_search=True, max_iters=2_000),
            run_deterministic("nesterov", _f, _g, np.zeros(_d), lr=1.0 / _L, max_iters=2_000),
            run_deterministic("newton", _f, _g, np.zeros(_d), hess=_h, max_iters=50),
            run_deterministic("lbfgs", _f, _g, np.zeros(_d), max_iters=2_000),
        ]

        def _lg(w, idx, Xb=_Xb, alpha=_alpha):
            s = 0.5 * (1.0 + np.tanh(0.5 * (Xb[idx] @ w)))
            return Xb[idx].T @ (s - _y[idx]) / len(idx) + alpha * w

        def _lg_noreg(w, idx, Xb=_Xb):
            s = 0.5 * (1.0 + np.tanh(0.5 * (Xb[idx] @ w)))
            return Xb[idx].T @ (s - _y[idx]) / len(idx)

        _makers = {"SGD + momentum": (lambda ps: SGD(ps, lr=0.5 / _L if _vname == "raw" else 0.1, momentum=0.9), _lg),
                   "Adagrad": (lambda ps: Adagrad(ps, lr=0.5), _lg),
                   "RMSProp": (lambda ps: RMSProp(ps, lr=LR_RMS), _lg),
                   "Adam": (lambda ps: Adam(ps, lr=LR_ADAM), _lg),
                   "AdamW": (lambda ps: AdamW(ps, lr=LR_ADAM, weight_decay=_alpha), _lg_noreg)}
        _stoch = []
        for _mname, (_make, _grad_fn) in _makers.items():
            _seed_runs = [run_stochastic(_make, _grad_fn, _f, np.zeros(_d), len(_y), epochs=EPOCHS, batch_size=64,
                                         rng=_s) for _s in range(SEEDS)]
            _fin = np.array([r["fun"] for r in _seed_runs]) - _fs
            _best = _seed_runs[int(np.argmin(_fin))]
            _best["method"] = f"{_mname} (best of {SEEDS})"
            _stoch.append(_best)
            _texts.append(f"| {_vname} | {_mname} | {np.median(_fin):.2e} | {_fin.min():.2e} – {_fin.max():.2e} |")
        plot_records(_runs + _stoch, _fs, _ax, f"{_vname} features (κ(H*) ≈ {_ev[-1] / _ev[0]:.1e})")
        _texts.insert(0, f"**{_vname} features** — condition number of the Hessian at the optimum "
                         f"{_ev[-1] / _ev[0]:.2e}\n\n" + report(_runs, _fs) + "\n")
    _figC.tight_layout()
    _det = [t for t in _texts if t.startswith("**")]
    _sto = [t for t in _texts if not t.startswith("**")]
    mo.vstack([_figC, mo.md("\n\n".join(_det) + "\n\n**minibatch methods** (final f − f* over seeds)\n\n"
                            "| features | optimizer | median | range |\n|---|---|---|---|\n" + "\n".join(_sto))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your convergence report (grade it with [`rubric.md`](rubric.md)).
    answer at least:

    1. **the ranking, and what it depends on.** from experiment A, rank the methods by cost
       to reach $10^{-8}$ on each problem, and separately by wall-clock time. where do the
       rankings disagree, and why? connect each method's behaviour to the theory: the
       $\kappa$ versus $\sqrt\kappa$ rates (topics 02–03), quadratic and superlinear
       convergence (topic 05).
    2. **nonconvexity.** from experiment B, which methods find the global minimum of
       Rastrigin, how often, and at what cost? is perturbed gradient descent an optimizer
       here, or a random search?
    3. **conditioning in practice.** from experiment C, compare the raw and standardized
       versions. which methods care, and by how much? explain with the condition numbers,
       and explain why Newton and (partly) the adaptive methods don't care. what does that
       say about feature scaling?
    4. **stochastic versus deterministic.** at equal cost (passes over the data), which
       minibatch method is best, and how do they compare with L-BFGS? report the spread over
       seeds, not one run. why does AdamW end at a different loss?
    5. **recommendations.** which optimizer would you use for (a) this logistic regression,
       (b) the same model on $10^8$ examples, (c) a small neural network? justify each with a
       number from your benchmark.
    """)
    return


if __name__ == "__main__":
    app.run()
