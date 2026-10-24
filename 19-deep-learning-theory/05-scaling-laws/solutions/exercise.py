import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import scipy.optimize
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # scaling laws · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours. `power_law`, `fit_power_law` and `compute_optimal` join **`mlfs.dltheory`**.

    | stage | you build |
    |---|---|
    | 1 | `power_law`, `local_exponents`: the model, and the slope between neighbouring points → `mlfs.dltheory` |
    | 2 | `fit_power_law`: $y = a x^{-b} + c$ by least squares in log space → `mlfs.dltheory` |
    | 3 | `compute_optimal`: the Chinchilla rule, $C = 6ND$ and 20 tokens per parameter → `mlfs.dltheory` |
    | 4 | `chinchilla_loss`, `optimal_allocation`: the parametric loss $L(N, D)$ and its compute-optimal split |
    | 5 | `non_embedding_params`: the parameter count scaling laws use |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def power_law(x, a, b, c=0.0):
    """stage 1: a * x**(-b) + c, elementwise (floats or arrays)."""
    return a * np.asarray(x, dtype=float) ** (-b) + c


@app.function
def local_exponents(x, y, c=0.0):
    """stage 1: (len - 1,) slopes -d log(y - c) / d log x between consecutive points (x increasing)."""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float) - c)
    return -np.diff(ly) / np.diff(lx)


@app.cell(hide_code=True)
def _():
    checks.run(1, power_law=power_law, local_exponents=local_exponents)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def fit_power_law(x, y, *, offset=True):
    """stage 2: (a, b, c) minimizing sum (log y - log(a x^-b + c))^2 with a > 0 and 0 <= c < min(y); c = 0 if not offset."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx, ly = np.log(x), np.log(y)

    def loglinear(c):
        slope, intercept = np.polyfit(lx, np.log(y - c), 1)
        return float(np.exp(intercept)), float(-slope)

    if not offset:
        a, b = loglinear(0.0)
        return a, b, 0.0

    def sse(params):
        a, b, c = params
        pred = a * x ** (-b) + c
        if a <= 0 or np.any(pred <= 0):
            return np.inf
        return float(np.sum((ly - np.log(pred)) ** 2))

    # profile over c: for a fixed offset the fit is linear in log space; then polish all three
    cs = np.min(y) * (1 - np.geomspace(1.0, 1e-6, 200))
    best = min(((*loglinear(c), c) for c in cs), key=sse)

    def resid(p):
        return ly - np.log(np.exp(p[0]) * x ** (-p[1]) + p[2])

    hi = np.min(y) * (1 - 1e-9)
    p0 = np.array([np.log(best[0]), best[1], min(max(best[2], 0.0), hi * (1 - 1e-9))])
    res = scipy.optimize.least_squares(resid, p0, bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, hi]),
                                       xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    cand = (float(np.exp(res.x[0])), float(res.x[1]), float(res.x[2]))
    return cand if sse(cand) <= sse(best) else tuple(float(v) for v in best)


@app.cell(hide_code=True)
def _():
    checks.run(2, fit_power_law=fit_power_law, power_law=power_law)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def compute_optimal(compute, *, tokens_per_param=20.0):
    """stage 3: (n_params, n_tokens) with compute = 6 N D and D = tokens_per_param * N."""
    n = np.sqrt(np.asarray(compute, dtype=float) / (6.0 * tokens_per_param))
    return n, tokens_per_param * n


@app.cell(hide_code=True)
def _():
    checks.run(3, compute_optimal=compute_optimal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def chinchilla_loss(n_params, n_tokens, *, E, A, B, alpha, beta):
    """stage 4: E + A / N**alpha + B / D**beta, elementwise."""
    N = np.asarray(n_params, dtype=float)
    D = np.asarray(n_tokens, dtype=float)
    return E + A / N ** alpha + B / D ** beta


@app.function
def optimal_allocation(compute, *, A, B, alpha, beta):
    """stage 4: (N_opt, D_opt) minimizing A/N^alpha + B/D^beta subject to 6 N D = compute (closed form)."""
    C = np.asarray(compute, dtype=float)
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    n = G * (C / 6.0) ** (beta / (alpha + beta))
    d = (C / 6.0) ** (alpha / (alpha + beta)) / G
    return n, d


@app.cell(hide_code=True)
def _():
    checks.run(4, chinchilla_loss=chinchilla_loss, optimal_allocation=optimal_allocation)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def non_embedding_params(model):
    """stage 5: number of parameters excluding model.tok_emb.weight and model.pos_emb.weight (if present); tied tensors counted once."""
    skip = set()
    for name in ("tok_emb", "pos_emb"):
        module = getattr(model, name, None)
        if module is not None and getattr(module, "weight", None) is not None:
            skip.add(id(module.weight))
    seen, total = set(), 0
    for p in model.parameters():  # parameters() yields each (possibly tied) tensor once
        if id(p) in seen or id(p) in skip:
            continue
        seen.add(id(p))
        total += p.numel()
    return int(total)


@app.cell(hide_code=True)
def _():
    checks.run(5, non_embedding_params=non_embedding_params)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.dltheory import fit_power_law, power_law, compute_optimal` fits and extrapolates
    scaling curves for any experiment in the course.
    """)
    return


if __name__ == "__main__":
    app.run()
