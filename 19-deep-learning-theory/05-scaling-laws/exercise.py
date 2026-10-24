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
    # scaling laws · build

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
    raise NotImplementedError("stage1: implement power_law")


@app.function
def local_exponents(x, y, c=0.0):
    """stage 1: (len - 1,) slopes -d log(y - c) / d log x between consecutive points (x increasing)."""
    raise NotImplementedError("stage1: implement local_exponents")


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
    raise NotImplementedError("stage2: implement fit_power_law")


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
    raise NotImplementedError("stage3: implement compute_optimal")


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
    raise NotImplementedError("stage4: implement chinchilla_loss")


@app.function
def optimal_allocation(compute, *, A, B, alpha, beta):
    """stage 4: (N_opt, D_opt) minimizing A/N^alpha + B/D^beta subject to 6 N D = compute (closed form)."""
    raise NotImplementedError("stage4: implement optimal_allocation")


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
    raise NotImplementedError("stage5: implement non_embedding_params")


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
