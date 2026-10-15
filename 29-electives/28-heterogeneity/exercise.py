import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # heterogeneity · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `partition_dirichlet`, `label_distribution`, `heterogeneity`: tunable label skew, and a number for it |
    | 2 | `global_optimum`, `quadratic_fedavg`, `quadratic_fixed_point`: client drift, exactly, on quadratics |
    | 3 | `quadratic_scaffold`: control variates that remove the drift |
    | 4 | `fedprox_client_update`: local SGD with a proximal term, ready for your `fedavg` |
    | 5 | `fine_tune`, `per_client_accuracy`: personalization by local fine-tuning |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def partition_dirichlet(y, n_clients, alpha, *, seed=None, min_size=1, max_tries=100):
    """stage 1: per class, split its (shuffled) indices across clients in Dirichlet(alpha) proportions."""
    raise NotImplementedError("stage1: implement partition_dirichlet")


@app.function
def label_distribution(parts, y, n_classes):
    """stage 1: (n_clients, n_classes) array; row k is client k's label histogram, normalized to sum to 1."""
    raise NotImplementedError("stage1: implement label_distribution")


@app.function
def heterogeneity(parts, y, n_classes):
    """stage 1: sum_k (n_k / n) * TV(client k's label distribution, the global one), a float in [0, 1]."""
    raise NotImplementedError("stage1: implement heterogeneity")


@app.cell(hide_code=True)
def _():
    checks.run(1, partition_dirichlet=partition_dirichlet, label_distribution=label_distribution,
               heterogeneity=heterogeneity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def global_optimum(As, cs, weights=None):
    """stage 2: the minimizer of sum_k p_k * 0.5 (w - c_k)^T A_k (w - c_k)."""
    raise NotImplementedError("stage2: implement global_optimum")


@app.function
def quadratic_fedavg(As, cs, w0, *, rounds, local_steps, lr, mu=0.0, weights=None):
    """stage 2: FedAvg (or FedProx if mu > 0) with full participation and local GD; (rounds + 1, d) iterates."""
    raise NotImplementedError("stage2: implement quadratic_fedavg")


@app.function
def quadratic_fixed_point(As, cs, *, local_steps, lr, mu=0.0, weights=None):
    """stage 2: the point quadratic_fedavg converges to (when it converges), in closed form."""
    raise NotImplementedError("stage2: implement quadratic_fixed_point")


@app.cell(hide_code=True)
def _():
    checks.run(2, global_optimum=global_optimum, quadratic_fedavg=quadratic_fedavg,
               quadratic_fixed_point=quadratic_fixed_point)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def quadratic_scaffold(As, cs, w0, *, rounds, local_steps, lr, weights=None):
    """stage 3: SCAFFOLD (full participation, option II control variates, server lr 1); (rounds + 1, d) iterates."""
    raise NotImplementedError("stage3: implement quadratic_scaffold")


@app.cell(hide_code=True)
def _():
    checks.run(3, quadratic_scaffold=quadratic_scaffold, quadratic_fedavg=quadratic_fedavg,
               global_optimum=global_optimum)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fedprox_client_update(model, X, y, *, mu=0.01, epochs=1, batch_size=10, lr=0.1, loss_fn=None, seed=None):
    """stage 4: client_update on loss + (mu / 2) ||w - w_global||^2, w_global = the weights at entry; (weights, n_steps)."""
    raise NotImplementedError("stage4: implement fedprox_client_update")


@app.cell(hide_code=True)
def _():
    checks.run(4, fedprox_client_update=fedprox_client_update)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def fine_tune(model, X, y, *, steps=10, lr=0.05, batch_size=10, head_only=False, loss_fn=None, seed=None):
    """stage 5: a fine-tuned deep copy of model after `steps` SGD steps on (X, y); the original is untouched."""
    raise NotImplementedError("stage5: implement fine_tune")


@app.function
def per_client_accuracy(models, client_data):
    """stage 5: accuracy of model k (or of one shared model) on client k's (X_k, y_k); a NumPy array."""
    raise NotImplementedError("stage5: implement per_client_accuracy")


@app.cell(hide_code=True)
def _():
    checks.run(5, fine_tune=fine_tune, per_client_accuracy=per_client_accuracy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`). it runs your topic 27
    `fedavg` on increasingly skewed clients, plugs your `fedprox_client_update` into it, and
    measures what personalization buys each client.
    """)
    return


if __name__ == "__main__":
    app.run()
