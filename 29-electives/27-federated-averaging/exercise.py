import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # federated averaging · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `get_weights`, `set_weights`, `fedavg_aggregate`: models as vectors, and the server's weighted average |
    | 2 | `client_update`: local SGD on one client's data |
    | 3 | `partition_iid`, `partition_shards`: splitting a dataset across clients |
    | 4 | `fedavg`: the simulator, with byte-level communication accounting |
    | 5 | `stochastic_quantize`: unbiased low-bit compression of an update |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def get_weights(model):
    """stage 1: every parameter of `model`, flattened in model.parameters() order into one detached copy."""
    raise NotImplementedError("stage1: implement get_weights")


@app.function
def set_weights(model, weights):
    """stage 1: copy the flat vector `weights` into model's parameters in place (ValueError on a size mismatch)."""
    raise NotImplementedError("stage1: implement set_weights")


@app.function
def fedavg_aggregate(client_weights, num_samples):
    """stage 1: the average of the client vectors, each weighted by its number of samples."""
    raise NotImplementedError("stage1: implement fedavg_aggregate")


@app.cell(hide_code=True)
def _():
    checks.run(1, get_weights=get_weights, set_weights=set_weights, fedavg_aggregate=fedavg_aggregate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def client_update(model, X, y, *, epochs=1, batch_size=10, lr=0.1, loss_fn=None, seed=None):
    """stage 2: `epochs` passes of minibatch SGD on (X, y), in place; returns (get_weights(model), n_steps)."""
    raise NotImplementedError("stage2: implement client_update")


@app.cell(hide_code=True)
def _():
    checks.run(2, client_update=client_update, get_weights=get_weights)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def partition_iid(n, n_clients, *, seed=None):
    """stage 3: a random split of range(n) into n_clients index arrays whose sizes differ by at most 1."""
    raise NotImplementedError("stage3: implement partition_iid")


@app.function
def partition_shards(y, n_clients, shards_per_client=2, *, seed=None):
    """stage 3: McMahan et al.'s pathological split: sort by label, cut into shards, deal shards_per_client to each."""
    raise NotImplementedError("stage3: implement partition_shards")


@app.cell(hide_code=True)
def _():
    checks.run(3, partition_iid=partition_iid, partition_shards=partition_shards)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fedavg(model, client_data, *, rounds, client_fraction=0.1, local_epochs=1, batch_size=10, lr=0.1,
           local_update=None, compress=None, eval_fn=None, eval_every=1, seed=None):
    """stage 4: FedAvg on client_data = [(X_k, y_k), ...], training `model` in place; returns a history dict."""
    raise NotImplementedError("stage4: implement fedavg")


@app.cell(hide_code=True)
def _():
    checks.run(4, fedavg=fedavg, get_weights=get_weights, set_weights=set_weights, client_update=client_update)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def stochastic_quantize(v, bits, *, rng=None):
    """stage 5: unbiased stochastic rounding of v onto 2**bits evenly spaced levels in [min v, max v]; (v_hat, n_bytes)."""
    raise NotImplementedError("stage5: implement stochastic_quantize")


@app.cell(hide_code=True)
def _():
    checks.run(5, stochastic_quantize=stochastic_quantize)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, you have a federated-learning simulator. open the lab
    (`marimo edit lab.py`) to find out how much local work buys you in communication rounds.
    topics 28 and 29 plug new client updates and new aggregators into your `fedavg`.
    """)
    return


if __name__ == "__main__":
    app.run()
