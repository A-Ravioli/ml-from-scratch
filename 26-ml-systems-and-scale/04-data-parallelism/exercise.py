import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.distributed as dist

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # data parallelism · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `ring_allreduce`: the bandwidth-optimal all-reduce, simulated in NumPy → `mlfs.systems` |
    | 2 | `allreduce_bytes`, `allreduce_time`, `ddp_efficiency`: what communication costs → `mlfs.systems` |
    | 3 | `accumulate_gradients`: a big batch's gradient from small micro-batches |
    | 4 | `shard_indices`: every rank its own disjoint slice of the data, every epoch reshuffled |
    | 5 | `average_gradients`, `ddp_train`: data-parallel training across processes with `torch.distributed` |

    stage 5's checks start two real processes that talk over `torch.distributed`'s gloo backend
    (CPU, no GPU needed). they load your notebook from disk, so save it first; in the notebook
    they're skipped and you run them from a terminal: `pytest -k stage5`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def ring_allreduce(arrays):
    """stage 1: p equal-shape arrays -> p arrays all equal to their sum, by ring reduce-scatter then all-gather."""
    raise NotImplementedError("stage1: implement ring_allreduce")


@app.cell(hide_code=True)
def _():
    checks.run(1, ring_allreduce=ring_allreduce)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def allreduce_bytes(n_bytes, world_size):
    """stage 2: bytes each worker sends in a ring all-reduce of n_bytes: 2 (p - 1) / p * n_bytes."""
    raise NotImplementedError("stage2: implement allreduce_bytes")


@app.function
def allreduce_time(n_bytes, world_size, *, bandwidth, latency=0.0):
    """stage 2: ring all-reduce time: 2 (p - 1) messages of n_bytes / p, each latency + size / bandwidth."""
    raise NotImplementedError("stage2: implement allreduce_time")


@app.function
def ddp_efficiency(t_compute, t_comm, *, overlap=0.0):
    """stage 2: t_compute / step time, step time = t_compute + the part of t_comm not overlapped: max(0, t_comm - overlap * t_compute)."""
    raise NotImplementedError("stage2: implement ddp_efficiency")


@app.cell(hide_code=True)
def _():
    checks.run(2, allreduce_bytes=allreduce_bytes, allreduce_time=allreduce_time, ddp_efficiency=ddp_efficiency)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def accumulate_gradients(model, loss_fn, micro_batches):
    """stage 3: .grad = gradient of the mean loss over all examples of the micro-batches [(x, y), ...]; returns that loss."""
    raise NotImplementedError("stage3: implement accumulate_gradients")


@app.cell(hide_code=True)
def _():
    checks.run(3, accumulate_gradients=accumulate_gradients)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def shard_indices(n, rank, world_size, *, epoch=0, seed=0):
    """stage 4: rank's indices: permutation of n by seed + epoch, tiled to a multiple of world_size, every world_size-th from rank."""
    raise NotImplementedError("stage4: implement shard_indices")


@app.cell(hide_code=True)
def _():
    checks.run(4, shard_indices=shard_indices)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def average_gradients(model, world_size):
    """stage 5: average every .grad across ranks with ONE dist.all_reduce on a flat buffer of all gradients."""
    raise NotImplementedError("stage5: implement average_gradients")


@app.function
def ddp_train(model, X, y, loss_fn, *, rank, world_size, epochs=1, batch_size=8, lr=0.1, seed=0):
    """stage 5: broadcast rank 0's params, then per epoch: shard, local batches, backward, average_gradients, SGD; returns local losses."""
    raise NotImplementedError("stage5: implement ddp_train")


@app.cell(hide_code=True)
def _():
    checks.run(5, average_gradients=average_gradients, ddp_train=ddp_train, shard_indices=shard_indices)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green (including `pytest -k stage5` in a terminal), open the lab:
    `marimo edit lab.py`. it times real all-reduces between processes and measures how well
    your `ddp_train` scales. from now on `from mlfs.systems import ring_allreduce,
    allreduce_bytes` gives you the communication model that topic 05 extends to sharded
    optimizers and model parallelism.
    """)
    return


if __name__ == "__main__":
    app.run()
