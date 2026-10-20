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
    # data parallelism · build (reference solution)

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
    p = len(arrays)
    shape = np.shape(arrays[0])
    # each worker splits its (flattened) array into p chunks; buf[i][c] is worker i's copy of chunk c
    buf = [[c.copy() for c in np.array_split(np.asarray(a).ravel(), p)] for a in arrays]
    # reduce-scatter: at step s, worker i sends chunk (i - s) % p to worker i + 1, which adds it to its own
    for s in range(p - 1):
        sent = [buf[i][(i - s) % p] for i in range(p)]  # everyone sends at once (copies taken before any add)
        sent = [x.copy() for x in sent]
        for i in range(p):
            c = (i - s) % p
            buf[(i + 1) % p][c] = buf[(i + 1) % p][c] + sent[i]
    # now worker i holds the complete sum of chunk (i + 1) % p
    # all-gather: at step s, worker i sends its complete chunk (i + 1 - s) % p to worker i + 1, which overwrites
    for s in range(p - 1):
        sent = [(i, (i + 1 - s) % p, buf[i][(i + 1 - s) % p].copy()) for i in range(p)]
        for i, c, x in sent:
            buf[(i + 1) % p][c] = x
    return [np.concatenate(b).reshape(shape) for b in buf]


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
    return 2 * (world_size - 1) / world_size * n_bytes


@app.function
def allreduce_time(n_bytes, world_size, *, bandwidth, latency=0.0):
    """stage 2: ring all-reduce time: 2 (p - 1) messages of n_bytes / p, each latency + size / bandwidth."""
    return 2 * (world_size - 1) * (latency + n_bytes / world_size / bandwidth)


@app.function
def ddp_efficiency(t_compute, t_comm, *, overlap=0.0):
    """stage 2: t_compute / step time, step time = t_compute + the part of t_comm not overlapped: max(0, t_comm - overlap * t_compute)."""
    return t_compute / (t_compute + max(0.0, t_comm - overlap * t_compute))


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
    for p in model.parameters():
        p.grad = None
    n_total = sum(len(x) for x, _ in micro_batches)
    total = 0.0
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) * (len(x) / n_total)  # each micro-batch's mean, weighted by its share
        loss.backward()  # gradients add up in .grad
        total += loss.item()
    return total


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
    perm = np.random.default_rng(seed + epoch).permutation(n)
    total = -(-n // world_size) * world_size
    perm = np.resize(perm, total)  # repeats the permutation cyclically, even when n < total - n
    return perm[rank::world_size]


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
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    offset = 0
    for g in grads:
        g.copy_(flat[offset:offset + g.numel()].view_as(g))
        offset += g.numel()


@app.function
def ddp_train(model, X, y, loss_fn, *, rank, world_size, epochs=1, batch_size=8, lr=0.1, seed=0):
    """stage 5: broadcast rank 0's params, then per epoch: shard, local batches, backward, average_gradients, SGD; returns local losses."""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)  # every rank starts from the same weights
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    model.train()
    for epoch in range(epochs):
        idx = shard_indices(len(X), rank, world_size, epoch=epoch, seed=seed)
        for start in range(0, len(idx), batch_size):
            b = torch.as_tensor(idx[start:start + batch_size])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X[b]), y[b])
            loss.backward()
            average_gradients(model, world_size)
            opt.step()
            losses.append(loss.item())
    return losses


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
