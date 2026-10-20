import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # model parallelism & memory · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `training_memory`: bytes per device for weights, gradients and optimizer state, with ZeRO sharding → `mlfs.systems` |
    | 2 | `saved_tensor_bytes`, `checkpoint`: measuring activation memory, and trading it for recomputation |
    | 3 | `split_mlp`, `tp_mlp_forward`: a tensor-parallel MLP (column- then row-parallel) |
    | 4 | `gpipe_schedule`, `bubble_fraction`: pipeline parallelism and its idle time |
    | 5 | `ZeroAdamW`: AdamW with its state sharded across ranks (ZeRO-1) → `mlfs.systems` |

    stage 5's multi-process check starts two gloo processes that load your notebook from disk;
    in the notebook it's skipped, so run `pytest -k stage5` in a terminal.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def training_memory(n_params, *, bytes_per_param=2, optimizer="adamw", zero_stage=0, world_size=1):
    """stage 1: per-device bytes {'params','grads','optimizer','total'}; fp32 master copy if bytes_per_param < 4; ZeRO shards."""
    raise NotImplementedError("stage1: implement training_memory")


@app.cell(hide_code=True)
def _():
    checks.run(1, training_memory=training_memory)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def saved_tensor_bytes(fn, *args, exclude=()):
    """stage 2: (fn(*args), bytes of the distinct storages autograd saves for backward during the call, minus `exclude`'s)."""
    raise NotImplementedError("stage2: implement saved_tensor_bytes")


@app.function
def checkpoint(fn, *args):
    """stage 2: fn(*args) (one tensor output) that saves only args and recomputes fn during backward, same RNG."""

    raise NotImplementedError("stage2: implement checkpoint")


@app.cell(hide_code=True)
def _():
    checks.run(2, saved_tensor_bytes=saved_tensor_bytes, checkpoint=checkpoint)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def split_mlp(fc1, fc2, world_size):
    """stage 3: per-rank (W1 rows, b1 slice, W2 columns) for hidden units [r h/p, (r+1) h/p); ValueError if p doesn't divide h."""
    raise NotImplementedError("stage3: implement split_mlp")


@app.function
def tp_mlp_forward(x, shards, b2):
    """stage 3: (out, partials): rank r's partial gelu(x W1_r^T + b1_r) W2_r^T; out = their sum (the all-reduce) + b2."""
    raise NotImplementedError("stage3: implement tp_mlp_forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, split_mlp=split_mlp, tp_mlp_forward=tp_mlp_forward)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def gpipe_schedule(n_stages, n_micro):
    """stage 4: grid[t][s] = ('F', m), ('B', m) or None: all forwards as a wavefront, then all backwards in reverse."""
    raise NotImplementedError("stage4: implement gpipe_schedule")


@app.function
def bubble_fraction(n_stages, n_micro):
    """stage 4: idle fraction of a GPipe schedule: (p - 1) / (m + p - 1)."""
    raise NotImplementedError("stage4: implement bubble_fraction")


@app.cell(hide_code=True)
def _():
    checks.run(4, gpipe_schedule=gpipe_schedule, bubble_fraction=bubble_fraction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class ZeroAdamW:
    """ZeRO-1 AdamW: each rank keeps AdamW state for its 1/world_size shard of the flattened parameters."""

    def __init__(self, params, *, rank, world_size, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        """stage 5: flatten sizes; this rank's shard = [rank S, (rank+1) S) with S = ceil(N / world_size); zero state."""
        raise NotImplementedError("stage5: implement ZeroAdamW.__init__")

    @torch.no_grad()
    def step(self):
        """stage 5: average flat grads over ranks (all_reduce), AdamW on this rank's shard, all_gather the new params."""
        raise NotImplementedError("stage5: implement ZeroAdamW.step")

    def zero_grad(self):
        """stage 5: set every parameter's .grad to None."""
        raise NotImplementedError("stage5: implement ZeroAdamW.zero_grad")


@app.cell(hide_code=True)
def _():
    checks.run(5, ZeroAdamW=ZeroAdamW)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green (including `pytest -k stage5` in a terminal), open the lab:
    `marimo edit lab.py`. from now on `from mlfs.systems import training_memory, ZeroAdamW`
    gives you the memory model and a sharded optimizer; chapter 27 uses the memory model to
    size its Llama, and the capstone uses `saved_tensor_bytes`'s idea to explain its memory.
    """)
    return


if __name__ == "__main__":
    app.run()
