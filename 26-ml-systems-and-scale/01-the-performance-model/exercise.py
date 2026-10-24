import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the performance model · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `matmul_flops`, `matmul_bytes`, `arithmetic_intensity`: the two numbers that decide an op's speed → `mlfs.systems` |
    | 2 | `roofline`, `ridge_point`, `bound`: what a machine can attain at a given intensity → `mlfs.systems` |
    | 3 | `benchmark`: timing an op honestly (warmup, device sync, best of n) |
    | 4 | `transformer_train_flops`, `gpt_block_ops`, `mfu`: the arithmetic-intensity calculator for a GPT block → `mlfs.systems` |
    | 5 | `profile_ops`: where a training step's time goes, op by op |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def matmul_flops(m, n, k):
    """stage 1: floating-point operations of an (m, k) @ (k, n) product: 2 m n k (a multiply and an add each)."""
    raise NotImplementedError("stage1: implement matmul_flops")


@app.function
def matmul_bytes(m, n, k, *, bytes_per_element=4):
    """stage 1: minimum bytes moved by (m, k) @ (k, n): read both inputs once, write the (m, n) output once."""
    raise NotImplementedError("stage1: implement matmul_bytes")


@app.function
def arithmetic_intensity(flops, bytes_moved):
    """stage 1: FLOPs per byte moved."""
    raise NotImplementedError("stage1: implement arithmetic_intensity")


@app.cell(hide_code=True)
def _():
    checks.run(1, matmul_flops=matmul_flops, matmul_bytes=matmul_bytes, arithmetic_intensity=arithmetic_intensity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def roofline(intensity, peak_flops, bandwidth):
    """stage 2: attainable FLOP/s = min(peak_flops, intensity * bandwidth); scalars or arrays (elementwise)."""
    raise NotImplementedError("stage2: implement roofline")


@app.function
def ridge_point(peak_flops, bandwidth):
    """stage 2: the intensity (FLOP/byte) where the bandwidth slope meets the compute roof."""
    raise NotImplementedError("stage2: implement ridge_point")


@app.function
def bound(intensity, peak_flops, bandwidth):
    """stage 2: 'memory' if intensity is below the ridge point, else 'compute'."""
    raise NotImplementedError("stage2: implement bound")


@app.cell(hide_code=True)
def _():
    checks.run(2, roofline=roofline, ridge_point=ridge_point, bound=bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def benchmark(fn, *, warmup=1, repeats=5, sync=None):
    """stage 3: time fn() repeats times after warmup calls; sync() before each clock read; {'best','median','times'}."""
    raise NotImplementedError("stage3: implement benchmark")


@app.cell(hide_code=True)
def _():
    checks.run(3, benchmark=benchmark)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def transformer_train_flops(n_params, n_tokens):
    """stage 4: approximate training FLOPs: 6 N D (2 N D forward, 4 N D backward)."""
    raise NotImplementedError("stage4: implement transformer_train_flops")


@app.function
def gpt_block_ops(batch, T, d_model, n_heads, *, d_ff=None, bytes_per_element=4):
    """stage 4: forward ops of one pre-LN GPT block: list of {'name','flops','bytes'} in execution order (spec table)."""
    raise NotImplementedError("stage4: implement gpt_block_ops")


@app.function
def mfu(flops_per_step, seconds_per_step, peak_flops):
    """stage 4: model FLOP utilization: achieved FLOP/s (flops_per_step / seconds_per_step) as a fraction of peak."""
    raise NotImplementedError("stage4: implement mfu")


@app.cell(hide_code=True)
def _():
    checks.run(4, transformer_train_flops=transformer_train_flops, gpt_block_ops=gpt_block_ops, mfu=mfu,
               arithmetic_intensity=arithmetic_intensity, bound=bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def profile_ops(step_fn, *, n_steps=3, warmup=1):
    """stage 5: torch.profiler over n_steps calls (after warmup unprofiled ones) -> [(op, self_cpu_us, calls)], slowest first."""
    raise NotImplementedError("stage5: implement profile_ops")


@app.cell(hide_code=True)
def _():
    checks.run(5, profile_ops=profile_ops)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. you'll measure your own
    machine's roofline, put every op of a GPT block on it, and profile a training step. from
    now on `from mlfs.systems import matmul_flops, arithmetic_intensity, roofline,
    transformer_train_flops` is your back-of-the-envelope kit; the capstone uses all of it.
    """)
    return


if __name__ == "__main__":
    app.run()
