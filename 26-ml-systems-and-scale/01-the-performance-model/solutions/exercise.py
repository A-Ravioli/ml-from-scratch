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
    # the performance model · build (reference solution)

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
    return 2 * m * n * k


@app.function
def matmul_bytes(m, n, k, *, bytes_per_element=4):
    """stage 1: minimum bytes moved by (m, k) @ (k, n): read both inputs once, write the (m, n) output once."""
    return (m * k + k * n + m * n) * bytes_per_element


@app.function
def arithmetic_intensity(flops, bytes_moved):
    """stage 1: FLOPs per byte moved."""
    return flops / bytes_moved


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
    out = np.minimum(peak_flops, np.asarray(intensity, dtype=float) * bandwidth)
    return float(out) if out.ndim == 0 else out


@app.function
def ridge_point(peak_flops, bandwidth):
    """stage 2: the intensity (FLOP/byte) where the bandwidth slope meets the compute roof."""
    return peak_flops / bandwidth


@app.function
def bound(intensity, peak_flops, bandwidth):
    """stage 2: 'memory' if intensity is below the ridge point, else 'compute'."""
    return "memory" if intensity < ridge_point(peak_flops, bandwidth) else "compute"


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
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()  # drain the warmup's queued work before the first clock read
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()  # wait for the device to finish before reading the clock
        times.append(time.perf_counter() - t0)
    return {"best": min(times), "median": float(np.median(times)), "times": times}


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
    return 6 * n_params * n_tokens


@app.function
def gpt_block_ops(batch, T, d_model, n_heads, *, d_ff=None, bytes_per_element=4):
    """stage 4: forward ops of one pre-LN GPT block: list of {'name','flops','bytes'} in execution order (spec table)."""
    d_ff = 4 * d_model if d_ff is None else d_ff
    b, BT, BH, dh = bytes_per_element, batch * T, batch * n_heads, d_model // n_heads

    def linear(name, d_in, d_out):
        return {"name": name, "flops": matmul_flops(BT, d_out, d_in), "bytes": matmul_bytes(BT, d_out, d_in, bytes_per_element=b)}

    def elementwise(name, n, flops_per, reads, writes):
        return {"name": name, "flops": flops_per * n, "bytes": (reads + writes) * n * b}

    return [
        elementwise("ln1", BT * d_model, 7, 1, 1),
        linear("q_proj", d_model, d_model),
        linear("k_proj", d_model, d_model),
        linear("v_proj", d_model, d_model),
        {"name": "scores", "flops": BH * matmul_flops(T, T, dh), "bytes": BH * matmul_bytes(T, T, dh, bytes_per_element=b)},
        elementwise("softmax", BH * T * T, 5, 1, 1),
        {"name": "attn_v", "flops": BH * matmul_flops(T, dh, T), "bytes": BH * matmul_bytes(T, dh, T, bytes_per_element=b)},
        linear("out_proj", d_model, d_model),
        elementwise("residual1", BT * d_model, 1, 2, 1),
        elementwise("ln2", BT * d_model, 7, 1, 1),
        linear("fc1", d_model, d_ff),
        elementwise("gelu", BT * d_ff, 8, 1, 1),
        linear("fc2", d_ff, d_model),
        elementwise("residual2", BT * d_model, 1, 2, 1),
    ]


@app.function
def mfu(flops_per_step, seconds_per_step, peak_flops):
    """stage 4: model FLOP utilization: achieved FLOP/s (flops_per_step / seconds_per_step) as a fraction of peak."""
    return flops_per_step / seconds_per_step / peak_flops


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
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        step_fn()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(n_steps):
            step_fn()
    rows = [(e.key, float(e.self_cpu_time_total), int(e.count)) for e in prof.key_averages()]
    return sorted(rows, key=lambda r: r[1], reverse=True)


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
