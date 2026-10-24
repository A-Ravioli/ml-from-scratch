# the performance model · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

three one-liners: `2 * m * n * k`, `(m * k + k * n + m * n) * bytes_per_element`, `flops / bytes_moved`. keep the first two in integers (no `float()`), so they stay exact for big sizes. the test uses three different dimensions on purpose: if a check fails, write out which matrix is $m \times k$, which is $k \times n$, and which is the output.

</details>

## stage 2

<details><summary>hint</summary>

`np.minimum(peak_flops, np.asarray(intensity, dtype=float) * bandwidth)` works for scalars and arrays; return `float(out)` when `out.ndim == 0`. Python's built-in `min` breaks on arrays. `bound` compares with `ridge_point(peak_flops, bandwidth)` using a strict `<` for `"memory"`.

</details>

## stage 3

<details><summary>hint</summary>

```python
for _ in range(warmup):
    fn()
if sync is not None:
    sync()
times = []
for _ in range(repeats):
    t0 = time.perf_counter()
    fn()
    if sync is not None:
        sync()
    times.append(time.perf_counter() - t0)
```

then return `{"best": min(times), "median": float(np.median(times)), "times": times}`.

</details>

## stage 4

<details><summary>hint</summary>

write two small local helpers inside `gpt_block_ops`: `linear(name, d_in, d_out)` returning `matmul_flops(B*T, d_out, d_in)` and `matmul_bytes(B*T, d_out, d_in, bytes_per_element=b)`, and `elementwise(name, n_elements, flops_per_element, reads, writes)` returning `flops_per_element * n` and `(reads + writes) * n * b`. the attention matmuls are `B * n_heads` copies of a single-head product: `B * H * matmul_flops(T, T, d_h)` for the scores (output $T \times T$, inner dimension $d_h$) and `B * H * matmul_flops(T, d_h, T)` for `attn_v`. the residual adds read two tensors and write one.

</details>

## stage 5

<details><summary>hint</summary>

```python
from torch.profiler import profile, ProfilerActivity
for _ in range(warmup):
    step_fn()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(n_steps):
        step_fn()
rows = [(e.key, float(e.self_cpu_time_total), int(e.count)) for e in prof.key_averages()]
```

then sort `rows` by the second element, descending. the import can live inside the function (only setup-cell imports are shared between cells, and an `@app.function` may import locally).

</details>
