# writing kernels · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

```python
offs = pid[0] * BLOCK + np.arange(BLOCK)
mask = offs < n
store(out, offs, load(x, offs, mask) + load(y, offs, mask), mask)
```

the launcher computes `grid = (cdiv(n, block),)`, calls `launch(add_kernel, grid, x, y, out, n, block)` and returns `grid`.

</details>

## stage 2

<details><summary>hint</summary>

`cols = np.arange(BLOCK)`, `mask = cols < n_cols`, `offs = pid[0] * stride + cols`; `row = load(x, offs, mask, other=-np.inf)`; then `z = np.exp(row - row.max())` and `store(out, offs, z / z.sum(), mask)`. in the launcher, the smallest power of two $\ge n$ is `1 << (n - 1).bit_length()` for $n \ge 1$ (it gives 1 for $n = 1$).

</details>

## stage 3

<details><summary>hint</summary>

```python
rm = pid[0] * BLOCK_M + np.arange(BLOCK_M)
rn = pid[1] * BLOCK_N + np.arange(BLOCK_N)
acc = np.zeros((BLOCK_M, BLOCK_N), dtype=np.float32)
for k0 in range(0, K, BLOCK_K):
    rk = k0 + np.arange(BLOCK_K)
    a_tile = load(a, rm[:, None] * K + rk[None, :], (rm[:, None] < M) & (rk[None, :] < K))
    b_tile = load(b, rk[:, None] * N + rn[None, :], (rk[:, None] < K) & (rn[None, :] < N))
    acc += a_tile @ b_tile
store(c, rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))
```

`matmul_traffic` is the lesson's formula with `cdiv`.

</details>

## stage 4

<details><summary>hint</summary>

the kernel's skeleton:

```metal
#include <metal_stdlib>
using namespace metal;
kernel void softmax_rows(device float* out [[buffer(0)]], device const float* x [[buffer(1)]],
                         constant uint& n_cols [[buffer(2)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint nthreads [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    device const float* xr = x + row * n_cols;
    // max: strided loop, scratch[tid] = m, barrier, tree reduction (barrier each round), m = scratch[0], barrier
    // sum of exp(xr[j] - m): the same pattern with +
    // strided loop writing exp(xr[j] - m) / sum
}
```

threads with `tid >= n_cols` simply skip the strided loops and contribute $-\infty$ (max) or $0$ (sum). in `metal_softmax`, call `compile_metal(metal_softmax_source()).softmax_rows(out, x, x.shape[1], threads=x.shape[0] * 256, group_size=256)` on a contiguous `x` with `out = torch.empty_like(x)`; on other devices wrap the data in `Buffer`s, call your `softmax`, and convert back with `torch.from_numpy(...)`.

</details>
