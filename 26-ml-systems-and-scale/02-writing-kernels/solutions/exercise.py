import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import functools
    import math

    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # writing kernels · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    stages 1–3 run on a small **block-program interpreter** defined just below: a NumPy model of
    how Triton (and CUDA, and Metal) kernels see memory. it runs anywhere and counts every
    element your kernels load and store, so the checks can verify not just *what* you compute
    but *how much memory traffic* it costs. stage 4 is a real GPU kernel in Metal, Apple's GPU
    language, run through `torch.mps.compile_shader`; on a machine without an Apple GPU it falls
    back to your stage-2 kernel.

    | stage | you build |
    |---|---|
    | 1 | `add_kernel`, `add`: the programming model: program ids, blocks of offsets, masks |
    | 2 | `softmax_kernel`, `softmax`: a fused row softmax that reads and writes each element once |
    | 3 | `matmul_kernel`, `matmul`, `matmul_traffic`: a tiled matmul, and the traffic it saves |
    | 4 | `metal_softmax_source`, `metal_softmax`: the fused softmax as a Metal GPU kernel |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the interpreter (given)

    `Buffer` is a flat array in "global memory" (DRAM). kernels may touch it **only** through
    `load` and `store`, which take an array of element offsets and an optional boolean mask, and
    count the elements actually moved. `launch(kernel, grid, *args)` calls
    `kernel(pid, *args)` once for every program id `pid` (a tuple) in the grid. reading
    `buf.data` directly from a kernel is cheating: the traffic checks will catch it.
    """)
    return


@app.class_definition
class Buffer:
    """a flat float32 array in global memory; counts elements loaded and stored through load/store."""

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32).reshape(-1)
        self.loads = 0
        self.stores = 0

    @classmethod
    def empty(cls, n):
        """a buffer of n elements (filled with NaN, so reading before writing shows up)."""
        return cls(np.full(n, np.nan, dtype=np.float32))

    def reset_counts(self):
        self.loads = self.stores = 0


@app.function
def check_offsets(buf, offsets, mask):
    bad = mask & ((offsets < 0) | (offsets >= buf.data.size))
    if bad.any():
        raise IndexError(
            f"out-of-bounds access at offset {int(offsets[bad].flat[0])} of a buffer with {buf.data.size} elements: "
            "on a GPU this reads garbage or crashes. mask the lanes past the end")


@app.function
def load(buf, offsets, mask=None, other=0.0):
    """read buf at integer `offsets` (any shape) where `mask` is True; masked-off lanes get `other`. counted."""
    offsets = np.asarray(offsets, dtype=np.int64)
    mask = np.ones(offsets.shape, dtype=bool) if mask is None else np.broadcast_to(np.asarray(mask, dtype=bool), offsets.shape)
    check_offsets(buf, offsets, mask)
    out = np.full(offsets.shape, other, dtype=np.float32)
    out[mask] = buf.data[offsets[mask]]
    buf.loads += int(mask.sum())
    return out


@app.function
def store(buf, offsets, values, mask=None):
    """write `values` (broadcast to offsets' shape) into buf at `offsets` where `mask` is True. counted."""
    offsets = np.asarray(offsets, dtype=np.int64)
    mask = np.ones(offsets.shape, dtype=bool) if mask is None else np.broadcast_to(np.asarray(mask, dtype=bool), offsets.shape)
    check_offsets(buf, offsets, mask)
    values = np.broadcast_to(np.asarray(values, dtype=np.float32), offsets.shape)
    buf.data[offsets[mask]] = values[mask]
    buf.stores += int(mask.sum())


@app.function
def launch(kernel, grid, *args):
    """run kernel(pid, *args) for every program id pid (a tuple of ints) in the grid (a tuple of sizes)."""
    for pid in np.ndindex(*grid):
        kernel(tuple(int(p) for p in pid), *args)


@app.function
def cdiv(a, b):
    """ceil(a / b) for positive ints: the number of blocks of size b needed to cover a elements."""
    return -(-a // b)


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def add_kernel(pid, x, y, out, n, BLOCK):
    """stage 1: program pid[0] adds elements pid[0]*BLOCK ... +BLOCK-1 of x and y into out, masking past n."""
    offs = pid[0] * BLOCK + np.arange(BLOCK)
    mask = offs < n
    store(out, offs, load(x, offs, mask) + load(y, offs, mask), mask)


@app.function
def add(x, y, out, n, *, block=128):
    """stage 1: launch add_kernel over cdiv(n, block) programs; returns the grid used."""
    grid = (cdiv(n, block),)
    launch(add_kernel, grid, x, y, out, n, block)
    return grid


@app.cell(hide_code=True)
def _():
    checks.run(1, add_kernel=add_kernel, add=add, Buffer=Buffer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def softmax_kernel(pid, x, out, n_cols, stride, BLOCK):
    """stage 2: program pid[0] softmaxes row pid[0] (starting at pid[0]*stride) in one load and one store."""
    cols = np.arange(BLOCK)
    mask = cols < n_cols
    offs = pid[0] * stride + cols
    row = load(x, offs, mask, other=-np.inf)  # padding lanes: -inf, so they don't change the max or the sum
    z = np.exp(row - row.max())
    store(out, offs, z / z.sum(), mask)


@app.function
def softmax(x, out, n_rows, n_cols, *, stride=None):
    """stage 2: one program per row, BLOCK = the smallest power of two >= n_cols; returns (grid, BLOCK)."""
    stride = n_cols if stride is None else stride
    block = 1 << max(n_cols - 1, 0).bit_length()
    grid = (n_rows,)
    launch(softmax_kernel, grid, x, out, n_cols, stride, block)
    return grid, block


@app.cell(hide_code=True)
def _():
    checks.run(2, softmax_kernel=softmax_kernel, softmax=softmax, Buffer=Buffer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def matmul_kernel(pid, a, b, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    """stage 3: program (pid_m, pid_n) computes the BLOCK_M x BLOCK_N tile of c = a @ b, looping over K in BLOCK_K steps."""
    rm = pid[0] * BLOCK_M + np.arange(BLOCK_M)
    rn = pid[1] * BLOCK_N + np.arange(BLOCK_N)
    acc = np.zeros((BLOCK_M, BLOCK_N), dtype=np.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + np.arange(BLOCK_K)
        a_tile = load(a, rm[:, None] * K + rk[None, :], (rm[:, None] < M) & (rk[None, :] < K))
        b_tile = load(b, rk[:, None] * N + rn[None, :], (rk[:, None] < K) & (rn[None, :] < N))
        acc += a_tile @ b_tile
    store(c, rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))


@app.function
def matmul(a, b, c, M, N, K, *, block_m=16, block_n=16, block_k=16):
    """stage 3: c (M, N) = a (M, K) @ b (K, N), all row-major, one program per output tile; returns the grid."""
    grid = (cdiv(M, block_m), cdiv(N, block_n))
    launch(matmul_kernel, grid, a, b, c, M, N, K, block_m, block_n, block_k)
    return grid


@app.function
def matmul_traffic(M, N, K, block_m, block_n):
    """stage 3: elements loaded by the tiled matmul: (a_loads, b_loads) = (M K cdiv(N, block_n), K N cdiv(M, block_m))."""
    return M * K * cdiv(N, block_n), K * N * cdiv(M, block_m)


@app.cell(hide_code=True)
def _():
    checks.run(3, matmul_kernel=matmul_kernel, matmul=matmul, matmul_traffic=matmul_traffic, Buffer=Buffer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def metal_softmax_source():
    """stage 4: Metal source of kernel softmax_rows(out, x, n_cols): one 256-thread threadgroup per contiguous row."""
    return r"""
    #include <metal_stdlib>
    using namespace metal;

    kernel void softmax_rows(device float* out [[buffer(0)]],
                             device const float* x [[buffer(1)]],
                             constant uint& n_cols [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint nthreads [[threads_per_threadgroup]]) {
        threadgroup float scratch[256];
        device const float* xr = x + row * n_cols;
        device float* outr = out + row * n_cols;

        // 1. each thread finds the max of its strided slice of the row, then a tree reduction
        float m = -INFINITY;
        for (uint j = tid; j < n_cols; j += nthreads) m = max(m, xr[j]);
        scratch[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] = max(scratch[tid], scratch[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        m = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. the same for the sum of exponentials
        float s_local = 0.0f;
        for (uint j = tid; j < n_cols; j += nthreads) s_local += exp(xr[j] - m);
        scratch[tid] = s_local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = 1.0f / scratch[0];

        // 3. write the normalized row
        for (uint j = tid; j < n_cols; j += nthreads) outr[j] = exp(xr[j] - m) * inv;
    }
    """


@app.function
@functools.cache
def compile_metal(source):
    """compile Metal source once per distinct string (torch.mps.compile_shader); given, not an exercise."""
    return torch.mps.compile_shader(source)


@app.function
def metal_softmax(x):
    """stage 4: row softmax of a 2-D float32 tensor: your Metal kernel on mps, your stage-2 kernel otherwise."""
    if x.device.type == "mps":
        x = x.contiguous()
        out = torch.empty_like(x)
        compile_metal(metal_softmax_source()).softmax_rows(out, x, x.shape[1], threads=x.shape[0] * 256,
                                                            group_size=256)
        return out
    n_rows, n_cols = x.shape
    xb, ob = Buffer(x.detach().cpu().numpy()), Buffer.empty(n_rows * n_cols)
    softmax(xb, ob, n_rows, n_cols)
    return torch.from_numpy(ob.data.reshape(n_rows, n_cols).copy()).to(x.device)


@app.cell(hide_code=True)
def _():
    checks.run(4, metal_softmax_source=metal_softmax_source, metal_softmax=metal_softmax, compile_metal=compile_metal)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it measures your kernels'
    traffic against eager PyTorch's, then (on an Apple GPU) races your Metal softmax and a tiled
    Metal matmul against PyTorch's own kernels. nothing from this topic joins `mlfs`: the point
    is the mental model, which topic 03's mixed precision and the capstone's fused attention
    and `torch.compile` both rely on.
    """)
    return


if __name__ == "__main__":
    app.run()
