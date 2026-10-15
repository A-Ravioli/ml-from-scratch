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
    # writing kernels · build

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
    raise NotImplementedError("stage1: implement add_kernel")


@app.function
def add(x, y, out, n, *, block=128):
    """stage 1: launch add_kernel over cdiv(n, block) programs; returns the grid used."""
    raise NotImplementedError("stage1: implement add")


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
    raise NotImplementedError("stage2: implement softmax_kernel")


@app.function
def softmax(x, out, n_rows, n_cols, *, stride=None):
    """stage 2: one program per row, BLOCK = the smallest power of two >= n_cols; returns (grid, BLOCK)."""
    raise NotImplementedError("stage2: implement softmax")


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
    raise NotImplementedError("stage3: implement matmul_kernel")


@app.function
def matmul(a, b, c, M, N, K, *, block_m=16, block_n=16, block_k=16):
    """stage 3: c (M, N) = a (M, K) @ b (K, N), all row-major, one program per output tile; returns the grid."""
    raise NotImplementedError("stage3: implement matmul")


@app.function
def matmul_traffic(M, N, K, block_m, block_n):
    """stage 3: elements loaded by the tiled matmul: (a_loads, b_loads) = (M K cdiv(N, block_n), K N cdiv(M, block_m))."""
    raise NotImplementedError("stage3: implement matmul_traffic")


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
    raise NotImplementedError("stage4: implement metal_softmax_source")


@app.function
@functools.cache
def compile_metal(source):
    """compile Metal source once per distinct string (torch.mps.compile_shader); given, not an exercise."""
    return torch.mps.compile_shader(source)


@app.function
def metal_softmax(x):
    """stage 4: row softmax of a 2-D float32 tensor: your Metal kernel on mps, your stage-2 kernel otherwise."""
    raise NotImplementedError("stage4: implement metal_softmax")


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
