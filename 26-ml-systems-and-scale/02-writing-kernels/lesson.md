# writing kernels

## why this matters

topic 01 ended with a table in which the softmax over attention scores, with about 1% of a GPT block's FLOPs, took a large share of its time. the reason is how eager PyTorch runs it. `torch.exp(x - x.amax(-1, keepdim=True))` and the division after it are four or five separate operations, and each one is a separate **kernel**: a program launched on the GPU that reads its inputs from DRAM and writes its output back to DRAM. the intermediate tensors go out to memory and come straight back in. for a memory-bound op, that round trip *is* the cost.

a kernel that does all of it at once, reading $x$ once and writing the softmax once, keeps the intermediates in registers and on-chip memory. that's **fusion**, and it's the single most important trick in making deep learning fast: FlashAttention is a fused attention kernel, `torch.compile` is mostly an automatic fuser, and the capstone's biggest win is a fused kernel. the other trick is **tiling**: reorganizing a matmul so each number loaded from DRAM is reused many times from fast memory. it's why a matmul can reach the compute roof at all.

this topic teaches both by writing kernels. the curriculum's original plan uses Triton, a Python language for GPU kernels; Triton needs an NVIDIA or AMD GPU, and this course runs on laptops. so you'll write Triton-style kernels for a small NumPy interpreter that runs anywhere and counts every byte your kernels move (stages 1–3, each shown next to the Triton code it mirrors), then write one real GPU kernel in **Metal**, the language of Apple GPUs, and race it against PyTorch (stage 4). if you have an NVIDIA GPU, the lesson's Triton kernels run as written (`pip install triton`); on Linux without one, `TRITON_INTERPRET=1` runs real `@triton.jit` kernels on the CPU through NumPy, slowly but exactly, so you can check the Triton versions of your kernels too.

## warm-up: predict first

1. eager PyTorch computes a row softmax as max, subtract, exp, sum, divide. if each op reads its inputs from DRAM and writes its output to DRAM, how many times more bytes does that move than a single fused kernel?
2. a GPU kernel is launched over a grid of 1,000 programs, each handling a block of 1,024 elements, for an array of 1,000,000 elements. what goes wrong in the last program, and how do you prevent it?
3. computing $C = AB$ for $1024 \times 1024$ matrices one output element at a time, each element reads a row of $A$ and a column of $B$. how many elements are loaded in total? how many would be enough?
4. you write a fused softmax that's 3× faster than eager PyTorch's five-op version. will it also beat `torch.softmax`?

## prerequisites check

- [ ] you can compute FLOPs, bytes and arithmetic intensity, and place an op on a roofline (→ topic 01)
- [ ] you know why softmax subtracts the max, and how online softmax handles a row in blocks (→ ch 00 · 03, ch 18 · 06)
- [ ] you can index NumPy arrays with integer arrays and boolean masks, and broadcast `a[:, None] + b[None, :]` (→ ch 00 · 02)
- [ ] you know a matmul as a sum of outer products over $k$ (→ ch 02)

## core ideas

### idea 1: programs, blocks and masks

a GPU runs thousands of threads at once, organized in groups that share fast on-chip memory: CUDA calls them *thread blocks*, Metal *threadgroups*. you write the code for one group and launch it over a **grid**; each instance learns its position from a **program id** and uses it to pick its share of the data.

Triton raises the level by one notch: you write code for a whole group at once, operating on **blocks** of values (vectors and small matrices) instead of single threads, and the compiler maps the block operations onto threads. the interpreter in your notebook models exactly that view, in NumPy. memory is a flat `Buffer` (DRAM). a kernel is a Python function `kernel(pid, ...)`, run once per program id by `launch(kernel, grid, ...)`. it may touch a buffer *only* through `load(buf, offsets, mask, other)` and `store(buf, offsets, values, mask)`, which take arrays of element offsets (the pointer arithmetic of a real kernel) and count every element moved. a vector add, in both:

```python
# your interpreter (runs anywhere)              # Triton (needs an NVIDIA/AMD GPU)
def add_kernel(pid, x, y, out, n, BLOCK):        @triton.jit
                                                 def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = pid[0] * BLOCK + np.arange(BLOCK)         offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n                                  mask = offs < n
    a = load(x, offs, mask)                          a = tl.load(x_ptr + offs, mask=mask)
    b = load(y, offs, mask)                          b = tl.load(y_ptr + offs, mask=mask)
    store(out, offs, a + b, mask)                    tl.store(out_ptr + offs, a + b, mask=mask)

launch(add_kernel, (cdiv(n, BLOCK),), ...)       add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=1024)
```

the **mask** handles the ragged end: with $n = 1{,}000{,}000$ and blocks of 1,024 you need $\lceil n / 1024 \rceil = 977$ programs, and the last one's offsets run past the end of the array. on a GPU an unmasked load there reads whatever lives next door, or crashes; the interpreter raises an `IndexError` instead. masked-off lanes of a load get the value `other`.

> **spec.** `add_kernel(pid, x, y, out, n, BLOCK)`: program `pid[0]` loads elements $[\text{pid}_0 \cdot \text{BLOCK}, (\text{pid}_0 + 1)\text{BLOCK})$ of `x` and `y` that are $< n$ and stores their sum in `out`. `add(x, y, out, n, *, block=128)` launches it over `(cdiv(n, block),)` programs and returns the grid. every element is loaded once from each input and stored once.

> **▶ build: stage 1.** implement `add_kernel` and `add`.

### idea 2: fusion

count the traffic of eager PyTorch's row softmax on an $r \times n$ matrix, op by op ($r \ll rn$, so ignore per-row vectors):

| op | reads | writes |
|---|---|---|
| `m = x.amax(-1, keepdim=True)` | $rn$ | – |
| `t = x - m` | $rn$ | $rn$ |
| `e = t.exp()` | $rn$ | $rn$ |
| `s = e.sum(-1, keepdim=True)` | $rn$ | – |
| `y = e / s` | $rn$ | $rn$ |

five reads and three writes: eight passes over the data. a fused kernel reads $x$ once and writes $y$ once: two passes, $4\times$ less traffic, and since the op is deeply memory-bound ($I < 1$), up to $4\times$ faster.

the fused kernel gives each program one row. it loads the whole row into a block (the next power of two at least as long as the row; Triton requires power-of-two block sizes, so you'll use the same rule), computes the max, the exponentials, the sum and the quotient on the block (all in registers and on-chip memory), and stores the result. the padding lanes past the row's end need care: load them with `other = -inf`, so they can't raise the max, and $e^{-\infty - m} = 0$ adds nothing to the sum. load them as 0 and every row's sum quietly gains a few extra $e^{-m}$ terms.

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
    z = tl.exp(x - tl.max(x, axis=0))
    tl.store(out_ptr + row * stride + cols, z / tl.sum(z, axis=0), mask=mask)
```

`stride` is the distance between the starts of consecutive rows: equal to `n_cols` for a contiguous matrix, larger for a slice of a wider one. this design needs the row to fit in one program's fast memory; for longer rows, the online softmax of chapter 18 processes the row in blocks with a running max, and FlashAttention fuses that with the two attention matmuls.

> **spec.** `softmax_kernel(pid, x, out, n_cols, stride, BLOCK)`: program `pid[0]` softmaxes row `pid[0]`, whose elements sit at offsets `pid[0] * stride + j` for $j < $ `n_cols`, with exactly one load of those elements and one store. `softmax(x, out, n_rows, n_cols, *, stride=None)` (`stride` defaults to `n_cols`) launches `(n_rows,)` programs with BLOCK the smallest power of two $\ge$ `n_cols`, and returns `(grid, BLOCK)`.

> **🔮 predict:** your fused kernel moves 2 elements per input element, eager PyTorch 8. on a real GPU, will the speedup be the full 4×? on a CPU?

> **▶ build: stage 2.** implement `softmax_kernel` and `softmax`.

### idea 3: tiling

a matmul has plenty of arithmetic to hide memory behind (topic 01: intensity $n/6$), but only if each loaded number is *reused* from fast memory. computing one output element at a time, $c_{ij} = \sum_k a_{ik} b_{kj}$ loads $2K$ elements for $2K$ FLOPs, and all of $C$ loads $2MNK$ elements: an intensity of a quarter FLOP per byte, as bad as an elementwise op.

**tiling** fixes it. each program computes a $B_M \times B_N$ tile of $C$. it walks along $K$ in steps of $B_K$, loading a $B_M \times B_K$ tile of $A$ and a $B_K \times B_N$ tile of $B$ into fast memory and accumulating their product into a $B_M \times B_N$ accumulator that never leaves registers. every element of the $A$ tile is used for $B_N$ outputs, every element of the $B$ tile for $B_M$. in total, $A$ is loaded once for each column of tiles and $B$ once for each row of tiles:

$$\text{loads}_A = MK \left\lceil \tfrac{N}{B_N} \right\rceil, \qquad \text{loads}_B = KN \left\lceil \tfrac{M}{B_M} \right\rceil.$$

with square tiles of side $b$ that's about $2MNK / b$ elements, $b$ times less than one-element-at-a-time, and an intensity of about $b/4$ FLOP/byte in float32. bigger tiles mean more reuse, up to the size of fast memory: a $b \times b$ float32 tile of each input costs $8b^2$ bytes, and Apple GPUs give a threadgroup 32 KB, so $b = 64$ is about the limit. real kernels add more levels (registers, SIMD groups, the L2 cache) and use the GPU's matrix units, which is why vendor libraries still beat a straightforward tiled kernel by a wide margin.

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
```

note the 2-D offsets: `rm[:, None] * K + rk[None, :]` is the $B_M \times B_K$ grid of row-major addresses of the tile, built by broadcasting. masked lanes load 0, which is harmless in a sum of products.

> **spec.** `matmul_kernel(pid, a, b, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)`: program `(pid[0], pid[1])` computes rows $[\text{pid}_0 B_M, (\text{pid}_0+1) B_M)$ and columns $[\text{pid}_1 B_N, (\text{pid}_1 + 1) B_N)$ of $C = AB$ (row-major `a` $(M, K)$, `b` $(K, N)$, `c` $(M, N)$), loading one $B_M \times B_K$ tile of `a` and one $B_K \times B_N$ tile of `b` per step along $K$, and storing its tile once at the end, masked at all three edges. `matmul(a, b, c, M, N, K, *, block_m=16, block_n=16, block_k=16)` launches `(cdiv(M, block_m), cdiv(N, block_n))` programs and returns the grid. `matmul_traffic(M, N, K, block_m, block_n)` returns `(loads_A, loads_B)` from the formula above.

> **▶ build: stage 3.** implement `matmul_kernel`, `matmul` and `matmul_traffic`.

### idea 4: a real kernel in Metal

Triton hides threads; Metal, like CUDA, doesn't. a Metal **kernel** is a C++-like function run by every thread of the grid; attributes tell each thread who it is: `[[threadgroup_position_in_grid]]` (which group: Triton's program id), `[[thread_position_in_threadgroup]]` (which thread inside it) and `[[threads_per_threadgroup]]`. arrays declared `threadgroup` live in the group's 32 KB of on-chip memory, shared by its threads; `threadgroup_barrier(mem_flags::mem_threadgroup)` makes every thread wait until all have reached it, so values written to threadgroup memory are visible to the others. `device` pointers are DRAM.

a block operation like Triton's `tl.max(x, axis=0)` becomes two steps. each of the group's threads walks its own strided slice of the row (columns `tid`, `tid + 256`, `tid + 512`, ...) and keeps a private maximum; then a **tree reduction** in threadgroup memory halves the number of active threads each round (256 → 128 → … → 1), with a barrier between rounds:

```metal
threadgroup float scratch[256];
float m = -INFINITY;
for (uint j = tid; j < n_cols; j += nthreads) m = max(m, xr[j]);
scratch[tid] = m;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint s = nthreads / 2; s > 0; s >>= 1) {
    if (tid < s) scratch[tid] = max(scratch[tid], scratch[tid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
m = scratch[0];   // every thread now knows the row max
```

the sum of exponentials is the same pattern with `+`, and a last strided loop writes the normalized row. the row's second and third reads hit the GPU's cache rather than DRAM for rows that aren't huge, so DRAM traffic stays close to one read and one write. one subtle bug: reading `scratch[0]` and then reusing `scratch` for the next reduction without a barrier in between lets a fast thread overwrite it before a slow one has read it.

PyTorch compiles Metal source at run time and calls its kernels on `mps` tensors:

```python
lib = torch.mps.compile_shader(source)          # compile once
lib.softmax_rows(out, x, n_cols, threads=n_rows * 256, group_size=256)
```

`threads` is the total number of threads in the grid and `group_size` the threads per group, so this launches one 256-thread group per row. tensor arguments bind to `device` pointers in order (`[[buffer(0)]]`, `[[buffer(1)]]`, …) and Python ints to `constant uint&`. the tests compile your source and call it exactly like that.

> **spec.** `metal_softmax_source()` returns Metal source defining `kernel void softmax_rows(device float* out [[buffer(0)]], device const float* x [[buffer(1)]], constant uint& n_cols [[buffer(2)]], ...)` that computes the softmax of each contiguous row of `x` (row $r$ at offset $r \cdot$ `n_cols`) into `out` when launched with `threads = n_rows * 256, group_size = 256`; correct for any `n_cols` $\ge 1$, numerically stable. `metal_softmax(x)`: for a 2-D float32 tensor on `mps`, runs your kernel (compile it with the given `compile_metal`, which caches); on any other device, runs your stage-2 `softmax` on the interpreter. returns a tensor of `x`'s shape on `x`'s device.

> **🔮 predict:** PyTorch's own `torch.softmax` is already one fused kernel, written by Apple for its GPUs. will your 30-line Metal kernel be slower, the same, or faster?

> **▶ build: stage 4.** implement `metal_softmax_source` and `metal_softmax`. without an Apple GPU, the Metal kernel can't run, so off Apple stage 4 is graded only structurally (the source defines `softmax_rows`, and the CPU path uses your stage-2 kernel); write the kernel anyway and compare it with the reference's line by line.

## worked example

**softmax over chapter 18's attention scores.** model M at batch 64: $384 \times 128$ rows of 128 scores, $rn = 6{,}291{,}456$ floats. eager: $8 \times 6{,}291{,}456 \times 4 = 201{,}326{,}592$ bytes. fused: $2 \times 6{,}291{,}456 \times 4 = 50{,}331{,}648$ bytes. at 100 GB/s, 2.01 ms against 0.50 ms per block, per forward pass. your `softmax` on the interpreter must report exactly one load and one store per element (the check uses $5 \times 37$).

**tiling a $1024^3$ matmul.** one element at a time: $2 \times 1024^3 = 2{,}147{,}483{,}648$ loads (8.6 GB in float32). with $16 \times 16$ tiles: $A$ is loaded $1024/16 = 64$ times, $1024^2 \times 64 = 67{,}108{,}864$ loads, the same for $B$: 537 MB, 16× less. with $64 \times 64$ tiles: 134 MB. the minimum, each input read once, is 8.4 MB. your `matmul_traffic(1024, 1024, 1024, 16, 16)` must return `(67108864, 67108864)`.

## on paper

1. **the tiled matmul's intensity.** with square $b \times b$ output tiles and $M = N = K = n$ divisible by $b$, show that the DRAM traffic of the tiled matmul is $4(2n^3/b + n^2)$ bytes in float32, and that its intensity tends to $b/4$ FLOP/byte for large $n$. with a ridge point of 30 FLOP/byte, what tile size makes a matmul compute-bound?

<details><summary>solution</summary>

loads $n^2 \cdot (n/b)$ for each input and $n^2$ stores: $2n^3/b + n^2$ elements, $\times 4$ bytes. intensity $2n^3 / (4(2n^3/b + n^2)) = 1/(4/b + 2/n) \to b/4$. compute-bound needs $b/4 \ge 30$, $b \ge 120$: more than fits in 32 KB as two float32 tiles ($8b^2 = 115$ KB). real kernels get there with rectangular tiles, register blocking and the cache, and by using bf16 (halving the bytes).

</details>

2. **why eager can't fuse.** eager PyTorch runs `y = torch.exp(x - m)` as two kernels. what would it need to know, at the moment it launches the subtraction, to fuse them? why can `torch.compile` do it?

<details><summary>solution</summary>

it would need to know the next operation, but in eager mode Python hasn't executed it yet: each op runs as soon as it's called and must leave a complete tensor behind, since any later Python code might read it. `torch.compile` captures the whole function as a graph first, sees that `x - m` is only consumed by `exp`, and generates one kernel for both (on NVIDIA GPUs, a Triton kernel).

</details>

3. **the bug the mask hides.** in the fused softmax, the padding lanes are loaded with `other = 0` instead of $-\infty$, for a row of 37 values in a block of 64 whose max is $m$. what is the computed sum of exponentials, and when is the error invisible?

<details><summary>solution</summary>

each of the 27 padding lanes contributes $e^{0 - m}$ (if $m \ge 0$, the max is unaffected; if every real value is negative, the padding's 0 *becomes* the max). the sum is $\sum_j e^{x_j - m} + 27 e^{-m}$, and every output is too small by that factor. it's invisible when $m$ is large (say $x \sim 10$), so tests on small random inputs, where $m \approx 2$, catch it.

</details>

## experiment

the lab counts your fused softmax's traffic against an eager five-kernel version on the interpreter and times the real thing on your CPU and GPU. it measures how the tiled matmul's traffic falls with tile size, then (on an Apple GPU) races a naive and a tiled Metal matmul against PyTorch's. finally your Metal softmax takes on eager PyTorch's composite softmax and `torch.softmax` across row lengths.

## check your understanding

1. what does the program id buy you, and what's its CUDA name?

<details><summary>answer</summary>

it's how one piece of code, launched thousands of times, knows which part of the data is its own: offsets are computed from it. in CUDA it's `blockIdx` (with `threadIdx` inside the block); in Metal, `threadgroup_position_in_grid`.

</details>

2. why does fusion speed up a memory-bound chain but barely change a compute-bound one?

<details><summary>answer</summary>

fusion removes DRAM round trips of intermediates; it doesn't remove any arithmetic. for a memory-bound chain the time is the traffic, so it falls; for a compute-bound op the time is the arithmetic, which is unchanged.

</details>

3. a tiled matmul's tiles are $16 \times 16$. you change them to $64 \times 64$. by how much does DRAM traffic fall, and what limits how far you can go?

<details><summary>answer</summary>

about 4× (loads scale as $1/b$). the limit is fast memory: the tiles (and the accumulator, in registers) must fit, and bigger tiles also mean fewer programs, which can leave parts of the GPU idle for small matrices.

</details>

4. why is the softmax kernel's block size a power of two, and why must padding lanes be $-\infty$?

<details><summary>answer</summary>

Triton requires block shapes that are powers of two (they map onto hardware lanes and vector registers). padding lanes must not change the result: $-\infty$ is the identity for max and gives $e^{-\infty} = 0$, the identity for the sum.

</details>

5. in the Metal kernel, which barriers are essential?

<details><summary>answer</summary>

one after every write to `scratch` that other threads will read (after each thread stores its partial result, and after each round of the tree), and one between reading the final `scratch[0]` and overwriting `scratch` for the next reduction. without them, threads read values that haven't been written yet or have already been replaced.

</details>

6. you fuse softmax and win 3×. why doesn't everyone write every op as a custom kernel?

<details><summary>answer</summary>

it's expensive engineering per op, per hardware target and per dtype, the kernels must also have backward passes, and a fusion only helps where it removes memory traffic. that's why automatic fusers (`torch.compile`, XLA) exist, and why hand-written kernels are kept for the few ops that dominate: attention, normalization, optimizers, quantized matmuls.

</details>

## common bugs

- **unmasked tails.** a launch with `cdiv(n, BLOCK)` programs whose last program loads past the end. the interpreter raises `IndexError`; a GPU silently reads garbage.
- **padding loaded as 0 in a softmax.** rows still sum to about 1 for large inputs, so the bug survives casual tests; use `other=-inf`.
- **assuming contiguous rows.** using `row * n_cols` instead of `row * stride` works until someone passes a slice.
- **reading `buf.data` inside a kernel.** the result is right and the traffic count is zero: you've skipped the memory model the exercise is about.
- **a missing threadgroup barrier in Metal.** results that are right most of the time, wrong for some sizes, and different between runs.
- **forgetting `[[buffer(i)]]` order.** Metal binds arguments by index; passing `(x, out)` to a kernel declared `(out, x)` overwrites your input.

## going deeper

- Tillet, Kung & Cox (2019), "Triton: an intermediate language and compiler for tiled neural network computations", *MAPL*. [doi:10.1145/3315508.3329973](https://doi.org/10.1145/3315508.3329973)
- the Triton tutorials (vector addition, fused softmax, matrix multiplication), which this topic's kernels mirror. [triton-lang.org/main/getting-started/tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- Dao, Fu, Ermon, Rudra & Ré (2022), "FlashAttention: fast and memory-efficient exact attention with IO-awareness". [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Apple, *Metal Shading Language Specification* (version 3.x), §4 (address spaces) and §5.2 (kernel attributes). [developer.apple.com/metal/Metal-Shading-Language-Specification.pdf](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- Kirk & Hwu, *Programming Massively Parallel Processors* (4th ed., 2022), ch. 5 (tiling) and ch. 10 (reductions).
- Ansel et al. (2024), "PyTorch 2: faster machine learning through dynamic Python bytecode transformation and graph compilation", *ASPLOS*: how `torch.compile` fuses. [doi:10.1145/3620665.3640366](https://doi.org/10.1145/3620665.3640366)

## connections

- ← builds on: traffic and the roofline (01), online softmax and FlashAttention (ch 18 · 06).
- → used in: mixed precision halves the bytes these kernels move (03); quantized matmuls dequantize inside the kernel (06); the capstone replaces your attention with PyTorch's fused kernel and lets `torch.compile` fuse the rest.
