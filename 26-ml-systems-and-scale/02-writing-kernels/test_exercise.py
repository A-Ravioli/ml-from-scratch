"""Checks for 26 · 02 writing kernels.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. kernels run on the
notebook's block-program interpreter, whose buffers count every element loaded and stored, so besides
correctness (against NumPy) the checks verify memory traffic exactly: a fused softmax touches each
element once each way, a tiled matmul loads what the tiling formula says, each program touches only
its own block, masks stop out-of-bounds lanes, and tile sizes are honoured (a recording buffer sees
the size of every load). the Metal kernel is compiled and run against torch.softmax when an Apple GPU
is present; everywhere else stage 4 checks the CPU fallback path.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")

HAS_MPS = torch.backends.mps.is_available()


def _buf(nb, a):
    return nb.Buffer(np.asarray(a, dtype=np.float32))


def _run(fn, *args, **kw):
    return bounded(fn, *args, budget=2_000_000, what=getattr(fn, "__name__", "kernel"), **kw)


class _Recording:
    """mixin for a Buffer subclass that records the size of every load and store."""

    def __init__(self, data):
        self.load_sizes, self.store_sizes = [], []
        self._l = self._s = 0
        super().__init__(data)

    @property
    def loads(self):
        return self._l

    @loads.setter
    def loads(self, v):
        if v > self._l:
            self.load_sizes.append(v - self._l)
        self._l = v

    @property
    def stores(self):
        return self._s

    @stores.setter
    def stores(self, v):
        if v > self._s:
            self.store_sizes.append(v - self._s)
        self._s = v


def _rec(nb, a):
    cls = type("Rec", (_Recording, nb.Buffer), {})
    return cls(np.asarray(a, dtype=np.float32))


def _np_softmax(x):
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- stage 1: the programming model


def test_stage1_add(nb):
    rng = np.random.default_rng(0)
    xs, ys = rng.normal(size=1000), rng.normal(size=1000)
    x, y, out = _buf(nb, xs), _buf(nb, ys), nb.Buffer.empty(1000)
    grid = _run(nb.add, x, y, out, 1000, block=128)
    assert tuple(grid) == (8,), f"1000 elements in blocks of 128 need cdiv(1000, 128) = 8 programs; grid {grid}"
    assert np.allclose(out.data, xs + ys, atol=1e-6), "out must hold x + y for all 1000 elements"
    assert (x.loads, y.loads, out.stores) == (1000, 1000, 1000), (
        f"each element of x and y is loaded once and each output stored once: (1000, 1000, 1000); got "
        f"{(x.loads, y.loads, out.stores)}. zero means you read buf.data directly; more means lanes past n weren't masked "
        "or blocks overlap")
    assert tuple(_run(nb.add, _buf(nb, xs[:300]), _buf(nb, ys[:300]), nb.Buffer.empty(300), 300)) == (3,), (
        "the default block is 128: 300 elements need 3 programs")
    assert tuple(_run(nb.add, _buf(nb, xs[:256]), _buf(nb, ys[:256]), nb.Buffer.empty(256), 256)) == (2,), (
        "256 elements in blocks of 128 need exactly 2 programs (no extra empty program when n divides evenly)")


def test_stage1_one_program(nb):
    x, y, out = _buf(nb, np.arange(10)), _buf(nb, 100 * np.ones(10)), nb.Buffer.empty(10)
    _run(nb.add_kernel, (1,), x, y, out, 10, 4)
    written = ~np.isnan(out.data)
    assert written.tolist() == [False] * 4 + [True] * 4 + [False] * 2, (
        f"program 1 with BLOCK=4 owns elements 4..7 and nothing else; it wrote {np.flatnonzero(written).tolist()}")
    assert np.allclose(out.data[4:8], [104, 105, 106, 107]) and x.loads == 4
    try:
        _run(nb.add_kernel, (2,), x, y, out, 10, 4)
    except IndexError as e:
        raise AssertionError(f"the last program covers offsets 8..11 but n = 10: mask lanes with offs < n ({e})") from None
    assert np.allclose(out.data[8:], [108, 109]) and x.loads == 6, "the last program loads and stores only 2 elements"


# ---------------------------------------------------------------- stage 2: fused softmax


def test_stage2_softmax_traffic(nb):
    rng = np.random.default_rng(1)
    xs = rng.normal(size=(5, 37)) * 3
    x, out = _buf(nb, xs), nb.Buffer.empty(5 * 37)
    grid, block = _run(nb.softmax, x, out, 5, 37)
    assert tuple(grid) == (5,) and block == 64, (
        f"one program per row, BLOCK = the next power of two >= 37 = 64; got grid {grid}, BLOCK {block}")
    got = out.data.reshape(5, 37)
    assert np.allclose(got, _np_softmax(xs), atol=1e-6), (
        "rows must equal softmax(x). if each row sums to a bit less than 1 and the values are slightly small, the "
        "padding lanes (37..63) were loaded as 0 and added exp(0 - max) to the sum: load them as -inf")
    assert (x.loads, out.stores) == (5 * 37, 5 * 37), (
        f"a fused softmax reads every element once and writes it once: ({5 * 37}, {5 * 37}); got {(x.loads, out.stores)}")


def test_stage2_blocks_and_stability(nb):
    for n, want in ((1, 1), (32, 32), (33, 64), (100, 128)):
        _, block = _run(nb.softmax, _buf(nb, np.zeros(2 * n)), nb.Buffer.empty(2 * n), 2, n)
        assert block == want, f"n_cols = {n}: BLOCK should be {want} (smallest power of two >= n_cols); got {block}"
    xs = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1000.0, -1000.0]])
    out = nb.Buffer.empty(6)
    _run(nb.softmax, _buf(nb, xs), out, 2, 3)
    assert np.all(np.isfinite(out.data)) and np.allclose(out.data.reshape(2, 3), _np_softmax(xs), atol=1e-6), (
        "softmax of [1000, 1001, 999] overflowed: subtract the row max before exponentiating")
    one = nb.Buffer.empty(3)
    _run(nb.softmax, _buf(nb, [[5.0], [-2.0], [0.0]]), one, 3, 1)
    assert np.allclose(one.data, 1.0), "a row of length 1 has softmax 1"


def test_stage2_stride_and_ownership(nb):
    rng = np.random.default_rng(2)
    full = np.full((4, 40), 1e30)
    full[:, :33] = rng.normal(size=(4, 33))
    x, out = _buf(nb, full), nb.Buffer(np.full(160, 7.0))
    _run(nb.softmax, x, out, 4, 33, stride=40)
    got = out.data.reshape(4, 40)
    assert np.allclose(got[:, :33], _np_softmax(full[:, :33]), atol=1e-6), (
        "with stride=40 row r starts at r * 40, and only its first n_cols = 33 entries belong to it")
    assert np.all(got[:, 33:] == 7.0), "the 7 padding columns of each row must not be written"
    assert x.loads == 4 * 33, f"only the 132 real elements are loaded, not the padding; got {x.loads}"
    x2, out2 = _buf(nb, rng.normal(size=(3, 8))), nb.Buffer.empty(24)
    _run(nb.softmax_kernel, (2,), x2, out2, 8, 8, 8)
    assert np.isnan(out2.data[:16]).all() and not np.isnan(out2.data[16:]).any(), "program 2 writes row 2 only"


# ---------------------------------------------------------------- stage 3: tiled matmul


def test_stage3_matmul(nb):
    rng = np.random.default_rng(3)
    M, N, K = 37, 29, 45
    A, B = rng.normal(size=(M, K)), rng.normal(size=(K, N))
    a, b, c = _buf(nb, A), _buf(nb, B), nb.Buffer.empty(M * N)
    grid = _run(nb.matmul, a, b, c, M, N, K)
    assert tuple(grid) == (3, 2), f"default 16x16 tiles: grid (cdiv(37, 16), cdiv(29, 16)) = (3, 2); got {grid}"
    assert np.allclose(c.data.reshape(M, N), (A.astype(np.float32) @ B.astype(np.float32)), atol=1e-4), (
        "c must equal a @ b, including the ragged last tiles (mask rows >= M, columns >= N and k >= K)")
    want = nb.matmul_traffic(M, N, K, 16, 16)
    assert (a.loads, b.loads) == tuple(want) == (M * K * 2, K * N * 3), (
        f"a is reloaded once per column of tiles (cdiv(N, 16) = 2 times) and b once per row of tiles (3 times): "
        f"want {(M * K * 2, K * N * 3)}, your kernel loaded {(a.loads, b.loads)}, your formula says {tuple(want)}")
    assert c.stores == M * N, f"each output element is stored exactly once; got {c.stores} stores"


def test_stage3_tiles_are_honoured(nb):
    rng = np.random.default_rng(4)
    M, N, K = 8, 16, 6
    A, B = rng.normal(size=(M, K)), rng.normal(size=(K, N))
    a, b, c = _rec(nb, A), _rec(nb, B), _rec(nb, np.full(M * N, np.nan))
    grid = _run(nb.matmul, a, b, c, M, N, K, block_m=4, block_n=8, block_k=2)
    assert tuple(grid) == (2, 2)
    assert np.allclose(c.data.reshape(M, N), A.astype(np.float32) @ B.astype(np.float32), atol=1e-4)
    assert a.load_sizes == [8] * 12, (
        f"with BLOCK_M=4, BLOCK_K=2 every load of a is one 4x2 tile, 3 per program (K=6), 4 programs: twelve loads of "
        f"8 elements; got {a.load_sizes}")
    assert b.load_sizes == [16] * 12, f"every load of b is one 2x8 tile: twelve loads of 16; got {b.load_sizes}"
    assert c.store_sizes == [32] * 4, f"each program stores its 4x8 tile once, at the end; got {c.store_sizes}"


def test_stage3_one_tile_and_formula(nb):
    rng = np.random.default_rng(5)
    A, B = rng.normal(size=(40, 10)), rng.normal(size=(10, 20))
    c = nb.Buffer.empty(800)
    _run(nb.matmul_kernel, (1, 0), _buf(nb, A), _buf(nb, B), c, 40, 20, 10, 16, 16, 4)
    written = ~np.isnan(c.data.reshape(40, 20))
    assert written[16:32, :16].all() and written.sum() == 256, "program (1, 0) writes rows 16..31, columns 0..15 only"
    assert tuple(nb.matmul_traffic(64, 64, 64, 8, 8)) == (64 * 64 * 8, 64 * 64 * 8)
    big = nb.matmul_traffic(64, 64, 64, 32, 32)
    assert tuple(big) == (64 * 64 * 2, 64 * 64 * 2), "4x larger tiles load each input 4x fewer times"
    assert tuple(nb.matmul_traffic(10, 30, 7, 4, 16)) == (10 * 7 * 2, 7 * 30 * 3), "block_m and block_n are separate"
    assert tuple(nb.matmul_traffic(1024, 1024, 1024, 16, 16)) == (67108864, 67108864), (
        "the lesson's worked example: 16x16 tiles of a 1024^3 matmul load each input 64 times")


# ---------------------------------------------------------------- stage 4: a real GPU kernel


def test_stage4_cpu_path(nb):
    src = nb.metal_softmax_source()
    assert isinstance(src, str) and "softmax_rows" in src, "metal_softmax_source returns Metal code defining softmax_rows"
    x = torch.randn(6, 300, generator=torch.Generator().manual_seed(0)) * 4
    y = _run(nb.metal_softmax, x)
    assert isinstance(y, torch.Tensor) and y.shape == x.shape and y.device == x.device and y.dtype == torch.float32, (
        "metal_softmax returns a float32 tensor of the input's shape, on the input's device")
    assert torch.allclose(y, torch.softmax(x, -1), atol=1e-6), "on the CPU, metal_softmax runs your stage-2 kernel"
    from torch.overrides import TorchFunctionMode

    class _Watch(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.names = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.names.append(getattr(func, "__name__", str(func)))
            return func(*args, **(kwargs or {}))

    with _Watch() as w:
        _run(nb.metal_softmax, x)
    used = {n for n in w.names if n in ("softmax", "log_softmax", "exp", "amax", "max", "sum", "logsumexp", "div",
                                         "__truediv__", "sub", "__sub__")}
    assert not used, (
        f"on the CPU, metal_softmax must compute the softmax with your stage-2 interpreter kernel, not with torch ops; it "
        f"called {sorted(used)}")


@pytest.mark.skipif(not HAS_MPS, reason="needs an Apple GPU (mps)")
def test_stage4_metal_kernel(nb):
    try:
        lib = torch.mps.compile_shader(nb.metal_softmax_source())
    except Exception as e:
        raise AssertionError(f"your Metal source doesn't compile: {str(e)[:800]}") from None
    g = torch.Generator().manual_seed(1)
    for n_cols in (1, 7, 100, 256, 257, 1000, 5000):
        x = (torch.randn(9, n_cols, generator=g) * 6).to("mps")
        out = torch.full_like(x, float("nan"))
        lib.softmax_rows(out, x, n_cols, threads=9 * 256, group_size=256)
        torch.mps.synchronize()
        err = (out.cpu() - torch.softmax(x.cpu(), -1)).abs().max().item()
        assert err < 1e-5, (
            f"softmax_rows with n_cols = {n_cols} (one 256-thread threadgroup per row) is off by {err:.2e}. if small "
            "n_cols fail, threads past the end of the row must not contribute to the max or the sum; if large ones "
            "fail, each thread must loop over columns tid, tid + 256, ...; NaNs often mean a missing barrier")
    big = torch.tensor([[1000.0, 1001.0, 999.0, 1000.5]], device="mps")
    out = torch.empty_like(big)
    lib.softmax_rows(out, big, 4, threads=256, group_size=256)
    assert torch.isfinite(out.cpu()).all(), "subtract the row max before exp: exp(1000) overflows float32"
    x = torch.randn(33, 777, generator=g).to("mps")
    y = nb.metal_softmax(x)
    assert y.device.type == "mps" and torch.allclose(y.cpu(), torch.softmax(x.cpu(), -1), atol=1e-5), (
        "metal_softmax on an mps tensor returns the softmax on mps")
