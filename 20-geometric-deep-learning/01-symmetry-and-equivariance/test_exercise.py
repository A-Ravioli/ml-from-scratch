"""Checks for exercise.py (symmetry & equivariance).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. group elements are
checked for their defining properties (orthogonality, determinant, Haar-uniformity in
distribution, permutation action); the tester on functions whose errors are known exactly; the
convolution against a direct loop and for exact shift-equivariance; group averaging for exact
invariance of an arbitrary function.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _as_np(t):
    return t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


# ---------------------------------------------------------------- stage 1: group elements


@pytest.mark.parametrize("dim", [2, 3, 5])
def test_stage1_rotations_are_rotations(nb, dim):
    for seed in range(5):
        r = nb.random_rotation(dim, rng=seed)
        assert isinstance(r, torch.Tensor) and r.dtype == torch.float64 and r.shape == (dim, dim), (
            f"random_rotation({dim}) returns a float64 tensor of shape ({dim}, {dim}); got "
            f"{type(r).__name__} {getattr(r, 'dtype', None)} {tuple(getattr(r, 'shape', ()))}")
        assert torch.allclose(r.T @ r, torch.eye(dim, dtype=torch.float64), atol=1e-12), (
            "a rotation is orthogonal: R^T R = I. did you take Q from a QR decomposition?")
        det = float(torch.linalg.det(r))
        assert abs(det - 1.0) < 1e-10, (
            f"a rotation has determinant +1; got {det:.3f}. half of all orthogonal matrices are reflections "
            "(det -1): flip the sign of one column to turn a reflection into a rotation.")
    assert torch.equal(nb.random_rotation(dim, rng=7), nb.random_rotation(dim, rng=7)), (
        "the same rng seed gives the same rotation (use as_rng(rng))")
    assert not torch.allclose(nb.random_rotation(dim, rng=7), nb.random_rotation(dim, rng=8))


def test_stage1_rotations_are_haar_uniform(nb):
    rng = np.random.default_rng(0)
    rs = np.stack([_as_np(nb.random_rotation(3, rng=rng)) for _ in range(3000)])
    mean = rs.mean(axis=0)
    assert np.abs(mean).max() < 0.06, (
        "averaged over many Haar-random rotations every entry should be about 0 (R and R composed with a fixed "
        f"rotation are equally likely). got mean\n{np.round(mean, 3)}\nthe usual cause is taking Q from "
        "np.linalg.qr without fixing signs: QR returns R with a sign convention on its diagonal, which biases Q. "
        "multiply each column of Q by the sign of the matching diagonal entry of R.")
    sq = (rs ** 2).mean(axis=0)
    assert np.abs(sq - 1 / 3).max() < 0.03, (
        f"each entry of a Haar rotation in 3-D has E[R_ij^2] = 1/3; got\n{np.round(sq, 3)}")


def test_stage1_orthogonal_includes_reflections(nb):
    dets = [float(torch.linalg.det(nb.random_orthogonal(3, rng=s))) for s in range(40)]
    for s in range(3):
        q = nb.random_orthogonal(4, rng=s)
        assert q.dtype == torch.float64 and torch.allclose(q.T @ q, torch.eye(4, dtype=torch.float64), atol=1e-12)
    assert all(abs(abs(d) - 1) < 1e-10 for d in dets), "an orthogonal matrix has determinant +1 or -1"
    n_ref = sum(d < 0 for d in dets)
    assert 8 <= n_ref <= 32, (
        f"random_orthogonal draws from all of O(n): about half should be reflections (det -1); got {n_ref}/40")


def test_stage1_permutations(nb):
    p = nb.random_permutation(10, rng=0)
    assert isinstance(p, torch.Tensor) and p.dtype == torch.int64 and p.shape == (10,), (
        "random_permutation(n) returns an int64 tensor of shape (n,)")
    assert sorted(p.tolist()) == list(range(10)), f"a permutation contains each of 0..n-1 once; got {p.tolist()}"
    firsts = [int(nb.random_permutation(4, rng=s)[0]) for s in range(400)]
    counts = np.bincount(firsts, minlength=4)
    assert counts.min() > 60, f"each element should come first about equally often; counts {counts.tolist()}"
    perm = torch.tensor([2, 0, 3, 1])
    P = nb.permutation_matrix(perm)
    assert P.dtype == torch.float64 and P.shape == (4, 4)
    x = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    assert torch.equal(P @ x, x[perm]), (
        f"permutation_matrix(perm) @ x must equal x[perm] (row i of the result is row perm[i] of x): "
        f"P[i, perm[i]] = 1. got P =\n{P}")
    assert torch.equal(P.T @ P, torch.eye(4, dtype=torch.float64))


# ---------------------------------------------------------------- stage 2: equivariance_error


def test_stage2_known_errors(nb):
    x = torch.tensor([[1.0, 2.0], [3.0, 5.0], [-4.0, 0.5]], dtype=torch.float64)
    perm = torch.tensor([1, 2, 0])

    def pin(t):
        return t[perm]

    e = nb.equivariance_error(lambda t: t.sum(0), x, pin, lambda y: y)
    assert isinstance(e, float), f"equivariance_error returns a Python float; got {type(e).__name__}"
    assert e < 1e-12, f"a sum over the set axis is permutation-invariant: error 0, got {e}"
    e = nb.equivariance_error(lambda t: t ** 2, x, pin, pin)
    assert e < 1e-12, f"an elementwise function is permutation-equivariant: error 0, got {e}"
    # f(x) = x - x[0]: f(x[perm]) - f(x)[perm] = x[perm][0] - x[0] broadcast: x[1] - x[0] = (2, 3)
    e = nb.equivariance_error(lambda t: t - t[0], x, pin, pin)
    assert abs(e - 3.0) < 1e-12, (
        f"for f(x) = x - x[0] and this permutation, the largest |f(perm x) - perm f(x)| is |5 - 2| = 3; got {e}. "
        "the error is the max absolute difference, f applied to the transformed input versus the transformed output")
    e = nb.equivariance_error(lambda t: t.sum(0), x, pin, None)
    assert e < 1e-12, "transform_out=None means the identity (an invariance test)"


def test_stage2_no_side_effects_and_modules(nb):
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 4).double()
    x = torch.randn(6, 3, dtype=torch.float64)
    x0 = x.clone()
    perm = nb.random_permutation(6, rng=1)
    e = nb.equivariance_error(lin, x, lambda t: t[perm], lambda y: y[perm])
    assert torch.equal(x, x0), "equivariance_error mustn't modify x"
    assert e < 1e-12, f"a Linear layer applied row by row is permutation-equivariant; got error {e}"
    e2 = nb.equivariance_error(lin, x, lambda t: t[perm], None)
    assert e2 > 1e-3, "...but it's not invariant: rows of the output move with the rows of the input"


# ---------------------------------------------------------------- stage 3: translation, convolution


def _conv_loop(x, w, b, stride):
    x, w = _as_np(x), _as_np(w)
    B, C, H, W = x.shape
    O, _, k, _ = w.shape
    p = k // 2
    out = np.zeros((B, O, (H + stride - 1) // stride, (W + stride - 1) // stride))
    for bi in range(B):
        for o in range(O):
            for oi, i in enumerate(range(0, H, stride)):
                for oj, j in enumerate(range(0, W, stride)):
                    s = 0.0
                    for c in range(C):
                        for di in range(k):
                            for dj in range(k):
                                s += w[o, c, di, dj] * x[bi, c, (i + di - p) % H, (j + dj - p) % W]
                    out[bi, o, oi, oj] = s + (0.0 if b is None else float(b[o]))
    return out


def test_stage3_translate(nb):
    x = torch.arange(2 * 4 * 5, dtype=torch.float64).reshape(2, 4, 5)
    got = nb.translate(x, (1, -2))
    want = np.roll(_as_np(x), shift=(1, -2), axis=(-2, -1))
    assert np.array_equal(_as_np(got), want), (
        "translate(x, (dy, dx)) moves content down by dy and right by dx, wrapping around: out[..., i, j] = "
        "x[..., i - dy, j - dx]")
    assert torch.equal(nb.translate(nb.translate(x, (3, 2)), (-3, -2)), x), "translations compose and invert"


@pytest.mark.parametrize("stride", [1, 2])
def test_stage3_conv_matches_loop(nb, stride):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 2, 6, 6, generator=g, dtype=torch.float64)
    w = torch.randn(3, 2, 3, 3, generator=g, dtype=torch.float64)
    b = torch.randn(3, generator=g, dtype=torch.float64)
    got = nb.circular_conv2d(x, w, b, stride=stride)
    want = _conv_loop(x, w, b, stride)
    assert tuple(got.shape) == want.shape, f"output shape should be {want.shape}; got {tuple(got.shape)}"
    assert np.allclose(_as_np(got), want, atol=1e-10), (
        "circular_conv2d is a cross-correlation (no kernel flip, like F.conv2d) of the circularly padded input: "
        "pad k // 2 on every side with wrap-around, then slide the kernel with the given stride")
    one = nb.circular_conv2d(x, torch.ones(1, 2, 1, 1, dtype=torch.float64))
    assert torch.allclose(one[:, 0], x.sum(1)), "a 1x1 kernel needs no padding"


def test_stage3_exact_shift_equivariance(nb):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 3, 8, 8, generator=g, dtype=torch.float64)
    w = torch.randn(4, 3, 5, 5, generator=g, dtype=torch.float64)

    def conv(t, s=1):
        return nb.circular_conv2d(t, w, stride=s)

    for shift in [(1, 0), (0, 3), (5, 7)]:
        e = nb.equivariance_error(conv, x, lambda t: nb.translate(t, shift), lambda y: nb.translate(y, shift))
        assert e < 1e-10, f"a stride-1 circular convolution commutes with every shift; error {e} for shift {shift}"
    e2 = nb.equivariance_error(lambda t: conv(t, 2), x, lambda t: nb.translate(t, (2, 4)),
                               lambda y: nb.translate(y, (1, 2)))
    assert e2 < 1e-10, f"with stride 2, shifting the input by (2, 4) shifts the output by exactly (1, 2); error {e2}"
    e3 = nb.equivariance_error(lambda t: conv(t, 2), x, lambda t: nb.translate(t, (1, 0)), lambda y: y)
    assert e3 > 1e-3, "an odd shift moves a stride-2 output by half a pixel: no exact relation holds"


# ---------------------------------------------------------------- stage 4: group averaging


def test_stage4_c4_is_a_group(nb):
    rots = nb.c4_rotations()
    assert len(rots) == 4, "C4 has 4 elements: rotations by 0, 90, 180, 270 degrees"
    x = torch.arange(2 * 3 * 3, dtype=torch.float64).reshape(2, 3, 3)
    assert torch.equal(rots[0](x), x), "element 0 is the identity"
    for k in range(4):
        assert torch.equal(rots[k](x), torch.rot90(x, k, dims=(-2, -1))), (
            f"element {k} rotates the last two axes by {90 * k} degrees, like torch.rot90(x, {k}, dims=(-2, -1)); "
            "check the default argument trick if all your lambdas behave like the last one")
    y = x
    for _ in range(4):
        y = rots[1](y)
    assert torch.equal(y, x), "four quarter turns are the identity"


def test_stage4_group_average_is_exactly_invariant(nb):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(25, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)).double()
    x = torch.randn(3, 1, 5, 5, dtype=torch.float64)
    rots = nb.c4_rotations()

    def avg(t):
        return nb.group_average(net, t, rots)

    assert tuple(avg(x).shape) == (3, 2), f"group_average keeps f's output shape; got {tuple(avg(x).shape)}"
    assert nb.equivariance_error(net, x, rots[1], None) > 1e-3, "(the raw network isn't rotation-invariant)"
    for k in range(4):
        e = nb.equivariance_error(avg, x, rots[k], None)
        assert e < 1e-12, f"the C4-average must be exactly invariant to every rotation in C4; error {e} for k={k}"
    want = torch.stack([net(torch.rot90(x, k, dims=(-2, -1))) for k in range(4)]).mean(0)
    assert torch.allclose(avg(x), want, atol=1e-12), "the average is the plain mean of f(g x) over the 4 elements"
    shifts = [lambda t, s=(i, j): nb.translate(t, s) for i in range(5) for j in range(5)]
    e = nb.equivariance_error(lambda t: nb.group_average(net, t, shifts), x, lambda t: nb.translate(t, (2, 3)), None)
    assert e < 1e-12, "averaging over all 25 circular shifts gives exact shift invariance"
