import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # symmetry & equivariance · build (reference solution)

    this chapter is spec-driven (L4): each stage's exact specification is in the lesson's
    **spec** box next to it, the checks below enforce it, and how you structure the code is up
    to you. you'll build the tools every later topic in this chapter tests itself with: random
    group elements, an equivariance tester, a translation-equivariant convolution, and group
    averaging, the bluntest way to make any function invariant.

    | stage | you build |
    |---|---|
    | 1 | `random_rotation`, `random_orthogonal`, `random_permutation`, `permutation_matrix`: group elements |
    | 2 | `equivariance_error`: the tester → `mlfs.gnn` |
    | 3 | `translate`, `circular_conv2d`: translations and the layer that commutes with them |
    | 4 | `c4_rotations`, `group_average`: invariance by averaging over a group |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def random_rotation(dim, *, rng=None):
    """stage 1: a Haar-random rotation in SO(dim): float64 tensor (dim, dim), R^T R = I, det R = +1."""
    q = random_orthogonal(dim, rng=rng)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@app.function
def random_orthogonal(dim, *, rng=None):
    """stage 1: a Haar-random orthogonal matrix in O(dim) (rotations and reflections): float64 tensor (dim, dim)."""
    g = as_rng(rng).normal(size=(dim, dim))
    q, r = np.linalg.qr(g)
    q = q * np.sign(np.diag(r))  # fixes QR's sign ambiguity, which would otherwise bias the distribution
    return torch.tensor(q, dtype=torch.float64)


@app.function
def random_permutation(n, *, rng=None):
    """stage 1: a uniformly random permutation of 0..n-1, an int64 tensor (n,)."""
    return torch.tensor(as_rng(rng).permutation(n), dtype=torch.int64)


@app.function
def permutation_matrix(perm):
    """stage 1: the float64 (n, n) matrix P with P @ x == x[perm] for every x of shape (n, ...)."""
    perm = torch.as_tensor(perm, dtype=torch.int64)
    n = len(perm)
    p = torch.zeros(n, n, dtype=torch.float64)
    p[torch.arange(n), perm] = 1.0
    return p


@app.cell(hide_code=True)
def _():
    checks.run(1, random_rotation=random_rotation, random_orthogonal=random_orthogonal,
               random_permutation=random_permutation, permutation_matrix=permutation_matrix)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def equivariance_error(f, x, transform_in, transform_out):
    """stage 2: max |f(transform_in(x)) - transform_out(f(x))| as a Python float (transform_out None = identity)."""
    with torch.no_grad():
        y = f(x)
        a = f(transform_in(x))
        b = y if transform_out is None else transform_out(y)
        return float((torch.as_tensor(a) - torch.as_tensor(b)).abs().max())


@app.cell(hide_code=True)
def _():
    checks.run(2, equivariance_error=equivariance_error, random_permutation=random_permutation)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def translate(x, shift):
    """stage 3: circularly shift the last two axes of x by shift = (dy, dx): out[..., i, j] = x[..., i - dy, j - dx]."""
    return torch.roll(x, shifts=(int(shift[0]), int(shift[1])), dims=(-2, -1))


@app.function
def circular_conv2d(x, weight, bias=None, *, stride=1):
    """stage 3: cross-correlation of x (B, C, H, W) with weight (O, C, k, k), k odd, circular padding k // 2, then stride."""
    k = weight.shape[-1]
    if k % 2 == 0:
        raise ValueError(f"circular_conv2d needs an odd kernel size, got {k}")
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="circular") if p else x
    return F.conv2d(xp, weight, bias, stride=stride)


@app.cell(hide_code=True)
def _():
    checks.run(3, translate=translate, circular_conv2d=circular_conv2d, equivariance_error=equivariance_error)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def c4_rotations():
    """stage 4: the group C4 as a list of 4 functions; element k rotates the last two axes by k * 90 degrees (torch.rot90)."""
    return [lambda x, k=k: torch.rot90(x, k, dims=(-2, -1)) for k in range(4)]


@app.function
def group_average(f, x, group):
    """stage 4: the mean over g in group (a list of functions) of f(g(x)): invariant if group is a finite group."""
    outs = [f(g(x)) for g in group]
    return torch.stack(outs).mean(dim=0)


@app.cell(hide_code=True)
def _():
    checks.run(4, c4_rotations=c4_rotations, group_average=group_average, translate=translate,
               equivariance_error=equivariance_error)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.gnn import equivariance_error` gives you the tester; every layer in this chapter
    is checked with it.
    """)
    return


if __name__ == "__main__":
    app.run()
