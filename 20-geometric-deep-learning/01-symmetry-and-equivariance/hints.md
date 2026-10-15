# symmetry & equivariance · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: there's one hint per stage, and it points at the idea, not the code.

## stage 1

<details><summary>hint</summary>

`g = as_rng(rng).normal(size=(dim, dim))`, `q, r = np.linalg.qr(g)`, then `q * np.sign(np.diag(r))` multiplies column $j$ by $\operatorname{sign}(r_{jj})$ (broadcasting along rows). that's `random_orthogonal`. for `random_rotation`, if `torch.linalg.det(q) < 0`, negate one column: a reflection composed with a fixed reflection is a rotation, and the distribution stays Haar. for `permutation_matrix`, set `P[torch.arange(n), perm] = 1`; check it on a tiny example: row $i$ of $Px$ must be row `perm[i]` of $x$.

</details>

## stage 2

<details><summary>hint</summary>

inside `with torch.no_grad():` compute `a = f(transform_in(x))` and `b = f(x)` (passed through `transform_out` unless it's `None`), and return `float((a - b).abs().max())`. don't transform `x` in place: `transform_in` must return a new tensor, and you mustn't write into `x` yourself.

</details>

## stage 3

<details><summary>hint</summary>

`translate` is one call to `torch.roll` with `dims=(-2, -1)`. for `circular_conv2d`, pad first with `F.pad(x, (p, p, p, p), mode="circular")` (the tuple is left, right, top, bottom), then call `F.conv2d(xp, weight, bias, stride=stride)` with no further padding. with padding $p = k/2$ rounded down, a stride-1 output has the input's size, and stride $s$ keeps rows and columns $0, s, 2s, \dots$, which is what the loop oracle in the checks computes.

</details>

## stage 4

<details><summary>hint</summary>

`[lambda x, k=k: torch.rot90(x, k, dims=(-2, -1)) for k in range(4)]`: the `k=k` default binds each lambda's own `k` (without it, all four see the loop's final value). `group_average` is `torch.stack([f(g(x)) for g in group]).mean(dim=0)`: stacking keeps the output shape and the autograd graph.

</details>
