# SVD & low-rank approximation · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

start with `W = np.array(A, dtype=float)` and `V = np.eye(n)`. an outer loop runs sweeps (at most `max_sweeps`), and each sweep is a double loop over pairs `p < q`. keep a flag `rotated = False` per sweep, and stop when a sweep ends with it still `False`.

</details>

<details><summary>hint 2: the approach</summary>

for each pair: `alpha = W[:,p] @ W[:,p]`, `beta = W[:,q] @ W[:,q]`, `gamma = W[:,p] @ W[:,q]`. skip the pair if `min(alpha, beta) <= (eps * norm(A))**2` or `abs(gamma) <= n * eps * sqrt(alpha * beta)` (with `eps = np.finfo(float).eps`). otherwise compute `zeta = (beta - alpha) / (2 * gamma)`, `t = sign(zeta) / (abs(zeta) + np.hypot(1, zeta))` (sign(0) = +1), `c = 1 / np.hypot(1, t)`, `s = t * c`, set `rotated = True`, and rotate columns `p, q` of **both** `W` and `V`: `new_p = c*old_p - s*old_q`, `new_q = s*old_p + c*old_q`.

after the sweeps: `s = norm(W, axis=0)`, `order = np.argsort(-s)`, permute `s`, `W[:, order]` and `V[:, order]`, then `U = W / s`.

</details>

<details><summary>hint 3: nearly the answer</summary>

the rotation, written with tuple assignment so the right-hand side uses the *old* columns:

```python
wp, wq = W[:, p], W[:, q]
...
W[:, p], W[:, q] = c * wp - s * wq, s * wp + c * wq
vp, vq = V[:, p].copy(), V[:, q].copy()
V[:, p], V[:, q] = c * vp - s * vq, s * vp + c * vq
```

return `U, s, V.T`. (for the zero columns that stage 2 will throw at it, divide only where `s > 0` and leave the other columns of `U` as zeros.)

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

handle the shapes first. if `m < n`, call `svd(A.T)` and return `(Vt.T, s, U.T)`. otherwise `Q, R = qr(A)` gives a square `R`, and stage 1 does the hard part on `R`.

</details>

<details><summary>hint 2: the approach</summary>

`_, _, Vt = jacobi_svd_square(R)` gives `V = Vt.T` with columns already sorted by decreasing `||R v||`. form `W = R @ Vt.T`, factor it with `Qw, Rw = qr(W)`, and read `s = abs(diag(Rw))` and `Uw = Qw * sign(diag(Rw))` (treat sign 0 as +1). finally `U = Q @ Uw`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
Q, R = qr(A)
_, _, Vt = jacobi_svd_square(R)
Qw, Rw = qr(R @ Vt.T)
d = np.diag(Rw)
s = np.abs(d)
Uw = Qw * np.where(d < 0, -1.0, 1.0)
order = np.argsort(-s, kind="stable")   # near-ties on the diagonal can swap
s, Uw, Vt = s[order], Uw[:, order], Vt[order]
return Q @ Uw, s, Vt
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

the only change is where $Q$ comes from: `qr(A, mode="complete")` gives an $m \times m$ orthogonal `Q`.

</details>

<details><summary>hint 2: the approach</summary>

with the complete `Q, R`, the square part is `R[:n]`. run stage 2's steps on it. at the end, `U = Q.copy()` and replace its first `n` columns by `Q[:, :n] @ Uw`. the wide case is handled by the transpose trick: `svd(A.T, full_matrices=True)` and swap.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
Q, R = qr(A, mode="complete" if full_matrices else "reduced")
R = R[:n]
...                       # exactly as in stage 2
if full_matrices:
    U = Q.copy()
    U[:, :n] = Q[:, :n] @ Uw
else:
    U = Q @ Uw
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

all three are short uses of `svd`. `truncated_svd` keeps the first `k` of everything, `low_rank_approx` multiplies them back together, and `pinv` inverts the singular values it trusts.

</details>

<details><summary>hint 2: the approach</summary>

`U, s, Vt = svd(A)`, check `1 <= k <= len(s)`, return `U[:, :k], s[:k], Vt[:k]`. for the approximation, `(U * s) @ Vt` scales the columns of `U` by `s` without building `diag(s)`. for `pinv`: `cutoff = rcond * s.max()` (careful: `s` can be all zero), `s_inv[s > cutoff] = 1 / s[s > cutoff]`, then `(Vt.T * s_inv) @ U.T`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
U, s, Vt = svd(A)
cutoff = rcond * (s.max() if s.size else 0.0)
s_inv = np.zeros_like(s)
keep = s > cutoff
s_inv[keep] = 1.0 / s[keep]
return (Vt.T * s_inv) @ U.T
```

the cutoff must be *relative* (`rcond * s.max()`): a matrix scaled by $10^6$ has the same pseudoinverse structure, just scaled by $10^{-6}$.

</details>
