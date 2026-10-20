# convex sets & functions · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

for `in_polyhedron`, compute all $m$ constraint values for all $k$ points at once: a `(k, m)` matrix. a point is inside when its whole row satisfies the inequality. for `set_convexity_violation`, first find which rows are members of the set, then loop over pairs of *members* and over `ts`.

</details>

<details><summary>hint 2: the approach</summary>

`X @ A.T` has shape `(k, m)`: entry `[p, i]` is $a_i^\top x_p$. compare with `b + tol` (broadcast along rows) and reduce with `np.all(..., axis=1)`. for the search: `members = [i for i in range(len(P)) if contains(P[i])]`, then `for a, i in enumerate(members): for j in members[a + 1:]: for t in ts:` and test `contains(t * P[i] + (1 - t) * P[j])`. return the original indices `i, j` (not positions in `members`).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# in_polyhedron
X = np.atleast_2d(np.asarray(X, dtype=float))
return np.all(X @ A.T <= b + tol, axis=1)

# set_convexity_violation
for a, i in enumerate(members):
    for j in members[a + 1:]:
        for t in ts:
            if not contains(t * P[i] + (1 - t) * P[j]):
                return (i, j, float(t))
return None
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`jensen_gap` is one line: chord minus function. for the minimum, precompute `F = [f(p) for p in P]` so the chord ends are free, and only call `f` for the interior point $tx_i + (1 - t)x_j$.

</details>

<details><summary>hint 2: the approach</summary>

keep a running `best = np.inf` and `where = None`. loop `i`, then `j > i`, then `t in ts`; compute `gap = t * F[i] + (1 - t) * F[j] - f(t * P[i] + (1 - t) * P[j])`. update only when `gap < best` (strictly), so the first location wins ties.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
P = np.asarray(points, dtype=float)
F = [float(f(p)) for p in P]
best, where = np.inf, None
for i in range(len(P)):
    for j in range(i + 1, len(P)):
        for t in ts:
            gap = t * F[i] + (1 - t) * F[j] - float(f(t * P[i] + (1 - t) * P[j]))
            if gap < best:
                best, where = gap, (i, j, float(t))
return float(best), where
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

stack the values `F` (shape `(k,)`) and the gradients `G` (shape `(k, n)`). the gap matrix has entry `[i, j] = F[j] - F[i] - G[i] @ (P[j] - P[i])`. expand the last term: `G[i] @ P[j] - G[i] @ P[i]`.

</details>

<details><summary>hint 2: the approach</summary>

`G @ P.T` is the `(k, k)` matrix of `G[i] @ P[j]`, and `np.sum(G * P, axis=1)` is the vector of `G[i] @ P[i]`. so `gaps = F[None, :] - F[:, None] - (G @ P.T - np.sum(G * P, axis=1)[:, None])`. set the diagonal to `np.inf` (`np.fill_diagonal`) and find the argmin with `np.unravel_index(np.argmin(gaps), gaps.shape)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
gaps = F[None, :] - F[:, None] - (G @ P.T - np.sum(G * P, axis=1)[:, None])
np.fill_diagonal(gaps, np.inf)
i, j = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
return float(gaps[i, j]), (int(i), int(j))
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`np.linalg.eigvalsh` returns the eigenvalues of a symmetric matrix in ascending order, so the smallest is `[0]`. for `check_convexity`, reuse `min_jensen_gap`, then decide with a tolerance scaled by the size of $f$.

</details>

<details><summary>hint 2: the approach</summary>

`lams = [eigvalsh(numerical_hessian(f, p, h=h))[0] for p in P]`, then the argmin. for the verdict: `gap, (i, j, t) = min_jensen_gap(f, P, ts)`, `scale = 1 + max(abs(f(p)) for p in P)`, and a violation means `gap < -tol * scale`. return `P[i].copy()` and `P[j].copy()`, not views into the caller's array.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
P = np.asarray(points, dtype=float)
gap, where = min_jensen_gap(f, P, ts)
scale = 1.0 + max(abs(float(f(p))) for p in P)
if gap < -tol * scale:
    i, j, t = where
    return False, (P[i].copy(), P[j].copy(), float(t))
return True, None
```

</details>
