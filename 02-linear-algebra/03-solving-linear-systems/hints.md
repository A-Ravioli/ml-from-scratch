# solving linear systems · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

start from a copy, `x = np.array(b, dtype=float)`, and overwrite `x[i]` in place as you go. by the time you reach row `i`, the entries you need (`x[:i]` for lower, `x[i+1:]` for upper) are already final.

</details>

<details><summary>hint 2: the approach</summary>

loop over rows in order (`range(n)` for lower, `range(n - 1, -1, -1)` for upper). for each row, subtract the dot product of the already-solved part of the row with the already-solved part of `x`, then divide by the diagonal unless `unit_diagonal`. if `x` is `(n, k)`, then `T[i, :i] @ x[:i]` is a length-`k` row, so the same line handles many right-hand sides.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for i in rows:
    if lower:
        x[i] -= T[i, :i] @ x[:i]
    else:
        x[i] -= T[i, i + 1:] @ x[i + 1:]
    if not unit_diagonal:
        if T[i, i] == 0:
            raise np.linalg.LinAlgError("zero pivot")
        x[i] /= T[i, i]
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

keep three things: a working copy `U` of `A`, an `L` that starts as `np.eye(n)`, and an integer array `perm = np.arange(n)` recording which original row sits where. build `P = np.eye(n)[perm]` at the end.

</details>

<details><summary>hint 2: the approach</summary>

for `k` in `range(n - 1)`:
1. `p = k + np.argmax(np.abs(U[k:, k]))`
2. if `p != k`, swap rows `k` and `p` of `U` (columns `k:` are enough), of `L` (columns `:k` only, the multipliers so far), and of `perm`
3. if the pivot `U[k, k]` is zero, skip to the next column
4. store multipliers `L[k+1:, k] = U[k+1:, k] / U[k, k]`
5. update `U[k+1:, k:] -= np.outer(L[k+1:, k], U[k, k:])`

fancy-index swaps like `U[[k, p]] = U[[p, k]]` work because the right side makes a copy.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if p != k:
    U[[k, p], k:] = U[[p, k], k:]
    L[[k, p], :k] = L[[p, k], :k]
    perm[[k, p]] = perm[[p, k]]
if U[k, k] == 0:
    continue
L[k + 1:, k] = U[k + 1:, k] / U[k, k]
U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
U[k + 1:, k] = 0.0
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

$LUx = Pb$. which of the two triangular factors has an implicit unit diagonal?

</details>

<details><summary>hint 2: the approach</summary>

unpack `P, L, U = factors`, compute `y` from `L y = P b` (lower, unit diagonal), then `x` from `U x = y` (upper). `P @ b` works for both `(n,)` and `(n, k)` shapes.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
y = solve_triangular(L, P @ b, lower=True, unit_diagonal=True)
return solve_triangular(U, y, lower=False)
```

and `solve` is one line: `return lu_solve(lu(A), b)`.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

$\det A = \det P \cdot \prod_i u_{ii}$, and $\det P$ is $+1$ or $-1$. the only real work is the sign.

</details>

<details><summary>hint 2: the approach</summary>

recover the permutation with `perm = P.argmax(axis=1)`. a permutation splits into cycles. a cycle of length $c$ is $c - 1$ swaps, so the sign is $(-1)^{\sum (c-1)}$. follow each cycle from an unvisited start: `j = perm[j]` until you're back.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
sign, seen = 1.0, np.zeros(n, dtype=bool)
for start in range(n):
    length, j = 0, start
    while not seen[j]:
        seen[j] = True
        j = perm[j]
        length += 1
    if length > 0 and length % 2 == 0:   # even-length cycles are odd permutations
        sign = -sign
```

(an alternative: count swaps inside `lu` itself. but `det` only gets `P` here, so cycles it is.)

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

work column by column. column `j` of `L` only needs columns `0..j-1`, which you already have.

</details>

<details><summary>hint 2: the approach</summary>

first reject non-symmetric input (`np.allclose(A, A.T)`). then for each `j`: `d = A[j, j] - L[j, :j] @ L[j, :j]`; if `d <= 0`, raise `LinAlgError`; `L[j, j] = sqrt(d)`; then fill the rest of the column with `(A[j+1:, j] - L[j+1:, :j] @ L[j, :j]) / L[j, j]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

`cho_solve` is two triangular solves:

```python
y = solve_triangular(L, b, lower=True)
return solve_triangular(L.T, y, lower=False)
```

</details>
