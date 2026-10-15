# linear maps & matmul · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

read the shapes first: `m, k = A.shape` and `n = B.shape[1]`. the result is `np.zeros((m, n))`, which is float even if the inputs are integers.

</details>

<details><summary>hint 2: the approach</summary>

loops over `i` (rows of `C`), `j` (columns of `C`) and, innermost, `p` (the shared index). accumulate `A[i, p] * B[p, j]` into a running total that starts at 0 for each `(i, j)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for i in range(m):
    for j in range(n):
        total = 0.0
        for p in range(k):
            total += A[i, p] * B[p, j]
        C[i, j] = total
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`A[:, p]` is a whole column and `B[p, :]` a whole row. scaling one by a number and adding it to a column or row of `C` is a single vectorized line.

</details>

<details><summary>hint 2: the approach</summary>

columns: for each `j` and each `p`, `C[:, j] += B[p, j] * A[:, p]`. rows: for each `i` and each `p`, `C[i, :] += A[i, p] * B[p, :]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

the inner `p` loop of the column version is itself a matrix-vector product, `C[:, j] = A @ B[:, j]`. that's the column view in one line, but write the loop here to see the combination explicitly.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

one loop, over the inner dimension `k`.

</details>

<details><summary>hint 2: the approach</summary>

start with `C = np.zeros((m, n))`, then for each `p`, `C += np.outer(A[:, p], B[p, :])`.

</details>

<details><summary>hint 3: nearly the answer</summary>

`np.outer(u, v)` builds the matrix with entries `u[i] * v[j]`, which is what `u[:, None] * v[None, :]` gives by broadcasting.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

the rows of `np.eye(n)` are the standard basis vectors $e_1, \dots, e_n$.

</details>

<details><summary>hint 2: the approach</summary>

`matrix_of`: compute `f(e)` for each basis vector and stack the results as **columns** (`np.column_stack`). `is_linear`: with `rng = np.random.default_rng(rng)`, repeat `trials` times: draw `x, y` (normal, length `n`) and scalars `a, b` (normal), compare `f(a*x + b*y)` with `a*f(x) + b*f(y)`, and return False on the first mismatch beyond `tol` (relative to the size of the values).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
lhs = np.asarray(f(a * x + b * y), dtype=float)
rhs = a * np.asarray(f(x), dtype=float) + b * np.asarray(f(y), dtype=float)
scale = max(1.0, np.abs(lhs).max(), np.abs(rhs).max())
if np.abs(lhs - rhs).max() > tol * scale:
    return False
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

it's the triple loop again, but the loop variables step by `block`, and each "entry" is a tile.

</details>

<details><summary>hint 2: the approach</summary>

`for i in range(0, m, block)`, `for j in range(0, n, block)`, `for p in range(0, k, block)`, then `C[i:i+block, j:j+block] += A[i:i+block, p:p+block] @ B[p:p+block, j:j+block]`. slicing past the end of an array just returns a shorter slice, so the edge tiles take care of themselves.

</details>

<details><summary>hint 3: nearly the answer</summary>

check the shapes of one tile product: `A[i:i+b, p:p+b]` is (rows of tile `i`) × (columns of tile `p`), and `B[p:p+b, j:j+b]` is (rows of tile `p`) × (columns of tile `j`). they agree on the inner size, and the result matches the shape of `C[i:i+b, j:j+b]`, including at the edges.

</details>
