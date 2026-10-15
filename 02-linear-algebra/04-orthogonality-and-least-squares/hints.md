# orthogonality & least squares · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

allocate `Q = np.zeros((m, n))` and `R = np.zeros((n, n))`, then fill column `j` of both in a loop over `j`. at step `j`, columns `0..j-1` of `Q` are finished.

</details>

<details><summary>hint 2: the approach</summary>

for column `j`: the coefficients against the earlier `q`'s are one matrix-vector product, `R[:j, j] = Q[:, :j].T @ A[:, j]`. subtract the projection, `v = A[:, j] - Q[:, :j] @ R[:j, j]`, then `R[j, j] = norm(v)` and `Q[:, j] = v / R[j, j]`. (for `j = 0` the slices are empty and everything still works.)

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for j in range(n):
    R[:j, j] = Q[:, :j].T @ A[:, j]
    v = A[:, j] - Q[:, :j] @ R[:j, j]
    R[j, j] = np.linalg.norm(v)
    if R[j, j] == 0:
        raise np.linalg.LinAlgError("dependent columns")
    Q[:, j] = v / R[j, j]
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

MGS fills `R` row by row instead of column by column. when `q_j` is ready, you immediately know all of `R[j, j+1:]`.

</details>

<details><summary>hint 2: the approach</summary>

keep `V = A.copy()`. for each `j`: normalize `V[:, j]` to get `Q[:, j]` (and `R[j, j]`), then compute the coefficients against the *current* remaining columns, `R[j, j+1:] = Q[:, j] @ V[:, j+1:]`, and remove them: `V[:, j+1:] -= np.outer(Q[:, j], R[j, j+1:])`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
V = A.copy()
for j in range(n):
    R[j, j] = np.linalg.norm(V[:, j])
    if R[j, j] == 0:
        raise np.linalg.LinAlgError("dependent columns")
    Q[:, j] = V[:, j] / R[j, j]
    R[j, j + 1:] = Q[:, j] @ V[:, j + 1:]
    V[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

work on a copy `R` of `A`. at step `k`, look only at `x = R[k:, k]` and reflect the block `R[k:, k:]`. store each unit vector `v` in a list: you need them again to build `Q`.

</details>

<details><summary>hint 2: the approach</summary>

there are `min(m - 1, n)` steps. for each: `v = x.copy(); v[0] += sign(x[0]) * norm(x)` (with sign(0) taken as +1), normalize `v`, then `R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])`. to build `Q`, start from `np.eye(m)` (complete) or `np.eye(m, min(m, n))` (reduced), and apply the reflections in **reverse** order to its rows `k:`: `Q[k:] -= 2 * np.outer(v, v @ Q[k:])`. finally `R = np.triu(R)`, and in reduced mode keep its first `min(m, n)` rows.

</details>

<details><summary>hint 3: nearly the answer</summary>

why reverse order? $Q = H_0H_1\cdots H_{p-1}$, so $Q\,I = H_0(H_1(\cdots(H_{p-1}I)))$: the last reflection hits the identity first. and since $H_k$ only touches rows $k$ and below, each update is `Q[k:] -= 2 * np.outer(v, v @ Q[k:])`. (applying them forward to `np.eye(m)` from the right, `Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v, v)`, also works, but it always builds the full $m \times m$ matrix, which is slow for tall $A$.)

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

$\|Ax - b\| = \|Rx - Q^\top b\|$ (plus a constant). make the first part zero.

</details>

<details><summary>hint 2: the approach</summary>

reduced `Q, R = qr(A)`, then a back substitution: `solve_triangular(R, Q.T @ b, lower=False)`. `Q.T @ b` works for `b` of shape `(m,)` and `(m, k)` alike.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
Q, R = qr(A)
return solve_triangular(R, Q.T @ np.asarray(b, dtype=float), lower=False)
```

if this raises `NotImplementedError` from `solve_triangular`, topic 03 isn't finished. run with `MLFS_REFERENCE=linalg` for now.

</details>
