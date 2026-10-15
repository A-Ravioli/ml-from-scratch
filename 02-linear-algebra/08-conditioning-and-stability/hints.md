# conditioning & stability · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

each norm is one line. the only care needed is dispatching on `ord`: compare strings with strings (`"fro"`) and numbers with numbers (`1`, `2`, `np.inf`).

</details>

<details><summary>hint 2: the approach</summary>

column sums are `np.abs(A).sum(axis=0)`, row sums are `np.abs(A).sum(axis=1)`. Frobenius is `np.sqrt(np.sum(A * A))`. the 2-norm is the first singular value from your `svd`. anything else: `raise ValueError(...)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if isinstance(ord, str):
    if ord == "fro":
        return float(np.sqrt(np.sum(A * A)))
elif ord == 1:
    return float(np.abs(A).sum(axis=0).max())
elif ord == np.inf:
    return float(np.abs(A).sum(axis=1).max())
elif ord == 2:
    return float(svd(A)[1][0])
raise ValueError(f"unsupported ord {ord!r}")
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

your `svd` returns the singular values sorted descending, so the largest is first and the smallest last.

</details>

<details><summary>hint 2: the approach</summary>

`s = svd(A)[1]`. if `s[-1] == 0`, return `np.inf`. otherwise return `s[0] / s[-1]` as a float.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
s = svd(A)[1]
return np.inf if s[-1] == 0 else float(s[0] / s[-1])
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

two loops, each a topic-05 algorithm applied to $M = A^\top A$, but without ever forming $M$. the first multiplies by $M$ as `A.T @ (A @ v)`. the second solves with $M$ using the LU factors of $A$.

</details>

<details><summary>hint 2: the approach</summary>

`P, L, U = lu(A)` once (return `np.inf` on a zero pivot). power loop: `w = A.T @ (A @ v)`, `lam = v @ w`, `v = w / norm(w)`. inverse loop: solve $A^\top y = u$ via `z = solve_triangular(U.T, u, lower=True)`, `y = P.T @ solve_triangular(L.T, z, lower=False, unit_diagonal=True)`, then `x = lu_solve((P, L, U), y)`. now `mu = u @ x` estimates $1/\sigma_{\min}^2$, and you set `u = x / norm(x)`. at the end, `sqrt(lam * mu)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

why does the transposed solve look like that? $PA = LU$ gives $A = P^\top LU$ (since $P^{-1} = P^\top$), so $A^\top = U^\top L^\top P$. solving $U^\top L^\top P y = u$ means peeling off one factor at a time from the left: first $U^\top z = u$ ($U^\top$ is lower-triangular), then $L^\top w = z$ ($L^\top$ is upper-triangular with a unit diagonal), then $Py = w$, so $y = P^\top w$. stop each loop early when its estimate changes by less than about $10^{-15}$ relative.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

one residual, three norms.

</details>

<details><summary>hint 2: the approach</summary>

`r = b - A @ x`. the denominator is `matrix_norm(A, 2) * norm(x) + norm(b)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
r = b - A @ x
return float(np.linalg.norm(r) / (matrix_norm(A, 2) * np.linalg.norm(x) + np.linalg.norm(b)))
```

</details>
