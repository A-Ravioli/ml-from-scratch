# vector spaces · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

keep two counters: the column you're looking at (`for col in range(n)`), and the row where the next pivot will go (`row`, starting at 0). the row only advances when you find a pivot.

</details>

<details><summary>hint 2: the approach</summary>

for each column, stop if `row == m`. otherwise find `p = row + argmax(|R[row:, col]|)`. if `|R[p, col]| <= tol`, zero out `R[row:, col]` and go to the next column (it's free). otherwise swap rows `row` and `p`, divide row `row` by its pivot, subtract `R[i, col] * R[row]` from every other row `i`, record `col`, and do `row += 1`.

</details>

<details><summary>hint 3: nearly the answer</summary>

clearing every other row at once:

```python
R[[row, p]] = R[[p, row]]
R[row] /= R[row, col]
others = np.arange(m) != row
R[others] -= np.outer(R[others, col], R[row])
R[others, col] = 0.0
pivots.append(col)
row += 1
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

your `rref` already returns everything `rank` needs.

</details>

<details><summary>hint 2: the approach</summary>

`rank` is `len(rref(A, tol=tol)[1])`. for `in_span`, build `[V | x]` with `np.column_stack([V, x])` and compare its rank with the rank of `V`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
return rank(np.column_stack([V, x]), tol=tol) == rank(V, tol=tol)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

a basis of $\mathbb{R}^n$ needs exactly $n$ vectors, all independent. check the shape first.

</details>

<details><summary>hint 2: the approach</summary>

`is_basis`: `k == n and rank(V, tol=tol) == n`. `coordinates`: raise `ValueError` if not `is_basis(V)`, otherwise row-reduce `np.column_stack([V, x])`. the last column of the result is `c`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
R, _ = rref(np.column_stack([V, x]))
return R[:, -1]
```

this works because the RREF of `[V | x]` is `[I | c]` when `V` is invertible.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

one basis vector per free column. the free columns are the ones not in `pivots`.

</details>

<details><summary>hint 2: the approach</summary>

allocate `N = np.zeros((n, number_of_free_columns))`. for the `j`-th free column `f`: put a 1 in row `f` of column `j`. then for each pivot `p_i` (the pivot of row `i` of `R`), put `-R[i, f]` in row `p_i`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for j, f in enumerate(free):
    N[f, j] = 1.0
    for i, p in enumerate(pivots):
        N[p, j] = -R[i, f]
```

row `i` of `R` reads $x_{p_i} + \sum_{\text{free } g} R_{i,g}x_g = 0$, so with $x_f = 1$ and the other free variables 0, $x_{p_i} = -R_{i,f}$.

</details>
