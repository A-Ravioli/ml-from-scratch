# VC dimension · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

convert `y` to bools. if there are no positives, return `True`. otherwise the smallest interval (box) containing the positives is the only candidate worth checking: `lo = x[y].min()`, `hi = x[y].max()` (for boxes, `axis=0` gives per-coordinate bounds). the labeling is realizable iff no negative lies inside `[lo, hi]` (every coordinate, boundaries included). reshape `X` to `(n, -1)` so the box version works for any $d$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
lo, hi = X[y].min(axis=0), X[y].max(axis=0)
inside = np.all((X >= lo) & (X <= hi), axis=1)
return not bool(np.any(inside & ~y))
```

the interval version is the same with 1-D arrays.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the unknowns are $z = (w, b) \in \mathbb R^{d+1}$. append a column of ones to `X`, so $w^\top x_i + b = [x_i, 1]\cdot z$. `linprog` wants constraints as `A_ub @ z <= b_ub`, so multiply $s_i\,[x_i, 1]\cdot z \ge 1$ by $-1$. use a zero objective vector and `bounds=(None, None)`; `res.status == 0` means a feasible point was found (2 means infeasible).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
s = np.where(y, 1.0, -1.0)
A = -s[:, None] * np.hstack([X, np.ones((len(X), 1))])
res = linprog(np.zeros(X.shape[1] + 1), A_ub=A, b_ub=-np.ones(len(X)), bounds=(None, None), method="highs")
return res.status == 0
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

check `len(X) > max_points` first and raise. `itertools.product((0, 1), repeat=n)` yields every labeling as a tuple; wrap each in `np.array(..., dtype=int)` before calling the oracle. for `is_shattered`, `all(...)` over a generator stops at the first `False` by itself.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
labelings = (np.array(lab, dtype=int) for lab in itertools.product((0, 1), repeat=n))
count = sum(bool(realizable(X, y)) for y in labelings)
```

and `is_shattered` is `all(realizable(X, y) for y in labelings)`.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

an outer loop over sizes `d`, an inner loop over `itertools.combinations(range(len(pool)), d)`. index the pool with a list of indices (`pool[list(idx)]`) and ask `is_shattered`. remember the first shattered subset and break out of the inner loop; if the inner loop finishes without finding one, break out of the outer loop too.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
best, witness = 0, ()
for d in range(1, min(max_d, len(pool)) + 1):
    found = next((idx for idx in itertools.combinations(range(len(pool)), d)
                  if is_shattered(pool[list(idx)], realizable)), None)
    if found is None:
        break
    best, witness = d, found
return best, witness
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`math.comb(m, i)` is exact and returns 0 for $i > m$, so `sum(math.comb(m, i) for i in range(d + 1))` already works (stopping at `min(d, m)` just saves time). for the growth bound, make one generator with `as_rng(rng)` and draw each subset with `gen.choice(len(pool), size=m, replace=False)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
gen = as_rng(rng)
best = 0
for _ in range(n_subsets):
    idx = gen.choice(len(pool), size=m, replace=False)
    best = max(best, count_labelings(pool[idx], realizable))
return best
```

</details>
