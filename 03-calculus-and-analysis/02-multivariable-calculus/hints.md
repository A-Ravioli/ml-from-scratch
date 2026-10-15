# multivariable calculus · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

the directional derivative is a 1-D derivative of $\phi(t) = f(x + tv)$ at $t = 0$. use the same central difference you'd use in one variable, with the step taken along $v$.

</details>

<details><summary>hint 2: the approach</summary>

`directional_derivative`: convert `x` and `v` to float arrays and return `(f(x + h*v) - f(x - h*v)) / (2*h)` as a float. don't normalize `v`. `steepest_direction`: get `g = numerical_gradient(f, x, h=h)`, its norm, raise `ValueError` if the norm is zero, and return `(g / norm, norm)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
g = numerical_gradient(f, np.asarray(x, dtype=float), h=h)
norm = float(np.linalg.norm(g))
if norm == 0.0:
    raise ValueError("zero gradient")
return g / norm, norm
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

each input coordinate gives one *column*: perturb $x_j$ up and down, difference the two output vectors. the outputs are length-$m$ vectors, so each column has $m$ entries.

</details>

<details><summary>hint 2: the approach</summary>

make a float copy of `x`. for each `j`, build `e = np.zeros_like(x); e[j] = h`, evaluate `f(x + e)` and `f(x - e)`, wrap each in `np.atleast_1d` (so a scalar output becomes length 1), and append their difference over `2*h` to a list of columns. finally `np.stack(cols, axis=1)` gives shape $(m, n)$.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.array(x, dtype=float)
cols = []
for j in range(x.size):
    e = np.zeros_like(x)
    e[j] = h
    cols.append((np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))) / (2 * h))
return np.stack(cols, axis=1)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

there are two stencils: three points for the diagonal, four points for each off-diagonal pair. the rows of `np.eye(n) * h` are the steps $he_i$.

</details>

<details><summary>hint 2: the approach</summary>

compute `f0 = f(x)` once. loop `i` over all coordinates and fill `H[i, i]` with the second difference. for each `j > i`, compute the four-point stencil once and assign it to both `H[i, j]` and `H[j, i]`. that's exactly symmetric by construction.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
E = np.eye(n) * h
for i in range(n):
    H[i, i] = (f(x + E[i]) - 2 * f0 + f(x - E[i])) / h**2
    for j in range(i + 1, n):
        H[i, j] = H[j, i] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j])
                             - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h**2)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

everything depends on the *signs* of the Hessian's eigenvalues, with "zero" meaning "smaller than a threshold". check for mixed signs first: that alone decides "saddle".

</details>

<details><summary>hint 2: the approach</summary>

`lam = np.linalg.eigvalsh(numerical_hessian(f, x))` and `s = tol * max(1.0, np.max(np.abs(lam)))`. then: any `lam > s` and any `lam < -s` → saddle; all `lam > s` → minimum; all `lam < -s` → maximum; anything else → degenerate.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
pos, neg = np.any(lam > s), np.any(lam < -s)
if pos and neg:
    return "saddle"
if np.all(lam > s):
    return "minimum"
if np.all(lam < -s):
    return "maximum"
return "degenerate"
```

</details>
