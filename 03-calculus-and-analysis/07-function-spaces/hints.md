# function spaces · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

an inner product of functions is an integral of their product. you built the midpoint rule in topic 06; here it's 1-D, so you can write it directly with `np.arange`.

</details>

<details><summary>hint 2: the approach</summary>

`h = (b - a) / n`, midpoints `x = a + (np.arange(n) + 0.5) * h`, and `<f, g> ≈ sum(f(x) * g(x)) * h`. the norm is the square root of `l2_inner(f, f, ...)`, passing `n` along.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
h = (b - a) / n
x = a + (np.arange(n) + 0.5) * h
return float(np.sum(f(x) * g(x)) * h)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

keep two arrays, the previous and the current polynomial, and step the recurrence forward until you reach degree `k`. handle `k = 0` separately.

</details>

<details><summary>hint 2: the approach</summary>

start with `p_prev = np.ones_like(x)` ($P_0$) and `p = x.copy()` ($P_1$). for `j = 1, ..., k - 1`, compute $P_{j+1} = ((2j + 1)xP_j - jP_{j-1})/(j + 1)$ and shift: `p_prev, p = p, new`. after the loop, `p` is $P_k$. return `p_prev` (ones) when `k == 0`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
p_prev, p = np.ones_like(x), x.copy()
if k == 0:
    return p_prev
for j in range(1, k):
    p_prev, p = p, ((2 * j + 1) * x * p - j * p_prev) / (j + 1)
return p
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

each coefficient is one inner product, divided by $\|P_k\|^2 = 2/(2k + 1)$. to pass $P_k$ to `l2_inner` you need a function of one argument: a `lambda`.

</details>

<details><summary>hint 2: the approach</summary>

for each `k` in `range(K + 1)`: `(2k + 1)/2 * l2_inner(f, lambda t, k=k: legendre(k, t), -1, 1, n=n)`. (the `k=k` default freezes the current `k` inside the lambda.) the series is `sum(c[k] * legendre(k, x))` over `k`, as an array shaped like `x`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# legendre_coefficients
return np.array([(2 * k + 1) / 2 * l2_inner(f, lambda t, k=k: legendre(k, t), -1.0, 1.0, n=n)
                 for k in range(K + 1)])

# legendre_series
x = np.asarray(x, dtype=float)
return sum(ck * legendre(k, x) for k, ck in enumerate(c)) + np.zeros_like(x)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

sample $f$ once on a uniform midpoint grid over $[-\pi, \pi]$; then every coefficient is a weighted sum of those samples. a matrix of $\cos(kx_j)$ (rows $k$, columns $j$) times the vector of samples gives all the $a_k$ at once.

</details>

<details><summary>hint 2: the approach</summary>

`h = 2*pi/n`, `x = -pi + (np.arange(n) + 0.5) * h`, `fx = f(x)`, `k = np.arange(K + 1)[:, None]`. then `a = np.cos(k * x) @ fx * h / pi` and `b = np.sin(k * x) @ fx * h / pi`. for the series: start from `a[0] / 2` and add `a[k] cos(kx) + b[k] sin(kx)` for `k >= 1`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# fourier_series
x = np.asarray(x, dtype=float)
out = np.full_like(x, a[0] / 2)
for k in range(1, len(a)):
    out = out + a[k] * np.cos(k * x) + b[k] * np.sin(k * x)
return out
```

</details>
