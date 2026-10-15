# forward-mode autodiff · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

every binary method has two cases: `other` is a `Dual`, or it's a plain number (a constant, i.e. a Dual with dual part 0). check with `isinstance(other, Dual)`. the reflected methods (`__radd__`, …) are only called with a plain number on the left.

</details>

<details><summary>hint 2: the approach</summary>

write the formulas from idea 1 directly: sum and difference part by part; product `Dual(a*c, a*d + b*c)`; quotient `Dual(a/c, (b*c - a*d)/(c*c))`. for a number `k`: `Dual(a + k, b)`, `Dual(a*k, b*k)`, `Dual(a/k, b/k)`. one more case comes first: if `other` is a NumPy array, `return NotImplemented`, so NumPy computes `d * array` entry by entry (an object array of Duals) instead of you building one Dual with array parts. `__rsub__(k)` is `k - self = Dual(k - a, -b)`, and `__rtruediv__(k)` is `Dual(k/a, -k*b/(a*a))`. each comparison compares `self.real` with `other.real` (or with `other` if it's a number).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def __mul__(self, other):
    if isinstance(other, Dual):
        return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)
    return Dual(self.real * other, self.dual * other)

def __rtruediv__(self, other):
    return Dual(other / self.real, -other * self.dual / (self.real * self.real))

def __lt__(self, other):
    return self.real < (other.real if isinstance(other, Dual) else other)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

every rule has the shape `Dual(g(a), g'(a) * b)` with `a = self.real` and `b = self.dual`. compute `g(a)` with NumPy (`np.exp`, `np.sin`, …), never `math`, so the rules keep working when `a` is itself a Dual.

</details>

<details><summary>hint 2: the approach</summary>

reuse values: `e = np.exp(a)` then `Dual(e, e * b)` (and `log1p` is `Dual(np.log1p(a), b / (1 + a))`, `expm1` is `Dual(np.expm1(a), np.exp(a) * b)`); `t = np.tanh(a)` then `Dual(t, (1 - t*t) * b)`; `s = np.sqrt(a)` then `Dual(s, b / (2*s))`. for `__pow__`: if `p` is an `np.ndarray`, return `NotImplemented`; if `p` is a Dual, return `(p * self.log()).exp()`; if `p == 0`, return `Dual(a ** 0, 0.0 * b)`; otherwise `Dual(a ** p, p * a ** (p - 1) * b)`. `__rpow__(c)`: `v = c ** a`, then `Dual(v, v * np.log(c) * b)`. `__abs__`: `-self` if `a < 0`, else a copy.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def log(self):
    return Dual(np.log(self.real), self.dual / self.real)

def cos(self):
    return Dual(np.cos(self.real), -np.sin(self.real) * self.dual)

def __pow__(self, p):
    if isinstance(p, Dual):
        return (p * self.log()).exp()
    if p == 0:
        return Dual(self.real ** 0, 0.0 * self.dual)
    return Dual(self.real ** p, p * self.real ** (p - 1) * self.dual)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`derivative` is one line once `dual_parts` exists. for the gradient, build an object array of Duals whose real parts are `x` and whose dual parts are the basis vector $e_i$, call `f` on it, and read the dual part of the result. repeat for each `i`.

</details>

<details><summary>hint 2: the approach</summary>

`dual_parts(y)`: a `Dual` → `(float(y.real), float(y.dual))`; a float array → `(y.astype(float), zeros)`; an object array → apply `dual_parts` to every entry of `y.ravel()` and reshape both results to `y.shape`; anything else is a number → `(float(y), 0.0)`. in `value_and_grad`, create `xd = np.empty(n, dtype=object)` and fill `xd[j] = Dual(float(x[j]), 1.0 if j == i else 0.0)`. keep the real part of the first pass as the value.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
n = x.size
g = np.empty(n)
value = None
for i in range(n):
    xd = np.empty(n, dtype=object)
    for j in range(n):
        xd[j] = Dual(float(x[j]), 1.0 if j == i else 0.0)
    val, g[i] = dual_parts(f(xd))
    if value is None:
        value = val
return float(value), g
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`jvp` is one pass of `value_and_grad` with the seed $e_i$ replaced by `v`. `jacobian` calls `jvp` once per input with $v = e_j$, and each call gives you a column.

</details>

<details><summary>hint 2: the approach</summary>

in `jvp`, fill `xd[j] = Dual(float(x[j]), float(v[j]))` and return `dual_parts(f(xd))`: that's already `(f(x), J v)` in the right shapes. in `jacobian`, collect `np.atleast_1d(jvp(f, x, e_j)[1])` for each `j` (so scalar outputs become length 1), then `np.stack(cols, axis=1)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
cols = []
for j in range(n):
    e = np.zeros(n)
    e[j] = 1.0
    cols.append(np.atleast_1d(jvp(f, x, e)[1]))
return np.stack(cols, axis=1)
```

</details>
