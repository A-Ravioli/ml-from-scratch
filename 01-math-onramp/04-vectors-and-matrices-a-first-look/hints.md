# vectors & matrices, a first look · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

for NumPy arrays, `u * v` multiplies entry by entry (it's *not* a dot product). what's left to do after that?

</details>

<details><summary>hint 2: the approach</summary>

`dot`: convert both to float arrays, compare `u.shape` with `v.shape` and raise `ValueError` if they differ (NumPy would otherwise "broadcast" a length-1 vector silently), then `float(np.sum(u * v))`. `norm`: `float(np.sqrt(dot(v, v)))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
u = np.asarray(u, dtype=float)
v = np.asarray(v, dtype=float)
if u.shape != v.shape:
    raise ValueError("dot needs vectors of the same length")
return float(np.sum(u * v))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

solve $u \cdot v = \|u\|\|v\|\cos\theta$ for $\theta$. you have `dot` and `norm` already.

</details>

<details><summary>hint 2: the approach</summary>

`angle`: `c = dot(u, v) / (norm(u) * norm(v))`, then `np.arccos(np.clip(c, -1.0, 1.0))`. `project`: compute the number `dot(u, v) / dot(v, v)` and multiply the (float array) `v` by it.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
v = np.asarray(v, dtype=float)
return dot(u, v) / dot(v, v) * v
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`A[:, j]` is column $j$ of `A` (all rows, column $j$). $Av$ adds up the columns, each multiplied by one entry of $v$.

</details>

<details><summary>hint 2: the approach</summary>

`matvec`: check `A.shape[1] == v.shape[0]`, start from `result = np.zeros(A.shape[0])`, and loop `for j in range(A.shape[1]):` adding `v[j] * A[:, j]`. `matmul`: check `A.shape[1] == B.shape[0]`, make `C = np.zeros((A.shape[0], B.shape[1]))`, and fill each column with `C[:, j] = matvec(A, B[:, j])`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
result = np.zeros(A.shape[0])
for j in range(A.shape[1]):
    result = result + v[j] * A[:, j]
return result
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

for each matrix, ask: where does $(1, 0)$ go? that's column 1. where does $(0, 1)$ go? that's column 2. `np.array([[a, b], [c, d]])` has first *column* $(a, c)$ and second column $(b, d)$: you write it row by row.

</details>

<details><summary>hint 2: the approach</summary>

rotation: column 1 is $(\cos\theta, \sin\theta)$ and column 2 is $(-\sin\theta, \cos\theta)$, so the rows are $(\cos\theta, -\sin\theta)$ and $(\sin\theta, \cos\theta)$. scaling: rows $(s_x, 0)$ and $(0, s_y)$. shear: columns $(1, 0)$ and $(k, 1)$, so rows $(1, k)$ and $(0, 1)$.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
c, s = np.cos(theta), np.sin(theta)
return np.array([[c, -s], [s, c]])
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`transform_points` is one line with your `matmul`, once you've put the points into columns. for the plot, everything is an `ax.plot(xs, ys)` call except the arrows.

</details>

<details><summary>hint 2: the approach</summary>

`transform_points`: `matmul(A, P.T).T`. `plot_transform`: make `ax` if needed, build `t = np.linspace(0, 2 * np.pi, n)` and `circle = np.column_stack([np.cos(t), np.sin(t)])`, plot it dashed, plot `transform_points(A, circle)` solid, draw an arrow to each column with `ax.annotate("", xy=A[:, j], xytext=(0, 0), arrowprops=dict(arrowstyle="->"))`, then `ax.set_aspect("equal")` and `return ax`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if ax is None:
    _, ax = plt.subplots(figsize=(4, 4))
t = np.linspace(0, 2 * np.pi, n)
circle = np.column_stack([np.cos(t), np.sin(t)])
ax.plot(circle[:, 0], circle[:, 1], "--", color="gray")
image = transform_points(A, circle)
ax.plot(image[:, 0], image[:, 1])
for j in range(2):
    ax.annotate("", xy=A[:, j], xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
ax.set_aspect("equal")
return ax
```

</details>

## stage 6

<details><summary>hint 1: a nudge</summary>

`det2` is one line. for `polygon_area`, you need each corner together with the *next* corner, where the last corner's "next" is the first. `np.roll(x, -1)` is `x` shifted left by one, with the first entry moved to the end.

</details>

<details><summary>hint 2: the approach</summary>

`det2`: `A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]`. `polygon_area`: with `x, y = P[:, 0], P[:, 1]` and `x_next, y_next = np.roll(x, -1), np.roll(y, -1)`, the signed area is half the sum of `x * y_next - x_next * y`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x, y = P[:, 0], P[:, 1]
return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
```

</details>
