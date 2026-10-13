# visualizing data · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

everything happens through the Axes object: `ax.plot`, `ax.set_xlabel`, `ax.set_ylabel`, `ax.set_yscale`, `ax.set_title`, `ax.legend`. the only tricky part is telling the two curve formats apart: a pair is a **tuple** of length 2 whose items are themselves sequences.

</details>

<details><summary>hint 2: the approach</summary>

loop over `curves.items()`. for each `(label, curve)`:

- if `isinstance(curve, tuple) and len(curve) == 2 and np.ndim(curve[0]) == 1 and np.ndim(curve[1]) == 1`, it's `(x, y)` (check the lengths match, else `ValueError`);
- otherwise `y = np.asarray(curve)` and `x = np.arange(len(y))`.

then `ax.plot(x, y, label=str(label))`. after the loop, set the labels, `ax.set_yscale("log")` if `logy`, the title if it isn't `None`, and `ax.legend()` if there were any curves.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if ax is None:
    _, ax = plt.subplots(figsize=(6, 3.5))
for label, curve in curves.items():
    if isinstance(curve, tuple) and len(curve) == 2 and np.ndim(curve[0]) == 1 and np.ndim(curve[1]) == 1:
        x, y = np.asarray(curve[0]), np.asarray(curve[1])
        if len(x) != len(y):
            raise ValueError("x and y lengths differ")
    else:
        y = np.asarray(curve)
        x = np.arange(len(y))
    ax.plot(x, y, label=str(label))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

there are three pieces: the edges (one `np.linspace` call), the bin index of every value (one `np.searchsorted` call, then a fix for the values on the last edge), and the counts (one `np.bincount` call).

</details>

<details><summary>hint 2: the approach</summary>

1. `x = np.asarray(x, dtype=float).ravel()`; raise `ValueError` for empty input or `bins < 1`.
2. `lo, hi = x.min(), x.max()`; if they're equal, use `lo - 0.5, hi + 0.5`.
3. `edges = np.linspace(lo, hi, bins + 1)`.
4. `idx = np.searchsorted(edges, x, side="right") - 1` is the number of edges `<=` each value, minus one. only values equal to `hi` get `idx == bins`; clamp them with `np.minimum(idx, bins - 1)`.
5. `counts = np.bincount(idx, minlength=bins)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
edges = np.linspace(lo, hi, bins + 1)
idx = np.minimum(np.searchsorted(edges, x, side="right") - 1, bins - 1)
return np.bincount(idx, minlength=bins), edges
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

do `make_grid` in three phases: normalize the input to a float `(N, H, W, C)` array; make a canvas filled with the pad value; copy each image into its tile. the formulas for the canvas size and each tile's corner are in idea 3 and the worked example. `image_grid` is mostly a call to `make_grid` plus `ax.imshow`.

</details>

<details><summary>hint 2: the approach</summary>

`make_grid`:
- uint8 → divide by 255; then `astype(float)`.
- 3-D input → `imgs[..., None]`. 4-D input whose last axis isn't 1 or 3 but whose axis 1 is → `imgs.transpose(0, 2, 3, 1)`. anything else → `ValueError`.
- `ncols = min(nrow, n)`, `nrows = math.ceil(n / ncols)`, canvas `np.full((nrows*h + (nrows+1)*padding, ncols*w + (ncols+1)*padding, c), fill)`.
- `for k in range(n): row, col = divmod(k, ncols)`, then slice-assign the image at `top = padding + row*(h + padding)`, `left = padding + col*(w + padding)`.
- return `grid[..., 0]` if `c == 1`.

`image_grid`: check the number of titles, make an Axes if needed, `ax.imshow(grid, cmap=cmap)` for 2-D grids (color grids: rescale into [0, 1] if needed, then `ax.imshow(grid)`), `ax.axis("off")`, then one `ax.text(...)` per title at the top center of its tile.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
fill = imgs.max() if pad_value is None else pad_value
grid = np.full((nrows * h + (nrows + 1) * padding, ncols * w + (ncols + 1) * padding, c), fill, dtype=float)
for k in range(n):
    row, col = divmod(k, ncols)
    top, left = padding + row * (h + padding), padding + col * (w + padding)
    grid[top:top + h, left:left + w] = imgs[k]
return grid[..., 0] if c == 1 else grid
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`make_meshgrid` is two `np.linspace` calls and one `np.meshgrid`. in `plot_decision_boundary`, the key line is the single call to `predict`: build an `(m, 2)` array of all grid points with `np.column_stack([xx.ravel(), yy.ravel()])` first.

</details>

<details><summary>hint 2: the approach</summary>

1. `xx, yy = make_meshgrid(X, resolution=resolution, padding=padding)`; `points = np.column_stack([xx.ravel(), yy.ravel()])`; `Z = np.asarray(predict(points))`; check `Z.shape == (len(points),)`.
2. `classes, y_idx = np.unique(y, return_inverse=True)`; `k = len(classes)`; a discrete colormap, e.g. `ListedColormap(plt.cm.tab10(np.arange(k) % 10))`.
3. if `np.all(np.isin(Z, classes))`: `Z_idx = np.searchsorted(classes, Z).reshape(xx.shape)` and `ax.contourf(xx, yy, Z_idx, levels=np.arange(k + 1) - 0.5, cmap=colors, alpha=0.3)`. otherwise contourf the raw scores.
4. `ax.scatter(X[:, 0], X[:, 1], c=y_idx, cmap=colors, vmin=-0.5, vmax=k - 0.5, edgecolors="k")`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
points = np.column_stack([xx.ravel(), yy.ravel()])
Z = np.asarray(predict(points))
if Z.shape != (len(points),):
    raise ValueError("predict must return one value per point")
classes, y_idx = np.unique(y, return_inverse=True)
```

the `vmin`/`vmax` in the scatter call make class index `i` land in the middle of the same color band as region `i`.

</details>
