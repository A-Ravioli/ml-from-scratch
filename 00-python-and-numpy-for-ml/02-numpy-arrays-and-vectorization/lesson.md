# numpy arrays & vectorization

## why this matters

suppose you have 10,000 images, each flattened to 3,072 numbers, and for every one you want its nearest neighbour among the others. that's 100 million pairs, each needing a distance over 3,072 coordinates: about $3 \times 10^{11}$ multiply-adds. the lab of topic 01 measured a Python loop at roughly 20 ns per multiply-add, so the loop version takes about **100 minutes**. the version you'll write in this topic does it in a few **seconds** on a laptop, and it's three lines long.

the difference isn't a clever algorithm. both versions do the same arithmetic. the difference is *who runs the loop*. in the slow version, the Python interpreter steps through every multiplication one at a time, checking types and allocating objects as it goes. in the fast version, you describe the whole computation as operations on entire **arrays** ("multiply these two grids of numbers", "sum along this axis", "multiply these two matrices"), and NumPy runs the loops in compiled code over raw, tightly packed numbers. writing code this way is called **vectorization**, and it's the single most important programming skill in numerical work. every ML library, from scikit-learn to PyTorch, is built on it.

this topic teaches the four ideas that make it possible: arrays and their **shapes**, **axes** for reductions, **indexing** (and when it copies data), and **broadcasting**. it ends with the payoff: a loop-free `pairwise_distances`, which becomes the first function of `mlfs.utils`, the utility library the rest of the course imports.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. `X` has shape `(100, 3)`. what shape does `X.sum(axis=0)` have? and `X.sum(axis=1)`?
2. `a = np.arange(5)`, `b = a[1:3]`, then `b[0] = 99`. what is `a`?
3. `x` has shape `(3,)` and `y` has shape `(3, 1)`. what shape is `x + y`?
4. `np.array([1, 2, 3]) / 2` gives `[0.5, 1. , 1.5]`. what does `a = np.array([1, 2, 3]); a[0] = 0.7; a` give?
5. to compute all distances between 1,000 points and 1,000 other points in 100 dimensions, the obvious vectorized code builds an intermediate array of how many numbers?

## prerequisites check

- [ ] you can write a class with `__add__` and `__mul__`, and know that `u + v` calls `u.__add__(v)` (→ ch 00 · topic 01)
- [ ] you know why a Python loop over a million floats is slow (→ ch 00 · topic 01, the lab)
- [ ] you can compute a dot product and a Euclidean distance by hand (→ ch 00 · topic 01)
- [ ] you know what a matrix is and how to multiply two small matrices by hand (row times column). if not, the first page of any linear algebra text, or [3Blue1Brown's chapter on matrix multiplication](https://www.3blue1brown.com/lessons/matrix-multiplication), is enough for now (ch 02 does it properly)

## core ideas

### idea 1: arrays have a shape, a dtype, and axes

a NumPy **array** (`np.ndarray`) is a grid of numbers that all have the same type, stored in one contiguous block of memory. that's the layout topic 01's lab found: 8 bytes per float, side by side, with none of the per-object overhead of a Python list.

```python
import numpy as np
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
X.shape   # (2, 3): 2 rows, 3 columns
X.ndim    # 2
X.dtype   # dtype('float64')
```

three attributes describe every array:

- **shape**: a tuple with the length along each dimension. a vector of $n$ numbers has shape `(n,)` (note the comma: a one-element tuple). a table with $n$ rows and $d$ columns has shape `(n, d)`. a batch of $N$ color images might be `(N, 32, 32, 3)`.
- **dtype**: the type of every element, such as `float64` (8-byte floats, the default for decimals), `float32`, `int64` or `bool`. it's fixed for the whole array.
- **axes**: dimension 0 is **axis 0** (the rows of a table), dimension 1 is **axis 1** (the columns), and so on.

in ML the convention, used everywhere in this course, is that a dataset is an array `X` of shape `(n_samples, n_features)`: **one row per example, one column per feature**.

**elementwise operations.** arithmetic between arrays of the same shape works entry by entry, in compiled code: `X + Y`, `X * Y` (elementwise, *not* a matrix product), `X ** 2`, `np.sqrt(X)`, `np.exp(X)`. this is your `Vector` arithmetic from topic 01, for any number of dimensions, a few hundred times faster.

**reductions and `axis`.** functions like `sum`, `mean`, `max` and `std` collapse an array. with no `axis`, they reduce everything to one number. with `axis=k`, they reduce *along* axis $k$, which removes that axis from the shape:

```python
X.sum()         # 21.0
X.sum(axis=0)   # [5., 7., 9.]   shape (3,): added down the rows, one total per column
X.sum(axis=1)   # [ 6., 15.]     shape (2,): added across the columns, one total per row
```

the rule to remember (and warm-up 1): **the axis you name is the one that disappears.** `(100, 3).sum(axis=0)` has shape `(3,)`, a per-feature summary; `.sum(axis=1)` has shape `(100,)`, a per-example summary. if you'd rather keep the reduced axis with length 1, pass `keepdims=True`: `X.sum(axis=1, keepdims=True)` has shape `(100, 1)`. that's handy when the result must line up with `X` again, as you'll see in idea 3 (and use constantly in topic 03).

put the two together and you can compute a dot product for every row at once. for two arrays $A$ and $B$ of shape $(n, d)$,

$$\text{out}_i = \sum_{j} A_{ij}B_{ij} \quad\Longrightarrow\quad \texttt{np.sum(A * B, axis=1)}.$$

that's one elementwise product and one reduction, no matter how big $n$ is. computing many small, independent operations at once like this is called **batching**, and it's how a neural network processes a batch of examples in one go.

> **▶ build: stage 1.** implement `batch_dot(A, B)` and `row_norms(X)` (open the notebook with `marimo edit exercise.py`). `batch_dot` raises `ValueError` if the shapes differ, and neither function may loop.

### idea 2: indexing — slices are views, fancy indices are copies

**basic slicing** works as for lists, with one slice per axis, separated by commas:

```python
X[0]        # the first row, shape (3,)
X[:, 0]     # the first column, shape (2,)
X[:, :-1]   # every column but the last, shape (2, 2)
X[1, 2]     # one element: 6.0
```

here's the crucial difference from lists. slicing a list copies it. **slicing an array doesn't**: it returns a **view**, a new array object that points into the *same* memory, just with a different shape and starting point. that makes slicing almost free (no data moves, even for a gigabyte array), but it means writing to a view writes to the original. that's warm-up 2:

```python
a = np.arange(5)    # [0, 1, 2, 3, 4]
b = a[1:3]          # a view of a's elements 1 and 2
b[0] = 99
a                   # [0, 99, 2, 3, 4]  -- a changed too
```

`np.shares_memory(a, b)` tells you whether two arrays overlap, and `b = a[1:3].copy()` makes an independent copy when you need one.

the same idea bites in function arguments. if a function does `X -= X.mean(axis=0)` on the array it was given, it modifies the **caller's** data, because the argument is the caller's array, not a copy. `X = X - X.mean(axis=0)` instead creates a new array and rebinds the local name, leaving the caller's alone. that one-character difference is a classic bug, and stage 3's tests check for it.

**dtypes bite here too** (warm-up 4). assigning into an array converts the value to the array's dtype, silently. `a = np.array([1, 2, 3])` has an integer dtype, so `a[0] = 0.7` stores `0`. whenever you'll put decimals into an array, create it as float (`np.zeros(n)` is float by default; `np.asarray(x, dtype=float)` converts).

**fancy indexing** means indexing with an *array of integers* (or booleans). it always returns a **copy**, and it's how you gather or scatter many elements at once:

```python
x = np.array([10, 20, 30, 40])
x[[3, 0, 0]]          # [40, 10, 10]: pick elements by position
x[x > 15]             # [20, 30, 40]: a boolean mask keeps the True positions
```

with one integer array per axis, the arrays are paired up element by element. `M[rows, cols]` picks out `M[rows[0], cols[0]], M[rows[1], cols[1]], ...`. that's exactly what a **one-hot encoding** needs. to turn class labels $[2, 0, 1]$ into rows with a single 1 in column `label`:

```python
out = np.zeros((3, n_classes))
out[np.arange(3), labels] = 1.0     # sets out[0, 2], out[1, 0], out[2, 1]
```

one warning: negative integers count from the end in fancy indexing too, so a label of $-1$ would silently set the *last* column. validate your labels first.

> **🔮 predict:** `a = np.arange(6).reshape(2, 3)`. you take `b = a[:, ::2]` (every other column, a slice) and `c = a[:, [0, 2]]` (the same columns, by a list). then `b[0, 0] = 99` and `c[1, 1] = -1`. what does `a` look like afterwards? (try it, then check "check your understanding".)

**a cumulative-sum trick.** slicing also turns loops into arithmetic. the moving average of a series over windows of length $w$ is

$$m_i = \frac{1}{w}\sum_{k=i}^{i+w-1} x_k.$$

the obvious code averages each window separately: $w$ additions for each of $n$ windows. instead, compute the running total $c_k = x_0 + \dots + x_{k-1}$ once (`np.cumsum`, with a 0 in front so that $c_0 = 0$). then every window sum is a difference of two running totals, $\sum_{k=i}^{i+w-1}x_k = c_{i+w} - c_i$, and all of them at once are one subtraction of two slices: `c[w:] - c[:-w]`.

> **▶ build: stage 2.** implement `split_features_target(data)` (it must return views), `one_hot(labels, n_classes)`, and `moving_average(x, window)` with the cumulative-sum trick.

### idea 3: broadcasting

what should `X - X.mean(axis=0)` do, when `X` has shape `(n, d)` and its column means have shape `(d,)`? clearly, subtract the mean row from *every* row. NumPy does exactly that, without ever copying the mean $n$ times. the rule that decides when arrays of different shapes can be combined is called **broadcasting**:

1. line the two shapes up **from the right**. if one has fewer dimensions, pad it with 1s on the left.
2. in each position, the sizes must be equal, or one of them must be 1.
3. a dimension of size 1 is **stretched** (virtually, with no copying) to match the other.

examples, with the result shape on the right:

| shape A | shape B | result |
|---|---|---|
| `(n, d)` | `(d,)` → `(1, d)` | `(n, d)`: the row is reused for every row |
| `(n, 1)` | `(1, m)` | `(n, m)`: every pair $(i, j)$ |
| `(3,)` → `(1, 3)` | `(3, 1)` | `(3, 3)` |
| `(n, d)` | `(n,)` → `(1, n)` | error, unless $d = n$ |

the third row is warm-up 3, and it's the most common source of silent bugs in all of numerical Python. a "vector" of shape `(3,)` plus a "column" of shape `(3, 1)` doesn't give an error. it gives a `(3, 3)` matrix of every pairwise sum. code that mixes up `(n,)` and `(n, 1)` often runs without complaint and computes garbage. **when an array's shape matters, check it**, which is what the tests do. the fourth row is the flip side: to divide each row of `X` by its own norm (shape `(n,)`), you need `X / norms[:, None]`, a column of shape `(n, 1)`.

that `[:, None]` is how you **add an axis** on purpose. indexing with `None` (also spelled `np.newaxis`) inserts a dimension of size 1: `v[:, None]` turns shape `(n,)` into `(n, 1)` and `v[None, :]` turns it into `(1, n)`. with it, you can make any pair of arrays line up the way you want.

for example, standardizing features (rescaling every column to mean 0 and standard deviation 1, which most ML methods want, as chapter 07 explains) is

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}, \qquad \mu_j = \frac1n\sum_i x_{ij},\quad \sigma_j = \sqrt{\frac1n\sum_i (x_{ij} - \mu_j)^2},$$

which in NumPy is `(X - X.mean(axis=0)) / X.std(axis=0)`: two reductions and two broadcast operations.

a constant column needs care: it has $\sigma_j = 0$, and dividing by 0 gives `nan`. you replace those divisors with **`np.where(condition, a, b)`**, the vectorized `if`: it builds an array taking `a` where `condition` is `True` and `b` elsewhere, broadcasting all three. so `np.where(const, 1.0, std)` divides constant columns by 1. but how do you find them? not with `std == 0`: a column of `0.1`s has a *computed* std of about $10^{-17}$, because 0.1 isn't exactly representable in binary and the computed mean differs from the entries in the last digit (topic 03 explains why). dividing that rounding noise by more rounding noise gives $\pm 1$, not 0. test what you actually mean, exactly: a column is constant when its maximum equals its minimum, `X.max(axis=0) == X.min(axis=0)`. one more detail: even with the divisor fixed to 1, `X - mean` can leave about $10^{-17}$ in a constant column (the computed mean is off in the last digit), so set those columns to exactly 0 at the end with a second `np.where(const, 0.0, Z)`.

and every difference between $n$ points $x_i$ and $m$ points $y_j$ in $d$ dimensions is a three-dimensional broadcast: `X[:, None, :]` has shape $(n, 1, d)$, `Y[None, :, :]` has shape $(1, m, d)$, and their difference has shape $(n, m, d)$, with entry $[i, j]$ equal to $x_i - y_j$.

> **▶ build: stage 3.** implement `standardize(X)` (it must not modify `X`) and `pairwise_differences(X, Y)`.

> **🔮 predict:** `pairwise_differences` is fully vectorized. if you use it to compute all distances between 1,000 points and 1,000 points in 100 dimensions, how much memory does the intermediate array take? (warm-up 5. the lab measures what that costs.)

### idea 4: distances without the difference array

from stage 3, all pairwise distances are one line: `np.sqrt((pairwise_differences(X, Y) ** 2).sum(axis=2))`. it's correct and loop-free. but for $n = m = 1{,}000$ and $d = 100$ it builds an intermediate array of $n \cdot m \cdot d = 10^8$ numbers, which is 800 MB, just to sum most of it away again. there's a better way, and it comes from expanding a square (on paper 1 of topic 01):

$$\|x - y\|^2 = (x - y)\cdot(x - y) = \|x\|^2 - 2\,x\cdot y + \|y\|^2.$$

each of the three terms is cheap to compute for all pairs at once:

- $\|x_i\|^2$ for every row of $X$: `np.sum(X * X, axis=1)`, shape $(n,)$. reshape it to a column, $(n, 1)$.
- $\|y_j\|^2$ for every row of $Y$: shape $(m,)$, reshaped to a row, $(1, m)$.
- $x_i \cdot y_j$ for **every pair**: that's the matrix product $XY^\top$, shape $(n, m)$, since entry $(i, j)$ of a matrix product is row $i$ of the first matrix dotted with column $j$ of the second.

broadcasting adds the column, the matrix and the row into the full $(n, m)$ matrix of squared distances. nothing of size $n \cdot m \cdot d$ is ever built. better still, the matrix product `X @ Y.T` is the single most optimized operation in numerical computing: NumPy hands it to a BLAS library that uses every trick your CPU has. chapter 02 explains why it's so fast. for now, it's the reason this version wins.

one catch, previewing topic 03. computers round every result to about 16 significant digits. when $x$ and $y$ are nearly equal and far from the origin, $\|x\|^2$ and $2\,x\cdot y$ are huge and nearly cancel, and the rounding errors can push the computed squared distance slightly **below zero**. its square root is then `nan`. the fix is to clip at zero before the square root, `np.maximum(d2, 0.0)`. (the lab shows that this formula has a deeper accuracy problem far from the origin, too.)

when `Y` is omitted, the reference compares `X` with itself and sets the diagonal to exactly 0, since every point is at distance 0 from itself, rounding or not.

> **▶ build: stage 4.** implement `pairwise_distances(X, Y=None, *, squared=False)` with the expansion above. it becomes `mlfs.utils.pairwise_distances`.

## worked example

compute all distances between the rows of

$$X = \begin{pmatrix}0 & 0\\ 3 & 4\end{pmatrix}, \qquad Y = \begin{pmatrix}0 & 0\\ 1 & 0\\ 3 & 0\end{pmatrix}.$$

**squared norms.** for $X$: $(0,\ 9 + 16) = (0, 25)$, as a column. for $Y$: $(0, 1, 9)$, as a row.

**all dot products.** $XY^\top$ has entry $(i, j) = x_i \cdot y_j$:

$$XY^\top = \begin{pmatrix}0 & 0 & 0\\ 0 & 3 & 9\end{pmatrix} \qquad\text{(row 2: } (3,4)\cdot(0,0) = 0,\ (3,4)\cdot(1,0) = 3,\ (3,4)\cdot(3,0) = 9\text{)}.$$

**combine.** $D^2_{ij} = \|x_i\|^2 - 2(XY^\top)_{ij} + \|y_j\|^2$:

$$D^2 = \begin{pmatrix}0 & 0 & 0\\ 25 & 25 & 25\end{pmatrix} - 2\begin{pmatrix}0 & 0 & 0\\ 0 & 3 & 9\end{pmatrix} + \begin{pmatrix}0 & 1 & 9\\ 0 & 1 & 9\end{pmatrix} = \begin{pmatrix}0 & 1 & 9\\ 25 & 20 & 16\end{pmatrix}.$$

the first matrix is the column $(0, 25)$ broadcast across three columns; the last is the row $(0, 1, 9)$ broadcast down two rows.

**square root.** $D = \begin{pmatrix}0 & 1 & 3\\ 5 & \sqrt{20} & 4\end{pmatrix} \approx \begin{pmatrix}0 & 1 & 3\\ 5 & 4.472 & 4\end{pmatrix}$.

check one directly: $(3, 4)$ to $(1, 0)$ is $\sqrt{2^2 + 4^2} = \sqrt{20}$. ✓ the tests check this exact matrix.

## on paper

1. **shapes.** give the shape of the result, or say it's an error: (a) `(5, 3) + (3,)`; (b) `(5, 3) + (5,)`; (c) `(5, 1) * (1, 4)`; (d) `(2, 1, 3) - (4, 1)`; (e) `np.ones((5, 3)).mean(axis=0)[:, None]`.

<details><summary>solution</summary>

(a) `(5, 3)`. (b) error: aligned from the right, 3 and 5 differ and neither is 1. (c) `(5, 4)`. (d) pad `(4, 1)` to `(1, 4, 1)`, then each position: 2 vs 1 → 2, 1 vs 4 → 4, 3 vs 1 → 3, so `(2, 4, 3)`. (e) the mean has shape `(3,)`, and `[:, None]` makes it `(3, 1)`.

</details>

2. **the cost of each approach.** for $n = m = 10^4$ points in $d = 100$ dimensions, how much memory (in bytes, with 8-byte floats) does (a) the difference-array method's intermediate take, and (b) the expansion method's largest array take? count the multiply-adds of each.

<details><summary>solution</summary>

(a) $n \cdot m \cdot d = 10^{10}$ numbers, 80 GB: it won't fit on a laptop at all. (b) the $(n, m)$ result, $10^8$ numbers, 800 MB (the norms are tiny). both do about $n \cdot m \cdot d = 10^{10}$ multiply-adds (the expansion's are in the matrix product), so the arithmetic is the same. the difference is memory, and the speed of memory, which is why the expansion wins.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. how much faster is your vectorized `pairwise_distances` than a double loop, and how does the speedup change with the number of points?
2. three vectorized ways of computing distances, all loop-free: which is fastest as the dimension grows, and how much memory does each need?
3. what happens to your fast formula's accuracy when the points move far from the origin?

record your predictions first.

## check your understanding

1. `X` has shape `(n, d)`. write the expression for the mean of each feature, and for the mean of each example.

<details><summary>answer</summary>

`X.mean(axis=0)` (shape `(d,)`, one per feature) and `X.mean(axis=1)` (shape `(n,)`, one per example). the named axis disappears.

</details>

2. why does `split_features_target` return views, and when would that be dangerous?

<details><summary>answer</summary>

views are free: no data is copied, which matters for big tables. the danger is writing through them: `X[:] = 0` (or any in-place operation on `X` or `y`) also changes the original table. if you need to modify the pieces independently, `.copy()` them.

</details>

3. you have `scores` of shape `(n,)` and want to subtract it from every column of `M`, shape `(n, k)`. why does `M - scores` fail or misbehave, and what's the fix?

<details><summary>answer</summary>

aligned from the right, `(n, k)` and `(n,)` compare `k` with `n`: it's an error, unless `k == n`, in which case it silently subtracts `scores` from every *row* instead. write `M - scores[:, None]`, a column of shape `(n, 1)`.

</details>

4. why can a squared distance computed with the expansion formula come out negative, and why can't the difference-array method's?

<details><summary>answer</summary>

the expansion subtracts large, nearly equal numbers ($\|x\|^2 + \|y\|^2$ and $2x\cdot y$), and their rounding errors survive the subtraction. the difference method squares each coordinate difference and adds the squares: a sum of non-negative numbers can't be negative.

</details>

5. `labels = np.array([0, 2, -1])`. what does `one_hot` without a range check do with the $-1$?

<details><summary>answer</summary>

`out[2, -1] = 1` sets the *last* column of row 2, as if the label were `n_classes - 1`. no error, wrong encoding. this is why `one_hot` validates its input.

</details>

6. (the predict callout in idea 2.) after `b = a[:, ::2]; c = a[:, [0, 2]]; b[0, 0] = 99; c[1, 1] = -1`, what is `a`?

<details><summary>answer</summary>

`[[99, 1, 2], [3, 4, 5]]`. `b` is a slice, so it's a view: writing to it writes to `a`. `c` selects the same columns with a list, which is fancy indexing, so it's a copy: writing to it leaves `a` alone. the two expressions give arrays with identical values and completely different behaviour, so when it matters, check with `np.shares_memory`.

</details>

7. (the predict callout in idea 3.) how big is the difference array for 1,000 × 1,000 points in 100 dimensions?

<details><summary>answer</summary>

$10^3 \cdot 10^3 \cdot 100 = 10^8$ floats, 800 MB. squaring it makes another 800 MB temporary array. the lab shows how much time all that memory traffic costs.

</details>

## common bugs

- **reducing along the wrong axis.** symptom: a result of the wrong shape, like `(d,)` where you expected `(n,)`, or a single number. the axis you name disappears: per-example summaries use `axis=1`, per-feature summaries use `axis=0`.
- **`(n,)` versus `(n, 1)`.** symptom: an `(n, n)` matrix appears where you expected a vector, or a "shapes could not be broadcast" error. print `.shape` early and often, and add axes on purpose with `[:, None]`.
- **modifying the caller's array.** symptom: data changes after you pass it to a function. `X -= ...`, `X[...] = ...` and `X /= ...` write into the caller's array. write `X = X - ...` instead, or copy first.
- **integer arrays.** symptom: values silently truncated to whole numbers when you assign into an integer array (`a[0] = 0.7` stores 0), or a `UFuncTypeError` from an in-place update like `X -= X.mean(axis=0)` on integer data. `np.array([1, 2, 3])` has an integer dtype: convert with `np.asarray(x, dtype=float)` before writing into it.
- **`sqrt` of a slightly negative number.** symptom: `nan` on the diagonal of a distance matrix, plus a `RuntimeWarning: invalid value encountered in sqrt`. clip with `np.maximum(d2, 0.0)`.
- **a Python loop over array elements.** symptom: correct but slow code, especially with large `n`. topic 01's lab showed that looping over an array is even slower than looping over a list.

## going deeper

- Harris et al. (2020), "Array programming with NumPy", *Nature* 585: 357–362. the design of NumPy, with clear diagrams of strides, views and broadcasting. [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- van der Walt, Colbert & Varoquaux (2011), "The NumPy array: a structure for efficient numerical computation", *Computing in Science & Engineering* 13(2): 22–30. why arrays are fast: memory layout, strides and vectorization. [doi:10.1109/MCSE.2011.37](https://doi.org/10.1109/MCSE.2011.37)
- the NumPy user guide: [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), [indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html) (including exactly when you get a view and when a copy), and [copies and views](https://numpy.org/doc/stable/user/basics.copies.html).
- VanderPlas (2016), *Python Data Science Handbook*, O'Reilly, chapter 2: a gentle tour of arrays, ufuncs, aggregations, broadcasting and fancy indexing. [free online](https://jakevdp.github.io/PythonDataScienceHandbook/)

## connections

- ← builds on: Python classes, operators and the cost of loops (topic 01).
- → used in: floating point and the cancellation that bit `pairwise_distances` (topic 03); `mlfs.utils.pairwise_distances` in nearest-neighbour classification (ch 02's capstone uses the same idea), kernels and k-NN (ch 09), and clustering (ch 11); `one_hot` in classification losses (ch 08, ch 14); `standardize` in preprocessing (ch 07); batched array code in every chapter that follows.
