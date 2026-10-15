# linear maps & matmul

## why this matters

a neural network layer is $y = Wx + b$. a batch of 512 inputs through that layer is one matrix product. a transformer spends most of its compute in matrix multiplications, and so does training one: the backward pass is matmuls too. GPUs exist, commercially, because they multiply matrices fast. if you understand one operation deeply in this course, make it this one.

"understand" has two halves. the **mathematical** half: a matrix *is* a linear map, matrix multiplication *is* composition of maps, and the same product can be read in four different ways (dot products, combinations of columns, combinations of rows, sums of rank-one pieces). each reading is the natural one for some later idea, from least squares to the SVD to backpropagation. the **computational** half: the textbook triple loop and `np.matmul` do exactly the same arithmetic, yet one is thousands to tens of thousands of times faster, depending on your machine. you'll write the product four ways, recover the matrix of any linear function, and measure where the speed comes from.

## warm-up: predict first

1. a linear map sends $e_1 = (1,0) \mapsto (2, 1)$ and $e_2 = (0,1) \mapsto (-1, 3)$. where does it send $(3, 2)$?
2. if $A$ is $3 \times 5$ and $B$ is $5 \times 2$, how many multiplications does computing $AB$ take?
3. is $AB = BA$ for square matrices?
4. a linear map from $\mathbb{R}^5$ to $\mathbb{R}^3$: can it be one-to-one (no two inputs with the same output)?
5. for $n \times n$ matrices $A, B$ and a vector $x$, is computing $(AB)x$ or $A(Bx)$ cheaper, and by how much?

## prerequisites check

- [ ] you can multiply small matrices by hand (→ ch 01 · topic 04)
- [ ] you know span, rank and null space, and can compute them with your topic-01 code (→ ch 02 · topic 01)
- [ ] you can write nested loops over NumPy arrays and use slicing (→ ch 00 · topic 02)
- [ ] you know why vectorized NumPy is faster than Python loops (→ ch 00 · topic 02)

## core ideas

### idea 1: a matrix is a linear map

a function $f: \mathbb{R}^n \to \mathbb{R}^m$ is **linear** if it respects linear combinations:

$$f(ax + by) = a\,f(x) + b\,f(y)\qquad\text{for all vectors } x, y \text{ and scalars } a, b.$$

linearity means $f$ is determined entirely by what it does to a basis. write $x = x_1e_1 + \dots + x_ne_n$. then $f(x) = x_1f(e_1) + \dots + x_nf(e_n)$. collect the images $f(e_j)$ as the columns of a matrix $A$, and this says

$$f(x) = Ax = x_1a_1 + x_2a_2 + \dots + x_na_n:$$

**the matrix-vector product is the combination of $A$'s columns with weights $x$**, and **column $j$ of $A$ is where $e_j$ lands**. (warm-up 1: $3(2,1) + 2(-1,3) = (4, 9)$.)

now compose two maps. if $g(x) = Bx$ and $f(y) = Ay$, what's the matrix of $f(g(x))$? its column $j$ is $f(g(e_j)) = A(Be_j) = A\,b_j$. so we *define* the product $AB$ as the matrix whose columns are $Ab_1, \dots, Ab_n$, and then $(AB)x = A(Bx)$ holds automatically. matrix multiplication is the way it is *because* it represents composition. expanding entry $(i, j)$ gives the familiar formula:

$$C = AB,\qquad c_{ij} = \sum_{p=1}^k a_{ip}b_{pj}\quad(\text{row } i \text{ of } A \text{ dotted with column } j \text{ of } B).$$

for $A$ of shape $m \times k$ and $B$ of shape $k \times n$, the result is $m \times n$. that's $mnk$ multiplications and about as many additions: $2mnk$ flops (warm-up 2: $30$ multiplications). for square matrices, that's $2n^3$.

> **▶ build: stage 1.** implement `matmul_entries(A, B)` in `exercise.py` (`marimo edit exercise.py`): three nested loops, one entry at a time. no `@`.

### idea 2: columns and rows

the dot-product formula hides structure. two other readings of the same sum are often more useful.

**column view.** $c_j = Ab_j = \sum_p b_{pj}\,a_p$: each column of $C$ is a combination of the *columns of $A$*. so the columns of $AB$ lie in the column space of $A$, and $\operatorname{rank}(AB) \le \operatorname{rank}(A)$.

**row view.** row $i$ of $C$ is $\sum_p a_{ip}\,(\text{row } p \text{ of } B)$: each row of $C$ is a combination of the *rows of $B$*. so $\operatorname{rank}(AB) \le \operatorname{rank}(B)$ as well.

these views explain facts that look mysterious from the entry formula. a product of a tall-thin and a short-wide matrix, $(m \times r)(r \times n)$, has rank at most $r$: that's the structure of low-rank models, from recommender systems to LoRA. and $AB \ne BA$ in general (warm-up 3): "do $B$, then $A$" is a different map from "do $A$, then $B$". rotate-then-shear isn't shear-then-rotate (experiment 3).

the views also map onto computation. the column view does $n$ matrix-vector products, the row view $m$ vector-matrix products. in NumPy each inner update is a vectorized operation on a whole column or row, so both views replace one level of Python loop with fast compiled code.

> **🔮 predict:** your triple loop does $2n^3$ flops in $n^3$ Python iterations. the column version does the same flops in $n^2$ iterations, each a vectorized update of length $n$. at $n = 128$, roughly how much faster do you expect it to be?

> **▶ build: stage 2.** implement `matmul_columns(A, B)` (build each column of $C$ as a combination of $A$'s columns) and `matmul_rows(A, B)` (build each row of $C$ as a combination of $B$'s rows).

### idea 3: a sum of outer products

the fourth reading groups the sum by the *inner* index $p$ instead:

$$AB = \sum_{p=1}^k a_p\,\tilde b_p^\top,$$

where $a_p$ is column $p$ of $A$ and $\tilde b_p^\top$ is row $p$ of $B$. each term $a_p\tilde b_p^\top$ is an **outer product**: an $m \times n$ matrix of rank one, every column a multiple of $a_p$. so every product is a sum of $k$ rank-one matrices.

this is the view behind several later ideas:

- the **SVD** (topic 06) writes any matrix as $\sum_i\sigma_iu_iv_i^\top$, a sum of rank-one pieces in decreasing order of importance. keeping the first few is the best low-rank approximation.
- the gradient of a linear layer's loss with respect to its weights, over a batch, is $\sum_i\delta_ix_i^\top$, a sum of outer products of error and input vectors (topic 07, chapter 14).
- a covariance matrix is $\frac{1}{n}\sum_i x_ix_i^\top$ (for centered data), one outer product per sample.

computationally the outer-product version loops only $k$ times, and each iteration updates the whole $m \times n$ matrix at once.

> **▶ build: stage 3.** implement `matmul_outer(A, B)`: start from zeros and add `np.outer(A[:, p], B[p, :])` for each `p`.

### idea 4: linear maps, kernels and images

every linear map $f(x) = Ax$ from $\mathbb{R}^n$ to $\mathbb{R}^m$ comes with two subspaces (you computed both in topic 01):

- the **image** (column space) $\{Ax\}$, everything $f$ can output, of dimension $\operatorname{rank}A$;
- the **kernel** (null space) $\{x : Ax = 0\}$, everything $f$ destroys.

the **rank–nullity theorem** ties them together:

$$\dim\ker A + \dim\operatorname{im}A = n\quad(\text{the dimension of the input space}).$$

each input dimension either survives into the output or is crushed to zero. there's no third option. consequences: a map from $\mathbb{R}^5$ to $\mathbb{R}^3$ has image dimension at most 3, so its kernel has dimension at least 2. it can't be one-to-one, because $f(x) = f(x + z)$ for any $z$ in the kernel (warm-up 4). a square matrix is invertible exactly when its kernel is $\{0\}$. a projection onto a line in $\mathbb{R}^2$ has a 1-D image (the line) and a 1-D kernel (the perpendicular direction).

in code, a linear map often arrives as a *function*, not a matrix: a rotation, a finite-difference operator, a layer of a network. linearity gives a recipe for its matrix: apply it to each standard basis vector and stack the results as columns. and since a nonlinear function can't be represented that way, it's worth being able to *test* linearity numerically: check $f(ax + by) = af(x) + bf(y)$ at random points with random scalars. test additivity and scaling *together*, at random points: a test of scaling alone, $f(ax) = af(x)$ with $a > 0$, is passed by $f(x) = |x|$ and by ReLU, which are positively homogeneous but not additive. at two random points with mixed signs, $|x + y| \ne |x| + |y|$, so the full test catches them.

> **▶ build: stage 4.** implement `matrix_of(f, n)` (column $j$ is $f(e_j)$) and `is_linear(f, n, *, trials=20, tol=1e-9, rng=None)`.

### idea 5: blocks, and why BLAS is fast

**the block view.** cut $A$, $B$ and $C$ into tiles of size $b \times b$. then the tiles multiply like numbers:

$$C_{IJ} = \sum_K A_{IK}B_{KJ},$$

where each $A_{IK}B_{KJ}$ is a small matrix product. it's the entry formula one level up. it holds for any partition with compatible sizes, including ragged tiles at the edges.

**why it matters for speed.** on a modern CPU, arithmetic is cheap and moving data is expensive. one multiply-add takes a fraction of a nanosecond. fetching a number from main memory takes about a hundred times longer. a matrix product does $2n^3$ flops on $3n^2$ numbers, so each number can be reused about $n$ times *if it stays in fast memory (cache) while it's needed*. the triple loop doesn't arrange that. tiling does: a $b \times b$ tile of $A$ and one of $B$ fit in cache, and each of their entries is used $b$ times before eviction.

optimized libraries (the BLAS: OpenBLAS, MKL, Apple Accelerate) stack every trick on top of this: several levels of tiling matched to the cache sizes, hand-written inner kernels that use the CPU's vector instructions (several multiply-adds per instruction), careful memory layout, and all cores at once. the result runs within a small factor of the hardware's peak. matrix-*vector* products can't benefit the same way: they do $2n^2$ flops on $n^2$ numbers, each used once. that's why, flop for flop, they run much slower (experiment 2).

in your `matmul_blocked`, each tile product uses `@`. the point isn't to beat BLAS. it's to get the block algebra right, including the edge tiles, and to see how much of the speed comes just from doing big chunks of work per Python-level step.

> **🔮 predict:** `(A @ B) @ x` and `A @ (B @ x)` give the same vector. for $n = 1500$, how many times more flops does the first one do?

> **▶ build: stage 5.** implement `matmul_blocked(A, B, *, block=32)`: three loops over tile indices, accumulating `A[I, K] @ B[K, J]` into `C[I, J]`. sizes need not be multiples of `block`.

## worked example

$$A = \begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}\ (3 \times 2),\qquad B = \begin{pmatrix}1&0&-1\\2&1&0\end{pmatrix}\ (2 \times 3).$$

**entries.** $c_{21} = (3, 4)\cdot(1, 2) = 11$, and so on for all nine entries.

**columns.** column 1 of $C$ is $1\cdot a_1 + 2\cdot a_2 = (1,3,5) + (4,8,12) = (5, 11, 17)$.

**rows.** row 1 of $C$ is $1\cdot(1,0,-1) + 2\cdot(2,1,0) = (5, 2, -1)$.

**outer products.**

$$a_1\tilde b_1^\top + a_2\tilde b_2^\top = \begin{pmatrix}1&0&-1\\3&0&-3\\5&0&-5\end{pmatrix} + \begin{pmatrix}4&2&0\\8&4&0\\12&6&0\end{pmatrix} = \begin{pmatrix}5&2&-1\\11&4&-3\\17&6&-5\end{pmatrix} = C.$$

**rank–nullity.** $C$ is $3 \times 3$ but factors through $\mathbb{R}^2$, so $\operatorname{rank}C \le 2$, and its kernel has dimension at least 1. one kernel vector comes from $B$'s kernel: $B(1, -2, 1)^\top = (1 - 0 - 1,\ 2 - 2 + 0) = 0$, so $C(1, -2, 1)^\top = A\cdot 0 = 0$. check row 1: $5 - 4 - 1 = 0$. ✓

**matrix of a map.** rotation by 90° sends $e_1 \mapsto (0, 1)$ and $e_2 \mapsto (-1, 0)$, so its matrix is $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$, with the images as *columns*.

the tests check that all five of your products reproduce $C$, and that `matrix_of` reproduces the rotation.

## on paper

1. **composition is multiplication.** show from the definition $c_j = Ab_j$ that $(AB)x = A(Bx)$ for every $x$. then derive associativity $(AB)C = A(BC)$ without writing a single index.

<details><summary>solution</summary>

$(AB)x = \sum_jx_jc_j = \sum_jx_jAb_j = A\sum_jx_jb_j = A(Bx)$, using linearity of $x \mapsto Ax$. so $AB$ is the matrix of the composite map. then $(AB)C$ and $A(BC)$ are both the matrix of "apply $C$, then $B$, then $A$". equal maps have equal matrices (they agree on every $e_j$), so they're equal.

</details>

2. **rank of a product.** using the column and row views, show $\operatorname{rank}(AB) \le \min(\operatorname{rank}A, \operatorname{rank}B)$. give an example where it's strictly smaller than both.

<details><summary>solution</summary>

the columns of $AB$ are combinations of the columns of $A$, so $\operatorname{col}(AB) \subseteq \operatorname{col}(A)$. its rows are combinations of rows of $B$, so $\operatorname{row}(AB) \subseteq \operatorname{row}(B)$. take dimensions. example: $A = \begin{pmatrix}1&0\\0&0\end{pmatrix}$, $B = \begin{pmatrix}0&0\\0&1\end{pmatrix}$. both have rank 1, and $AB = 0$ has rank 0. $B$'s image is exactly $A$'s kernel.

</details>

3. **flop counts.** count the flops of $(AB)x$ and of $A(Bx)$ for $n \times n$ $A, B$. for a chain $A_1A_2\cdots A_Lx$ of $n \times n$ matrices, what's the cost of each strategy?

<details><summary>solution</summary>

$(AB)x$: $2n^3$ for $AB$ plus $2n^2$ for the matrix-vector product. $A(Bx)$: two matrix-vector products, $4n^2$. the ratio is about $n/2$. for a chain: right to left, it's $L$ matrix-vector products, $2Ln^2$. left to right, it's $(L-1)$ matrix-matrix products, about $2(L-1)n^3$. the cheap right-to-left order (push a vector through, never form a matrix product) is exactly how forward-mode autodiff computes Jacobian-vector products; reverse mode does the same from the other end, pushing a row vector $v^\top$ through left to right as vector-Jacobian products. in general, choosing the cheapest parenthesization (*matrix-chain ordering*) is an optimization problem in its own right.

</details>

4. **rank–nullity for a projection.** let $P = uu^\top$ with $\|u\| = 1$ in $\mathbb{R}^n$. find $\operatorname{im}P$ and $\ker P$, and verify rank–nullity.

<details><summary>solution</summary>

$Px = u(u^\top x)$ is always a multiple of $u$, and $Pu = u$, so $\operatorname{im}P = \operatorname{span}(u)$ (dimension 1). $Px = 0 \iff u^\top x = 0$, so $\ker P$ is the hyperplane perpendicular to $u$ (dimension $n - 1$). $1 + (n - 1) = n$. ✓

</details>

## experiment

`lab.py` runs three experiments on your code:

1. timing all your matmuls against `np.matmul` across sizes, in GFLOP/s.
2. timing $(AB)x$ against $A(Bx)$.
3. composing a rotation and a shear in both orders with your `matrix_of` and `matmul_entries`, and watching a projection collapse the plane onto a line.

## check your understanding

1. why is column $j$ of a matrix the image of $e_j$, and not row $j$?

<details><summary>answer</summary>

$Ae_j$ picks out column $j$: the combination of columns with weights $e_j$ is just the $j$-th column. rows are what you dot with $x$, to get one output coordinate at a time.

</details>

2. which of the four views would you use to explain why $\operatorname{rank}(uv^\top) = 1$?

<details><summary>answer</summary>

the column view (every column $v_ju$ is a multiple of $u$), or the outer-product view, where it's the $k = 1$ case of a sum of rank-one terms.

</details>

3. if $f$ is linear, why must $f(0) = 0$? which simple test catches $x \mapsto x + 1$?

<details><summary>answer</summary>

$f(0) = f(0\cdot x) = 0\cdot f(x) = 0$. so $x \mapsto x + 1$ fails right away. so does the scaling test: $f(2x) = 2x + 1 \ne 2f(x) = 2x + 2$.

</details>

4. all four matmul views do $2mnk$ flops. why aren't they equally fast in NumPy?

<details><summary>answer</summary>

they differ in how much work each *Python-level* step does: $mnk$ scalar steps for the triple loop, $nk$ (columns) or $mk$ (rows) vector steps, and $k$ matrix-sized steps for outer products. the interpreter overhead per step (about 50–100 ns, or more) dominates when each step does little. memory access patterns matter too.

</details>

5. a $1000 \times 50$ matrix times a $50 \times 1000$ matrix: what's the largest possible rank of the product? what does that say about the kernel of the product?

<details><summary>answer</summary>

rank at most 50. the product is $1000 \times 1000$, so by rank–nullity its kernel has dimension at least 950.

</details>

6. is the composition of two linear maps linear? is the inverse of an invertible linear map linear?

<details><summary>answer</summary>

yes: $f(g(ax + by)) = f(ag(x) + bg(y)) = af(g(x)) + bf(g(y))$. for the inverse, write $x = f(u)$, $y = f(v)$; then $f(au + bv) = ax + by$, so $f^{-1}(ax + by) = au + bv = af^{-1}(x) + bf^{-1}(y)$.

</details>

## common bugs

- **swapping the loop bounds.** `C = np.zeros((m, n))` with the wrong `k`. symptom: works for square matrices, crashes or silently truncates for rectangular ones. test with three distinct sizes.
- **accumulating into an integer array.** `C = np.zeros_like(A)` inherits `A`'s dtype. symptom: products of integer matrices are fine, but float results get truncated. allocate `np.zeros((m, n))`, which is float.
- **forgetting to reset the running sum.** in the triple loop, `total` must restart at 0 for each $(i, j)$. symptom: entries that grow along each row.
- **rows of the map instead of columns.** `np.vstack([f(e) for e in ...])` gives the transpose. symptom: `matrix_of` is right only for symmetric maps.
- **testing only scaling by positive numbers.** symptom: $|x|$ and ReLU pass as "linear". test $f(ax + by) = af(x) + bf(y)$ at random points with random scalars, which checks additivity too.
- **dropping edge tiles.** `range(0, m - block, block)` misses the last partial tile. symptom: wrong values in the last rows or columns whenever the size isn't a multiple of the block.

## going deeper

- Strang (2016), *Introduction to Linear Algebra*, 5th ed., Wellesley-Cambridge Press, §1.3 and chapter 2 (matrix multiplication and its four ways) and §8.1–8.2 (linear transformations).
- Axler (2024), *Linear Algebra Done Right*, 4th ed., Springer, chapter 3 (linear maps, null spaces and ranges, the fundamental theorem of linear maps). [doi:10.1007/978-3-031-41026-0](https://doi.org/10.1007/978-3-031-41026-0)
- Goto & van de Geijn (2008), "Anatomy of high-performance matrix multiplication", *ACM Transactions on Mathematical Software* 34(3): 12. how BLAS really does it. [doi:10.1145/1356052.1356053](https://doi.org/10.1145/1356052.1356053)
- Strassen (1969), "Gaussian elimination is not optimal", *Numerische Mathematik* 13: 354–356. a product in fewer than $n^3$ multiplications. [doi:10.1007/BF02165411](https://doi.org/10.1007/BF02165411)
- Golub & Van Loan (2013), *Matrix Computations*, 4th ed., Johns Hopkins University Press, chapter 1 (matrix multiplication: the views, blocking and the memory hierarchy).

## connections

- ← builds on: span, rank and null space (topic 01), vectorization (ch 00 · topic 02).
- → used in: elimination as a product of matrices (topic 03), projections $QQ^\top$ (topic 04), diagonalization $A = V\Lambda V^{-1}$ (topic 05), the SVD as a sum of rank-one pieces (topic 06), gradients of linear layers (topic 07, ch 14), and every network layer from ch 14 on.
