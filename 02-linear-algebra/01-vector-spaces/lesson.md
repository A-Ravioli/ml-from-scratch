# vector spaces

## why this matters

a dataset of 10,000 images, each with 784 pixels, is 10,000 points in $\mathbb{R}^{784}$. but handwritten digits don't fill that space. they lie near a much smaller set of directions, and much of machine learning is about finding them. before you can find low-dimensional structure, you need precise words for it: when are some vectors "really" fewer vectors? how many directions does a set of data actually use? which inputs does a linear model ignore completely?

those are questions about **span**, **independence**, **basis**, **dimension** and **subspaces**: the vocabulary of every later topic. this topic makes each one computable with a single algorithm you already half know: row reduction. you'll build a reduced-row-echelon-form routine and read everything off it: the rank of a matrix, whether a vector is in a span, whether vectors form a basis, the coordinates of a vector in that basis, and a basis for the null space.

there's a catch, and it's the first appearance of a theme that runs through the whole chapter. in floating point, "is this exactly zero?" is the wrong question. so "what is the rank?" turns out to depend on a tolerance you choose. the lab makes that vivid.

## warm-up: predict first

1. three vectors in $\mathbb{R}^3$, none a multiple of another. must they span $\mathbb{R}^3$?
2. a $5 \times 8$ matrix: what's the largest its rank can be? the smallest its null space's dimension can be?
3. you add a vector to a set and the rank doesn't change. what does that tell you about the new vector?
4. in $\mathbb{R}^{1000}$, what's the typical angle between two random vectors?
5. is the set of vectors $(x, y)$ with $x + y = 1$ a subspace of $\mathbb{R}^2$?

## prerequisites check

- [ ] you can add vectors, scale them, and take dot products (→ ch 01 · topic 04)
- [ ] you can solve a 3×3 system by elimination by hand (→ ch 01 · topic 04)
- [ ] you can slice, index and update NumPy arrays in place, including whole rows (→ ch 00 · topic 02)
- [ ] you know that floating point arithmetic rounds, so $0.1 + 0.2 \ne 0.3$ exactly (→ ch 00 · topic 03)

## core ideas

### idea 1: row reduction, the one algorithm behind it all

a **linear combination** of vectors $v_1, \dots, v_k$ is $c_1v_1 + \dots + c_kv_k$, and their **span** is the set of all such combinations. stack the vectors as the columns of a matrix $V$, and a linear combination is just $Vc$. almost every question in this topic is a question about the solutions of $Vc = x$ or $Vc = 0$, and Gaussian elimination answers those.

three **row operations** (swap two rows, scale a row by a nonzero number, add a multiple of one row to another) never change the solutions of $Ax = b$, because each one can be undone. apply them systematically, column by column, and you reach the **reduced row echelon form** (RREF):

- each nonzero row starts with a 1, the **pivot**, further right than the pivot of the row above;
- each pivot is the only nonzero entry in its column;
- all-zero rows sit at the bottom.

for example,

$$\begin{pmatrix}1&2&3\\2&4&7\\1&2&4\end{pmatrix} \;\longrightarrow\; \begin{pmatrix}1&2&0\\0&0&1\\0&0&0\end{pmatrix}, \qquad\text{pivots in columns } 1 \text{ and } 3.$$

(the math in this lesson counts rows and columns from 1. NumPy counts from 0, so in code these pivots are `(0, 2)`.)

the RREF of a matrix is unique, whatever order you do the operations in. the columns *with* pivots are the **pivot columns**. the others are **free columns**, and the free columns are where all the interesting structure lives.

**in floating point** two refinements matter. first, choose as pivot the entry of largest magnitude in the column (on or below the current row). this is *partial pivoting*, and it avoids dividing by small numbers that magnify rounding errors (topic 03 shows how badly that can go). second, decide "is this zero?" with a tolerance: after a few eliminations, an entry that should be $0$ is typically $10^{-16}$, and treating it as a pivot would be wrong. so entries with $|a| \le \text{tol}$ count as zero.

> **▶ build: stage 1.** implement `rref(A, *, tol=1e-10)` in `exercise.py` (`marimo edit exercise.py`). return the float RREF and a tuple of pivot columns. for each column: pick the largest-magnitude candidate, skip the column if it's within `tol` of zero, otherwise swap it up, scale it to 1, and clear the column in every other row.

### idea 2: independence and rank

vectors are **linearly independent** if the only combination giving zero is the trivial one: $Vc = 0 \Rightarrow c = 0$. otherwise some vector is a combination of the others and adds nothing new to the span. in the RREF, every free column is exactly such a redundant vector. in the example, column 2 is $2\times$ column 1, and the RREF records that as the entry $2$ in row 1 of column 2.

the **rank** of a matrix is its number of pivots. it's simultaneously

- the number of independent columns (the dimension of the column space),
- the number of independent rows (the dimension of the row space),
- the dimension of the set of outputs $\{Ax\}$.

"row rank = column rank" is a genuine theorem, and RREF makes it believable: row operations preserve the row space *and* the dependencies among the columns, and each pivot accounts for one independent row and one independent column. so a $5 \times 8$ matrix has rank at most 5 (warm-up 2).

rank gives a one-line test for span membership (warm-up 3). $x$ is in the span of $V$'s columns exactly when appending it doesn't create a new pivot:

$$x \in \operatorname{span}(V) \iff \operatorname{rank}[\,V \;\; x\,] = \operatorname{rank}V.$$

if the rank went up, $x$ points in a new direction. otherwise it was already reachable.

> **🔮 predict:** a $30 \times 20$ matrix is built as (30×3)·(3×20), so it has rank exactly 3. add noise of size $10^{-8}$ to every entry. what rank will your function report?

> **▶ build: stage 2.** implement `rank(A, *, tol=1e-10)` (count the pivots) and `in_span(V, x, *, tol=1e-10)`.

### idea 3: bases, dimension and coordinates

a **basis** of a space is a set of vectors that is independent *and* spans it: enough vectors to reach everything, none wasted. the central fact of the subject is that **every basis of a space has the same number of vectors**. that number is the space's **dimension**. so a basis of $\mathbb{R}^n$ has exactly $n$ vectors. fewer can't span, and more can't be independent. that's why three vectors that aren't multiples of each other still needn't span $\mathbb{R}^3$: they can all lie in one plane (warm-up 1). pairwise "not parallel" is weaker than independent.

for an $n \times n$ matrix $V$, these are all equivalent: the columns form a basis, $\operatorname{rank}V = n$, the RREF is the identity, $Vc = x$ has exactly one solution for every $x$, and $V$ is invertible.

that unique solution gives the **coordinates** of $x$ in the basis: $x = c_1v_1 + \dots + c_nv_n$. the standard coordinates $(x_1, \dots, x_n)$ are just coordinates in the standard basis $e_1, \dots, e_n$. a different basis describes the same vector with different numbers. finding the "right" basis, one where your data or your matrix looks simple, is the theme of topics 05 (eigenvectors) and 06 (singular vectors).

computationally: row-reduce the augmented matrix $[V \mid x]$. if $V$ is a basis, the left block becomes $I$ and the last column becomes $c$.

> **🔮 predict:** fill a $d \times d$ matrix with random coin flips, $\pm1$. for $d = 2$, how often are its columns a basis? does that get better or worse for $d = 4$? (the lab finds out.)

> **▶ build: stage 3.** implement `is_basis(V, *, tol=1e-10)` and `coordinates(V, x)`, which raises `ValueError` when `V`'s columns aren't a basis.

### idea 4: subspaces and the null space

a **subspace** is a set of vectors closed under addition and scaling. equivalently, it's closed under linear combinations, so it always contains $0$. lines and planes through the origin are subspaces of $\mathbb{R}^3$. a line *not* through the origin isn't (warm-up 5: $x + y = 1$ fails, since $(0,0)$ isn't in it, and doubling $(1, 0)$ leaves the set). every span is a subspace, and every subspace of $\mathbb{R}^n$ is the span of some basis.

two subspaces come with every matrix $A$ ($m \times n$):

- the **column space** $\{Ax\} \subseteq \mathbb{R}^m$, of dimension $\operatorname{rank}A$. a basis: the pivot columns *of $A$ itself*. (not of the RREF! row operations change the column space, but they preserve which columns are independent.)
- the **null space** $\{x : Ax = 0\} \subseteq \mathbb{R}^n$: the inputs that $A$ sends to zero, the directions a linear model is blind to.

the RREF hands you a null-space basis. each free variable can be chosen freely, and the pivot variables are then determined. setting one free variable to 1 and the others to 0 gives one basis vector per free column. if $R$ has pivots in columns $p_1, p_2, \dots$ (in rows $1, 2, \dots$) and $f$ is a free column, the vector $z$ has

$$z_f = 1,\qquad z_{p_i} = -R_{i,f},\qquad \text{all other free entries } 0.$$

counting columns gives the **rank–nullity theorem**: pivot columns plus free columns make $n$, so

$$\dim\operatorname{null}(A) = n - \operatorname{rank}(A).$$

a $5 \times 8$ matrix therefore has a null space of dimension at least $3$ (warm-up 2): some 3-D family of inputs is invisible to it. topic 02 explains this from the linear-map point of view.

> **▶ build: stage 4.** implement `null_space(A, *, tol=1e-10)`, returning a basis as the columns of an $(n, n - \text{rank})$ array.

## worked example

take $A = \begin{pmatrix}1&2&3\\2&4&7\\1&2&4\end{pmatrix}$, with columns $a_1, a_2, a_3$.

**RREF with partial pivoting.** column 1: the largest entry is the 2 in row 2, so swap it to the top and scale: $(1, 2, 3.5)$. eliminate: row $(1,2,3) - (1,2,3.5) = (0, 0, -0.5)$ and row $(1,2,4) - (1,2,3.5) = (0, 0, 0.5)$. column 2 has nothing nonzero at or below row 2, so it's **free**. column 3: the candidates are $-0.5$ and $0.5$, a tie, so take the first. scale: $(0,0,1)$. clear it from the other rows: $(1, 2, 3.5) - 3.5(0,0,1) = (1, 2, 0)$ and $(0,0,0.5) - 0.5(0,0,1) = 0$. so

$$R = \begin{pmatrix}1&2&0\\0&0&1\\0&0&0\end{pmatrix},\qquad\text{pivots in columns } 1, 3 \text{ (in code: } (0, 2)).$$

**rank** $= 2$, so the columns are dependent and $A$ is not a basis of $\mathbb{R}^3$. the column space is the plane spanned by $a_1 = (1,2,1)$ and $a_3 = (3,7,4)$, the pivot columns of $A$.

**null space.** column 2 is free. set $x_2 = 1$, then row 1 of $R$ says $x_1 + 2x_2 = 0$ and row 2 says $x_3 = 0$, so $z = (-2, 1, 0)$. check: $-2a_1 + a_2 = -2(1,2,1) + (2,4,2) = 0$. ✓ rank–nullity: $2 + 1 = 3$ columns.

**span membership.** is $(0, 0, 1)$ in the column space? solving $\alpha a_1 + \beta a_3 = (0,0,1)$: the first two equations force $\alpha = \beta = 0$, and then the third says $0 = 1$. so no: appending it raises the rank to 3.

**coordinates.** in the basis $v_1 = (1, 1)$, $v_2 = (1, -1)$ of $\mathbb{R}^2$, the vector $(3, 1)$ has coordinates $(2, 1)$, since $2(1,1) + 1(1,-1) = (3,1)$.

the tests check $R$, the pivots, the rank, the null-space vector $(-2, 1, 0)$ and the coordinates $(2, 1)$.

## on paper

1. **the null space is a subspace.** show that if $Ax = 0$ and $Ay = 0$, then $A(\alpha x + \beta y) = 0$. is the solution set of $Ax = b$ with $b \ne 0$ a subspace?

<details><summary>solution</summary>

$A(\alpha x + \beta y) = \alpha Ax + \beta Ay = 0$. for $b \ne 0$, no: $x = 0$ isn't a solution, and if $Ax = Ay = b$ then $A(x + y) = 2b \ne b$. the solution set is a *shifted* copy $x_p + \operatorname{null}(A)$ of the null space (one particular solution plus anything $A$ ignores).

</details>

2. **more than $n$ vectors in $\mathbb{R}^n$ are dependent.** using RREF, explain why any $n + 1$ vectors in $\mathbb{R}^n$ (the columns of an $n \times (n+1)$ matrix) must be dependent.

<details><summary>solution</summary>

there are at most $n$ pivots (at most one per row), but $n + 1$ columns, so at least one column is free. setting that free variable to 1 gives a nonzero solution of $Vc = 0$, which is a nontrivial dependency.

</details>

3. **coordinates are unique.** show that if the $v_i$ are independent and $\sum c_iv_i = \sum d_iv_i$, then $c = d$.

<details><summary>solution</summary>

subtract: $\sum(c_i - d_i)v_i = 0$. independence forces every $c_i - d_i = 0$.

</details>

4. **row operations preserve column dependencies.** let $E$ be an invertible matrix (a product of row operations). show that $Vc = 0 \iff (EV)c = 0$, so $V$ and its RREF $R = EV$ have the same dependencies among columns. why does this justify using the pivot columns of $A$ (not of $R$) as a column-space basis?

<details><summary>solution</summary>

if $Vc = 0$ then $EVc = 0$. conversely, if $EVc = 0$, multiply by $E^{-1}$ to get $Vc = 0$. so a set of columns of $V$ is independent exactly when the same columns of $R$ are. in $R$, the pivot columns are distinct unit vectors (independent), and every free column is a combination of them. the same is then true for $A$'s columns. $R$'s columns themselves span a *different* space (the RREF of the worked example has zero third coordinates everywhere), which is why you take the columns from $A$.

</details>

## experiment

`lab.py` runs three experiments:

1. the distribution of angles between random vectors in $\mathbb{R}^d$ as $d$ grows.
2. the rank your `rank` reports for a rank-3 matrix plus small noise, as a function of the tolerance.
3. how often $d$ random coin-flip vectors in $\mathbb{R}^d$ form a basis (asked of your `is_basis`), compared with Gaussian vectors.

## check your understanding

1. why must every subspace contain the zero vector?

<details><summary>answer</summary>

it's closed under scaling, and $0 \cdot v = 0$ for any $v$ in it. (the empty set is excluded by definition.)

</details>

2. a matrix has RREF with pivots in columns $1, 2, 4$ out of $5$ columns. what's its rank, the dimension of its null space, and how many vectors are in a null-space basis?

<details><summary>answer</summary>

rank 3. nullity $5 - 3 = 2$, so 2 basis vectors, one each for free columns 3 and 5.

</details>

3. why does `rref` need a tolerance, but a pen-and-paper RREF doesn't?

<details><summary>answer</summary>

on paper, arithmetic is exact, so a would-be-zero entry *is* zero. in floating point, cancellation leaves residues around $10^{-16}$ times the size of the numbers involved. without a tolerance, such a residue becomes a pivot, inflating the rank and producing nonsense rows scaled by $10^{16}$.

</details>

4. is "rank" a continuous function of the matrix?

<details><summary>answer</summary>

no. arbitrarily small perturbations can increase it: rank-3 plus $10^{-15}$ noise has full rank mathematically. it can never decrease under small perturbations, though. the singular values (topic 06) are continuous and tell you *how far* a matrix is from each lower rank, which is the robust replacement.

</details>

5. vectors $u, v, w$ are pairwise independent (no two parallel). are they independent?

<details><summary>answer</summary>

not necessarily: $(1,0)$, $(0,1)$, $(1,1)$ are pairwise non-parallel but dependent ($w = u + v$). independence is a property of the whole set.

</details>

6. in experiment 1, why is near-orthogonality of random vectors good news for some ML methods and bad news for others?

<details><summary>answer</summary>

good: random projections and random features keep distances roughly intact (chapter 09), and you can pack exponentially many nearly orthogonal directions into a space, which is one explanation of how networks store many features in superposition (chapter 28). bad: distances between random points concentrate, so "nearest" neighbours aren't much nearer than anything else. that's the curse of dimensionality for kNN (chapter 09).

</details>

## common bugs

- **integer input.** `R[row] /= R[row, col]` on an integer array truncates. symptom: wrong RREF for hand examples typed as integers. convert with `np.array(A, dtype=float)`.
- **clearing only below the pivot.** that gives row echelon form, not *reduced*. symptom: correct pivots, but nonzero entries above them, and wrong coordinates and null-space vectors.
- **advancing the row counter on a free column.** when a column has no pivot, move to the next column but stay on the same row. symptom: too few pivots, rank too low.
- **taking basis vectors from $R$.** the column space of $A$ is spanned by the pivot columns of $A$, not of $R$.
- **comparing with `== 0`.** symptom: rank always full on real data. use the tolerance.

## going deeper

- Strang (2016), *Introduction to Linear Algebra*, 5th ed., Wellesley-Cambridge Press, chapter 3 (vector spaces and subspaces: the four subspaces, rank, independence, basis and dimension).
- Axler (2024), *Linear Algebra Done Right*, 4th ed., Springer, chapter 2 (finite-dimensional vector spaces). the abstract, determinant-free treatment, free online. [doi:10.1007/978-3-031-41026-0](https://doi.org/10.1007/978-3-031-41026-0)
- Vershynin (2018), *High-Dimensional Probability*, Cambridge University Press, chapter 3. why random vectors in high dimensions are nearly orthogonal (experiment 1), made precise. [doi:10.1017/9781108231596](https://doi.org/10.1017/9781108231596)
- Tikhomirov (2020), "Singularity of random Bernoulli matrices", *Annals of Mathematics* 191(2): 593–634. the probability that a random $\pm1$ matrix is singular is $(1/2 + o(1))^n$: experiment 3's asymptotics. [arXiv:1812.09016](https://arxiv.org/abs/1812.09016)

## connections

- ← builds on: vectors and matrices (ch 01 · topic 04), NumPy arrays (ch 00 · topic 02), floating point (ch 00 · topic 03).
- → used in: kernel, image and rank–nullity for linear maps (topic 02), Gaussian elimination as a factorization (topic 03), orthonormal bases (topic 04), eigenbases (topic 05), numerical rank via singular values (topic 06), feature collinearity in regression (ch 08), the curse of dimensionality (ch 09).
