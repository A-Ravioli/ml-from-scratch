# solving linear systems

## why this matters

fitting a linear regression comes down to solving $X^\top X\, w = X^\top y$. a Newton step in optimization solves $H\,d = -g$. sampling from a multivariate Gaussian needs a matrix $L$ with $LL^\top = \Sigma$. a Gaussian process prediction solves $K\alpha = y$. underneath a lot of machine learning, the expensive line is "solve $Ax = b$".

the method you learned in school, Gaussian elimination, is the right algorithm. but the textbook version has two problems. it fails on matrices that are perfectly harmless, like $\begin{pmatrix}0 & 1\\ 1 & 1\end{pmatrix}$, and it gives badly wrong answers on matrices that are *almost* like that one. and if you have to solve with the same $A$ for a hundred different right-hand sides, it redoes all the expensive work a hundred times.

this topic fixes both. you'll rewrite elimination as a **factorization** $PA = LU$: do the expensive work once, then each solve is cheap. you'll add **pivoting**, so that elimination never divides by a tiny number. and you'll build the faster, more specialized **Cholesky** factorization for symmetric positive definite matrices, the kind that covariance matrices, kernel matrices and Hessians at a nondegenerate minimum all are.

what you write here becomes `mlfs.linalg`. chapter 04 samples Gaussians with your `cholesky`, chapter 06 takes Newton steps with your `solve`, and chapter 08 fits regressions with both.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. solving a triangular $n \times n$ system takes about how many arithmetic operations: $n$, $n^2$, or $n^3$?
2. you need to solve $Ax = b$ for 100 different vectors $b$ with the same $n \times n$ matrix $A$. which is cheaper: computing $A^{-1}$ once and multiplying, or something else?
3. elimination on $\begin{pmatrix}10^{-20} & 1\\ 1 & 1\end{pmatrix}x = \begin{pmatrix}1\\2\end{pmatrix}$ in floating point: the exact answer is very close to $x = (1, 1)$. what does textbook elimination (no row swaps) return?
4. the cofactor expansion of a determinant needs about $n!$ multiplications. how would you compute $\det A$ for a $100 \times 100$ matrix instead?
5. is every symmetric matrix with positive diagonal entries positive definite?

## prerequisites check

- [ ] you can row-reduce a small matrix by hand and read off its rank (→ ch 02 · topic 01)
- [ ] you can read matrix-vector products as combinations of columns, and know that $(AB)^{-1} = B^{-1}A^{-1}$ (→ ch 02 · topic 02)
- [ ] you know floating point numbers carry about 16 significant digits, and that $1 + 10^{-17}$ rounds to $1$ (→ ch 00 · topic 03)
- [ ] you can write a NumPy loop that updates a slice of an array in place (→ ch 00 · topic 02)

## core ideas

### idea 1: triangular systems are easy

some systems need no cleverness at all. if $L$ is lower-triangular,

$$\begin{pmatrix}\ell_{11} & 0 & 0\\ \ell_{21} & \ell_{22} & 0\\ \ell_{31} & \ell_{32} & \ell_{33}\end{pmatrix}\begin{pmatrix}x_1\\x_2\\x_3\end{pmatrix} = \begin{pmatrix}b_1\\b_2\\b_3\end{pmatrix},$$

the first equation involves only $x_1$, so $x_1 = b_1/\ell_{11}$. the second involves only $x_1$ (now known) and $x_2$, so $x_2 = (b_2 - \ell_{21}x_1)/\ell_{22}$. in general, **forward substitution** is

$$x_i = \frac{1}{\ell_{ii}}\Big(b_i - \sum_{j<i}\ell_{ij}x_j\Big), \qquad i = 1, 2, \dots, n.$$

an upper-triangular system runs the same way from the bottom up: **back substitution**, $x_i = (b_i - \sum_{j>i}u_{ij}x_j)/u_{ii}$ for $i = n, \dots, 1$.

row $i$ costs about $2i$ operations (a dot product of length $i$), so the whole solve costs about $\sum_i 2i \approx n^2$. that answers warm-up 1: $n^2$, the same order as just reading the matrix. a triangular solve fails only if some diagonal entry is zero, which is exactly when a triangular matrix is singular (its determinant is the product of its diagonal).

two details make your function more useful later. first, $b$ may be a matrix $B$ of shape $(n, k)$, meaning $k$ right-hand sides at once. the same formula works if you treat $x_i$ as a whole row of $X$, and NumPy's `@` handles it for you. second, a flag `unit_diagonal=True` says "the diagonal is all ones; don't even look at it". you'll need that for the $L$ factor of LU, which has an implicit unit diagonal.

> **▶ build: stage 1.** implement `solve_triangular(T, b, *, lower=True, unit_diagonal=False)` in `exercise.py` (`marimo edit exercise.py`). it must read only the relevant triangle, accept `b` of shape `(n,)` or `(n, k)`, and raise `np.linalg.LinAlgError` on a zero pivot.

### idea 2: elimination is a factorization

Gaussian elimination subtracts multiples of a pivot row from the rows below it, until the matrix is upper-triangular. here's the key observation: **each of those row operations is a matrix multiplication**. subtracting $\ell_{21}$ times row 1 from row 2 is multiplying on the left by

$$E_{21} = \begin{pmatrix}1 & 0 & 0\\ -\ell_{21} & 1 & 0\\ 0 & 0 & 1\end{pmatrix}.$$

eliminating column 1 multiplies by a unit lower-triangular $E_1$ that holds all of column 1's multipliers, and so on. after $n-1$ steps, $E_{n-1}\cdots E_2E_1 A = U$, which is upper-triangular. move the $E$'s to the other side:

$$A = \underbrace{E_1^{-1}E_2^{-1}\cdots E_{n-1}^{-1}}_{L}\;U.$$

two facts make this practical (you'll prove both on paper). inverting $E_k$ just flips the signs of its multipliers. and multiplying the inverses in this order doesn't mix them: $L$ is simply the unit lower-triangular matrix with $\ell_{ik}$ (the multiplier that eliminated row $i$ using pivot row $k$) in position $(i, k)$. **elimination already computes $L$.** you just have to store the multipliers instead of throwing them away.

the cost: step $k$ updates an $(n-k) \times (n-k)$ block, about $2(n-k)^2$ operations, so the total is $\sum_k 2(n-k)^2 \approx \tfrac{2}{3}n^3$. the factorization is the expensive part.

**pivoting.** step $k$ divides by the pivot $u_{kk}$. if it's zero, elimination stops, even when $A$ is invertible: $\begin{pmatrix}0&1\\1&1\end{pmatrix}$ is a perfectly good matrix. the fix is to swap rows so that a nonzero entry sits in the pivot position. in floating point, "nonzero" isn't enough. take warm-up 3:

$$\begin{pmatrix}\delta & 1\\ 1 & 1\end{pmatrix},\quad \delta = 10^{-20}: \qquad \ell_{21} = 10^{20},\qquad u_{22} = 1 - 10^{20}\cdot 1 \;\to\; -10^{20}\ \text{(the 1 is rounded away)}.$$

the "1" in the bottom-right corner of $A$ has been wiped out by rounding. the factors now describe a different matrix, and back substitution confidently returns $x_1 = 0$ instead of $1$. the root cause is the huge multiplier. **partial pivoting** fixes it: at step $k$, swap up the row with the largest $|u_{ik}|$ in column $k$ (on or below the diagonal). then every multiplier satisfies $|\ell_{ik}| \le 1$, and nothing gets amplified at the moment of elimination.

each swap is a permutation matrix applied on the left. collecting them gives

$$PA = LU,$$

where $P$ is a permutation matrix (the identity with its rows reordered). there's one bookkeeping trap. when you swap rows $k$ and $p$ of the working matrix at step $k$, the multipliers you've already stored in columns $1, \dots, k-1$ of $L$ belong to those rows, so they have to be swapped too.

> **🔮 predict:** partial pivoting guarantees $|\ell_{ik}| \le 1$. does that guarantee the entries of $U$ stay about as large as those of $A$? (the lab finds out.)

> **▶ build: stage 2.** implement `lu(A)` returning `(P, L, U)` with `P @ A == L @ U`. use the largest-magnitude pivot (ties go to the earliest row), don't modify `A`, and let singular matrices factor without crashing.

### idea 3: factor once, solve many

with $PA = LU$, solving $Ax = b$ is two triangular solves:

$$Ax = b \iff PAx = Pb \iff L\underbrace{(Ux)}_{y} = Pb: \qquad \text{solve } Ly = Pb,\ \text{then } Ux = y.$$

the factorization costs $\tfrac{2}{3}n^3$. each later solve costs $2n^2$. so for warm-up 2: factor once, then do 100 cheap solves. what about the inverse? forming $A^{-1}$ costs about $2n^3$ (an LU plus $n$ solves), then each product $A^{-1}b$ costs $2n^2$, so it's three times the up-front work for no gain. it's also *less accurate*. numerical analysts have a saying: **you almost never need the inverse; you need its action on a vector.** whenever you see $A^{-1}b$ in a formula, read it as "solve".

this is also why your API passes around `factors = (P, L, U)` instead of a matrix: `lu_solve(factors, b)` reuses the work.

> **▶ build: stage 3.** implement `lu_solve(factors, b)` (two calls to your `solve_triangular`) and `solve(A, b) = lu_solve(lu(A), b)`. a singular $A$ should raise `LinAlgError`, which happens automatically when back substitution hits the zero pivot.

### idea 4: the determinant comes for free

the determinant is the factor by which $A$ scales volumes: the unit cube maps to a parallelepiped of volume $|\det A|$, and the sign says whether orientation flips. it's zero exactly when $A$ squashes space into a lower dimension, which is when $A$ is singular.

the determinant is multiplicative, $\det(XY) = \det X\,\det Y$, and the determinant of a triangular matrix is the product of its diagonal. apply both to $PA = LU$:

$$\det P\cdot\det A = \det L\cdot\det U = 1\cdot\prod_i u_{ii} \quad\Longrightarrow\quad \det A = \det(P)\prod_i u_{ii}.$$

$\det P = \pm 1$: each row swap flips the sign, so it's $(-1)^{\#\text{swaps}}$. if you only have $P$, you can count swaps through the cycles of the permutation: a cycle of length $c$ takes $c-1$ swaps to undo. so that's warm-up 4: an LU in $\tfrac{2}{3}n^3$ operations, about $7 \times 10^5$ for $n = 100$, against $100! \approx 10^{158}$ for cofactor expansion.

one warning for later chapters: for large $n$, $\prod u_{ii}$ easily overflows or underflows. ML code almost always wants $\log|\det A| = \sum_i \log|u_{ii}|$ instead (chapter 04 uses this for Gaussian log-densities).

> **🔮 predict:** a $100 \times 100$ matrix with independent standard normal entries is very unlikely to be singular. roughly what size is its determinant: about 1, about $10^{10}$, or about $10^{80}$?

> **▶ build: stage 4.** implement `det(A)` from your `lu`. get the sign of $P$ right.

### idea 5: Cholesky, the symmetric positive definite case

a symmetric matrix $A$ is **positive definite** (SPD) if $v^\top A v > 0$ for every $v \ne 0$. geometrically, the quadratic $v^\top A v$ is an upward bowl in every direction. covariance matrices are SPD (or semidefinite), and so are kernel matrices, $X^\top X$ for full-rank $X$, and Hessians at a nondegenerate minimum (one where the Hessian has no zero eigenvalue). for these matrices elimination simplifies: you never need pivoting, and you can factor symmetrically,

$$A = LL^\top,$$

with $L$ lower-triangular with positive diagonal. that's the **Cholesky factorization**. compare entries of $A = LL^\top$ in column $j$ and you can solve for $L$ one column at a time:

$$\ell_{jj} = \sqrt{a_{jj} - \sum_{k<j}\ell_{jk}^2}, \qquad \ell_{ij} = \frac{1}{\ell_{jj}}\Big(a_{ij} - \sum_{k<j}\ell_{ik}\ell_{jk}\Big)\quad (i > j).$$

it takes about $\tfrac{1}{3}n^3$ operations, half of LU, because symmetry means you only compute one triangle. and it's stable without pivoting, because $\sum_k \ell_{jk}^2 \le a_{jj}$ keeps every entry of $L$ bounded by $\sqrt{\max_j a_{jj}}$.

the square root is where non-SPD matrices give themselves away. if the quantity under the root is $\le 0$, $A$ isn't positive definite. that makes `cholesky` the cheapest reliable test for positive definiteness: just try it. it also answers warm-up 5. $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ has a positive diagonal, but the second pivot would be $1 - 2^2 = -3$. it's indefinite: $v = (1, -1)$ gives $v^\top A v = -2$.

solving with the factor is two triangular solves again: $LL^\top x = b$ means $Ly = b$, then $L^\top x = y$.

> **▶ build: stage 5.** implement `cholesky(A)` (raise `LinAlgError` for non-symmetric or non-positive-definite input) and `cho_solve(L, b)`.

## worked example

solve $Ax = b$ with

$$A = \begin{pmatrix}2 & 1 & 1\\ 4 & -6 & 0\\ -2 & 7 & 2\end{pmatrix}, \qquad b = \begin{pmatrix}5\\-2\\9\end{pmatrix}.$$

**column 1.** the largest $|a_{i1}|$ is the 4 in row 2, so swap rows 1 and 2. the multipliers are $\ell_{21} = 2/4 = 0.5$ (old row 1) and $\ell_{31} = -2/4 = -0.5$. eliminate:

$$\begin{pmatrix}4 & -6 & 0\\ 0 & 4 & 1\\ 0 & 4 & 2\end{pmatrix}\qquad\text{since } (2,1,1) - 0.5(4,-6,0) = (0,4,1),\ \ (-2,7,2) + 0.5(4,-6,0) = (0,4,2).$$

**column 2.** the candidates are $|4|$ and $|4|$, a tie, so keep the earlier row (no swap). $\ell_{32} = 4/4 = 1$, and the last row becomes $(0,4,2) - (0,4,1) = (0,0,1)$. so

$$P = \begin{pmatrix}0&1&0\\1&0&0\\0&0&1\end{pmatrix},\quad L = \begin{pmatrix}1&0&0\\0.5&1&0\\-0.5&1&1\end{pmatrix},\quad U = \begin{pmatrix}4&-6&0\\0&4&1\\0&0&1\end{pmatrix}.$$

check one entry: row 3 of $LU$ is $-0.5(4,-6,0) + 1(0,4,1) + 1(0,0,1) = (-2, 7, 2)$, which is row 3 of $PA$. ✓

**solve.** $Pb = (-2, 5, 9)$. forward substitution: $y_1 = -2$, $y_2 = 5 - 0.5(-2) = 6$, $y_3 = 9 + 0.5(-2) - 1(6) = 2$. back substitution: $x_3 = 2$, $x_2 = (6 - 1\cdot 2)/4 = 1$, $x_1 = (-2 + 6\cdot 1)/4 = 1$. so $x = (1, 1, 2)$.

**determinant.** one swap, so $\det P = -1$, and $\det A = -1 \cdot (4 \cdot 4 \cdot 1) = -16$. (cofactor expansion along row 1 gives $2(-12) - 1(8) + 1(16) = -16$. ✓)

**Cholesky.** for $S = \begin{pmatrix}4&2&2\\2&5&3\\2&3&6\end{pmatrix}$: $\ell_{11} = 2$, $\ell_{21} = 2/2 = 1$, $\ell_{31} = 1$; $\ell_{22} = \sqrt{5 - 1} = 2$, $\ell_{32} = (3 - 1\cdot 1)/2 = 1$; $\ell_{33} = \sqrt{6 - 1 - 1} = 2$. so $L = \begin{pmatrix}2&0&0\\1&2&0\\1&1&2\end{pmatrix}$, and $Sx = (8, 10, 11)$ has solution $x = (1,1,1)$.

the tests check every number here: your `lu`, `solve`, `det`, `cholesky` and `cho_solve` must reproduce them exactly.

## on paper

1. **inverting an elimination matrix.** let $E = I - \ell e_k^\top$, where $\ell$ has nonzeros only below position $k$. show that $E^{-1} = I + \ell e_k^\top$. then show that $(I + \ell^{(1)} e_1^\top)(I + \ell^{(2)} e_2^\top) = I + \ell^{(1)}e_1^\top + \ell^{(2)}e_2^\top$, which is why $L$ just collects the multipliers.

<details><summary>solution</summary>

$(I - \ell e_k^\top)(I + \ell e_k^\top) = I - \ell e_k^\top\ell e_k^\top = I - \ell(e_k^\top \ell)e_k^\top$. now $e_k^\top\ell = \ell_k = 0$, because $\ell$ is zero at and above position $k$. so the product is $I$.

for the product: $(I + \ell^{(1)}e_1^\top)(I + \ell^{(2)}e_2^\top) = I + \ell^{(1)}e_1^\top + \ell^{(2)}e_2^\top + \ell^{(1)}(e_1^\top\ell^{(2)})e_2^\top$, and $e_1^\top\ell^{(2)} = \ell^{(2)}_1 = 0$ because $\ell^{(2)}$ is zero in positions $1, 2$. the cross term vanishes. (in the other order, $E_2E_1$, the cross term $\ell^{(2)}(e_2^\top\ell^{(1)})e_1^\top$ does *not* vanish, which is why $E_{n-1}\cdots E_1$ is messy but its inverse $L$ is clean.)

</details>

2. **operation counts.** show that LU without pivoting costs $\sum_{k=1}^{n-1}\big[(n-k) + 2(n-k)^2\big] = \tfrac{2}{3}n^3 + O(n^2)$ flops, and that a triangular solve costs $n^2 + O(n)$.

<details><summary>solution</summary>

at step $k$ there are $n-k$ rows below the pivot. computing their multipliers is $n-k$ divisions. updating the $(n-k) \times (n-k)$ trailing block is one multiply and one subtract per entry, $2(n-k)^2$. with $\sum_{m=1}^{n-1} m^2 = \tfrac{(n-1)n(2n-1)}{6} = \tfrac{n^3}{3} + O(n^2)$, the total is $\tfrac{2}{3}n^3 + O(n^2)$. for the triangular solve, row $i$ needs $i-1$ multiplies, $i-1$ subtractions, and one division: $\sum_i (2i - 1) = n^2$.

</details>

3. **derive Cholesky.** write out $A = LL^\top$ entrywise for $i \ge j$, $a_{ij} = \sum_{k \le j}\ell_{ik}\ell_{jk}$, and solve for $\ell_{jj}$ and $\ell_{ij}$ to get the formulas from idea 5. in what order must the entries be computed?

<details><summary>solution</summary>

for $i = j$: $a_{jj} = \sum_{k<j}\ell_{jk}^2 + \ell_{jj}^2$, so $\ell_{jj} = \sqrt{a_{jj} - \sum_{k<j}\ell_{jk}^2}$. for $i > j$: $a_{ij} = \sum_{k<j}\ell_{ik}\ell_{jk} + \ell_{ij}\ell_{jj}$, so $\ell_{ij} = (a_{ij} - \sum_{k<j}\ell_{ik}\ell_{jk})/\ell_{jj}$. column $j$ uses only columns $1, \dots, j-1$ of $L$, so compute column by column, left to right. within a column, the diagonal comes first.

</details>

4. **why SPD needs no pivoting.** write the first step of elimination on an SPD $A = \begin{pmatrix}\alpha & w^\top\\ w & B\end{pmatrix}$ as $A = \begin{pmatrix}1 & 0\\ w/\alpha & I\end{pmatrix}\begin{pmatrix}\alpha & w^\top\\ 0 & S\end{pmatrix}$ with $S = B - ww^\top/\alpha$. show that $\alpha > 0$ and that $S$ is again SPD. conclude by induction that every pivot is positive.

<details><summary>solution</summary>

$\alpha = e_1^\top A e_1 > 0$. for $S$: take any $v \ne 0$ and set $u = (-w^\top v/\alpha,\ v)$. then $u^\top A u = \alpha(w^\top v)^2/\alpha^2 - 2(w^\top v)^2/\alpha + v^\top B v = v^\top B v - (w^\top v)^2/\alpha = v^\top S v$. since $u \ne 0$, this is $> 0$. so $S$ is SPD (and symmetric, clearly). the trailing matrix after each step is SPD, so its top-left entry, the next pivot, is positive. no zero or negative pivots ever appear, so no swaps are needed.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. elimination without pivoting on $\begin{pmatrix}\delta&1\\1&1\end{pmatrix}$ as $\delta \to 0$, against your pivoted `solve`.
2. the growth factor $\max|U|/\max|A|$ of your `lu` on random matrices and on one carefully built matrix, and what that does to the accuracy of `solve`.
3. using your `cholesky` as a positive-definiteness test: what fraction of random symmetric matrices pass?

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. why does `lu_solve` take the factors rather than the matrix $A$?

<details><summary>answer</summary>

so the $\tfrac{2}{3}n^3$ factorization is paid once and every subsequent solve costs only $2n^2$. an API that took $A$ would refactor every call.

</details>

2. your colleague computes `np.linalg.inv(A) @ b`. give two reasons to replace it with `solve(A, b)`.

<details><summary>answer</summary>

it does about three times the arithmetic ($2n^3$ against $\tfrac{2}{3}n^3$), and it's less accurate: the computed inverse carries its own errors, and multiplying by it doesn't have the backward-stability guarantee that a pivoted LU solve has in practice (chapter 02 · topic 08 makes "backward stable" precise).

</details>

3. partial pivoting makes all $|\ell_{ik}| \le 1$. why isn't that enough to guarantee accuracy?

<details><summary>answer</summary>

the entries of $U$ can still grow. each elimination step can at most double the largest entry, so the growth can reach $2^{n-1}$. the rounding errors committed at each step are relative to the size of the numbers involved, so big entries in $U$ mean big absolute errors. in practice the growth is almost always small. the lab shows a matrix where it isn't.

</details>

4. a matrix has $\det A = 10^{-30}$. is it nearly singular?

<details><summary>answer</summary>

not necessarily. $10^{-1} I_{30}$ has determinant $10^{-30}$ and is perfectly conditioned: it just scales everything by $0.1$. the determinant mixes up scale and singularity. the condition number (topic 08) is the right measure of nearness to singularity.

</details>

5. why does $P$ (the permutation) multiply $b$ in `lu_solve`, but no permutation appears in `cho_solve`?

<details><summary>answer</summary>

Cholesky needs no pivoting for SPD matrices (on paper 4), so there's no permutation to undo. LU's row swaps reorder the equations, so the right-hand side has to be reordered the same way.

</details>

6. if `cholesky` succeeds on a matrix, what have you learned about its eigenvalues?

<details><summary>answer</summary>

they're all positive. success means the matrix is SPD, and SPD is equivalent to all eigenvalues being positive (topic 05). (in floating point, "succeeds" means positive definite up to rounding. a matrix with an eigenvalue of $10^{-18}$ can go either way.)

</details>

7. what's the cost of solving $Ax = b$ when $A$ is already lower-triangular, and why would calling `solve` on it be wasteful?

<details><summary>answer</summary>

$n^2$ with `solve_triangular`. `solve` would run a full $\tfrac{2}{3}n^3$ factorization first, and pivoting would even scramble the triangle.

</details>

8. (the predict callout in idea 4.) how big is the determinant of a $100 \times 100$ matrix of independent standard normals?

<details><summary>answer</summary>

enormous. expanding $\det A = \sum_\sigma \operatorname{sgn}\sigma \prod_i a_{i\sigma(i)}$ and squaring, all the cross terms have mean zero, so $\mathbb{E}[\det A^2] = n!$: the root-mean-square size is $\sqrt{100!} \approx 10^{79}$, and a typical value is a little smaller, around $10^{78}$ (the distribution of $\log|\det A|$ is skewed). geometrically, $|\det A|$ is the volume spanned by the rows, which is the product of each row's distance from the span of the rows before it. the $k$-th row has about $n - k + 1$ directions left to stick out into, so that distance is about $\sqrt{n - k + 1}$, and the product is about $\sqrt{n!}$. nothing about the matrix is unusual. this is why you work with $\log|\det A| = \sum\log|u_{ii}|$, and why "small determinant" (or large) is no test for singularity.

</details>

## common bugs

- **integer input.** `np.array([[2, 1], [4, 3]])` has an integer dtype, so in-place updates like `U[k+1:] -= ...` silently truncate to integers. symptom: wrong answers on small hand examples, right answers on random floats. fix: start with `np.array(A, dtype=float)`.
- **swapping rows of $U$ but not of $L$.** the multipliers already stored in `L[:, :k]` (code indexing) belong to rows, and must move with them. symptom: `P @ A` and `L @ U` differ in a few rows whenever a swap happens after the first column.
- **forgetting to permute $b$.** symptom: correct answers only when $P = I$. solve $Ly = Pb$, not $Ly = b$.
- **the sign of the determinant.** symptom: $|\det|$ right, sign wrong about half the time. a swap flips the sign. a 3-cycle is two swaps, so it doesn't.
- **mutating the caller's arrays.** symptom: a test passes on its own but fails when run after another one that reuses the same matrix. copy before you write.
- **reading the wrong triangle.** using all of row $i$ in forward substitution includes junk from the upper part. symptom: works on genuinely triangular inputs, fails when the other triangle is nonzero (as with LU's packed storage or a full symmetric matrix).

## going deeper

- Trefethen & Bau (1997), *Numerical Linear Algebra*, SIAM, lectures 20–23: Gaussian elimination, pivoting, stability of elimination, and Cholesky. the clearest treatment of why growth factors matter. [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
- Golub & Van Loan (2013), *Matrix Computations*, 4th ed., Johns Hopkins University Press, chapter 3 (general linear systems) and §4.2 (Cholesky). the reference for blocked, cache-efficient versions of these algorithms.
- Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, chapters 9–10: the definitive error analysis of LU and Cholesky. [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
- Wilkinson (1961), "Error analysis of direct methods of matrix inversion", *Journal of the ACM* 8(3): 281–330. where the growth-factor analysis (and the matrix in the lab) comes from. [doi:10.1145/321075.321076](https://doi.org/10.1145/321075.321076)
- Trefethen & Schreiber (1990), "Average-case stability of Gaussian elimination", *SIAM J. Matrix Anal. Appl.* 11(3): 335–360. why exponential growth essentially never happens for random matrices. [doi:10.1137/0611023](https://doi.org/10.1137/0611023)

## connections

- ← builds on: row reduction and rank (topic 01), matrix products as compositions (topic 02), floating point (ch 00 · topic 03).
- → used in: `lstsq` via triangular solves (topic 04), inverse iteration (topic 05), condition estimation (topic 08), multivariate Gaussians via `cholesky` (ch 04), Newton's method (ch 06), linear and ridge regression (ch 08), Gaussian processes (ch 09).
