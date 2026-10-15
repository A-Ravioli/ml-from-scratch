# SVD & low-rank approximation

## why this matters

a 400 × 4096 matrix of face images, a million users × ten thousand movies of ratings, a word × document count table, the weight matrix of a neural network layer: all of these are big, and nearly all of their content lives in a handful of directions. the **singular value decomposition** finds those directions, ranks them by importance, and tells you exactly how much you lose by keeping only the top $k$. that's PCA, image compression, recommender systems, latent semantic analysis, and the LoRA fine-tuning method of chapter 27.

the SVD exists for *every* matrix: rectangular, rank-deficient, non-symmetric, anything. eigendecompositions don't. it also answers the questions the last few topics left hanging. what's the "right" solution of a least-squares problem with dependent columns? (the pseudoinverse.) how close is a matrix to singular? (its smallest singular value.) what is the condition number, really? (topic 08: $\sigma_{\max}/\sigma_{\min}$.) and how should you compute all this without squaring away half your digits?

you'll build the SVD with a beautifully simple algorithm: rotate pairs of columns until they're all perpendicular. then you'll use it for the best low-rank approximation and the pseudoinverse.

## warm-up: predict first

1. a 2×2 matrix maps the unit circle to some curve. what shape is it, always?
2. a 1000 × 1000 image is stored as its best rank-20 approximation. what fraction of the original numbers do you keep?
3. an image of an upright filled square and an image of the same square rotated by 45°: which has lower rank?
4. the singular values of $A$ are the square roots of the eigenvalues of $A^\top A$. is that a good way to compute them?
5. $A = \begin{pmatrix}1&1\\1&1\end{pmatrix}$ has no inverse. what's the most sensible "inverse" it could have?

## prerequisites check

- [ ] you know orthogonal matrices preserve lengths, and you can compute a QR factorization with your `qr` (→ ch 02 · topic 04)
- [ ] you know the spectral theorem for symmetric matrices, and how a Jacobi rotation zeroes an off-diagonal pair (→ ch 02 · topic 05)
- [ ] you can explain column space, null space and rank (→ ch 02 · topics 01–02)
- [ ] you know least squares minimizes $\|Ax - b\|$ (→ ch 02 · topic 04)

## core ideas

### idea 1: every matrix is a rotation, a stretch, and a rotation

take any $m \times n$ matrix $A$ and look at what it does to the unit sphere: it maps it to an ellipsoid, possibly flattened (warm-up 1: an ellipse, or a line segment if $A$ is singular). the ellipsoid's semi-axes are $\sigma_1u_1, \dots, \sigma_ru_r$, with orthonormal $u_i$ and lengths $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. they're the images of an orthonormal set of input directions $v_i$: $Av_i = \sigma_iu_i$. stack these into matrices and you get the **singular value decomposition**

$$A = U\Sigma V^\top = \sum_{i=1}^r \sigma_i\,u_iv_i^\top, \qquad U^\top U = I,\ V^\top V = I,\ \Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_r).$$

read right to left: rotate (by $V^\top$) so the special input directions line up with the axes, stretch axis $i$ by $\sigma_i$, then rotate (by $U$) into the output space. with $r = \min(m, n)$, this is the *reduced* SVD: $U$ is $m \times r$, $V^\top$ is $r \times n$.

**why it exists.** $A^\top A$ is symmetric positive semidefinite, so by the spectral theorem $A^\top A = V\Lambda V^\top$ with $\lambda_i \ge 0$. set $\sigma_i = \sqrt{\lambda_i}$ and $u_i = Av_i/\sigma_i$. then $u_i^\top u_j = v_i^\top A^\top Av_j/(\sigma_i\sigma_j) = \lambda_j\delta_{ij}/(\sigma_i\sigma_j) = \delta_{ij}$. that's a fine proof, and a bad algorithm (warm-up 4). forming $A^\top A$ squares the condition number: singular values below about $\sqrt{\varepsilon}\,\sigma_1 \approx 10^{-8}\sigma_1$ come out as noise. experiment 2 shows it.

**computing it: one-sided Jacobi.** look at the SVD differently: $AV = U\Sigma$. so we want an orthogonal $V$ such that the columns of $W = AV$ are **mutually orthogonal**. their norms are then the $\sigma_i$, and their directions the $u_i$. we can build $V$ from plane rotations, as in topic 05. start from $W = A$, $V = I$, take two columns $w_p, w_q$ and rotate them,

$$\begin{pmatrix}w_p' & w_q'\end{pmatrix} = \begin{pmatrix}w_p & w_q\end{pmatrix}\begin{pmatrix}c & s\\ -s & c\end{pmatrix},$$

choosing the angle so that $w_p'^\top w_q' = 0$. with $\alpha = \|w_p\|^2$, $\beta = \|w_q\|^2$, $\gamma = w_p^\top w_q$, this is *exactly* the Jacobi rotation that diagonalizes the 2×2 Gram block $\begin{pmatrix}\alpha & \gamma\\ \gamma & \beta\end{pmatrix}$:

$$\zeta = \frac{\beta - \alpha}{2\gamma},\qquad t = \frac{\operatorname{sign}(\zeta)}{|\zeta| + \sqrt{1 + \zeta^2}},\qquad c = \frac{1}{\sqrt{1 + t^2}},\qquad s = tc$$

(with $\operatorname{sign}(0) = +1$; `np.hypot(1, zeta)` computes $\sqrt{1+\zeta^2}$ without overflow). apply the same rotation to the columns of $V$, so $W = AV$ stays true. this is Hestenes' method (1958): two-sided Jacobi on $A^\top A$ *without ever forming* $A^\top A$, which is why it keeps full accuracy.

**when to stop.** a **sweep** visits every pair $p < q$ once. a pair is left alone when it's already orthogonal *relative to its own columns*,

$$|\gamma| \le n\,\varepsilon\,\sqrt{\alpha\beta},$$

or when one of the two columns is numerically zero, $\min(\alpha, \beta) \le (\varepsilon\|A\|_F)^2$ (there's no direction left to orthogonalize, and dividing by a $\gamma$ of $10^{-300}$ would overflow). you're done after the first sweep that makes **no rotation at all**. cap the number of sweeps (100 is plenty; typical matrices need 6–10). the relative test matters: an absolute test like $|\gamma| \le 10^{-15}$ would stop early on columns of size $10^{-8}$ and leave their singular values wrong.

**read off the answer.** $s_i = \|w_i\|$. sort by decreasing norm, permuting $s$, the columns of $W$ and the columns of $V$ together. for a full-rank matrix, $u_i = w_i/s_i$. because the stopping test is relative, these are orthonormal to working precision even when $s_i$ is tiny.

> **▶ build: stage 1.** implement `jacobi_svd_square(A, *, max_sweeps=100)` for a square, full-rank `A`: sweeps with the stopping rule above, then sort, then `U = W / s`. return `(U, s, Vt)`.

### idea 2: from a square matrix to any matrix

stage 1's function handles square, full-rank matrices. three more steps turn it into an `svd` for any $m \times n$ matrix:

1. **wide matrices: transpose.** if $m < n$, compute the SVD of the tall $A^\top = U'\Sigma V'^\top$. then $A = V'\Sigma U'^\top$, so swap the roles of $U$ and $V$.
2. **tall matrices: QR first.** for $m > n$, take the reduced $A = QR$ with your Householder `qr`, and run Jacobi on the small $n \times n$ matrix $R$. if $R = U_R\Sigma V^\top$, then $A = (QU_R)\Sigma V^\top$, and $QU_R$ still has orthonormal columns. each rotation now costs $O(n)$ instead of $O(m)$.
3. **rank deficiency: a QR of $W$.** when $s_i$ is *exactly* zero, or a column was skipped as numerically zero, $w_i/s_i$ gives no direction at all, yet the reduced $U$ must still have $\min(m,n)$ orthonormal columns. (for tiny but nonzero $s_i$, $w_i/s_i$ is fine, thanks to the relative stopping test.) the fix is to take the sorted $W = RV$ and factor it, $W = Q_WR_W$. its columns are already orthogonal, so $R_W$ is diagonal up to rounding, $s_i = |(R_W)_{ii}|$, and the columns of $Q_W$ (signs flipped to make that diagonal positive) are orthonormal left singular vectors, *with a completion for the zero columns*. for nonzero $s_i$ they agree with $w_i/s_i$.

**why the sort must come before that QR.** QR handles columns left to right and puts on the diagonal only the part of each column that's orthogonal to the *earlier* ones. a zero column contributes no direction, so if it comes first, the next column's component along the first basis vector lands *above* the diagonal and is lost. for $A = \begin{pmatrix}0&3\\0&4\end{pmatrix}$, unsorted QR gives $s = (4, 0)$ instead of $(5, 0)$. with the nonzero, mutually orthogonal columns first, $R_W$ really is diagonal.

> **▶ build: stage 2.** implement `svd(A)` (the default `full_matrices=False`) for any shape and rank: transpose if wide, then `qr`, then `jacobi_svd_square` on $R$ to get $V$, then a QR of the sorted $W = RV$ for $U$ and $s$. singular values down to $10^{-12}$ must come out with full absolute accuracy.

### idea 3: the full SVD and the four subspaces

the reduced SVD uses $r = \min(m, n)$ singular vectors on each side. the **full** SVD completes them to square orthogonal matrices: $U$ is $m \times m$, $V$ is $n \times n$, and $\Sigma$ is $m \times n$ with zeros padding it out. the extra vectors meet $A$ only through zeros, but they carry real information. if $A$ has rank $\rho$ ($\sigma_1, \dots, \sigma_\rho > 0$, the rest zero),

| subspace | basis | dimension |
|---|---|---|
| column space of $A$ | $u_1, \dots, u_\rho$ | $\rho$ |
| null space of $A^\top$ (left null space) | $u_{\rho+1}, \dots, u_m$ | $m - \rho$ |
| row space of $A$ | $v_1, \dots, v_\rho$ | $\rho$ |
| null space of $A$ | $v_{\rho+1}, \dots, v_n$ | $n - \rho$ |

all four of Strang's fundamental subspaces, with orthonormal bases, from one factorization. rank–nullity (topic 02) is just counting columns. and the **numerical rank** is the number of singular values above a threshold, which is the only notion of rank that survives floating point (recall topic 01's lab).

computing it costs almost nothing extra: ask your `qr` for the *complete* $Q$ and multiply only its first $n$ columns by $U_W$. the last $m - n$ columns of $Q$ are orthogonal to the first $n$. when $A$ has full column rank, the first $n$ span the column space, so the last $m - n$ are an orthonormal basis of the left null space. (if $A$ is rank-deficient, the left null space is bigger, and the QR of $W$ supplies the rest of its basis among the first $n$ columns.)

> **▶ build: stage 3.** support `full_matrices=True`: `U` is $(m, m)$, `Vt` is $(n, n)$, `s` still has length $\min(m, n)$, and the extra columns span the null spaces.

### idea 4: the best low-rank approximation, and the pseudoinverse

the SVD writes $A$ as a sum of rank-one pieces in decreasing order of importance:

$$A = \sigma_1u_1v_1^\top + \sigma_2u_2v_2^\top + \dots + \sigma_ru_rv_r^\top.$$

keep the first $k$ terms: $A_k = U_k\Sigma_kV_k^\top$. storing $A_k$ takes $k(m + n + 1)$ numbers instead of $mn$, so for warm-up 2, $20 \cdot 2001/10^6 \approx 4\%$. the remarkable fact is that you can't do better.

**Eckart–Young theorem.** for every matrix $B$ of rank at most $k$,

$$\|A - B\|_2 \ge \|A - A_k\|_2 = \sigma_{k+1}, \qquad \|A - B\|_F \ge \|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \dots + \sigma_r^2}.$$

(Eckart & Young proved the Frobenius case in 1936. Mirsky extended it to the 2-norm and every other unitarily invariant norm in 1960.) the singular values are a complete report on compressibility: $\sigma_{k+1}$ is the price of rank $k$. a proof sketch for the 2-norm: any rank-$k$ matrix $B$ has a null space of dimension at least $n - k$, which must intersect the $(k+1)$-dimensional span of $v_1, \dots, v_{k+1}$. take a unit vector $x$ in the intersection. then $\|(A - B)x\| = \|Ax\| \ge \sigma_{k+1}$.

the fraction of "energy" kept is $\sum_{i \le k}\sigma_i^2 / \sum_i\sigma_i^2$, because $\|A\|_F^2 = \sum_i\sigma_i^2$. how fast the singular values decay *is* how compressible the data is. low rank is about alignment, not visual simplicity: an upright square is an outer product (rank one), while the same square rotated has a staircase edge that needs many terms (warm-up 3; experiment 1).

**PCA is a truncated SVD.** center a data matrix $X$ (rows are samples) by subtracting the column means, $X_c = X - \bar x$. then $X_c^\top X_c/(n-1)$ is the sample covariance, and its eigenvectors, the principal directions, are the right singular vectors of $X_c$, with variances $\sigma_i^2/(n-1)$. projecting onto the top $k$ of them is the rank-$k$ approximation that keeps the most variance. (forget to center, and the top direction points at the mean instead: experiment 3.) the capstone builds face recognition on exactly this.

> **🔮 predict:** for an $n \times n$ matrix of independent random noise, how quickly do the singular values decay? would you expect a rank-10 approximation to capture most of its energy?

**the pseudoinverse.** if $A = U\Sigma V^\top$ were square and invertible, $A^{-1} = V\Sigma^{-1}U^\top$. the **Moore–Penrose pseudoinverse** extends this to every matrix: invert the singular values you can, and zero the ones you can't.

$$A^+ = V\Sigma^+U^\top,\qquad \Sigma^+ = \operatorname{diag}(\sigma_i^+),\quad \sigma_i^+ = \begin{cases}1/\sigma_i & \sigma_i > \text{cutoff}\\ 0 & \text{otherwise.}\end{cases}$$

$x = A^+b$ is the **minimum-norm least-squares solution**: among all $x$ minimizing $\|Ax - b\|$, it's the shortest. for full-column-rank tall $A$ it equals `lstsq`. for square invertible $A$ it equals $A^{-1}b$. for rank-deficient $A$ it makes a choice, and the choice is principled: it ignores the null space of $A$ entirely, putting no component of $x$ along directions the data can't see. for warm-up 5, $\begin{pmatrix}1&1\\1&1\end{pmatrix} = 2\,uu^\top$ with $u = (1,1)/\sqrt2$, so $A^+ = \tfrac12 uu^\top = \tfrac14\begin{pmatrix}1&1\\1&1\end{pmatrix}$.

the **cutoff** matters. in floating point, a mathematically zero singular value comes out as $10^{-16}$, and inverting it gives $10^{16}$, garbage that swamps everything. so treat $\sigma_i \le \text{rcond}\cdot\sigma_1$ as zero. the default `rcond = 1e-15` only catches values at the rounding level. a larger `rcond` acts as **regularization**: it throws away directions that are technically present but too weak to trust. ridge regression (chapter 08) does a smooth version of the same thing, replacing $1/\sigma_i$ with $\sigma_i/(\sigma_i^2 + \lambda)$.

$A^+$ is characterized by four **Penrose conditions**: $AA^+A = A$, $A^+AA^+ = A^+$, and $AA^+$ and $A^+A$ symmetric. $AA^+$ is the orthogonal projector onto the column space, and $A^+A$ the projector onto the row space.

> **🔮 predict:** you add a tiny amount of noise ($10^{-10}$) to the matrix $\begin{pmatrix}1&1\\1&1\end{pmatrix}$ and compute `pinv` with the default `rcond`. what happens to the entries of the result?

> **▶ build: stage 4.** implement `truncated_svd(A, k)` (raise `ValueError` unless $1 \le k \le \min(m, n)$), `low_rank_approx(A, k)` and `pinv(A, *, rcond=1e-15)` (an $(n, m)$ matrix, with the cutoff *relative* to $\sigma_1$).

**a note on speed.** your `jacobi_svd_square` is a teaching implementation: clear and accurate, but its rotations run as Python loops over $n^2/2$ pairs per sweep. on a laptop it takes about a second at $n = 140$ and about 5 s at $n = 280$ (the full eigenfaces capstone), growing like $n^3$ from there. LAPACK's SVD (behind `np.linalg.svd`) first reduces to bidiagonal form and runs in compiled code, hundreds of times faster. from here on, the course uses `np.linalg` for heavy lifting, and your `mlfs.linalg` where the point is to see your own code at work.

## worked example

$A = \begin{pmatrix}3&0\\4&5\end{pmatrix}$. the columns are $a_1 = (3,4)$ and $a_2 = (0,5)$, so $\alpha = 25$, $\beta = 25$, $\gamma = 20$.

**one rotation.** $\zeta = (25 - 25)/40 = 0$, so $t = 1$ and $c = s = 1/\sqrt2$:

$$w_1 = \tfrac{1}{\sqrt2}(a_1 - a_2) = \tfrac{1}{\sqrt2}(3, -1),\qquad w_2 = \tfrac{1}{\sqrt2}(a_1 + a_2) = \tfrac{1}{\sqrt2}(3, 9).$$

check: $w_1^\top w_2 = (9 - 9)/2 = 0$. ✓ the 2×2 case finishes in one rotation.

**read off the SVD.** $\|w_2\| = \sqrt{90/2} = 3\sqrt5$ and $\|w_1\| = \sqrt{10/2} = \sqrt5$. sorted: $\sigma_1 = 3\sqrt5 \approx 6.708$, $\sigma_2 = \sqrt5 \approx 2.236$. the right singular vectors are the matching columns of $V = \tfrac{1}{\sqrt2}\begin{pmatrix}1&1\\-1&1\end{pmatrix}$: $v_1 = (1,1)/\sqrt2$ and $v_2 = (1,-1)/\sqrt2$. the left ones are $u_1 = w_2/\sigma_1 = (1,3)/\sqrt{10}$ and $u_2 = w_1/\sigma_2 = (3,-1)/\sqrt{10}$. sanity checks: $\sigma_1\sigma_2 = 15 = |\det A|$ ✓ and $\sigma_1^2 + \sigma_2^2 = 50 = \|A\|_F^2$ ✓.

**best rank-1 approximation.** $A_1 = \sigma_1u_1v_1^\top = 3\sqrt5\cdot\frac{1}{\sqrt{20}}\begin{pmatrix}1\\3\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix} = \begin{pmatrix}1.5&1.5\\4.5&4.5\end{pmatrix}$. the error $A - A_1 = \begin{pmatrix}1.5&-1.5\\-0.5&0.5\end{pmatrix}$ has Frobenius norm $\sqrt{2.25 + 2.25 + 0.25 + 0.25} = \sqrt5 = \sigma_2$, as Eckart–Young promises.

**pseudoinverse.** $A$ is invertible, so $A^+ = A^{-1} = \frac{1}{15}\begin{pmatrix}5&0\\-4&3\end{pmatrix}$. and for $B = \begin{pmatrix}1&1\\1&1\end{pmatrix}$ (singular values $2, 0$), $B^+ = \tfrac14 B$.

the tests check $\sigma = (3\sqrt5, \sqrt5)$, $v_1 \parallel (1,1)$, $A_1$, and both pseudoinverses.

## on paper

1. **Frobenius norm and singular values.** show that $\|A\|_F^2 = \operatorname{tr}(A^\top A) = \sum_i\sigma_i^2$ and $\|A\|_2 = \sigma_1$ (where $\|A\|_2 = \max_{\|x\|=1}\|Ax\|$).

<details><summary>solution</summary>

$\|A\|_F^2 = \sum_{ij}a_{ij}^2 = \operatorname{tr}(A^\top A) = \operatorname{tr}(V\Sigma^2V^\top) = \operatorname{tr}(\Sigma^2V^\top V) = \sum\sigma_i^2$. for the 2-norm, write $x = Vy$ with $\|y\| = \|x\| = 1$. then $\|Ax\| = \|U\Sigma y\| = \|\Sigma y\| = \sqrt{\sum\sigma_i^2y_i^2} \le \sigma_1$, with equality at $y = e_1$, i.e. $x = v_1$.

</details>

2. **minimum-norm least squares.** show that $x^+ = A^+b$ minimizes $\|Ax - b\|$, and that every other minimizer has the form $x^+ + z$ with $z \in \operatorname{null}(A)$ and $\|x^+ + z\| \ge \|x^+\|$.

<details><summary>solution</summary>

in the full SVD coordinates $y = V^\top x$, $c = U^\top b$: $\|Ax - b\|^2 = \sum_{i \le \rho}(\sigma_iy_i - c_i)^2 + \sum_{i > \rho}c_i^2$. this is minimized by $y_i = c_i/\sigma_i$ for $i \le \rho$, with $y_i$ for $i > \rho$ arbitrary (those are the null-space directions $z$). $A^+b$ sets them to zero. since the $y$-coordinates are orthogonal, $\|x\|^2 = \sum_{i \le \rho}y_i^2 + \sum_{i > \rho}y_i^2$, which is smallest when the second sum is zero.

</details>

3. **one-sided Jacobi's rotation.** with $w_p' = cw_p - sw_q$ and $w_q' = sw_p + cw_q$, show that $w_p'^\top w_q' = cs(\alpha - \beta) + (c^2 - s^2)\gamma$. setting this to zero, derive $t^2 + 2\zeta t - 1 = 0$ with $\zeta = (\beta - \alpha)/(2\gamma)$.

<details><summary>solution</summary>

expanding, $w_p'^\top w_q' = cs\,\alpha + c^2\gamma - s^2\gamma - sc\,\beta = cs(\alpha - \beta) + (c^2 - s^2)\gamma$. divide by $c^2\gamma$ and write $t = s/c$: $t(\alpha - \beta)/\gamma + 1 - t^2 = 0$, i.e. $t^2 + t(\beta - \alpha)/\gamma - 1 = 0$, which is $t^2 + 2\zeta t - 1 = 0$. the roots multiply to $-1$, so one of them has $|t| \le 1$ (angle at most $45°$). that's the one in idea 1.

</details>

4. **PCA through the SVD.** with $X_c = U\Sigma V^\top$ (centered data, $n$ rows), show that the sample covariance $C = X_c^\top X_c/(n-1)$ has eigenvectors $v_i$ with eigenvalues $\sigma_i^2/(n-1)$, and that the projected data $X_cV_k$ equals $U_k\Sigma_k$.

<details><summary>solution</summary>

$C = V\Sigma U^\top U\Sigma V^\top/(n-1) = V\frac{\Sigma^2}{n-1}V^\top$, an eigendecomposition. $X_cV_k = U\Sigma V^\top V_k = U\Sigma\begin{pmatrix}I_k\\0\end{pmatrix} = U_k\Sigma_k$. so the principal component scores come straight out of the SVD, with no extra multiplication.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. compressing a face image with `low_rank_approx`, and comparing the rank needed by a square and by the same square rotated 45°.
2. computing singular values from `eigh(AᵀA)` (topic 05) against your `svd`, on a matrix with singular values from $1$ down to $10^{-15}$.
3. PCA by SVD on a 2-D point cloud, with and without centering.

## check your understanding

1. why is it legitimate to run Jacobi on $R$ instead of $A$ (after $A = QR$)?

<details><summary>answer</summary>

if $R = U_R\Sigma V^\top$, then $A = QR = (QU_R)\Sigma V^\top$, and $QU_R$ has orthonormal columns because it's a product of a matrix with orthonormal columns and an orthogonal matrix. the singular values and right singular vectors of $A$ and $R$ coincide. working on the $n \times n$ matrix $R$ instead of the $m \times n$ matrix $A$ makes each rotation cost $O(n)$ instead of $O(m)$.

</details>

2. a matrix has singular values $(10, 5, 10^{-9}, 10^{-17})$. what's its rank? its numerical rank?

<details><summary>answer</summary>

mathematically it has rank 4, if those numbers are exact. numerically, $10^{-17}$ is below rounding level ($\varepsilon\sigma_1 \approx 2 \times 10^{-15}$), so the numerical rank is 3. whether to count $10^{-9}$ depends on how accurate your data is. that's exactly the judgment `rcond` encodes.

</details>

3. what does Eckart–Young say about compressing a matrix whose singular values are all equal (like an orthogonal matrix)?

<details><summary>answer</summary>

that you can't. every rank-$k$ approximation loses $\sigma_{k+1} = \sigma_1$ in the 2-norm, the full size of the matrix. an orthogonal matrix has no preferred directions to keep.

</details>

4. why does `pinv` need a cutoff, when `lstsq` (topic 04) doesn't have one?

<details><summary>answer</summary>

`lstsq` assumes full column rank, and just fails (or returns garbage) on rank-deficient input. `pinv` is designed for rank deficiency, and in floating point, "zero" singular values come out around $10^{-16}$. without a cutoff it would invert them into $10^{16}$-sized components along directions the data doesn't determine.

</details>

5. how do the singular values of $A$ relate to the eigenvalues of $A$ when $A$ is symmetric? when it's not?

<details><summary>answer</summary>

for symmetric $A = Q\Lambda Q^\top$, the singular values are $|\lambda_i|$: absorb the signs into $U$. for non-symmetric matrices there's no simple relation. $\begin{pmatrix}0&1\\0&0\end{pmatrix}$ has eigenvalues $0, 0$ but singular values $1, 0$. only $\prod\sigma_i = |\det A| = \prod|\lambda_i|$ holds in general (for square $A$).

</details>

6. in idea 1, $u_i = w_i/\sigma_i$. why does `svd` still use a QR of $W$?

<details><summary>answer</summary>

for nonzero $\sigma_i$, even tiny ones, $w_i/\sigma_i$ is fine: the relative stopping rule keeps those columns orthogonal to working precision. but when $\sigma_i$ is exactly zero (or the column was skipped as numerically zero), $w_i/\sigma_i$ has no direction at all, and the reduced $U$ still needs an orthonormal column there. QR of the sorted $W$ produces an orthonormal basis no matter what, and for nonzero $\sigma_i$ it agrees with $w_i/\sigma_i$ up to sign.

</details>

7. (the predict callout in idea 3.) how compressible is pure noise?

<details><summary>answer</summary>

barely. for a $100 \times 100$ matrix of independent standard normals, the singular values spread fairly evenly from about $2\sqrt{n} = 20$ down to $0$ (the quarter-circle law), with no gap. the top 10 hold only about 30% of the energy: the 10th singular value is still about 16, against 20 for the 1st. low rank is a property of *structured* data. when a dataset's spectrum falls off fast, that's evidence of structure, which is the premise of PCA.

</details>

8. (the predict callout in idea 4.) what does `pinv` do to $\begin{pmatrix}1&1\\1&1\end{pmatrix}$ plus noise of size $10^{-10}$?

<details><summary>answer</summary>

the noise turns the zero singular value into one of about $10^{-10}$, well above the default cutoff $10^{-15}\cdot 2$, so `pinv` inverts it. the result has entries around $10^{9}$–$10^{10}$ instead of $0.25$. the pseudoinverse is a *discontinuous* function of the matrix whenever the rank changes. the fix is a cutoff that matches the noise level (`rcond=1e-8` here), or regularization.

</details>

## common bugs

- **computing $\sigma$ from $A^\top A$.** symptom: small singular values off by orders of magnitude, sometimes `NaN` from the square root of a slightly negative eigenvalue. orthogonalize $A$'s columns directly.
- **forgetting to rotate $V$ alongside $W$.** symptom: correct singular values, but $U\Sigma V^\top \ne A$.
- **sorting $s$ but not $U$ and $V$.** symptom: reconstruction fails after sorting. apply the same permutation to $s$, the columns of $U$ and the columns of $V$ (rows of $V^\top$).
- **dividing by zero singular values.** symptom: `NaN`s in $U$ (in `svd`) or enormous entries (in `pinv`). use QR of $W$ for $U$, and the cutoff for $A^+$.
- **returning $V$ instead of $V^\top$.** symptom: reconstruction works only for symmetric $A$. the contract returns `Vt`, whose *rows* are the right singular vectors.
- **wide matrices.** symptom: shape errors, or only $m$ right singular vectors in full mode. transpose, compute, and transpose back (swapping $U$ and $V$).

## going deeper

- Trefethen & Bau (1997), *Numerical Linear Algebra*, SIAM, lectures 4–5 (the SVD and what it reveals) and 31 (computing the SVD). [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
- Eckart & Young (1936), "The approximation of one matrix by another of lower rank", *Psychometrika* 1(3): 211–218. [doi:10.1007/BF02288367](https://doi.org/10.1007/BF02288367)
- Hestenes (1958), "Inversion of matrices by biorthogonalization and related results", *Journal of the SIAM* 6(1): 51–90. the one-sided Jacobi idea. [doi:10.1137/0106005](https://doi.org/10.1137/0106005)
- Mirsky (1960), "Symmetric gauge functions and unitarily invariant norms", *Quarterly Journal of Mathematics* 11(1): 50–59. Eckart–Young for the 2-norm and beyond. [doi:10.1093/qmath/11.1.50](https://doi.org/10.1093/qmath/11.1.50)
- Brent & Luk (1985), "The solution of singular-value and symmetric eigenvalue problems on multiprocessor arrays", *SIAM J. Sci. Stat. Comput.* 6(1): 69–84. the round-robin ordering that rotates $n/2$ disjoint pairs at once, which is how you would vectorize or parallelize your Jacobi sweeps. [doi:10.1137/0906007](https://doi.org/10.1137/0906007)
- Golub & Kahan (1965), "Calculating the singular values and pseudo-inverse of a matrix", *SIAM J. Numer. Anal.* 2(2): 205–224. the bidiagonalization approach that LAPACK uses. [doi:10.1137/0702016](https://doi.org/10.1137/0702016)
- Halko, Martinsson & Tropp (2011), "Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions", *SIAM Review* 53(2): 217–288. truncated SVDs of huge matrices. [arXiv:0909.4061](https://arxiv.org/abs/0909.4061)
- Strang (2019), *Linear Algebra and Learning from Data*, Wellesley-Cambridge Press, part I (the SVD, PCA and low rank).

## connections

- ← builds on: Householder `qr` (topic 04), Jacobi rotations and the spectral theorem (topic 05), the four subspaces (topics 01–02).
- → used in: condition numbers $\sigma_{\max}/\sigma_{\min}$ (topic 08), the eigenfaces capstone, `PCA` (ch 11), ridge regression's shrinkage (ch 08), whitening and spectral normalization (ch 15), low-rank adapters (LoRA, ch 27).
