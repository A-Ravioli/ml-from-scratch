# 02 · linear algebra

> think geometrically about matrices and build the numerical linear algebra ML rests on.

a dataset is a matrix. so is a layer of a neural network, a covariance, a kernel, a Hessian, a Markov chain, an image. almost every algorithm in this course ends up multiplying, factoring or decomposing one, and its failures (a regression that won't fit, a covariance that "isn't positive definite", a loss that diverges) are usually linear algebra failures. this chapter gives you two things at once: the **geometry** (a matrix is a map, the SVD is rotate–stretch–rotate, least squares is a projection) and the **numerics** (why the textbook formula loses half your digits, and which algorithm doesn't). you'll build LU, Cholesky, Householder QR, eigensolvers and an SVD yourself, and they become the library the rest of the course runs on.

**you will build:** reduced row echelon form, rank and null spaces; matrix multiplication four ways and blocked; LU with partial pivoting, triangular solves, determinants and Cholesky; Gram–Schmidt (classical and modified), Householder QR and least squares; power and inverse iteration, the QR algorithm and a Jacobi `eigh`; a one-sided Jacobi SVD, low-rank approximation and the pseudoinverse; a finite-difference gradient checker and ten hand-derived matrix gradients; matrix norms, condition numbers (exact and estimated) and backward errors. modules: **`mlfs.linalg`** (topics 03–06, 08) and **`mlfs.numgrad`** (topic 07).
**prerequisites:** chapter 00 (NumPy, vectorization, floating point) and chapter 01 (vectors and matrices, derivatives, finite differences). no prior linear algebra course is assumed, but it helps.
**time · scaffolding:** about 110 hours · L2 (supported): every function comes with its signature and a docstring with shapes. hints have three levels. the capstone gives you data loading and evaluation.
**newly unlocked library components:** after this chapter you may use `np.linalg` (`solve`, `cholesky`, `qr`, `eigh`, `svd`, `pinv`, `lstsq`, `norm`, `cond`, `det`) and `scipy.linalg` freely in labs and experiments, since you've built each of them. inside `mlfs` modules, later chapters use *your* versions through `mlfs.linalg` and `mlfs.numgrad`, per the contracts in [`CONTRACTS.md`](../CONTRACTS.md). (`@` / `np.matmul` are fair game from topic 02's lab on.)

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [vector spaces](01-vector-spaces/) | `rref`, `rank`, `in_span`, `is_basis`, `coordinates`, `null_space` | random vectors in high dimensions are nearly orthogonal, and the "rank" of a measured matrix depends on your tolerance |
| 02 | [linear maps & matmul](02-linear-maps-and-matmul/) | `matmul_entries`/`_columns`/`_rows`/`_outer`, `matrix_of`, `is_linear`, `matmul_blocked` | `np.matmul` is thousands to tens of thousands of times faster than your triple loop, for the same flops |
| 03 | [solving linear systems](03-solving-linear-systems/) | `solve_triangular`, `lu`, `lu_solve`, `solve`, `det`, `cholesky`, `cho_solve` → `mlfs.linalg` | a perfectly nice matrix breaks elimination without pivoting, and one matrix breaks it even *with* pivoting |
| 04 | [orthogonality & least squares](04-orthogonality-and-least-squares/) | `gram_schmidt` (classical and modified), Householder `qr`, `lstsq` → `mlfs.linalg` | classical Gram–Schmidt's "orthonormal" vectors end up nearly parallel. modified GS and Householder don't |
| 05 | [eigenvalues & eigenvectors](05-eigenvalues-and-eigenvectors/) | `power_iteration`, `inverse_iteration`, `qr_algorithm`, Jacobi `eigh` → `mlfs.linalg` | PageRank: a page with one link beats a page with three. and the QR algorithm freezes on a 2×2 symmetric matrix |
| 06 | [SVD & low-rank approximation](06-svd-and-low-rank-approximation/) | `jacobi_svd_square`, then a general `svd` (one-sided Jacobi), `truncated_svd`, `low_rank_approx`, `pinv` → `mlfs.linalg` | an upright square is rank 1, the same square rotated is rank 22. SVD via $A^\top A$ loses half the spectrum |
| 07 | [matrix calculus](07-matrix-calculus/) | `numerical_gradient`, `check_gradient` → `mlfs.numgrad`, plus 10 hand-derived gradients | smaller $h$ is worse below $10^{-5}$, and a textbook gradient bug passes 3 of 4 gradient checks |
| 08 | [conditioning & stability](08-conditioning-and-stability/) | `matrix_norm`, `cond`, `cond_estimate` → `mlfs.linalg`, `backward_error` | Hilbert systems lose every digit, while the solver's backward error stays at $10^{-17}$ |

## capstone

*eigenfaces*: face recognition with PCA, using only `mlfs.linalg`. you compute the eigenfaces of the Olivetti faces with your own SVD, describe each face by a few dozen coordinates, and recognize new faces by nearest neighbour, then evaluate against baselines with honest error bars. small CPU config: the synthetic stand-in (offline) in a few seconds. full config: the real 400 faces in under a minute.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (an LU with pivoting, a least-squares fit and an SVD, a gradient and a condition number), and a 60-minute blind rebuild of `mlfs.linalg`'s solvers.

## notation

| symbol | meaning |
|---|---|
| $x, b, v$ (lowercase) | column vectors; $x_i$ is the $i$-th entry |
| $A, B, X$ (uppercase) | matrices; $a_{ij}$ an entry, $a_j$ the $j$-th column |
| $e_j$ | the $j$-th standard basis vector |
| $I$, $I_n$ | the identity matrix ($n \times n$) |
| $A^\top$, $A^{-1}$, $A^{-\top}$ | transpose, inverse, inverse transpose |
| $A^+$ | Moore–Penrose pseudoinverse |
| $\operatorname{rank}A$, $\ker A$, $\operatorname{im}A$ | rank, null space (kernel), column space (image) |
| $\operatorname{null}A$, $\operatorname{col}A$, $\operatorname{row}A$ | synonyms used in some lessons: $\operatorname{null}A = \ker A$, $\operatorname{col}A = \operatorname{im}A$; $\operatorname{row}A = \operatorname{im}A^\top$ is the row space |
| indices | math counts rows and columns from 1 ($a_{11}$ is the top-left entry); NumPy code counts from 0 (`A[0, 0]`) |
| $\operatorname{tr}A$, $\det A$ | trace, determinant |
| $PA = LU$ | LU factorization with row permutation $P$, unit lower-triangular $L$, upper-triangular $U$ |
| $A = LL^\top$ | Cholesky factorization ($L$ lower-triangular, positive diagonal) |
| $A = QR$ | QR factorization ($Q^\top Q = I$, $R$ upper-triangular) |
| $A = Q\Lambda Q^\top$ | spectral (eigen)decomposition of a symmetric matrix; $\lambda_i$ eigenvalues |
| $A = U\Sigma V^\top$ | singular value decomposition; $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$; $u_i, v_i$ singular vectors |
| $A_k$ | best rank-$k$ approximation $\sum_{i \le k}\sigma_iu_iv_i^\top$ |
| $\|x\|$, $\|x\|_p$ | vector 2-norm (default), $p$-norm |
| $\|A\|_2$, $\|A\|_1$, $\|A\|_\infty$, $\|A\|_F$ | induced matrix norms, Frobenius norm |
| $\kappa(A)$ | condition number $\|A\|\,\|A^{-1}\|$; in the 2-norm, $\sigma_{\max}/\sigma_{\min}$ |
| $\varepsilon$ | unit roundoff of double precision, $\approx 1.1 \times 10^{-16}$ |
| $\eta(\hat x)$ | normwise backward error of a computed solution |
| $\nabla f$, $df$ | gradient (same shape as the variable), differential |
| $\langle A, B\rangle = \operatorname{tr}(A^\top B)$ | Frobenius inner product of matrices |

## how this connects

- builds on: NumPy, vectorization and floating point (ch 00), vectors, matrices and finite differences (ch 01).
- used later in: multivariate calculus and Jacobians (ch 03), multivariate Gaussians via `cholesky` (ch 04), Newton's method and conditioning in optimization (ch 06), linear and ridge regression via `cholesky`/`lstsq` (ch 08), kernel methods and Gaussian processes (ch 09), PCA via `svd` (ch 11), gradient checks with `mlfs.numgrad` in every chapter that derives a gradient, and backpropagation (ch 14), which is matrix calculus run in reverse.

## core references

- Trefethen & Bau (1997), *Numerical Linear Algebra*, SIAM. the main companion for topics 03–08. [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
- Strang (2016), *Introduction to Linear Algebra*, 5th ed., Wellesley-Cambridge Press. the geometric picture of topics 01–02, 04 and 06.
- Strang (2019), *Linear Algebra and Learning from Data*, Wellesley-Cambridge Press. the SVD, PCA and low rank, with an ML focus.
- Axler (2024), *Linear Algebra Done Right*, 4th ed., Springer. the abstract theory, free online. [doi:10.1007/978-3-031-41026-0](https://doi.org/10.1007/978-3-031-41026-0)
- Golub & Van Loan (2013), *Matrix Computations*, 4th ed., Johns Hopkins University Press. the reference for every algorithm here.
- Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM. conditioning, stability and error analysis in depth. [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
