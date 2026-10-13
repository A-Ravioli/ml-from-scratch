# 02 · linear algebra · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. what are the column space and null space of a $5 \times 8$ matrix of rank 3? give their dimensions and the spaces they live in.

<details><summary>answer</summary>

the column space is a 3-dimensional subspace of $\mathbb{R}^5$. the null space is a $(8 - 3) = 5$-dimensional subspace of $\mathbb{R}^8$ (rank–nullity).

</details>

2. why is column $j$ of a matrix the image of the basis vector $e_j$, and why does that force the definition of matrix multiplication?

<details><summary>answer</summary>

$Ae_j$ is the combination of $A$'s columns with weights $e_j$, which is column $j$. for $AB$ to represent "apply $B$, then $A$", its column $j$ must be $A(Be_j) = Ab_j$. that's the column view, and the entry formula follows.

</details>

3. why does Gaussian elimination need pivoting even for perfectly well-conditioned matrices?

<details><summary>answer</summary>

a tiny pivot creates huge multipliers, and the subsequent subtraction rounds away the original data (the $\begin{pmatrix}\delta&1\\1&1\end{pmatrix}$ example). the problem is the algorithm's choice of pivot, not the matrix. partial pivoting keeps all multipliers at most 1 in magnitude.

</details>

4. you need $A^{-1}b$ for 1,000 different vectors $b$. what do you compute, and what does it cost?

<details><summary>answer</summary>

factor $PA = LU$ once ($\tfrac23n^3$), then two triangular solves per $b$ ($2n^2$ each). never form $A^{-1}$.

</details>

5. when does Cholesky succeed, and why is it preferred over LU when it applies?

<details><summary>answer</summary>

exactly when the (symmetric) matrix is positive definite. it costs half as much ($\tfrac13n^3$), needs no pivoting, is backward stable, and doubles as a test for positive definiteness.

</details>

6. classical and modified Gram–Schmidt are algebraically identical. why does MGS give a more orthogonal $Q$, and what gives a fully orthogonal one?

<details><summary>answer</summary>

MGS computes each projection coefficient from the partially orthogonalized vector, so it removes the errors made in earlier projections. CGS uses the original column, so errors compound to $\varepsilon\kappa^2$ (against $\varepsilon\kappa$). Householder QR builds $Q$ as a product of exact reflections, so it's orthogonal to machine precision regardless of $\kappa$.

</details>

7. why shouldn't you solve least-squares problems through the normal equations when $A$ is ill-conditioned?

<details><summary>answer</summary>

$\kappa(A^\top A) = \kappa(A)^2$. forming $A^\top A$ loses about twice as many digits as working with $A$ directly via QR.

</details>

8. power iteration converges at what rate? what does inverse iteration with shift $\mu$ converge to, and at what rate?

<details><summary>answer</summary>

power iteration: the eigenvector error shrinks like $|\lambda_2/\lambda_1|^k$ (the eigenvalue error like the square of that, for symmetric matrices). inverse iteration converges to the eigenvalue nearest $\mu$, at rate $|\lambda_\text{nearest} - \mu|/|\lambda_\text{second nearest} - \mu|$.

</details>

9. state the spectral theorem. what's the analogous statement for a rectangular matrix?

<details><summary>answer</summary>

every real symmetric $A$ factors as $Q\Lambda Q^\top$ with $Q$ orthogonal and $\Lambda$ real diagonal. for any $m \times n$ matrix, the SVD gives $A = U\Sigma V^\top$ with orthonormal $U, V$ and non-negative, decreasing singular values.

</details>

10. what's the best rank-$k$ approximation of $A$ in the 2-norm, and what's its error?

<details><summary>answer</summary>

the truncated SVD $A_k = \sum_{i \le k}\sigma_iu_iv_i^\top$ (Eckart–Young). its error is $\|A - A_k\|_2 = \sigma_{k+1}$, and in the Frobenius norm $\sqrt{\sum_{i>k}\sigma_i^2}$.

</details>

11. what does the pseudoinverse do with a zero singular value, and why does it need a cutoff in practice?

<details><summary>answer</summary>

it leaves it at zero instead of inverting it, which gives the minimum-norm least-squares solution. in floating point, "zero" singular values come out around $10^{-16}$, and inverting them would produce enormous meaningless components. so values below $\text{rcond}\cdot\sigma_1$ are treated as zero.

</details>

12. what's the gradient of $f(x) = x^\top Ax$ for a non-symmetric $A$? of $f(X) = \log|\det X|$?

<details><summary>answer</summary>

$(A + A^\top)x$, and $X^{-\top}$.

</details>

13. why is a gradient check at $x = 0$, or with symmetric test matrices, unreliable?

<details><summary>answer</summary>

many wrong gradients coincide with the right one there. for example, $2Ax$ equals $(A + A^\top)x$ when $A$ is symmetric or $x = 0$. use random points and random non-symmetric, non-square matrices.

</details>

14. distinguish conditioning from stability, and state the rule that connects forward error, backward error and κ.

<details><summary>answer</summary>

conditioning is a property of the problem: how much the exact solution moves when the data moves. stability is a property of the algorithm: whether it gives the exact solution of a nearby problem. forward error ≲ κ × backward error, so a backward-stable algorithm loses about $\log_{10}\kappa$ digits.

</details>

15. a matrix has $\det A = 10^{-40}$. is it nearly singular? what would you check instead?

<details><summary>answer</summary>

not necessarily. $0.1 I_{40}$ has that determinant and $\kappa = 1$. check the condition number, $\sigma_{\max}/\sigma_{\min}$.

</details>

## 2. by hand (no computer)

1. **LU and a solve.** factor $A = \begin{pmatrix}2 & 1 & 1\\ 4 & 3 & 3\\ 8 & 7 & 9\end{pmatrix}$ as $PA = LU$ with partial pivoting, compute $\det A$, and solve $Ax = (4, 10, 24)^\top$.

<details><summary>solution</summary>

column 1: pivot 8 (row 3), so swap rows 1 and 3, giving rows $(8,7,9)$, $(4,3,3)$, $(2,1,1)$. the multipliers are $4/8 = 0.5$ and $2/8 = 0.25$. after elimination: $(4,3,3) - 0.5(8,7,9) = (0, -0.5, -1.5)$ and $(2,1,1) - 0.25(8,7,9) = (0, -0.75, -1.25)$.

column 2: the candidates are $-0.5$ and $-0.75$, so the pivot is $-0.75$: swap rows 2 and 3 (the multipliers swap too). the multiplier is $-0.5/-0.75 = 2/3$, and $(0, -0.5, -1.5) - \tfrac23(0, -0.75, -1.25) = (0, 0, -2/3)$.

so $P$ takes rows $(3, 1, 2)$ of $A$, i.e. $P = \begin{pmatrix}0&0&1\\1&0&0\\0&1&0\end{pmatrix}$, with $L = \begin{pmatrix}1&0&0\\0.25&1&0\\0.5&2/3&1\end{pmatrix}$ and $U = \begin{pmatrix}8&7&9\\0&-0.75&-1.25\\0&0&-2/3\end{pmatrix}$.

$P$ is a 3-cycle (two swaps), so $\det P = +1$ and $\det A = 8 \cdot (-0.75)\cdot(-2/3) = 4$. (check by cofactors: $2(27 - 21) - 1(36 - 24) + 1(28 - 24) = 12 - 12 + 4 = 4$. ✓)

solve: $Pb = (24, 4, 10)$. forward: $y_1 = 24$, $y_2 = 4 - 0.25 \cdot 24 = -2$, $y_3 = 10 - 0.5 \cdot 24 - \tfrac23(-2) = -2 + \tfrac43 = -\tfrac23$. back: $x_3 = (-\tfrac23)/(-\tfrac23) = 1$, $x_2 = (-2 + 1.25 \cdot 1)/(-0.75) = 1$, $x_1 = (24 - 7 - 9)/8 = 1$. so $x = (1, 1, 1)$. ✓ ($A\mathbf 1 = (4, 10, 24)$.)

</details>

2. **least squares and SVD.** (a) fit $y = c + dt$ to the points $(-1, 0), (0, 1), (1, 5)$ by least squares. (b) find the singular values of $\begin{pmatrix}2 & 0\\ 0 & 0\\ 0 & 1\end{pmatrix}$, its best rank-1 approximation, and its pseudoinverse.

<details><summary>solution</summary>

(a) $A = \begin{pmatrix}1&-1\\1&0\\1&1\end{pmatrix}$ has orthogonal columns, so $A^\top A = \operatorname{diag}(3, 2)$ and $A^\top y = (6, 5)$. so $c = 2$ and $d = 2.5$. the residual $y - Ax = (0, 1, 5) - (-0.5, 2, 4.5) = (0.5, -1, 0.5)$ is orthogonal to both columns. ✓

(b) the columns are already orthogonal, with norms 2 and 1, so $\sigma_1 = 2$, $\sigma_2 = 1$, with $v_1 = e_1$, $u_1 = e_1$, $v_2 = e_2$, $u_2 = e_3$. the best rank-1 approximation is $2e_1e_1^\top = \begin{pmatrix}2&0\\0&0\\0&0\end{pmatrix}$, with 2-norm error $\sigma_2 = 1$. the pseudoinverse is $V\Sigma^{-1}U^\top = \begin{pmatrix}1/2 & 0 & 0\\ 0 & 0 & 1\end{pmatrix}$.

</details>

3. **a gradient and a condition number.** (a) derive $\nabla_W\,\tfrac12\|XW - Y\|_F^2$ with the trace trick. (b) for $A = \begin{pmatrix}1 & 0\\ 0 & 10^{-3}\end{pmatrix}$, what's $\kappa_2(A)$? if $b = (1, 0)$ is perturbed by $\delta b = (0, 10^{-6})$, what is the relative change in the solution of $Ax = b$, and how does it compare with the bound?

<details><summary>solution</summary>

(a) $f = \tfrac12\operatorname{tr}(R^\top R)$ with $R = XW - Y$, so $df = \operatorname{tr}(R^\top dR) = \operatorname{tr}(R^\top X\,dW)$, giving $\nabla_W f = X^\top(XW - Y)$.

(b) $\kappa_2 = 1/10^{-3} = 1000$. $x = (1, 0)$ and $\delta x = (0, 10^{-3})$, so $\|\delta x\|/\|x\| = 10^{-3}$, while $\|\delta b\|/\|b\| = 10^{-6}$. the amplification is $1000 = \kappa$: the bound is attained, because $b$ lies along the top singular direction and $\delta b$ along the bottom one.

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **solving linear systems**, the foundation everything else in `mlfs.linalg` stands on. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=02-linear-algebra/03-solving-linear-systems
mkdir -p /tmp/rebuild-03
git show HEAD:$T/exercise.py > /tmp/rebuild-03/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-03/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-03/solutions/` and run `python scripts/stubify.py /tmp/rebuild-03`.) open `/tmp/rebuild-03/exercise.py` with `marimo edit`, implement `solve_triangular`, `lu`, `lu_solve`, `solve`, `det`, `cholesky` and `cho_solve` from memory, and run the checks against your copy:

```bash
pytest /tmp/rebuild-03/test_exercise.py
```

you pass if every check is green within the hour.

**harder variant** (for testing out of the chapter with confidence): rebuild Householder `qr` (topic 04) and the SVD (`jacobi_svd_square` and `svd`, topic 06) the same way, each in its own scratch directory, and pass these within 90 minutes in total:

```bash
pytest /tmp/rebuild-04/test_exercise.py -k stage3
pytest /tmp/rebuild-06/test_exercise.py -k "stage1 or stage2"
```

(run them from the repo root: those notebooks import `mlfs.linalg`, which the tests resolve to the reference solutions, so only the functions you rebuilt are being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each part of a problem separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the idea, and the topic READMEs and lessons are organized by those ideas.
