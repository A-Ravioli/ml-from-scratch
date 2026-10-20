# eigenvalues & eigenvectors

## why this matters

in 1998 a search engine needed to rank a web of a few hundred million pages by importance. the answer Google settled on: a page is important if important pages link to it. that definition is circular, and circular definitions of this kind are **eigenvector equations**: the importance vector $\pi$ satisfies $\pi = G\pi$ for a matrix $G$ built from the links. the first PageRank computations were a few dozen matrix-vector products, which is power iteration, the first algorithm you'll build here.

eigenvectors are the directions in which a matrix acts simply, by stretching. that makes them the natural coordinates for anything you do with a matrix repeatedly: run a Markov chain, iterate a dynamical system, take gradient steps on a quadratic (the eigenvalues of the Hessian decide whether gradient descent converges, chapter 06), or find the directions of greatest variance in data (PCA, the capstone). and for symmetric matrices, which is what covariances, Hessians, kernels and graph Laplacians all are, the eigenvectors are orthogonal. you can always rotate to a basis where the matrix is diagonal.

you can't solve for eigenvalues with a formula: for $n \ge 5$ no formula in radicals exists (Abel–Ruffini). every eigenvalue algorithm is iterative. you'll build four, from the one-vector power method to Jacobi's method, which finds every eigenpair of a symmetric matrix to full precision.

## warm-up: predict first

1. you multiply a random vector by the same matrix 50 times, normalizing each time. what happens to its direction?
2. if $A$ has eigenvalues $3$ and $1$, what are the eigenvalues of $(A - 0.9I)^{-1}$?
3. a symmetric matrix has a negative entry on its diagonal. must it have a negative eigenvalue?
4. you know an approximate eigenvector $v$ with error $10^{-4}$. how accurate is the eigenvalue estimate $v^\top Av$ for symmetric $A$: about $10^{-4}$, or better?
5. can a real matrix have no real eigenvectors at all?

## prerequisites check

- [ ] you can multiply matrices fluently and know that $(AB)^{-1} = B^{-1}A^{-1}$ (→ ch 02 · topic 02)
- [ ] you can factor and solve with your `lu` and `lu_solve` (→ ch 02 · topic 03)
- [ ] you can compute a QR factorization with your `qr`, and you know that orthogonal $Q$ satisfies $Q^\top Q = I$ (→ ch 02 · topic 04)
- [ ] you know that $\det(A - \lambda I) = 0$ defines the eigenvalues (→ ch 01 · topic 04)

## core ideas

### idea 1: power iteration

$v \ne 0$ is an **eigenvector** of $A$ with **eigenvalue** $\lambda$ if $Av = \lambda v$: along $v$, $A$ just scales by $\lambda$.

**diagonalization.** if an $n \times n$ matrix has $n$ independent eigenvectors, stack them as the columns of $V$ and the eigenvalues on the diagonal of $\Lambda$. then $AV = V\Lambda$, and since $V$ is invertible,

$$A = V\Lambda V^{-1}, \qquad A^k = V\Lambda^kV^{-1}.$$

in the eigenvector basis, $A$ is just a diagonal scaling, and powers of $A$ are powers of numbers. when does such a basis exist? always if the $n$ eigenvalues are distinct (eigenvectors for distinct eigenvalues are independent), and always for symmetric matrices (idea 3). but not always: $J = \begin{pmatrix}1&1\\0&1\end{pmatrix}$ has the eigenvalue 1 twice, yet $(J - I)v = 0$ only for multiples of $(1, 0)$: one eigenvector direction, not two. such a matrix is **defective**. it isn't a scaling in any basis: $J^k = \begin{pmatrix}1&k\\0&1\end{pmatrix}$ grows linearly in $k$ even though $|\lambda| = 1$, a shear rather than a stretch.

if $A$ has $n$ independent eigenvectors $v_1, \dots, v_n$, any starting vector is a combination of them, $x_0 = \sum_i c_i v_i$, and multiplying by $A$ scales each component by its own eigenvalue:

$$A^kx_0 = \sum_i c_i\lambda_i^kv_i = \lambda_1^k\Big(c_1v_1 + \sum_{i>1}c_i\big(\tfrac{\lambda_i}{\lambda_1}\big)^kv_i\Big).$$

if $|\lambda_1| > |\lambda_2| \ge \dots$, every ratio $(\lambda_i/\lambda_1)^k$ goes to zero. the direction of $A^kx_0$ swings toward $v_1$, with error shrinking like $|\lambda_2/\lambda_1|^k$ (warm-up 1). **power iteration** is just this, with normalization so the numbers don't overflow:

$$v_{k+1} = \frac{Av_k}{\|Av_k\|}.$$

to read off the eigenvalue, use the **Rayleigh quotient** $r(v) = v^\top Av / v^\top v$, the "best eigenvalue" for a given $v$ (it minimizes $\|Av - rv\|$ over $r$). it keeps the sign of $\lambda$, which $\|Av\|$ doesn't. and for symmetric $A$ it's *quadratically* accurate: an eigenvector error of $\delta$ gives an eigenvalue error of order $\delta^2$ (warm-up 4; on paper 2).

when do you stop? the honest test is the **residual**: $\|Av - \lambda v\|$ is zero exactly for an eigenpair, and it's cheap, since you already computed $Av$. don't stop when "$v$ stops changing": if $\lambda_1 < 0$, $v$ flips sign every step and never stops changing.

> **▶ build: stage 1.** implement `power_iteration(A, *, max_iters=1000, tol=1e-10, rng=None)`. draw the starting vector from `np.random.default_rng(rng)`, return `(eigenvalue, unit eigenvector)`, and stop when $\|Av - \lambda v\| \le \text{tol}\cdot\|Av\|$.

> **🔮 predict:** what does power iteration do on $\begin{pmatrix}5&0\\0&-5\end{pmatrix}$, where the two largest eigenvalues have equal magnitude?

### idea 2: shift and invert

power iteration finds only the largest eigenvalue in magnitude. to find others, change the matrix without changing its eigenvectors. if $Av = \lambda v$, then

$$(A - \mu I)v = (\lambda - \mu)v \qquad\text{and}\qquad (A - \mu I)^{-1}v = \frac{1}{\lambda - \mu}v.$$

same eigenvectors, new eigenvalues $1/(\lambda_i - \mu)$. the eigenvalue of $A$ **closest to the shift** $\mu$ becomes the *largest* eigenvalue of $(A - \mu I)^{-1}$ (warm-up 2: $3, 1 \mapsto 1/2.1 \approx 0.48$ and $1/0.1 = 10$). so power iteration on the inverse finds it, at rate

$$\frac{|\lambda_{\text{nearest}} - \mu|}{|\lambda_{\text{second nearest}} - \mu|},$$

which you control. a good shift gives very fast convergence. this is **inverse iteration**.

you never form the inverse (topic 03): "multiply by $(A - \mu I)^{-1}$" means "solve with $A - \mu I$". the matrix is the same every step, so factor it once with your `lu` and call `lu_solve` each iteration: $O(n^2)$ per step after one $O(n^3)$ factorization.

what if $\mu$ *is* an eigenvalue? then $A - \mu I$ is singular and the solve fails. but a shift that's merely *very close* is the best case: rounding errors in the solve are enormous, yet they point almost exactly along the wanted eigenvector, which is all you need after normalizing. so nudge $\mu$ by a relative $10^{-10}$ and carry on.

update the shift every step to the current Rayleigh quotient and you get **Rayleigh quotient iteration**, which converges *cubically* for symmetric matrices. the price is a new factorization every step.

> **▶ build: stage 2.** implement `inverse_iteration(A, mu=0.0, *, max_iters=1000, tol=1e-10, rng=None)` using your `lu` and `lu_solve`. estimate the eigenvalue with the Rayleigh quotient using $A$ itself, and use the same residual stopping rule.

### idea 3: all eigenvalues at once: the QR algorithm

the **spectral theorem** says every real symmetric matrix can be written

$$A = Q\Lambda Q^\top,\qquad Q^\top Q = I,\ \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)\ \text{real}.$$

symmetric matrices have real eigenvalues and orthonormal eigenvectors. a useful consequence: in the eigenbasis the Rayleigh quotient is a weighted average of the eigenvalues, so $\lambda_{\min} \le r(v) \le \lambda_{\max}$ for every $v$. at $v = e_i$ that says $\lambda_{\min} \le a_{ii} \le \lambda_{\max}$. that settles warm-up 3: yes, one negative diagonal entry forces a negative eigenvalue. (the converse fails. $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ has a positive diagonal and the eigenvalue $-1$.) for warm-up 5: yes, a rotation of the plane by 90° has eigenvalues $\pm i$ and moves every real vector. the spectral theorem only promises real eigenvectors for *symmetric* matrices.

to find all of $\Lambda$, we want orthogonal **similarity transformations** $A \mapsto Q^\top AQ$, which keep the eigenvalues, chosen to drive $A$ toward diagonal (or, for non-symmetric matrices, upper-triangular) form. the **QR algorithm** is startlingly simple:

$$A_0 = A, \qquad Q_k R_k = A_k\ \text{(your qr)}, \qquad A_{k+1} = R_kQ_k.$$

since $R_k = Q_k^\top A_k$, we get $A_{k+1} = Q_k^\top A_kQ_k$: a similarity transform, so the eigenvalues never change. why does it converge? the product $Q_0Q_1\cdots Q_{k-1}$ turns out to be exactly what you'd get by running power iteration on *$n$ vectors at once*, re-orthogonalizing them each step (subspace iteration). the first column converges to the dominant eigenvector, the first two span the dominant 2-D eigenspace, and so on. in the rotated coordinates, $A_k$ becomes upper-triangular, with the eigenvalues appearing on the diagonal in order of decreasing magnitude. the $(i+1, i)$ entry decays like $|\lambda_{i+1}/\lambda_i|^k$.

this is the *unshifted* algorithm, and it inherits power iteration's weakness. eigenvalues of nearly equal magnitude converge slowly, and equal magnitudes don't converge at all (experiment 3 breaks it). the industrial version (Francis, 1961) first reduces $A$ to tridiagonal or Hessenberg form, then uses **shifts** (inverse iteration's trick again) and deflation. with the Wilkinson shift, the symmetric case converges cubically (non-symmetric matrices with good shifts converge quadratically). it's what `np.linalg.eigh` and `np.linalg.eig` run through LAPACK.

> **▶ build: stage 3.** implement `qr_algorithm(A, *, max_iters=1000, tol=1e-10)` with your `qr`. stop when the strictly lower-triangular part of $A_k$ is small relative to $\|A\|_F$, and return the diagonal sorted descending.

### idea 4: Jacobi's method: diagonalize by rotations

for symmetric matrices there's a direct way to use the spectral theorem: build $Q$ as a product of plane rotations, each one killing a single off-diagonal pair. look at any 2×2 symmetric block in rows and columns $p, q$:

$$\begin{pmatrix}c & -s\\ s & c\end{pmatrix}\begin{pmatrix}a_{pp} & a_{pq}\\ a_{pq} & a_{qq}\end{pmatrix}\begin{pmatrix}c & s\\ -s & c\end{pmatrix},\qquad c = \cos\theta,\ s = \sin\theta.$$

the new off-diagonal entry is $(c^2 - s^2)a_{pq} + cs(a_{pp} - a_{qq})$. setting it to zero and writing $t = \tan\theta$, $\tau = (a_{qq} - a_{pp})/(2a_{pq})$ gives $t^2 + 2\tau t - 1 = 0$. take the smaller root, so the rotation angle stays below $45°$ and disturbs the rest of the matrix as little as possible:

$$t = \frac{\operatorname{sign}(\tau)}{|\tau| + \sqrt{1 + \tau^2}},\qquad c = \frac{1}{\sqrt{1 + t^2}},\qquad s = tc.$$

(this form never subtracts nearly equal numbers. the textbook $t = -\tau + \sqrt{\tau^2 + 1}$ does, for large $\tau$.) embed the 2×2 rotation in the identity to get $J$, and update $A \leftarrow J^\top AJ$ and $V \leftarrow VJ$. only rows and columns $p$ and $q$ change.

**why it converges.** orthogonal similarity preserves the Frobenius norm $\sum_{ij}a_{ij}^2$. a rotation moves exactly $2a_{pq}^2$ of it from the off-diagonal onto the diagonal:

$$\operatorname{off}(J^\top AJ)^2 = \operatorname{off}(A)^2 - 2a_{pq}^2, \qquad \operatorname{off}(A)^2 = \sum_{i \ne j}a_{ij}^2.$$

later rotations refill entries you zeroed earlier, but the total off-diagonal mass only ever goes down. a **sweep** rotates every pair $(p, q)$ once. after a few sweeps the convergence becomes quadratic, and for typical matrices 6–10 sweeps reach machine precision. when $\operatorname{off}(A) \approx 0$, the diagonal holds the eigenvalues, and the accumulated $V = J_1J_2\cdots$ holds orthonormal eigenvectors: orthonormal because $V$ is a product of rotations, never formed by subtraction.

Jacobi (1846) is older than computers. it lost to the QR algorithm on speed: each sweep costs about $9n^3$ flops (with eigenvectors), and you need several sweeps, while tridiagonal QR does the whole job in roughly $9n^3$. but it's simple, very robust, and often *more accurate* for small eigenvalues (Demmel & Veselić, 1992). for the matrix sizes in this course (up to a few hundred), it's a fine `eigh`.

**a note on speed.** your `eigh` is a teaching implementation: each rotation is a handful of NumPy calls inside a Python double loop, so it takes about 0.4 s at $n = 100$ and about 2 s at $n = 200$ on a laptop, growing like $n^3$. LAPACK's `np.linalg.eigh` handles $n = 1000$ in well under a second. from here on the course uses `np.linalg` for heavy lifting, and your `mlfs.linalg` where seeing your own code at work is the point.

> **🔮 predict:** a rotation zeroes $a_{pq}$. after the next rotation (on a different pair that shares row $p$), is $a_{pq}$ still zero?

> **▶ build: stage 4.** implement `eigh(A, *, max_iters=10000, tol=1e-12)` by cyclic Jacobi sweeps (`max_iters` caps the number of sweeps). return eigenvalues **ascending** with the matching columns of `V`.

## worked example

take $A = \begin{pmatrix}2&1\\1&2\end{pmatrix}$: eigenvalues $3$ (eigenvector $(1,1)/\sqrt2$) and $1$ (eigenvector $(1,-1)/\sqrt2$).

**power iteration** from $x_0 = (1, 0)$ (unnormalized, to keep the numbers clean): $Ax_0 = (2, 1)$, $A^2x_0 = (5, 4)$, $A^3x_0 = (14, 13)$. the direction approaches $(1,1)$. the Rayleigh quotients are $r(2,1) = \frac{(2,1)\cdot(5,4)}{5} = 2.8$ and $r(5,4) = \frac{(5,4)\cdot(14,13)}{41} = \frac{122}{41} \approx 2.976$. the errors $0.2$ and $0.024$ shrink by about $(1/3)^2 = 1/9$ per step, quadratic in the vector's error ratio $1/3$, just as idea 1 promised.

**inverse iteration** with $\mu = 0.9$: $(A - 0.9I)^{-1}$ has eigenvalues $1/2.1$ and $1/0.1 = 10$. it converges to $\lambda = 1$, with the error shrinking by $0.1/2.1 \approx 0.05$ per step.

**one QR step**, with Gram–Schmidt signs: $q_1 = (2,1)/\sqrt5$, $q_2 = (-1,2)/\sqrt5$, $R = \begin{pmatrix}\sqrt5 & 4/\sqrt5\\ 0 & 3/\sqrt5\end{pmatrix}$, and

$$A_1 = RQ = \begin{pmatrix}2.8 & 0.6\\ 0.6 & 1.2\end{pmatrix}.$$

the trace is still $4$ and the determinant still $3$ (a similarity transform). the diagonal is heading to $(3, 1)$.

**one Jacobi rotation**: $\tau = (2 - 2)/2 = 0$, so $t = 1$ (take $\operatorname{sign}(0) = +1$), $c = s = 1/\sqrt2$. the new diagonal entries are $c^2a_{pp} - 2cs\,a_{pq} + s^2a_{qq} = 1 - 1 + 1 = 1$ and $s^2a_{pp} + 2cs\,a_{pq} + c^2a_{qq} = 3$. so one rotation diagonalizes a 2×2, and $V = J = \frac{1}{\sqrt2}\begin{pmatrix}1&1\\-1&1\end{pmatrix}$: column 1 is the eigenvector $(1,-1)/\sqrt2$ for $\lambda = 1$.

the tests check all four algorithms on this matrix: $3$ and $(1,1)$ from `power_iteration`, $1$ and $(1,-1)$ from `inverse_iteration(A, 0.9)`, $[3, 1]$ from `qr_algorithm`, and $w = [1, 3]$ with $V[:, 0] \parallel (1, -1)$ from `eigh`.

## on paper

1. **eigenvalues of a symmetric matrix are real, and eigenvectors for distinct eigenvalues are orthogonal.** prove both for real symmetric $A$. (hint: for the first, allow complex $v$ and compute $\bar v^\top Av$ two ways.)

<details><summary>solution</summary>

if $Av = \lambda v$ with $v \in \mathbb{C}^n$, $v \ne 0$, then $\bar v^\top Av = \lambda\,\bar v^\top v = \lambda\|v\|^2$. conjugating $Av = \lambda v$ gives $A\bar v = \bar\lambda\bar v$ (since $A$ is real), and using $A = A^\top$: $\bar v^\top Av = (A^\top\bar v)^\top v = (A\bar v)^\top v = \bar\lambda\,\bar v^\top v = \bar\lambda\|v\|^2$. so $\lambda\|v\|^2 = \bar\lambda\|v\|^2$, and $\lambda$ is real. for orthogonality: if $Au = \lambda u$ and $Av = \mu v$ with $\lambda \ne \mu$, then $\lambda u^\top v = (Au)^\top v = u^\top Av = \mu u^\top v$, so $(\lambda - \mu)u^\top v = 0$, forcing $u^\top v = 0$.

</details>

2. **the Rayleigh quotient is quadratically accurate.** for symmetric $A$ with unit eigenvectors $q_1, q_2, \dots$, write a unit vector as $v = \cos\theta\,q_1 + \sin\theta\,w$ with $w \perp q_1$ a unit vector. show that $r(v) - \lambda_1 = \sin^2\theta\,(w^\top Aw - \lambda_1)$, so the eigenvalue error is $O(\theta^2)$.

<details><summary>solution</summary>

$v^\top Av = \cos^2\theta\,\lambda_1 + 2\cos\theta\sin\theta\,q_1^\top Aw + \sin^2\theta\,w^\top Aw$. the cross term vanishes: $q_1^\top Aw = (Aq_1)^\top w = \lambda_1 q_1^\top w = 0$. so $r(v) = \lambda_1 + \sin^2\theta(w^\top Aw - \lambda_1)$. the vector error is $\sin\theta \approx \theta$, the eigenvalue error $\propto \theta^2$. this is why experiment 2's eigenvalue curves fall twice as steeply as the vector curves.

</details>

3. **one QR step is a similarity.** show that $A_{k+1} = R_kQ_k = Q_k^\top A_kQ_k$, and that $A_k = (Q_0\cdots Q_{k-1})^\top A\,(Q_0\cdots Q_{k-1})$. why does that mean symmetric $A_k$ stay symmetric?

<details><summary>solution</summary>

$R_k = Q_k^\top A_k$ (multiply $A_k = Q_kR_k$ by $Q_k^\top$), so $R_kQ_k = Q_k^\top A_kQ_k$. by induction, $A_k = \hat Q_k^\top A\hat Q_k$ with $\hat Q_k = Q_0\cdots Q_{k-1}$, a product of orthogonal matrices, hence orthogonal. and $(\hat Q^\top A\hat Q)^\top = \hat Q^\top A^\top\hat Q = \hat Q^\top A\hat Q$ when $A$ is symmetric. so a symmetric $A_k$ that becomes upper-triangular must become *diagonal*.

</details>

4. **a Jacobi rotation moves $2a_{pq}^2$ to the diagonal.** using that the Frobenius norm of the 2×2 block is unchanged by the rotation, show that $a_{pp}'^2 + a_{qq}'^2 = a_{pp}^2 + a_{qq}^2 + 2a_{pq}^2$ when $a_{pq}' = 0$. why does this imply $\operatorname{off}(A)^2$ drops by exactly $2a_{pq}^2$, even though the other entries in rows $p$ and $q$ change?

<details><summary>solution</summary>

the block's squared Frobenius norm is $a_{pp}^2 + a_{qq}^2 + 2a_{pq}^2$ before and $a_{pp}'^2 + a_{qq}'^2 + 0$ after, and orthogonal similarity preserves it. for the rest of rows/columns $p, q$: each pair $(a_{kp}, a_{kq})$ with $k \ne p, q$ is rotated by a 2-D rotation, so $a_{kp}^2 + a_{kq}^2$ is unchanged. those entries all stay off-diagonal. entries outside rows and columns $p, q$ don't change at all. so the only change in $\operatorname{off}^2$ is losing $2a_{pq}^2$.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. PageRank on a nine-page web with your `power_iteration`: who ranks first, and is it who you'd guess from the link counts?
2. the convergence of power iteration for different eigenvalue gaps, measuring the eigenvector and the eigenvalue separately.
3. watching the QR algorithm drive a matrix to diagonal form, and then a symmetric 2×2 matrix where it never moves at all.

## check your understanding

1. power iteration on a matrix with eigenvalues $10, 9.9, 1$: roughly how many iterations to reduce the eigenvector error by $10^{-6}$?

<details><summary>answer</summary>

the error shrinks by $0.99$ per step, so you need $k \approx \ln(10^{-6})/\ln(0.99) \approx 1375$ iterations. a shift helps here: power iteration on $A - 5I$ (eigenvalues $5, 4.9, -4$) has ratio $0.98$, which is faster but still slow. inverse iteration with $\mu = 10.2$ has ratio $0.2/0.3$ per step.

</details>

2. why does `inverse_iteration` factor once and solve every step, instead of calling `solve` every step?

<details><summary>answer</summary>

the matrix $A - \mu I$ never changes, so the $O(n^3)$ factorization can be reused. each iteration is then two $O(n^2)$ triangular solves. calling `solve` would refactor every iteration.

</details>

3. the QR algorithm leaves the eigenvalues unchanged at every step. which line of algebra shows it?

<details><summary>answer</summary>

$A_{k+1} = R_kQ_k = Q_k^\top(Q_kR_k)Q_k = Q_k^\top A_kQ_k$: an orthogonal similarity transform. similar matrices have the same characteristic polynomial.

</details>

4. in `eigh`, after the sweeps you sort the eigenvalues. what else must you reorder, and what breaks if you forget?

<details><summary>answer</summary>

the columns of $V$, with the same permutation. if you sort $w$ alone, then $AV = V\operatorname{diag}(w)$ fails: column $j$ of $V$ is no longer the eigenvector for $w[j]$.

</details>

5. why is the Jacobi rotation angle chosen with $|\theta| \le \pi/4$ (the smaller root)?

<details><summary>answer</summary>

both roots zero $a_{pq}$. the larger angle (more than $45°$) essentially swaps the roles of $p$ and $q$ and changes the other entries in those rows and columns by more, undoing earlier progress. with the small angle, the rotation tends to the identity as $a_{pq} \to 0$, so late rotations barely disturb a nearly diagonal matrix. that's the choice under which cyclic Jacobi is known to converge, and quadratically in its final phase (Golub & Van Loan, §8.5). the formula for the small root also avoids cancellation.

</details>

6. a Markov chain's transition matrix $M$ is column-stochastic. why is $1$ always an eigenvalue, and why does power iteration on $M$ compute a stationary distribution?

<details><summary>answer</summary>

the columns sum to one, so $\mathbf 1^\top M = \mathbf 1^\top$: $1$ is an eigenvalue of $M^\top$ with eigenvector $\mathbf 1$, and $M$ and $M^\top$ share eigenvalues. all eigenvalues satisfy $|\lambda| \le 1$ (since $\|M\|_1 = 1$), so if the chain is irreducible and aperiodic, $1$ is strictly dominant and power iteration converges to its eigenvector $\pi = M\pi$, the stationary distribution. PageRank's teleportation term guarantees exactly these conditions, with $|\lambda_2| \le \alpha$.

</details>

7. (the predict callout in idea 1.) what does power iteration do on $\begin{pmatrix}5&0\\0&-5\end{pmatrix}$?

<details><summary>answer</summary>

it never converges. writing $v_0 = (c_1, c_2)$, the iterates are $\propto (c_1, (-1)^kc_2)$: the vector flips back and forth between two directions, and neither is an eigenvector (unless $c_1$ or $c_2$ happened to be zero). the Rayleigh quotient sits at the constant $5(c_1^2 - c_2^2)/(c_1^2 + c_2^2)$, a meaningless weighted average of $5$ and $-5$, and the residual never shrinks, so the loop runs to `max_iters`. power iteration needs a *strictly* dominant eigenvalue in magnitude. a shift ($A - \mu I$ with $\mu \ne 0$) breaks the tie.

</details>

## common bugs

- **returning $\|Av\|$ as the eigenvalue.** symptom: correct magnitude, wrong sign for negative dominant eigenvalues. use the Rayleigh quotient.
- **stopping when $v$ stops changing.** symptom: never stops for $\lambda < 0$ (the sign alternates) and returns after `max_iters` regardless. test the residual $\|Av - \lambda v\|$.
- **inverting instead of factoring.** forming `inv(A - mu*I)` or calling `solve` inside the loop. symptom: correct but slow, plus a crash when $\mu$ hits an eigenvalue exactly.
- **non-reproducible randomness.** using `np.random.randn` (global state) instead of the `rng` argument. symptom: results change from run to run, and the reproducibility check fails.
- **updating rows before saving them.** in a Jacobi rotation, `A[:, p] = c*A[:, p] - s*A[:, q]` followed by `A[:, q] = s*A[:, p] + c*A[:, q]` uses the *new* column $p$ in the second line. symptom: off-diagonal mass grows instead of shrinking. save copies, or update both in one tuple assignment.
- **sorting $w$ without $V$.** symptom: correct eigenvalues, failing $AV = V\operatorname{diag}(w)$.

## going deeper

- Trefethen & Bau (1997), *Numerical Linear Algebra*, SIAM, lectures 24–30: eigenvalue problems, Rayleigh quotient and inverse iteration, the QR algorithm with and without shifts, and Jacobi. [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
- Golub & Van Loan (2013), *Matrix Computations*, 4th ed., Johns Hopkins University Press, chapter 7 (unsymmetric QR) and chapter 8 (symmetric eigenproblems; §8.5 on Jacobi methods).
- Demmel & Veselić (1992), "Jacobi's method is more accurate than QR", *SIAM J. Matrix Anal. Appl.* 13(4): 1204–1245. [doi:10.1137/0613074](https://doi.org/10.1137/0613074)
- Francis (1961), "The QR transformation: a unitary analogue to the LR transformation, part 1", *The Computer Journal* 4(3): 265–271. [doi:10.1093/comjnl/4.3.265](https://doi.org/10.1093/comjnl/4.3.265)
- Page, Brin, Motwani & Winograd (1999), "The PageRank citation ranking: bringing order to the web", Stanford InfoLab technical report. [ilpubs.stanford.edu:8090/422](http://ilpubs.stanford.edu:8090/422/)
- Bryan & Leise (2006), "The \$25,000,000,000 eigenvector: the linear algebra behind Google", *SIAM Review* 48(3): 569–581. a short, readable account of experiment 1. [doi:10.1137/050623280](https://doi.org/10.1137/050623280)

## connections

- ← builds on: `lu`/`lu_solve` (topic 03), `qr` (topic 04), diagonalization as a change of basis (topic 02).
- → used in: the SVD (topic 06: singular values are square roots of eigenvalues of $A^\top A$), condition-number estimation (topic 08), the eigenfaces capstone, Markov chains and their stationary distributions (ch 04), step sizes and Newton's method from Hessian eigenvalues (ch 06), PCA and spectral clustering (ch 11), graph Laplacians in GNNs (ch 20).
