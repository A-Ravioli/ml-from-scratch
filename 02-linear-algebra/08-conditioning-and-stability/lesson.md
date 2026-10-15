# conditioning & stability

## why this matters

you fit a polynomial and the coefficients come out as $\pm 10^{7}$. you solve a linear system and the answer changes in the third digit when you round the data in the twelfth. a colleague's covariance matrix "isn't positive definite" though it obviously should be. is the algorithm broken, or the problem?

numerical linear algebra has a precise answer, and it's the most useful idea in this chapter. two separate questions decide whether a computed answer can be trusted:

- **conditioning** is a property of the *problem*: how much can the exact answer change when the input changes slightly? an ill-conditioned problem amplifies the unavoidable errors in its data (measurement noise, rounding to 16 digits) no matter how you solve it.
- **stability** is a property of the *algorithm*: does it give the exact answer to a slightly perturbed problem? a backward-stable algorithm adds no more error than rounding the input already did.

put together they give the rule of thumb that explains every numerical surprise in this chapter: **forward error ≲ condition number × backward error.** for a backward-stable solve in double precision, you lose about $\log_{10}\kappa$ of your 16 digits. you'll build the tools to measure each piece: matrix norms, the condition number (exact via your SVD, and a cheap estimate via your LU), and the backward error of any computed solution.

## warm-up: predict first

1. $\det(0.1 I_{50}) = 10^{-50}$. is this matrix nearly singular?
2. a system has condition number $10^{10}$, and you solve it in double precision (about 16 digits). how many digits of the answer can you trust?
3. your solver returns an $x$ whose forward error is 100%. is the solver necessarily bad?
4. the residual $\|b - Ax\|$ is tiny. is $x$ necessarily close to the true solution?
5. to estimate $\kappa(A) = \|A\|\,\|A^{-1}\|$, do you need to compute $A^{-1}$?

## prerequisites check

- [ ] you know floating point stores about 16 significant digits, with unit roundoff $\varepsilon \approx 1.1 \times 10^{-16}$ (→ ch 00 · topic 03)
- [ ] you can factor $PA = LU$ and solve with the factors (→ ch 02 · topic 03)
- [ ] you know power iteration and inverse iteration (→ ch 02 · topic 05)
- [ ] you can compute an SVD, and know $\|A\|_2 = \sigma_1$ (→ ch 02 · topic 06)

## core ideas

### idea 1: measuring the size of a matrix

to talk about "small changes" we need sizes. for vectors, the common norms are $\|x\|_1 = \sum|x_i|$, $\|x\|_2 = \sqrt{\sum x_i^2}$ and $\|x\|_\infty = \max|x_i|$. for a matrix, the natural size is **how much it can stretch a vector**:

$$\|A\| = \max_{x \ne 0}\frac{\|Ax\|}{\|x\|}.$$

this is the **induced** (or operator) norm. by construction, $\|Ax\| \le \|A\|\,\|x\|$ and $\|AB\| \le \|A\|\,\|B\|$, the two inequalities every error bound is built from. each vector norm induces its own matrix norm, and three of them have closed forms:

- $\|A\|_1$ = the largest absolute **column** sum (put all of $x$'s 1-norm budget on the best column);
- $\|A\|_\infty$ = the largest absolute **row** sum (take $x = (\pm1, \dots, \pm1)$ matching the signs of the best row);
- $\|A\|_2 = \sigma_1$, the largest singular value (topic 06: the longest semi-axis of the ellipse).

the **Frobenius norm** $\|A\|_F = \sqrt{\sum_{ij}a_{ij}^2} = \sqrt{\sum\sigma_i^2}$ isn't induced by any vector norm, but it's cheap and it bounds the 2-norm: $\|A\|_2 \le \|A\|_F \le \sqrt{\operatorname{rank}A}\,\|A\|_2$. all of these agree to within factors of $n$, so for questions like "is the error $10^{-15}$ or $10^{-3}$", the choice rarely matters. we'll default to the 2-norm.

> **▶ build: stage 1.** implement `matrix_norm(A, ord=2)` for `ord` in `{1, 2, np.inf, "fro"}`. use your `svd` for the 2-norm, and raise `ValueError` for anything else.

### idea 2: the condition number

perturb the right-hand side of $Ax = b$ to $b + \delta b$. the solution moves by $\delta x = A^{-1}\delta b$, so $\|\delta x\| \le \|A^{-1}\|\,\|\delta b\|$. also $\|b\| \le \|A\|\,\|x\|$. divide the two:

$$\frac{\|\delta x\|}{\|x\|} \le \underbrace{\|A\|\,\|A^{-1}\|}_{\kappa(A)}\;\frac{\|\delta b\|}{\|b\|}.$$

the **condition number** $\kappa(A)$ is the worst-case amplification of *relative* error. a similar bound holds for perturbations of $A$ itself, to first order. in the 2-norm, $\|A^{-1}\|_2 = 1/\sigma_n$, so

$$\kappa_2(A) = \frac{\sigma_1}{\sigma_n}.$$

it's the ratio of the ellipse's longest axis to its shortest: how lopsided $A$ makes the unit sphere. by convention, $\kappa = \infty$ for singular $A$, and for rectangular full-rank $A$ the same ratio governs least squares (topic 04).

**what κ means.** $\kappa \ge 1$ always, with $\kappa = 1$ exactly for scaled orthogonal matrices. there's a beautiful geometric reading too: $\sigma_n$ is the 2-norm distance from $A$ to the nearest singular matrix (on paper 3), so $1/\kappa = \sigma_n/\sigma_1$ is the *relative* distance to singularity. note what κ doesn't care about: overall scale. that answers warm-up 1. $0.1I$ has a microscopic determinant but $\kappa = 1$: it's as far from singular as a matrix can be. the determinant measures volume, not near-singularity.

**the rule of thumb.** your data is only known to relative accuracy $\varepsilon \approx 10^{-16}$, if only because it's stored in floating point. so even a perfect algorithm can only promise a relative error of about $\kappa\varepsilon$: you lose about $\log_{10}\kappa$ digits (warm-up 2: $\kappa = 10^{10}$ leaves about 6). when $\kappa \gtrsim 1/\varepsilon \approx 10^{16}$, the matrix is singular to working precision, and the computed answer may have no correct digits at all.

> **🔮 predict:** the $n \times n$ Hilbert matrix $H_{ij} = 1/(i + j - 1)$ looks harmless: all entries between 0 and 1, symmetric, positive definite. at what size $n$ do you think it becomes singular to working precision?

> **▶ build: stage 2.** implement `cond(A)` = $\sigma_{\max}/\sigma_{\min}$ from your `svd`, returning `np.inf` when $\sigma_{\min}$ is exactly zero.

### idea 3: estimating κ without an SVD

an SVD costs much more than a solve, and you usually want κ *alongside* a solve, as a warning light. but you have just computed $PA = LU$, and that's enough to estimate κ in $O(n^2)$ extra work (warm-up 5: no inverse needed).

- $\sigma_1^2$ is the largest eigenvalue of $A^\top A$. power iteration finds it using only products $v \mapsto A^\top(Av)$, at $O(n^2)$ each.
- $\sigma_n^2$ is the *smallest* eigenvalue of $A^\top A$, so it's the largest of $(A^\top A)^{-1}$. inverse iteration (topic 05) finds it, and each step needs solves with $A^\top A$: $x = A^{-1}A^{-\top}u$. both solves come from the LU you already have. $A x = y$ is your `lu_solve`. $A^\top y = u$ uses the same factors transposed: since $A = P^\top LU$, we have $A^\top = U^\top L^\top P$, so solve $U^\top z = u$ (lower-triangular), then $L^\top w = z$ (upper, unit diagonal), then $y = P^\top w$.

then $\kappa \approx \sqrt{\lambda_{\max}(A^\top A)/\lambda_{\min}(A^\top A)}$. the convergence rates are the eigenvalue ratios of $A^\top A$, which are the *squares* of the singular value ratios, so a few dozen iterations usually give several digits. production libraries go further: LAPACK's `xGECON` estimates $\|A^{-1}\|_1$ with Hager's method as refined by Higham (1988), using about five solves, accurate to within a small factor. a condition *estimate* only has to tell $10^3$ from $10^{12}$.

> **▶ build: stage 3.** implement `cond_estimate(A, *, max_iters=100, rng=None)`: one `lu`, then power iteration on $A^\top A$ and inverse iteration on $A^\top A$ through the LU factors. return `np.inf` if $U$ has an exactly zero pivot.

### idea 4: backward error and stability

how do you judge a computed solution $\hat x$ when you don't know the true $x$? the **forward error** $\|\hat x - x\|/\|x\|$ needs the answer. the **residual** $r = b - A\hat x$ is computable, but its size alone means little, because it depends on scaling. the right normalization is the **backward error**:

$$\eta(\hat x) = \frac{\|b - A\hat x\|}{\|A\|\,\|\hat x\| + \|b\|}.$$

Rigal & Gaches (1967) proved that $\eta$ is exactly the size of the smallest relative perturbation, $\|\delta A\| \le \eta\|A\|$ and $\|\delta b\| \le \eta\|b\|$, for which $\hat x$ solves $(A + \delta A)\hat x = b + \delta b$ *exactly*. so $\eta$ answers the question "for how nearby a problem is $\hat x$ the exact answer?" if $\eta \approx 10^{-16}$, $\hat x$ is as good as any algorithm could produce from data stored in floating point.

an algorithm is **backward stable** if it always produces solutions with $\eta = O(\varepsilon)$. combine that with idea 2, and to first order:

$$\underbrace{\frac{\|\hat x - x\|}{\|x\|}}_{\text{forward error}} \;\lesssim\; \kappa(A)\cdot\underbrace{\eta(\hat x)}_{\text{backward error}}.$$

this rule untangles every case. a large forward error with a tiny backward error means the *problem* is ill-conditioned, and no algorithm could do better (warm-up 3: not necessarily the solver's fault). a large backward error means the *algorithm* is unstable. and warm-up 4: a tiny residual guarantees a tiny backward error, *not* a small forward error. that still needs small κ.

the scorecard for what you've built: Householder QR and `lstsq` via QR are backward stable. Gaussian elimination with partial pivoting is backward stable *in practice*: its backward error is bounded by the growth factor (topic 03's lab), which is small except on matrices like Wilkinson's. elimination *without* pivoting is unstable. classical Gram–Schmidt is unstable as a way to get an orthogonal $Q$. the normal equations are not backward stable for least squares, because they solve a problem with condition number $\kappa^2$. Cholesky on SPD matrices is backward stable with no pivoting at all.

> **🔮 predict:** you solve the $12 \times 12$ Hilbert system $Hx = H\mathbf 1$ with your pivoted `solve`. the forward error turns out to be about 20%. what do you expect the backward error to be?

> **▶ build: stage 4.** implement `backward_error(A, x, b)` with 2-norms (use your `matrix_norm`).

## worked example

$$A = \begin{pmatrix}1 & 1\\ 1 & 1.0001\end{pmatrix},\qquad b = \begin{pmatrix}2\\2.0001\end{pmatrix},\qquad x = \begin{pmatrix}1\\1\end{pmatrix}.$$

**norms.** for this $A$, column sums and row sums both peak at $2.0001$, so $\|A\|_1 = \|A\|_\infty = 2.0001$. $A$ is symmetric, so its singular values are the absolute eigenvalues: $\lambda = \frac{2.0001 \pm \sqrt{0.0001^2 + 4}}{2}$, giving $\sigma_1 \approx 2.00005$ and $\sigma_2 \approx 4.999875 \times 10^{-5}$. (for the non-symmetric norm examples in the tests: $\begin{pmatrix}1&-2\\3&4\end{pmatrix}$ has $\|\cdot\|_1 = 6$, $\|\cdot\|_\infty = 7$, and $\|\cdot\|_F = \sqrt{30}$.)

**condition number.** $\kappa_2 = \sigma_1/\sigma_2 \approx 2.00005/4.999875 \times 10^{-5} \approx 40{,}002$. the inverse is $A^{-1} = 10^4\begin{pmatrix}1.0001 & -1\\ -1 & 1\end{pmatrix}$, with entries of size $10^4$. the matrix is about $1/40{,}000$ of the way (relatively) from being singular.

**amplification in action.** change $b_2$ from $2.0001$ to $2.0002$, a relative change of $\|\delta b\|/\|b\| \approx 10^{-4}/2.83 \approx 3.5 \times 10^{-5}$. the new solution is $(0, 2)$. the relative change in $x$ is $\|(-1, 1)\|/\|(1,1)\| = 1$, which is 100%. the amplification is about $1/3.5 \times 10^{-5} \approx 28{,}000$, close to the worst case $\kappa = 40{,}002$. why so close? $b = (2, 2.0001)$ points almost exactly along $u_1 \approx (1, 1)/\sqrt2$, which makes $\|x\|$ as small as it can be relative to $\|b\|$. and $\delta b = (0, 10^{-4})$ is at $45°$ to $u_2 \approx (1, -1)/\sqrt2$, so a fraction $\cos 45° \approx 0.707$ of it lies along the weakest direction: $0.707 \times 40{,}002 \approx 28{,}000$.

**backward error.** read $\hat x = (0, 2)$ as an attempted solution of the *original* system. the residual is $b - A\hat x = (2 - 2,\ 2.0001 - 2.0002) = (0, -10^{-4})$, so

$$\eta = \frac{10^{-4}}{2.00005 \cdot 2 + 2.8285} \approx 1.46 \times 10^{-5}.$$

an answer that's 100% wrong in the forward sense is the exact solution of a problem perturbed by only $0.0015\%$. consistent with the rule: $\kappa\,\eta \approx 40{,}002 \times 1.46 \times 10^{-5} \approx 0.6$, the same order as the forward error.

the tests check $\kappa = 40{,}002$, the three hand norms, and $\eta \approx 1.464 \times 10^{-5}$.

## on paper

1. **the 1-norm is the max column sum.** show that $\|Ax\|_1 \le (\max_j\sum_i|a_{ij}|)\,\|x\|_1$, and find an $x$ that attains it.

<details><summary>solution</summary>

$\|Ax\|_1 = \sum_i|\sum_ja_{ij}x_j| \le \sum_j|x_j|\sum_i|a_{ij}| \le \max_j(\sum_i|a_{ij}|)\sum_j|x_j|$. equality holds for $x = e_{j^*}$, where $j^*$ is the column with the largest absolute sum: then $Ax$ is that column, and its 1-norm is the column sum.

</details>

2. **perturbing $A$.** suppose $(A + \delta A)(x + \delta x) = b$ with $Ax = b$. show that, to first order, $\frac{\|\delta x\|}{\|x\|} \lesssim \kappa(A)\frac{\|\delta A\|}{\|A\|}$.

<details><summary>solution</summary>

expand: $Ax + A\,\delta x + \delta A\,x + \delta A\,\delta x = b$. drop the second-order term $\delta A\,\delta x$ and use $Ax = b$: $\delta x \approx -A^{-1}\delta A\,x$. so $\|\delta x\| \lesssim \|A^{-1}\|\,\|\delta A\|\,\|x\|$, i.e. $\frac{\|\delta x\|}{\|x\|} \lesssim \|A^{-1}\|\,\|A\|\frac{\|\delta A\|}{\|A\|}$.

</details>

3. **distance to singularity.** show that $A + E$ with $E = -\sigma_nu_nv_n^\top$ is singular and $\|E\|_2 = \sigma_n$. (it's also true that no smaller $E$ works.) conclude that $\sigma_n/\sigma_1 = 1/\kappa_2$ is the relative 2-norm distance to the nearest singular matrix.

<details><summary>solution</summary>

$(A + E)v_n = \sigma_nu_n - \sigma_nu_n(v_n^\top v_n) = 0$, so $v_n$ is in the null space and $A + E$ is singular. $E$ is rank one with singular value $\sigma_n$, so $\|E\|_2 = \sigma_n$. relative to $\|A\|_2 = \sigma_1$, that distance is $\sigma_n/\sigma_1 = 1/\kappa_2$. (minimality is Eckart–Young from topic 06, with $k = n - 1$.)

</details>

4. **backward error for perturbations of $A$ only.** given $\hat x \ne 0$ with residual $r = b - A\hat x$, show that $\delta A = r\hat x^\top/\|\hat x\|^2$ satisfies $(A + \delta A)\hat x = b$, with $\|\delta A\|_2 = \|r\|/\|\hat x\|$, and that no smaller $\delta A$ works.

<details><summary>solution</summary>

$(A + \delta A)\hat x = A\hat x + r(\hat x^\top\hat x)/\|\hat x\|^2 = A\hat x + r = b$. $\delta A$ is rank one, $\|r\hat x^\top\|_2 = \|r\|\,\|\hat x\|$, so $\|\delta A\|_2 = \|r\|/\|\hat x\|$. any $\delta A$ with $\delta A\,\hat x = r$ has $\|\delta A\| \ge \|r\|/\|\hat x\|$ by the definition of the induced norm. so the relative backward error in $A$ alone is $\|r\|/(\|A\|\,\|\hat x\|)$. Rigal–Gaches adds $\|b\|$ to the denominator when $b$ may be perturbed too.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. solving Hilbert systems of growing size with your `solve`, and tracking the condition number, forward error and backward error together.
2. your `cond` (SVD) against your `cond_estimate` (one LU plus iterations): speed and agreement.
3. choosing the directions of $b$ and of a perturbation $\delta b$, to see when the bound $\kappa$ is actually attained.

## check your understanding

1. why is κ defined with *relative* errors rather than absolute ones?

<details><summary>answer</summary>

so it's independent of units and scale: $\kappa(cA) = \kappa(A)$. floating point errors are relative too (about $\varepsilon$ times the number's size), so relative amplification is exactly what predicts the loss of digits.

</details>

2. two algorithms solve the same ill-conditioned system: A gives forward error $10^{-4}$ and backward error $10^{-16}$, B gives forward error $10^{-5}$ and backward error $10^{-9}$. which is the better algorithm?

<details><summary>answer</summary>

A. its result is exactly right for data perturbed at the rounding level, so it's as good as the problem allows. B's smaller forward error on this one instance is luck. its backward error shows it introduces errors $10^7$ times larger than necessary, and on another right-hand side it could be much worse.

</details>

3. the rule "you lose $\log_{10}\kappa$ digits" is a worst case. when is it pessimistic?

<details><summary>answer</summary>

when the right-hand side and the errors don't line up with the worst singular directions (experiment 3: random perturbations amplify far less than κ for most right-hand sides). also when the errors are structured. for example, componentwise-small perturbations of a badly *scaled* (but otherwise well-behaved) matrix are governed by a smaller, componentwise condition number. row scaling can make κ huge without making the problem hard.

</details>

4. why do the normal equations for least squares lose about twice as many digits as QR?

<details><summary>answer</summary>

they explicitly solve a system with matrix $A^\top A$, whose condition number is $\kappa(A)^2$, and the rounding errors committed while forming $A^\top A$ count as perturbations of *that* problem. QR works with $A$ and is backward stable for the least-squares problem itself, whose sensitivity is governed by $\kappa(A)$ (plus a $\kappa^2$ term weighted by the residual size, which vanishes for consistent systems).

</details>

5. your covariance matrix fails `cholesky` though it should be positive semidefinite. what's going on, and what are the usual fixes?

<details><summary>answer</summary>

it's positive *semi*definite or nearly so (more features than samples, collinear features), so its smallest eigenvalues are at the rounding level, and rounding pushes some slightly negative. κ is effectively infinite. fixes: add a small ridge $\lambda I$ (which raises every eigenvalue by $\lambda$ and caps κ at about $\lambda_{\max}/\lambda$), drop redundant features, or use an SVD/eigendecomposition that tolerates semidefiniteness.

</details>

6. a residual check `np.allclose(A @ x, b)` passes. what have you verified?

<details><summary>answer</summary>

that the backward error is small, so the solver behaved. you haven't verified that $x$ is accurate. for that you also need κ to be small (forward error ≲ κ × backward error).

</details>

## common bugs

- **using the determinant as a singularity test.** symptom: perfectly conditioned matrices flagged as singular (tiny det from scaling), and nearly singular ones passed (det of order 1). use κ, or $\sigma_{\min}/\sigma_{\max}$.
- **the Frobenius norm where the 2-norm was meant.** symptom: condition numbers or errors off by a factor up to $\sqrt n$. fine for orders of magnitude, wrong for exact checks.
- **mixing up row and column sums.** $\|A\|_1$ is the max *column* sum, $\|A\|_\infty$ the max *row* sum. symptom: the two swapped on non-symmetric matrices.
- **solving with $A$ when you meant $A^\top$** in `cond_estimate`. symptom: a sensible-looking but wrong $\sigma_{\min}$ (it's then estimating something about $A^2$, not $A^\top A$). derive the transposed solve from $A^\top = U^\top L^\top P$, and test against `cond`.
- **judging a solution by its residual alone.** symptom: "my residual is $10^{-14}$ but the answer is wrong". that's ill-conditioning, not a bug.

## going deeper

- Trefethen & Bau (1997), *Numerical Linear Algebra*, SIAM, lectures 3 (norms), 12–15 (conditioning, floating point, stability, backward stability) and 22 (stability of Gaussian elimination). [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
- Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, chapters 1–3 (principles of finite precision), 7 (perturbation theory for linear systems) and 15 (condition number estimation). [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
- Rigal & Gaches (1967), "On the compatibility of a given solution with the data of a linear system", *Journal of the ACM* 14(3): 543–548. the backward error formula of stage 4. [doi:10.1145/321406.321416](https://doi.org/10.1145/321406.321416)
- Higham (1988), "FORTRAN codes for estimating the one-norm of a real or complex matrix, with applications to condition estimation", *ACM Transactions on Mathematical Software* 14(4): 381–396. the refinement of Hager's estimator that LAPACK uses. [doi:10.1145/50063.214386](https://doi.org/10.1145/50063.214386)
- Hager (1984), "Condition estimates", *SIAM J. Sci. Stat. Comput.* 5(2): 311–316. the cheap 1-norm estimator behind LAPACK. [doi:10.1137/0905023](https://doi.org/10.1137/0905023)
- Goldberg (1991), "What every computer scientist should know about floating-point arithmetic", *ACM Computing Surveys* 23(1): 5–48. [doi:10.1145/103162.103163](https://doi.org/10.1145/103162.103163)

## connections

- ← builds on: floating point (ch 00 · topic 03), LU and growth factors (topic 03), QR versus normal equations (topic 04), power and inverse iteration (topic 05), the SVD (topic 06).
- → used in: interpreting every fit and every solve from here on. the condition number of the Hessian sets the convergence rate of gradient descent (ch 06), ridge regularization caps κ (ch 08), kernel matrices and GP jitter (ch 09), why normalization layers help training (ch 15), and mixed-precision training (ch 26).
