# orthogonality & least squares

## why this matters

you measure a quantity at 1,000 points and want the line, or polynomial, or linear model, that best explains them. that's 1,000 equations in a handful of unknowns, $Ax = b$ with $A$ tall and thin. no exact solution exists, because real data is noisy, so "solve" has to mean something else: find the $x$ that makes $Ax$ as close to $b$ as possible. this is **least squares**, and it's the first model you'll train in chapter 08, the inner loop of Gauss–Newton, and the projection step in PCA.

two tempting shortcuts both cause trouble. the textbook formula $x = (A^\top A)^{-1}A^\top b$ is correct on paper. in floating point it can throw away half your significant digits (experiment 2 shows this). and building an orthonormal basis by the obvious Gram–Schmidt procedure can produce "orthonormal" vectors that are nowhere near orthogonal (experiment 1). the cure for both is the same idea: **orthogonality**, used carefully. orthogonal matrices are the one kind of matrix that never amplifies errors, so the reliable algorithms are built from them.

by the end you'll have three ways to compute $A = QR$ and a least-squares solver that the rest of the course builds on.

## warm-up: predict first

1. you have an orthonormal basis $q_1, \dots, q_n$ of $\mathbb{R}^n$. what does it cost to write a vector $v$ in that basis: solving an $n \times n$ system, or something cheaper?
2. classical and modified Gram–Schmidt perform the same arithmetic in exact math. will they give the same $Q$ in floating point?
3. you fit the best line to some points. is the vector of residuals (data minus fit) orthogonal to anything in particular?
4. if $A$ has condition number $10^8$, what's the condition number of $A^\top A$?
5. how many digits of a least-squares solution can you trust if $\kappa(A) = 10^8$ and your data has 16 significant digits?

## prerequisites check

- [ ] you can compute dot products and norms, and know that $u \cdot v = 0$ means perpendicular (→ ch 01 · topic 04)
- [ ] you know what span, basis and dimension mean (→ ch 02 · topic 01)
- [ ] you can solve an upper-triangular system with your `solve_triangular` (→ ch 02 · topic 03)
- [ ] you can differentiate a quadratic like $(ax - b)^2$ and set the derivative to zero (→ ch 01 · topic 02)

## core ideas

### idea 1: orthonormal bases and projections

vectors $q_1, \dots, q_k$ are **orthonormal** if $q_i^\top q_j = 0$ for $i \ne j$ and $\|q_i\| = 1$. stack them as the columns of $Q$ and the whole definition fits in one equation: $Q^\top Q = I$. orthonormal bases are the convenient ones because coordinates become dot products: if $v = \sum_i c_i q_i$, then dotting with $q_j$ kills every term except one, so $c_j = q_j^\top v$. no system to solve (warm-up 1). and multiplying by $Q$ preserves lengths and angles, $\|Qx\| = \|x\|$, which is why orthogonal matrices never amplify rounding errors.

**projection.** the closest point to $v$ on the line through a unit vector $q$ is $(q^\top v)\,q$. what's left over, $v - (q^\top v)q$, is perpendicular to $q$. for a subspace with orthonormal basis $Q$ (columns $q_1..q_k$), the closest point is the sum of the one-dimensional projections:

$$\operatorname{proj}(v) = \sum_i (q_i^\top v)\,q_i = QQ^\top v, \qquad v - QQ^\top v \perp \text{every } q_i.$$

**Gram–Schmidt** turns any independent columns $a_1, \dots, a_n$ into an orthonormal basis of the same span by projecting away what you've already got:

$$v_j = a_j - \sum_{i<j}(q_i^\top a_j)\,q_i, \qquad q_j = v_j / \|v_j\|.$$

record the coefficients $r_{ij} = q_i^\top a_j$ and $r_{jj} = \|v_j\|$, and rearrange: $a_j = \sum_{i \le j} r_{ij} q_i$. that's $A = QR$, with $R$ upper-triangular because $a_j$ only uses $q_1..q_j$. so Gram–Schmidt is a matrix factorization, just as elimination was.

> **▶ build: stage 1.** implement `gram_schmidt(A, *, modified=True)` for the case `modified=False`: classical Gram–Schmidt, returning the reduced `Q` (m, n) and `R` (n, n) with a positive diagonal. compute column $j$'s coefficients all at once, `R[:j, j] = Q[:, :j].T @ A[:, j]`.

### idea 2: the same arithmetic, in a better order

in exact arithmetic the projection coefficients don't care whether you compute them from the original $a_j$ or from a partly cleaned-up version of it, because $q_i^\top q_l = 0$. in floating point they do care. classical Gram–Schmidt (CGS) computes every coefficient for $a_j$ from the *original* vector. if $a_j$ is nearly in the span of the earlier columns, $v_j$ is the small difference of large, nearly equal vectors. cancellation wipes out most of its correct digits, and the errors that remain point back along the earlier $q$'s. the computed $q_j$ isn't orthogonal to them.

**modified Gram–Schmidt** (MGS) reorders the loops. as soon as $q_j$ is known, subtract its component from *all remaining* columns:

$$\text{for } j = 1..n:\quad q_j = v_j/\|v_j\|, \qquad v_l \leftarrow v_l - (q_j^\top v_l)\,q_j \ \text{ for all } l > j.$$

now each coefficient is computed from a vector that's already had the earlier directions removed, *including the errors made while removing them*. each projection partly cleans up after the previous ones.

the difference is dramatic. for a matrix with condition number $\kappa$ (largest over smallest singular value; topic 08 makes this precise), the loss of orthogonality $\|Q^\top Q - I\|$ is about $\varepsilon\kappa$ for MGS (Björck, 1967), but grows like $\varepsilon\kappa^2$ for CGS (Giraud, Langou, Rozložník & van den Eshof, 2005), where $\varepsilon \approx 10^{-16}$ is machine precision. at $\kappa = 10^8$, that's about $10^{-8}$ for MGS against about $10^{-2}$ for CGS, and by $\kappa \approx 10^9$ the CGS columns are no longer orthogonal at all.

> **🔮 predict:** which of the two is easier to *parallelize*: computing all of $R[:j, j]$ at once (CGS), or the strictly sequential updates of MGS? (this is why CGS, with a second "reorthogonalization" pass, is still used in practice.)

> **▶ build: stage 2.** add the `modified=True` branch to `gram_schmidt` (it's the default). keep a working copy `V` of the columns, and after normalizing column $j$, update all of `V[:, j+1:]` at once.

### idea 3: Householder reflections

MGS is better, but its $Q$ still drifts from orthogonality as $\kappa$ grows. the fix is to *never form $Q$ by subtraction at all*. build it as a product of exactly orthogonal matrices instead.

a **Householder reflection** reflects space across the hyperplane perpendicular to a unit vector $v$:

$$H = I - 2vv^\top, \qquad H^\top = H, \qquad H^2 = I.$$

$H$ is orthogonal and symmetric, and it's its own inverse. given any vector $x$, you can pick $v$ so that $H$ sends $x$ onto the first axis: $Hx = \mp\|x\|e_1$. reflecting $x$ to the point $y$ with the same length means reflecting across the perpendicular bisector of $x$ and $y$, so take $v \propto x - y$. with $y = -\operatorname{sign}(x_1)\|x\|e_1$,

$$v = \frac{x + \operatorname{sign}(x_1)\|x\|e_1}{\big\|x + \operatorname{sign}(x_1)\|x\|e_1\big\|}.$$

(take $\operatorname{sign}(0) = +1$, so a column that's already zero below its top entry, like a column of the identity, still gets a nonzero $v$.) why that sign? the other choice, $x - \operatorname{sign}(x_1)\|x\|e_1$, subtracts two nearly equal numbers in the first entry whenever $x$ already points mostly along $e_1$. that's catastrophic cancellation, and a garbage $v$. with the sign matching $x_1$, you always add.

**QR by reflections.** pick $H_1$ to zero column 1 below the diagonal. then pick $H_2$ that acts only on rows $2..m$ (so column 1 is undisturbed) to zero column 2 below the diagonal, and so on:

$$H_n\cdots H_2H_1A = R \quad\Longrightarrow\quad A = \underbrace{H_1H_2\cdots H_n}_{Q}R.$$

each $H_k$ is orthogonal *by construction*. the rounding errors land in $R$ (where they're harmless and small, relative to $\|A\|$), not in the orthogonality of $Q$. Householder's $Q$ is orthogonal to machine precision **regardless of $\kappa$**.

two practical points. you never form $H$ as a matrix. apply it as $X \leftarrow X - 2v(v^\top X)$, which costs $O(\text{size of } X)$ rather than a full matrix product. and $Q$ comes from applying the stored reflections to the identity. this gives the *complete* $Q$ ($m \times m$), whose last $m - n$ columns are an orthonormal basis for everything $A$'s columns miss. the *reduced* $Q$ is its first $n$ columns. the signs on $R$'s diagonal come out however the reflections leave them, so $\operatorname{diag}(R)$ may be negative. that's fine: flipping the sign of row $i$ of $R$ and column $i$ of $Q$ changes nothing.

> **▶ build: stage 3.** implement `qr(A, mode="reduced")` with Householder reflections. support any shape, both modes, and skip a reflection when the column below the diagonal is already zero.

### idea 4: least squares

we want $\min_x \|Ax - b\|_2$ for tall $A$ ($m > n$, full column rank). geometrically, $Ax$ ranges over the column space of $A$, so the best $Ax$ is the point of that subspace closest to $b$: the **orthogonal projection** of $b$. the residual $r = b - Ax$ is perpendicular to the column space, so it's orthogonal to every column (warm-up 3):

$$A^\top(b - Ax) = 0 \quad\Longleftrightarrow\quad A^\top A\,x = A^\top b.$$

these are the **normal equations**. calculus gives the same thing: $\nabla_x\|Ax - b\|^2 = 2A^\top(Ax - b)$ (you'll derive this in topic 07).

solving the normal equations is the obvious approach, and it's dangerous. $A^\top A$ has condition number $\kappa(A)^2$ (warm-up 4: $10^{16}$). in double precision, that means essentially no correct digits. the information was in $A$ all along; forming $A^\top A$ squared away half of it.

**least squares by QR.** write $A = QR$ (reduced) and complete $Q$ to an orthogonal $[Q\ Q_\perp]$. orthogonal matrices preserve length, so

$$\|Ax - b\|^2 = \left\|\begin{pmatrix}Q^\top\\Q_\perp^\top\end{pmatrix}(QRx - b)\right\|^2 = \|Rx - Q^\top b\|^2 + \|Q_\perp^\top b\|^2.$$

the second term doesn't depend on $x$. the first can be made exactly zero. so

$$\boxed{\,Rx = Q^\top b\,}$$

is a single triangular solve, and it works with $A$'s condition number, not its square. because Householder QR is backward stable, perturbation theory bounds the relative error in $x$ by about

$$\varepsilon\Big(\kappa + \kappa^2\tan\theta\Big),\qquad \theta = \text{the angle between } b \text{ and its fit } Ax,$$

so when the fit is good (small residual, $\tan\theta \approx 0$), $\kappa = 10^8$ leaves you about 8 digits (warm-up 5). as the residual grows, the $\kappa^2\tan\theta$ term takes over, and not just in the worst case: at $\kappa = 10^8$, a residual of 10% of $\|b\|$ already costs about 7 digits (an error around $3 \times 10^{-3}$ instead of $3 \times 10^{-10}$). least squares itself becomes that sensitive, whatever algorithm you use. the normal equations pay $\varepsilon\kappa^2$ *even when the residual is zero*, which at $\kappa = 10^8$ means about no correct digits.

> **🔮 predict:** your data is $b = Ax_\text{true}$ *exactly* (no noise). does least squares by QR return $x_\text{true}$ exactly? what about the normal equations?

> **▶ build: stage 4.** implement `lstsq(A, b)`: one call to your `qr` and one to your topic-03 `solve_triangular`. it should accept `b` of shape `(m,)` or `(m, k)`.

## worked example

fit the line $y = c + dt$ through the points $(0, 6), (1, 0), (2, 0)$:

$$A = \begin{pmatrix}1&0\\1&1\\1&2\end{pmatrix},\qquad b = \begin{pmatrix}6\\0\\0\end{pmatrix}.$$

**Gram–Schmidt.** $r_{11} = \|a_1\| = \sqrt3$ and $q_1 = (1,1,1)/\sqrt3$. next, $r_{12} = q_1^\top a_2 = (0+1+2)/\sqrt3 = \sqrt3$, so $v_2 = a_2 - \sqrt3\,q_1 = (0,1,2) - (1,1,1) = (-1, 0, 1)$. then $r_{22} = \sqrt2$ and $q_2 = (-1,0,1)/\sqrt2$:

$$Q = \begin{pmatrix}1/\sqrt3 & -1/\sqrt2\\ 1/\sqrt3 & 0\\ 1/\sqrt3 & 1/\sqrt2\end{pmatrix},\qquad R = \begin{pmatrix}\sqrt3 & \sqrt3\\ 0 & \sqrt2\end{pmatrix}.$$

**least squares.** $Q^\top b = (6/\sqrt3,\ -6/\sqrt2) = (2\sqrt3,\ -3\sqrt2)$. back substitution on $Rx = Q^\top b$: $d = -3\sqrt2/\sqrt2 = -3$, then $\sqrt3\,c + \sqrt3(-3) = 2\sqrt3$, so $c = 5$. the best line is $y = 5 - 3t$.

**check the geometry.** the fit is $Ax = (5, 2, -1)$ and the residual is $r = (1, -2, 1)$. $a_1^\top r = 1 - 2 + 1 = 0$ and $a_2^\top r = 0 - 2 + 2 = 0$. ✓ the normal equations agree: $A^\top A = \begin{pmatrix}3&3\\3&5\end{pmatrix}$, $A^\top b = (6, 0)$, solved by $(5, -3)$.

**Householder** gives the same $R$ up to signs. here the first reflection sends $a_1 = (1,1,1)$ to $(-\sqrt3, 0, 0)$, so `qr` returns $r_{11} = -\sqrt3$.

the tests check $Q$, $R$, $|R|$ from `qr`, and $x = (5, -3)$.

## on paper

1. **projections.** show that $P = QQ^\top$ (with $Q^\top Q = I$) satisfies $P^2 = P$ and $P^\top = P$, and that $I - P$ projects onto the orthogonal complement. why is $P$ *not* the identity when $Q$ is tall?

<details><summary>solution</summary>

$P^2 = Q(Q^\top Q)Q^\top = QQ^\top = P$, and $P^\top = (Q^\top)^\top Q^\top = P$. $(I-P)^2 = I - 2P + P^2 = I - P$, and $Q^\top(I-P)v = Q^\top v - Q^\top v = 0$, so $(I-P)v$ is orthogonal to every column of $Q$. for tall $Q$ ($m > k$), $P$ has rank $k < m$, so it can't be $I$. $Q^\top Q = I_k$, but $QQ^\top \ne I_m$.

</details>

2. **the normal equations by calculus.** expand $f(x) = \|Ax - b\|^2 = x^\top A^\top Ax - 2b^\top Ax + b^\top b$ and show that its gradient is $2(A^\top Ax - A^\top b)$. why is the stationary point a minimum when $A$ has full column rank?

<details><summary>solution</summary>

the gradient of $x^\top Mx$ is $(M + M^\top)x = 2Mx$ for symmetric $M = A^\top A$, and the gradient of $c^\top x$ is $c$ (with $c = A^\top b$). so $\nabla f = 2A^\top Ax - 2A^\top b$. the Hessian is $2A^\top A$, and $v^\top A^\top Av = \|Av\|^2 > 0$ for $v \ne 0$ when the columns are independent, so $f$ is strictly convex and the stationary point is the unique minimum.

</details>

3. **Householder hits its target.** with $v = x + \sigma\|x\|e_1$ (unnormalized, $\sigma = \operatorname{sign}(x_1)$), show that $\left(I - 2\frac{vv^\top}{v^\top v}\right)x = -\sigma\|x\|e_1$.

<details><summary>solution</summary>

$v^\top x = \|x\|^2 + \sigma\|x\|x_1$ and $v^\top v = \|x\|^2 + 2\sigma\|x\|x_1 + \|x\|^2 = 2(\|x\|^2 + \sigma\|x\|x_1)$. so $2v^\top x / v^\top v = 1$, and $Hx = x - v = -\sigma\|x\|e_1$. note that $v^\top v \ge 2\|x\|^2$ with this sign choice: no cancellation, ever.

</details>

4. **conditioning of $A^\top A$.** using the SVD $A = U\Sigma V^\top$ (topic 06; take it on faith for now), show that $A^\top A = V\Sigma^2V^\top$, and hence $\kappa(A^\top A) = \kappa(A)^2$.

<details><summary>solution</summary>

$A^\top A = V\Sigma U^\top U\Sigma V^\top = V\Sigma^2V^\top$, an eigendecomposition with eigenvalues $\sigma_i^2$. the 2-norm condition number of an SPD matrix is $\lambda_{\max}/\lambda_{\min} = \sigma_{\max}^2/\sigma_{\min}^2 = \kappa(A)^2$.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how orthogonal the $Q$ from classical GS, modified GS and Householder really is, as the condition number grows.
2. fitting polynomials of rising degree with your `lstsq` and with the normal equations: how many digits each one keeps.
3. how much a single moved data point changes a fitted line, depending on *where* that point sits.

## check your understanding

1. why is $\|Qx\| = \|x\|$ for orthonormal $Q$ the key fact behind the QR approach to least squares?

<details><summary>answer</summary>

it lets you multiply the residual by $[Q\ Q_\perp]^\top$ without changing its length, which splits $\|Ax - b\|^2$ into a part you can zero ($\|Rx - Q^\top b\|^2$) and a part you can't. it's also why multiplying by $Q$ doesn't amplify errors.

</details>

2. modified Gram–Schmidt and Householder both give you $Q$ and $R$. when would you still pick MGS?

<details><summary>answer</summary>

when you want the vectors $q_j$ one at a time as you go, as in iterative methods like Arnoldi and Lanczos, which build an orthonormal basis column by column. or when you only want the reduced $Q$ explicitly and $\kappa$ is modest. and if you run MGS on the augmented matrix $[A\ b]$ (treating $b$ as one more column), the least-squares solution it gives is backward stable, even though its $Q$ drifts (Björck & Paige, 1992).

</details>

3. least squares says the residual is orthogonal to the column space. is it also orthogonal to $b$?

<details><summary>answer</summary>

no. $r^\top b = r^\top(Ax + r) = \|r\|^2$, which is positive unless the fit is exact. the residual is orthogonal to the *fit* $Ax$, not to the data.

</details>

4. in the worked example, what are the projection matrix $P = A(A^\top A)^{-1}A^\top$ applied to $b$ and the residual? and what is $\|b\|^2 = \|Pb\|^2 + \|r\|^2$ numerically?

<details><summary>answer</summary>

$Pb = (5, 2, -1)$ and $r = (1, -2, 1)$. $\|b\|^2 = 36$, $\|Pb\|^2 = 25 + 4 + 1 = 30$, $\|r\|^2 = 6$. $30 + 6 = 36$: Pythagoras, because $Pb \perp r$.

</details>

5. what goes wrong in `lstsq` if two columns of $A$ are identical?

<details><summary>answer</summary>

$A$ is rank-deficient, so the minimizer isn't unique. Householder still produces $Q$ and $R$, but $R$ has a (near-)zero diagonal entry, and back substitution divides by it: `LinAlgError` if it's exactly zero, a huge meaningless answer if it's $10^{-17}$. the fix is the pseudoinverse (topic 06), which picks the minimum-norm solution.

</details>

6. why does `qr` in reduced mode return $R$ of shape $(k, n)$ with $k = \min(m, n)$, instead of always $(n, n)$?

<details><summary>answer</summary>

a wide matrix ($m < n$) has rank at most $m$, so only $m$ orthonormal columns exist. $Q$ is $(m, m)$, and $R$ must then be $(m, n)$ to make $QR$ have $A$'s shape.

</details>

## common bugs

- **CGS when you meant MGS.** computing `Q[:, j] @ A[:, l]` (original column) instead of `Q[:, j] @ V[:, l]` (updated column) quietly turns MGS back into CGS. symptom: tests pass on random matrices but orthogonality fails on ill-conditioned ones.
- **the wrong Householder sign.** `v[0] -= norm(x)` when `x[0] > 0` causes cancellation. symptom: `Q` loses orthogonality, or turns into NaN, on matrices whose columns already point along the axes (including the identity!).
- **forming $H$ explicitly.** building `np.eye(m) - 2*np.outer(v, v)` and multiplying is $O(m^2)$ memory and $O(m^2n)$ per step. symptom: correct but painfully slow for tall matrices. apply $X - 2v(v^\top X)$ instead.
- **forgetting to zero below the diagonal.** after the reflections, entries below the diagonal of $R$ are around $10^{-17}$ rather than exactly $0$. symptom: "R is not upper-triangular" with `np.allclose` passing but exact checks failing. use `np.triu`.
- **solving the normal equations.** `solve(A.T @ A, A.T @ b)` works on nice problems and silently loses half the digits on ill-conditioned ones. symptom: polynomial fits whose coefficients change wildly with the degree.

## going deeper

- Trefethen & Bau (1997), *Numerical Linear Algebra*, SIAM, lectures 6–11 (projectors, QR, Gram–Schmidt, Householder, least squares) and 18–19 (conditioning and stability of least squares). [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
- Björck (1967), "Solving linear least squares problems by Gram–Schmidt orthogonalization", *BIT Numerical Mathematics* 7: 1–21. the $\varepsilon\kappa$ analysis of modified GS behind experiment 1. [doi:10.1007/BF01934122](https://doi.org/10.1007/BF01934122)
- Björck & Paige (1992), "Loss and recapture of orthogonality in the modified Gram–Schmidt algorithm", *SIAM J. Matrix Anal. Appl.* 13(1): 176–190. why MGS least squares works despite its drifting $Q$. [doi:10.1137/0613015](https://doi.org/10.1137/0613015)
- Giraud, Langou, Rozložník & van den Eshof (2005), "Rounding error analysis of the classical Gram–Schmidt orthogonalization process", *Numerische Mathematik* 101: 87–100. the $\varepsilon\kappa^2$ bound for classical GS. [doi:10.1007/s00211-005-0615-4](https://doi.org/10.1007/s00211-005-0615-4)
- Householder (1958), "Unitary triangularization of a nonsymmetric matrix", *Journal of the ACM* 5(4): 339–342. the original reflections paper. [doi:10.1145/320941.320947](https://doi.org/10.1145/320941.320947)
- Golub & Van Loan (2013), *Matrix Computations*, 4th ed., Johns Hopkins University Press, chapter 5 (orthogonalization and least squares).
- Strang (2016), *Introduction to Linear Algebra*, 5th ed., Wellesley-Cambridge Press, chapter 4 (orthogonality, projections, least squares). gentler, with lots of pictures.

## connections

- ← builds on: span and bases (topic 01), triangular solves (topic 03).
- → used in: the QR algorithm for eigenvalues (topic 05), the SVD (topic 06), conditioning of least squares (topic 08), `LinearRegression(solver="qr")` (ch 08), Gauss–Newton (ch 06), orthogonal initialization of neural networks (ch 15).
