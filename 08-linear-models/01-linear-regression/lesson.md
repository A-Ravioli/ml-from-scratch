# linear regression

## why this matters

every model in this course gets compared with a linear one, and a surprising number of them lose. linear regression is fast, has no hyperparameters, comes with a closed-form answer and a century of statistical theory, and on many tabular problems it's within a few percent of anything fancier. when a deep network beats it by a mile, that tells you something about the data; when it doesn't, that tells you something too.

it's also where three themes of this course meet for the first time in one model. **geometry**: the fitted values are an orthogonal projection, so chapter 02's subspaces and QR decompositions are the whole story. **numerics**: the textbook formula $w = (X^\top X)^{-1}X^\top y$ is the one you should never compute, and you'll see it lose every digit on a problem QR solves to 8 or 9 digits. **probability**: least squares is exactly maximum likelihood under Gaussian noise, which is where the loss comes from and what the noise estimate means.

you'll build `LinearRegression` with three solvers (normal equations, QR, gradient descent) and an intercept done the numerically careful way, plus the likelihood view and the basis expansions that let a linear model fit curves. it's the first class in `mlfs.linear`, and it follows chapter 07's estimator protocol to the letter.

## warm-up: predict first

1. you fit a line to four points and get residuals $r_i$. what's $\sum_i r_i$, exactly, if the model has an intercept? what about without one?
2. $X$ has condition number $10^8$. how many correct digits do you expect from solving the normal equations in double precision (about 16 digits)? from QR?
3. you multiply one feature by 10. how do the least-squares coefficients change? how does the number of gradient-descent steps change?
4. you fit 15 parameters to 15 points. what's the training error? what do you expect on new points?
5. the noise has variance $\sigma^2$. is $\frac1n\sum_i r_i^2$ on the *training* data too big, too small, or right on average?

## prerequisites check

- [ ] you can solve a linear system with a Cholesky factorization and say when it exists (→ ch 02 · topic 03)
- [ ] you know what QR does and why least squares via QR is stable (→ ch 02 · topic 04)
- [ ] you know the condition number and what it does to rounding errors (→ ch 02 · topic 08)
- [ ] you can derive a gradient of a quadratic form (→ ch 02 · topic 07)
- [ ] you know gradient descent's step-size limit and its rate on a quadratic (→ ch 06 · topic 02)
- [ ] you know maximum likelihood estimation (→ ch 05 · topic 02)
- [ ] you've built `Regressor`, `check_X_y` and `PolynomialRegressor` (→ ch 07 · topics 01 and 04)

## core ideas

### idea 1: the model and the loss

a linear model predicts with a weighted sum of the features: $\hat y = x^\top w + b$. stack the $n$ examples as rows of $X \in \mathbb R^{n \times d}$, and the predictions are $\hat y = Xw + b\mathbf 1$. for now, fold the intercept into $X$ as a column of ones and write $\hat y = Xw$. (idea 3 does the intercept properly.)

which $w$? the classic answer measures misfit by the sum of squared residuals $r = Xw - y$. this course uses the *mean*, with a factor $\tfrac12$ that makes the gradient clean:

$$L(w) = \frac{1}{2n}\|Xw - y\|^2 = \frac{1}{2n}\sum_{i=1}^n (x_i^\top w - y_i)^2, \qquad \nabla L(w) = \frac1n X^\top (Xw - y).$$

the gradient follows from expanding $\|Xw - y\|^2 = w^\top X^\top X w - 2 w^\top X^\top y + y^\top y$ and differentiating each term (chapter 02, topic 07). the Hessian is $X^\top X / n$, which is positive semidefinite, so $L$ is a convex bowl: every point where the gradient vanishes is a global minimum.

the geometry is worth keeping in your head. $Xw$ ranges over the column space of $X$, a $d$-dimensional subspace of $\mathbb R^n$. the closest point in it to $y$ is the orthogonal projection, and "orthogonal" means the residual is perpendicular to every column: $X^\top (y - Xw) = 0$. that's the gradient set to zero. so warm-up 1: with a ones column, the residuals are orthogonal to $\mathbf 1$, which means they sum to exactly zero. without an intercept, nothing forces that.

> **▶ build: stage 1.** implement `mse_loss(w, X, y)` and `mse_gradient(w, X, y)`: two lines each, fully vectorized, and consistent with each other (the check compares your gradient with finite differences of your loss).

### idea 2: two exact solvers, and why one of them is a trap

setting the gradient to zero gives the **normal equations**

$$X^\top X\, w = X^\top y.$$

if $X$ has full column rank, $X^\top X$ is symmetric positive definite, so you can factor it as $LL^\top$ with your Cholesky and solve two triangular systems. this costs about $nd^2$ flops to form $X^\top X$ and $d^3/3$ to factor it, and it's what most people write. it has one problem: **conditioning**. the condition number of $X^\top X$ is the *square* of that of $X$:

$$\kappa(X^\top X) = \frac{\sigma_{\max}^2}{\sigma_{\min}^2} = \kappa(X)^2.$$

rounding errors in double precision are about $\epsilon \approx 10^{-16}$, and a backward-stable solver loses about $\log_{10}\kappa$ digits of that. so with $\kappa(X) = 10^8$, solving with $X^\top X$ (condition $10^{16}$) can leave you no correct digits at all, and Cholesky may even fail, because the rounded $X^\top X$ is no longer positive definite (warm-up 2).

**QR avoids squaring.** with the reduced factorization $X = QR$ ($Q$ with orthonormal columns, $R$ upper triangular), and because $Q$ preserves lengths on its range,

$$\|Xw - y\|^2 = \|Rw - Q^\top y\|^2 + \|(I - QQ^\top)y\|^2.$$

the second term doesn't depend on $w$, and the first is zero when $Rw = Q^\top y$, a triangular solve. this works with $X$ itself, and its error grows like $\kappa(X)\,\epsilon$ (plus a $\kappa^2$ term multiplied by the size of the residual, which is small when the model fits well). it costs about twice as much as the normal equations, which is nearly always a good trade. it's what your chapter 02 `lstsq` does, and what `np.linalg.lstsq` does with an SVD.

> **🔮 predict:** the Läuchli matrix $X = \begin{pmatrix}1 & 1\\ \varepsilon & 0\\ 0 & \varepsilon\end{pmatrix}$ with $\varepsilon = 10^{-8}$ has $\kappa(X) \approx 1.4 \times 10^8$. what is $X^\top X$ after rounding to double precision? what will Cholesky do with it?

> **▶ build: stage 2.** implement `solve_normal_equations(X, y)` with your `cholesky` and `cho_solve`, and `solve_qr(X, y)` with your `qr` and `solve_triangular`. don't catch the `LinAlgError` Cholesky raises on a matrix that isn't positive definite: it's information.

### idea 3: the intercept, done by centering

with an intercept, the model is $\hat y = Xw + b\mathbf 1$ and $b$ is *not* penalized or treated like the other weights (that matters from topic 02 on). you could add a column of ones and solve as before, but there's a better way. the derivative of $\|Xw + b\mathbf 1 - y\|^2$ with respect to $b$ is zero when

$$b = \bar y - \bar x^\top w,$$

where $\bar x$ is the vector of column means and $\bar y$ the mean target (on paper 2). substitute it back, and the problem becomes plain least squares on **centered** data:

$$\min_w \|(X - \mathbf 1\bar x^\top) w - (y - \bar y\mathbf 1)\|^2.$$

so: subtract the means, solve for $w$ with no ones column, and recover $b$ at the end. this is exactly equivalent, and numerically much better. imagine features measured near 10,000 (a year, a temperature in kelvin, a price): a ones column next to them is nearly parallel to them, $\kappa$ explodes, and the normal equations lose every digit. after centering, the columns are about as well conditioned as their spreads allow. scikit-learn does this, and so will your `Ridge` and `Lasso`.

> **▶ build: stage 3.** implement `LinearRegression(*, fit_intercept=True, solver="normal", lr=0.01, max_iters=1000, tol=1e-06)`: `fit` validates with `check_X_y`, centers when `fit_intercept`, dispatches to your stage 2 solvers (or to stage 4's `solve_gd` for `"gd"`), and sets `coef_` (shape `(d,)`) and `intercept_` (a float); `predict` is `X @ coef_ + intercept_`. `score` (R²) comes free from `Regressor`.

### idea 4: gradient descent, and the shape of the bowl

the exact solvers cost $O(nd^2 + d^3)$ and need the whole $X$ in memory. gradient descent costs $O(nd)$ per step and only needs products with $X$ and $X^\top$:

$$w_{k+1} = w_k - \eta\,\frac1n X^\top(Xw_k - y).$$

for linear regression it's rarely the best choice, but it's the algorithm that scales to everything else in this course, and least squares is where you can see exactly how it behaves. write $H = X^\top X/n$ with eigenvalues $\mu = \lambda_{\min}$ and $L = \lambda_{\max}$. the error $e_k = w_k - w^\star$ satisfies $e_{k+1} = (I - \eta H)e_k$ (on paper 3), so:

- it converges if and only if $\eta < 2/L$. with a bigger step, the component along the top eigenvector grows by a factor $|1 - \eta L| > 1$ every step, and the loss explodes.
- with $\eta = 1/L$, the slowest component shrinks by $1 - \mu/L = 1 - 1/\kappa$ per step, so reaching a tolerance $\varepsilon$ takes about $\kappa \log(1/\varepsilon)$ steps, where $\kappa = L/\mu$ is the condition number of $H$.

and $\kappa$ depends on your units. multiply one feature by 10 and its eigenvalue grows by 100: the bowl becomes a long narrow valley, and gradient descent zigzags across it (warm-up 3). the exact solvers don't care (the coefficient simply shrinks by 10), but gradient descent does, which is the conditioning argument behind chapter 07's scalers.

you've already built the loop: your chapter 06 `gradient_descent` takes $f$, its gradient, a start, `lr`, `max_iters` and `tol`, stops when $\|\nabla f\| \le$ `tol`, and stops early if the loss stops being finite. here you only need to hand it the right functions.

> **▶ build: stage 4.** implement `solve_gd(X, y, *, lr=0.01, max_iters=1000, tol=1e-06, w0=None)` on top of your `gradient_descent`, returning its `OptimResult`. `LinearRegression(solver="gd")` runs it on the centered data and stores the step count in `n_iter_`.

### idea 5: least squares is maximum likelihood

why squares, and not absolute values or fourth powers? one answer: assume the data came from a linear model plus Gaussian noise,

$$y_i = x_i^\top w + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal N(0, \sigma^2) \text{ independent}.$$

then the mean negative log-likelihood of the data is

$$\text{NLL}(w, \sigma^2) = \frac12\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\cdot\frac1n\sum_i (y_i - x_i^\top w)^2.$$

for any fixed $\sigma^2$, the $w$ that minimizes this minimizes the squared error: least squares *is* the maximum-likelihood estimate. the loss you chose in idea 1 is a modelling assumption in disguise, and other noise models give other losses: Laplace noise gives absolute errors (robust to outliers), Poisson counts give topic 06's GLMs. minimizing over $\sigma^2$ too gives

$$\hat\sigma^2_{\text{MLE}} = \frac{\text{RSS}}{n}, \qquad \text{RSS} = \sum_i (y_i - x_i^\top \hat w)^2.$$

this is biased low (warm-up 5). the fit chose $\hat w$ to make the training residuals small, so they're smaller than the true noise: on average $\text{RSS} = \sigma^2(n - p)$ for $p$ fitted parameters (on paper 4), and the unbiased estimate divides by $n - p$. that's the same $n - 1$ as in the sample variance (which fits one parameter, the mean), and it matters when $p$ is not small compared with $n$.

> **▶ build: stage 5.** implement `gaussian_nll(y, y_pred, sigma2)` (the mean NLL above) and `noise_variance(y, y_pred, n_params=0)` (RSS divided by $n - $ `n_params`).

### idea 6: basis expansions

"linear" means linear in the *parameters*, not in the inputs. replace $x$ by any fixed features $\phi(x) = (\phi_1(x), \dots, \phi_m(x))$ and fit $\hat y = \phi(x)^\top w + b$ with exactly the same machinery. two classic choices for a scalar input:

- **polynomials** $\phi(x) = (x, x^2, \dots, x^k)$: any smooth function is close to a polynomial on an interval, so this can fit nearly anything. it's also badly conditioned (the columns $x^9$ and $x^{10}$ on $[0, 1]$ are nearly parallel) and wild outside the data.
- **radial basis functions** $\phi_j(x) = \exp\left(-\frac{(x - c_j)^2}{2s^2}\right)$: bumps of width $s$ centered at points $c_j$. each feature only matters near its center, so the fit is local and better behaved.

chapter 07's `PolynomialFeatures` generalized the first to many inputs. both give you a dial (degree, number of bumps) that trades bias for variance, and turning it too far gives the textbook picture of overfitting: as many parameters as points means zero training error and nonsense in between (warm-up 4). there are two ways out: fewer features, or **regularization**, which is topic 02. chapter 09 takes the idea to its limit, with infinitely many features handled through a kernel.

> **🔮 predict:** with `fit_intercept=True`, what goes wrong if your polynomial basis also includes the constant column $x^0 = 1$? (center it.)

> **▶ build: stage 6.** implement `polynomial_basis(x, degree)` (columns $x, \dots, x^{\text{degree}}$, no constant) and `rbf_basis(x, centers, width)`, both by broadcasting.

## worked example

four points: $x = (0, 1, 2, 3)$, $y = (1, 3, 2, 5)$.

**centering.** $\bar x = 1.5$, $\bar y = 2.75$. centered: $x - \bar x = (-1.5, -0.5, 0.5, 1.5)$, $y - \bar y = (-1.75, 0.25, -0.75, 2.25)$.

**slope.** with one feature the normal equation is a single division: $w = \frac{\sum (x_i - \bar x)(y_i - \bar y)}{\sum (x_i - \bar x)^2} = \frac{2.625 - 0.125 - 0.375 + 3.375}{2.25 + 0.25 + 0.25 + 2.25} = \frac{5.5}{5} = 1.1$.

**intercept.** $b = \bar y - \bar x w = 2.75 - 1.65 = 1.1$. the line is $\hat y = 1.1 + 1.1x$.

**residuals.** predictions $(1.1, 2.2, 3.3, 4.4)$, residuals $y - \hat y = (-0.1, 0.8, -1.3, 0.6)$. they sum to zero (idea 1). $\text{RSS} = 0.01 + 0.64 + 1.69 + 0.36 = 2.7$, and $\text{TSS} = \sum(y - \bar y)^2 = 8.75$, so $R^2 = 1 - 2.7/8.75 \approx 0.691$.

**the loss and its gradient** with a ones column, $A = [\mathbf 1, x]$: at $w = (0, 0)$ the residuals are $-y$, so $L = \frac{1 + 9 + 4 + 25}{2 \cdot 4} = 4.875$ and $\nabla L = -\frac14 A^\top y = -\frac14(11, 22) = (-2.75, -5.5)$. at $w = (1.1, 1.1)$, $L = 2.7/8 = 0.3375$ and the gradient is exactly zero.

**noise.** $\hat\sigma^2_{\text{MLE}} = 2.7/4 = 0.675$; with $p = 2$ fitted parameters, the unbiased estimate is $2.7/2 = 1.35$. at $\sigma^2 = 0.675$ the mean NLL is $\frac12\log(2\pi \cdot 0.675) + \frac12$.

the tests check every number here.

## on paper

1. **the normal equations.** show that $\nabla L(w) = 0$ is equivalent to $X^\top X w = X^\top y$, that any solution minimizes $L$, and that the solution is unique if and only if $X$ has full column rank.

<details><summary>solution</summary>

$\nabla L = \frac1n X^\top(Xw - y)$, which is zero exactly when $X^\top X w = X^\top y$. for any $w$ and a solution $w^\star$, write $Xw - y = X(w - w^\star) + (Xw^\star - y)$; the cross term $2(w - w^\star)^\top X^\top(Xw^\star - y)$ vanishes by the normal equations, so $\|Xw - y\|^2 = \|X(w - w^\star)\|^2 + \|Xw^\star - y\|^2 \ge \|Xw^\star - y\|^2$. equality holds iff $X(w - w^\star) = 0$, so the minimizer is unique iff the null space of $X$ is $\{0\}$, i.e. full column rank, which is also when $X^\top X$ is invertible.

</details>

2. **the intercept by centering.** minimize $\|Xw + b\mathbf 1 - y\|^2$ over $b$ for fixed $w$, and show that what's left is least squares on centered data.

<details><summary>solution</summary>

the derivative in $b$ is $2\mathbf 1^\top(Xw + b\mathbf 1 - y) = 2n(\bar x^\top w + b - \bar y)$, zero at $b = \bar y - \bar x^\top w$ (a minimum: the second derivative is $2n > 0$). substituting, $Xw + b\mathbf 1 - y = (X - \mathbf 1\bar x^\top)w - (y - \bar y\mathbf 1)$. so $w$ minimizes the centered problem, and $b$ follows from it.

</details>

3. **gradient descent on a quadratic.** show that the error $e_k = w_k - w^\star$ of gradient descent on $L$ satisfies $e_{k+1} = (I - \eta H)e_k$ with $H = X^\top X/n$. for which $\eta$ does it converge from every start? what is the contraction factor per step for $\eta = 1/L$?

<details><summary>solution</summary>

$\nabla L(w_k) = H w_k - \frac1n X^\top y = H(w_k - w^\star)$, using $H w^\star = \frac1n X^\top y$. so $e_{k+1} = e_k - \eta H e_k$. in $H$'s eigenbasis, each component is multiplied by $1 - \eta\lambda_i$ per step, so convergence for every start needs $|1 - \eta\lambda_i| < 1$ for all $i$ with $\lambda_i > 0$, i.e. $0 < \eta < 2/L$. (components with $\lambda_i = 0$ never move: rank-deficient problems converge to the minimizer closest to $w_0$.) with $\eta = 1/L$ the factors are $1 - \lambda_i/L \in [0, 1 - 1/\kappa]$, so the error shrinks by at least $1 - 1/\kappa$ per step.

</details>

4. **the residual sum of squares, on average.** with $y = Xw + \varepsilon$, $\varepsilon \sim \mathcal N(0, \sigma^2 I)$, and $X$ of full column rank $p$, show that $E[\text{RSS}] = \sigma^2(n - p)$. (hint: the fitted values are $Hy$ with the "hat matrix" $H = X(X^\top X)^{-1}X^\top$, a projection.)

<details><summary>solution</summary>

the residual is $(I - H)y = (I - H)\varepsilon$, since $(I - H)Xw = 0$. so $\text{RSS} = \varepsilon^\top(I - H)^\top(I - H)\varepsilon = \varepsilon^\top(I - H)\varepsilon$ (a projection is symmetric and idempotent). $E[\varepsilon^\top A\varepsilon] = \sigma^2\operatorname{tr}A$, and $\operatorname{tr}(I - H) = n - \operatorname{tr}((X^\top X)^{-1}X^\top X) = n - p$. so $E[\text{RSS}/(n - p)] = \sigma^2$.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. polynomial fits of every degree from 1 to 14 on 15 noisy points, with training and test error.
2. the normal equations against QR on increasingly ill-conditioned polynomial design matrices, with a known exact answer.
3. gradient descent's step count as one feature is rescaled, before and after standardizing.

## check your understanding

1. your training residuals sum to 0.03, not 0, and the model has an intercept. what's the most likely bug?

<details><summary>answer</summary>

the intercept isn't really the least-squares one: for example, it was computed from means taken *after* something changed the data, or `predict` uses a different `x_mean` than `fit`, or the solve was inaccurate (normal equations with a ones column next to large features). with an exact least-squares intercept, the residuals are orthogonal to $\mathbf 1$ up to rounding.

</details>

2. why doesn't `LinearRegression` divide by $n$ in the exact solvers, when the loss is the mean?

<details><summary>answer</summary>

scaling the loss by a positive constant doesn't move its minimizer: $X^\top X w = X^\top y$ and $\frac1n X^\top X w = \frac1n X^\top y$ have the same solutions. the $1/n$ matters for gradient descent, where it sets the scale of the gradient and therefore which step sizes work, independently of $n$.

</details>

3. a colleague computes `np.linalg.inv(X.T @ X) @ X.T @ y`. name two things wrong with it.

<details><summary>answer</summary>

it forms $X^\top X$, squaring the condition number, and it computes an explicit inverse, which is slower and less accurate than solving the system. use QR (or `np.linalg.lstsq`); if you must use the normal equations, factor with Cholesky and solve.

</details>

4. you double every target value. what happens to `coef_`, `intercept_` and $R^2$? you double one feature?

<details><summary>answer</summary>

doubling $y$ doubles both `coef_` and `intercept_` (the solution is linear in $y$), and $R^2$ doesn't change (RSS and TSS both scale by 4). doubling feature $j$ halves `coef_[j]` and changes nothing else: predictions and $R^2$ are identical. least squares is equivariant to feature scaling, which is exactly what regularization (topic 02) and gradient descent are *not*.

</details>

5. gradient descent with `lr=0.01` hasn't converged after 1,000 steps. list three things that could be going on.

<details><summary>answer</summary>

the step is far below $1/L$ (each step makes tiny progress), the problem is badly conditioned (unscaled features, $\kappa$ huge, so $\kappa\log(1/\varepsilon)$ steps are needed), or `tol` is unreachable relative to the gradient's scale. diagnose with the loss history and the eigenvalues of $X^\top X/n$: set $\eta = 1/L$, standardize, or use an exact solver.

</details>

6. the Gaussian-noise model is clearly wrong for your data (the noise is skewed and heavier-tailed). is the least-squares fit still useful?

<details><summary>answer</summary>

often, yes. least squares estimates the best linear approximation to $E[y \mid x]$ whatever the noise distribution, and it's unbiased whenever the linear model for the mean is right. what breaks is what relies on Gaussianity: the likelihood, exact confidence intervals for small $n$, and efficiency (with heavy tails a few points dominate the squared loss, and a robust loss can do much better).

</details>

7. why does the degree-14 polynomial in the lab do worst near the ends of the interval?

<details><summary>answer</summary>

high-degree polynomials grow fastest where $|x|$ is largest, and near the ends there are fewer data points on one side to pin the curve down. the interpolant oscillates between and beyond the outermost points (Runge's phenomenon), so its variance concentrates at the edges (chapter 07, topic 04 measured the same thing).

</details>

## common bugs

- **a gradient off by a factor of 2.** a loss of $\frac1n\|r\|^2$ with a gradient of $\frac1n X^\top r$ (or the reverse). symptom: the finite-difference check fails by exactly a factor of 2, and gradient descent with $\eta = 1/L$ converges at the wrong speed.
- **intercept forgotten in `predict`.** `fit` centered the data, `predict` returns `X @ coef_`. symptom: predictions offset by a constant, $R^2$ much worse on the test set than the training fit suggests.
- **forming $X^\top X$ inside "QR".** calling `qr(X.T @ X)`, or `lstsq(X.T @ X, X.T @ y)`. symptom: the Läuchli check fails; results match the normal equations' errors exactly.
- **a constant column plus an intercept.** a basis that includes $x^0 = 1$ becomes a column of zeros after centering, so $X^\top X$ is singular. symptom: `LinAlgError` from Cholesky, and from QR too (its $R$ has an exact zero on the diagonal, so the triangular solve fails); if rounding leaves the column tiny but nonzero, you get huge coefficients instead.
- **a step size that's too big.** `lr` above $2/L$, often because the features weren't scaled. symptom: the loss grows every step and becomes `inf` or `nan` within a few dozen iterations.
- **centering $X$ but not $y$** (or the reverse). symptom: the intercept is wrong by $\bar y$ (or by $\bar x^\top w$), while the slopes look right.

## going deeper

- Trefethen & Bau (1997), *Numerical Linear Algebra*, SIAM, lectures 11 and 18–19: least squares by normal equations, QR and SVD, and their conditioning, including the $\kappa^2\cdot$residual term. [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, sections 3.1–3.2: least squares, its geometry and its sampling distribution. [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, sections 3.1–3.2: basis functions, maximum likelihood and least squares. [microsoft.com/en-us/research/people/cmbishop/prml-book](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, chapter 20: the stability of least-squares solvers, in detail. [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
- Stigler (1981), "Gauss and the invention of least squares", *The Annals of Statistics* 9(3): 465–474. how least squares and the normal distribution were born together. [doi:10.1214/aos/1176345451](https://doi.org/10.1214/aos/1176345451)

## connections

- ← builds on: Cholesky, QR and conditioning (ch 02), gradients of quadratic forms (ch 02 · topic 07), gradient descent and its rate (ch 06), maximum likelihood (ch 05), the estimator protocol and polynomial features (ch 07).
- → used in: ridge, lasso and elastic net (topic 02, which reuse the centering trick), logistic regression's IRLS, which solves a weighted least-squares problem every step (topic 04), GLMs (topic 06), Bayesian linear regression (ch 12), kernel ridge regression and Gaussian processes (ch 09), linear probes on learned representations (ch 25), and every "is the fancy model better than a linear one?" baseline from here on.
