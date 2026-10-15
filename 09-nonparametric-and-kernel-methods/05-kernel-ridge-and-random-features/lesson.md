# kernel ridge & random features

## why this matters

topic 03 ended with interpolants that passed through every noisy point and paid for it with wild swings. the cure is the same as in chapter 08's ridge regression: stop insisting on a perfect fit and penalize complexity. in an RKHS, "complexity" is the norm, and minimizing squared error plus $\alpha\|f\|^2_{\mathcal H}$ has a closed-form solution: **kernel ridge regression** (KRR). it's three lines of code, it's the posterior mean of a Gaussian process (topic 06), and it's what an infinitely wide neural network computes when trained by gradient descent (the neural tangent kernel, chapter 19).

it also has a scaling problem that becomes the main topic here. KRR solves an $n \times n$ linear system: $O(n^2)$ memory and $O(n^3)$ time. 10,000 points is a second; 100,000 points is 80 GB of Gram matrix and an hour of factorization; a million is out of the question. the fix is to go *back* to explicit features, but cleverly: find a modest number $D$ of features whose inner products approximate the kernel, then run ordinary linear ridge on them in $O(nD^2)$. two classic ways to do it: **random Fourier features**, which sample the kernel's frequency spectrum (Rahimi & Recht, 2007), and the **Nyström method**, which builds features from a random subset of the data (Williams & Seeger, 2001).

## warm-up: predict first

1. kernel ridge with the linear kernel: what model is that?
2. as $\alpha \to 0$, what does KRR do on noisy data? as $\alpha \to \infty$?
3. the RBF kernel is an inner product of infinitely many features. can a *random* finite set of features approximate it? how should the error shrink with the number of features $D$?
4. Nyström uses $m$ landmark points from the training set. what is its approximation of $K$ on the landmarks themselves?
5. with a budget of 100 features, which do you expect to approximate a Gram matrix better: 100 random frequencies or 100 random data points?

## prerequisites check

- [ ] you know ridge regression's closed form $(X^\top X + \alpha I)^{-1}X^\top y$ and why it's always solvable for $\alpha > 0$ (→ ch 02 · topic 04; ch 08 · topic 02)
- [ ] you can build Gram matrices with `get_kernel`, and know the representer theorem (→ topic 03)
- [ ] you can compute $A^{-1/2}$ for a symmetric PSD matrix from its eigendecomposition (→ ch 02 · topic 05)
- [ ] you know Monte Carlo estimates and their $1/\sqrt n$ error (→ ch 01 · topic 05; ch 04)
- [ ] you've used the `Transformer` protocol: `fit`, `transform`, `fit_transform` (→ ch 07 · topic 05)

## core ideas

### idea 1: kernel ridge regression

in feature space, ridge regression minimizes $\sum_i (y_i - w^\top\varphi(x_i))^2 + \alpha\|w\|^2$, with solution $w = (\Phi^\top\Phi + \alpha I)^{-1}\Phi^\top y$, where $\Phi$ is $(n, D)$. for a huge (or infinite) $D$ that's useless, but the **push-through identity**

$$(\Phi^\top\Phi + \alpha I_D)^{-1}\Phi^\top = \Phi^\top(\Phi\Phi^\top + \alpha I_n)^{-1}$$

(on paper 1) rewrites it as $w = \Phi^\top\beta$ with $\beta = (K + \alpha I)^{-1}y$, where $K = \Phi\Phi^\top$ is the Gram matrix. predictions need only kernel values:

$$\hat f(x) = w^\top\varphi(x) = \sum_{i=1}^n \beta_i\, k(x_i, x), \qquad \beta = (K + \alpha I)^{-1} y.$$

it's the representer theorem of topic 03 applied to the squared loss: the minimizer of $\sum_i (y_i - f(x_i))^2 + \alpha\|f\|^2_{\mathcal H}$ is $\sum_i\beta_i k(x_i, \cdot)$, and setting the gradient in $\beta$ to zero gives the same system. with the linear kernel, it's exactly ridge regression (warm-up 1).

$\alpha$ is the regularization strength (warm-up 2). as $\alpha \to 0$, $\beta \to K^{-1}y$, the interpolant of topic 03 with all its noise. as $\alpha \to \infty$, $\beta \to y/\alpha \to 0$, and every prediction shrinks to zero. this model has **no intercept**: with the RBF kernel, predictions far from the data fall back to 0, so centre $y$ first if its mean isn't near 0. between the extremes, a useful measure of complexity is the **effective degrees of freedom**, $\mathrm{df}(\alpha) = \mathrm{tr}\big(K(K + \alpha I)^{-1}\big) = \sum_k \frac{\lambda_k}{\lambda_k + \alpha}$ over the eigenvalues $\lambda_k$ of $K$: each eigendirection is kept ($\lambda_k \gg \alpha$) or shrunk away ($\lambda_k \ll \alpha$).

the cost: forming $K$ is $O(n^2 d)$, storing it is $O(n^2)$, and solving is $O(n^3)$ (a Cholesky factorization, since $K + \alpha I$ is symmetric positive definite). predicting costs $O(n d)$ per point, because every training point is in the expansion: unlike the SVM, KRR isn't sparse.

> **▶ build: stage 1.** implement `KernelRidge.fit` (`kernel_` from `get_kernel`, `X_fit_`, and `dual_coef_` from `np.linalg.solve(K + alpha * I, y)`; `alpha < 0` raises `ValueError`) and `predict`.

### idea 2: random Fourier features

the RBF kernel depends only on the difference $\delta = x - y$. **Bochner's theorem** says a continuous, shift-invariant kernel is PSD exactly when it's the Fourier transform of a nonnegative measure, which you can normalize to a probability distribution $p(\omega)$:

$$k(x - y) = \int p(\omega)\, e^{i\omega^\top(x - y)}\,d\omega = \mathbb E_{\omega \sim p}\big[\cos\big(\omega^\top(x - y)\big)\big].$$

for $k(\delta) = e^{-\gamma\|\delta\|^2}$, $p$ is a Gaussian, $\omega \sim \mathcal N(0, 2\gamma I)$ (on paper 2). so the kernel is an *expectation*, and you can estimate it by Monte Carlo: draw $D$ frequencies and average. to turn this into features of $x$ and $y$ separately, add a random phase $b \sim \mathcal U[0, 2\pi)$ and use $2\cos(a)\cos(c) = \cos(a - c) + \cos(a + c)$: the second term averages to zero over $b$, so

$$z(x) = \sqrt{\tfrac{2}{D}}\big(\cos(\omega_1^\top x + b_1), \dots, \cos(\omega_D^\top x + b_D)\big), \qquad \mathbb E\big[z(x)^\top z(y)\big] = k(x, y).$$

$z(x)^\top z(y)$ is an unbiased average of $D$ independent bounded terms, so its error shrinks like $1/\sqrt D$ (warm-up 3): 100 times more features for one more decimal digit. that's the price of Monte Carlo, and it's independent of the data. with $D$ features, linear ridge costs $O(nD^2 + D^3)$ to fit and $O(D d)$ per prediction, which beats $O(n^3)$ as soon as $D \ll n$.

> **▶ build: stage 2.** implement `RandomFourierFeatures.fit` (with one generator `as_rng(rng)`: `random_weights_` of shape $(d, D)$ from $\mathcal N(0, 2\gamma)$, i.e. `sqrt(2 * gamma) * standard_normal((d, D))`, and `random_offset_` of shape $(D,)$ uniform on $[0, 2\pi)$) and `transform`.

> **🔮 predict:** the RFF approximation error with $D = 100$ is 0.2. what will it be with $D = 1{,}600$?

### idea 3: the Nyström method

random Fourier features ignore the data. the **Nyström method** uses it. pick $m$ landmark points $L$ from the training set at random, and compute just two blocks of the kernel matrix: $W = K_{LL}$, $(m, m)$, among the landmarks, and $C = K_{nL}$, $(n, m)$, between all points and the landmarks. the approximation is

$$K \approx C\,W^{-1}C^\top.$$

it's exact on the landmarks (warm-up 4): the $LL$ block of $CW^{-1}C^\top$ is $WW^{-1}W = W$, and the $nL$ block is $C$. elsewhere, it's the kernel of the projection of each point's feature vector onto the span of the landmarks' feature vectors (on paper 3), so it's accurate when the landmarks span the directions that matter. to use it as features, factor $W^{-1} = W^{-1/2}W^{-1/2}$:

$$z(x) = K(x, L)\,W^{-1/2}, \qquad z(x)^\top z(y) = K(x, L)W^{-1}K(L, y).$$

compute $W^{-1/2}$ from the eigendecomposition $W = U\,\mathrm{diag}(w)\,U^\top$ as $U\,\mathrm{diag}(1/\sqrt{w})\,U^\top$, with tiny eigenvalues clipped (at $10^{-12}$) so that near-duplicate landmarks don't blow it up. any kernel works, not just shift-invariant ones.

because it adapts to the data, Nyström is usually far more accurate than random features with the same $D = m$, especially when the kernel matrix's eigenvalues decay fast, which is typical: the error of the best rank-$m$ approximation is set by the eigenvalues beyond the $m$-th, and random landmarks capture the large eigendirections well, though they fall further and further behind the optimum as $m$ grows into the tail of the spectrum (warm-up 5). the lab measures both gaps.

> **▶ build: stage 3.** implement `Nystroem.fit` (`kernel_` via `get_kernel`; `m = min(n_components, n)` landmark indices drawn with `as_rng(rng).choice(n, size=m, replace=False)` into `component_indices_`, the rows into `components_`, and `normalization_` $= W^{-1/2}$) and `transform`.

### idea 4: linear ridge on approximate features

with any feature map $Z = z(X)$ of shape $(n, D)$, the approximate model is plain ridge regression,

$$w = (Z^\top Z + \alpha I_D)^{-1} Z^\top y, \qquad \hat f(x) = z(x)^\top w,$$

a $D \times D$ system. by the push-through identity it's the same as kernel ridge with the *approximate* kernel $ZZ^\top$, so its quality is controlled by how well $ZZ^\top$ approximates $K$. a convenient summary is the relative Frobenius error $\|ZZ^\top - K\|_F / \|K\|_F$. when the Nyström landmarks are *all* the training points, $ZZ^\top = K$ exactly, and ridge on those features reproduces kernel ridge regression to round-off, on new points too (on paper 1 again).

the trade-off is between $D$ and accuracy:

| method | fit | predict (per point) | memory | error |
|---|---|---|---|---|
| exact KRR | $O(n^3)$ | $O(nd)$ | $O(n^2)$ | none |
| random Fourier features | $O(nD^2 + D^3)$ | $O(Dd)$ | $O(nD)$ | $\sim 1/\sqrt D$, data-independent |
| Nyström, $m = D$ | $O(nm^2 + m^3)$ | $O(md)$ | $O(nm)$ | set by the spectrum of $K$ |

in practice the generalization error saturates long before $ZZ^\top$ is exact: the statistical error of learning from $n$ noisy points is already of order $1/\sqrt n$, and a kernel approximation better than that doesn't help. results by Rudi & Rosasco (2017) show that $D$ of order $\sqrt n \log n$ features can be enough for the same learning rate as exact KRR.

> **🔮 predict:** if Nyström features approximate the Gram matrix ten times better than random features with the same $D$, how much better will the resulting ridge model predict new data?

> **▶ build: stage 4.** implement `primal_ridge(Z, y, alpha)` and `gram_error(Z, K)`. the checks confirm that ridge on all-points Nyström features equals your `KernelRidge`, and that ridge on many random features approaches it.

## worked example

kernel ridge on two points, RBF kernel with $\gamma = 1$, data $x = (0, 1)$, $y = (1, 2)$, $\alpha = 0.5$:

$$K + \alpha I = \begin{pmatrix}1.5 & e^{-1}\\ e^{-1} & 1.5\end{pmatrix}, \qquad \det = 2.25 - e^{-2} = 2.114665.$$

$$\beta = \frac{1}{2.114665}\begin{pmatrix}1.5 \cdot 1 - e^{-1}\cdot 2\\ 1.5 \cdot 2 - e^{-1}\cdot 1\end{pmatrix} = \begin{pmatrix}0.361400\\ 1.244702\end{pmatrix}.$$

the fitted values are $f(0) = \beta_1 + e^{-1}\beta_2 = 0.819303$ and $f(1) = e^{-1}\beta_1 + \beta_2 = 1.377654$: both pulled towards 0 compared with the targets 1 and 2 (topic 03's interpolant, with $\alpha = 0$, hit them exactly). the residuals are $y - f = (0.181, 0.622)$, and they equal $\alpha\beta$, which is the defining equation $(K + \alpha I)\beta = y$ rearranged. the tests check $\beta$ and the fitted values.

## on paper

1. **the push-through identity.** show that $(\Phi^\top\Phi + \alpha I)^{-1}\Phi^\top = \Phi^\top(\Phi\Phi^\top + \alpha I)^{-1}$ for $\alpha > 0$. use it to show that ridge on Nyström features with all $n$ points as landmarks equals kernel ridge, at any new point.

<details><summary>solution</summary>

$\Phi^\top(\Phi\Phi^\top + \alpha I) = (\Phi^\top\Phi + \alpha I)\Phi^\top$ (both are $\Phi^\top\Phi\Phi^\top + \alpha\Phi^\top$). multiply on the left by $(\Phi^\top\Phi + \alpha I)^{-1}$ and on the right by $(\Phi\Phi^\top + \alpha I)^{-1}$; both inverses exist for $\alpha > 0$. with all points as landmarks, $Z = K K^{-1/2} = K^{1/2}$ and $z(x) = k(x)^\top K^{-1/2}$ with $k(x) = K(X, x)$. then $\hat f(x) = z(x)^\top(Z^\top Z + \alpha I)^{-1}Z^\top y = k(x)^\top K^{-1/2}(K + \alpha I)^{-1}K^{1/2}y = k(x)^\top(K + \alpha I)^{-1}y$, since $K^{\pm1/2}$ and $(K + \alpha I)^{-1}$ share eigenvectors and commute. that's the KRR prediction.

</details>

2. **the spectrum of the RBF kernel.** show that $\mathbb E_{\omega \sim \mathcal N(0, \sigma^2 I)}[\cos(\omega^\top\delta)] = e^{-\sigma^2\|\delta\|^2/2}$, and conclude that $\sigma^2 = 2\gamma$ gives $e^{-\gamma\|\delta\|^2}$. then show $\mathbb E_b[2\cos(a + b)\cos(c + b)] = \cos(a - c)$ for $b \sim \mathcal U[0, 2\pi)$.

<details><summary>solution</summary>

$\omega^\top\delta \sim \mathcal N(0, \sigma^2\|\delta\|^2)$, and for $u \sim \mathcal N(0, s^2)$ the characteristic function gives $\mathbb E[e^{iu}] = e^{-s^2/2}$; its real part is $\mathbb E\cos u$, and the imaginary part vanishes by symmetry. so the expectation is $e^{-\sigma^2\|\delta\|^2/2}$, which is $e^{-\gamma\|\delta\|^2}$ when $\sigma^2 = 2\gamma$. for the phase: $2\cos(a + b)\cos(c + b) = \cos(a - c) + \cos(a + c + 2b)$, and the second term averages to 0 over a full period of $b$.

</details>

3. **Nyström as a projection.** let $P$ be the orthogonal projection in feature space onto $\mathrm{span}\{\varphi(\ell) : \ell \in L\}$. show that $\langle P\varphi(x), P\varphi(y)\rangle = K(x, L)\,W^{-1}K(L, y)$ (assume $W$ invertible).

<details><summary>solution</summary>

with $\Phi_L$ the $(m, \cdot)$ matrix of landmark features, $P = \Phi_L^\top(\Phi_L\Phi_L^\top)^{-1}\Phi_L = \Phi_L^\top W^{-1}\Phi_L$. then $\langle P\varphi(x), P\varphi(y)\rangle = \varphi(x)^\top P\varphi(y)$ (as $P$ is symmetric and idempotent) $= (\Phi_L\varphi(x))^\top W^{-1}(\Phi_L\varphi(y)) = K(L, x)^\top W^{-1}K(L, y)$. the Nyström kernel is the true kernel after projecting every point onto what the landmarks can represent; the error $K - CW^{-1}C^\top$ is the Gram matrix of the residuals $(I - P)\varphi$, which is PSD, so Nyström never overestimates $K_{ii}$.

</details>

4. **degrees of freedom.** show that $\mathrm{df}(\alpha) = \mathrm{tr}\big(K(K + \alpha I)^{-1}\big)$ equals $\sum_k \lambda_k/(\lambda_k + \alpha)$, and evaluate its limits as $\alpha \to 0$ and $\alpha \to \infty$.

<details><summary>solution</summary>

with $K = U\Lambda U^\top$, $K(K + \alpha I)^{-1} = U\Lambda(\Lambda + \alpha I)^{-1}U^\top$, whose trace is $\sum_k\lambda_k/(\lambda_k + \alpha)$. as $\alpha \to 0$, each positive eigenvalue contributes 1, so $\mathrm{df} \to \mathrm{rank}(K)$ ($n$ for the RBF kernel on distinct points: interpolation). as $\alpha \to \infty$, $\mathrm{df} \to 0$. $K(K + \alpha I)^{-1}$ is the "hat matrix" mapping $y$ to the fitted values, as in linear regression.

</details>

## experiment

`lab.py` uses your `KernelRidge`, `RandomFourierFeatures`, `Nystroem` and `primal_ridge`:

1. how well do random features and Nyström approximate a real Gram matrix (California housing), as $D$ grows?
2. kernel ridge on 4,000 California districts against ridge on $D$ approximate features: test error and fit time.
3. kernel ridge on noisy 1-D data, from interpolation to over-smoothing, with the effective degrees of freedom.

## check your understanding

1. KRR and the SVM both use kernels. why is the SVM's solution sparse and KRR's dense?

<details><summary>answer</summary>

the squared loss penalizes every residual, however small, so every training point gets a nonzero coefficient $\beta_i = (y_i - f(x_i))/\alpha$. the hinge loss is exactly zero beyond the margin, so most $\alpha_i$ are zero.

</details>

2. why does a KRR model with the RBF kernel predict about 0 far from the data, and what should you do about it?

<details><summary>answer</summary>

every term $\beta_i k(x_i, x)$ decays to 0 as $x$ moves away from all training points, and there's no intercept. centre the targets (subtract their training mean, add it back to predictions), or add a constant to the kernel, which acts like an unpenalized-ish intercept.

</details>

3. both approximations replace $K$ by $ZZ^\top$. which can be used with a polynomial kernel?

<details><summary>answer</summary>

Nyström works with any kernel: it only needs kernel evaluations. random Fourier features need a shift-invariant kernel $k(x - y)$ and its spectral distribution; the polynomial kernel isn't shift-invariant (other random-feature constructions exist for it, e.g. random Maclaurin features).

</details>

4. you double $D$ for random Fourier features. how much does the Gram approximation error shrink? and the test error of the ridge model?

<details><summary>answer</summary>

the Gram error by about $1/\sqrt2$ (Monte Carlo). the test error by less, and eventually not at all: once the kernel approximation error is below the statistical error of learning from $n$ noisy points, more features buy nothing.

</details>

5. what goes wrong in `Nystroem.fit` if two landmarks are identical, and how does clipping the eigenvalues help?

<details><summary>answer</summary>

$W$ has two identical rows, so it's singular, and $W^{-1/2}$ has an infinite eigenvalue. clipping eigenvalues at $10^{-12}$ makes the computation finite; the corresponding direction then carries almost no information (it's a pseudo-inverse in effect). without clipping you get `inf` or `nan` features.

</details>

6. KRR with $n = 50{,}000$: what's the bottleneck, and what would you do?

<details><summary>answer</summary>

the Gram matrix alone is $50{,}000^2 \times 8$ bytes = 20 GB, and the solve is $\sim 4 \times 10^{13}$ flops. use Nyström or random features with a few thousand components ($O(nD^2)$), or iterative solvers (conjugate gradients with fast kernel-vector products) that never store $K$.

</details>

## common bugs

- **adding $\alpha$ to every entry instead of the diagonal.** `K + alpha` instead of `K + alpha * np.eye(n)`. symptom: no regularization effect (or a singular system), and sklearn comparisons fail.
- **inverting instead of solving.** `np.linalg.inv(K + alpha I) @ y` is slower and less accurate. solve.
- **wrong frequency scale.** `standard_normal` frequencies without the $\sqrt{2\gamma}$, or with $\gamma$ instead of $2\gamma$. symptom: the features approximate an RBF kernel with the wrong width; errors don't shrink with $D$.
- **forgetting the phase or the $\sqrt{2/D}$.** without $b$, $z(x)^\top z(y)$ estimates $\frac12[\cos(\cdot - \cdot) + \cos(\cdot + \cdot)]$-type terms that don't average out; without the scale, everything is off by a constant.
- **redrawing random features in `transform`.** the features must be fixed at `fit`: train and test points need the *same* $\omega$ and $b$. symptom: test predictions are noise.
- **Nyström landmarks with replacement.** duplicates make $W$ singular. sample without replacement.

## going deeper

- Rahimi & Recht (2007), "Random features for large-scale kernel machines", *NeurIPS*. [papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines)
- Williams & Seeger (2001), "Using the Nyström method to speed up kernel machines", *NeurIPS*. [papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines](https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines)
- Yang, Li, Mahdavi, Jin & Zhou (2012), "Nyström method vs random Fourier features: a theoretical and empirical comparison", *NeurIPS*. why data-dependent features win when the spectrum decays. [papers.nips.cc/paper/4588](https://papers.nips.cc/paper/4588-nystrom-method-vs-random-fourier-features-a-theoretical-and-empirical-comparison)
- Rudi & Rosasco (2017), "Generalization properties of learning with random features", *NeurIPS*. how many features are enough. [arXiv:1602.04474](https://arxiv.org/abs/1602.04474)
- Saunders, Gammerman & Vovk (1998), "Ridge regression learning algorithm in dual variables", *ICML*. kernel ridge regression.
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., sections 5.8 and 12.3.7 (regularization in an RKHS, effective degrees of freedom). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)

## connections

- ← builds on: ridge regression and the pseudoinverse (ch 02, ch 08), eigendecompositions and matrix square roots (ch 02), Monte Carlo error (ch 01, ch 04), kernels, Gram matrices and the representer theorem (topic 03), the `Transformer` protocol (ch 07).
- → used in: Gaussian processes, whose posterior mean is exactly KRR with $\alpha$ = the noise variance (topic 06); the Bayesian-optimization capstone; kernel PCA with Nyström (ch 11); the neural tangent kernel, where a wide network trains like KRR (ch 19); random features as a model of two-layer networks (ch 19).
