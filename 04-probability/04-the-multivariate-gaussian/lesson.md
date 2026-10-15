# the multivariate gaussian

## why this matters

one distribution shows up in more of machine learning than any other. measurement noise is modeled as Gaussian, so least-squares regression is maximum likelihood under it (ch 08). Gaussian processes put a Gaussian over entire functions (ch 09). Kalman filters track a moving object by updating a Gaussian belief every time a measurement arrives. variational autoencoders and diffusion models push data through Gaussian noise and learn to pull it back out (ch 21–22). Gaussian mixtures cluster data (ch 11).

the reason is not that the world is Gaussian. it's that the Gaussian is the one distribution where **every operation you want stays inside the family and reduces to linear algebra**. apply a linear map: still Gaussian. forget some coordinates: still Gaussian, and you just delete rows and columns. observe some coordinates: still Gaussian, with a mean and covariance given by a formula you can write on one line. sums of independent effects: Gaussian (that's topic 05). no other multivariate family is this cooperative.

the price is that you have to do the linear algebra carefully. textbooks write $\Sigma^{-1}$ and $|\Sigma|$ freely. compute them literally and you'll get densities of `inf`, `nan`, or wrong answers as soon as $d$ is more than a handful or the covariance is a bit ill-conditioned. in this topic you'll build `MultivariateNormal` the way numerical libraries do: everything through one Cholesky factor, built with *your* `mlfs.linalg.cholesky`.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. you can sample independent standard normals. how do you sample a 2-D Gaussian whose coordinates have correlation 0.8?
2. $(X_1, X_2)$ is jointly Gaussian. is $X_1$ on its own Gaussian? is $X_1 + X_2$?
3. $X_1, X_2$ jointly Gaussian with correlation 0.8. you observe $X_1 = 3$, or alternatively $X_1 = 0$. in which case are you more uncertain about $X_2$?
4. to compute the density of a 1000-dimensional Gaussian, do you need $\Sigma^{-1}$?
5. in 100 dimensions, how far from the mean is a typical sample from $\mathcal{N}(0, I)$? is the mean itself a typical sample?

## prerequisites check

- [ ] you can sample standard normals (→ ch 04 · topic 02, `mlfs.sampling.normal`)
- [ ] you know $\operatorname{Cov}(AX + b) = A\Sigma A^\top$ and that covariance matrices are symmetric positive semidefinite (→ ch 04 · topic 03)
- [ ] you can compute a Cholesky factorization and solve triangular systems (→ ch 02 · topic 03, `mlfs.linalg`)
- [ ] you know how densities transform under an invertible linear map (Jacobian determinant) (→ ch 03 · topic 06)

## core ideas

### idea 1: a Gaussian is a linear map applied to white noise

start with $z = (z_1, \dots, z_d)$, independent standard normals: "white noise", $z \sim \mathcal{N}(0, I)$. its density is a product of $d$ bells, $\prod_i \frac{1}{\sqrt{2\pi}}e^{-z_i^2/2} = (2\pi)^{-d/2}e^{-\|z\|^2/2}$, perfectly round. now stretch, rotate and shift it with a matrix $L$ and a vector $\mu$:

$$x = \mu + Lz.$$

that's the definition worth remembering: **a multivariate Gaussian is an affine image of white noise**. its mean is $\mu$, and by topic 03's rule its covariance is $L\,I\,L^\top = LL^\top$. so to sample $\mathcal{N}(\mu, \Sigma)$, find any $L$ with $LL^\top = \Sigma$ and push standard normals through. the Cholesky factor is the natural choice: it's triangular, cheap ($\tfrac13 d^3$), and you already built it. that answers warm-up 1: for unit variances and correlation 0.8, $\Sigma = \begin{pmatrix}1 & 0.8\\0.8 & 1\end{pmatrix}$, $L = \begin{pmatrix}1 & 0\\ 0.8 & 0.6\end{pmatrix}$, so $x_1 = z_1$ and $x_2 = 0.8z_1 + 0.6z_2$.

for a batch, stack $n$ noise vectors as the rows of $Z$ ($n \times d$). then each row should become $\mu + Lz$, which in row form is

$$X = \mathbf{1}\mu^\top + ZL^\top.$$

writing `Z @ L` instead of `Z @ L.T` is the classic bug: it gives samples with covariance $L^\top L$, which is a different matrix whenever $L$ isn't symmetric.

the picture: the round contours of white noise become **ellipses** (ellipsoids in higher dimensions). their axes point along the eigenvectors of $\Sigma$, with half-lengths proportional to the square roots of its eigenvalues (ch 02 · topic 05). correlation tilts the ellipse; unequal variances stretch it.

the affine-image view also settles half of warm-up 2: if $x = \mu + Lz$, then $Ax + b = (A\mu + b) + (AL)z$ is again an affine image of white noise. **linear functions of Gaussians are Gaussian**, with mean $A\mu + b$ and covariance $A\Sigma A^\top$. $X_1 + X_2$ is $(1, 1)\,x$, so it's Gaussian.

> **▶ build: stage 1.** implement `MultivariateNormal.__init__(mean, cov)` (store copies, `dim`, and `chol` from `mlfs.linalg.cholesky`) and `sample(n=1, *, rng=None)`, which uses *your* `mlfs.sampling.normal`, in `exercise.py` (`marimo edit exercise.py`). all four stages fill in methods of the same class.

### idea 2: the density, without ever inverting

the change-of-variables formula (ch 03 · topic 06) turns the density of $z$ into the density of $x = \mu + Lz$: $z = L^{-1}(x - \mu)$, and the Jacobian contributes $1/|\det L|$. with $\det\Sigma = (\det L)^2$ and $\|L^{-1}(x - \mu)\|^2 = (x - \mu)^\top\Sigma^{-1}(x - \mu)$, you get the familiar formula

$$p(x) = (2\pi)^{-d/2}\,|\Sigma|^{-1/2}\exp\!\Big(-\tfrac12(x - \mu)^\top\Sigma^{-1}(x - \mu)\Big).$$

the quantity in the exponent, $\Delta^2 = (x - \mu)^\top\Sigma^{-1}(x - \mu)$, is the squared **Mahalanobis distance**: distance from the mean measured in units of the distribution's own spread, in each direction. the contours of the density are the sets of constant $\Delta$.

in ML you almost always want the **log** density, both because likelihoods of many points multiply (and logs add) and because $p(x)$ underflows to 0 for points more than about 38 standard deviations out, while $\log p(x)$ is a perfectly ordinary number:

$$\log p(x) = -\tfrac12\Big[\Delta^2 + \log|\Sigma| + d\log 2\pi\Big].$$

now warm-up 4. you need neither $\Sigma^{-1}$ nor $|\Sigma|$. with $\Sigma = LL^\top$:

- **the quadratic form.** solve the triangular system $Lw = x - \mu$ (forward substitution, $d^2$ operations). then $\Delta^2 = \|w\|^2$.
- **the log determinant.** $\log|\Sigma| = 2\log|\det L| = 2\sum_i\log L_{ii}$. a sum of logs never overflows, while $|\Sigma|$ itself for a $1000 \times 1000$ covariance with eigenvalues near 0.1 is $10^{-1000}$, which is 0 in floating point.

forming $\Sigma^{-1}$ would cost more, lose accuracy when $\Sigma$ is ill-conditioned (ch 02 · topic 08), and buy nothing.

one more summary of the distribution you'll meet in chapter 05: the **differential entropy**, the expected value of $-\log p(x)$. since $\mathbb{E}[\Delta^2] = d$ (on paper 3),

$$H = \tfrac{d}{2}(1 + \log 2\pi) + \tfrac12\log|\Sigma|.$$

it depends only on the volume of the ellipse, not on where it is.

> **🔮 predict:** a 2-D Gaussian has correlation 0.99 between its coordinates. compared with the uncorrelated version (same variances), is its entropy higher or lower? its peak density?

> **▶ build: stage 2.** in the `MultivariateNormal` class cell (under idea 1 in the notebook), implement `logpdf(x)` (a point `(d,)` gives a float, a batch `(n, d)` gives `(n,)`), `pdf(x)`, and `entropy()`. use your `solve_triangular` and the diagonal of `chol`.

### idea 3: marginals are selections

you have a joint Gaussian over $(x_a, x_b)$ and you want the distribution of $x_a$ alone. in general, marginalizing means integrating out $x_b$ (topic 01's sum, as an integral), which is hard. for a Gaussian it's free. selecting coordinates is a linear map: $x_a = Px$ where $P$ picks out the rows in $a$. by idea 1, $x_a$ is Gaussian with mean $P\mu = \mu_a$ and covariance $P\Sigma P^\top = \Sigma_{aa}$, the submatrix of rows and columns in $a$. so

$$x_a \sim \mathcal{N}(\mu_a, \Sigma_{aa}).$$

just delete the rows and columns you don't care about. that's the rest of warm-up 2: $X_1$ alone is $\mathcal{N}(\mu_1, \Sigma_{11})$.

a warning about the converse: two variables that are each Gaussian need not be *jointly* Gaussian. take $X \sim \mathcal{N}(0, 1)$ and $Y = SX$ with $S = \pm 1$ a fair coin flip. $Y$ is standard normal too, but $X + Y$ is 0 half the time, which no Gaussian does. "jointly Gaussian" is a property of the whole vector.

> **▶ build: stage 3.** in the class cell under idea 1, implement `marginal(idx)`, returning a new `MultivariateNormal` over `x[idx]` in the order given.

### idea 4: conditionals are Gaussian too

now you *observe* some coordinates, $x_a = v$, and want your updated belief about the rest. split everything into blocks:

$$\mu = \begin{pmatrix}\mu_a\\\mu_b\end{pmatrix}, \qquad \Sigma = \begin{pmatrix}\Sigma_{aa} & \Sigma_{ab}\\ \Sigma_{ba} & \Sigma_{bb}\end{pmatrix}.$$

then $x_b \mid x_a = v$ is Gaussian with

$$\mu_{b|a} = \mu_b + \Sigma_{ba}\Sigma_{aa}^{-1}(v - \mu_a), \qquad \Sigma_{b|a} = \Sigma_{bb} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}.$$

here's a quick way to see where this comes from (on paper 2 fills in the details). let $K = \Sigma_{ba}\Sigma_{aa}^{-1}$ and define the residual $r = x_b - Kx_a$. a covariance computation shows $\operatorname{Cov}(r, x_a) = \Sigma_{ba} - K\Sigma_{aa} = 0$, and for jointly Gaussian vectors, uncorrelated means independent. so $r$ doesn't care what $x_a$ turned out to be. writing $x_b = Kx_a + r$ and plugging in $x_a = v$ gives a Gaussian with mean $Kv + \mathbb{E}[r]$ and covariance $\operatorname{Cov}(r)$, which are the two formulas.

read them slowly, because each piece means something:

- the conditional mean is the prior mean plus a correction that is **linear** in the surprise $v - \mu_a$. $K$ is exactly the coefficient matrix of a linear regression of $x_b$ on $x_a$. for a Gaussian, the best predictor is linear.
- the conditional covariance is the prior covariance **minus** a positive semidefinite term: observing never increases uncertainty, on average or otherwise. $\Sigma_{b|a}$ is the **Schur complement** of $\Sigma_{aa}$ in $\Sigma$ (you met it in ch 02 · topic 03's "why SPD needs no pivoting").
- $v$ does not appear in $\Sigma_{b|a}$ at all. that's warm-up 3: observing $X_1 = 3$ or $X_1 = 0$ leaves you *equally* uncertain about $X_2$. what you observe moves the mean; *that* you observe shrinks the variance. (this is special to Gaussians.)

geometrically, **conditioning is slicing**: $p(x_b \mid x_a = v) \propto p(x_b, v)$, so the conditional density is the joint density along the slice $x_a = v$, rescaled to integrate to 1. a slice through an ellipse is a line segment, centered where the slice crosses the ellipse's tilted axis, with a width that doesn't depend on where you cut. the lab lets you drag the slice around.

in code, $\Sigma_{aa}^{-1}$ appears twice, and as always you never form it: factor $\Sigma_{aa} = L_aL_a^\top$ once with `cholesky`, and use `cho_solve` for both $\Sigma_{aa}^{-1}(v - \mu_a)$ and $\Sigma_{aa}^{-1}\Sigma_{ab}$. keep the remaining coordinates in their original order, and symmetrize the result, $(C + C^\top)/2$, since rounding makes the subtraction very slightly asymmetric and your `cholesky` checks symmetry.

> **🔮 predict:** five variables form a chain, each one a noisy copy of the previous. the covariance matrix has no zeros: everything is correlated with everything. if you condition on the middle variable, what happens to the correlation between the first and the last?

> **▶ build: stage 4.** in the class cell under idea 1, implement `condition(idx, values)`, returning a `MultivariateNormal` over the remaining coordinates.

## worked example

take

$$\mu = \begin{pmatrix}0\\1\end{pmatrix}, \qquad \Sigma = \begin{pmatrix}4 & 2\\2 & 3\end{pmatrix}.$$

**Cholesky.** $L_{11} = \sqrt4 = 2$, $L_{21} = 2/2 = 1$, $L_{22} = \sqrt{3 - 1^2} = \sqrt2$. so $L = \begin{pmatrix}2 & 0\\1 & \sqrt2\end{pmatrix}$, and indeed $LL^\top = \begin{pmatrix}4 & 2\\2 & 3\end{pmatrix}$.

**a sample.** with noise $z = (1, -1)$: $x = \mu + Lz = (0 + 2,\ 1 + 1 - \sqrt 2) \approx (2, 0.586)$.

**the log density at $x = (1, 2)$.** $x - \mu = (1, 1)$. forward substitution in $Lw = (1, 1)$: $w_1 = 1/2$, $w_2 = (1 - 1 \cdot \tfrac12)/\sqrt2 = \tfrac{1}{2\sqrt2}$. so $\Delta^2 = \tfrac14 + \tfrac18 = \tfrac38$. (check with the inverse: $\Sigma^{-1} = \tfrac18\begin{pmatrix}3 & -2\\-2 & 4\end{pmatrix}$, and $(1, 1)\Sigma^{-1}(1, 1)^\top = \tfrac{3 - 2 - 2 + 4}{8} = \tfrac38$. ✓) $\log|\Sigma| = 2(\log 2 + \log\sqrt2) = \log 8$. so

$$\log p(1, 2) = -\tfrac12\Big[\tfrac38 + \log 8 + 2\log 2\pi\Big] \approx -\tfrac12[0.375 + 2.079 + 3.676] \approx -3.065.$$

**marginals.** $x_1 \sim \mathcal{N}(0, 4)$ and $x_2 \sim \mathcal{N}(1, 3)$.

**a conditional.** observe $x_1 = 2$. with $a = \{1\}$, $b = \{2\}$: $K = \Sigma_{21}/\Sigma_{11} = 2/4 = 0.5$, so

$$\mu_{2|1} = 1 + 0.5\,(2 - 0) = 2, \qquad \Sigma_{2|1} = 3 - \tfrac{2^2}{4} = 2.$$

the variance dropped from 3 to 2 because the coordinates are correlated ($\rho = 2/\sqrt{12} \approx 0.58$), and it would be 2 for any observed value.

the tests check the log density, the marginal, and the conditional exactly.

## on paper

1. **sampling is correct.** show that $x = \mu + Lz$ with $z \sim \mathcal{N}(0, I)$ and $LL^\top = \Sigma$ has mean $\mu$ and covariance $\Sigma$. then use the change-of-variables formula to derive the density of $x$ from the density of $z$.

<details><summary>solution</summary>

$\mathbb{E}[x] = \mu + L\,\mathbb{E}[z] = \mu$. $\operatorname{Cov}(x) = L\operatorname{Cov}(z)L^\top = LL^\top = \Sigma$. for the density: $z = L^{-1}(x - \mu)$ and $|\det(\partial z/\partial x)| = 1/|\det L| = |\Sigma|^{-1/2}$. so $p_x(x) = p_z(L^{-1}(x-\mu))\,|\Sigma|^{-1/2} = (2\pi)^{-d/2}|\Sigma|^{-1/2}\exp(-\tfrac12\|L^{-1}(x-\mu)\|^2)$, and $\|L^{-1}(x-\mu)\|^2 = (x-\mu)^\top L^{-\top}L^{-1}(x-\mu) = (x-\mu)^\top\Sigma^{-1}(x-\mu)$.

</details>

2. **the conditional.** with $K = \Sigma_{ba}\Sigma_{aa}^{-1}$ and $r = x_b - Kx_a$, show that $\operatorname{Cov}(r, x_a) = 0$ and $\operatorname{Cov}(r) = \Sigma_{bb} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}$. then derive $\mu_{b|a}$ and $\Sigma_{b|a}$, using the fact that uncorrelated jointly Gaussian vectors are independent.

<details><summary>solution</summary>

$\operatorname{Cov}(r, x_a) = \operatorname{Cov}(x_b, x_a) - K\operatorname{Cov}(x_a, x_a) = \Sigma_{ba} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{aa} = 0$. $(r, x_a)$ is a linear function of $x$, hence jointly Gaussian, so $r$ is independent of $x_a$. $\operatorname{Cov}(r) = \Sigma_{bb} - K\Sigma_{ab} - \Sigma_{ba}K^\top + K\Sigma_{aa}K^\top = \Sigma_{bb} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}$ (the last three terms are each $\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}$, with signs $-, -, +$). given $x_a = v$, $x_b = Kv + r$ where $r$ still has its unconditional distribution, $\mathcal{N}(\mu_b - K\mu_a, \operatorname{Cov}(r))$. so $\mu_{b|a} = \mu_b + K(v - \mu_a)$ and $\Sigma_{b|a} = \operatorname{Cov}(r)$.

</details>

3. **the entropy.** show that $\mathbb{E}[(x - \mu)^\top\Sigma^{-1}(x - \mu)] = d$ for $x \sim \mathcal{N}(\mu, \Sigma)$, and use it to derive $H = \tfrac d2(1 + \log 2\pi) + \tfrac12\log|\Sigma|$.

<details><summary>solution</summary>

the quadratic form equals $\|z\|^2 = \sum_i z_i^2$ with $z = L^{-1}(x - \mu) \sim \mathcal{N}(0, I)$, so its expectation is $\sum_i\mathbb{E}[z_i^2] = d$. (it's chi-square with $d$ degrees of freedom.) then $H = \mathbb{E}[-\log p(x)] = \tfrac12\big(\mathbb{E}[\Delta^2] + \log|\Sigma| + d\log 2\pi\big) = \tfrac12(d + \log|\Sigma| + d\log 2\pi)$.

</details>

4. **why a typical sample is far from the mean.** for $x \sim \mathcal{N}(0, I_d)$, find $\mathbb{E}\|x\|^2$ and $\operatorname{Var}\|x\|^2$ (use $\mathbb{E}[z^4] = 3$ for a standard normal). what does that say about $\|x\|/\sqrt d$ for large $d$?

<details><summary>solution</summary>

$\|x\|^2 = \sum_i x_i^2$ with independent terms of mean 1 and variance $\mathbb{E}[z^4] - 1 = 2$. so $\mathbb{E}\|x\|^2 = d$ and $\operatorname{Var}\|x\|^2 = 2d$. the standard deviation $\sqrt{2d}$ is tiny compared with the mean $d$, so $\|x\|^2/d \to 1$, and $\|x\| \approx \sqrt d$ with fluctuations of order 1 (about $1/\sqrt2$). almost all the probability sits in a thin shell of radius $\sqrt d$, even though the density is highest at the origin. that's warm-up 5: in 100 dimensions, a typical sample is about 10 units from the mean, and the mean itself is wildly atypical.

</details>

## experiment

`lab.py` runs three experiments on your `MultivariateNormal`:

1. conditioning as slicing: drag a slice across a 2-D Gaussian and compare the renormalized slice with your `condition`.
2. a chain of five variables: what conditioning on one of them does to the correlations of the others, and what the precision matrix knew all along.
3. the shell: where samples from a high-dimensional Gaussian actually live, and how their densities compare with the density at the mean.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. why is any $L$ with $LL^\top = \Sigma$ good enough for sampling, but you use the Cholesky factor in particular?

<details><summary>answer</summary>

the samples $\mu + Lz$ have covariance $LL^\top = \Sigma$ whichever square root you use (Cholesky, symmetric square root, $Q\Lambda^{1/2}$ from the eigendecomposition). Cholesky is the cheapest to compute ($\tfrac13d^3$), it's triangular so solves with it are $d^2$, and the same factor gives the log determinant for free.

</details>

2. your logpdf returns `-inf` for a point 50 standard deviations from the mean. what went wrong?

<details><summary>answer</summary>

you computed `log(pdf(x))`. the density underflows to 0 below about $e^{-745}$, about 38 standard deviations out, so its log is $-\infty$. compute $-\tfrac12(\Delta^2 + \log|\Sigma| + d\log2\pi)$ directly: $\Delta^2 = 2500$ is no problem at all.

</details>

3. the conditional covariance doesn't depend on the observed value. give an example of a non-Gaussian distribution where it does.

<details><summary>answer</summary>

a heteroscedastic model: $X \sim \text{Uniform}(1, 10)$ and $Y \mid X \sim \mathcal{N}(0, X^2)$. observing $X = 9$ leaves $Y$ far more uncertain than observing $X = 1$. (a mixture of two Gaussians works too: conditioning can even make you *more* uncertain.)

</details>

4. you compute $\Sigma_{b|a}$ and your `cholesky` complains that it isn't symmetric. why, and what's the fix?

<details><summary>answer</summary>

$\Sigma_{bb} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}$ is symmetric in exact arithmetic, but rounding in the matrix products makes the two triangles differ in the last bits. symmetrize with $(C + C^\top)/2$, which changes nothing that matters.

</details>

5. what's the conditional distribution when $x_a$ and $x_b$ are uncorrelated ($\Sigma_{ab} = 0$)?

<details><summary>answer</summary>

$K = 0$, so $\mu_{b|a} = \mu_b$ and $\Sigma_{b|a} = \Sigma_{bb}$: observing $x_a$ changes nothing. for jointly Gaussian vectors, zero correlation means independence.

</details>

6. how does a Gaussian process prediction (ch 09) relate to `condition`?

<details><summary>answer</summary>

it *is* `condition`. a GP says the function values at the training inputs and the test inputs are jointly Gaussian, with covariance given by a kernel. observing the training values and conditioning gives the predictive mean and variance at the test inputs, with exactly idea 4's formulas ($\Sigma_{aa}$ is the kernel matrix of the training inputs, plus noise).

</details>

7. the marginal of a Gaussian is "just delete rows and columns". why isn't the conditional "just delete rows and columns of $\Sigma^{-1}$"?

<details><summary>answer</summary>

it almost is. with the precision matrix $\Lambda = \Sigma^{-1}$, the conditional covariance is $\Sigma_{b|a} = (\Lambda_{bb})^{-1}$: delete rows and columns of the precision and invert. marginalizing is easy in the covariance parameterization, conditioning in the precision one. that duality is why zeros in $\Lambda$ mean conditional independence (the lab's chain), and why sparse precision matrices are the backbone of Gaussian graphical models (ch 12).

</details>

8. (the predict callout in idea 2.) two unit-variance coordinates with correlation 0.99: how do the entropy and the peak density compare with the uncorrelated case?

<details><summary>answer</summary>

$|\Sigma| = 1 - \rho^2 = 0.0199$ instead of 1. the entropy drops by $\tfrac12\log(1/0.0199) \approx 1.96$ nats, and the peak density $(2\pi)^{-1}|\Sigma|^{-1/2}$ is $1/\sqrt{0.0199} \approx 7.1$ times higher. the ellipse is a thin needle: the same total probability squeezed into about a seventh of the area. strong correlation means the pair carries much less uncertainty than the two coordinates separately suggest.

</details>

## common bugs

- **`Z @ L` instead of `Z @ L.T`.** symptom: samples with the right variances but the wrong correlations (covariance $L^\top L$). invisible for diagonal or 1-D covariances, which is why the tests use a lopsided $L$.
- **`log(pdf(x))`.** symptom: `-inf` for points far from the mean, and slow code.
- **forgetting the 2 in $\log|\Sigma| = 2\sum\log L_{ii}$.** symptom: logpdf off by a constant that depends on $\Sigma$.
- **`np.linalg.inv`.** works on the test cases, loses digits when $\Sigma$ is ill-conditioned, and is slower. use triangular solves and `cho_solve`.
- **reordering the remaining coordinates.** `condition` must return them in their original order, and `values[k]` belongs to `idx[k]`. symptom: correct numbers in the wrong places when `idx` isn't sorted.
- **slicing rows only.** `cov[idx]` takes rows; the marginal covariance needs rows *and* columns, `cov[np.ix_(idx, idx)]`.

## going deeper

- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, §2.3: the Gaussian distribution, conditionals and marginals (with the completing-the-square derivation), and the precision-matrix view. free from Microsoft Research.
- Murphy (2022), *Probabilistic Machine Learning: An Introduction*, MIT Press, §3.2: the multivariate Gaussian, including the "inference in jointly Gaussian systems" theorems. free at [probml.github.io/pml-book](https://probml.github.io/pml-book/).
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press, appendix A.2 and chapter 2: Gaussian identities, and conditioning as prediction. free at [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml/).
- Petersen & Pedersen (2012), *The Matrix Cookbook*, §8: a dense reference of Gaussian identities. free online.
- Vershynin (2018), *High-Dimensional Probability*, Cambridge University Press, §3.1: concentration of the norm, the shell of on paper 4.

## connections

- ← builds on: `normal` (topic 02), $\operatorname{Cov}(AX) = A\Sigma A^\top$ (topic 03), `cholesky`, `cho_solve`, `solve_triangular` (ch 02 · topic 03), eigenvectors as ellipse axes (ch 02 · topic 05), change of variables (ch 03 · topic 06).
- → used in: the CLT, where Gaussians appear as limits (topic 05), Brownian motion, a Gaussian process in time (topic 07), Gaussian entropy and KL divergence (ch 05), Gaussian discriminant analysis (ch 08), Gaussian processes (ch 09), Gaussian mixtures and PCA's probabilistic view (ch 11), Kalman filters and Gaussian graphical models (ch 12), variational inference and the VAE's reparameterization (ch 12, 21), and diffusion models (ch 22).
