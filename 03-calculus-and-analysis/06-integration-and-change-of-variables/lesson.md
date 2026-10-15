# integration & change of variables

## why this matters

probability is integration. the probability of an event is the integral of a density over a region; an expectation is the integral of a function against a density; a marginal likelihood is an integral over every parameter setting. in machine learning those integrals live in many dimensions, and you need two skills to handle them: **computing** them (by grids when the dimension is small, by random sampling when it isn't) and **transforming** them (changing variables to make an integral easy, or to build a new distribution out of a simple one).

the second skill is the engine of a whole family of generative models. a **normalizing flow** (chapter 21) takes Gaussian noise $Z$, pushes it through an invertible neural network $T$, and produces images, molecules or audio $X = T(Z)$. to train it by maximum likelihood you need the density of $X$, and the change-of-variables formula gives it: the density of $Z$ at $T^{-1}(x)$, corrected by how much $T$ stretches volume near that point, a **Jacobian determinant**. get the correction wrong and your "density" no longer integrates to 1. this topic builds the pieces: grid and Monte Carlo integrators in $d$ dimensions, log-Jacobian determinants, and densities of transformed random variables, and checks that a transformed Gaussian still integrates to 1.

## warm-up: predict first

1. a 1-D integral needs 100 grid points for the accuracy you want. how many does the same accuracy need in 10 dimensions, on a grid?
2. Monte Carlo's error shrinks like $1/\sqrt n$. does that rate depend on the dimension?
3. the map $T(z) = 2z$ (in 1-D) doubles every length. if $Z$ has density $p_Z$, is $p_X(x) = p_Z(x/2)$ a valid density for $X = 2Z$?
4. polar coordinates: a small rectangle $dr \times d\theta$ at radius $r$ becomes a small patch in the plane. what is its area?
5. $Z$ is standard normal, so its most likely value is 0. $X = e^Z$. is the most likely value of $X$ equal to $e^0 = 1$?

## prerequisites check

- [ ] you can integrate in 1-D with Riemann sums, the trapezoid and Simpson rules, and know their error orders (→ ch 01 · topic 03)
- [ ] you can estimate a mean with an error bar by simulation, and know why the error falls like $1/\sqrt n$ (→ ch 01 · topic 05)
- [ ] you can compute numerical Jacobians with `numerical_jacobian` (→ topic 02)
- [ ] you know that $|\det A|$ is the factor by which the linear map $A$ scales volume (→ ch 02 · topic 03)
- [ ] you know what a probability density is and that it integrates to 1 (→ ch 01 · topic 05)

## core ideas

### idea 1: integrating on a grid, in $d$ dimensions

a double integral over a rectangle can be done one variable at a time (Fubini's theorem): $\int_a^b\int_c^d f(x, y)\,dy\,dx$. so a grid rule in $d$ dimensions is a 1-D rule applied along every axis: split each side of the box into $n$ cells, evaluate $f$ at the center of each of the $n^d$ little boxes, and add up value × cell volume. that's the **tensor-product midpoint rule**:

$$\int_{\text{box}} f(x)\,dx \approx \sum_{\text{cells}} f(\text{cell center})\cdot\prod_{k=1}^d h_k,\qquad h_k = \frac{\text{high}_k - \text{low}_k}{n}.$$

in 1-D the midpoint rule's error is $O(h^2)$, and the same holds per axis in $d$ dimensions: the error is $O(h^2)$, and the midpoint rule is exact for any function that's linear in each variable separately, like $xy$.

the catch is the cost. $n$ cells per axis means $n^d$ evaluations. for a fixed budget of $N$ evaluations, $h \propto N^{-1/d}$ and the error falls like $N^{-2/d}$: in 1-D that's superb, in 10-D it's dreadful. matching 100 points' accuracy in 10-D takes $100^{10} = 10^{20}$ points (warm-up 1). this is the **curse of dimensionality** in its original form (Bellman's), and it's why grids stop being an option somewhere around 4 to 6 dimensions.

in NumPy, build the cell centers along each axis, combine them with `np.meshgrid(..., indexing="ij")`, and reshape to an $(n^d, d)$ array of points: one call to a vectorized `f` evaluates everything.

> **▶ build: stage 1.** implement `midpoint_rule(f, low, high, n)`, a $d$-dimensional tensor-product midpoint rule with $n$ cells per axis, calling the vectorized `f` on an $(n^d, d)$ array of cell centers.

### idea 2: integrating with random points

rewrite the integral as an expectation. if $U$ is uniform on a box of volume $V$, its density is $1/V$, so

$$I = \int_{\text{box}} f(x)\,dx = V\int f(x)\frac1V\,dx = V\,\mathbb E[f(U)].$$

estimate the expectation by an average over $n$ random points: $\hat I = V\cdot\frac1n\sum_i f(U_i)$. this **Monte Carlo** estimator is unbiased, and by the central limit theorem its error is approximately normal with standard deviation

$$\operatorname{se}(\hat I) = \frac{V\sigma_f}{\sqrt n},$$

where $\sigma_f$ is the standard deviation of $f(U)$, which you estimate from the same samples (with `ddof=1`). the rate $n^{-1/2}$ **doesn't depend on the dimension** (warm-up 2). dimension enters only through $\sigma_f$, which can grow, but often modestly. somewhere around $d = 4$, where $N^{-2/d}$ and $N^{-1/2}$ cross, Monte Carlo starts beating grids, and in high dimensions it's the only game in town (experiment 2).

an estimate without an error bar is half an answer: the standard error tells you when to stop sampling. a well-calibrated $\hat I \pm 2\operatorname{se}$ interval contains the true value about 95% of the time. it's also a random answer: always take an `rng` argument (`None`, a seed, or a `Generator`, normalized with `np.random.default_rng(rng)`) so that results are reproducible.

> **🔮 predict:** you run your Monte Carlo integrator 200 times with different seeds and form the interval estimate ± 2 se each time. how many of the 200 intervals contain the true value? what would change if you divided by $n$ instead of $\sqrt n$?

> **▶ build: stage 2.** implement `monte_carlo_integrate(f, low, high, n, *, rng=None)`, returning `(estimate, standard_error)`.

### idea 3: changing variables and the Jacobian determinant

in 1-D you know substitution: with $x = T(z)$, $\int f(x)\,dx = \int f(T(z))\,|T'(z)|\,dz$. the factor $|T'(z)|$ says how much $T$ stretches a little interval $dz$. in $d$ dimensions, near $z$ the map looks like its Jacobian (topic 02), $T(z + dz) \approx T(z) + J_T(z)\,dz$, and a linear map scales every volume by $|\det J_T(z)|$. so a tiny box of volume $dz$ around $z$ becomes a tiny parallelogram of volume $|\det J_T(z)|\,dz$ around $T(z)$, and

$$\int f(x)\,dx = \int f(T(z))\,\big|\det J_T(z)\big|\,dz.$$

**polar coordinates** are the classic example (warm-up 4). $T(r, \theta) = (r\cos\theta, r\sin\theta)$ has $\det J_T = r$: a rectangle $dr \times d\theta$ at radius $r$ becomes a curved patch of width $dr$ and length $r\,d\theta$. that's the $r$ in $dx\,dy = r\,dr\,d\theta$, and it turns the Gaussian integral into something you can do by hand (worked example).

in high dimensions, $\det J$ can be astronomically large or small (a product of $d$ stretch factors), so always work with $\log|\det J|$. `np.linalg.slogdet` returns the sign and the log of the absolute value directly from an LU factorization, without ever forming the determinant. a negative determinant just means $T$ flips orientation (like a reflection), which changes nothing about volumes: take the absolute value.

> **▶ build: stage 3.** implement `log_abs_det_jacobian(T, z)` with your `numerical_jacobian` and `np.linalg.slogdet`.

### idea 4: densities under invertible maps

now the probability version. $Z$ has density $p_Z$, $T$ is invertible and differentiable, and $X = T(Z)$. probability is conserved: the probability that $X$ lands in a tiny region around $x$ equals the probability that $Z$ lands in the corresponding region around $z = T^{-1}(x)$. but the regions have different volumes, related by $|\det J_T(z)|$. density is probability per unit volume, so

$$p_X(x) = p_Z(z)\,\frac{1}{|\det J_T(z)|} = p_Z\big(T^{-1}(x)\big)\,\big|\det J_{T^{-1}}(x)\big|,\qquad z = T^{-1}(x).$$

(the two forms agree because the Jacobian of the inverse is the inverse of the Jacobian.) where $T$ spreads points out ($|\det J_T| > 1$), the density thins; where it squeezes them together, the density piles up. warm-up 3: for $X = 2Z$, $p_X(x) = p_Z(x/2)\cdot\tfrac12$; without the $\tfrac12$ it would integrate to 2. on a log scale, which is how flows are trained,

$$\log p_X(x) = \log p_Z\big(T^{-1}(x)\big) + \log\big|\det J_{T^{-1}}(x)\big|.$$

two consequences surprise people. first, some nonlinear maps don't need the correction at all: if $\det J_T = 1$ everywhere ("volume-preserving", like a shear), the density just moves. normalizing flows exploit this by stacking cheap triangular maps whose determinants are trivial to compute. second, **the peak moves**. the Jacobian factor varies from place to place, so the mode of $X$ is generally *not* $T$(the mode of $Z$) (warm-up 5). quantiles, by contrast, do transform through a monotone $T$: the median of $e^Z$ is $e^{\text{median of }Z}$. "the most likely value" depends on how you parameterize the problem; probabilities of events don't.

> **🔮 predict:** you integrate $p_Z(T^{-1}(x))$ (forgetting the Jacobian factor) over the plane for the linear map $T(z) = Az$ with $\det A = 2$. what number comes out?

> **▶ build: stage 4.** implement `pushforward_log_density(log_pz, T_inv, x)`, for one point $(d,)$ or a batch $(m, d)$, using your `log_abs_det_jacobian` on $T^{-1}$.

## worked example

**the Gaussian integral, by changing variables.** let $I = \int_{-\infty}^\infty e^{-x^2}dx$. then

$$I^2 = \iint e^{-(x^2 + y^2)}\,dx\,dy = \int_0^{2\pi}\int_0^\infty e^{-r^2}\,r\,dr\,d\theta = 2\pi\cdot\tfrac12 = \pi,$$

so $I = \sqrt\pi$. the factor $r$ is $\det J$ of the polar map, and it's what makes the inner integral elementary ($\frac{d}{dr}(-\tfrac12e^{-r^2}) = re^{-r^2}$). your `midpoint_rule` should give $\pi$ for $\iint e^{-x^2-y^2}$ over $[-5, 5]^2$, and your `log_abs_det_jacobian` of the polar map at $r = 2$ should give $\log 2$.

**a transformed density.** $Z \sim \mathcal N(0, 1)$ and $X = e^Z$, so $T^{-1}(x) = \log x$ and $|dT^{-1}/dx| = 1/x$:

$$p_X(x) = \varphi(\log x)\cdot\frac1x,\qquad \varphi(z) = \frac{e^{-z^2/2}}{\sqrt{2\pi}}.$$

at $x = 1$: $\varphi(0) = 0.398942$, so $\log p_X(1) = -\tfrac12\log 2\pi = -0.918939$. at $x = e$: $\varphi(1)/e$, so $\log p_X(e) = -0.918939 - 0.5 - 1 = -2.418939$. the "$-1$" is the log-Jacobian term $\log(1/e)$. your `pushforward_log_density` must reproduce both, and the density of an affine map of a Gaussian, $X = AZ + b \sim \mathcal N(b, AA^\top)$.

## on paper

1. **the 1-D formula from CDFs.** for an increasing differentiable $T$, show $F_X(x) = F_Z(T^{-1}(x))$ and differentiate to get $p_X(x) = p_Z(T^{-1}(x))\,(T^{-1})'(x)$. what changes for a decreasing $T$?

<details><summary>solution</summary>

$F_X(x) = P(T(Z) \le x) = P(Z \le T^{-1}(x)) = F_Z(T^{-1}(x))$, since $T$ is increasing. the chain rule gives $p_X(x) = p_Z(T^{-1}(x))\,(T^{-1})'(x)$. for decreasing $T$, $P(T(Z) \le x) = P(Z \ge T^{-1}(x)) = 1 - F_Z(T^{-1}(x))$, whose derivative is $-p_Z(T^{-1}(x))(T^{-1})'(x)$ with $(T^{-1})' < 0$: the same formula with an absolute value, which is why the general formula has $|\det|$.

</details>

2. **the mode of a lognormal.** for $Z \sim \mathcal N(\mu, \sigma^2)$ and $X = e^Z$, show that $p_X$ is maximized at $x = e^{\mu - \sigma^2}$, while the median of $X$ is $e^\mu$.

<details><summary>solution</summary>

$\log p_X(x) = -\frac{(\log x - \mu)^2}{2\sigma^2} - \log x + \text{const}$. setting the derivative to zero: $-\frac{\log x - \mu}{\sigma^2x} - \frac1x = 0$, so $\log x = \mu - \sigma^2$. the median: $P(X \le e^\mu) = P(Z \le \mu) = \tfrac12$. the $-\log x$ term, the log-Jacobian, pulls the mode below the median, more so for larger $\sigma$.

</details>

3. **volume-preserving maps.** show that the shear $T(z) = (z_1,\ z_2 + g(z_1))$ has $\det J_T = 1$ for any differentiable $g$. what is $p_X$ for $X = T(Z)$?

<details><summary>solution</summary>

$J_T = \begin{pmatrix}1 & 0\\ g'(z_1) & 1\end{pmatrix}$, lower triangular with ones on the diagonal, so $\det J_T = 1$. hence $p_X(x) = p_Z(T^{-1}(x)) = p_Z(x_1,\ x_2 - g(x_1))$: no correction. triangular Jacobians make determinants cheap in general (the product of the diagonal), which is the design principle of coupling and autoregressive flows.

</details>

4. **how many samples?** you want a Monte Carlo estimate with relative standard error 1%. the integrand has $\sigma_f/|\mathbb E f| = 3$. how many samples do you need, and how does that compare with a grid in $d = 8$ if the grid's error is about $h^2$?

<details><summary>solution</summary>

relative se $= 3/\sqrt n \le 0.01$ needs $n \ge 90{,}000$, in any dimension. a grid with error $h^2 = 0.01$ needs $h = 0.1$, so 10 cells per axis on the unit cube, and $10^8 = 100{,}000{,}000$ points in $d = 8$: over a thousand times more. in $d = 2$ the grid needs 100 points. the crossover is the whole story of experiment 2.

</details>

## experiment

`lab.py` investigates three things with your code:

1. whether a Gaussian pushed through three different invertible maps still integrates to 1, with and without the Jacobian factor.
2. your grid integrator against your Monte Carlo integrator, at the same budget of function evaluations, from $d = 1$ to $d = 8$.
3. where the peak of the density of $X = e^Z$ sits, compared with $e^{\text{peak of }Z}$.

## check your understanding

1. why does the midpoint rule integrate $xy$ exactly over a rectangle, but not $x^2y$?

<details><summary>answer</summary>

on each cell, the midpoint rule is exact for functions linear in each variable: $xy$ is bilinear, and its average over a cell equals its value at the center. $x^2$ is curved, and its cell average exceeds the center value by $h^2/12$, so the rule underestimates it by that much per cell.

</details>

2. your Monte Carlo estimate is $2.31 \pm 0.04$ (one standard error) with $n = 10{,}000$. how many samples for $\pm 0.01$?

<details><summary>answer</summary>

the error scales like $1/\sqrt n$, so 4× smaller error needs 16× more samples: $160{,}000$.

</details>

3. (the predict callout in idea 2.) how many of 200 intervals "estimate ± 2 se" contain the truth?

<details><summary>answer</summary>

about 190 (95%), give or take a few: the estimate is approximately normal with standard deviation se, and ±2 standard deviations cover 95% of a normal. with $\sigma/n$ in place of $\sigma/\sqrt n$, the "standard error" is $\sqrt n$ times too small, the intervals are absurdly narrow, and almost none of them cover the truth. the stage 2 tests check exactly this coverage.

</details>

4. (the predict callout in idea 4.) what does $\int p_Z(T^{-1}(x))\,dx$ give for $T(z) = Az$ with $\det A = 2$?

<details><summary>answer</summary>

substitute $x = Az$: $\int p_Z(z)\,|\det A|\,dz = 2$. leaving out the Jacobian factor multiplies the total mass by $|\det A|$, exactly the factor by which $T$ inflates volume (experiment 1, "scale").

</details>

5. why does a normalizing flow need $T$ to be invertible, and why does it care about how cheap $\det J_T$ is?

<details><summary>answer</summary>

the likelihood of a data point needs $T^{-1}(x)$ (to find the noise that produced it) and $\log|\det J_{T^{-1}}(x)|$. a general $d \times d$ determinant costs $O(d^3)$, impossible for $d$ = number of pixels at every training step; triangular or volume-preserving designs make it $O(d)$ or free.

</details>

6. is the Jacobian factor needed for computing $\mathbb E[g(X)]$ by sampling, if you can sample $Z$?

<details><summary>answer</summary>

no: $\mathbb E[g(X)] = \mathbb E[g(T(Z))]$, so just sample $Z$, push it through $T$ and average $g$. the Jacobian appears only when you need the *density* of $X$ at a given point (likelihoods, importance weights). sampling is easy in a flow; evaluating densities is where the determinant earns its keep.

</details>

7. why is $\log|\det|$ computed with `slogdet` rather than `np.log(abs(np.linalg.det(J)))`?

<details><summary>answer</summary>

$\det J$ is a product of $d$ factors and overflows or underflows quickly: $1.1^{1000}$ is fine as a log ($95.3$) but `det` returns huge or tiny floats, `inf` or 0, in high dimensions. `slogdet` sums the logs of the LU pivots instead.

</details>

8. (warm-up 5.) the posterior mode of a variance parameter $\sigma^2$ and the posterior mode of $\log\sigma^2$ correspond to different values of $\sigma^2$. which one is "right"?

<details><summary>answer</summary>

neither is privileged: the mode isn't invariant under reparameterization, because densities pick up a Jacobian factor (experiment 3). quantities that *are* invariant (posterior probabilities of intervals, medians, expectations of functions of the parameter) are safer summaries. it's one argument for full Bayesian inference over point estimates (chapter 12).

</details>

## common bugs

- **left endpoints instead of midpoints.** symptom: an $O(h)$ error instead of $O(h^2)$, e.g. $1/3 - 1/(2n) + \dots$ for $\int_0^1x^2$. cell $i$'s center is $\text{low} + (i + \tfrac12)h$.
- **one cell width for a non-square box.** symptom: wrong answers whenever the sides differ. each axis has its own $h_k$, and the cell volume is their product.
- **forgetting the box volume.** symptom: Monte Carlo on $[0, 2] \times [0, 3]$ is 6× too small. the estimate is $V$ times the sample mean, and the standard error picks up the same $V$.
- **$\sigma/n$ instead of $\sigma/\sqrt n$.** symptom: absurdly confident error bars, and intervals that almost never cover the truth.
- **the Jacobian of $T$ where you need $T^{-1}$.** symptom: the log-density is off by twice the log-determinant, with the wrong sign. evaluate $\log|\det J_{T^{-1}}|$ at $x$ (or subtract $\log|\det J_T|$ at $z$).
- **`log(det)` without the absolute value.** symptom: `nan` for maps that flip orientation. use `slogdet` and its second output.

## going deeper

- Owen (2013), *Monte Carlo Theory, Methods and Examples*, chapters 1–2. the dimension-free $n^{-1/2}$ rate, error estimates and comparisons with quadrature; free online. [artowen.su.domains/mc](https://artowen.su.domains/mc/)
- Hubbard & Hubbard (2015), *Vector Calculus, Linear Algebra, and Differential Forms*, 5th ed., Matrix Editions, chapter 4. multiple integrals and the change-of-variables theorem.
- Papamakarios, Nalisnick, Rezende, Mohamed & Lakshminarayanan (2021), "Normalizing flows for probabilistic modeling and inference", *JMLR* 22(57): 1–64. the change-of-variables formula as a modeling tool. [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)
- Dinh, Sohl-Dickstein & Bengio (2017), "Density estimation using Real NVP", *ICLR*. flows built from volume-friendly triangular maps. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)

## connections

- ← builds on: 1-D quadrature (ch 01 · topic 03), Monte Carlo with error bars (ch 01 · topic 05), Jacobians (topic 02), determinants as volume scaling (ch 02 · topic 03).
- → used in: expectations, the CLT and the multivariate Gaussian (ch 04), Monte Carlo inference and the reparameterization trick (ch 12), importance sampling in RL (ch 23), normalizing flows such as RealNVP (ch 21), and the change of variables behind flow matching and diffusion (ch 22).
