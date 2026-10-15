# smoothness & lipschitz continuity

## why this matters

every time you train a model you pick a learning rate, and you've probably done it by trial and error: too small and nothing happens, too large and the loss explodes to `nan`. it turns out there's a number hiding in your loss function that predicts where "too large" begins. it's called the **smoothness constant** $L$: a bound on how fast the gradient can change. for a quadratic, gradient descent converges for every step size below $2/L$ and diverges for every step size above it, a knife edge you can compute in advance.

the same family of ideas answers other questions you'll meet repeatedly. how much can a classifier's output change if an attacker nudges the input by $\varepsilon$? (its **Lipschitz constant** bounds it.) why do some problems converge in 10 iterations and others in 10,000? (the **condition number** $L/\mu$, the ratio of the largest to the smallest curvature.) why do Wasserstein GANs clip or penalize gradients? (to enforce a Lipschitz constraint.) this topic makes these constants concrete: you'll estimate them from samples and from Hessians, verify the inequality that all of first-order optimization theory rests on, and measure how well $L$ predicts the largest stable step, including where it doesn't.

## warm-up: predict first

1. $f(x) = \sin(3x)$. what's the largest possible value of $|f(x) - f(y)|/|x - y|$?
2. you estimate a Lipschitz constant by taking the largest slope between pairs of random sample points. is the estimate too high, too low, or either?
3. for $f(x) = \tfrac12(x_1^2 + 10x_2^2)$, gradient descent with step $\eta$ multiplies $x_2$ by $(1 - 10\eta)$ each step. what's the largest $\eta$ that doesn't diverge?
4. $|x|$ is Lipschitz with constant 1. is it "smooth" in the sense of having a Lipschitz gradient?
5. in 50 dimensions, would 5,000 random pairs of points give you a good estimate of a quadratic's largest curvature?

## prerequisites check

- [ ] you can compute gradients and Hessians, and you know that the eigenvalues of the Hessian are the curvatures along its eigenvectors (→ topic 02)
- [ ] you know Taylor's theorem with remainder, including the multivariate version along a line (→ topic 03)
- [ ] you know what linear convergence with rate $C$ means (→ topic 01)
- [ ] you can compute pairwise distances with broadcasting (→ ch 00 · topic 02)
- [ ] you know the 2-norm of a matrix is its largest singular value (→ ch 02 · topic 06)

## core ideas

### idea 1: Lipschitz continuity: a speed limit

a function is **Lipschitz continuous** with constant $L$ if

$$\|f(x) - f(y)\| \le L\,\|x - y\|\quad\text{for all } x, y.$$

the output can't change faster than $L$ times the input. continuity says small changes in $x$ give small changes in $f(x)$; Lipschitz continuity says *how* small, uniformly, everywhere. the smallest valid $L$ is the Lipschitz constant.

for a differentiable function, the mean value theorem connects $L$ to derivatives: $f(y) - f(x) = \nabla f(\xi)\cdot(y - x)$ for some $\xi$ on the segment, so the best $L$ is $\sup_x\|\nabla f(x)\|$ (on a convex domain). warm-up 1: $|f'| = 3|\cos 3x| \le 3$, so $L = 3$. for a linear map $x \mapsto Ax$, $L = \|A\|_2$, the largest singular value. for a composition, the constants multiply: $L_{g\circ f} \le L_gL_f$. a neural network with 1-Lipschitz activations (ReLU, tanh) has $L \le \prod_\ell\|W_\ell\|_2$, a bound that's easy to compute and usually very loose, and the reason spectral normalization (which divides each $W_\ell$ by its largest singular value) controls a network's sensitivity.

some functions aren't Lipschitz at all: $\sqrt{|x|}$ has infinite slope at 0, $x^2$ has unbounded slope on $\mathbb{R}$ (it's Lipschitz only on bounded sets).

**estimating it.** given sample points $x_1, \dots, x_k$, the largest observed slope

$$\hat L = \max_{i<j}\frac{\|f(x_i) - f(x_j)\|}{\|x_i - x_j\|}$$

is always a **lower bound** on the true $L$ (warm-up 2): each ratio is a slope the function actually has, and the supremum might live between the samples. with $k$ points there are $k(k - 1)/2$ pairs; evaluate $f$ once per point and compute all pairwise distances with broadcasting.

> **▶ build: stage 1.** implement `lipschitz_estimate(f, points)`: evaluate `f` at each row, then take the largest ratio of output distance to input distance over all pairs $i < j$, skipping identical points. vectorize over pairs.

### idea 2: smoothness and strong convexity: bounds on curvature

for optimization the key constant isn't about $f$, it's about its gradient. $f$ is **$L$-smooth** if its gradient is $L$-Lipschitz:

$$\|\nabla f(x) - \nabla f(y)\| \le L\,\|x - y\|.$$

for twice-differentiable $f$ this is the same as $\|H(x)\|_2 \le L$ everywhere: every curvature, positive or negative, is at most $L$ in size. smoothness means the gradient you measure at $x$ stays a reasonable guide in a neighbourhood of $x$, a neighbourhood whose size is about $1/L$.

there's a matching *lower* bound. $f$ is **$\mu$-strongly convex** (with $\mu > 0$) if every curvature is at least $\mu$: $H(x) \succeq \mu I$, or without second derivatives,

$$(\nabla f(x) - \nabla f(y))\cdot(x - y) \ge \mu\,\|x - y\|^2.$$

(chapter 06 · topic 01 develops convexity properly; here it's a preview.) a strongly convex function curves up at least as fast as $\tfrac\mu2\|x\|^2$, so it has a unique minimum and no flat regions. for a quadratic $\tfrac12x^\top Ax$ with symmetric positive definite $A$, $L = \lambda_{\max}(A)$ and $\mu = \lambda_{\min}(A)$, and their ratio

$$\kappa = \frac{L}{\mu}$$

is the **condition number** of the optimization problem, the same $\kappa$ you met for linear systems in chapter 02 · topic 08. it measures how stretched the level sets are, and it will set the speed of gradient descent (idea 4).

both constants can be estimated from pairs of points. the smoothness estimate is `lipschitz_estimate` applied to the gradient (a lower bound on $L$). the strong-convexity estimate is the *smallest* "curvature along a chord" $(\nabla f(x_i) - \nabla f(x_j))\cdot(x_i - x_j)/\|x_i - x_j\|^2$ over pairs: an upper bound on $\mu$. if it comes out negative, you've proved the function isn't convex.

> **▶ build: stage 2.** implement `smoothness_estimate(grad_f, points)` and `strong_convexity_estimate(grad_f, points)`.

### idea 3: the descent lemma

smoothness has a consequence that all of first-order optimization theory rests on. apply Taylor's theorem along the segment from $x$ to $y$ (topic 03, on paper 2), with every curvature bounded by $L$:

$$f(y) \le f(x) + \nabla f(x)^\top(y - x) + \frac L2\|y - x\|^2.$$

this is the **descent lemma**, or the **quadratic upper bound**: at every point, $f$ lies below a parabola that touches it there and has curvature $L$. (with strong convexity, $f$ also lies *above* a parabola of curvature $\mu$.) it turns a local fact (the gradient) into a global guarantee about how much $f$ can rise when you move. the gap between the bound and $f(y)$ is non-negative for a valid $L$; a negative gap anywhere proves $L$ is too small.

a single global $L$ can be pessimistic. the curvature near a minimum is often much smaller than the worst curvature anywhere, so it's useful to measure **local smoothness**, the largest absolute eigenvalue of the Hessian at a point. the absolute value matters: a large *negative* curvature also limits how far a gradient step can be trusted.

> **🔮 predict:** for $f(x) = \|x\|^2$, which is exactly 2-smooth, what is the descent-lemma gap for $L = 2$, for any $x$ and $y$?

> **▶ build: stage 3.** implement `descent_lemma_gap(f, grad_f, L, x, y)` and `hessian_smoothness(f, x)` (max absolute eigenvalue of your `numerical_hessian`, via `np.linalg.eigvalsh`).

### idea 4: what $L$ says about step sizes

take a gradient step, $y = x - \eta\nabla f(x)$, and plug it into the descent lemma:

$$f(x - \eta\nabla f(x)) \le f(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\|\nabla f(x)\|^2.$$

for $0 < \eta < 2/L$ the bracket is positive and every step decreases $f$, by an amount proportional to $\|\nabla f\|^2$. the guaranteed decrease is largest at $\eta = 1/L$, which is why "$1/L$" is the textbook step size. for a quadratic $\tfrac12x^\top Ax$, gradient descent is the linear iteration $x_{k+1} = (I - \eta A)x_k$, and in the eigenbasis each coordinate is multiplied by $(1 - \eta\lambda_i)$ every step. it converges if and only if every $|1 - \eta\lambda_i| < 1$, i.e.

$$0 < \eta < \frac{2}{\lambda_{\max}} = \frac2L$$

(warm-up 3: $\eta < 0.2$). at $\eta = 2/L$ the top coordinate flips sign forever; above it, it grows geometrically. the slowest coordinate shrinks by $1 - \eta\mu$, so even at the best fixed step, the error contracts like $1 - O(1/\kappa)$ per step: ill-conditioned problems (large $\kappa$) are slow, which chapter 06 explores in depth.

two caveats, both measured in the lab. for non-quadratic losses, $2/L$ is a *guarantee*, not a prediction: the threshold that actually matters is set by the curvature where the iterates spend their time, often much smaller than the global worst case. and for non-smooth functions like $|x|$ (warm-up 4: no, its gradient jumps from $-1$ to $+1$), no step size works: the gradient doesn't shrink near the minimum, so a fixed step overshoots back and forth forever.

> **🔮 predict:** you estimate $L$ for a 50-dimensional quadratic from 5,000 random pairs (warm-up 5) and run gradient descent at $1.9/\hat L$. what happens?

to *measure* the threshold you need an operational definition. here "stable" means "converges within budget": after `n_iters` steps, the gradient norm has dropped below `tol` times its starting value. it's a relative test, so it doesn't depend on how far from the minimum you start. on a quadratic it measures slightly less than $2/L$: the top coordinate shrinks by $|1 - \eta L|$ per step, and $|1 - \eta L|^n \le \text{tol}$ means $\eta \le (1 + \text{tol}^{1/n})/L \approx 1.93/L$ at the defaults ($n = 100$, tol $= 10^{-3}$). very *small* steps can fail it too, because the flattest coordinate, shrinking by $1 - \eta\mu$, doesn't get there in $n$ steps. and above the threshold, gradient descent doesn't always blow up: on a non-quadratic loss it can settle into a bounded back-and-forth (a 2-cycle). that run "survives" without ever converging, and the test correctly rejects it.

> **▶ build: stage 4.** implement `gradient_descent(grad_f, x0, step, n_iters)`, returning all iterates, and `largest_stable_step(grad_f, x0, steps, *, n_iters=100, tol=1e-3)`, which tries *every* step (they needn't be sorted) and returns the largest one for which the final gradient norm drops below `tol` times the initial one.

## worked example

$f(x) = \tfrac12(x_1^2 + 10x_2^2)$, with $\nabla f = (x_1, 10x_2)$ and $H = \operatorname{diag}(1, 10)$. so $L = 10$, $\mu = 1$, $\kappa = 10$.

**descent lemma.** with $L = 10$, the gap at $x$, $y$ is $\tfrac12(10 - 1)(y_1 - x_1)^2 + \tfrac12(10 - 10)(y_2 - x_2)^2 \ge 0$: exact along $x_2$ (the bound is tight in the direction of largest curvature), loose along $x_1$.

**gradient descent from $(1, 1)$** multiplies $x_1$ by $1 - \eta$ and $x_2$ by $1 - 10\eta$:

| step $\eta$ | factors $(1 - \eta,\ 1 - 10\eta)$ | iterates |
|---|---|---|
| $0.1 = 1/L$ | $(0.9,\ 0)$ | $x_2$ is solved in one step; $x_1$ shrinks by 0.9 per step |
| $0.19$ | $(0.81,\ -0.9)$ | converges, with $x_2$ oscillating in sign |
| $0.2 = 2/L$ | $(0.8,\ -1)$ | $x_2$ flips between $\pm1$ forever |
| $0.21$ | $(0.79,\ -1.1)$ | $(1, 1) \to (0.79, -1.1) \to (0.6241, 1.21) \to (0.4930, -1.331) \to \dots$ diverges |

the best fixed step balances the two factors, $1 - \eta\mu = -(1 - \eta L)$, at $\eta = 2/(\mu + L) = 2/11$, where both have size $(\kappa - 1)/(\kappa + 1) = 9/11 \approx 0.82$. your `gradient_descent` must reproduce the $\eta = 0.21$ iterates, and your `largest_stable_step` must find that 0.19 works and 0.2 doesn't.

## on paper

1. **Lipschitz from gradients.** show that if $\|\nabla f(z)\| \le G$ for every $z$ on the segment between $x$ and $y$, then $|f(x) - f(y)| \le G\|x - y\|$. conversely, show that a differentiable $L$-Lipschitz $f$ has $\|\nabla f(x)\| \le L$ everywhere.

<details><summary>solution</summary>

let $\phi(t) = f(x + t(y - x))$. by the mean value theorem, $f(y) - f(x) = \phi'(\tau) = \nabla f(z)\cdot(y - x)$ for some $z$ on the segment, and Cauchy–Schwarz gives $|f(y) - f(x)| \le G\|y - x\|$. conversely, take $y = x + tu$ with $u = \nabla f(x)/\|\nabla f(x)\|$: $|f(x + tu) - f(x)|/t \le L$ for all $t > 0$, and the left side tends to $\nabla f(x)\cdot u = \|\nabla f(x)\|$ as $t \to 0$.

</details>

2. **the descent lemma.** prove $f(y) \le f(x) + \nabla f(x)^\top(y - x) + \tfrac L2\|y - x\|^2$ for $L$-smooth $f$, using $f(y) - f(x) = \int_0^1\nabla f(x + t(y - x))\cdot(y - x)\,dt$.

<details><summary>solution</summary>

subtract $\nabla f(x)\cdot(y - x) = \int_0^1\nabla f(x)\cdot(y - x)\,dt$:

$$f(y) - f(x) - \nabla f(x)^\top(y - x) = \int_0^1\big(\nabla f(x + t(y - x)) - \nabla f(x)\big)\cdot(y - x)\,dt \le \int_0^1 Lt\|y - x\|^2\,dt = \frac L2\|y - x\|^2,$$

using Cauchy–Schwarz and $\|\nabla f(x + td) - \nabla f(x)\| \le Lt\|d\|$.

</details>

3. **gradient descent on a quadratic.** for $f = \tfrac12x^\top Ax$ with symmetric positive definite $A$, show $x_k = (I - \eta A)^kx_0$, and that $x_k \to 0$ for every $x_0$ iff $0 < \eta < 2/\lambda_{\max}$. which fixed $\eta$ minimizes $\max_i|1 - \eta\lambda_i|$?

<details><summary>solution</summary>

$\nabla f = Ax$, so $x_{k+1} = x_k - \eta Ax_k = (I - \eta A)x_k$. writing $x_0 = \sum c_iq_i$ in $A$'s eigenbasis, $x_k = \sum_ic_i(1 - \eta\lambda_i)^kq_i$, which tends to 0 for all $c$ iff every $|1 - \eta\lambda_i| < 1$, i.e. $0 < \eta\lambda_i < 2$ for all $i$, i.e. $\eta < 2/\lambda_{\max}$. $\max_i|1 - \eta\lambda_i| = \max(|1 - \eta\mu|, |1 - \eta L|)$; it's minimized where $1 - \eta\mu = \eta L - 1$, at $\eta = 2/(\mu + L)$, with value $(L - \mu)/(L + \mu) = (\kappa - 1)/(\kappa + 1)$.

</details>

4. **the guaranteed decrease.** from the descent lemma, show that gradient descent with $\eta = 1/L$ satisfies $f(x_{k+1}) \le f(x_k) - \tfrac1{2L}\|\nabla f(x_k)\|^2$, and deduce that $\min_{k<K}\|\nabla f(x_k)\|^2 \le 2L(f(x_0) - f^*)/K$ for any $f$ bounded below by $f^*$, convex or not.

<details><summary>solution</summary>

with $y = x - \tfrac1L\nabla f(x)$: $f(y) \le f(x) - \tfrac1L\|\nabla f\|^2 + \tfrac L2\cdot\tfrac1{L^2}\|\nabla f\|^2 = f(x) - \tfrac1{2L}\|\nabla f\|^2$. summing over $k = 0, \dots, K - 1$ telescopes: $\tfrac1{2L}\sum_k\|\nabla f(x_k)\|^2 \le f(x_0) - f(x_K) \le f(x_0) - f^*$, and the minimum is at most the average. this is the basic guarantee for gradient descent on non-convex losses: the gradient gets small at a rate $O(1/\sqrt K)$.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how well $2/\hat L$ predicts the largest stable gradient-descent step on a regularized logistic-regression loss.
2. how good a sampled estimate of $L$ is for a quadratic in many dimensions, and what happens to gradient descent if you trust it.
3. gradient descent with a fixed step on a Lipschitz but non-smooth function, compared with its smooth (Huber) cousin.

## check your understanding

1. why is a sampled Lipschitz estimate always a lower bound, never an upper bound?

<details><summary>answer</summary>

each pair gives a slope the function really has between those two points, and the true constant is the supremum over *all* pairs. a finite sample can miss the steepest region, never exceed it. getting an upper bound needs structure: a bound on the gradient, or a product of layer norms.

</details>

2. what are $L$, $\mu$ and $\kappa$ for $f(x) = 3x_1^2 + x_1x_2 + 3x_2^2$?

<details><summary>answer</summary>

$H = \begin{pmatrix}6 & 1\\ 1 & 6\end{pmatrix}$ with eigenvalues 7 and 5, so $L = 7$, $\mu = 5$, $\kappa = 1.4$: a well-conditioned problem.

</details>

3. (the predict callout in idea 3.) what's the descent-lemma gap for $\|x\|^2$ with $L = 2$?

<details><summary>answer</summary>

exactly 0 for all $x$, $y$: $\|x\|^2 + 2x\cdot(y - x) + \|y - x\|^2 = \|y\|^2$. the quadratic upper bound of a quadratic with the right $L$ is the function itself.

</details>

4. why does the step-size limit depend on the *largest* curvature, while the convergence speed depends on the *smallest*?

<details><summary>answer</summary>

stability needs every eigen-coordinate to shrink, and the most curved one is the first to overshoot, so $\eta < 2/\lambda_{\max}$. once $\eta$ is capped at about $1/L$, the flattest direction moves by only a factor $1 - \eta\mu \approx 1 - \mu/L$ per step, so it sets the speed. the ratio $\kappa = L/\mu$ is the price of using one step size for all directions.

</details>

5. (the predict callout in idea 4.) what happens when you trust a sampled $\hat L$ in 50 dimensions?

<details><summary>answer</summary>

it diverges. the sampled estimate comes out far below the true $L = 100$ (about 45 in experiment 2), because random directions are almost orthogonal to the single sharp direction. a step of $1.9/\hat L$ is then about $4/L$, beyond the stability limit. the Hessian's eigenvalues find the sharp direction exactly.

</details>

6. gradient clipping caps $\|\nabla f\|$ at some threshold. which constant does it control, and why does it help with exploding losses?

<details><summary>answer</summary>

it caps the length of each step at $\eta\cdot$threshold, an effective Lipschitz bound on the update. where the loss surface has a cliff (huge local curvature, so steps far beyond $2/L_{\text{local}}$), the step can't catapult the parameters arbitrarily far.

</details>

7. why can't gradient descent with a fixed step converge on $|x|$?

<details><summary>answer</summary>

the gradient is $\pm1$ right up to the minimum, so each step has length exactly $\eta$, and the iterate ends up bouncing around 0 at a distance of order $\eta$. convergence needs either shrinking steps ($\eta_k \to 0$, the subgradient method) or a smooth approximation (Huber, softplus) with a step below $2/L$.

</details>

## common bugs

- **comparing only neighbouring points.** symptom: the Lipschitz estimate depends on the order of the rows. use all pairs $i < j$.
- **dividing by zero on duplicate points.** symptom: `nan` or `inf`. mask out pairs with zero distance (and the diagonal).
- **a norm instead of a dot product for $\mu$.** symptom: the strong-convexity estimate is never negative, even for non-convex functions. the chord curvature is $(g_i - g_j)\cdot(x_i - x_j)/\|x_i - x_j\|^2$.
- **the largest eigenvalue instead of the largest absolute eigenvalue.** symptom: $L$ is underestimated near saddles and maxima.
- **calling an oscillation stable.** symptom: `largest_stable_step` returns exactly $2/L$. at $2/L$ the iterates bounce forever; demand real progress, not just "no explosion".
- **modifying `x0` in place.** symptom: the second run of gradient descent starts somewhere else. copy before stepping.

## going deeper

- Nesterov (2018), *Lectures on Convex Optimization*, 2nd ed., Springer, §2.1. smooth and strongly convex functions, the descent lemma, and the optimal rates. [doi:10.1007/978-3-319-91578-4](https://doi.org/10.1007/978-3-319-91578-4)
- Bubeck (2015), "Convex optimization: algorithms and complexity", *Foundations and Trends in Machine Learning* 8(3–4). a concise path from smoothness to convergence rates. [arXiv:1405.4980](https://arxiv.org/abs/1405.4980)
- Miyato, Kataoka, Koyama & Yoshida (2018), "Spectral normalization for generative adversarial networks", *ICLR*. controlling a network's Lipschitz constant layer by layer. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
- Cohen, Kaur, Li, Kolter & Talwalkar (2021), "Gradient descent on neural networks typically occurs at the edge of stability", *ICLR*. what happens to the local curvature when you train above $2/L$. [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)

## connections

- ← builds on: Hessians and their eigenvalues (topic 02), Taylor's theorem along a line (topic 03), linear convergence (topic 01), condition numbers (ch 02 · topic 08).
- → used in: convexity, gradient descent rates and line searches (ch 06), learning-rate finders and gradient clipping (ch 15), Lipschitz bounds and margins in learning theory (ch 13), Wasserstein GANs and spectral normalization (ch 21), adversarial robustness (ch 28).
