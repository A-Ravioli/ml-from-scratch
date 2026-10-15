# nonconvex optimization

## why this matters

everything that made the earlier topics clean (every stationary point is a global minimum, rates are guarantees, the answer doesn't depend on where you start) came from convexity. the losses you'll spend most of this course minimizing, those of neural networks, aren't convex. topic 01's lab already found that a single tanh unit, a two-layer linear network and matrix factorization all fail the convexity test. so what *can* go wrong when you run gradient descent on a nonconvex function, and how much of it matters?

three things, it turns out. **local minima**: valleys that aren't the deepest, where gradient descent stops for good. **saddle points**: flat spots that are downhill in some direction, where gradient descent can slow to a crawl. and **bad geometry** that isn't strictly about minima at all: curved, narrow valleys where the path is long. this topic builds the two classic test functions that isolate these (Rastrigin's egg carton and Rosenbrock's banana valley), a gradient method that provably escapes saddles by adding noise, and the tools researchers use to look at landscapes in millions of dimensions: 2-D slices, and the plane an optimizer's path actually moved in. the lab ends with a warning about what those pictures can and can't show.

## warm-up: predict first

1. gradient descent from $(1, 0)$ on $f(x, y) = \tfrac12x^2 - \tfrac12y^2 + \tfrac14y^4$. where does it converge?
2. the same, from $(1, 10^{-6})$. roughly how many steps (with step 0.1) until it has clearly left the origin?
3. you add small random noise whenever the gradient is small. does that help gradient descent out of a local minimum? out of a saddle?
4. Rosenbrock's function in 2-D has exactly one local minimum. how many does it have in 6-D?
5. you plot a 2-D slice of a 1,000-dimensional loss along two random directions and see a smooth bowl. is the function nearly convex?

## prerequisites check

- [ ] you can classify a critical point by the signs of its Hessian's eigenvalues (→ ch 03 · topic 02)
- [ ] you know why gradient descent on a quadratic multiplies each eigen-coordinate by $1 - \eta\lambda_i$ (→ ch 03 · topic 05)
- [ ] you've built `gradient_descent`, `OptimResult` and `lbfgs`, and know what convexity guarantees (→ topics 01, 02, 05)
- [ ] you can sample from a distribution by inverting its CDF (→ ch 04 · topic 02)
- [ ] you can compute an SVD and know that its right singular vectors are the principal directions of the rows (→ ch 02 · topic 06)

## core ideas

### idea 1: two landscapes: the egg carton and the banana

two functions have become the standard first test for any optimizer, because each isolates one difficulty.

**Rastrigin**, $f(x) = An + \sum_{i=1}^n\big(x_i^2 - A\cos(2\pi x_i)\big)$ with $A = 10$, is a bowl with an egg carton pressed into it: a global minimum 0 at the origin, and a local minimum near every point with integer coordinates. in $[-5.12, 5.12]^n$ that's about $11^n$ local minima. its gradient, $2x + 2\pi A\sin(2\pi x)$, is a sum of independent coordinate terms, so gradient descent simply rolls into the nearest dimple: **local minima are the problem**.

**Rosenbrock**, $f(x) = \sum_{i=1}^{n-1}\big[b(x_{i+1} - x_i^2)^2 + (a - x_i)^2\big]$ with $a = 1$, $b = 100$, has its minimum 0 at $x = (1, \dots, 1)$, at the end of a narrow parabolic valley $x_{i+1} \approx x_i^2$. in 2-D it has no other minimum, and its trouble is geometric: the valley floor is curved and nearly flat, the walls are steep (curvature around 200 across, about 1 along), and gradient descent zig-zags across instead of following the bend. it's nonconvex (its Hessian is indefinite at $(0, 1)$, on paper 1), but in 2-D every descent path ends at the right answer. in 4 to 7 dimensions a second, local minimum appears near $(-1, 1, \dots, 1)$ (warm-up 4: two; the lab finds it).

the gradient of the chained Rosenbrock needs care: each middle coordinate $x_i$ appears in two terms, as the "$x_i$" of term $i$ and as the "$x_{i+1}$" of term $i - 1$, and both contributions add.

> **▶ build: stage 1.** implement `rosenbrock`, `rosenbrock_grad`, `rastrigin` and `rastrigin_grad`, vectorized for any $n$.

### idea 2: saddles, and escaping them with noise

a critical point with a negative-curvature direction ($\lambda_{\min}(\nabla^2f) < 0$) is a **strict saddle**. near it, gradient descent acts like it does on the quadratic model: along an eigenvector with eigenvalue $-\gamma < 0$, the coordinate is multiplied by $1 + \eta\gamma$ each step, so it *grows*, and the iterate escapes. but it grows from wherever it starts: from a distance $\delta$ off the saddle's "stable manifold" (the set of points that flow into it), escaping to distance $c$ takes about

$$k \approx \frac{\ln(c/\delta)}{\ln(1 + \eta\gamma)}\ \text{steps},$$

and from *exactly* on it ($\delta = 0$), forever (warm-up 1: from $(1, 0)$ the $y$-coordinate stays 0 and gradient descent converges to the saddle at the origin, a point that isn't a minimum at all). from a random start the stable manifold has probability zero, so plain gradient descent almost surely avoids saddles (Lee et al., 2016). but "eventually" can be very slow: each nearby saddle can cost $\ln(1/\delta)/(\eta\gamma)$ steps, and chains of saddles can cost exponential time (Du et al., 2017). and in high dimensions saddles are far more common than local minima: a critical point is a minimum only if *all* $n$ curvatures are positive (Dauphin et al., 2014).

the fix is cheap: when the gradient is small (you might be near a saddle), add a random kick. **perturbed gradient descent** (Jin et al., 2017) does

$$x \leftarrow \begin{cases}x + \xi,\ \ \xi \sim \text{Uniform}(\text{ball of radius } r) & \text{if } \|\nabla f(x)\| < g_{\text{thr}}\\ x - \eta\nabla f(x) & \text{otherwise.}\end{cases}$$

a kick puts the iterate off the stable manifold with probability 1, at a typical distance $\sim r$, and the negative curvature does the rest in $O(\ln(1/r)/(\eta\gamma))$ steps. near a true minimum the kicks just jiggle the iterate (it rolls back each time), so the method keeps the **best** iterate seen rather than the last. one detail: "uniform in the ball" is not "uniform direction times uniform radius". the fraction of the ball's volume within radius $\rho$ is $(\rho/r)^n$, so by inverse-CDF sampling the radius is $r\,U^{1/n}$ (on paper 3); in high dimensions almost all of the ball is near its surface.

what noise does *not* do is get you out of a genuine local minimum, unless the kicks are big enough to jump to another basin (warm-up 3: no for local minima with small kicks, yes for saddles). that's the lab's first experiment.

> **🔮 predict:** from $(1, 10^{-6})$ with step 0.1 on the saddle function above, after how many steps is $|y| > 0.5$?

> **▶ build: stage 2.** implement `perturbed_gradient_descent(f, grad, x0, *, lr, noise_radius, grad_threshold, max_iters, rng, callback)`: kick or step, then record; return the best iterate.

### idea 3: looking at a landscape: 2-D slices

you can't look at a function of a million variables, but you can look at it on a plane: pick a center $x_0$ and two directions $d_1, d_2$, and evaluate

$$Z_{ij} = f(x_0 + \alpha_id_1 + \beta_jd_2)$$

on a grid of $(\alpha_i, \beta_j)$. this is how most pictures of neural-network "loss landscapes" are made (Goodfellow et al., 2015, used 1-D slices between initialization and solution; Li et al., 2018, used 2-D slices). the picture is honest about the plane and silent about everything else, so the choice of directions is everything:

- **coordinate directions** show how $f$ depends on one or two parameters: fine in 2-D, meaningless in a million.
- **random directions** are the default, but in high dimensions they're almost orthogonal to anything special (ch 02 · topic 01). a random unit vector has coordinates of size about $1/\sqrt n$, so moving a distance $\alpha$ along it moves each coordinate by only about $\alpha/\sqrt n$: features at the scale of 1 in each coordinate become invisible unless $\alpha$ is about $\sqrt n$. the lab shows what that does to Rastrigin.
- **directions that matter**: the line between two solutions (to ask whether they're connected by a low path, ch 19), or the directions an optimizer actually moved in (idea 4).

the grid's orientation is a convention worth fixing: rows follow $d_1$, columns follow $d_2$, so `Z[i, j]` is the loss at $(\alpha_i, \beta_j)$, and a contour plot with $\alpha$ on the horizontal axis needs `Z.T`.

> **▶ build: stage 3.** implement `loss_landscape_2d(f, x0, d1, d2, *, span, resolution)`, returning `(alphas, betas, Z)`.

### idea 4: the plane a trajectory lives in

an optimizer's path through a million-dimensional space is, surprisingly often, nearly two-dimensional: it moves mostly along a few directions. principal component analysis finds them. stack the iterates $x_0, \dots, x_K$, center them at the final point ($D_k = x_k - x_K$), and take the top two right singular vectors of $D$ (ch 02 · topic 06) as $d_1, d_2$. the coordinates $D[d_1\ d_2]$ place every iterate in that plane, and $s_i^2/\sum_js_j^2$ says what fraction of the path's (squared) movement each direction explains. centering at the *final* point, not the mean, makes the plane pass through the solution, so a 2-D slice through $x_K$ along $d_1, d_2$ shows the path on its actual landscape (Li et al., 2018). if the two directions explain 95% of the movement, the picture shows 95% of what the optimizer did, and nothing about the other $n - 2$ directions, which is where, in a nonconvex problem, the other minima live.

> **🔮 predict:** gradient descent takes 3,000 steps on a 6-dimensional function. how many PCA directions do you need to capture 99% of its path's movement?

> **▶ build: stage 4.** implement `trajectory_pca(xs)`, returning the two directions, the path's coordinates in their plane, and the explained fractions.

## worked example

**the saddle function** $f(x, y) = \tfrac12x^2 - \tfrac12y^2 + \tfrac14y^4$. its gradient is $(x,\ -y + y^3)$, zero at $(0, 0)$ and $(0, \pm1)$. the Hessian is $\operatorname{diag}(1, -1 + 3y^2)$: $\operatorname{diag}(1, -1)$ at the origin (a strict saddle, curvature $-1$ along $y$) and $\operatorname{diag}(1, 2)$ at $(0, \pm1)$ (minima, with $f = -\tfrac14$).

gradient descent with $\eta = 0.1$ multiplies $x$ by $0.9$ each step, and near the origin multiplies $y$ by about $1 + 0.1 = 1.1$. from $(1, 0)$: $y$ stays exactly 0 and $x_k = 0.9^k \to 0$, so the run converges to the saddle with $f = 0$ and a vanishing gradient, reporting `converged=True` at a point that isn't a minimum. from $(1, 10^{-6})$: $|y|$ reaches $0.5$ after about $\ln(0.5/10^{-6})/\ln 1.1 \approx 138$ steps (your code takes 139, since the cubic term only matters near the end), and then it slides to $(0, \pm1)$. perturbed gradient descent from $(1, 0)$ with $r = 0.01$: as soon as $\|\nabla f\| < g_{\text{thr}}$ near the origin, a kick of size $\le 0.01$ moves $y$ off zero, and about $\ln(0.5/0.01)/\ln 1.1 \approx 40$ steps later it's in a minimum.

**test-function values.** $\text{Rosenbrock}(-1.2, 1) = 100(1 - 1.44)^2 + (1 + 1.2)^2 = 19.36 + 4.84 = 24.2$. $\text{Rastrigin}(1, 1) = 20 + 2 - 20\cos 2\pi = 2$ (a local minimum is *near* $(1, 1)$, where $2x + 20\pi\sin 2\pi x = 0$, at about $x \approx 0.995$), and $\text{Rastrigin}(0.5) = 10 + 0.25 + 10 = 20.25$ (a local maximum between dimples).

**uniform in a disk.** in 2-D, the fraction of the disk of radius $r$ within radius $\rho$ is $(\rho/r)^2$, so half the samples should have $\|\xi\| \le r/\sqrt2 \approx 0.71r$, not $\le 0.5r$. sampling the radius as $rU$ instead of $r\sqrt U$ would put half of them inside $0.5r$, crowding the center.

## on paper

1. **Rosenbrock's Hessian.** for $n = 2$, show $\nabla^2f = \begin{pmatrix}2 - 400(y - 3x^2) & -400x\\-400x & 200\end{pmatrix}$. evaluate it at $(1, 1)$ and at $(0, 1)$. what do you conclude about convexity and about the conditioning of the minimum?

<details><summary>solution</summary>

$\partial_xf = -400x(y - x^2) - 2(1 - x)$, $\partial_yf = 200(y - x^2)$; differentiating again gives the matrix. at $(1, 1)$: $\begin{pmatrix}802 & -400\\-400 & 200\end{pmatrix}$, positive definite (determinant $802\cdot200 - 160000 = 400 > 0$) but with eigenvalues about $1001.6$ and $0.4$: condition number about 2,500, which is why gradient descent crawls along the valley. at $(0, 1)$: $\operatorname{diag}(-398, 200)$, indefinite, so $f$ is not convex.

</details>

2. **escape time.** near a strict saddle at 0, linearize gradient descent as $x_{k+1} = (I - \eta H)x_k$ with $H$ having eigenvalue $-\gamma$ along unit vector $v$. show that $v^\top x_k = (1 + \eta\gamma)^kv^\top x_0$, and derive the number of steps to reach $|v^\top x| = c$ from $|v^\top x_0| = \delta$. why does the dependence on $\delta$ make perturbations so effective?

<details><summary>solution</summary>

$v^\top x_{k+1} = v^\top x_k - \eta v^\top Hx_k = (1 + \eta\gamma)v^\top x_k$, since $v^\top H = -\gamma v^\top$. so $|v^\top x_k| = (1 + \eta\gamma)^k\delta$, which reaches $c$ at $k = \ln(c/\delta)/\ln(1 + \eta\gamma) \approx \ln(c/\delta)/(\eta\gamma)$. the cost is *logarithmic* in $1/\delta$: a kick of size $r$ turns an unbounded wait ($\delta = 0$) or a long one ($\delta = 10^{-12}$) into $\ln(c/r)/(\eta\gamma)$ steps, for any $r$ that isn't absurdly small.

</details>

3. **uniform in a ball.** a point uniform in the $n$-ball of radius $r$ has $P(\|\xi\| \le \rho) = (\rho/r)^n$. show that $\rho = rU^{1/n}$ with $U \sim \text{Uniform}(0, 1)$ has this distribution, and that combining it with an independent uniform direction (a normalized standard normal) gives the uniform distribution on the ball. what fraction of a 100-dimensional ball lies outside 95% of its radius?

<details><summary>solution</summary>

$P(rU^{1/n} \le \rho) = P(U \le (\rho/r)^n) = (\rho/r)^n$. the uniform distribution on the ball is rotation invariant, so given its radius the direction is uniform on the sphere, which is what a normalized standard normal vector gives (its density depends only on $\|u\|$). $1 - 0.95^{100} \approx 0.994$: more than 99% of the volume is in the outer 5% shell.

</details>

4. **why random slices smooth things out.** for a unit vector $d$ with independent random coordinates (each about $N(0, 1/n)$), consider one Rastrigin term $\cos(2\pi(x_i + \alpha d_i))$. over $\alpha \in [-s, s]$, how many periods does it go through? for $n = 1{,}000$ and $s = 3$, what does that tell you about the slice?

<details><summary>solution</summary>

the argument changes by $2\pi\cdot2s|d_i|$, i.e. $2s|d_i|$ periods, with $|d_i| \approx 1/\sqrt n$: about $6/\sqrt{1000} \approx 0.19$ of a period. every term is nearly linear (or gently curved) over the whole slice, so their sum is a smooth function: the egg carton is invisible. to see a full period along a typical coordinate you need $s \approx \sqrt n/2 \approx 16$; the lab's span-3 slice shows no local minima, while a span-30 slice of the same function shows about ten.

</details>

## experiment

`lab.py` investigates three things with your code:

1. gradient descent and perturbed gradient descent on 2-D Rastrigin from 100 random starts: how often they find the global minimum.
2. L-BFGS on 6-D Rosenbrock from 60 random starts, and one gradient-descent path drawn on the plane your `trajectory_pca` finds.
3. 2-D slices of Rastrigin along random directions, in 2 dimensions and in 1,000.

## check your understanding

1. why does gradient descent almost never converge to a strict saddle from a random start, and why is that not the end of the story?

<details><summary>answer</summary>

the set of starting points that flow into a strict saddle (its stable manifold) has lower dimension, so probability zero under a random start (Lee et al., 2016). but nearby starts spend a long time near the saddle, $\approx\ln(1/\delta)/(\eta\gamma)$ steps, and a path can pass many saddles in sequence; in the worst case gradient descent needs exponential time (Du et al., 2017). noise turns "almost surely, eventually" into "with high probability, quickly".

</details>

2. (the predict callout in idea 2.) from $(1, 10^{-6})$, when does $|y|$ pass 0.5?

<details><summary>answer</summary>

after about $\ln(5\times10^5)/\ln1.1 \approx 138$ steps (139 in practice). from $10^{-12}$ it would be about 283: halving the exponent of $\delta$ doubles the wait, but it's still only logarithmic.

</details>

3. why does perturbed gradient descent return the best iterate rather than the last?

<details><summary>answer</summary>

near a minimum the gradient is small, so the method keeps kicking the iterate away by up to $r$ and letting it roll back: the last iterate is a random point near the minimum, while the best one seen is the closest to it. (the original algorithm instead stops after a kick fails to decrease $f$ by enough within a set number of steps, which certifies an approximate local minimum.)

</details>

4. in high dimensions, why are saddles more common than local minima?

<details><summary>answer</summary>

a critical point is a local minimum only if all $n$ Hessian eigenvalues are positive. for "generic" critical points (in random-function models, Dauphin et al., 2014), the probability that all $n$ signs come out positive shrinks rapidly with $n$, especially at high loss values; most critical points have some negative curvature. near the bottom of the landscape the balance shifts towards minima, and those tend to have similar losses.

</details>

5. a 2-D slice through the final weights of a network along two random directions shows a wide, smooth bowl. what can you conclude?

<details><summary>answer</summary>

that the loss rises smoothly along *those two* directions near that point, nothing more. random directions miss low-dimensional structure (narrow valleys, other minima), and features at the scale of single weights are smoothed out (on paper 4). claims about "flatness" from such pictures also depend on how the directions are scaled; Li et al. (2018) normalize them filter by filter for that reason.

</details>

6. why might an optimizer's path be explained almost entirely by two PCA directions, even in a million dimensions?

<details><summary>answer</summary>

gradient methods are dominated by the few directions with the largest consistent gradients (the top eigen-directions of the curvature and the drift towards the solution); along most of the remaining directions the iterates barely move. the path is long but low-dimensional. the flip side is that a 2-D picture of the path says nothing about the directions it didn't explore.

</details>

7. (the predict callout in idea 4.) how many directions for 99% of a 3,000-step path in 6-D?

<details><summary>answer</summary>

often just one or two. in lab experiment 2, on 6-D Rosenbrock, the first direction alone explains 99.8%: a short drop down the valley wall and a long crawl along the floor. paths are long but low-dimensional (question 6 explains why).

</details>

## common bugs

- **one term too few in the Rosenbrock gradient.** symptom: the gradient check fails in the middle coordinates only. each $x_i$ with $0 < i < n - 1$ appears in two terms.
- **noise on the sphere, or in a cube.** symptom: the radius histogram is wrong (all kicks the same size, or the corners overweighted). use a normalized normal vector times $rU^{1/n}$.
- **noise with `np.random`.** symptom: runs aren't reproducible from `rng`. use `as_rng(rng)`.
- **returning the last iterate of perturbed gradient descent.** symptom: the reported value jumps around from run to run near a minimum. keep the best.
- **`Z` transposed.** symptom: the landscape looks mirrored across the diagonal relative to the path. `Z[i, j]` is at $(\alpha_i, \beta_j)$; plot `Z.T` with $\alpha$ horizontal.
- **centering the trajectory at its mean.** symptom: the PCA plane doesn't pass through the solution, so the path and the landscape don't line up. center at the final iterate.

## going deeper

- Jin, Ge, Netrapalli, Kakade & Jordan (2017), "How to escape saddle points efficiently", *ICML*. perturbed gradient descent and its guarantees. [arXiv:1703.00887](https://arxiv.org/abs/1703.00887)
- Lee, Simchowitz, Jordan & Recht (2016), "Gradient descent only converges to minimizers", *COLT*. [arXiv:1602.04915](https://arxiv.org/abs/1602.04915)
- Du, Jin, Lee, Jordan, Póczos & Singh (2017), "Gradient descent can take exponential time to escape saddle points", *NeurIPS*. [arXiv:1705.10412](https://arxiv.org/abs/1705.10412)
- Dauphin, Pascanu, Gulcehre, Cho, Ganguli & Bengio (2014), "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization", *NeurIPS*. [arXiv:1406.2572](https://arxiv.org/abs/1406.2572)
- Li, Xu, Taylor, Studer & Goldstein (2018), "Visualizing the loss landscape of neural nets", *NeurIPS*. filter-normalized slices and trajectory PCA. [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
- Goodfellow, Vinyals & Saxe (2015), "Qualitatively characterizing neural network optimization problems", *ICLR*. 1-D interpolation slices. [arXiv:1412.6544](https://arxiv.org/abs/1412.6544)

## connections

- ← builds on: classifying critical points with the Hessian (ch 03 · topic 02), gradient descent on quadratics (ch 03 · topic 05), convexity and what it guarantees (topic 01), `gradient_descent` and `lbfgs` (topics 02 and 05), inverse-CDF sampling (ch 04 · topic 02), the SVD (ch 02 · topic 06).
- → used in: the capstone's benchmark (Rosenbrock and Rastrigin), training neural networks (ch 14–15), loss-landscape slices and mode connectivity in `mlfs.dltheory` (ch 19), and the nonconvex objectives of matrix factorization and mixture models (ch 11).
