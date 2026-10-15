# natural gradient

## why this matters

fit a Gaussian to data by gradient descent, once with the standard deviation $\sigma$ as a parameter and once with $\log\sigma$. same model, same data, same step size. in this topic's lab, at step size 0.5 the $\log\sigma$ run oscillates and the $\sigma$ run drives $\sigma$ negative and dies; at step size 0.1 the two end up in different places after 60 steps. nothing about the *model* differs between the runs. what differs is the description, and gradient descent can't tell the two apart.

the reason is that "steepest descent" silently assumes a way of measuring step length, and gradient descent uses the Euclidean length of the parameter vector, which changes whenever you reparameterize. topic 18 gave you a better ruler: the Fisher metric, which measures a step by how much it changes the model's distribution and doesn't care about coordinates. steepest descent under that ruler is the **natural gradient** (Amari, 1998). it is invariant to how you parameterize the model, it is Newton's method for many classical models, it is asymptotically efficient as an online estimator, and it's the idea behind TRPO (ch 24), K-FAC (topic 20) and a good part of the second-order optimization literature. its catch is cost: the Fisher of a network with $P$ parameters is a $P\times P$ matrix. this topic builds natural gradient descent exactly for small models and matrix-free for networks, and the lab measures what it buys and what it costs.

## warm-up: predict first

1. you fit a Gaussian by gradient descent with $\sigma$ in metres, then again with $\sigma$ in centimetres (same data, same step size). do you get the same sequence of models?
2. the loss is $L(\theta) = g^\top\theta$ near the current point, and you may take any step $d$ with $d^\top Md \le r^2$ for a positive-definite $M$. is the best step parallel to $-g$?
3. what does a natural-gradient step with step size 1 do on least-squares linear regression?
4. a network has 25,000 parameters. how much memory does its Fisher matrix take in float64, and roughly how long would inverting it take?
5. Adam divides each gradient coordinate by the square root of a running average of its square. is that a natural-gradient method?

## prerequisites check

- [ ] you know the Fisher metric, its pullback under reparameterization, and the empirical Fisher (→ topic 18)
- [ ] you know Newton's method and why it needs a Hessian solve (→ ch 06 · topic 05)
- [ ] you know conjugate gradients, or at least symmetric positive-definite systems (→ ch 24 · topic 04, ch 02 · topic 03)
- [ ] you can use `torch.autograd.grad`, including `grad_outputs` and `create_graph` (→ ch 15)

## core ideas

### idea 1: steepest descent depends on how you measure a step

near the current parameters the loss is approximately linear, $L(\theta + d) \approx L(\theta) + g^\top d$ with $g = \nabla L(\theta)$. "the steepest direction" means: among all steps of a given length, the one that decreases this linear model most. with length measured by a positive-definite matrix $M$, that's

$$\min_d\ g^\top d\quad\text{subject to}\quad d^\top Md = r^2,\qquad\text{solved by}\qquad d^* = -r\,\frac{M^{-1}g}{\sqrt{g^\top M^{-1}g}}$$

(on paper 1). with $M = I$ you recover gradient descent's direction $-g/\lVert g\rVert$. with any other $M$ the best step is *not* parallel to the gradient (warm-up 2): it leans towards directions where steps are cheap.

the gradient itself is not a direction in any coordinate-free sense. if you rescale a coordinate, $\theta_1 = c\,\psi_1$, the partial derivative transforms as $\partial L/\partial\psi_1 = c\,\partial L/\partial\theta_1$, the opposite of how a displacement transforms ($d\psi_1 = d\theta_1/c$). gradient descent adds the two as if they were the same kind of object, so its updates have the wrong units whenever coordinates aren't all alike (warm-up 1: with $\sigma$ in centimetres, the gradient-descent step on $\sigma$, converted back to metres, is $10^4$ times smaller than with $\sigma$ in metres). the matrix $M^{-1}$ converts a gradient into a displacement, and the result is consistent across coordinate systems whenever $M$ transforms like a metric, $M_\psi = J^\top M_\theta J$ (on paper 2).

the **natural gradient** chooses $M = F$, the Fisher metric:

$$\tilde\nabla L = F(\theta)^{-1}\nabla L(\theta).$$

equivalently, it's the minimizer of the linear model plus a KL penalty, $\min_d\ g^\top d + \frac1\eta\cdot\tfrac12d^\top Fd$, which gives $d = -\eta F^{-1}g$: the best step among those that don't move the model's *distribution* too far, the same trust-region idea as TRPO.

> **spec.** `steepest_direction(grad, metric, radius)`: `grad` `(P,)`, `metric` `(P, P)` symmetric positive definite, `radius` a float → $d^*$ above, `(P,)`. `natural_gradient(grad, fisher, *, damping=0.0)` → the solution of $(F + \text{damping}\cdot I)x = g$, by a linear solve (don't form an inverse, and don't modify `fisher`).

> **▶ build: stage 1.** implement `steepest_direction` and `natural_gradient`.

### idea 2: natural gradient descent, and what's invariant

natural gradient descent iterates

$$\theta_{k+1} = \theta_k - \eta\,\big(F(\theta_k) + \lambda I\big)^{-1}\nabla L(\theta_k).$$

**invariance.** if you change coordinates by an *affine* map $\theta = A\psi + b$, the gradient pulls back as $A^\top g$ and the Fisher as $A^\top FA$, so the step in $\psi$ is $(A^\top FA)^{-1}A^\top g = A^{-1}F^{-1}g$, which pushed through $A$ is exactly the step in $\theta$. the iterates trace the same models at every step size (on paper 2). for a smooth *nonlinear* reparameterization ($\sigma$ versus $\log\sigma$) the step direction is still the same at each point, but a finite step is a straight line in whichever coordinates you use, and straight lines in different coordinates are different curves. so the discrete iterates agree to first order in $\eta$ per step, and exactly in the limit $\eta \to 0$: the **natural gradient flow** $\dot\theta = -F^{-1}\nabla L$ is a coordinate-free object, and NGD with a small step follows it in any coordinates. damping breaks invariance a little too, since $\lambda I$ isn't a metric that transforms correctly; keep $\lambda$ small relative to $F$.

**it's Newton's method, often.** for the negative log-likelihood of an exponential family in its natural parameters, or a generalized linear model with its canonical link (softmax regression, logistic regression, least squares), the Fisher equals the Hessian of the average loss (topic 18, idea 3). NGD with $\eta = 1$ is then Newton's method, which statisticians call **Fisher scoring** (warm-up 3: one step solves least squares). for a Gaussian in $(\mu, \log\sigma)$ coordinates the natural step on $\mu$ is exactly $\bar x - \mu$, so with $\eta = 1$ the mean lands on the sample mean in one step. for nonlinear models the Fisher is the generalized Gauss–Newton matrix, a positive semidefinite stand-in for the Hessian that is often *better* than the Hessian as a preconditioner, because it never points uphill.

**efficiency.** Amari (1998) showed that online NGD on i.i.d. data, with step size $1/t$, gives estimates whose covariance reaches the Cramér–Rao bound asymptotically (ch 05): it extracts as much information per example as the maximum-likelihood estimator does, which no plain SGD with a scalar step size can do in general.

**damping.** $F$ is singular for networks (symmetries, topic 18's softmax shift) and the quadratic model behind the step only holds locally (topic 18's lab: the KL ran 2.5× above the quadratic form at large steps). adding $\lambda I$ fixes both: it makes the solve well posed and shrinks the step in directions where $F$ is small and the quadratic model can't be trusted. it's the Levenberg–Marquardt idea, and choosing $\lambda$ well is half the engineering of every practical natural-gradient method.

> **🔮 predict:** you run NGD on the same Gaussian in $(\mu, \sigma)$ and in $(\mu, \log\sigma)$ coordinates, step size 0.01, 600 steps. by how much do the two runs' $\sigma$ values differ at worst along the way? (gradient descent's runs differ by about 0.2.)

> **spec.** `natural_gradient_descent(grad_fn, fisher_fn, theta0, *, lr, n_steps, damping=0.0)`: `grad_fn(theta)` → `(P,)`, `fisher_fn(theta)` → `(P, P)`, `theta0` `(P,)` (don't modify it). takes `n_steps` steps of the update above (with your `natural_gradient`) and returns the path, `(n_steps + 1, P)`, starting with `theta0`.

> **▶ build: stage 2.** implement `natural_gradient_descent`.

### idea 3: never form the Fisher

the network Fisher is $F = \frac1N\sum_nJ_n^\top H_nJ_n$ with $J_n$ the logit Jacobian and $H_n = \operatorname{diag}(p_n) - p_np_n^\top$ (topic 18). for 25,000 parameters, $F$ has $6\cdot10^8$ entries (5 GB in float64) and a dense solve costs about $P^3/3 \approx 5\cdot10^{12}$ flops (warm-up 4). but you never need $F$, only $F$ times a vector, and that factorizes:

$$Fv = \frac1N\sum_nJ_n^\top\Big(H_n\,(J_nv)\Big).$$

three cheap pieces. **$J v$**, the change in every logit when the parameters move along $v$, is a forward-mode derivative (a Jacobian–vector product). reverse-mode autograd can get it with the **double-backward trick**: $u(w) = J^\top w$ is what `torch.autograd.grad(logits, params, grad_outputs=w, create_graph=True)` returns, it's linear in $w$, and $\nabla_w\big(u(w)\cdot v\big) = Jv$ (on paper 4). **$H(Jv)$** is $p \odot u - p\,(p^\top u)$ row by row. **$J^\top(\cdot)$** is an ordinary backward pass with `grad_outputs`. altogether a Fisher-vector product costs a forward pass plus about three backward-sized passes (the double backward through the first backward counts as two).

then solve $(F + \lambda I)x = g$ with **conjugate gradients**, which needs only matrix–vector products. CG minimizes the quadratic $\tfrac12x^\top Ax - b^\top x$ along a sequence of $A$-conjugate directions; after $k$ iterations its iterate is optimal over a $k$-dimensional Krylov subspace, it is exact after at most $P$ iterations, and with a good spread of eigenvalues 10–50 iterations get close enough. stopping early is itself a kind of damping: the first CG iterates move mostly along directions of large curvature. this combination, Gauss–Newton products plus truncated CG, is Martens' (2010) **Hessian-free optimization**. you built the same CG for TRPO in chapter 24; here you'll write it again for flat vectors with an explicit residual tolerance.

> **spec.** `conjugate_gradient(mvp, b, *, max_iters=100, tol=1e-10)`: CG from $x_0 = 0$ for an SPD matrix given as `mvp(v) -> A v`. before each iteration, stop if the squared residual norm $r^\top r \le$ `tol` (so `b = 0` costs no products); otherwise one `mvp` call per iteration, at most `max_iters`. returns $x$. `fisher_vector_product(model, X, v, *, damping=0.0)`: `model` maps `(N, d)` → logits `(N, K)`; `v` is flat `(P,)` in `model.parameters()` order. returns $(F + \text{damping}\cdot I)v$ for the true Fisher (topic 18's `true_fisher`), flat and detached, without forming $F$, and without touching any parameter's `.grad`.

> **▶ build: stage 3.** implement `conjugate_gradient` and `fisher_vector_product`.

### idea 4: a natural-gradient step on a classifier

put the pieces together: compute the mean cross-entropy and its gradient $g$, solve $(F + \lambda I)\,\Delta = g$ with CG and your Fisher-vector product, and update $\theta \leftarrow \theta - \eta\Delta$. for softmax regression with $\lambda = 0$ and enough CG iterations this is exactly Newton's step (the Fisher is the Hessian), which the checks test. for an MLP it's a damped Gauss–Newton step.

why the *true* Fisher and not the cheaper empirical one? the empirical Fisher would make each step $\big(\frac1N\sum g_ng_n^\top\big)^{-1}g$, which depends on the residuals rather than the curvature: far from the optimum the residuals are large, so the steps are tiny, and near the optimum its scale is the data's noise level rather than the model's. the lab shows both failures on a two-parameter problem (Kunstner et al., 2019). Adam's preconditioner, the square root of a running average of the squared minibatch gradient (often loosely called a diagonal empirical Fisher, though it squares the minibatch *mean*), inherits these issues and adds the square root, which has no information-geometric justification at all (warm-up 5); Adam works for other reasons (Martens, 2020, discusses what can and can't be said about it).

how many steps natural gradient saves, and whether that survives the extra cost per step, is an empirical question, and the lab's last experiment measures it on MNIST.

> **🔮 predict:** each natural-gradient step with 10 CG iterations costs roughly 30–40 backward-sized passes. if natural gradient needs 10 times fewer steps than Adam to reach a given loss, which one gets there first in wall-clock time?

> **spec.** `ngd_step(model, X, y, *, lr=1.0, damping=1e-3, cg_iters=50)`: computes the mean cross-entropy of `model(X)` against integer labels `y` and its gradient $g$ (flat, `parameters()` order); solves $(F + \text{damping}\cdot I)\Delta = g$ with `conjugate_gradient(..., max_iters=cg_iters)` (default `tol`) and your `fisher_vector_product`; subtracts `lr` $\cdot\,\Delta$ from the parameters in place. returns the loss **before** the step as a Python float.

> **▶ build: stage 4.** implement `ngd_step`.

## worked example

**a steepest step.** $g = (1, -1)$, $M = \begin{pmatrix}2 & 0.5\\0.5 & 1\end{pmatrix}$, $r = 0.1$. $M^{-1}g = \frac{1}{1.75}(1.5, -2.5) = (0.857, -1.429)$ and $g^\top M^{-1}g = 4/1.75 = 2.286$, so $d^* = -0.1\,(0.857, -1.429)/\sqrt{2.286} = (-0.0567, 0.0945)$. it's not parallel to $-g = (-1, 1)$: it moves almost twice as far along the second coordinate, where the metric is cheaper. $(0.857, -1.429)$ is also the natural gradient if $M$ is a Fisher matrix.

**one natural step on a Gaussian.** data $(1, 2, 3, 6)$: $\bar x = 3$, and from $(\mu, \log\sigma) = (0, \log 2)$, $m_2 = \frac14\sum(x_i - 0)^2 = 12.5$. the mean NLL's gradient is $\big(-(\bar x - \mu)/\sigma^2,\ 1 - m_2/\sigma^2\big) = (-0.75, -2.125)$ and $F = \operatorname{diag}(1/\sigma^2, 2) = \operatorname{diag}(0.25, 2)$, so the natural gradient is $(-3, -1.0625)$. with $\eta = 1$: $\mu \to 3$ (the sample mean, in one step) and $\log\sigma \to \log 2 + 1.0625$. gradient descent with $\eta = 1$ would have moved $\mu$ only to 0.75.

**one CG iteration.** $A = \operatorname{diag}(1, 4, 10)$, $b = (1, 1, 1)$: from $x_0 = 0$ the first iterate is the exact line search along $b$, $x_1 = \frac{b^\top b}{b^\top Ab}\,b = \frac{3}{15}\,b = (0.2, 0.2, 0.2)$. three iterations reach the solution $(1, 0.25, 0.1)$ exactly (three distinct eigenvalues).

your code must reproduce these numbers.

## on paper

1. **the steepest step under a metric.** derive $d^*$ of idea 1 with a Lagrange multiplier, and show that it beats every other $d$ with $d^\top Md = r^2$.

<details><summary>solution</summary>

$\mathcal L = g^\top d + \frac\mu2(d^\top Md - r^2)$; stationarity gives $g + \mu Md = 0$, so $d = -M^{-1}g/\mu$. the constraint gives $g^\top M^{-1}g/\mu^2 = r^2$, so $\mu = \sqrt{g^\top M^{-1}g}/r$ (the positive root gives the minimum). for optimality, write $u = M^{1/2}d$, so $\lVert u\rVert = r$ and $g^\top d = (M^{-1/2}g)^\top u \ge -r\lVert M^{-1/2}g\rVert$ by Cauchy–Schwarz, with equality exactly at $u \propto -M^{-1/2}g$, which is $d^*$.

</details>

2. **affine invariance, and why gradient descent lacks it.** with $\theta = A\psi + b$ ($A$ invertible), show that one NGD step in $\psi$, mapped to $\theta$, equals one NGD step in $\theta$. then compute what one GD step in $\psi$ does to $\theta$.

<details><summary>solution</summary>

$\nabla_\psi L = A^\top g$ and $F_\psi = A^\top FA$ (pullback). the step in $\psi$ is $-\eta(A^\top FA)^{-1}A^\top g = -\eta A^{-1}F^{-1}A^{-\top}A^\top g = -\eta A^{-1}F^{-1}g$; mapped through $A$: $-\eta F^{-1}g$, the step in $\theta$. by induction the whole paths coincide. a GD step in $\psi$ is $-\eta A^\top g$, which moves $\theta$ by $-\eta AA^\top g$: gradient descent in $\theta$ preconditioned by $AA^\top$, which depends on the arbitrary choice of $A$.

</details>

3. **Fisher scoring is Newton for least squares.** for $L(w) = \frac1{2N}\lVert y - \Phi w\rVert^2$ (a Gaussian likelihood with unit noise), show that the Fisher is $\frac1N\Phi^\top\Phi$, equal to the Hessian, and that one NGD step with $\eta = 1$ from any $w$ lands on the least-squares solution.

<details><summary>solution</summary>

the model is $y_n \sim \mathcal N(\phi_n^\top w, 1)$, with score $(y_n - \phi_n^\top w)\phi_n$ for one example; under the model the residual has variance 1, so the per-example Fisher is $\phi_n\phi_n^\top$ and the average is $\frac1N\Phi^\top\Phi = \nabla^2L$. the step: $w - (\Phi^\top\Phi)^{-1}\Phi^\top(\Phi w - y) = (\Phi^\top\Phi)^{-1}\Phi^\top y$, the normal-equations solution, whatever $w$ was. (the *empirical* Fisher instead weights $\phi_n\phi_n^\top$ by the squared residual, which is the source of the lab's second experiment.)

</details>

4. **the double-backward trick.** let $u(w) = J^\top w$ for a fixed $K\times P$ matrix $J$. show $\nabla_w\big(u(w)^\top v\big) = Jv$. why does this need `create_graph=True` in the first `autograd.grad` call?

<details><summary>solution</summary>

$u(w)^\top v = w^\top Jv$, whose gradient in $w$ is $Jv$. in a network, $J$ depends on the parameters but not on $w$, and autograd computes $u(w)$ as a vector–Jacobian product; to differentiate that result with respect to $w$, the backward pass itself must be recorded as a graph, which is what `create_graph=True` does. the value of $w$ doesn't matter (use zeros), only the graph's dependence on it.

</details>

## experiment

`lab.py` investigates, with your natural-gradient code:

1. fitting one Gaussian in three coordinate systems, with gradient descent and with natural gradient descent, at several step sizes.
2. natural gradient with the Fisher versus the empirical Fisher on a two-parameter regression, when the model is right about the noise level and when it isn't.
3. natural gradient versus SGD and Adam on an MLP for MNIST, per step and per second.

## check your understanding

1. natural gradient is "invariant to parameterization". state precisely what is invariant, and what isn't.

<details><summary>answer</summary>

the step *direction* at a point, viewed as a tangent vector to the manifold of distributions, is invariant under any smooth reparameterization; so is the continuous-time flow $\dot\theta = -F^{-1}\nabla L$. discrete iterates with a finite step size are exactly invariant only under affine reparameterizations; under nonlinear ones they agree to first order in the step size. damping $\lambda I$ breaks invariance slightly, because $I$ isn't transformed with the coordinates.

</details>

2. (the predict callout in idea 2.) how far apart are the $(\mu, \sigma)$ and $(\mu, \log\sigma)$ NGD runs?

<details><summary>answer</summary>

about 0.001 at worst along the whole path with step size 0.01 (the checks require below 0.01), against about 0.2 for the two gradient-descent runs. halving the step size roughly halves the gap: it's a first-order discretization effect, not a difference in the method.

</details>

3. why does natural gradient need damping in practice, even when $F$ is invertible?

<details><summary>answer</summary>

$F^{-1}$ amplifies the gradient most in the directions where $F$ is smallest, which are exactly the directions where the quadratic approximation of the KL (and the curvature estimate from a minibatch) is least reliable. without damping, a tiny eigenvalue turns a small, noisy gradient component into a huge step. damping caps the amplification at $1/\lambda$.

</details>

4. for softmax regression, NGD with step size 1 is Newton's method. is that true for an MLP?

<details><summary>answer</summary>

no. for an MLP the Fisher is the generalized Gauss–Newton matrix, which drops the term of the Hessian that multiplies the residuals by the network's second derivatives. so NGD is a Gauss–Newton method: close to Newton near a good fit (small residuals), and better behaved than Newton far from it, because the GGN is never indefinite.

</details>

5. why is conjugate gradients a better fit here than, say, computing $F$ column by column and solving exactly?

<details><summary>answer</summary>

each column of $F$ costs one Fisher-vector product, so forming $F$ costs $P$ products and $P^2$ memory, then $O(P^3)$ to solve. CG gets a useful approximate solution from 10–50 products and $O(P)$ memory. its early iterates also act as regularization, concentrating on high-curvature directions, so truncating it is a feature, not only a shortcut.

</details>

6. you replace the true Fisher in `ngd_step` by the empirical Fisher of the minibatch. what goes wrong far from the optimum? at the optimum of a misspecified model?

<details><summary>answer</summary>

far from the optimum the residuals are large, the empirical Fisher is large, and the steps become tiny: progress slows down exactly when it should be fastest. at the optimum of a misspecified model the empirical Fisher's scale reflects the actual residual size rather than the model's own predicted variance, so the effective step size is wrong by that ratio: too large (bouncing around the optimum) if the model is overconfident, too small if it's underconfident. the lab shows both.

</details>

## common bugs

- **inverting instead of solving.** `torch.linalg.inv(F) @ g` is slower and less accurate than `torch.linalg.solve(F, g)`, and fails outright for singular $F$ where damping would have rescued a solve.
- **damping added in place.** `fisher += damping * I` modifies the caller's matrix; repeated calls keep adding to it.
- **CG without a residual check.** symptom: `nan` after the residual hits exactly zero (division by $p^\top Ap = 0$ or by $r^\top r = 0$).
- **`.backward()` inside the Fisher-vector product.** symptom: gradients accumulate in `.grad` across CG iterations and pollute the next step's loss gradient. use `torch.autograd.grad`.
- **forgetting `create_graph=True`** in the double-backward trick. symptom: an error that the output doesn't require grad, or a zero $Jv$.
- **returning the loss after the step.** the spec asks for the loss before it (that's the loss whose gradient you used).

## going deeper

- Amari (1998), "Natural gradient works efficiently in learning", *Neural Computation* 10(2). [doi:10.1162/089976698300017746](https://doi.org/10.1162/089976698300017746)
- Martens (2020), "New insights and perspectives on the natural gradient method", *JMLR* 21(146). [arXiv:1412.1193](https://arxiv.org/abs/1412.1193). the best single reference: invariance, the GGN connection, damping, and what natural gradient does and doesn't explain.
- Martens (2010), "Deep learning via Hessian-free optimization", *ICML*. [PDF](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf). Gauss–Newton products plus truncated CG.
- Pascanu & Bengio (2014), "Revisiting natural gradient for deep networks", *ICLR*. [arXiv:1301.3584](https://arxiv.org/abs/1301.3584)
- Kunstner, Balles & Hennig (2019), "Limitations of the empirical Fisher approximation for natural gradient descent", *NeurIPS*. [arXiv:1905.12558](https://arxiv.org/abs/1905.12558). the lab's second experiment follows their linear-regression example.
- Kakade (2001), "A natural policy gradient", *NeurIPS* 14. the bridge to TRPO.
- Shewchuk (1994), "An introduction to the conjugate gradient method without the agonizing pain", CMU technical report. [PDF](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

## connections

- ← builds on: the Fisher metric, pullbacks and the empirical Fisher (topic 18), Newton's method (ch 06 · topic 05), conjugate gradients and TRPO's natural step (ch 24 · topic 04), maximum likelihood and the Cramér–Rao bound (ch 05).
- → used in: K-FAC (topic 20) replaces the CG solve with a cheap Kronecker-factored inverse; EWC (topic 13, in the continual-learning block) uses the Fisher as a penalty metric; natural policy gradients and TRPO (ch 24) are this topic applied to policies.
