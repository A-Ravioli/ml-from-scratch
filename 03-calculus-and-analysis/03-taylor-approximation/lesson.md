# taylor approximation

## why this matters

nearly every optimizer in machine learning makes the same move: it can't see the whole loss surface, so it builds a simple **local model** of it, a line or a parabola, and acts on the model. gradient descent trusts a linear model and takes a small step downhill. Newton's method trusts a quadratic model and jumps to its minimum. trust-region methods ask, explicitly, how far the model can be believed. even the analysis of these methods ("why does a step size of $1/L$ work?", topic 05) is an argument about local models.

Taylor's theorem is the guarantee behind all of this: a smooth function, seen up close, *is* a polynomial, up to an error you can bound. but "up close" is doing a lot of work. move too far from the center and the model is wrong, sometimes wildly, and in ways that aren't obvious from the function. a perfectly smooth, bounded function can have a Taylor series that diverges a short distance away. a quadratic model that's excellent in one direction can be useless in another. and Newton's method, which blindly trusts its model, will happily march to a saddle point.

in this topic you build Taylor polynomials and their error bounds, a multivariate Taylor approximator from your own numerical gradient and Hessian, and a tool that measures, direction by direction, where a quadratic model stops being trustworthy.

## warm-up: predict first

1. what's the degree-2 Taylor polynomial of $e^x$ at 0, and how far off is it at $x = 0.5$?
2. you halve the distance from the center. how much smaller does the error of a degree-2 Taylor approximation get?
3. $1/(1 + x^2)$ is smooth everywhere on the real line. does its Taylor series at 0 converge at $x = 2$?
4. a quadratic model $m$ of $f$ predicts that a step will decrease $f$ by 1.0, and $f$ actually decreases by 0.4. what does that tell you about the step?
5. Newton's method jumps to the stationary point of the quadratic model. is that always a minimum of the model?

## prerequisites check

- [ ] you can differentiate repeatedly and know the derivatives of $e^x$, $\sin$, $\log$ (→ ch 01 · topic 02)
- [ ] you can compute gradients and Hessians numerically, and know what $d^\top Hd$ means (→ topic 02)
- [ ] you know what order of convergence and $O(h^p)$ mean, and how to read them off a log-log plot (→ topic 01)
- [ ] you can solve a linear system with `np.linalg.solve` (→ ch 02 · topic 03)

## core ideas

### idea 1: polynomials that match derivatives

the tangent line $f(x_0) + f'(x_0)(x - x_0)$ is the best straight-line approximation to $f$ near $x_0$: it matches the value and the slope. to do better, match the curvature too, and then the third derivative, and so on. the polynomial of degree $K$ whose first $K$ derivatives at $x_0$ match $f$'s is the **Taylor polynomial**

$$T_K(x) = \sum_{k=0}^{K}\frac{f^{(k)}(x_0)}{k!}(x - x_0)^k.$$

the $k!$ is forced: differentiating $(x - x_0)^k$ $k$ times gives $k!$, so dividing by it makes the $k$-th derivative of $T_K$ at $x_0$ exactly $f^{(k)}(x_0)$. some you know by heart: $e^x \approx 1 + x + x^2/2 + x^3/6$, $\sin x \approx x - x^3/6 + x^5/120$, $\log(1 + x) \approx x - x^2/2 + x^3/3$, all at $x_0 = 0$.

to evaluate $T_K$ for many $x$ at once, build the terms incrementally: $t_0 = 1$ and $t_k = t_{k-1}\cdot(x - x_0)/k$, so $t_k = (x - x_0)^k/k!$ without ever computing a large power or a large factorial separately (both overflow quickly; their ratio is fine).

> **▶ build: stage 1.** implement `taylor_poly_1d(derivs, x0, x)`, which evaluates $\sum_k \text{derivs}[k]\,(x - x_0)^k/k!$ for a float or an array `x`.

### idea 2: how wrong is it? Taylor's theorem

**Taylor's theorem** (Lagrange form): if $f$ has $K + 1$ continuous derivatives between $x_0$ and $x$, then for some $\xi$ between them,

$$f(x) - T_K(x) = \frac{f^{(K+1)}(\xi)}{(K + 1)!}(x - x_0)^{K+1}.$$

you don't know $\xi$, but if $|f^{(K+1)}| \le M$ on the interval, the **Lagrange bound** follows:

$$|f(x) - T_K(x)| \le \frac{M\,|x - x_0|^{K+1}}{(K + 1)!}.$$

two lessons. first, near the center the error is $O(|x - x_0|^{K+1})$: halve the distance and a degree-2 approximation gets $2^3 = 8$ times more accurate (warm-up 2). on a log-log plot of error against distance, the slope is $K + 1$, and fitting that slope is how you check numerically that a model is as good as it claims (you'll use `loglog_slope` on your multivariate models). second, far from the center everything depends on how fast $M$ grows with $K$. for $e^x$ or $\sin x$ the derivatives stay modest and $(K + 1)!$ wins: the series converges everywhere. for $\log(1 + x)$ the $K$-th derivative is about $K!$, the factorial cancels, and the bound only shrinks when $|x| < 1$.

**smooth isn't enough.** a function whose Taylor series converges to it near every point is called **analytic**. most functions you meet are, but the series can still have a finite **radius of convergence**. $1/(1 + x^2)$ is infinitely differentiable on the whole real line, yet its series at 0, $1 - x^2 + x^4 - x^6 + \dots$, diverges for $|x| \ge 1$ (warm-up 3: no). the reason is invisible on the real line: as a function of a complex variable, $1/(1 + z^2)$ blows up at $z = \pm i$, at distance 1 from the center, and a power series can't converge past the nearest singularity. (stranger still, $e^{-1/x^2}$, extended by 0 at the origin, is smooth with *every* derivative zero at 0: its Taylor series is identically zero and matches the function only at one point.) experiment 1 shows the divergence.

> **🔮 predict:** for $\log(1 + x)$ at $x_0 = 0$, does adding terms improve the approximation at $x = 0.5$? at $x = 1.5$?

> **▶ build: stage 2.** implement `lagrange_bound(M, x0, x, order)` and `loglog_slope(hs, errors)`, the least-squares slope of $\log(\text{errors})$ against $\log(h)$.

### idea 3: Taylor in several variables

for $f: \mathbb{R}^n \to \mathbb{R}$, reduce to one variable. walk from $x_0$ in direction $d$: $\phi(t) = f(x_0 + td)$. topic 02 gave $\phi'(0) = \nabla f(x_0)^\top d$ and $\phi''(0) = d^\top H(x_0)\,d$. the 1-D Taylor expansion of $\phi$ at $t = 0$, evaluated at $t = 1$, is

$$f(x_0 + d) = \underbrace{f(x_0) + \nabla f(x_0)^\top d + \tfrac12\,d^\top H(x_0)\,d}_{m(x_0 + d)} + O(\|d\|^3).$$

the **quadratic model** $m$ is the multivariate Taylor polynomial of degree 2. dropping the last term gives the **linear model** (the tangent plane). their geometry is everything an optimizer knows about the function:

- the linear model has no minimum: it decreases forever along $-\nabla f$. that's why gradient descent needs a step size.
- the quadratic model's shape is set by the eigenvalues of $H$: a bowl if all are positive, a dome if all are negative, a saddle if mixed. if $H$ is positive definite, the model has a unique minimum where $\nabla m = g + Hd = 0$, i.e. at the **Newton step** $d = -H^{-1}g$.

higher-order terms exist (the cubic term involves a third-derivative tensor with $n^3$ entries), but optimization almost never goes past second order: the Hessian's $n^2$ entries are already expensive.

> **▶ build: stage 3.** implement `quadratic_model(x0, f0, g, H)`, returning a function `m` that evaluates the model at one point $(n,)$ or at a batch of points $(k, n)$.

### idea 4: models from numerical derivatives

in practice you rarely have formulas for $\nabla f$ and $H$. with your tools from topic 02 and chapter 02 you can build the model of *any* function numerically: $f_0 = f(x_0)$, $g$ from `numerical_gradient`, $H$ from `numerical_hessian`. the order of the model is how many of these you include: order 0 is the constant $f_0$, order 1 the tangent plane, order 2 the full quadratic.

two errors are now in play. the **model error** $O(\|d\|^{\text{order}+1})$ is the Taylor remainder: it shrinks as you approach $x_0$. the **derivative error** from finite differences (around $10^{-10}$ for the gradient, $10^{-8}$ for the Hessian) is baked into $g$ and $H$, and it doesn't shrink faster than $\|d\|$: it contributes about $10^{-10}\|d\| + 10^{-8}\|d\|^2$ to the model error. that's far below the Taylor term $\sim\|d\|^3$ for every step an optimizer cares about. measured on $e^x\sin y$, the model error falls with a clean log-log slope of 3 from $\|d\| = 10^{-1}$ (error $8 \times 10^{-5}$) down to $\|d\| = 10^{-4}$ (error $8 \times 10^{-14}$), and below that it simply hits the rounding level of $f$ itself, about $\varepsilon|f| \approx 10^{-16}$. (topic 04's forward-mode autodiff removes the derivative error entirely; the capstone uses it.)

> **▶ build: stage 4.** implement `taylor_model(f, x0, *, order=2)` with `numerical_gradient` and `numerical_hessian`, returning a `quadratic_model` (with zero Hessian, or zero gradient and Hessian, for the lower orders).

### idea 5: when can you trust the model?

an optimizer proposes a step $d$ because the model says $f$ will drop by $m(x_0) - m(x_0 + d)$. it then measures what actually happened. the **agreement ratio**

$$\rho = \frac{f(x_0) - f(x_0 + d)}{m(x_0) - m(x_0 + d)} = \frac{\text{actual reduction}}{\text{predicted reduction}}$$

summarizes the model's honesty over that step. $\rho \approx 1$: the model was right. $\rho = 0.4$ (warm-up 4): you got less than half the promised decrease, so the step was probably too long. $\rho < 0$: $f$ went *up*, and the model was badly wrong. trust-region methods (chapter 06) run on exactly this number: accept the step and grow the region if $\rho$ is close to 1, shrink it if $\rho$ is small.

for a single direction $u$, the **trust radius** is how far you can walk before $|\rho - 1|$ first exceeds a tolerance. by Taylor's theorem, the actual reduction differs from the predicted one by the cubic remainder, about $\tfrac16\|d\|^3|f'''|$, while the predicted reduction is about $\|d\|\,|g\cdot u|$ (or $\tfrac12\|d\|^2u^\top Hu$ near a critical point). so the trust radius is large in directions where the model predicts a big change and the third derivatives are small, and tiny where the function curves in ways the quadratic can't follow. it's a *direction-dependent* quantity: the trustworthy region is rarely a disk (experiment 2).

and the quadratic model can be trustworthy and still lead you astray. if $H$ isn't positive definite, the model's stationary point $-H^{-1}g$ is a saddle or a maximum of the model (warm-up 5: no), and the Newton step heads there faithfully. Newton's method finds points where $\nabla f = 0$, and it doesn't care which kind (experiment 3).

> **🔮 predict:** for $f(x) = x_1^2 + x_1^3 + x_2^2$ and its exact quadratic model at the origin, what is $\rho$ for the step $(r, 0)$? for $(-r, 0)$?

> **▶ build: stage 5.** implement `agreement_ratio(f, model, x0, x)` and `trust_radius(f, model, x0, direction, radii, *, tol=0.25)`, which walks out through `radii` along the normalized direction and returns the last radius before $|\rho - 1| > \text{tol}$ first happens (0.0 if the first radius already fails).

## worked example

**one variable.** for $e^x$ at $x_0 = 0$ (every derivative is 1), $T_2(x) = 1 + x + x^2/2$, so $T_2(0.5) = 1.625$. the true value is $e^{0.5} = 1.648721$, an error of $0.023721$. on $[0, 0.5]$, $|f'''| = e^\xi \le e^{0.5} = M$, so the Lagrange bound is $M \cdot 0.5^3/3! = 1.648721 \times 0.125/6 = 0.034350$. the bound holds, and it's within a factor of 1.5 of the truth.

**two variables.** $f(x, y) = e^x\sin y$ at $x_0 = (0, \pi/2)$:

$$f_0 = 1,\qquad \nabla f = (e^x\sin y,\ e^x\cos y) = (1, 0),\qquad H = \begin{pmatrix}e^x\sin y & e^x\cos y\\ e^x\cos y & -e^x\sin y\end{pmatrix} = \begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix}.$$

the model is $m(x_0 + d) = 1 + d_1 + \tfrac12(d_1^2 - d_2^2)$: a saddle, curving up in $x$ and down in $y$. at $d = (0.1, 0.1)$ it gives $1 + 0.1 + \tfrac12(0.01 - 0.01) = 1.1$. the truth is $e^{0.1}\cos(0.1) = 1.105171 \times 0.995004 = 1.099649$, an error of $3.5 \times 10^{-4}$, about $\|d\|^3 = 2.8 \times 10^{-3}$ times a modest constant. the order-1 model gives the same 1.1 here (the quadratic terms happen to cancel at this $d$), but at $d = (0.1, 0)$ it gives 1.1 while the order-2 model gives 1.105, against the truth 1.105171.

your `taylor_poly_1d`, `lagrange_bound` and `taylor_model` must reproduce these numbers.

## on paper

1. **a Taylor polynomial and its bound.** find $T_3$ for $\log(1 + x)$ at 0, and bound $|\log 1.5 - T_3(0.5)|$ with the Lagrange form. compare with the actual error.

<details><summary>solution</summary>

$f' = (1 + x)^{-1}$, $f'' = -(1 + x)^{-2}$, $f''' = 2(1 + x)^{-3}$, $f'''' = -6(1 + x)^{-4}$. at 0: $0, 1, -1, 2$, so $T_3 = x - x^2/2 + x^3/3$ and $T_3(0.5) = 0.5 - 0.125 + 0.041667 = 0.416667$. on $[0, 0.5]$, $|f''''| \le 6$ (at $\xi = 0$), so the bound is $6 \cdot 0.5^4/24 = 0.015625$. the truth is $\log 1.5 = 0.405465$, an error of $0.011202$. ✓

</details>

2. **the multivariate expansion.** with $\phi(t) = f(x_0 + td)$, show $\phi'(t) = \nabla f(x_0 + td)^\top d$ and $\phi''(t) = d^\top H(x_0 + td)\,d$. then use the 1-D Lagrange form to write the exact remainder of the *linear* model in terms of $H$ at an intermediate point.

<details><summary>solution</summary>

the chain rule gives $\phi'(t) = \sum_i\partial_if(x_0 + td)\,d_i$. differentiating again, $\phi''(t) = \sum_{i,j}\partial_j\partial_if(x_0 + td)\,d_id_j = d^\top H(x_0 + td)\,d$. the 1-D theorem with $K = 1$ gives $\phi(1) = \phi(0) + \phi'(0) + \tfrac12\phi''(\tau)$ for some $\tau \in (0, 1)$: $f(x_0 + d) = f(x_0) + \nabla f(x_0)^\top d + \tfrac12d^\top H(x_0 + \tau d)\,d$. if $\|H\| \le L$ everywhere, the linear model's error is at most $\tfrac L2\|d\|^2$: that's the "descent lemma" of topic 05.

</details>

3. **an agreement ratio by hand.** for $f(x) = x_1^2 + x_1^3 + x_2^2$ at the origin, find the exact quadratic model, compute $\rho$ for the steps $(r, 0)$ and $(-r, 0)$, and find the trust radius in each direction for $\text{tol} = 0.25$.

<details><summary>solution</summary>

$\nabla f(0) = 0$ and $H(0) = \operatorname{diag}(2, 2)$, so $m(d) = d_1^2 + d_2^2$. for $d = (r, 0)$: actual reduction $-(r^2 + r^3)$, predicted $-r^2$, so $\rho = 1 + r$. for $(-r, 0)$: $f = r^2 - r^3$, so $\rho = 1 - r$. both give $|\rho - 1| = r$, so the trust radius is 0.25 in both directions. along $x_2$ the model is exact, $\rho = 1$ for every $r$, and the trust radius is unlimited: the region is very much not a disk.

</details>

4. **the Newton step.** show that the stationary point of $m(x_0 + d) = f_0 + g^\top d + \tfrac12d^\top Hd$ is $d^* = -H^{-1}g$ (for invertible $H$), that the predicted reduction there is $\tfrac12g^\top H^{-1}g$, and that it is a minimum of $m$ exactly when $H$ is positive definite.

<details><summary>solution</summary>

$\nabla_dm = g + Hd = 0$ gives $d^* = -H^{-1}g$. then $m(x_0) - m(x_0 + d^*) = -g^\top d^* - \tfrac12d^{*\top}Hd^* = g^\top H^{-1}g - \tfrac12g^\top H^{-1}g = \tfrac12g^\top H^{-1}g$. $m$ has Hessian $H$ everywhere, so its stationary point is a minimum iff $H$ is positive definite, and then the predicted reduction is positive. if $H$ has a negative eigenvalue, $g^\top H^{-1}g$ can be negative: the "Newton step" moves to a point the model itself says is *higher*.

</details>

## experiment

`lab.py` investigates three things with your code:

1. Taylor polynomials of $1/(1 + x^2)$ of increasing degree, around two different centers, compared with $e^x$.
2. the shape of the region where your quadratic model is trustworthy, at points in and around the curved valley of a Rosenbrock function.
3. where Newton's method, built on the quadratic model, takes you from a grid of starting points.

## check your understanding

1. why does the $k$-th Taylor coefficient have a $k!$ in the denominator?

<details><summary>answer</summary>

so that the $k$-th derivative of the polynomial at $x_0$ equals $f^{(k)}(x_0)$: $\frac{d^k}{dx^k}(x - x_0)^k = k!$, and every other term's $k$-th derivative vanishes at $x_0$.

</details>

2. (the predict callout in idea 2.) does adding terms help for $\log(1 + x)$ at $x = 0.5$ and at $x = 1.5$?

<details><summary>answer</summary>

at 0.5, yes: the terms are $\pm0.5^k/k$, which shrink geometrically. at 1.5, no: the terms are $\pm1.5^k/k$, which grow, so the partial sums oscillate with growing amplitude. the radius of convergence is 1, set by the singularity of $\log(1 + x)$ at $x = -1$.

</details>

3. a model built from finite-difference derivatives has coefficient errors around $10^{-10}$ (gradient) and $10^{-8}$ (Hessian). where does that show up in the model error as $r \to 0$?

<details><summary>answer</summary>

the coefficient errors multiply $r$ and $r^2$, contributing about $10^{-10}r + 10^{-8}r^2$. for a well-scaled $f$ that's below the Taylor remainder ($\sim r^3$ times a modest constant) until $r \approx 10^{-4}$ to $10^{-5}$, where the model error is already near $10^{-14}$, and from there the error is dominated by rounding in $f$ itself, $\varepsilon|f| \approx 10^{-16}$. so in practice you see a slope of 3 down to the rounding floor; the coefficient errors would only matter for a badly scaled $f$ (large $|f|$ or tiny curvature) or a much cruder step $h$.

</details>

4. (the predict callout in idea 5.) what is $\rho$ for the steps $(\pm r, 0)$ on $x_1^2 + x_1^3 + x_2^2$?

<details><summary>answer</summary>

$1 + r$ for $(r, 0)$ and $1 - r$ for $(-r, 0)$ (on paper 3). the cubic term helps in one direction and hurts in the other, but the model's error is the same size, $r^3$, either way.

</details>

5. why do trust-region methods use $\rho$ instead of just checking whether $f$ went down?

<details><summary>answer</summary>

"$f$ went down" says the step was acceptable, not whether the model can be trusted further. $\rho$ compares the model's prediction with reality: $\rho \approx 1$ justifies taking bigger steps next time, $0 < \rho \ll 1$ means progress but a poor model (shrink the region), $\rho < 0$ means reject. it adapts the step length to how good the local model actually is.

</details>

6. can a quadratic model be exact everywhere?

<details><summary>answer</summary>

yes, if $f$ is itself quadratic: then the Taylor remainder is zero, $\rho = 1$ for every step, and a single Newton step lands on the minimum (if $H$ is positive definite). least squares is the main example in ML, and it's why the normal equations solve it in "one step".

</details>

7. Newton's method converged, and $\nabla f = 0$ at the result. is it a minimum?

<details><summary>answer</summary>

not necessarily: Newton's iteration finds stationary points of every kind. check the Hessian's eigenvalues (topic 02's `classify_critical_point`). experiment 3 shows Newton converging to a saddle from half of the starting points.

</details>

## common bugs

- **forgetting the factorials.** symptom: $T_2(0.5) = 1.75$ for $e^x$ instead of 1.625. each term is $f^{(k)}(x_0)(x - x_0)^k/k!$.
- **powers of $x$ instead of $x - x_0$.** symptom: everything works at $x_0 = 0$ and fails elsewhere.
- **an off-by-one in the remainder.** symptom: the "bound" is smaller than the actual error. the remainder of a degree-$K$ polynomial involves the $(K + 1)$-th derivative and $|x - x_0|^{K+1}/(K + 1)!$.
- **dropping the $\tfrac12$ in the quadratic model.** symptom: the model is exact to first order but its curvature is doubled; Newton steps come out half as long as they should.
- **batches that don't broadcast.** symptom: `m(X)` returns one number for a $(k, n)$ batch. compute $d^\top Hd$ row by row, e.g. with `np.einsum("...i,ij,...j->...", d, H, d)` or `np.sum((d @ H) * d, axis=-1)`.
- **taking the best radius instead of the first failure.** symptom: `trust_radius` reports a radius beyond a region where the model failed. a trust radius is a contiguous stretch from the center: stop at the first failure.

## going deeper

- Apostol (1967), *Calculus, Volume 1*, 2nd ed., Wiley, chapter 7. Taylor's theorem and its remainders, rigorously.
- Trefethen (2019), *Approximation Theory and Approximation Practice*, extended ed., SIAM, chapters 8 and 12 (convergence for analytic functions; potential theory). why singularities in the complex plane govern real approximations. [doi:10.1137/1.9781611975949](https://doi.org/10.1137/1.9781611975949)
- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer, chapter 4 (trust-region methods and the agreement ratio). [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Conn, Gould & Toint (2000), *Trust-Region Methods*, SIAM. the full theory of trusting local models. [doi:10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

## connections

- ← builds on: derivatives (ch 01 · topic 02), gradients and Hessians, `numerical_hessian` (topic 02), orders of convergence (topic 01).
- → used in: forward-mode autodiff, which is Taylor's theorem with $\varepsilon^2 = 0$ (topic 04), the descent lemma and step sizes (topic 05), the Newton capstone, gradient descent, Newton's method and trust regions (ch 06), Laplace approximations (ch 12), and loss-landscape curvature in deep learning (ch 19).
