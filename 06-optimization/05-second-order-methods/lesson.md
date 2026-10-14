# second-order methods

## why this matters

every first-order method so far pays for the condition number: $\kappa$ iterations for gradient descent, $\sqrt\kappa$ with momentum, and the lower bound of topic 03 says that's the best gradients alone can do. Newton's method doesn't pay at all. it uses the Hessian to rescale space so that every direction has the same curvature, and on a quadratic it lands on the minimum in **one step**, whatever $\kappa$ is. near any smooth minimum it converges quadratically: the number of correct digits doubles every step.

the catch is cost. a Newton step needs the $n \times n$ Hessian and a linear solve, $O(n^3)$, which is fine for $n = 100$ and hopeless for a million parameters. this topic is about the clever middle ground. **quasi-Newton** methods (BFGS) learn an approximation of the inverse Hessian from nothing but the gradients they've already seen, at $O(n^2)$ per step, and still converge superlinearly. **L-BFGS** keeps only the last few steps, at $O(mn)$, and is the default optimizer for smooth, deterministic problems in scientific computing and classical ML: scikit-learn's `LogisticRegression` uses it by default, SciPy uses its bound-constrained variant, and in chapter 09 your Gaussian process will fit its hyperparameters with *your* `lbfgs`. you'll build Newton, BFGS and L-BFGS, the line search that makes quasi-Newton methods work, and a single `minimize` front end for every deterministic optimizer in `mlfs.optim`.

## warm-up: predict first

1. Newton's method on $f(x) = \tfrac12x^\top Ax - b^\top x$ with $\kappa(A) = 10^6$. how many iterations?
2. you change the units of one parameter (multiply it by 1,000). does gradient descent's path change? Newton's?
3. at a point where the Hessian has a negative eigenvalue, does the Newton step always go downhill?
4. BFGS keeps every $(s, y)$ pair it has seen; L-BFGS keeps only the last 10. on a 20-parameter problem, which needs fewer iterations?
5. L-BFGS with memory $m = 5$ on a problem with $n = 10^6$ parameters: how much memory does it need, compared with BFGS?

## prerequisites check

- [ ] you know the quadratic model $m(x + d) = f + g^\top d + \tfrac12d^\top Hd$ and when it can be trusted (→ ch 03 · topic 03)
- [ ] you've met Newton's method for roots and its quadratic convergence (→ ch 03 · topic 01, and the ch 03 capstone)
- [ ] you can solve linear systems with your `mlfs.linalg.solve`, and know that a singular matrix makes it fail (→ ch 02 · topic 03)
- [ ] you've built `backtracking_line_search`, `gradient_descent`, `heavy_ball`, `nesterov` and know their rates (→ topics 02–03)
- [ ] you know rank-one updates and outer products (→ ch 02 · topic 02)

## core ideas

### idea 1: Newton's method, and how to make it safe

at the current point $x$, replace $f$ by its quadratic model $m(d) = f(x) + g^\top d + \tfrac12d^\top Hd$ (with $g = \nabla f(x)$, $H = \nabla^2f(x)$) and jump to the model's stationary point: solve

$$Hd = -g,\qquad x \leftarrow x + d.$$

on a quadratic the model *is* the function, so one step lands on the minimizer (warm-up 1: one iteration, for any $\kappa$). in general, near a minimum $x^*$ with $H(x^*) \succ 0$, the error satisfies $\|x_{k+1} - x^*\| \le C\|x_k - x^*\|^2$: **quadratic convergence** (on paper 2). and Newton's method is **affine invariant**: if you change variables $x = Az$, the Newton iterates in $z$ are exactly the transformed Newton iterates in $x$ (on paper 1). rescaling a parameter (warm-up 2) changes gradient descent's path completely and Newton's not at all. that's what "no dependence on $\kappa$" means: Newton sees every problem in its own natural coordinates.

far from a minimum, pure Newton is dangerous, and the ch 03 capstone showed you how. if $H$ has a negative eigenvalue, the model's stationary point can be a saddle or a maximum, and Newton walks straight to it (warm-up 3: no, not always downhill). if $H$ is singular, the step doesn't exist. and even with $H \succ 0$, the full step can overshoot where the model is poor. a practical Newton method adds three safeguards:

- **a descent check.** $d$ is a descent direction iff $g^\top d < 0$. if the Newton direction isn't one, fall back to the steepest-descent direction $-g$ for this step.
- **a line search.** take $x + \alpha d$ with $\alpha$ from your Armijo backtracking, starting at $\alpha_0 = 1$ (the natural Newton step, which is accepted near the minimum, so the quadratic rate survives).
- **damping** (Levenberg's idea): solve $(H + \lambda I)d = -g$ instead. for $\lambda = 0$ it's Newton; as $\lambda \to \infty$, $d \approx -g/\lambda$, a short gradient step. adding $\lambda I$ also lifts small or negative eigenvalues (if $\lambda > -\lambda_{\min}(H)$ the system is positive definite). trust-region methods choose $\lambda$ adaptively; here it's a fixed knob.

if the solve fails outright (`np.linalg.LinAlgError` from a singular matrix), use $-g$ for that step.

> **🔮 predict:** Newton's method on $f(x) = e^x - 2x$ from $x_0 = 0$. what are $x_1$ and $x_2$, and how many correct digits does each new step add?

> **▶ build: stage 1.** implement `newton(f, grad, hess, x0, *, damping, line_search, max_iters, tol, callback)`, solving with your `mlfs.linalg.solve`, with both fallbacks and your `backtracking_line_search`. with `line_search=False` it's pure (damped) Newton, safeguards off.

### idea 2: quasi-Newton: learn the curvature from the gradients

the Hessian is expensive, but every step tells you something about it for free. after a step $s = x_{k+1} - x_k$, the gradient changed by $y = \nabla f(x_{k+1}) - \nabla f(x_k)$, and by the mean value theorem $y = \bar Hs$ for an average Hessian $\bar H$ along the step. so any good inverse-Hessian approximation should satisfy the **secant condition**

$$H_{k+1}\,y = s.$$

that's $n$ equations for $n(n+1)/2$ unknowns, so you choose the update that changes $H_k$ the least while keeping it symmetric. the answer (in a particular weighted norm) is the **BFGS update**, with $\rho = 1/(y^\top s)$:

$$H_{k+1} = (I - \rho\,sy^\top)\,H_k\,(I - \rho\,ys^\top) + \rho\,ss^\top.$$

it satisfies the secant condition, keeps $H$ symmetric, and keeps it **positive definite whenever $y^\top s > 0$** (on paper 3). with $H$ positive definite, $d = -Hg$ is always a descent direction. so the method is: $d = -H_kg$, line search, step, update. the very first step uses $H_0 = I$ (a gradient step); before the first update, rescale $H_0 = \frac{s^\top y}{y^\top y}I$, an estimate of the inverse curvature along the first step, so that $H$ starts out with the right size.

**the curvature condition, and the Wolfe line search.** positivity hinges on $y^\top s > 0$: the gradient's slope along the step must increase. on a convex function that's automatic; on a nonconvex one it isn't, and Armijo backtracking doesn't help: it only shortens steps, and short steps can still have $y^\top s \le 0$. (with Armijo alone, L-BFGS on Rosenbrock gets stuck taking hundreds of tiny steps: the bad pairs are skipped, the curvature information stops updating, and the step stays too short.) the fix is the **Wolfe conditions**: Armijo *plus* the curvature condition

$$\nabla f(x + \alpha d)^\top d \ge c_2\,\nabla f(x)^\top d,\qquad c_1 < c_2 < 1\ (\text{typically } c_2 = 0.9),$$

which says "the slope has flattened enough, so you didn't stop too early". it implies $y^\top s = \alpha\big(\nabla f(x + \alpha d) - \nabla f(x)\big)^\top d \ge \alpha(c_2 - 1)\nabla f(x)^\top d > 0$ (on paper 4). a simple way to find such a step is to keep a bracket: start at $\alpha = 1$; if Armijo fails the step is too long (shrink the bracket from above); if the curvature condition fails it's too short (from below); bisect the bracket, or double $\alpha$ while it has no upper end. as a bonus, the line search has already computed $f$ and $\nabla f$ at the accepted point, so the next iteration reuses them.

near a minimum (with a nonsingular Hessian), BFGS converges **superlinearly**: faster than any linear rate, though not quadratically. that's a statement about the limit, and how long the limit takes to arrive depends on the starting matrix and the line search: with an exact line search on a quadratic, BFGS finishes in about $d$ steps, with a loose Wolfe search it can take several times that (lab experiment 2). remarkably, it doesn't need $H_k$ to converge to the true inverse Hessian; it only needs $H_k$ to be accurate *along the directions it's actually stepping in* (the Dennis–Moré condition).

> **🔮 predict:** with $H = I$, $s = (1, 0)$ and $y = (2, 1)$, what is the updated $H$? check that it maps $y$ to $s$.

> **▶ build: stage 2.** implement `wolfe_line_search(f, grad, x, fx, gx, direction, ...)` (expand-and-bisect, returning $\alpha$, $f$ and $\nabla f$ at the new point), `bfgs_update(Hinv, s, y)`, and `bfgs(f, grad, x0, ...)`.

### idea 3: L-BFGS: the last $m$ steps are enough

dense BFGS stores an $n \times n$ matrix and spends $O(n^2)$ per step: fine for thousands of parameters, not for millions. but $H_k$ is entirely determined by $H_0$ and the pairs $(s_i, y_i)$, so you never need to *store* it, only to *apply* it to a vector. **L-BFGS** keeps the last $m$ pairs (typically 5–20), starts each time from $H_0 = \gamma I$ with $\gamma = \frac{s^\top y}{y^\top y}$ from the newest pair, and computes $H_kg$ with the **two-loop recursion**:

$$\begin{aligned}
&q \leftarrow g;\quad\text{for } i = \text{newest} \dots \text{oldest}:\ \ a_i = \rho_is_i^\top q,\ \ q \leftarrow q - a_iy_i\\
&r \leftarrow \gamma q;\quad\text{for } i = \text{oldest} \dots \text{newest}:\ \ b = \rho_iy_i^\top r,\ \ r \leftarrow r + (a_i - b)\,s_i
\end{aligned}$$

and $r = H_kg$ exactly, the same vector as $m$ explicit BFGS updates of $\gamma I$ applied to $g$ (each factor $(I - \rho sy^\top)$ in the update becomes one step of a loop). the cost is about $4mn$ flops and $2mn$ numbers of memory (warm-up 5: for $n = 10^6$ and $m = 5$, about 80 MB instead of BFGS's 8 TB). a pair is only stored if it passes the curvature check, and the oldest pair falls off when the memory is full: L-BFGS adapts to the *recent* curvature. it also re-seeds its starting matrix with the newest $\gamma$ at *every* iteration, while BFGS seeds once with the first step's $\gamma_0$ and must correct that guess one direction per update. in practice this re-scaling often makes L-BFGS need *fewer* iterations than full BFGS (warm-up 4; lab experiments 2–3).

> **▶ build: stage 3.** implement `two_loop(g, s_hist, y_hist)` and `lbfgs(f, grad, x0, *, history_size, ...)` (your `bfgs` loop, with a `deque(maxlen=history_size)` of pairs).

### idea 4: one front end, and how to choose

with six deterministic minimizers in `mlfs.optim`, later chapters shouldn't need to know each one's quirks. `minimize(f, grad, x0, method=..., hess=..., max_iters, tol, callback, **options)` dispatches to the right one and forwards everything else (`lr`, `momentum`, `damping`, `history_size`, ...). how to pick:

| method | per-iteration cost | memory | rate near a minimum | needs |
|---|---|---|---|---|
| gradient descent | $O(n)$ + one gradient | $O(n)$ | linear, $1 - O(1/\kappa)$ | `lr` (or a line search) |
| Nesterov / heavy ball | $O(n)$ + one or two gradients | $O(n)$ | linear, $1 - O(1/\sqrt\kappa)$ | `lr`, momentum |
| Newton | $O(n^3)$ + Hessian | $O(n^2)$ | quadratic | `hess` |
| BFGS | $O(n^2)$ + gradients | $O(n^2)$ | superlinear | nothing |
| L-BFGS | $O(mn)$ + gradients | $O(mn)$ | linear, but fast in practice | nothing |

rules of thumb: for small, smooth problems where the Hessian is cheap (a logistic regression with tens of features), Newton wins on time as well as iterations. for medium problems without Hessians, BFGS. for large deterministic problems, L-BFGS. for stochastic gradients (minibatches), none of these: quasi-Newton updates are corrupted by gradient noise, and you're back to topic 04. iterations are not time: the lab measures both.

one practical caution for every method that uses a line search on $f$: it can only accept a step if it can *measure* the decrease, and $f$ is only known to about $\epsilon_{\text{mach}}|f|$. near the minimum the decrease from a good step is about $\|g\|^2/2L$, so asking for $\|g\| \le$ `tol` below roughly $\sqrt{2L\epsilon_{\text{mach}}|f|}$ can't succeed: the line search fails and the method stops with `converged=False`. choose `tol` with the scale of the problem in mind.

> **▶ build: stage 4.** implement `minimize(f, grad, x0, *, method='lbfgs', hess=None, max_iters, tol, callback, **options)`. `method='newton'` without `hess` and unknown methods raise `ValueError`.

## worked example

**Newton on $f(x) = e^x - 2x$** (minimum at $\ln 2 = 0.693147$), with $f' = e^x - 2$ and $f'' = e^x$, from $x_0 = 0$:

| $k$ | $x_k$ | error $\lvert x_k - \ln 2\rvert$ |
|---|---|---|
| 0 | 0 | $6.9\times10^{-1}$ |
| 1 | $0 - (1 - 2)/1 = 1$ | $3.1\times10^{-1}$ |
| 2 | $1 - (e - 2)/e = 2/e = 0.735759$ | $4.3\times10^{-2}$ |
| 3 | $0.694042$ | $9.0\times10^{-4}$ |
| 4 | $0.69314758$ | $4.0\times10^{-7}$ |
| 5 | $0.693147180559945$ | $8.0\times10^{-14}$ |

every full step passes the Armijo test, and the error squares each step: $e_{k+1} \approx \tfrac12e_k^2$ (the constant is $f'''/2f'' = \tfrac12$ at the minimum).

**one BFGS update.** $H = I$, $s = (1, 0)$, $y = (2, 1)$: $y^\top s = 2$, $\rho = \tfrac12$, and $I - \rho sy^\top = \begin{pmatrix}0 & -\tfrac12\\0 & 1\end{pmatrix}$. then

$$H_+ = \begin{pmatrix}0 & -\tfrac12\\0 & 1\end{pmatrix}\begin{pmatrix}0 & 0\\-\tfrac12 & 1\end{pmatrix} + \begin{pmatrix}\tfrac12 & 0\\0 & 0\end{pmatrix} = \begin{pmatrix}\tfrac34 & -\tfrac12\\-\tfrac12 & 1\end{pmatrix},$$

and $H_+y = (\tfrac32 - \tfrac12, -1 + 1) = (1, 0) = s$ ✓. its eigenvalues are $\frac{7 \pm \sqrt{17}}{8} > 0$: still positive definite.

**the same pair in L-BFGS.** with only this pair, $\gamma = \frac{s^\top y}{y^\top y} = \frac25$, so $H = (I - \rho sy^\top)\,\tfrac25I\,(I - \rho ys^\top) + \rho ss^\top = \begin{pmatrix}0.6 & -0.2\\-0.2 & 0.4\end{pmatrix}$, and `two_loop((1, 1), [s], [y])` returns $H(1, 1) = (0.4, 0.2)$. (the two loops: $a = \tfrac12s^\top g = \tfrac12$, $q = (1, 1) - \tfrac12(2, 1) = (0, \tfrac12)$, $r = \tfrac25q = (0, 0.2)$, $b = \tfrac12y^\top r = 0.1$, $r = (0, 0.2) + (0.5 - 0.1)(1, 0) = (0.4, 0.2)$.)

**a Wolfe step.** $f(x) = x^2$ at $x = 1$ ($g = 2$) along the short direction $d = -0.01$: slope $-0.02$. steps 1, 2, 4, 8 all pass Armijo but have slope $2(1 - 0.01\alpha)(-0.01) < 0.9 \times (-0.02)$: too short. doubling reaches $\alpha = 16$, where $x = 0.84$ and the slope $-0.0168 \ge -0.018$: accepted. Armijo backtracking alone would have returned $\alpha = 1$.

## on paper

1. **affine invariance.** let $\tilde f(z) = f(Az + b)$ with $A$ invertible. show $\nabla\tilde f(z) = A^\top\nabla f(x)$ and $\nabla^2\tilde f(z) = A^\top\nabla^2f(x)A$ at $x = Az + b$, and that the Newton step in $z$ maps to the Newton step in $x$: $A\,d_z = d_x$. why is gradient descent not affine invariant?

<details><summary>solution</summary>

chain rule: $\nabla\tilde f = A^\top g$ and $\nabla^2\tilde f = A^\top HA$. the $z$-step solves $A^\top HAd_z = -A^\top g$, i.e. $HAd_z = -g$, so $Ad_z = -H^{-1}g = d_x$. gradient descent's step in $z$ is $-\eta A^\top g$, which maps to $-\eta AA^\top g$ in $x$: a different direction unless $AA^\top \propto I$ (a rotation and uniform scaling). that's why rescaling features changes gradient descent but not Newton.

</details>

2. **quadratic convergence in 1-D.** for minimizing $f$ with $f'(x^*) = 0$, $f''(x^*) > 0$, Newton is $x_{k+1} = x_k - f'(x_k)/f''(x_k)$. using Taylor's theorem, $0 = f'(x^*) = f'(x_k) + f''(x_k)(x^* - x_k) + \tfrac12f'''(\xi)(x^* - x_k)^2$, show $x_{k+1} - x^* = \frac{f'''(\xi)}{2f''(x_k)}(x_k - x^*)^2$.

<details><summary>solution</summary>

divide the Taylor expansion by $f''(x_k)$: $\frac{f'(x_k)}{f''(x_k)} + (x^* - x_k) = -\frac{f'''(\xi)}{2f''(x_k)}(x^* - x_k)^2$. the left side is $x_k - x_{k+1} + x^* - x_k = x^* - x_{k+1}$. so $x_{k+1} - x^* = \frac{f'''(\xi)}{2f''(x_k)}(x_k - x^*)^2$. near $x^*$ the factor tends to $\frac{f'''(x^*)}{2f''(x^*)}$; for $e^x - 2x$ it's $\tfrac12$, as in the worked example.

</details>

3. **BFGS keeps $H$ positive definite.** with $H \succ 0$ and $y^\top s > 0$, show $v^\top H_+v > 0$ for every $v \ne 0$, where $H_+ = V^\top HV + \rho ss^\top$ and $V = I - \rho ys^\top$. also check the secant condition $H_+y = s$.

<details><summary>solution</summary>

$v^\top H_+v = (Vv)^\top H(Vv) + \rho(s^\top v)^2$. both terms are $\ge 0$ ($H \succ 0$, $\rho > 0$). if both are zero, then $s^\top v = 0$, so $Vv = v - \rho y(s^\top v) = v \ne 0$, and $(Vv)^\top H(Vv) = v^\top Hv > 0$: contradiction. so $H_+ \succ 0$. secant: $Vy = y - \rho y(s^\top y) = 0$, so $H_+y = V^\top H\cdot0 + \rho s(s^\top y) = s$. (note $V^\top = I - \rho sy^\top$, the factor on the left in idea 2.)

</details>

4. **Wolfe implies positive curvature.** show that a step $\alpha > 0$ along a descent direction $d$ satisfying the curvature condition with $c_2 < 1$ gives $y^\top s > 0$. then show that Armijo alone can't guarantee it, with a 1-D example.

<details><summary>solution</summary>

$y^\top s = \alpha(\nabla f(x + \alpha d) - \nabla f(x))^\top d \ge \alpha(c_2 - 1)\nabla f(x)^\top d$, and both $c_2 - 1 < 0$ and $\nabla f(x)^\top d < 0$, so the product is positive. counterexample for Armijo: $f(x) = -x^2$ near $x = 1$ with $d = 1$ (slope $-2$): every step decreases $f$ by more than the Armijo threshold, but $y^\top s = (-2(1 + \alpha) + 2)\alpha = -2\alpha^2 < 0$. on concave stretches, the gradient's slope along the step *decreases*, and only the curvature condition (which can't be met there, so the search keeps expanding until the function turns convex, or fails) protects the update.

</details>

## experiment

`lab.py` investigates three things with your code:

1. iterations versus wall-clock time for gradient descent, Newton, BFGS and L-BFGS on logistic regressions with 10 to 400 features.
2. how accurate BFGS's inverse-Hessian approximation is when it has converged, overall and along the directions it stepped in.
3. how much memory L-BFGS needs: iterations as a function of `history_size`.

## check your understanding

1. why is the full Newton step ($\alpha = 1$) the natural first trial in the line search?

<details><summary>answer</summary>

near a minimum with $H \succ 0$ the quadratic model is accurate and the unit step is (asymptotically) the exact minimizer of the model, so it passes Armijo, and accepting it keeps the quadratic rate. starting from a larger trial would waste evaluations; starting smaller would throw the rate away.

</details>

2. (the predict callout in idea 2.) the BFGS update of $I$ with $s = (1, 0)$, $y = (2, 1)$?

<details><summary>answer</summary>

$\begin{pmatrix}0.75 & -0.5\\-0.5 & 1\end{pmatrix}$, which maps $y$ to $s$ and is positive definite (worked example).

</details>

3. you run `newton` with a large `damping` on a well-conditioned problem. what do you see?

<details><summary>answer</summary>

$(H + \lambda I)^{-1} \approx I/\lambda$ for $\lambda \gg \|H\|$, so each step is about $-g/\lambda$: gradient descent with learning rate $1/\lambda$, with its linear rate. damping buys robustness far from the minimum and costs the quadratic rate near it, which is why practical methods shrink $\lambda$ as they approach the solution (Levenberg–Marquardt, trust regions).

</details>

4. why does BFGS skip an update when $y^\top s \le 0$, rather than applying it?

<details><summary>answer</summary>

the update with $\rho = 1/(y^\top s) \le 0$ (or infinite) would destroy positive definiteness, and then $-Hg$ might not be a descent direction. skipping keeps the old (positive definite) $H$. with a Wolfe line search it happens only through rounding, because the curvature condition guarantees $y^\top s > 0$.

</details>

5. L-BFGS with $m = n$ on an $n$-dimensional problem: is it the same as BFGS?

<details><summary>answer</summary>

not exactly. after $n$ steps both use the same pairs, but L-BFGS restarts from $\gamma_kI$ with the *newest* $\gamma$ at every iteration, while BFGS started from one scaled identity and kept updating it. and once L-BFGS has more than $m$ steps, it forgets the oldest. on a quadratic with exact line searches, BFGS and L-BFGS (with enough memory) both reduce to the conjugate gradient method, but that equivalence breaks with inexact line searches.

</details>

6. you ask L-BFGS for `tol=1e-12` on a problem with $f \approx 10^3$ and $L \approx 10^4$. what happens, and why?

<details><summary>answer</summary>

it stops early with `converged=False`. a step that reduces the gradient from $\|g\|$ decreases $f$ by about $\|g\|^2/2L$, and once that's below the rounding level of $f$, about $10^{-16} \times 10^3 = 10^{-13}$, the line search can't tell it's a decrease. that happens around $\|g\| \approx \sqrt{2 \times 10^4 \times 10^{-13}} \approx 5\times10^{-5}$: `tol=1e-12` is unreachable by any $f$-based line search. (a line search on the gradient alone could go further.)

</details>

7. (the predict callout in idea 1.) Newton on $e^x - 2x$ from 0?

<details><summary>answer</summary>

$x_1 = 0 - (e^0 - 2)/e^0 = 1$ and $x_2 = 1 - (e - 2)/e = 2/e \approx 0.7358$. near the minimum $\ln 2$ the error roughly squares each step ($e_{k+1} \approx \tfrac12e_k^2$), so the number of correct digits roughly doubles: $4\times10^{-2}$, $9\times10^{-4}$, $4\times10^{-7}$, $8\times10^{-14}$ (worked example).

</details>

## common bugs

- **inverting the Hessian.** symptom: slow and inaccurate. solve $Hd = -g$ with `solve`; never form $H^{-1}$.
- **no descent check.** symptom: Newton converges to a saddle or a maximum with `line_search=True`. if $g^\top d \ge 0$, use $-g$.
- **Armijo backtracking for BFGS.** symptom: hundreds of tiny steps on Rosenbrock, or `nan` from an update with $y^\top s \le 0$. use the Wolfe line search, and skip updates that fail the curvature check.
- **the BFGS update's transposes swapped.** symptom: $H_+y \ne s$. the factor on the left is $I - \rho sy^\top$, on the right $I - \rho ys^\top$.
- **the two loops in the same order, or $\gamma$ from the oldest pair.** symptom: L-BFGS converges, but slowly, and `two_loop` doesn't match the dense update. newest-to-oldest first, oldest-to-newest second, $\gamma$ from the newest.
- **a history that never forgets.** symptom: L-BFGS's memory and cost grow every iteration. use a `deque(maxlen=history_size)`.
- **`tol` below the rounding floor.** symptom: the run ends with `converged=False` at a gradient around $10^{-6}$–$10^{-8}$ although nothing is wrong. see idea 4.

## going deeper

- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer, chapters 3 (Wolfe conditions), 6 (BFGS, including the Dennis–Moré condition) and 7 (L-BFGS and the two-loop recursion). [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Liu & Nocedal (1989), "On the limited memory BFGS method for large scale optimization", *Mathematical Programming* 45. [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116)
- Dennis & Moré (1977), "Quasi-Newton methods, motivation and theory", *SIAM Review* 19(1). [doi:10.1137/1019005](https://doi.org/10.1137/1019005)
- Lewis & Overton (2013), "Nonsmooth optimization via quasi-Newton methods", *Mathematical Programming* 141. the expand-and-bisect weak Wolfe line search used here. [doi:10.1007/s10107-012-0514-2](https://doi.org/10.1007/s10107-012-0514-2)
- Boyd & Vandenberghe (2004), *Convex Optimization*, §9.5 (Newton's method, affine invariance and its convergence analysis). [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/)

## connections

- ← builds on: quadratic models and Newton's method (ch 03 · topics 01 and 03, and the ch 03 capstone), linear solves (ch 02 · topic 03), line searches and `OptimResult` (topic 02), momentum methods (topic 03).
- → used in: logistic regression with `solver='irls'` (Newton for GLMs, ch 08), Gaussian process hyperparameters fitted with `mlfs.optim.lbfgs` (ch 09), the capstone's benchmark, Newton steps in the KKT system (topic 06), natural gradients and second-order ideas in deep learning (ch 15, ch 19).
