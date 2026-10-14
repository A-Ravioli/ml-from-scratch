# gradient descent

## why this matters

in ch 03 · topic 05 you met gradient descent as a knife edge: on a quadratic, any step below $2/L$ converges and any step above diverges. that answered "does it converge?". this topic answers the questions you actually face when training a model. **how fast** does it converge, and what does the speed depend on? **how do you pick the step** when you don't know $L$, which is always? and **why does it zig-zag**, crawling down a long narrow valley in a sawtooth instead of walking along it?

the answers are three numbers and one idea. for smooth convex functions the error falls like $1/k$; for strongly convex ones it falls geometrically, by a factor that depends only on the condition number $\kappa = L/\mu$; and even the *best possible* step size can't beat that factor, because the zig-zag is built into following the gradient. the idea is the **line search**: try a step, check that it decreased the function "enough", and shrink it if not. it frees you from knowing $L$ and it's what most deterministic optimizers use. you'll write both, and they become the first pieces of **`mlfs.optim`**, the optimizer library the rest of the course trains models with.

## warm-up: predict first

1. gradient descent with step $1/L$ on a strongly convex quadratic with $\kappa = 100$. roughly how many iterations to shrink the error by a factor of $10^6$: 14, 140, 1,400 or 14,000?
2. an "exact" line search picks, at every iteration, the step that minimizes $f$ along the negative gradient. is it much faster than the best *fixed* step?
3. with an exact line search on a quadratic, what's the angle between two successive steps?
4. a backtracking line search accepts a step when $f$ decreases "enough". why not accept any step that decreases $f$ at all?
5. gradient descent on $f(x) = x^4$ (convex, minimum at 0, but not strongly convex). does the error shrink geometrically?

## prerequisites check

- [ ] you know the descent lemma $f(y) \le f(x) + \nabla f(x)^\top(y - x) + \tfrac L2\|y - x\|^2$ and that gradient descent on a quadratic is stable for steps below $2/L$ (→ ch 03 · topic 05)
- [ ] you know what linear and sublinear convergence mean, and how to read a rate off a log plot (→ ch 03 · topic 01)
- [ ] you know the first-order condition for convexity, $f(y) \ge f(x) + \nabla f(x)^\top(y - x)$, and strong convexity (→ topic 01)
- [ ] you can diagonalize a symmetric matrix and work in its eigenbasis (→ ch 02 · topic 05)

## core ideas

### idea 1: the loop, and what it guarantees

gradient descent repeats one line, $x_{k+1} = x_k - \eta\nabla f(x_k)$, until the gradient is small: $\|\nabla f(x_k)\|_2 \le \text{tol}$. small gradient is the right stopping test because it's what a minimum looks like locally, and for a convex function it's a certificate (topic 01: a stationary point is a global minimum). "the step was tiny" is *not* a good test: a too-small learning rate makes every step tiny, far from any minimum.

a real optimizer also needs to report what happened. every minimizer in `mlfs.optim` returns an `OptimResult` record: the final `x`, its value `fun`, the number of steps `n_iters`, whether the gradient test passed (`converged`), and the `history` of $f$ values, starting with $f(x_0)$. it stops early if $f$ becomes `inf` or `nan` (diverged) or after `max_iters` steps, and it can call a `callback(x, k)` after step $k$ so you can record iterates or plot progress. the iterates should be *new* arrays at each step (never `x -= ...`), so a callback that keeps them sees the actual path.

how fast does the loop converge? with step $\eta = 1/L$ on an $L$-smooth function, there are three classic answers, one per class of function:

| $f$ is | guarantee after $k$ steps | rate |
|---|---|---|
| smooth (possibly nonconvex), bounded below | $\min_{j<k}\|\nabla f(x_j)\|^2 \le \frac{2L(f(x_0) - f^*)}{k}$ | gradient $\to 0$ like $1/\sqrt k$ |
| smooth and convex | $f(x_k) - f^* \le \frac{L\|x_0 - x^*\|^2}{2k}$ | sublinear, $O(1/k)$ |
| smooth and $\mu$-strongly convex | $f(x_k) - f^* \le \big(1 - \frac\mu L\big)^k\big(f(x_0) - f^*\big)$ | linear (geometric) |

the first you proved in ch 03 · topic 05. the second and third are on paper 1 and 2. the third is the one that matters most in practice: the error shrinks by a constant factor per step, and the factor is set by $\kappa = L/\mu$. to shrink the error by $\epsilon$ you need about $k \approx \kappa\ln(1/\epsilon)$ steps (from $(1 - 1/\kappa)^k \approx e^{-k/\kappa}$). warm-up 1: $\kappa = 100$, $\epsilon = 10^{-6}$ gives $100 \times 13.8 \approx 1{,}400$ iterations. ten times worse conditioning costs ten times more iterations.

the middle row is where warm-up 5 lives. $x^4$ is convex but has zero curvature at its minimum, so it isn't strongly convex near 0, and gradient descent slows down as it gets close: the steps $4\eta x^3$ shrink faster than the distance $x$. you get a sublinear rate (covered by the $O(1/k)$ guarantee), not a geometric one.

> **▶ build: stage 1.** implement `OptimResult` (a plain record that converts its fields to the right types) and `gradient_descent` with a fixed step (`line_search=False`): call `grad` once per iterate, append $f$ after every step, call `callback(x, k)`, and stop on convergence, divergence or `max_iters`.

### idea 2: backtracking: let the function choose the step

the rates above assume you know $L$. you almost never do (ch 03 · topic 05 showed how badly sampled estimates can underestimate it), and a single global $L$ is pessimistic anyway, since curvature varies. the fix is to choose the step at every iteration by trying it.

the simplest rule, "accept any step that decreases $f$", isn't enough (warm-up 4). steps that decrease $f$ by a vanishing amount can converge to a point that isn't a minimum, so you demand a **sufficient decrease**, proportional to the step and the slope. along a direction $d$ with slope $\nabla f(x)^\top d < 0$, the **Armijo condition** is

$$f(x + \alpha d) \le f(x) + c\,\alpha\,\nabla f(x)^\top d,\qquad 0 < c < 1.$$

the right side is the tangent line at $x$, tilted up by the factor $c$ (typically $10^{-4}$). near $\alpha = 0$ the function follows its tangent, which lies below the tilted line, so small steps always pass; large steps that overshoot the valley fail. **backtracking** finds a step that passes, starting large: try $\alpha_0$, then $\rho\alpha_0$, $\rho^2\alpha_0$, ... (with $\rho = 1/2$, say), and return the first one that passes.

does it terminate? the descent lemma says that along $d = -\nabla f$, any $\alpha \le 2(1 - c)/L$ passes (on paper 3). so backtracking stops after a bounded number of halvings, at a step no smaller than $\min\big(\alpha_0, 2\rho(1 - c)/L\big)$: within a constant factor of $1/L$, without ever knowing $L$. that's enough to keep all three rates from idea 1 (with slightly worse constants). in floating point there's one more case: if $f$ is so flat (or so non-smooth) that no trial step shows a measurable decrease, give up after `max_iters` trials and return $0$ so the caller can stop.

two details make it cheap. you already know $f(x)$ and $\nabla f(x)$ when you call the line search, so pass them in: each trial then costs exactly one evaluation of $f$. and a trial where $f$ overflows to `inf` or `nan` simply fails the test (`nan <= anything` is `False`), so backtracking also protects you from steps that blow up.

> **🔮 predict:** $f(x) = x_1^2 + 5x_2^2$ at $(1, 1)$, direction $-\nabla f = (-2, -10)$, starting from $\alpha_0 = 1$ and halving. which step does backtracking accept?

> **▶ build: stage 2.** implement `backtracking_line_search(f, x, fx, gx, direction, *, alpha0, rho, c, max_iters)`: return the first $\alpha_0\rho^j$ that passes the Armijo condition, or `0.0` after `max_iters` failures. one call of `f` per trial.

### idea 3: gradient descent with a line search

put the two together: at each iterate, compute $g = \nabla f(x)$, call the line search along $d = -g$ starting from $\alpha_0 = $ `lr`, and step by the accepted $\alpha$. three things change compared with a fixed step.

first, **$f$ never increases**: every accepted step decreases it. so the method can't diverge, and a too-large `lr` only costs a few extra function evaluations per iteration while backtracking shrinks it.

second, **the step adapts to local curvature**. where the function is gently curved, large steps pass; in a sharp region, backtracking shrinks them. ch 03 · topic 05's lab showed that the curvature near a minimum is often far below the global $L$; a line search exploits that automatically.

third, **the cost per iteration varies**. a fixed step costs one gradient and one function value per iteration; a line search costs one gradient and *at least* one function value. that's why comparisons between optimizers count function and gradient evaluations, not just iterations (the capstone does).

the line search also gives the loop a new way to end: if it returns $0$, no step along $-g$ makes measurable progress. on a smooth function that happens only at the rounding floor near a minimum; on a non-smooth one like $|x|$, whose gradient is $\pm1$ right up to the minimum, it happens as soon as $|x|$ is around $10^{-15}$. either way the right response is to stop and report `converged=False` unless the gradient test passed.

> **▶ build: stage 3.** fill in the `line_search=True` branch of `gradient_descent`: backtracking with `alpha0=lr` at every step, stopping (not converged) if it returns 0.

### idea 4: the condition number and the zig-zag

the most a line search could ever do is pick the *best* step along $-g$, the one that minimizes $f(x - \alpha g)$: an **exact line search**. on a quadratic $f(x) = \tfrac12x^\top Ax - b^\top x$ with gradient $g = Ax - b$, it has a closed form,

$$\alpha^* = \frac{g^\top g}{g^\top Ag},$$

(set $\frac{d}{d\alpha}f(x - \alpha g) = -g^\top(A(x - \alpha g) - b) = -g^\top g + \alpha\,g^\top Ag$ to zero). it's the inverse of the curvature along $g$. you might expect the best step to beat any fixed one by a lot. it doesn't (warm-up 2), and the reason is geometric.

at the exact minimizer along the line, the directional derivative is zero: the *new* gradient is perpendicular to the old search direction. so successive steps are at right angles (warm-up 3). on a round bowl ($\kappa = 1$) the first step lands on the minimum. on a long, narrow valley, every step heads mostly across the valley (that's where the gradient points), overshoots to the opposite wall, and turns 90°: a zig-zag, with only a small component of each step along the valley floor.

how small? the **Kantorovich inequality** gives the worst case: with exact line search,

$$f(x_{k+1}) - f^* \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^2\big(f(x_k) - f^*\big).$$

for large $\kappa$, $\big(\frac{\kappa - 1}{\kappa + 1}\big)^2 \approx 1 - 4/\kappa$: still $O(\kappa)$ iterations per factor of $e$, the same order as the fixed step $1/L$. the bound is attained: in the eigenbasis of $A$, starting from $x_0 - x^* \propto (1/\lambda_{\min}, 1/\lambda_{\max})$ (only the two extreme eigen-directions), every step reduces the error by exactly that factor and flips the sign of the second coordinate (the worked example). the best *fixed* step, $2/(\mu + L)$ (ch 03 · topic 05), shrinks the distance by $(\kappa - 1)/(\kappa + 1)$ per step, so $f - f^*$ by its square: the same factor. **no choice of step size along the gradient can escape the condition number.** to beat it you have to change the *direction*: by remembering previous steps (momentum, topic 03), by rescaling coordinates (adaptive methods, topic 04), or by using curvature (Newton and quasi-Newton, topic 05).

> **🔮 predict:** on $f = \tfrac12(x_1^2 + 10x_2^2)$ from $(10, 1)$, exact line search: where are the first two iterates, and by what factor does $f$ shrink per step?

> **▶ build: stage 4.** implement `exact_step(A, g)` and `steepest_descent_quadratic(A, b, x0, n_iters)`, which runs gradient descent with the exact step on a quadratic and returns every iterate.

## worked example

**backtracking.** $f(x) = x_1^2 + 5x_2^2$, i.e. $A = \operatorname{diag}(2, 10)$, at $x = (1, 1)$: $f = 6$, $g = (2, 10)$, $d = -g$, slope $g^\top d = -104$. with $c = 10^{-4}$ the Armijo threshold is $6 - 0.0104\,\alpha$.

| $\alpha$ | $x + \alpha d$ | $f$ | passes? |
|---|---|---|---|
| 1 | $(-1, -9)$ | 406 | no |
| 0.5 | $(0, -4)$ | 80 | no |
| 0.25 | $(0.5, -1.5)$ | 11.5 | no |
| 0.125 | $(0.75, -0.25)$ | 0.875 | yes |

`backtracking_line_search` returns $0.125$ after 4 calls of $f$, and `gradient_descent(..., lr=1.0, line_search=True)` takes its first step to $(0.75, -0.25)$. the exact step here is $\alpha^* = 104/(2\cdot4 + 10\cdot100) = 104/1008 \approx 0.103$, with $f = 0.635$: backtracking accepted a slightly longer, slightly worse step, which is all it promises. (a fixed step must stay below $2/L = 0.2$; backtracking found a good one from $\alpha_0 = 1$.)

**fixed step.** with $\eta = 0.1$ on the same function, the second coordinate is multiplied by $1 - 10 \cdot 0.1 = 0$: it's solved in one step. the first is multiplied by $1 - 0.2 = 0.8$ each step: $(1, 1) \to (0.8, 0) \to (0.64, 0) \to (0.512, 0)$, with $f = 6, 0.64, 0.4096, 0.262$. the gradient is $(2\cdot0.8^k, 0)$, so the test $\|g\| \le 10^{-6}$ first passes at $k = 66$.

**the zig-zag.** $A = \operatorname{diag}(1, 10)$, $b = 0$, $x_0 = (10, 1)$, which is $\propto (1/\lambda_{\min}, 1/\lambda_{\max})$. then $g_0 = (10, 10)$, $\alpha^* = 200/(100 + 1000) = 2/11$, and

$$x_1 = (10, 1) - \tfrac2{11}(10, 10) = \left(\tfrac{90}{11}, -\tfrac{9}{11}\right) = \tfrac{9}{11}(10, -1).$$

$x_1$ is the same shape as $x_0$ with the second coordinate flipped, so the next step is the mirror image, again with $\alpha^* = 2/11$: $x_k = (9/11)^k(10, (-1)^k)$. $f(x_0) = \tfrac12(100 + 10) = 55$, and each step multiplies $f$ by $(9/11)^2 \approx 0.669 = \big(\frac{\kappa - 1}{\kappa + 1}\big)^2$ with $\kappa = 10$: the Kantorovich bound, with equality. successive gradients $g_0 = (10, 10)$ and $g_1 = \tfrac9{11}(10, -10)$ are orthogonal. your `steepest_descent_quadratic` must reproduce these iterates.

## on paper

1. **the $O(1/k)$ rate.** for convex, $L$-smooth $f$ with step $1/L$, show $f(x_k) - f^* \le \frac{L\|x_0 - x^*\|^2}{2k}$. use two facts: the descent lemma, $f(x_{j+1}) \le f(x_j) - \frac1{2L}\|g_j\|^2$, and convexity, $f(x_j) - f^* \le g_j^\top(x_j - x^*)$. hint: show $f(x_{j+1}) - f^* \le \frac L2\big(\|x_j - x^*\|^2 - \|x_{j+1} - x^*\|^2\big)$ and sum.

<details><summary>solution</summary>

with $x_{j+1} = x_j - g_j/L$: $\|x_{j+1} - x^*\|^2 = \|x_j - x^*\|^2 - \frac2Lg_j^\top(x_j - x^*) + \frac1{L^2}\|g_j\|^2$, so $g_j^\top(x_j - x^*) - \frac1{2L}\|g_j\|^2 = \frac L2\big(\|x_j - x^*\|^2 - \|x_{j+1} - x^*\|^2\big)$. combining the two facts, $f(x_{j+1}) - f^* \le g_j^\top(x_j - x^*) - \frac1{2L}\|g_j\|^2$, which is the hint. summing $j = 0, \dots, k - 1$ telescopes the right side to at most $\frac L2\|x_0 - x^*\|^2$. the left side is a sum of $k$ terms, and since $f(x_j)$ decreases, each is at least $f(x_k) - f^*$. so $k\big(f(x_k) - f^*\big) \le \frac L2\|x_0 - x^*\|^2$.

</details>

2. **the linear rate.** a $\mu$-strongly convex $f$ satisfies $\|\nabla f(x)\|^2 \ge 2\mu\big(f(x) - f^*\big)$ (the Polyak–Łojasiewicz inequality). prove it, then combine it with the descent lemma to get $f(x_{k+1}) - f^* \le (1 - \mu/L)\big(f(x_k) - f^*\big)$ for step $1/L$.

<details><summary>solution</summary>

strong convexity gives $f(y) \ge f(x) + g^\top(y - x) + \frac\mu2\|y - x\|^2$ for all $y$. minimize both sides over $y$: the left gives $f^*$, the right is minimized at $y = x - g/\mu$ with value $f(x) - \frac1{2\mu}\|g\|^2$. so $f^* \ge f(x) - \frac1{2\mu}\|g\|^2$, which is PL. then $f(x_{k+1}) - f^* \le f(x_k) - f^* - \frac1{2L}\|g_k\|^2 \le f(x_k) - f^* - \frac\mu L\big(f(x_k) - f^*\big)$.

</details>

3. **backtracking terminates.** using the descent lemma, show that the Armijo condition along $d = -\nabla f(x)$ holds for every $\alpha \in (0, 2(1 - c)/L]$. deduce that backtracking from $\alpha_0$ with factor $\rho$ returns $\alpha \ge \min(\alpha_0, 2\rho(1 - c)/L)$.

<details><summary>solution</summary>

the descent lemma with $y = x - \alpha g$ gives $f(x - \alpha g) \le f(x) - \alpha\|g\|^2 + \frac L2\alpha^2\|g\|^2 = f(x) - \alpha\big(1 - \frac{L\alpha}2\big)\|g\|^2$. Armijo needs $f(x - \alpha g) \le f(x) - c\alpha\|g\|^2$, which holds whenever $1 - L\alpha/2 \ge c$, i.e. $\alpha \le 2(1 - c)/L$. backtracking returns the first trial in this interval, or $\alpha_0$ if that already passes; the trial before the accepted one was larger than $2(1 - c)/L$ (else it would have passed), so the accepted one is more than $\rho\cdot2(1 - c)/L$.

</details>

4. **exact line search on a quadratic.** derive $\alpha^* = g^\top g/g^\top Ag$, show that the next gradient $g_{k+1} = g_k - \alpha^*Ag_k$ is orthogonal to $g_k$, and check that with $A = \operatorname{diag}(\lambda_1, \lambda_2)$, $b = 0$ and $x_0 = (1/\lambda_1, 1/\lambda_2)$, the first step gives $x_1 = \frac{\lambda_2 - \lambda_1}{\lambda_2 + \lambda_1}(1/\lambda_1, -1/\lambda_2)$.

<details><summary>solution</summary>

$\frac{d}{d\alpha}f(x - \alpha g) = -\nabla f(x - \alpha g)^\top g = -(g - \alpha Ag)^\top g$, zero at $\alpha^* = g^\top g/g^\top Ag$ (a minimum, since $f$ is a convex quadratic along the line). then $g_{k+1}^\top g_k = g^\top g - \alpha^*g^\top Ag = 0$. with $x_0 = (1/\lambda_1, 1/\lambda_2)$: $g_0 = Ax_0 = (1, 1)$, $g^\top g = 2$, $g^\top Ag = \lambda_1 + \lambda_2$, so $\alpha^* = 2/(\lambda_1 + \lambda_2)$ and $x_1 = \big(\frac1{\lambda_1} - \frac{2}{\lambda_1 + \lambda_2}, \frac1{\lambda_2} - \frac2{\lambda_1 + \lambda_2}\big) = \frac{\lambda_2 - \lambda_1}{\lambda_1 + \lambda_2}\big(\frac1{\lambda_1}, -\frac1{\lambda_2}\big)$. the distance shrinks by $\frac{\kappa - 1}{\kappa + 1}$ and $f = \frac12x^\top Ax$ by its square, with the same shape (and a sign flip) every step.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how the zig-zag and the iteration count depend on the condition number $\kappa$, for a fixed step and for your exact line search.
2. measured convergence factors against the theory, for the step $1/L$, the step $2/(\mu + L)$, backtracking and exact line search.
3. gradient descent on logistic regression when the data are linearly separable: what happens to the loss, the weights and their direction.

## check your understanding

1. why is "the gradient norm is small" a better stopping test than "the step is small"?

<details><summary>answer</summary>

a small step can just mean a small learning rate, far from any minimum. a small gradient means the function is locally flat, which is what a minimum looks like, and for a convex function $\|\nabla f\|$ bounds the suboptimality (with strong convexity, $f - f^* \le \|\nabla f\|^2/2\mu$ by the PL inequality).

</details>

2. (the predict callout in idea 2.) which step does backtracking accept for $x_1^2 + 5x_2^2$ at $(1, 1)$?

<details><summary>answer</summary>

$0.125$, after rejecting 1, 0.5 and 0.25 (see the worked example). note that $0.125 < 2/L = 0.2$: backtracking always lands inside the stable range within a factor of $\rho$.

</details>

3. with $c$ close to 1, what goes wrong with Armijo backtracking? with $c$ close to 0?

<details><summary>answer</summary>

with $c \to 1$ you demand the full decrease the tangent line predicts, which only infinitesimal steps achieve on a curved function: backtracking shrinks the step far too much. with $c \to 0$ any decrease passes, which allows steps that barely help (in theory, convergence to non-stationary points). $c = 10^{-4}$ is the usual compromise: almost any decrease, but proportional to the slope.

</details>

4. (the predict callout in idea 4.) iterates of exact line search on $\tfrac12(x_1^2 + 10x_2^2)$ from $(10, 1)$?

<details><summary>answer</summary>

$x_1 = \frac9{11}(10, -1)$, $x_2 = \frac{81}{121}(10, 1)$: the same shape every step, the second coordinate flipping sign. $f$ shrinks by $(9/11)^2 \approx 0.67$ per step.

</details>

5. the fixed step $2/(\mu + L)$ and exact line search have the same worst-case rate. so why use a line search at all?

<details><summary>answer</summary>

the fixed step needs $\mu$ and $L$, which you don't know, and a too-large step diverges. a line search needs neither, never increases $f$, and adapts when the local curvature differs from the global worst case (often by a lot). exact line searches are rarely worth their cost outside quadratics; backtracking gets most of the benefit for a few extra function evaluations.

</details>

6. gradient descent has made $f$ go down by a factor of 2 every 100 iterations for a while. what does that tell you about $\kappa$ near the current iterate?

<details><summary>answer</summary>

a linear rate $r$ per step with $r^{100} = 0.5$ means $r \approx 0.9931$. with a step near $1/L$, $r \approx 1 - 1/\kappa$, so $\kappa \approx 145$ in the region the iterates are in (with exact line search, $r \approx 1 - 4/\kappa$, so $\kappa \approx 580$). the measured rate is a cheap estimate of the local conditioning.

</details>

7. why does gradient descent on $x^4$ converge sublinearly, when it converges linearly on $x^2$?

<details><summary>answer</summary>

with a fixed step, $x_{k+1} = x_k - 4\eta x_k^3 = x_k(1 - 4\eta x_k^2)$: the contraction factor $1 - 4\eta x_k^2 \to 1$ as $x_k \to 0$, so the steps shrink faster than the error. at the minimum the curvature is 0 ($\mu = 0$, $\kappa = \infty$). the error decays like $1/\sqrt k$ (solve $dx/dk = -4\eta x^3$), so $f = x^4$ like $1/k^2$: sublinear, inside the $O(1/k)$ guarantee from idea 1.

</details>

## common bugs

- **updating `x` in place** (`x -= lr * g`). symptom: a callback that stores the iterates gets the same array $k$ times, or the caller's `x0` changes. build a new array each step.
- **recomputing $f(x)$ inside the line search.** symptom: one extra function call per iteration (counted in the capstone). pass `fx` in.
- **an infinite backtracking loop.** symptom: the program hangs on a non-smooth function or near the rounding floor. cap the trials at `max_iters` and return `0.0`.
- **forgetting the sign of the slope.** symptom: Armijo accepts every step (or none). the slope $\nabla f^\top d$ is negative for a descent direction, so the threshold $f(x) + c\alpha\nabla f^\top d$ is *below* $f(x)$.
- **checking convergence on the old gradient.** symptom: `converged` is `True` for a result whose own gradient is large, or the loop calls `grad` twice per step. compute the gradient once per iterate and use it for both the test and the step.
- **reporting `converged=True` after `max_iters`.** symptom: callers trust an answer that isn't one. `converged` means the gradient test passed at the returned point, nothing else.

## going deeper

- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer, chapter 3 (line searches, Armijo and Wolfe conditions, the rate of steepest descent). [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Nesterov (2018), *Lectures on Convex Optimization*, 2nd ed., Springer, §2.1.5 (the gradient method for smooth convex and strongly convex functions). [doi:10.1007/978-3-319-91578-4](https://doi.org/10.1007/978-3-319-91578-4)
- Karimi, Nutini & Schmidt (2016), "Linear convergence of gradient and proximal-gradient methods under the Polyak–Łojasiewicz condition", *ECML PKDD*. [arXiv:1608.04636](https://arxiv.org/abs/1608.04636)
- Soudry, Hoffer, Nacson, Gunasekar & Srebro (2018), "The implicit bias of gradient descent on separable data", *JMLR* 19(70). what lab experiment 3 shows. [arXiv:1710.10345](https://arxiv.org/abs/1710.10345)
- Boyd & Vandenberghe (2004), *Convex Optimization*, §9.2–9.3 (descent methods, backtracking, the effect of condition number). [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/)

## connections

- ← builds on: the descent lemma and the $2/L$ threshold (ch 03 · topic 05), convergence rates (ch 03 · topic 01), convexity and strong convexity (topic 01), eigen-decompositions (ch 02 · topic 05).
- → used in: momentum (topic 03) and second-order methods (topic 05) reuse `OptimResult` and `backtracking_line_search`; `minimize` dispatches to `gradient_descent` (topic 05); linear and logistic regression with `solver='gd'` (ch 08); learning-rate finders and schedules (ch 15).
