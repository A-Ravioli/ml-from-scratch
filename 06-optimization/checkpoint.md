# 06 · optimization · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. a function is convex on a convex set, and you've found a point where its gradient is zero. what can you conclude? what if you only know that its Jensen gap is nonnegative on 1,000 random chords?

<details><summary>answer</summary>

the point is a global minimizer (first-order condition: $f(y) \ge f(x) + \nabla f(x)^\top(y - x) = f(x)$). the sampled Jensen check proves nothing: a numerical convexity test can only refute convexity, by finding a chord below the graph. (topic 01, idea 4)

</details>

2. is $\max(0, 1 - y\,w^\top x)$ convex in $w$? is $(\sigma(w^\top x) - y)^2$?

<details><summary>answer</summary>

the hinge loss is a maximum of two affine functions of $w$, so convex. the squared error of a sigmoid output isn't: the sigmoid isn't affine, and the composition rule fails (it has flat plateaus with nonconvex bends between them). (topic 01, idea 4 and lab)

</details>

3. gradient descent with step $1/L$ on a $\mu$-strongly convex, $L$-smooth function: how many iterations, roughly, to gain 6 digits when $\kappa = 1{,}000$? and with tuned momentum?

<details><summary>answer</summary>

the error shrinks like $(1 - 1/\kappa)^k$, so about $\kappa\ln(10^6) \approx 14{,}000$ iterations. tuned heavy ball or Nesterov have rate $1 - O(1/\sqrt\kappa)$: about $\sqrt\kappa\ln(10^6) \approx 440$ (heavy ball, rate $1 - 2/\sqrt\kappa$, needs about half of that). (topics 02–03)

</details>

4. what does the Armijo condition demand, and why isn't "any decrease" enough?

<details><summary>answer</summary>

$f(x + \alpha d) \le f(x) + c\alpha\nabla f(x)^\top d$: a decrease proportional to the step and the slope. accepting any decrease allows steps that decrease $f$ by vanishing amounts, which can stall far from a minimum. Armijo backtracking always terminates on a smooth function, with a step at least $\min(\alpha_0, 2\rho(1 - c)/L)$. (topic 02, idea 2)

</details>

5. with an exact line search on a quadratic, successive gradients are orthogonal. why does that make the method slow on ill-conditioned problems?

<details><summary>answer</summary>

each step goes mostly across the narrow valley (where the gradient points), overshoots to the far wall, and the next step turns 90°: a zig-zag with only a small component along the valley floor. the rate is $\big(\frac{\kappa - 1}{\kappa + 1}\big)^2 \approx 1 - 4/\kappa$ per step on $f$: still $O(\kappa)$ iterations, no better than the best fixed step. (topic 02, idea 4)

</details>

6. why can no first-order method beat $O(1/k^2)$ on smooth convex functions in general?

<details><summary>answer</summary>

on Nesterov's tridiagonal worst-case quadratic, each gradient reveals one more coordinate, so after $k$ steps from 0 any method whose iterates lie in the span of past gradients has touched only $k$ coordinates; even the best point in that subspace is $\frac{3L\|x_0 - x^*\|^2}{32(k + 1)^2}$ above the optimum (with $n = 2k + 1$). Nesterov's method matches this up to a constant. (topic 03, idea 4)

</details>

7. in the `mlfs.optim` protocol, why must `step()` update `p.data` in place, and what does `zero_grad()` do?

<details><summary>answer</summary>

models (and later chapters' `Tensor`s) hold references to their weight arrays; rebinding `p.data` to a new array would leave them pointing at the old, untrained weights. `zero_grad()` sets every `.grad` to `None`, and `step()` skips parameters whose gradient is `None` (no decay, no momentum, no state update). (topic 04, idea 1)

</details>

8. constant-step SGD on a strongly convex problem stops improving at some loss level. what sets that level, and how do you get below it?

<details><summary>answer</summary>

the gradient noise: the expected excess loss settles at about $\frac\eta4\sum_is_i$ (for small $\eta a_i$), proportional to the learning rate and to the noise variance, which falls like $1/b$ with the batch size. decay the learning rate (Robbins–Monro), increase the batch, or average the iterates. (topic 04, idea 5)

</details>

9. what does Adam's bias correction fix, and how big is Adam's first step?

<details><summary>answer</summary>

the moving averages $m$ and $v$ start at 0, so early on they underestimate the gradient's mean and second moment by factors $1 - \beta_1^t$ and $1 - \beta_2^t$; dividing by those removes the bias. after one step $\hat m = g$, $\hat v = g^2$, so the step is $\eta\,\text{sign}(g)$ in every coordinate, whatever the gradient's size. (topic 04, idea 4)

</details>

10. why does Adam beat SGD on an axis-aligned ill-conditioned quadratic, and lose that advantage after a random rotation?

<details><summary>answer</summary>

Adam divides each coordinate by an estimate of that coordinate's gradient scale: a diagonal preconditioner, which removes ill-conditioning that's aligned with the axes. a rotation mixes stiff and flat directions in every coordinate, so no per-coordinate scaling can separate them; SGD is rotation-invariant and doesn't care. (topic 04, lab)

</details>

11. Newton's method needs one step on any strongly convex quadratic. why use L-BFGS at all?

<details><summary>answer</summary>

a Newton step needs the $d \times d$ Hessian and an $O(d^3)$ solve, impossible for millions of parameters and expensive for thousands. L-BFGS uses only gradients, $O(md)$ memory and work per step, and still converges fast. far from a minimum, or where the Hessian is indefinite, Newton also needs safeguards (descent check, line search, damping). (topic 05, idea 4)

</details>

12. what is the secant condition, and why does BFGS need a Wolfe line search rather than Armijo alone?

<details><summary>answer</summary>

$H_{k+1}y = s$: the new inverse-Hessian approximation must map the observed gradient change to the step. the BFGS update keeps $H$ positive definite only if $y^\top s > 0$; the Wolfe curvature condition guarantees it ($y^\top s \ge \alpha(c_2 - 1)\nabla f^\top d > 0$), while Armijo only shortens steps and can accept ones with $y^\top s \le 0$. (topic 05, idea 2)

</details>

13. at the solution of $\min f$ subject to $Gx \le h$, a constraint is inactive. what is its multiplier, and what happens to the optimal value if you loosen that constraint slightly?

<details><summary>answer</summary>

zero (complementary slackness), and nothing: a multiplier is the sensitivity of the optimal value, $\partial f^*/\partial h_i = -\lambda_i$. in an SVM these are the points off the margin, which don't affect the classifier. (topic 06, ideas 4–5 and lab)

</details>

14. why does projected gradient descent stop on $\|x_{k+1} - x_k\| \le \eta\,\text{tol}$ rather than on $\|\nabla f\|$?

<details><summary>answer</summary>

at a constrained minimum the gradient generally isn't zero (it points out of the feasible set). the gradient mapping $(x - P(x - \eta\nabla f(x)))/\eta$ is zero exactly at constrained stationary points, and the distance moved is $\eta$ times its norm. (topic 06, idea 2)

</details>

15. gradient descent from a random start almost never converges to a strict saddle. why does perturbed gradient descent exist, then?

<details><summary>answer</summary>

"almost never converges to" isn't "escapes quickly": from a distance $\delta$ off the saddle's stable manifold, escaping takes about $\ln(1/\delta)/(\eta\gamma)$ steps, and paths can pass many saddles. a random kick of radius $r$ when the gradient is small bounds the time near each saddle by about $\ln(1/r)/(\eta\gamma)$. kicks don't help with genuine local minima (all curvatures positive), unless they're large enough to jump basins. (topic 07, idea 2)

</details>

## 2. by hand (no computer)

1. **backtracking and a zig-zag.** (a) $f(x) = 2x_1^2 + x_2^2$ at $(1, 1)$, direction $-\nabla f$, $\alpha_0 = 1$, $\rho = \tfrac12$, $c = 10^{-4}$: which step does Armijo backtracking accept? (b) with an exact line search on $f = \tfrac12(x_1^2 + 4x_2^2)$ from $(4, 1)$, find $x_1$ and the factor by which $f$ shrinks per step.

<details><summary>solution</summary>

(a) $g = (4, 2)$, $f = 3$, slope $-20$. $\alpha = 1$: $(-3, -1)$, $f = 19$: fails. $\alpha = \tfrac12$: $(-1, 0)$, $f = 2 \le 3 - 0.001$: accepted, $\alpha = 0.5$. (b) $g_0 = (4, 4)$, $g^\top g = 32$, $g^\top Ag = 16 + 64 = 80$, $\alpha^* = 0.4$, $x_1 = (4, 1) - 0.4(4, 4) = (2.4, -0.6) = 0.6(4, -1)$. the start is $\propto(1/\lambda_1, 1/\lambda_2)$, the worst case, so every step multiplies the distance by $\frac{\kappa - 1}{\kappa + 1} = 0.6$ and $f$ by $0.36$ ($f_0 = 10$, $f_1 = 3.6$).

</details>

2. **momentum and Adam, one step at a time.** (a) SGD with momentum 0.5 and $\eta = 0.1$ on $f(x) = x^2$ from $x_0 = 1$ (PyTorch semantics, buffer $b = g$ on the first step): find $x_1$, $x_2$. (b) Adam with $\eta = 0.01$, default betas, on a parameter whose gradients are $g_1 = 4$, $g_2 = -4$: find the two steps.

<details><summary>solution</summary>

(a) $g_0 = 2$, $b = 2$, $x_1 = 1 - 0.2 = 0.8$. $g_1 = 1.6$, $b = 0.5 \cdot 2 + 1.6 = 2.6$, $x_2 = 0.8 - 0.26 = 0.54$. (b) step 1: $m = 0.4$, $v = 0.016$, $\hat m = 4$, $\hat v = 16$, step $-0.01 \cdot 4/4 = -0.01$. step 2: $m = 0.36 - 0.4 = -0.04$, $\hat m = -0.04/0.19 = -0.2105$; $v = 0.015984 + 0.016 = 0.031984$, $\hat v = 0.031984/0.001999 = 16.0$; step $-0.01 \cdot (-0.2105)/4 = +0.000526$. the momentum average nearly cancels, so the second step is tiny.

</details>

3. **a QP by KKT.** minimize $\tfrac12(x_1^2 + x_2^2) - x_1 - x_2$ subject to $x_1 + 2x_2 \le 1$. (a) is the constraint active at the unconstrained minimum? (b) solve the KKT conditions; (c) write the dual function $g(\lambda)$ and check that its maximum equals the optimal value.

<details><summary>solution</summary>

(a) the unconstrained minimizer is $(1, 1)$, with $1 + 2 = 3 > 1$: infeasible, so the constraint is active. (b) stationarity: $x - (1, 1) + \lambda(1, 2) = 0$, so $x = (1 - \lambda, 1 - 2\lambda)$; activity: $1 - \lambda + 2 - 4\lambda = 1$, so $\lambda = 0.4 \ge 0$ and $x^* = (0.6, 0.2)$, $f^* = \tfrac12(0.36 + 0.04) - 0.8 = -0.6$. (c) with $Q = I$, $c = (-1, -1)$, $G = (1\ 2)$, $h = 1$: $c + G^\top\lambda = (\lambda - 1, 2\lambda - 1)$, so $g(\lambda) = -\tfrac12\big((\lambda - 1)^2 + (2\lambda - 1)^2\big) - \lambda$. $g'(\lambda) = -(\lambda - 1) - 2(2\lambda - 1) - 1 = 2 - 5\lambda = 0$ at $\lambda = 0.4$, where $g = -\tfrac12(0.36 + 0.04) - 0.4 = -0.6 = f^*$. ✓

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **the stateful optimizer family** of topic 04, the part of `mlfs.optim` that the rest of the course trains with. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=06-optimization/04-stochastic-optimization
mkdir -p /tmp/rebuild-0604
git show HEAD:$T/exercise.py > /tmp/rebuild-0604/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-0604/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-0604/solutions/` and run `python scripts/stubify.py /tmp/rebuild-0604`.) open `/tmp/rebuild-0604/exercise.py` with `marimo edit`, implement `Param`, `Optimizer`, `minibatch_indices`, `SGD` (momentum, Nesterov, weight decay), `Adagrad`, `RMSProp`, `Adam` and `AdamW` from memory, and run the checks against your copy:

```bash
pytest /tmp/rebuild-0604/test_exercise.py -k "stage1 or stage2 or stage3 or stage4"
```

you pass if every check is green within the hour. (the checks compare every optimizer with PyTorch's update rules on random gradient sequences, including steps with missing gradients and a learning-rate change, so "almost right" fails.)

**harder variant** (for testing out of the chapter with confidence): in a further 60 minutes, rebuild topic 02's `backtracking_line_search` and `gradient_descent`, and topic 05's `wolfe_line_search`, `bfgs_update`, `two_loop` and `lbfgs`, each in its own scratch directory, and pass:

```bash
pytest /tmp/rebuild-0602/test_exercise.py -k "stage1 or stage2 or stage3"
pytest /tmp/rebuild-0605/test_exercise.py -k "stage2 or stage3"
```

(run them from the repo root: the notebooks import earlier topics through `mlfs.optim`, which the tests resolve to the reference solutions, so only the functions you rebuilt are being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each lettered part separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the topic and idea it comes from.
