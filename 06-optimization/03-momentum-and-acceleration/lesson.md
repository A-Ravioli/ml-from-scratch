# momentum & acceleration

## why this matters

topic 02 ended with a wall: on a problem with condition number $\kappa$, gradient descent needs about $\kappa\log(1/\epsilon)$ iterations, and no choice of step size changes that. for a mildly ill-conditioned problem with $\kappa = 10^4$ (common: think of features on wildly different scales, or a deep network's loss surface), that's tens of thousands of iterations per digit of accuracy.

momentum breaks through the wall with a one-line change: keep moving in the direction you were already moving. on a quadratic it cuts the iteration count from $\kappa\log(1/\epsilon)$ to about $\sqrt\kappa\log(1/\epsilon)$: for $\kappa = 10^4$, a hundred times fewer steps. every optimizer you'll use to train neural networks (SGD with momentum, Adam) has a momentum term. and then a remarkable fact: you can *prove* that no method that only sees gradients can do better than $\sqrt\kappa$ in the worst case. Nesterov's accelerated gradient method hits that limit, so in a precise sense it's an optimal algorithm. this topic builds both methods, predicts their speed exactly on quadratics, and constructs the function that makes every first-order method slow. it also shows where the predictions fail.

## warm-up: predict first

1. a ball rolls down a long, gently sloping valley with steep sides. compared with gradient descent, does giving it momentum help it go *along* the valley, *across* it, or both?
2. momentum $\beta = 0.9$ with step size $\eta$. on a long, constant slope, how big does each step eventually get?
3. heavy ball with step $0.1$ and momentum $0.5$ on $f(x) = 2x^2$. faster or slower than plain gradient descent with step $0.1$?
4. for $\kappa = 100$, how many times fewer iterations does well-tuned momentum need than gradient descent: 2, 10, or 100?
5. a method only ever sees gradients at points it has visited. can it reach a coordinate of $x$ that no gradient has touched yet?

## prerequisites check

- [ ] you know gradient descent's rates and why $\kappa$ controls them, and you've built `gradient_descent` and `OptimResult` (→ topic 02)
- [ ] you can analyze an iteration on a quadratic coordinate by coordinate in the eigenbasis (→ ch 03 · topic 05, on paper 3)
- [ ] you can solve a quadratic equation with complex roots and compute their modulus (→ ch 01)
- [ ] you know that the spectral radius of an iteration matrix decides whether it converges, and how fast (→ ch 02 · topic 05)

## core ideas

### idea 1: the heavy ball

gradient descent has no memory: each step looks only at the current gradient. picture what that does in a narrow valley. the gradient points mostly across the valley, so the iterate bounces between the walls, and the component along the floor is tiny. but notice: the *across* components alternate in sign from step to step, while the *along* components all point the same way.

Polyak's **heavy-ball** method adds a fraction of the previous step to the current one:

$$x_{k+1} = x_k - \eta\nabla f(x_k) + \beta(x_k - x_{k-1}),\qquad 0 \le \beta < 1,$$

with $x_{-1} = x_0$ (so the first step is a plain gradient step). think of a ball with mass rolling on the loss surface with friction: $\beta$ is how much velocity survives each step. unroll the recursion and the step is an exponentially weighted sum of past gradients, $x_{k+1} - x_k = -\eta\sum_{j=0}^k\beta^{k-j}\nabla f(x_j)$ (if the first step had no prior velocity). in that sum, alternating components cancel and consistent ones add up (warm-up 1: momentum helps *along* the valley by accumulating, and damps the bouncing *across* it). on a constant slope $g$ the step converges to $\eta g/(1 - \beta)$: with $\beta = 0.9$, ten times the plain step (warm-up 2). momentum is a way of taking big steps where the landscape is consistent without taking big steps where it's sharp.

it's not free. a ball with momentum can overshoot the minimum and oscillate around it, and with badly chosen parameters it converges *slower* than gradient descent (the worked example) or not at all.

> **▶ build: stage 1.** implement `heavy_ball(f, grad, x0, *, lr, momentum, max_iters, tol, callback)` with the same loop conventions as your `gradient_descent` (copy `x0`, one gradient per iterate, stop on convergence, divergence or `max_iters`, history and callback, new arrays every step), returning an `OptimResult`.

### idea 2: Nesterov's look-ahead

Nesterov's method changes one thing: it evaluates the gradient where momentum is about to take you, not where you are.

$$y_k = x_k + \beta_k(x_k - x_{k-1}),\qquad x_{k+1} = y_k - \eta\nabla f(y_k).$$

first move with the momentum, then correct with a gradient step from the look-ahead point. written as a single update, $x_{k+1} = x_k - \eta\nabla f(y_k) + \beta_k(x_k - x_{k-1})$: heavy ball with the gradient taken at $y_k$. if the momentum is carrying you up the far wall of a valley, the gradient at $y_k$ already points back, and the correction comes one step earlier. that small change makes Nesterov's method stable in situations where heavy ball isn't, and it's what makes the theory work beyond quadratics.

there are two standard choices of $\beta_k$:

- for **strongly convex** $f$ with known $\mu$ and $L$, a constant $\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$ with $\eta = 1/L$, which gives $f(x_k) - f^* = O\big((1 - 1/\sqrt\kappa)^k\big)$.
- for **convex** $f$ (no $\mu$ needed), the increasing schedule $\beta_k = \frac{k - 1}{k + 2}$ at step $k = 1, 2, \dots$ (so $0, \tfrac14, \tfrac25, \tfrac12, \dots \to 1$), with $\eta = 1/L$. this gives

$$f(x_k) - f^* \le \frac{2L\|x_0 - x^*\|^2}{(k + 1)^2},$$

$O(1/k^2)$ instead of gradient descent's $O(1/k)$: a hundred steps of Nesterov do the work of ten thousand gradient steps. (the proof uses an "estimate sequence" or a Lyapunov function; see going deeper.)

accelerated methods pay for their speed with **non-monotonicity**: $f(x_k)$ doesn't always decrease. with the schedule, the momentum grows towards 1, the iterates overshoot, and $f$ ripples on the way down. the lab looks at those ripples, and at the simple fix (restarting the momentum).

> **🔮 predict:** Nesterov with constant momentum $0.5$ and step $0.1$ on $f(x) = 2x^2$ from $x_0 = 1$. what are $x_1$, $x_2$, $x_3$?

> **▶ build: stage 2.** implement `nesterov(f, grad, x0, *, lr, momentum=None, ...)`: `momentum=None` means the schedule $(k - 1)/(k + 2)$, a number means that constant (including `0.0`, which is plain gradient descent). the stopping test uses the gradient at $x_k$, so each step costs two gradients.

### idea 3: exact rates on a quadratic

on a quadratic $f(x) = \tfrac12x^\top Ax - b^\top x$, both methods are linear iterations, and you can predict their speed exactly, as you did for gradient descent in ch 03 · topic 05. in $A$'s eigenbasis the coordinates decouple. along an eigenvector with eigenvalue $\lambda$, the error $e_k$ satisfies a two-term recurrence,

$$e_{k+1} = a\,e_k - b\,e_{k-1},\qquad\begin{cases}\text{heavy ball:} & a = 1 + \beta - \eta\lambda,\ \ b = \beta\\ \text{Nesterov:} & a = (1 + \beta)(1 - \eta\lambda),\ \ b = \beta(1 - \eta\lambda)\end{cases}$$

(on paper 1). a recurrence like this is solved by $e_k = c_1z_1^k + c_2z_2^k$, where $z_{1,2}$ are the roots of $z^2 - az + b = 0$. so the error shrinks like $\max(|z_1|, |z_2|)^k$, and the method's rate on the whole quadratic is the **spectral radius** $\rho = \max_\lambda\max(|z_1|, |z_2|)$ over $A$'s eigenvalues. when the roots are complex ($a^2 < 4b$), they have equal modulus $\sqrt b$ and the error spirals: that's the oscillation of an underdamped ball.

this formula turns tuning into algebra. for heavy ball, the best choice makes the roots complex with the *same* modulus $\sqrt\beta$ for every $\lambda \in [\mu, L]$ (on paper 2):

$$\eta = \frac{4}{(\sqrt L + \sqrt\mu)^2},\qquad \beta = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2,\qquad \rho = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} \approx 1 - \frac{2}{\sqrt\kappa}.$$

compare gradient descent's best, $\frac{\kappa - 1}{\kappa + 1} \approx 1 - \frac2\kappa$: the $\kappa$ became $\sqrt\kappa$. for Nesterov with $\eta = 1/L$ and $\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$, the rate is $1 - 1/\sqrt\kappa$ (the smallest eigenvalue gives a double root there). for $\kappa = 100$: gradient descent with $1/L$ shrinks the error by $0.99$ per step, Nesterov by $0.9$, heavy ball by $0.82$. to gain six digits that's about 1,400, 130 and 70 iterations (warm-up 4: 10 to 20 times fewer).

two cautions. these rates need $\mu$ and $L$, which you rarely know; a misestimated $\mu$ costs speed (or causes ringing). and the heavy-ball analysis is *only* for quadratics: on a strongly convex function that isn't quadratic, heavy ball with these "optimal" parameters can fail to converge at all, while Nesterov's method keeps its guarantee. the lab shows a 1-D example where heavy ball goes round a cycle forever.

> **▶ build: stage 3.** implement `momentum_spectral_radius(eigs, lr, momentum, *, nesterov=False)` (the larger root modulus for every eigenvalue, vectorized with complex square roots; the max over eigenvalues) and `tuned_momentum(L, mu, *, nesterov=False)`, the textbook parameters above.

### idea 4: the lower bound: why $\sqrt\kappa$ is the best possible

is there a cleverer method that beats $\sqrt\kappa$, or $1/k^2$? not if it only uses gradients. call a method **first-order** if every iterate lies in $x_0 + \operatorname{span}\{\nabla f(x_0), \dots, \nabla f(x_{k-1})\}$. gradient descent, heavy ball, Nesterov and any fixed mix of them qualify.

Nesterov's argument is a function designed so that gradients reveal information as slowly as possible. take $f(x) = \frac L4\big(\tfrac12x^\top Tx - x_1\big)$ in $n$ dimensions, where $T$ is tridiagonal with 2 on the diagonal and $-1$ beside it: $\tfrac12x^\top Tx = \tfrac12\big(x_1^2 + \sum_{i=1}^{n-1}(x_i - x_{i+1})^2 + x_n^2\big)$, a chain of springs. its gradient $\frac L4(Tx - e_1)$ at a point whose nonzero entries are among the first $j$ coordinates has nonzero entries only among the first $j + 1$: each gradient reaches **one coordinate further down the chain** (warm-up 5: no, it can't). starting from $x_0 = 0$, after $k$ steps a first-order method has touched only $x_1, \dots, x_k$, while the minimizer $x^*_i = 1 - i/(n + 1)$ is spread over all $n$ coordinates. even the *best* point in that $k$-dimensional subspace is far from optimal. working it out (on paper 3) with $n = 2k + 1$ gives

$$f(x_k) - f^* \ge \frac{3L\|x_0 - x^*\|^2}{32(k + 1)^2}$$

for **every** first-order method. Nesterov's upper bound $\frac{2L\|x_0 - x^*\|^2}{(k + 1)^2}$ matches it up to the constant: acceleration is optimal. a similar construction for strongly convex functions shows that, in the worst case, no first-order method shrinks the squared distance $\|x_k - x^*\|^2$ by more than a factor $\big(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\big)^2$ per step: tuned heavy ball's rate on quadratics is exactly that. (the bound needs the dimension to be large compared with $k$. in low dimensions, methods that exploit the dimension, like conjugate gradients on a quadratic, finish in $n$ steps.)

> **🔮 predict:** run three steps of heavy ball from $x_0 = 0$ on this function in $n = 10$ dimensions. which coordinates of $x_3$ can be nonzero?

> **▶ build: stage 4.** implement `worst_case_quadratic(n, L)`, returning $(A, b)$ with $A = \frac L4T$ and $b = \frac L4e_1$, and `first_order_lower_bound(L, R, k)`. the checks run your `heavy_ball` and `nesterov` on it: watch the iterates fill in one coordinate per step.

## worked example

$f(x) = 2x^2$ (so $\nabla f = 4x$, $\lambda = 4$), from $x_0 = 1$, step $\eta = 0.1$. gradient descent multiplies $x$ by $1 - 0.4 = 0.6$ each step.

**heavy ball, $\beta = 0.5$.** the first step has no velocity: $x_1 = 1 - 0.4 = 0.6$. then

$$x_2 = 0.6 - 0.1(2.4) + 0.5(0.6 - 1) = 0.16,\qquad x_3 = 0.16 - 0.1(0.64) + 0.5(0.16 - 0.6) = -0.124.$$

it overshot zero. the recurrence has $a = 1 + 0.5 - 0.4 = 1.1$, $b = 0.5$; $a^2 - 4b = -0.79 < 0$, so the roots are complex with modulus $\sqrt{0.5} = 0.707$. that's *slower* than gradient descent's $0.6$ (warm-up 3): on a well-conditioned problem with a well-chosen step, momentum only adds oscillation.

**Nesterov, constant $\beta = 0.5$.** $y = x_0$ at the first step, so $x_1 = 0.6$. then $y = 0.6 + 0.5(0.6 - 1) = 0.4$ and $x_2 = 0.4 - 0.1 \cdot 1.6 = 0.24$; then $y = 0.24 + 0.5(0.24 - 0.6) = 0.06$ and $x_3 = 0.06 \cdot 0.6 = 0.036$. here $a = 1.5 \cdot 0.6 = 0.9$, $b = 0.5 \cdot 0.6 = 0.3$, complex roots of modulus $\sqrt{0.3} = 0.548$: faster than gradient descent. the look-ahead damps the overshoot.

**Nesterov, schedule.** $\beta_1 = 0$, $\beta_2 = \tfrac14$, $\beta_3 = \tfrac25$: $x_1 = 0.6$; $y = 0.6 - 0.1 = 0.5$, $x_2 = 0.3$; $y = 0.3 + 0.4(-0.3) = 0.18$, $x_3 = 0.108$.

**tuned, $\kappa = 100$** ($\mu = 1$, $L = 100$). heavy ball: $\eta = 4/121 \approx 0.0331$, $\beta = (9/11)^2 \approx 0.669$, rate $9/11 \approx 0.818$. Nesterov: $\eta = 0.01$, $\beta = 9/11 \approx 0.818$, rate $0.9$. gradient descent with $1/L$: rate $0.99$. your functions must reproduce all of these numbers.

## on paper

1. **the recurrence.** for heavy ball on $f = \tfrac12x^\top Ax$ (minimizer 0), write $x_k = \sum_ic_{k,i}q_i$ in $A$'s eigenbasis and show $c_{k+1} = (1 + \beta - \eta\lambda)c_k - \beta c_{k-1}$. do the same for Nesterov and get $c_{k+1} = (1 + \beta)(1 - \eta\lambda)c_k - \beta(1 - \eta\lambda)c_{k-1}$.

<details><summary>solution</summary>

$\nabla f = Ax$, and $Aq_i = \lambda_iq_i$, so each coordinate evolves separately. heavy ball: $c_{k+1} = c_k - \eta\lambda c_k + \beta(c_k - c_{k-1})$. Nesterov: the look-ahead coordinate is $d_k = c_k + \beta(c_k - c_{k-1})$, and $c_{k+1} = d_k - \eta\lambda d_k = (1 - \eta\lambda)\big((1 + \beta)c_k - \beta c_{k-1}\big)$.

</details>

2. **tuning heavy ball.** with $b = \beta$, the roots are complex with modulus $\sqrt\beta$ whenever $(1 + \beta - \eta\lambda)^2 \le 4\beta$, i.e. $(1 - \sqrt\beta)^2 \le \eta\lambda \le (1 + \sqrt\beta)^2$. choose $\eta$ and $\beta$ so that this holds exactly at the ends, $\eta\mu = (1 - \sqrt\beta)^2$ and $\eta L = (1 + \sqrt\beta)^2$, and derive the formulas in idea 3.

<details><summary>solution</summary>

$(1 + \beta - \eta\lambda)^2 \le 4\beta \iff |1 + \beta - \eta\lambda| \le 2\sqrt\beta \iff 1 + \beta - 2\sqrt\beta \le \eta\lambda \le 1 + \beta + 2\sqrt\beta$. dividing the two end conditions: $\kappa = \big(\frac{1 + \sqrt\beta}{1 - \sqrt\beta}\big)^2$, so $\sqrt\kappa(1 - \sqrt\beta) = 1 + \sqrt\beta$ and $\sqrt\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$. adding the square roots of the end conditions: $\sqrt{\eta}(\sqrt L + \sqrt\mu) = 2$, so $\eta = 4/(\sqrt L + \sqrt\mu)^2$. every eigenvalue in $[\mu, L]$ then has complex roots of modulus $\sqrt\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$, and making $\beta$ any smaller would push an end eigenvalue into real roots, one of them larger.

</details>

3. **the lower bound.** for the worst-case function with $L = 4$, i.e. $f = \tfrac12x^\top Tx - x_1$: (a) show by induction that a first-order method started at 0 has $x_k \in \operatorname{span}(e_1, \dots, e_k)$; (b) show that the minimum of $f$ over that subspace is $-\frac12\frac{k}{k + 1}$, and that $f^* = -\frac12\frac{n}{n + 1}$; (c) with $n = 2k + 1$, conclude $f(x_k) - f^* \ge \frac{1}{4(k + 1)}$; (d) using $\|x^*\|^2 = \sum_{i=1}^n\big(1 - \frac{i}{n + 1}\big)^2 \le \frac{n + 1}{3}$, turn this into $\frac{3L\|x_0 - x^*\|^2}{32(k + 1)^2}$.

<details><summary>solution</summary>

(a) $\nabla f(x) = Tx - e_1$. if $x$ is supported on the first $j$ coordinates, $Tx$ is supported on the first $j + 1$ (tridiagonal), so each new gradient, and hence each new iterate, reaches at most one coordinate further. (b) restricted to the first $k$ coordinates, $f$ is the same function in dimension $k$; its minimizer solves $T_kx = e_1$, and the minimum value is $-\tfrac12e_1^\top T_k^{-1}e_1 = -\tfrac12x^*_1 = -\tfrac12\big(1 - \tfrac1{k + 1}\big)$. the same formula with $n$ gives $f^*$. (c) $f(x_k) - f^* \ge \tfrac12\big(\tfrac{n}{n + 1} - \tfrac{k}{k + 1}\big) = \tfrac12\big(\tfrac1{k + 1} - \tfrac1{n + 1}\big)$, and with $n + 1 = 2(k + 1)$ this is $\tfrac1{4(k + 1)}$. (d) $\|x^*\|^2 \le \tfrac{n + 1}{3} = \tfrac{2(k + 1)}{3}$, so $k + 1 \ge \tfrac32\|x^*\|^2$ and $\tfrac{1}{4(k + 1)} = \tfrac{k + 1}{4(k + 1)^2} \ge \tfrac{3\|x^*\|^2}{8(k + 1)^2}$, which is $\tfrac{3L\|x^*\|^2}{32(k + 1)^2}$ with $L = 4$ and $x_0 = 0$. (the sum of squares is $\tfrac{n(2n + 1)}{6(n + 1)}$, and $n(2n + 1) \le 2(n + 1)^2$.)

</details>

4. **Nesterov is heavy ball with a corrected gradient.** show that Nesterov's two lines are equivalent to $x_{k+1} = x_k - \eta\nabla f(y_k) + \beta(x_k - x_{k-1})$, and that on a quadratic $\nabla f(y_k) = \nabla f(x_k) + \beta A(x_k - x_{k-1})$. interpret the extra term.

<details><summary>solution</summary>

substitute $y_k$: $x_{k+1} = x_k + \beta(x_k - x_{k-1}) - \eta\nabla f(y_k)$. on a quadratic the gradient is affine, so $\nabla f(y_k) = A(x_k + \beta(x_k - x_{k-1})) - b = \nabla f(x_k) + \beta A(x_k - x_{k-1})$. the extra term $-\eta\beta A(x_k - x_{k-1})$ is a damping force proportional to the curvature along the last step: it brakes hardest in the sharp directions, exactly where heavy ball overshoots.

</details>

## experiment

`lab.py` investigates three things with your code:

1. a race on an ill-conditioned quadratic ($\kappa = 1{,}000$): gradient descent, heavy ball, Nesterov with tuned constant momentum, and Nesterov with the schedule.
2. the schedule's ripples on a strongly convex problem, and what restarting the momentum does.
3. heavy ball with its "optimal" quadratic tuning on a strongly convex function that is *not* a quadratic.

## check your understanding

1. why does momentum help in a narrow valley but not on a round bowl?

<details><summary>answer</summary>

in a narrow valley the gradient components across the valley alternate in sign and cancel in the momentum sum, while the components along the floor add up, so the effective step along the floor grows by up to $1/(1 - \beta)$. on a round bowl ($\kappa = 1$) gradient descent with $\eta = 1/L$ already lands on the minimum in one step; momentum can only add overshoot.

</details>

2. (the predict callout in idea 2.) Nesterov with constant 0.5 on $2x^2$ from 1?

<details><summary>answer</summary>

$x_1 = 0.6$, $x_2 = 0.24$, $x_3 = 0.036$ (see the worked example), a rate of $\sqrt{0.3} \approx 0.55$ per step, faster than gradient descent's 0.6.

</details>

3. heavy ball's tuned rate on a quadratic is $\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$, the best possible. why isn't heavy ball the standard accelerated method, then?

<details><summary>answer</summary>

its guarantee is only for quadratics. on general strongly convex functions it can fail to converge with those parameters (lab experiment 3, from Lessard, Recht & Packard, 2016), while Nesterov's method provably achieves $1 - 1/\sqrt\kappa$ for every $L$-smooth, $\mu$-strongly convex function. in deep learning, where nothing is quadratic and the theory doesn't apply anyway, both are used, and "SGD with momentum" usually means heavy ball.

</details>

4. you overestimate $\kappa$ by a factor of 100 when tuning heavy ball. what happens?

<details><summary>answer</summary>

$\beta$ is too close to 1 and $\eta$ too small. the roots for every eigenvalue stay complex with modulus $\sqrt\beta$, now about $1 - 2/\sqrt{\kappa_{\text{assumed}}}$: still convergent, but at the rate for the larger $\kappa$, 10 times slower than tuned, with lots of oscillation. underestimating $\kappa$ is worse for the small eigenvalues: their roots turn real, one close to 1, and the slow directions crawl. `momentum_spectral_radius` lets you compute both.

</details>

5. Nesterov's rate for convex functions is $O(1/k^2)$, and the lower bound is $\Omega(1/k^2)$. does that mean nothing converges faster on *your* convex problem?

<details><summary>answer</summary>

no: the lower bound is about the *worst* function in the class, in high enough dimension ($n \ge 2k + 1$). a specific problem can be much easier: strongly convex (linear rates), low-dimensional (conjugate gradients finishes a quadratic in $n$ steps), or well-conditioned. lower bounds tell you when to stop looking for a better general-purpose method, not how fast your problem is.

</details>

6. why does each step of `nesterov` cost two gradient evaluations in your implementation, and how could you get it down to one?

<details><summary>answer</summary>

the step needs $\nabla f(y_k)$, and the stopping test needs $\nabla f(x_k)$. you could test $\|\nabla f(y_k)\|$ instead (and return $y_k$ when it's small), or check convergence only every few steps. the version here keeps the same convergence test as every other method in `mlfs.optim`, so their results are comparable.

</details>

7. (the predict callout in idea 4.) three heavy-ball steps from 0 on the tridiagonal worst case: which coordinates of $x_3$ are nonzero?

<details><summary>answer</summary>

only $x_1, x_2, x_3$. $\nabla f(0) = -\frac L4e_1$ touches only the first coordinate; each later gradient $\frac L4(Tx - e_1)$ reaches one coordinate further, and the momentum term only recombines earlier iterates. coordinates 4 to 10 are exactly zero, however the method is tuned, and the minimizer has all of them nonzero: that's the lower bound in one picture.

</details>

## common bugs

- **momentum on the first step.** symptom: the first step is too big, or `x_prev` is undefined. set $x_{-1} = x_0$, so the first step is a plain gradient step.
- **velocity with the wrong sign.** symptom: momentum slows everything down. the term is $+\beta(x_k - x_{k-1})$: keep going the way you were going.
- **gradient at $x_k$ instead of $y_k$ in Nesterov.** symptom: you've implemented heavy ball. the look-ahead point is the whole idea.
- **`if not momentum:` for the schedule.** symptom: `momentum=0.0` uses the schedule instead of plain gradient descent. test `momentum is None`.
- **an off-by-one schedule.** symptom: iterates differ from the hand example from step 2. $\beta_k = (k - 1)/(k + 2)$ with $k = 1$ for the first step, so $\beta_1 = 0$.
- **taking only real roots.** symptom: the spectral radius is wrong exactly when the method oscillates. when $a^2 < 4b$ the roots are complex with modulus $\sqrt b$; use a complex square root.

## going deeper

- Polyak (1964), "Some methods of speeding up the convergence of iteration methods", *USSR Computational Mathematics and Mathematical Physics* 4(5). the heavy-ball method. [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5)
- Nesterov (2018), *Lectures on Convex Optimization*, 2nd ed., Springer, §2.1.2 (lower complexity bounds) and §2.2 (optimal methods). [doi:10.1007/978-3-319-91578-4](https://doi.org/10.1007/978-3-319-91578-4)
- Su, Boyd & Candès (2016), "A differential equation for modeling Nesterov's accelerated gradient method", *JMLR* 17(153). where the $(k - 1)/(k + 2)$ schedule comes from. [arXiv:1503.01243](https://arxiv.org/abs/1503.01243)
- O'Donoghue & Candès (2015), "Adaptive restart for accelerated gradient schemes", *Foundations of Computational Mathematics* 15. the fix for the ripples in lab experiment 2. [arXiv:1204.3982](https://arxiv.org/abs/1204.3982)
- Lessard, Recht & Packard (2016), "Analysis and design of optimization algorithms via integral quadratic constraints", *SIAM Journal on Optimization* 26(1). the counterexample in lab experiment 3. [arXiv:1408.3595](https://arxiv.org/abs/1408.3595)
- Goh (2017), "Why momentum really works", *Distill*. interactive pictures of everything in idea 3. [doi:10.23915/distill.00006](https://doi.org/10.23915/distill.00006)

## connections

- ← builds on: gradient descent, `OptimResult` and the condition number (topic 02), linear iterations on quadratics (ch 03 · topic 05), eigenvalues and spectral radii (ch 02 · topic 05).
- → used in: SGD with momentum and Adam (topic 04), the conjugate gradient connection and quasi-Newton methods (topic 05), training neural networks with momentum (ch 14–15), and the implicit regularization of momentum methods (ch 19).
