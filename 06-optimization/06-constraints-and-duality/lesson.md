# constraints & duality

## why this matters

real optimization problems come with fences. probabilities must be nonnegative and sum to 1. a portfolio's weights must sum to your budget. a robust model's perturbation must stay inside an $\epsilon$-ball. a support vector machine wants the widest margin *subject to* classifying every training point correctly. "minimize $f$" is really "minimize $f$ over a set $C$".

there are two classic ways in, and this topic builds both. the first is geometric: take a gradient step, and if it leaves the set, snap back to the nearest feasible point. that's **projected gradient descent**, and it's as simple as it sounds whenever projecting is cheap (boxes, balls, the simplex). the second is algebraic: attach a price, a **Lagrange multiplier**, to every constraint, and turn the constrained problem into an unconstrained one about prices. that leads to the **KKT conditions**, which certify that a point is optimal, and to **duality**: every constrained minimization has a twin maximization whose value bounds it from below, and for convex problems the two meet. the support vector machine of chapter 09 is solved exactly this way, through its dual, and its "support vectors" are just the constraints whose price isn't zero. you'll see them appear in the lab.

## warm-up: predict first

1. you project $v = (3, 4)$ onto the unit disk. where does it land? what about $(0.3, 0.4)$?
2. projecting $(0.5, 0.5, 0.5)$ onto the probability simplex: is it $(1/3, 1/3, 1/3)$, i.e. "divide by the sum"? what about $(1, 0.5, -1)$?
3. at the minimizer of $f$ over a set, must the gradient of $f$ be zero?
4. the optimal value of a constrained problem is 10. you relax one constraint slightly, by $\epsilon$. how could you predict the new optimal value without re-solving?
5. you have a feasible point with objective 12 and a "dual" lower bound of 11.9. what can you say about the true optimum?

## prerequisites check

- [ ] you know convex sets and the first-order condition for convexity (→ topic 01)
- [ ] you've built `gradient_descent` and `OptimResult`, and know the rate of gradient descent with step $1/L$ (→ topic 02)
- [ ] you can solve square linear systems with `mlfs.linalg.solve`, and know why Cholesky needs a positive definite matrix (→ ch 02 · topic 03)
- [ ] you can compute gradients of quadratics: $\nabla(\tfrac12x^\top Qx + c^\top x) = Qx + c$ for symmetric $Q$ (→ ch 02 · topic 07)

## core ideas

### idea 1: projections: the closest feasible point

for a closed convex set $C$, every point $v$ has a unique closest point in $C$, its **Euclidean projection**

$$P_C(v) = \arg\min_{x \in C}\tfrac12\|x - v\|^2.$$

it's characterized by an angle: $p = P_C(v)$ iff $p \in C$ and

$$(v - p)^\top(z - p) \le 0\quad\text{for every } z \in C,$$

the vector back to $v$ makes an obtuse angle with every direction into the set (on paper 1 gives the reason). projections onto convex sets are **nonexpansive**, $\|P_C(u) - P_C(v)\| \le \|u - v\|$: snapping back never amplifies differences, which is why it doesn't break gradient descent's convergence.

three sets have one-line projections:

- **a box** $\{l \le x \le u\}$: clip each coordinate, $P(v) = \operatorname{clip}(v, l, u)$. coordinates don't interact.
- **a ball** $\{\|x\|_2 \le r\}$: if $\|v\| \le r$, $v$ is already feasible; otherwise scale it radially, $P(v) = rv/\|v\|$ (warm-up 1: $(0.6, 0.8)$, and $(0.3, 0.4)$ stays put).
- **the probability simplex** $\Delta = \{x \ge 0, \sum_ix_i = 1\}$: the answer has the form $P(v) = \max(v - \theta, 0)$ for a single threshold $\theta$, chosen so the result sums to 1 (on paper 2 derives it from the KKT conditions). sorting $v$ in decreasing order as $u_1 \ge u_2 \ge \dots$, the number of positive entries is the largest $j$ with $u_j - \frac{1}{j}\big(\sum_{i\le j}u_i - 1\big) > 0$, and $\theta = \frac1j\big(\sum_{i\le j}u_i - 1\big)$ (Duchi et al., 2008). it's *not* "divide by the sum" (warm-up 2): it subtracts a constant and clips, so small entries become exactly zero.

> **▶ build: stage 1.** implement `project_box(v, lower, upper)`, `project_l2_ball(v, radius)` and `project_simplex(v)` (sort-based, vectorized).

### idea 2: projected gradient descent

to minimize a smooth $f$ over a convex $C$ with an easy projection, take a gradient step and project:

$$x_{k+1} = P_C\big(x_k - \eta\nabla f(x_k)\big).$$

at a constrained minimum the gradient usually isn't zero (warm-up 3): it points *out of* the set, and the projection cancels the step. the right measure of progress is the **gradient mapping** $G_\eta(x) = \big(x - P_C(x - \eta\nabla f(x))\big)/\eta$, which is $\nabla f(x)$ when the constraint doesn't bind and zero exactly at constrained stationary points. so the stopping test is $\|x_{k+1} - x_k\| \le \eta\cdot\text{tol}$. with $\eta \le 1/L$, projected gradient descent keeps every rate from topic 02: $O(1/k)$ for convex $f$, linear for strongly convex $f$ (the proofs go through because projections are nonexpansive).

two practical details: $x_0$ may be infeasible (the first projection fixes that), and the projection must be cheap, because it runs every step. for sets without an easy projection (a polyhedron $\{Gx \le h\}$, say), you need the second approach.

> **▶ build: stage 2.** implement `projected_gradient_descent(f, grad, project, x0, *, lr, max_iters, tol, callback)`, stopping when $\|x_{k+1} - x_k\| \le \text{tol}\cdot\text{lr}$ (and still taking that step).

### idea 3: equality constraints, Lagrange multipliers, and the KKT system

minimize $f(x)$ subject to $Ax = b$. at a constrained minimum $x^*$, moving along the constraint surface can't decrease $f$, so $\nabla f(x^*)$ must be perpendicular to the surface: it's a combination of the constraint normals, the rows of $A$. write that as

$$\nabla f(x^*) + A^\top\lambda = 0,\qquad Ax^* = b,$$

for some vector $\lambda$ of **Lagrange multipliers**. both conditions are the stationarity conditions of the **Lagrangian** $\mathcal L(x, \lambda) = f(x) + \lambda^\top(Ax - b)$ in $x$ and $\lambda$. for a quadratic objective $f = \tfrac12x^\top Qx + c^\top x$ they're linear, the **KKT system**:

$$\begin{pmatrix}Q & A^\top\\A & 0\end{pmatrix}\begin{pmatrix}x\\\lambda\end{pmatrix} = \begin{pmatrix}-c\\b\end{pmatrix}.$$

one solve gives both the minimizer and the multipliers. the matrix is symmetric but **indefinite** (it has negative eigenvalues: the zero block makes sure of that), so Cholesky won't work; LU with pivoting (your `solve`) does. $Q$ itself may even be singular, as long as the constraints pin down its flat directions.

what *are* the multipliers? they're prices. if you change $b$ to $b + \delta$, the optimal value changes by about $-\lambda^\top\delta$ (on paper 3): $\lambda_i$ is the rate at which the optimum gets worse as you tighten constraint $i$ (warm-up 4). that's why economists call them shadow prices, and it's what lab experiment 3 measures.

> **🔮 predict:** for $\min\tfrac12(x_1^2 + x_2^2)$ subject to $x_1 + x_2 = 1$, what are $x^*$ and $\lambda$, with the sign convention above?

> **▶ build: stage 3.** implement `solve_eq_qp(Q, c, A, b)`, building the KKT matrix with `np.block` and solving it with `mlfs.linalg.solve`. return `(x, lam)`.

### idea 4: inequality constraints: KKT conditions and the dual function

now minimize $f(x) = \tfrac12x^\top Qx + c^\top x$ (with $Q \succ 0$) subject to $Gx \le h$, $m$ inequality constraints. attach a multiplier $\lambda_i \ge 0$ to each: $\mathcal L(x, \lambda) = f(x) + \lambda^\top(Gx - h)$. for feasible $x$ the extra term is $\le 0$, so $\mathcal L \le f$: the multipliers *reward* staying feasible. the **Karush–Kuhn–Tucker (KKT) conditions** for $(x^*, \lambda^*)$ are

1. **stationarity:** $Qx^* + c + G^\top\lambda^* = 0$,
2. **primal feasibility:** $Gx^* \le h$,
3. **dual feasibility:** $\lambda^* \ge 0$,
4. **complementary slackness:** $\lambda_i^*(Gx^* - h)_i = 0$ for every $i$: a constraint either binds ($= h_i$) or has price zero.

for a convex problem they're sufficient for optimality, and (under a mild condition, below) necessary. the residuals of the four conditions are a **certificate**: anyone can check them without trusting your solver.

the **dual function** is the best the Lagrangian can do for given prices: $g(\lambda) = \min_x\mathcal L(x, \lambda)$. for the QP the minimizing $x$ is $x(\lambda) = -Q^{-1}(c + G^\top\lambda)$, and

$$g(\lambda) = -\tfrac12(c + G^\top\lambda)^\top Q^{-1}(c + G^\top\lambda) - h^\top\lambda.$$

$g$ is concave (a minimum of affine functions of $\lambda$), and for every $\lambda \ge 0$ and every feasible $x$, $g(\lambda) \le \mathcal L(x, \lambda) \le f(x)$: **weak duality**. any $\lambda \ge 0$ gives a lower bound on the optimum, and any feasible $x$ an upper bound; the difference $f(x) - g(\lambda)$, the **duality gap**, bounds how suboptimal $x$ is (warm-up 5: the optimum is between 11.9 and 12, so $x$ is within 0.1 of optimal).

> **🔮 predict:** for $\min\tfrac12x^2$ subject to $x \ge 1$, what is $g(\lambda)$, which $\lambda$ maximizes it, and how does the maximum compare with the optimal value?

> **▶ build: stage 4.** implement `qp_dual(Q, c, G, h, lam)`, returning $(g(\lambda), x(\lambda))$, and `kkt_residuals(Q, c, G, h, x, lam)`, returning the four residuals as a dict.

### idea 5: strong duality, and solving through the dual

for convex problems with a strictly feasible point ($Gx < h$ for some $x$: **Slater's condition**), **strong duality** holds: $\max_{\lambda\ge0}g(\lambda) = \min_{Gx\le h}f(x)$, the gap closes, and the maximizing $\lambda^*$ are the KKT multipliers. complementary slackness is then forced (on paper 4): if $f(x^*) = g(\lambda^*)$, every term of $\lambda^{*\top}(Gx^* - h) \le 0$ must vanish.

that suggests a different algorithm: maximize $g$ instead of minimizing $f$. the dual's only constraint is $\lambda \ge 0$, a box, whose projection is a clip. so projected gradient *ascent* on $g$ works, with gradient $\nabla g(\lambda) = Gx(\lambda) - h$ (the constraint violations at $x(\lambda)$: raise the price of violated constraints, lower the price of slack ones) and step $1/L$, where $L = \lambda_{\max}(GQ^{-1}G^\top)$ is the dual's curvature. at the end, $x(\lambda)$ is the primal solution, and the gap $f(x(\lambda)) - g(\lambda)$ tells you how done you are. this is the strategy of SVM solvers: the SVM's dual has a simple constraint set, its dimension is the number of training points rather than features, and only the points with $\lambda_i > 0$, the **support vectors**, determine the solution. (chapter 09's SMO algorithm optimizes the same dual two multipliers at a time.)

> **▶ build: stage 5.** implement `solve_qp_dual(Q, c, G, h, *, lr, max_iters, tol)` using your `projected_gradient_descent` on $-g$ with a projection onto $\lambda \ge 0$. return `(x, lam, gap)`.

## worked example

**projections.** $P_{\text{ball}}((3, 4)) = (3, 4)/5 = (0.6, 0.8)$; with radius 2 it's $(1.2, 1.6)$. for the simplex and $v = (1, 0.5, -1)$: sorted $u = (1, 0.5, -1)$, cumulative sums minus 1 are $c = (0, 0.5, -0.5)$, and $u_j - c_j/j$ is $1, 0.25, -0.833$ for $j = 1, 2, 3$: the last positive one is $j = 2$, so $\theta = c_2/2 = 0.25$ and $P(v) = \max((0.75, 0.25, -1.25), 0) = (0.75, 0.25, 0)$. ✓ it sums to 1. for $(0.5, 0.5, 0.5)$: $\theta = 0.5/3 = 1/6$ and $P = (1/3, 1/3, 1/3)$ (subtracting $1/6$ from each happens to equal dividing by 1.5 here, but only because all entries are equal).

**an equality QP.** $\min\tfrac12(x_1^2 + x_2^2)$ subject to $x_1 + x_2 = 1$: $Q = I$, $c = 0$, $A = (1\ \ 1)$, $b = 1$. the KKT system is

$$\begin{pmatrix}1 & 0 & 1\\0 & 1 & 1\\1 & 1 & 0\end{pmatrix}\begin{pmatrix}x_1\\x_2\\\lambda\end{pmatrix} = \begin{pmatrix}0\\0\\1\end{pmatrix}\ \Rightarrow\ x_1 = x_2 = -\lambda,\ -2\lambda = 1:\quad x^* = (\tfrac12, \tfrac12),\ \lambda = -\tfrac12.$$

the optimal value is $\tfrac14$. moving the constraint to $x_1 + x_2 = 1 + \delta$ gives $x^* = (\tfrac{1+\delta}2, \tfrac{1+\delta}2)$ and value $\tfrac{(1 + \delta)^2}{4} \approx \tfrac14 + \tfrac\delta2 = \tfrac14 - \lambda\delta$ ✓.

**an inequality QP and its dual.** $\min\tfrac12x^2$ subject to $x \ge 1$, written as $-x \le -1$ ($G = -1$, $h = -1$). then $x(\lambda) = -(0 - \lambda) = \lambda$ and $g(\lambda) = -\tfrac12\lambda^2 + \lambda$: at $\lambda = 0, \tfrac12, 1, 2$ it's $0, 0.375, 0.5, 0$. the maximum is at $\lambda^* = 1$, where $x(\lambda^*) = 1 = x^*$ and $g = \tfrac12 = f(x^*)$: zero gap. the KKT conditions check out: stationarity $x + G^\top\lambda = 1 - 1 = 0$, the constraint binds ($x = 1$) and $\lambda^* = 1 > 0$. if the constraint were $x \ge -1$ instead ($h = 1$), $g(\lambda) = -\tfrac12\lambda^2 - \lambda$ is maximized over $\lambda \ge 0$ at $\lambda^* = 0$: the constraint is slack, its price is zero, and $x^* = 0$.

## on paper

1. **the projection inequality.** let $p = P_C(v)$ for a closed convex $C$. for $z \in C$ and $t \in (0, 1]$, the point $p + t(z - p)$ is in $C$. expand $\|p + t(z - p) - v\|^2 \ge \|p - v\|^2$ and let $t \to 0$ to show $(v - p)^\top(z - p) \le 0$.

<details><summary>solution</summary>

$\|p - v + t(z - p)\|^2 = \|p - v\|^2 + 2t(p - v)^\top(z - p) + t^2\|z - p\|^2 \ge \|p - v\|^2$, so $2(p - v)^\top(z - p) + t\|z - p\|^2 \ge 0$ for all $t \in (0, 1]$. letting $t \to 0$: $(p - v)^\top(z - p) \ge 0$, i.e. $(v - p)^\top(z - p) \le 0$.

</details>

2. **the simplex threshold.** write the KKT conditions of $\min\tfrac12\|x - v\|^2$ subject to $\sum_ix_i = 1$ and $x \ge 0$ (multiplier $\theta$ for the equality, $\mu_i \ge 0$ for $-x_i \le 0$), and show that $x_i = \max(v_i - \theta, 0)$.

<details><summary>solution</summary>

stationarity: $x_i - v_i + \theta - \mu_i = 0$, so $x_i = v_i - \theta + \mu_i$. complementary slackness: $\mu_ix_i = 0$. if $x_i > 0$ then $\mu_i = 0$ and $x_i = v_i - \theta$ (which must be $> 0$). if $x_i = 0$ then $\mu_i = \theta - v_i \ge 0$, i.e. $v_i \le \theta$. both cases are $x_i = \max(v_i - \theta, 0)$. $\theta$ is fixed by $\sum_i\max(v_i - \theta, 0) = 1$, a decreasing piecewise-linear function of $\theta$, which the sort-based formula solves exactly.

</details>

3. **multipliers are sensitivities.** for the equality QP with $Q \succ 0$, let $x^*(b)$, $\lambda(b)$ solve the KKT system and $p(b) = f(x^*(b))$. show $\nabla_bp = -\lambda$. hint: differentiate $Ax^*(b) = b$ and use stationarity.

<details><summary>solution</summary>

$\frac{\partial p}{\partial b} = \big(\frac{\partial x^*}{\partial b}\big)^\top\nabla f(x^*) = -\big(\frac{\partial x^*}{\partial b}\big)^\top A^\top\lambda$ by stationarity. differentiating $Ax^*(b) = b$ gives $A\frac{\partial x^*}{\partial b} = I$, so the right side is $-(A\frac{\partial x^*}{\partial b})^\top\lambda = -\lambda$. (for inequalities the same holds for small changes that don't change which constraints are active: an inactive constraint has $\lambda_i = 0$ and zero sensitivity.)

</details>

4. **strong duality forces complementary slackness.** suppose $x^*$ is feasible, $\lambda^* \ge 0$ and $f(x^*) = g(\lambda^*)$. show that $\lambda_i^*(Gx^* - h)_i = 0$ for every $i$, and that $x^*$ minimizes $\mathcal L(\cdot, \lambda^*)$.

<details><summary>solution</summary>

$g(\lambda^*) = \min_x\mathcal L(x, \lambda^*) \le \mathcal L(x^*, \lambda^*) = f(x^*) + \sum_i\lambda_i^*(Gx^* - h)_i \le f(x^*)$, since each term of the sum is (nonnegative) × (nonpositive). equality at both ends forces both inequalities to be equalities: the sum is 0, and since every term is $\le 0$, every term is 0; and $x^*$ attains the minimum of the Lagrangian, which (for $Q \succ 0$) means $Qx^* + c + G^\top\lambda^* = 0$: stationarity.

</details>

## experiment

`lab.py` investigates three things with your code:

1. a hard-margin linear SVM (no bias) on 40 separable points, solved through its dual: the primal and dual objectives converging to each other, and which training points end up with nonzero multipliers.
2. least squares constrained to the simplex versus to a ball: what the solutions look like.
3. Lagrange multipliers as prices: perturb each constraint and compare the change in the optimum with $-\lambda_i$.

## check your understanding

1. why does projected gradient descent stop on the gradient mapping rather than on $\|\nabla f\|$?

<details><summary>answer</summary>

at a constrained minimum $\nabla f$ is usually nonzero (it points out of the set), so $\|\nabla f\| \le \text{tol}$ would never trigger. the gradient mapping $(x - P(x - \eta\nabla f(x)))/\eta$ is zero exactly when the projected step doesn't move, which is the constrained stationarity condition; when no constraint is active it equals $\nabla f$.

</details>

2. (the predict callout in idea 3.) $x^*$ and $\lambda$ for $\min\tfrac12\|x\|^2$ s.t. $x_1 + x_2 = 1$?

<details><summary>answer</summary>

$x^* = (\tfrac12, \tfrac12)$ and $\lambda = -\tfrac12$ with $\mathcal L = f + \lambda(x_1 + x_2 - 1)$. the negative sign says that raising $b$ *increases* the optimal value (you're pushed further from the origin): $\partial p/\partial b = -\lambda = \tfrac12$.

</details>

3. why can't you solve the KKT system with a Cholesky factorization?

<details><summary>answer</summary>

it isn't positive definite: $(0, \mu)^\top K(0, \mu) = 0$ for every $\mu$, and with a nonzero $A$ there are directions with negative curvature. it's a saddle-point matrix (the Lagrangian is minimized in $x$ and maximized in $\lambda$). LU with pivoting, or a specialized symmetric-indefinite factorization, is needed.

</details>

4. (the predict callout in idea 4.) the dual of $\min\tfrac12x^2$ s.t. $x \ge 1$?

<details><summary>answer</summary>

$g(\lambda) = \lambda - \tfrac12\lambda^2$, maximized at $\lambda^* = 1$ with $g = \tfrac12$, equal to the optimal value $f(1) = \tfrac12$: strong duality. the multiplier says that requiring $x \ge 1 + \delta$ costs about $\delta$ more.

</details>

5. a constraint is strictly satisfied at the optimum ($Gx^* < h$ in row $i$). what's $\lambda_i^*$, and what happens to the optimum if you remove the constraint?

<details><summary>answer</summary>

$\lambda_i^* = 0$ by complementary slackness, and removing (or slightly moving) the constraint doesn't change the optimum: it wasn't doing anything. in an SVM, that's every training point outside the margin: removing it doesn't change the classifier.

</details>

6. when can strong duality fail?

<details><summary>answer</summary>

for nonconvex problems, generally (the dual is still concave and still a lower bound, but the gap can be positive). for convex problems it needs a constraint qualification such as Slater's condition. a classic example without one is $\min x$ subject to $x^2 \le 0$: the only feasible point is 0, no multiplier satisfies stationarity there ($1 + 2\lambda\cdot0 = 0$ has no solution), and the dual $g(\lambda) = -1/(4\lambda)$ approaches the optimal value 0 only as $\lambda \to \infty$, never attaining it. for linear constraints and a convex quadratic objective, strong duality holds whenever the problem is feasible.

</details>

## common bugs

- **"normalizing" onto the simplex.** symptom: negative entries survive, or the result isn't the closest point. subtract the threshold and clip; don't divide by the sum.
- **sorting ascending.** symptom: the simplex projection is wrong except for already-sorted inputs. the threshold formula needs $u$ in *decreasing* order.
- **projecting before the step, or checking convergence before stepping.** symptom: the hand iterates are off by one step. step, project, then test the distance moved.
- **an absolute tolerance on the gradient mapping.** symptom: tiny learning rates "converge" at once. the step moves $\eta\|G_\eta\|$, so compare $\|x_{k+1} - x_k\|$ with $\text{tol}\cdot\eta$.
- **sign conventions for multipliers.** symptom: $\lambda$ has the wrong sign, or the dual value exceeds the primal. with $\mathcal L = f + \lambda^\top(Gx - h)$ and $Gx \le h$, multipliers are $\ge 0$ and $x(\lambda) = -Q^{-1}(c + G^\top\lambda)$.
- **forming $Q^{-1}$.** symptom: inaccuracy for ill-conditioned $Q$. solve with $Q$ instead.

## going deeper

- Boyd & Vandenberghe (2004), *Convex Optimization*, chapter 5 (duality, KKT conditions, sensitivity, Slater's condition) and §10.1 (equality-constrained minimization and the KKT system). [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/)
- Nocedal & Wright (2006), *Numerical Optimization*, chapters 12 (theory of constrained optimization) and 16 (quadratic programming, including active-set methods). [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Duchi, Shalev-Shwartz, Singer & Chandra (2008), "Efficient projections onto the ℓ1-ball for learning in high dimensions", *ICML*. the sort-based simplex projection. [doi:10.1145/1390156.1390191](https://doi.org/10.1145/1390156.1390191)
- Beck (2017), *First-Order Methods in Optimization*, SIAM, chapters 9–10 (projected and proximal gradient methods, the gradient mapping). [doi:10.1137/1.9781611974997](https://doi.org/10.1137/1.9781611974997)
- Cortes & Vapnik (1995), "Support-vector networks", *Machine Learning* 20. the SVM as a quadratic program, solved through its dual. [doi:10.1007/BF00994018](https://doi.org/10.1007/BF00994018)

## connections

- ← builds on: convex sets and functions (topic 01), gradient descent, its rates and `OptimResult` (topic 02), linear solves with `mlfs.linalg` (ch 02 · topic 03), quadratic forms and their gradients (ch 02 · topic 07).
- → used in: the lasso and constrained regression (ch 08), the SVM dual and SMO (ch 09), maximum entropy and the simplex in probabilistic models (ch 12), projected gradient attacks (PGD) inside an $\epsilon$-ball (ch 28), and trust regions and constrained policy updates in RL (ch 24).
