# convex sets & functions

## why this matters

you train a logistic regression twice, from two different random initializations, and get the same weights to eight decimal places. you train a two-layer network twice and get two completely different sets of weights, with different test accuracies. the difference isn't the optimizer, the learning rate, or luck. it's a property of the loss surface called **convexity**: the logistic loss is a convex function of its weights, and the network's loss isn't.

convexity is the dividing line in optimization. on the convex side, every local minimum is a global minimum, any point where the gradient vanishes is optimal, and the convergence rates you'll prove in the next topics are guarantees. on the other side, you get saddles, spurious minima, and dependence on initialization (topic 07). so before building optimizers, it pays to be able to tell which side of the line a problem is on. that's harder than it sounds, because "convex" is a statement about *every* pair of points, and a computer can only check finitely many. this topic gives you four ways of hunting for a counterexample, and a clear sense of what it means when you don't find one.

## warm-up: predict first

1. is the set of points $(x, y)$ with $x^2 + y^2 \ge 1$ (everything *outside* the unit disk) convex?
2. squared error is convex, and the sigmoid is a smooth, monotone function. is $\big(\sigma(w x) - y\big)^2$ convex in $w$?
3. $f(a, b) = (ab - 1)^2$ is the loss of a "network" with two linear layers of width 1. the model $x \mapsto abx$ is linear in $x$. is the loss convex in $(a, b)$?
4. you check Jensen's inequality on 1,000 random pairs of points and it holds every time. is the function convex?
5. the maximum of two convex functions: convex? the minimum?

## prerequisites check

- [ ] you can compute gradients and Hessians, and read curvature off the Hessian's eigenvalues (→ ch 03 · topic 02)
- [ ] you know that a symmetric matrix is positive semidefinite iff all its eigenvalues are $\ge 0$ (→ ch 02 · topic 05)
- [ ] you know what the smoothness and strong-convexity constants $L$ and $\mu$ measure (→ ch 03 · topic 05)
- [ ] you can write a matrix inequality $Ax \le b$ row by row, and evaluate it for many points at once with broadcasting (→ ch 00 · topic 02)

## core ideas

### idea 1: convex sets: every chord stays inside

a set $C \subseteq \mathbb{R}^n$ is **convex** if the segment between any two of its points stays in it:

$$x, y \in C,\ t \in [0, 1] \implies tx + (1 - t)y \in C.$$

a disk, a square, a half-plane, a line and a single point are convex. a crescent, an annulus and two separate disks aren't: you can find two points whose chord leaves the set (warm-up 1: the outside of a disk is not convex, since the chord between $(1, 0)$ and $(-1, 0)$ passes through the origin).

the convex sets you'll meet most in ML are built from a few pieces:

- a **halfspace** $\{x : a^\top x \le b\}$ is convex, and so is a hyperplane $\{x : a^\top x = b\}$.
- a **norm ball** $\{x : \|x - c\| \le r\}$ is convex for any norm, by the triangle inequality.
- an **intersection** of convex sets is convex (on paper 1). so a **polyhedron** $\{x : Ax \le b\}$, the intersection of $m$ halfspaces, is convex, and so is the **probability simplex** $\{x : x \ge 0,\ \sum_i x_i = 1\}$.
- the set of positive semidefinite matrices is convex (a "cone": it's closed under positive scaling too).

the *union* of convex sets usually isn't convex, and that asymmetry (intersections yes, unions no) will come back for functions.

how would a computer decide whether a set is convex? it can't check every chord. what it *can* do is search for a **witness** of non-convexity: two points of the set and a $t$ whose combination falls outside. one witness is a proof that the set isn't convex. no number of failed searches is a proof that it is.

> **▶ build: stage 1.** implement `in_polyhedron(A, b, X)` (vectorized membership in $\{Ax \le b\}$, with a small tolerance so that boundary points count as inside) and `set_convexity_violation(contains, points, ts)`, which tests the chord points $t x_i + (1 - t) x_j$ for every pair of member points and returns the first witness, or `None`.

### idea 2: convex functions: every chord lies above the graph

a function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if the chord between any two points of its graph lies on or above the graph:

$$f\big(tx + (1 - t)y\big) \le t f(x) + (1 - t) f(y)\quad\text{for all } x, y,\ t \in [0, 1].$$

this is **Jensen's inequality** (in its two-point form). the difference

$$\text{gap}(x, y, t) = t f(x) + (1 - t) f(y) - f\big(tx + (1 - t)y\big)$$

is $\ge 0$ everywhere for a convex function, and a single negative value is a witness that $f$ isn't convex. the two ideas are linked: $f$ is convex exactly when its **epigraph**, the region $\{(x, s) : s \ge f(x)\}$ above its graph, is a convex set. and every **sublevel set** $\{x : f(x) \le c\}$ of a convex function is convex (the converse fails, as the lab shows).

why does this one property matter so much? because it makes local information global.

**local minima are global.** suppose $x$ is a local minimum of a convex $f$, but some $y$ has $f(y) < f(x)$. for small $t > 0$, the point $z = ty + (1 - t)x$ is as close to $x$ as you like, and Jensen gives $f(z) \le t f(y) + (1 - t)f(x) < f(x)$: points arbitrarily close to $x$ are lower, so $x$ wasn't a local minimum after all. on a convex function, there's no such thing as a "bad" local minimum, and the set of minimizers is itself convex (a point, a segment, a flat region, never two separate valleys).

that's why the logistic regression in "why this matters" lands on the same weights every time (its minimizer is unique, because the ridge penalty makes it strictly convex), and why the network doesn't.

> **🔮 predict:** $f(x) = x^4 - 2x^2$ has two valleys, at $x = \pm1$. at which pair of the points $-1, 0, 1, 2$, and which $t$, is its Jensen gap most negative?

> **▶ build: stage 2.** implement `jensen_gap(f, x, y, t)` and `min_jensen_gap(f, points, ts)`, which returns the most negative gap over all pairs $i < j$ and all $t$ in `ts`, with its location. evaluate $f$ only once per point for the chord ends.

### idea 3: the first-order condition: tangent planes are global underestimators

for a differentiable $f$, convexity has an equivalent form that uses the gradient:

$$f(y) \ge f(x) + \nabla f(x)^\top (y - x)\quad\text{for all } x, y.$$

the tangent plane at any point lies below the whole graph, not just near the point. (on paper 2 derives it from Jensen: divide the gap by $t$ and let $t \to 0$.) compare this with the descent lemma from ch 03 · topic 05, which put a *parabola* above $f$; convexity puts a *plane* below it. together they sandwich $f$, and that sandwich is what all the convergence proofs in topics 02–03 are made of.

the inequality has an immediate consequence: if $\nabla f(x^*) = 0$, then $f(y) \ge f(x^*)$ for every $y$. **for a convex function, a stationary point is a global minimum.** this is what makes "run gradient descent until the gradient is small" a sound stopping rule for convex problems, and a hopeful one for nonconvex problems.

the gap $D_f(y, x) = f(y) - f(x) - \nabla f(x)^\top(y - x)$ has a name, the **Bregman divergence** of $f$. for $f = \tfrac12\|x\|^2$ it's $\tfrac12\|y - x\|^2$; for a quadratic $\tfrac12 x^\top Ax$ it's $\tfrac12(y - x)^\top A(y - x)$. it isn't symmetric in $x$ and $y$, which matters for the checker: the tangent at $x_i$ tested at $x_j$ is a different test from the tangent at $x_j$ tested at $x_i$, and only one of them may fail.

> **▶ build: stage 3.** implement `first_order_gap(f, grad, x, y)` and `min_first_order_gap(f, grad, points)`, which builds the $(k, k)$ matrix of gaps for all **ordered** pairs with broadcasting and returns the smallest off-diagonal entry with its location.

### idea 4: the second-order condition, building blocks, and an honest verdict

for a twice-differentiable $f$, convexity is a condition on curvature: $f$ is convex iff its Hessian is positive semidefinite everywhere,

$$\nabla^2 f(x) \succeq 0\quad\text{for all } x,\qquad\text{i.e.}\quad\lambda_{\min}\big(\nabla^2 f(x)\big) \ge 0.$$

this is the version you'll use most when *proving* convexity by hand, and the strengthened form $\nabla^2 f \succeq \mu I$ with $\mu > 0$ is the **strong convexity** from ch 03 · topic 05. numerically it's a pointwise test: compute the Hessian at a sample point (your `numerical_hessian`), take its smallest eigenvalue, and a negative value anywhere is a witness. for $x^4 - 2x^2$, $f''(x) = 12x^2 - 4$ is negative on $|x| < 1/\sqrt3$.

proving convexity in practice rarely means computing Hessians from scratch. instead you assemble a function from pieces that **preserve convexity**:

- **nonnegative sums**: $\sum_i w_i f_i$ with $w_i \ge 0$. (an average of convex losses over a dataset is convex; a ridge penalty keeps it convex.)
- **affine precomposition**: $f(Ax + b)$ is convex if $f$ is. (a convex loss of a *linear* model's output is convex in the weights: $\ell(w^\top x_i, y_i)$.)
- **pointwise maximum**: $\max_i f_i(x)$ is convex if every $f_i$ is (its epigraph is an intersection of epigraphs). the hinge loss $\max(0, 1 - y\,w^\top x)$ is a max of two affine functions. a pointwise *minimum* usually isn't convex (warm-up 5), just like a union of convex sets.
- **composition**: $h(g(x))$ is convex if $g$ is convex and $h$ is convex and nondecreasing. ($e^{g(x)}$ is convex for convex $g$.) without the monotonicity, it can fail.
- **log-sum-exp** $\log\sum_i e^{x_i}$ is convex (on paper 3), which makes the softmax cross-entropy convex in the logits.

these rules also explain the warm-ups. $(\sigma(wx) - y)^2$ is a convex function (squared error) of a *nonlinear* function of $w$: the affine rule doesn't apply, and it isn't convex. $(ab - 1)^2$ is a convex function of $ab$, but $ab$ isn't affine in $(a, b)$, and the loss isn't convex: $(1, 1)$ and $(-1, -1)$ are both global minima, and their midpoint $(0, 0)$ has loss 1. the model being linear in the *input* doesn't matter; convexity is about the *parameters*.

> **🔮 predict:** your checker tests Jensen on 45 pairs of random points and finds no violation. what can you conclude?

which brings us to the verdict. all four tests (chords of sets, Jensen, tangent planes, Hessians) can **refute** convexity with a single witness. none can **prove** it from finitely many samples: a function can be convex everywhere except in a region your samples missed. so a numerical convexity checker has an asymmetric answer: "not convex, and here's the witness" or "no violation found". the lab makes this concrete. there's also rounding to deal with: at $f \approx 10^{12}$, values are rounded to steps of about $10^{-4}$, so a perfectly convex function can show a Jensen gap of $-10^{-4}$. a sensible checker only counts gaps below $-\text{tol}\cdot(1 + \max|f|)$, a tolerance relative to the size of $f$.

> **▶ build: stage 4.** implement `min_hessian_eigenvalue(f, points)` (smallest eigenvalue of `numerical_hessian` over the points) and `check_convexity(f, points, *, ts, tol)`, which returns `(False, (x, y, t))` with a witness when the most negative Jensen gap is below the relative tolerance, and `(True, None)` otherwise.

## worked example

$f(x) = x^4 - 2x^2$, a double well, with sample points $-1, 0, 1, 2$ and $t \in \{0.25, 0.5, 0.75\}$. the values are $f(-1) = f(1) = -1$, $f(0) = 0$, $f(2) = 8$.

**Jensen.** for the pair $(-1, 1)$ at $t = 0.5$: the chord has height $\tfrac12(-1) + \tfrac12(-1) = -1$ at $x = 0$, while $f(0) = 0$. the gap is $-1 - 0 = -1$: the graph pokes *above* the chord, so $f$ is not convex. all the gaps:

| pair | $t = 0.25$ | $t = 0.5$ | $t = 0.75$ |
|---|---|---|---|
| $(-1, 0)$ | $-0.1289$ | $-0.0625$ | $0.0586$ |
| $(-1, 1)$ | $-0.5625$ | $\mathbf{-1}$ | $-0.5625$ |
| $(-1, 2)$ | $6.4336$ | $3.9375$ | $1.3711$ |
| $(0, 1)$ | $0.0586$ | $-0.0625$ | $-0.1289$ |
| $(0, 2)$ | $5.4375$ | $5$ | $2.4375$ |
| $(1, 2)$ | $2.4961$ | $2.9375$ | $1.9336$ |

so `min_jensen_gap` returns $-1$ at $(i, j, t) = (0, 2, 0.5)$, and `check_convexity` returns `(False, (array([-1.]), array([1.]), 0.5))`. (check one entry: for $(-1, 0)$ at $t = 0.25$ the point is $-0.25$, with $f(-0.25) = 0.00390625 - 0.125 = -0.12109$, and the chord is $0.25 \cdot (-1) + 0.75 \cdot 0 = -0.25$: gap $-0.25 + 0.12109 = -0.1289$.)

**tangent planes.** $f'(x) = 4x^3 - 4x$ is $0$ at $-1$, $0$ and $1$, and $24$ at $2$. the tangent at $x = 0$ is the horizontal line $y = 0$, and $f(-1) = -1$ lies below it: the first-order gap $f(-1) - f(0) - 0 = -1$. for the ordered pair (tangent at $-1$, tested at $0$) the gap is $0 - (-1) - 0 = +1$: the same two points, the other order, no violation. over all ordered pairs the minimum is $-1$, first reached at $(i, j) = (1, 0)$.

**curvature.** $f''(x) = 12x^2 - 4$ is $8, -4, 8, 44$ at the four points: `min_hessian_eigenvalue` returns $(-4, 1)$.

three independent witnesses, all pointing at the hump between the two valleys.

## on paper

1. **intersections.** show that the intersection of any collection of convex sets is convex, and deduce that $\{x : Ax \le b\}$ and the probability simplex are convex. give an example of two convex sets whose union isn't.

<details><summary>solution</summary>

if $x, y$ are in every $C_\alpha$, then for $t \in [0, 1]$, $tx + (1 - t)y$ is in every $C_\alpha$ (each is convex), hence in the intersection. a halfspace $\{a_i^\top x \le b_i\}$ is convex because $a_i^\top(tx + (1 - t)y) = ta_i^\top x + (1 - t)a_i^\top y \le tb_i + (1 - t)b_i = b_i$; the polyhedron is the intersection of $m$ of them. the simplex is the intersection of the halfspaces $x_i \ge 0$ with the hyperplane $\sum x_i = 1$ (two halfspaces). two disjoint disks: each is convex, the union isn't (the chord between their centres leaves it).

</details>

2. **from Jensen to tangent planes.** for differentiable convex $f$, show $f(y) \ge f(x) + \nabla f(x)^\top(y - x)$. hint: rewrite Jensen with $z_t = x + t(y - x)$ as $\frac{f(x + t(y - x)) - f(x)}{t} \le f(y) - f(x)$ and let $t \to 0^+$. then show that if $\nabla f(x^*) = 0$, $x^*$ is a global minimum.

<details><summary>solution</summary>

Jensen with weight $t$ on $y$: $f(x + t(y - x)) = f(ty + (1 - t)x) \le tf(y) + (1 - t)f(x)$, so $f(x + t(y - x)) - f(x) \le t\big(f(y) - f(x)\big)$. divide by $t > 0$; the left side is a difference quotient that tends to the directional derivative $\nabla f(x)^\top(y - x)$ as $t \to 0^+$. so $\nabla f(x)^\top(y - x) \le f(y) - f(x)$. with $\nabla f(x^*) = 0$ this reads $f(y) \ge f(x^*)$ for every $y$.

</details>

3. **log-sum-exp is convex.** for $f(x) = \log\sum_i e^{x_i}$, show that $\nabla f = p$ with $p_i = e^{x_i}/\sum_j e^{x_j}$ (the softmax) and $\nabla^2 f = \operatorname{diag}(p) - pp^\top$. then show $v^\top(\operatorname{diag}(p) - pp^\top)v \ge 0$ for every $v$.

<details><summary>solution</summary>

$\partial f/\partial x_i = e^{x_i}/\sum_j e^{x_j} = p_i$, and $\partial p_i/\partial x_k = p_i\mathbb{1}[i = k] - p_ip_k$, which is the stated Hessian. then $v^\top\nabla^2 f\,v = \sum_i p_iv_i^2 - \big(\sum_i p_iv_i\big)^2$. with $p$ a probability vector, this is $\mathbb{E}[V^2] - (\mathbb{E}V)^2 = \operatorname{Var}(V) \ge 0$ for a random variable $V$ that takes the value $v_i$ with probability $p_i$. it's zero along $v = \mathbf 1$: log-sum-exp is convex but not strictly ($f(x + c\mathbf 1) = f(x) + c$ is linear along that direction).

</details>

4. **logistic regression is convex.** for labels $y_i \in \{0, 1\}$, the mean negative log-likelihood is $\ell(w) = \frac1n\sum_i\big[\log(1 + e^{z_i}) - y_iz_i\big]$ with $z_i = x_i^\top w$. show $\nabla^2\ell(w) = \frac1n X^\top SX$ with $S = \operatorname{diag}\big(\sigma(z_i)(1 - \sigma(z_i))\big)$, and that it's positive semidefinite. when is it positive definite?

<details><summary>solution</summary>

$\frac{d}{dz}\big[\log(1 + e^z) - yz\big] = \sigma(z) - y$ and $\frac{d^2}{dz^2} = \sigma(z)(1 - \sigma(z)) =: s(z) > 0$. by the chain rule with $z_i = x_i^\top w$, $\nabla\ell = \frac1n\sum_i(\sigma(z_i) - y_i)x_i$ and $\nabla^2\ell = \frac1n\sum_i s_ix_ix_i^\top = \frac1n X^\top SX$. then $v^\top\nabla^2\ell\,v = \frac1n\sum_i s_i(x_i^\top v)^2 \ge 0$. it's positive definite iff no nonzero $v$ has $Xv = 0$, i.e. iff $X$ has full column rank (and adding a ridge term $\tfrac\alpha2\|w\|^2$ makes it positive definite always). the same argument is "affine precomposition + nonnegative sum" from idea 4.

</details>

## experiment

`lab.py` investigates three things with your checker:

1. which of eight common ML losses are convex in their parameters, according to your `check_convexity` and `min_hessian_eigenvalue`, and whether that matches the rules from idea 4.
2. how many sample points it takes to catch a function that's nonconvex only in one small region, and what "no violation found" is worth.
3. a function whose sublevel sets are all convex, but which isn't itself convex.

## check your understanding

1. the set of matrices with rank at most 1: convex?

<details><summary>answer</summary>

no. $\begin{pmatrix}1 & 0\\0 & 0\end{pmatrix}$ and $\begin{pmatrix}0 & 0\\0 & 1\end{pmatrix}$ have rank 1, and their midpoint $\tfrac12 I$ has rank 2. (this is why low-rank matrix problems are nonconvex, and why ch 08's nuclear-norm relaxations exist.)

</details>

2. (the predict callout in idea 2.) where is the Jensen gap of $x^4 - 2x^2$ most negative, among $-1, 0, 1, 2$?

<details><summary>answer</summary>

$-1$, for the pair $(-1, 1)$ at $t = 0.5$: the chord joins the two valley floors at height $-1$, and the hump at $x = 0$ rises to 0. see the worked example.

</details>

3. a function is convex on $[0, 1]$ and convex on $[1, 2]$. is it convex on $[0, 2]$?

<details><summary>answer</summary>

not necessarily: $f(x) = -|x - 1|$ is linear (so convex) on each piece, and concave across the kink. convexity is a global property; you can't assemble it from pieces without checking the joins (the slope must not decrease).

</details>

4. (the predict callout in idea 4.) 45 pairs, no violation. what can you conclude?

<details><summary>answer</summary>

only that those 45 chords (at the $t$ values tested) lie above the graph. if $f$ is nonconvex in a region that no tested chord crosses, the checker can't see it. "no violation found" is evidence, and it gets stronger with denser sampling where the function is complicated, but it's never a proof; for that you need an argument (the second-order condition, or the composition rules). lab experiment 2 measures how much sampling it takes.

</details>

5. why is $\max(0, 1 - y\,w^\top x)$ (the hinge loss) convex in $w$, even though it has a kink and no Hessian there?

<details><summary>answer</summary>

it's a pointwise maximum of two affine functions of $w$ ($0$ and $1 - yx^\top w$), and both are convex. the definition (Jensen) doesn't need derivatives at all. only the first- and second-order *tests* need them; at a kink you use subgradients instead (ch 08).

</details>

6. you know $f$ is convex and $\nabla f(x) = 0$. is $x$ the *unique* minimizer?

<details><summary>answer</summary>

it's a global minimizer, but not necessarily unique: $f(x_1, x_2) = x_1^2$ is minimized on the whole line $x_1 = 0$. uniqueness needs strict convexity (e.g. strong convexity, $\nabla^2 f \succeq \mu I$, $\mu > 0$). a ridge penalty is the usual way to buy it.

</details>

7. is $f(w) = \|Xw - y\|_2$ (the norm, not squared) convex? is $\sqrt{f}$?

<details><summary>answer</summary>

$\|Xw - y\|$ is a norm composed with an affine map: convex. $\sqrt{\|Xw - y\|}$ isn't in general: in 1-D with $X = 1$, $y = 0$ it's $\sqrt{|w|}$, which has a cusp at 0 and is concave on each side. (squaring a nonnegative convex function keeps it convex, since $s^2$ is convex and nondecreasing on $s \ge 0$; taking a square root doesn't.)

</details>

## common bugs

- **testing only $t = 0.5$.** symptom: a set made of two disks of very different sizes passes, or a lopsided violation is missed. test several values of $t$, and keep $t$ attached to the first point: $tx_i + (1 - t)x_j$.
- **testing tangent planes only for $i < j$.** symptom: $x^4 - 2x^2$ at the points $-1, 0$ looks convex. the first-order condition isn't symmetric; test every ordered pair.
- **forgetting the diagonal.** symptom: the minimum first-order gap is always exactly 0, at some $(i, i)$. a point's tangent plane touches the function at that point; exclude $i = j$.
- **the largest Hessian eigenvalue instead of the smallest.** symptom: every function looks convex. convexity needs *all* curvatures $\ge 0$.
- **an absolute tolerance.** symptom: a convex function shifted by $10^{12}$ is reported as nonconvex, with a gap of $-10^{-4}$. scale the tolerance by the size of $f$.
- **reading "no violation" as "convex".** symptom: a confident claim, refuted by the next random seed. the checker can only refute.

## going deeper

- Boyd & Vandenberghe (2004), *Convex Optimization*, Cambridge University Press, chapters 2 (convex sets) and 3 (convex functions, including the composition rules). free online. [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/)
- Nesterov (2018), *Lectures on Convex Optimization*, 2nd ed., Springer, §2.1 (smooth convex functions and their equivalent characterizations). [doi:10.1007/978-3-319-91578-4](https://doi.org/10.1007/978-3-319-91578-4)
- Rockafellar (1970), *Convex Analysis*, Princeton University Press. the classic, rigorous reference. [doi:10.1515/9781400873173](https://doi.org/10.1515/9781400873173)
- Bubeck (2015), "Convex optimization: algorithms and complexity", *Foundations and Trends in Machine Learning* 8(3–4), chapter 1. [arXiv:1405.4980](https://arxiv.org/abs/1405.4980)

## connections

- ← builds on: gradients, Hessians and `numerical_hessian` (ch 03 · topic 02), smoothness and strong convexity (ch 03 · topic 05), positive semidefinite matrices and eigenvalues (ch 02 · topic 05).
- → used in: convergence rates for convex and strongly convex functions (topics 02–03), duality and the KKT conditions (topic 06), what goes wrong without convexity (topic 07); convex losses for linear and logistic regression and the lasso (ch 08), the SVM's convex quadratic program (ch 09), and the nonconvex landscapes of neural networks (ch 14, ch 19).
