# wasserstein distances

## why this matters

take two distributions on a line: a thin bump at 0 and the same bump moved to 3. how far apart are they? KL divergence says *infinitely* far, and it says the same about a bump moved to 0.3 or to 3,000. Jensen–Shannon says $\log 2$ for all three. neither notices *how far* the mass has to move, because both compare densities point by point, and where one density is zero the comparison is useless. that's the everyday situation in generative modelling: a generator's samples and the training images live on thin, low-dimensional sets that usually don't overlap at all, and a loss that can't tell "close" from "far" gives the generator no gradient.

**optimal transport** measures distance the way someone moving earth would: the cheapest way to turn one pile of mass into the other, where moving a unit of mass a distance $d$ costs $d$ (or $d^2$). that cost, the **Wasserstein distance**, is $\theta$ for a bump moved by $\theta$. you met its dual form in chapter 21 as the WGAN critic. this block computes it exactly (this topic), makes it fast (topic 16) and uses it in machine learning (topic 17), ending with a project in which optimal-transport couplings straighten your chapter 22 flow-matching paths.

this topic builds the 1-D closed form, the Kantorovich linear program, an exact transportation solver with its dual potentials, and the Gaussian closed form.

## warm-up: predict first

1. $\delta_0$ and $\delta_\theta$ are point masses at 0 and $\theta$. what are $\mathrm{KL}(\delta_0\,\|\,\delta_\theta)$, $\mathrm{JS}(\delta_0, \delta_\theta)$ and the earth mover's distance, as functions of $\theta$?
2. points at 0, 1, 2 must be matched one-to-one with points at 0.5, 1.5, 2.5 at squared-distance cost. which matching is cheapest?
3. one unit of mass sits at 0 and must be sent to two points, 1 and $-1$, half each. is there a *function* $T$ that does this?
4. a transport problem between $n$ and $m$ points is a linear program in how many variables? how many of them can be nonzero at a vertex of its feasible set?
5. two samples of 1,000 points each are drawn from the same 10-dimensional standard Gaussian. is the Wasserstein-2 distance between them closer to 0.01, 0.3 or 2?

## prerequisites check

- [ ] KL divergence, Jensen–Shannon, and when KL is infinite (→ ch 05 · topic 05)
- [ ] linear programs, duality, and `scipy.optimize.linprog` (→ ch 06 · topic 06)
- [ ] quantile functions, inverse CDFs (→ ch 04)
- [ ] symmetric matrix square roots by eigendecomposition (→ ch 02)
- [ ] the WGAN critic and the Kantorovich–Rubinstein dual, and the Fréchet distance of FID (→ ch 21 · topics 01 and 06)

## core ideas

### idea 1: moving mass, and the 1-D closed form

**notation.** a discrete distribution is $\mu = \sum_{i=1}^n a_i\delta_{x_i}$: mass $a_i \ge 0$ at the point $x_i$. the target is $\nu = \sum_{j=1}^m b_j\delta_{y_j}$ with the same total mass. moving a unit of mass from $x$ to $y$ costs $c(x, y) = \|x - y\|^p$, and $C_{ij} = c(x_i, y_j)$ is the **cost matrix**.

**Monge** (1781) asked for a *map* $T$ that sends each point to one destination, so that the mass arriving at every $y_j$ is right (written $T_\#\mu = \nu$, "$T$ pushes $\mu$ forward to $\nu$"), at the least total cost $\sum_i a_i c(x_i, T(x_i))$. that's natural, but sometimes impossible: one unit at 0 can't be sent half to 1 and half to $-1$ by any function. **Kantorovich** (1942) relaxed the map to a **transport plan** $P \in \mathbb R^{n\times m}_{\ge 0}$, where $P_{ij}$ is the mass moved from $x_i$ to $y_j$. mass is conserved, so the rows sum to $a$ and the columns to $b$:

$$U(a, b) = \{P \ge 0 : P\mathbf 1 = a,\ P^\top\mathbf 1 = b\},\qquad W_p(\mu, \nu) = \Big(\min_{P \in U(a, b)}\ \sum_{ij}P_{ij}C_{ij}\Big)^{1/p}.$$

$W_p$ is a true metric on distributions (Villani, 2009, ch. 6): symmetric, zero only for equal distributions, and it satisfies the triangle inequality. $W_1$ is the earth mover's distance of chapter 21.

**on a line, the problem solves itself.** in 1-D with a convex cost, crossing paths never pay: if $x < x'$ and $y < y'$, sending $x \to y$ and $x' \to y'$ costs no more than $x \to y'$ and $x' \to y$ (on paper 2). so the optimal plan is **monotone**: the leftmost mass goes to the leftmost destination, and so on. formally, with $F^{-1}$ and $G^{-1}$ the quantile functions (inverse CDFs) of $\mu$ and $\nu$,

$$W_p(\mu, \nu)^p = \int_0^1\big|F^{-1}(t) - G^{-1}(t)\big|^p\,dt .$$

for discrete distributions both quantile functions are step functions that jump at the cumulative weights of the sorted points, so the integral is a finite sum over the merged breakpoints of both. for $p = 1$ (only) it also equals $\int|F - G|\,dx$, the area between the CDFs: the same area measured horizontally instead of vertically.

> **spec.** `wasserstein_1d(u, v, p=1, *, u_weights=None, v_weights=None)`: $W_p$ (not $W_p^p$) between $\sum_i w_i\delta_{u_i}$ and $\sum_j w'_j\delta_{v_j}$ for 1-D arrays `u` (n,) and `v` (m,), any $p \ge 1$. inputs need not be sorted; weights default to uniform, need not sum to 1 (only their ratios matter), and a point of weight 0 carries no mass. returns a float.

> **▶ build: stage 1.** implement `wasserstein_1d` with the quantile formula.

### idea 2: the Kantorovich problem is a linear program

in more than one dimension there's no ordering to exploit, but the problem above is still a **linear program**: the objective $\langle C, P\rangle = \sum_{ij}C_{ij}P_{ij}$ is linear, and $U(a, b)$ is cut out by $n + m$ linear equations and $nm$ sign constraints. flatten $P$ row by row into a vector of length $nm$ (entry $(i, j)$ at index $im + j$); then the row sums are $(I_n \otimes \mathbf 1_m^\top)\,\mathrm{vec}(P)$ and the column sums $(\mathbf 1_n^\top \otimes I_m)\,\mathrm{vec}(P)$, two Kronecker products stacked into one $(n + m)\times nm$ constraint matrix $A$. one of its rows is redundant (both blocks sum to the total mass), which LP solvers handle on their own.

two consequences of the LP view matter for the rest of the block. first, **the optimum sits at a vertex** of the polytope $U(a, b)$, and a vertex has at most $n + m - 1$ nonzero entries: an optimal plan is sparse. second, when $n = m$ and all weights are $1/n$, the vertices of $U$ are exactly the permutation matrices divided by $n$ (Birkhoff's theorem), so the Kantorovich optimum *is* a Monge map, an **assignment**. that's why SciPy's `linear_sum_assignment` (the Hungarian algorithm) solves uniform, equal-size problems exactly and fast.

> **spec.** `cost_matrix(X, Y, p=2)`: the `(n, m)` array of $\|x_i - y_j\|^p$ (Euclidean norm) for `X` (n, d) and `Y` (m, d); 1-D arrays are points on a line. `transport_constraints(n, m)`: the `(n + m, n*m)` matrix $A$ with `A @ P.ravel()` equal to `concatenate([P.sum(1), P.sum(0)])` for every `(n, m)` array `P`. `kantorovich_lp(a, b, C)`: solves the LP with `scipy.optimize.linprog(..., method="highs")` and returns `(P, cost)` with `P` the `(n, m)` optimal plan and `cost` $= \langle C, P\rangle$. `a` and `b` must have the same total mass (not necessarily 1; `ValueError` otherwise, relative tolerance $10^{-9}$).

> **▶ build: stage 2.** implement `cost_matrix`, `transport_constraints` and `kantorovich_lp`.

### idea 3: duality, and a solver of your own

every LP has a dual. for the transport problem it reads

$$\max_{f \in \mathbb R^n,\ g \in \mathbb R^m}\ \langle a, f\rangle + \langle b, g\rangle\quad\text{subject to}\quad f_i + g_j \le C_{ij}\ \text{for all } i, j .$$

think of a shipper charging $f_i$ per unit picked up at $x_i$ and $g_j$ per unit delivered at $y_j$: you only accept if no route is cheaper to do yourself ($f_i + g_j \le C_{ij}$), and the shipper maximizes its income. **strong duality** says its best income equals your cheapest plan, and **complementary slackness** says the optimal plan only uses routes where the price is exactly fair: $P_{ij} > 0 \Rightarrow f_i + g_j = C_{ij}$. the $f$ and $g$ are the **Kantorovich potentials**. for $W_1$, the constraint becomes "$f$ is 1-Lipschitz" with $g = -f$, which is precisely the WGAN critic of chapter 21 (on paper 3).

the **transportation simplex** (Dantzig, 1951) uses these conditions directly, and it's the algorithm you'll build:

1. **start** with any vertex: the **north-west corner rule** fills the top-left cell with as much mass as fits, $\min(a_0, b_0)$, then moves down if row 0 is used up and right otherwise, and repeats until it reaches the bottom-right cell. it visits exactly $n + m - 1$ cells, the **basis**, a spanning tree of the bipartite graph of rows and columns (some basic cells may carry zero mass; that's **degeneracy**, and it's common).
2. **prices**: set $f_0 = 0$ and solve $f_i + g_j = C_{ij}$ on the basic cells.
3. **check**: if every reduced cost $C_{ij} - f_i - g_j$ is $\ge 0$, the prices are dual feasible, complementary slackness holds, and the plan is optimal. stop.
4. **pivot**: otherwise a cell with negative reduced cost **enters**. adding it to the tree closes one cycle, alternating rows and columns. shift mass $\theta$ around the cycle, $+$ on the entering cell, $-$ on the next, $+$, $-$, …, with $\theta$ the smallest mass on a $-$ cell; that cell **leaves** the basis. go to 2.

on degenerate problems $\theta$ is often 0, and with an unlucky choice of entering and leaving cells the method can, in principle, cycle forever (it's rare in practice). **Bland's rule** prevents it: always enter the eligible cell with the smallest flat index $im + j$, and on ties for leaving, remove the smallest index. it's slower, but it always terminates.

> **spec.** `northwest_corner(a, b)`: returns `(P, basis)`: the north-west corner plan and the list of its $n + m - 1$ visited cells `(i, j)` in order. at each cell allocate $\min(\text{row left}, \text{column left})$ (never negative); then, in the last row move right, in the last column move down, otherwise move down if the row's remaining mass is $\le$ the column's (so ties move **down**), else right. `transport_simplex(a, b, C, *, max_iters=10_000)`: returns `(P, f, g)`, an optimal plan at a vertex (at most $n + m - 1$ nonzeros) and potentials with $f_0 = 0$, $f_i + g_j \le C_{ij}$ everywhere (up to a relative tolerance like $10^{-9}$) and equality on the plan's support. starts from the north-west corner, performs at most `max_iters` pivots, and raises `RuntimeError` if the plan is still not optimal after them (so `max_iters=0` returns the north-west corner plan if, and only if, it is already optimal). it must terminate on degenerate problems (Bland's rule guarantees it). `ValueError` if the masses differ.

> **🔮 predict:** for a sorted 1-D problem with a convex cost, how many pivots does the simplex need after the north-west corner start?

> **▶ build: stage 3.** implement `northwest_corner` and `transport_simplex`.

### idea 4: back to maps, and the Gaussian closed form

a plan tells you where mass goes; often you want a **map**, a function you can apply to new points. the **barycentric projection** of a plan sends each source point to the average of its destinations, weighted by how much mass goes where:

$$T(x_i) = \frac{\sum_j P_{ij}\,y_j}{\sum_j P_{ij}} .$$

when the plan is a permutation (the uniform equal-size case), $T$ is exactly the optimal Monge map. when mass splits, $T$ is a compromise: in the warm-up's example it sends 0 to the midpoint of $1$ and $-1$, which isn't a valid transport at all. topic 17 uses barycentric maps to carry labelled source data onto an unlabelled target domain.

in continuous space, **Brenier's theorem** (1991) says that for the squared cost, if $\mu$ has a density, the optimal plan *is* a map, and that map is the gradient of a convex function. for Gaussians everything is explicit. the optimal map from $\mathcal N(m_1, \Sigma_1)$ to $\mathcal N(m_2, \Sigma_2)$ is affine with a symmetric positive-definite matrix,

$$T(x) = m_2 + A(x - m_1),\qquad A = \Sigma_1^{-1/2}\big(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}\big)^{1/2}\Sigma_1^{-1/2},$$

and the distance has the **Bures** closed form

$$W_2^2 = \|m_1 - m_2\|^2 + \operatorname{tr}\Big(\Sigma_1 + \Sigma_2 - 2\big(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}\big)^{1/2}\Big).$$

you've computed it before: it's the **Fréchet distance** behind FID (ch 21). many matrices push $\Sigma_1$ to $\Sigma_2$ ($T$ maps $\mathcal N(m_1, \Sigma_1)$ to a Gaussian with covariance $A\Sigma_1A^\top$, so for instance $A = \Sigma_2^{1/2}\Sigma_1^{-1/2}$ works); only the symmetric one is optimal, because only it is the gradient of a convex function.

> **spec.** `barycentric_map(P, Y)`: the `(n, d)` array of $T(x_i)$ for a plan `P` (n, m) with positive row sums and targets `Y` (m, d), or `(n,)` for 1-D targets `(m,)`; rows are divided by their own mass. `gaussian_w2(m1, S1, m2, S2)`: $W_2$ (not squared) between Gaussians with means `(d,)` and covariances `(d, d)` (positive semi-definite; use symmetric square roots). `gaussian_monge_map(m1, S1, m2, S2)`: `(A, t)` with `A` symmetric `(d, d)` and `t` `(d,)` such that `x @ A + t` is the optimal map applied to a row vector `x` (for invertible `S1`). (on rows, `x @ A` acts as $A^\top$ does on columns; for the symmetric optimal $A$ the two agree.)

> **🔮 predict:** $W_2(\mathcal N(0, \sigma^2), \mathcal N(0, \tau^2))$ in 1-D: is it $|\sigma - \tau|$, $|\sigma^2 - \tau^2|$ or $\sqrt{\sigma^2 + \tau^2}$?

> **▶ build: stage 4.** implement `barycentric_map`, `gaussian_w2` and `gaussian_monge_map`.

## worked example

$\mu = \tfrac12\delta_0 + \tfrac12\delta_2$, $\nu = \tfrac14\delta_1 + \tfrac34\delta_3$.

**1-D closed form.** $F^{-1}(t) = 0$ for $t \le \tfrac12$ and 2 after; $G^{-1}(t) = 1$ for $t \le \tfrac14$ and 3 after. the merged breakpoints $\tfrac14, \tfrac12$ give three pieces: on $(0, \tfrac14]$ the gap is $|0 - 1| = 1$, on $(\tfrac14, \tfrac12]$ it's $|0 - 3| = 3$, on $(\tfrac12, 1]$ it's $|2 - 3| = 1$. so $W_1 = \tfrac14\cdot1 + \tfrac14\cdot3 + \tfrac12\cdot1 = 1.5$ and $W_2 = \sqrt{\tfrac14 + \tfrac94 + \tfrac12} = \sqrt3 \approx 1.732$.

**the plan.** the pieces say: $\tfrac14$ of mass goes $0 \to 1$, $\tfrac14$ goes $0 \to 3$, $\tfrac12$ goes $2 \to 3$:

$$P = \begin{pmatrix}\tfrac14 & \tfrac14\\ 0 & \tfrac12\end{pmatrix},$$

which is exactly the north-west corner plan of $a = (\tfrac12, \tfrac12)$, $b = (\tfrac14, \tfrac34)$: fill $(0, 0)$ with $\tfrac14$, column 0 is used up, move right, $(0, 1)$ takes $\tfrac14$, row 0 is used up, move down, $(1, 1)$ takes $\tfrac12$. basis $(0,0), (0,1), (1,1)$.

**prices.** with the squared cost, $C = \begin{pmatrix}1 & 9\\ 1 & 1\end{pmatrix}$. $f_0 = 0$ gives $g_0 = 1$ and $g_1 = 9$, then $f_1 = C_{11} - g_1 = -8$. the one non-basic cell has reduced cost $C_{10} - f_1 - g_0 = 1 + 8 - 1 = 8 \ge 0$: optimal, without a single pivot. the dual value is $\tfrac12\cdot0 + \tfrac12\cdot(-8) + \tfrac14\cdot1 + \tfrac34\cdot9 = 3 = W_2^2$.

**maps.** the barycentric map sends $0 \mapsto \tfrac12\cdot1 + \tfrac12\cdot3 = 2$ (its mass splits, so no Monge map exists) and $2 \mapsto 3$. **Gaussians:** $W_2(\mathcal N(1, 4), \mathcal N(-2, 1))^2 = (1 + 2)^2 + (2 - 1)^2 = 10$, and the optimal map is $T(x) = -2 + \tfrac12(x - 1)$, i.e. $A = \tfrac12$, $t = -2.5$.

your `wasserstein_1d`, `kantorovich_lp`, `northwest_corner`, `transport_simplex`, `barycentric_map`, `gaussian_w2` and `gaussian_monge_map` must reproduce these numbers.

## on paper

1. **point masses.** compute $\mathrm{KL}(\delta_0\,\|\,\delta_\theta)$, $\mathrm{JS}(\delta_0, \delta_\theta)$ and $W_p(\delta_0, \delta_\theta)$ for $\theta \ne 0$, and their derivatives in $\theta$.

<details><summary>solution</summary>

for $\theta \ne 0$, $\delta_\theta$ puts zero mass where $\delta_0$ puts mass 1, so $\mathrm{KL} = 1\cdot\log(1/0) = \infty$. the mixture $m = \tfrac12(\delta_0 + \delta_\theta)$ gives $\mathrm{KL}(\delta_0\|m) = \log 2$ and the same for $\delta_\theta$, so $\mathrm{JS} = \log 2$, constant, with derivative 0. the only plan moves all the mass by $|\theta|$, so $W_p = |\theta|$ for every $p$, with derivative $\operatorname{sign}\theta$: a gradient that always points the right way and never vanishes.

</details>

2. **no crossings in 1-D.** let $x < x'$, $y < y'$ and $c(x, y) = h(y - x)$ with $h$ convex. show $c(x, y) + c(x', y') \le c(x, y') + c(x', y)$.

<details><summary>solution</summary>

write $s = y - x'$, $t = y' - x$; the four differences satisfy $y - x' \le \{y - x,\ y' - x'\} \le y' - x$ and $(y - x) + (y' - x') = (y' - x) + (y - x')$. so $y - x = \lambda s + (1 - \lambda)t$ and $y' - x' = (1 - \lambda)s + \lambda t$ for some $\lambda \in [0, 1]$. by convexity, $h(y - x) + h(y' - x') \le \lambda h(s) + (1 - \lambda)h(t) + (1 - \lambda)h(s) + \lambda h(t) = h(s) + h(t)$. uncrossing two routes never costs more, so a monotone plan is optimal; it's what the quantile formula computes. (for a concave cost like $\sqrt{|x - y|}$ the inequality can reverse, and the 1-D formula can fail.)

</details>

3. **the dual, and the critic.** derive the dual LP from the Lagrangian $\langle C, P\rangle + \langle f, a - P\mathbf 1\rangle + \langle g, b - P^\top\mathbf 1\rangle$ with $P \ge 0$. then show that for $c(x, y) = |x - y|$ on a set of points one may take $g = -f$ with $|f(x) - f(y)| \le |x - y|$.

<details><summary>solution</summary>

the Lagrangian equals $\langle a, f\rangle + \langle b, g\rangle + \sum_{ij}P_{ij}(C_{ij} - f_i - g_j)$. minimizing over $P \ge 0$ gives $-\infty$ unless every $C_{ij} - f_i - g_j \ge 0$, in which case the minimum is $\langle a, f\rangle + \langle b, g\rangle$: that's the dual. for a metric cost, given feasible $(f, g)$, replace $g$ by $f^c(y) = \min_x\big(c(x, y) - f(x)\big) \ge g(y)$ (still feasible, objective no smaller), then $f$ by $f^{cc}$. a function of the form $\min_x(|x - y| - f(x))$ is 1-Lipschitz, and for 1-Lipschitz $\phi$, $\phi^c = -\phi$ (take $x = y$ for $\le$; Lipschitz for $\ge$). so $W_1 = \max_{\phi\ \text{1-Lip}}\ \mathbb E_\mu\phi - \mathbb E_\nu\phi$, the Kantorovich–Rubinstein formula behind the WGAN critic.

</details>

4. **1-D Gaussians.** using the quantile formula, show $W_2(\mathcal N(m_1, \sigma_1^2), \mathcal N(m_2, \sigma_2^2))^2 = (m_1 - m_2)^2 + (\sigma_1 - \sigma_2)^2$.

<details><summary>solution</summary>

$F^{-1}(t) = m_1 + \sigma_1\Phi^{-1}(t)$ and $G^{-1}(t) = m_2 + \sigma_2\Phi^{-1}(t)$, so $W_2^2 = \int_0^1\big((m_1 - m_2) + (\sigma_1 - \sigma_2)z(t)\big)^2dt$ with $z = \Phi^{-1}(t)$. substituting $t = \Phi(z)$ turns it into $\mathbb E\big[((m_1 - m_2) + (\sigma_1 - \sigma_2)Z)^2\big]$ for $Z \sim \mathcal N(0, 1)$, which is $(m_1 - m_2)^2 + (\sigma_1 - \sigma_2)^2$. the map is $T(x) = m_2 + (\sigma_2/\sigma_1)(x - m_1)$, the 1-D case of $A$.

</details>

## experiment

experiment 1 compares KL, JS and $W_1$ between two sliding bumps, and fits the shift by gradient descent on each. experiment 2 times your exact solvers against HiGHS and the Hungarian algorithm. experiment 3 measures $W_2$ between two samples of the *same* distribution, in 1 to 10 dimensions.

## check your understanding

1. why can't Monge's formulation handle $\delta_0 \to \tfrac12\delta_{-1} + \tfrac12\delta_1$, and how does Kantorovich's fix it?

<details><summary>answer</summary>

a map sends all the mass at a point to a single destination, so it can't split the unit at 0. a plan is a joint distribution: it can put $\tfrac12$ on $(0, -1)$ and $\tfrac12$ on $(0, 1)$. Kantorovich's problem is also convex (an LP), while Monge's isn't even guaranteed to have a feasible point.

</details>

2. the transport LP has $nm$ variables. why does an optimal vertex have at most $n + m - 1$ nonzero entries?

<details><summary>answer</summary>

at a vertex of $\{x \ge 0 : Ax = b\}$ the nonzero variables have linearly independent columns of $A$, so there are at most $\operatorname{rank}A$ of them. the constraint matrix has $n + m$ rows but rank $n + m - 1$ (the row-sum and column-sum blocks both add up to the total mass). graphically, the support of a vertex plan is a forest in the bipartite graph of rows and columns: no cycles, so at most $n + m - 1$ edges.

</details>

3. your simplex says the plan is optimal when every reduced cost is $\ge 0$. why is that enough?

<details><summary>answer</summary>

nonnegative reduced costs mean $f_i + g_j \le C_{ij}$ everywhere, so $(f, g)$ is dual feasible; by construction $f_i + g_j = C_{ij}$ on every basic cell, which contains the plan's support, so $\langle C, P\rangle = \sum_{ij}P_{ij}(f_i + g_j) = \langle a, f\rangle + \langle b, g\rangle$. a primal feasible plan whose cost equals a dual feasible value is optimal by weak duality.

</details>

4. Bland's rule is slower than "enter the most negative reduced cost". why use it?

<details><summary>answer</summary>

pivots with $\theta = 0$ change the basis but not the plan, and (rarely) a sequence of them returns to an earlier basis and cycles. Bland's rules provably never revisit a basis. uniform assignment problems are as degenerate as it gets: $2n - 1$ basic cells, only $n$ nonzero.

</details>

5. a colleague reports a $W_2$ of 1.9 between 500 generated and 500 real samples in a 10-dimensional feature space and concludes the model is poor. what would you ask?

<details><summary>answer</summary>

what $W_2$ two independent samples of 500 *real* points have. in 10 dimensions the empirical distance converges like $n^{-1/10}$, so for unit-variance features even identical distributions give a value near 2 at this sample size (the lab measures it). only the difference from that floor, at the same $n$, says anything.

</details>

## common bugs

- **returning $W_p^p$**: the worked example then gives 3 instead of $\sqrt3$ for $W_2$.
- **sorting the points but not their weights**, or comparing $F$ and $G$ at the same *index* when the samples have different sizes. merge the breakpoints.
- **flattening $P$ in one order and building $A$ in the other**: the "plan" then has the wrong marginals whenever $n \ne m$.
- **dropping degenerate basic cells.** when a row and a column run out together, the north-west corner must still record a zero-mass cell, or the basis stops being a spanning tree.
- **wrong signs around the cycle**: the cell next to the entering one, in the entering cell's column, is a $-$ cell. with the signs flipped, mass goes negative.
- **non-symmetric matrix square roots** in the Gaussian formulas. take $\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}$, which is symmetric, and use `eigh`.

## going deeper

- Villani (2009), *Optimal Transport: Old and New*, Springer. chapters 4–6: existence, Kantorovich duality, the Wasserstein metric. [doi:10.1007/978-3-540-71050-9](https://doi.org/10.1007/978-3-540-71050-9)
- Peyré & Cuturi (2019), *Computational Optimal Transport*, Foundations and Trends in Machine Learning 11(5–6). chapters 2–3: the LP, the north-west corner rule, network simplex, the 1-D and Gaussian cases. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
- Santambrogio (2015), *Optimal Transport for Applied Mathematicians*, Birkhäuser. chapter 2 (1-D) and Brenier's theorem. [doi:10.1007/978-3-319-20828-2](https://doi.org/10.1007/978-3-319-20828-2)
- Dantzig (1951), "Application of the simplex method to a transportation problem", in Koopmans (ed.), *Activity Analysis of Production and Allocation*, Wiley. the transportation simplex.
- Brenier (1991), "Polar factorization and monotone rearrangement of vector-valued functions", *Communications on Pure and Applied Mathematics* 44(4). [doi:10.1002/cpa.3160440402](https://doi.org/10.1002/cpa.3160440402)
- Bertsimas & Tsitsiklis (1997), *Introduction to Linear Optimization*, Athena Scientific. chapter 7: the network simplex method; chapter 3: Bland's rule and cycling.
- Dowson & Landau (1982), "The Fréchet distance between multivariate normal distributions", *Journal of Multivariate Analysis* 12(3). [doi:10.1016/0047-259X(82)90077-X](https://doi.org/10.1016/0047-259X(82)90077-X)
- Fournier & Guillin (2015), "On the rate of convergence in Wasserstein distance of the empirical measure", *Probability Theory and Related Fields* 162. [arXiv:1312.2128](https://arxiv.org/abs/1312.2128)
- Arjovsky, Chintala & Bottou (2017), "Wasserstein generative adversarial networks", *ICML*. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

## connections

- ← builds on: KL and JS divergences (ch 05); linear programming and duality (ch 06); the WGAN critic and the Fréchet distance (ch 21 · topics 06 and 01); quantile functions (ch 04).
- → used in: topic 16 smooths this LP into Sinkhorn's algorithm and compares against your exact solver; topic 17 uses exact minibatch couplings (assignments) in flow matching and barycentric maps for domain adaptation. after this topic you may use `scipy.optimize.linear_sum_assignment` and `linprog` for exact OT, and `scipy.stats.wasserstein_distance` for 1-D $W_1$, since you've built each of them.
