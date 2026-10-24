# entropic OT & sinkhorn

## why this matters

topic 15's exact solvers have two problems for machine learning. they're slow: the LP between two clouds of a few hundred points takes about a second, and you'd want one per training step. and their output is brittle: the optimal plan jumps from one vertex of the polytope to another when a point moves slightly, so the plan isn't a smooth function of the data, and neither is anything you'd like to differentiate through it.

Cuturi (2013) fixed both with one idea: add an entropy term to the objective. the regularized problem has a unique, smooth, strictly positive solution, and it's computed by **Sinkhorn's algorithm**, which alternately rescales the rows and columns of a matrix, nothing but matrix–vector products. it parallelizes perfectly (it's how optimal transport runs on GPUs), it's differentiable, and it trades exactness for speed through a single knob, the regularization strength $\varepsilon$.

the price comes in two parts, and this topic measures both. the naive algorithm overflows or underflows exactly when you want a sharp answer, so you'll build the **log-domain** version that stays finite at any $\varepsilon$. and the regularized cost is **biased**: it's not zero between a distribution and itself, and minimizing it pulls a model towards a shrunken copy of the data. the **debiased Sinkhorn divergence** (Feydy et al., 2019) removes that bias, and it's the loss topic 17 builds on.

## warm-up: predict first

1. as $\varepsilon \to \infty$, the entropic plan between $a$ and $b$ tends to what? as $\varepsilon \to 0$?
2. one Sinkhorn iteration on an $n\times n$ problem costs how many operations? how does that compare with one LP solve?
3. what is `np.exp(-1000.0)` in float64? and `np.exp(-700.0)`?
4. is the entropic transport cost between a distribution and *itself* zero?
5. you move a point cloud by gradient descent to minimize its entropic transport cost to a ring of points. where do the points end up?

## prerequisites check

- [ ] the Kantorovich LP, its dual, and Kantorovich potentials $f, g$ (→ topic 15)
- [ ] entropy and KL divergence (→ ch 05 · topic 05)
- [ ] a stable `logsumexp` and why it subtracts the max (→ ch 00 · topic 03)
- [ ] Lagrangians and coordinate ascent (→ ch 06)

## core ideas

### idea 1: regularize with entropy, solve by scaling

**notation** (as in topic 15): $a \in \mathbb R^n$, $b \in \mathbb R^m$ are positive weights summing to 1, $C$ the $n\times m$ cost matrix, $U(a, b)$ the plans with marginals $a$ and $b$. the **entropic OT** problem adds a KL penalty that pulls the plan towards the independent coupling $ab^\top$:

$$\mathrm{OT}_\varepsilon(a, b) = \min_{P \in U(a, b)}\ \langle C, P\rangle + \varepsilon\,\mathrm{KL}(P\,\|\,ab^\top),\qquad \mathrm{KL}(P\,\|\,Q) = \sum_{ij}P_{ij}\log\frac{P_{ij}}{Q_{ij}} - P_{ij} + Q_{ij}.$$

(the last two terms make KL nonnegative for any nonnegative matrices; on $U(a, b)$ they cancel. Cuturi's original penalty, $-\varepsilon H(P)$, gives the same plan, since on $U(a, b)$ the two differ by a constant.) a large $\varepsilon$ prefers spread-out, high-entropy plans; as $\varepsilon \to 0$ the problem returns to topic 15's LP.

KL is strictly convex, so the solution is unique, and the Lagrangian (on paper 1) shows its form: with multipliers $f$ for the row constraints and $g$ for the columns,

$$P_{ij} = a_ib_j\,e^{(f_i + g_j - C_{ij})/\varepsilon} = u_i\,K_{ij}\,v_j,\qquad K = e^{-C/\varepsilon}\ \text{(elementwise)}.$$

the optimal plan is the **Gibbs kernel** $K$ rescaled by a positive vector on each side, $P = \operatorname{diag}(u)\,K\operatorname{diag}(v)$. so solving the problem means finding $u$ and $v$, and the marginal constraints say $u \odot (Kv) = a$ and $v \odot (K^\top u) = b$. **Sinkhorn's algorithm** (Sinkhorn & Knopp, 1967) satisfies them alternately:

$$u \leftarrow a \oslash (Kv),\qquad v \leftarrow b \oslash (K^\top u),$$

starting from $v = \mathbf 1$. after a $v$-update the columns are exact and the rows are off; the row error $\|P\mathbf 1 - a\|_1$ measures convergence. each iteration is two matrix–vector products, $O(nm)$, with no linear solves.

a useful invariant: since $P_{ij} = u_iK_{ij}v_j$, the ratio $P_{ij}P_{i'j'}/(P_{ij'}P_{i'j})$ equals $e^{-(C_{ij} + C_{i'j'} - C_{ij'} - C_{i'j})/\varepsilon}$, whatever $u$ and $v$ are. the plan's "shape" is fixed by the cost; the scalings only fix the marginals.

> **spec.** `sinkhorn(a, b, C, eps, *, max_iters=1000, tol=1e-9)`: with $K = e^{-C/\varepsilon}$ and $v = \mathbf 1$, repeat (at most `max_iters` times) $u = a/(Kv)$, $v = b/(K^\top u)$, $P = \operatorname{diag}(u)K\operatorname{diag}(v)$, and stop as soon as $\|P\mathbf 1 - a\|_1 \le$ `tol`. returns the `(n, m)` plan. no safeguards: at small $\varepsilon$ this version breaks, and stage 2 fixes it.

> **▶ build: stage 1.** implement `sinkhorn`.

### idea 2: the log domain

$K_{ij} = e^{-C_{ij}/\varepsilon}$ underflows to exactly 0 in float64 once $C_{ij}/\varepsilon > 745$. with costs around 1 that's $\varepsilon < 0.0013$. once a whole row of $K$ is 0, $u = a/(Kv)$ is infinite, the next product is $\infty\cdot0$, and the plan is NaN; and the scalings $u, v$ themselves span hundreds of orders of magnitude well before that, so they overflow even when $K$ doesn't.

the fix is to never form $K$. write the scalings as potentials, $u_i = a_ie^{f_i/\varepsilon}$ and $v_j = b_je^{g_j/\varepsilon}$, and take logs of the updates. $u \leftarrow a\oslash(Kv)$ becomes

$$f_i \leftarrow -\varepsilon\log\sum_j b_j\,e^{(g_j - C_{ij})/\varepsilon} = -\varepsilon\,\mathrm{LSE}_j\Big(\log b_j + \frac{g_j - C_{ij}}{\varepsilon}\Big),$$

and symmetrically $g_j \leftarrow -\varepsilon\,\mathrm{LSE}_i\big(\log a_i + (f_i - C_{ij})/\varepsilon\big)$. a log-sum-exp subtracts its largest term before exponentiating, so every exponential is at most 1 and nothing overflows; terms that underflow are genuinely negligible next to the largest one. these updates are the **soft $c$-transforms**: as $\varepsilon \to 0$, $-\varepsilon\,\mathrm{LSE}$ becomes a minimum and $f_i \leftarrow \min_j(C_{ij} - g_j)$, exactly topic 15's $c$-transform. the iterations are mathematically identical to stage 1's (with a different starting point, $g = 0$), and cost about the same.

the log domain fixes the arithmetic, not the speed. a Sinkhorn step moves the potentials by an amount that shrinks with $\varepsilon$, so the number of iterations grows roughly like $1/\varepsilon$ (Altschuler et al., 2017, prove a bound of this flavour). the lab measures it.

> **spec.** `sinkhorn_log(a, b, C, eps, *, max_iters=1000, tol=1e-9)`: start from $g = 0$; repeat (at most `max_iters` times) the $f$-update, then the $g$-update above (with your ch 00 `logsumexp`), form $P_{ij} = a_ib_je^{(f_i + g_j - C_{ij})/\varepsilon}$ and stop as soon as $\|P\mathbf 1 - a\|_1 \le$ `tol`. returns `(P, f, g)`. it must stay finite, without overflow warnings, for any $C/\varepsilon$.

> **🔮 predict:** with costs around 1, how many times more iterations does Sinkhorn need at $\varepsilon = 0.001$ than at $\varepsilon = 0.1$? and at which $\varepsilon$ does the naive version first fail?

> **▶ build: stage 2.** implement `sinkhorn_log`.

### idea 3: the dual, and what Sinkhorn is really doing

the entropic problem has an unconstrained dual (on paper 1):

$$\mathrm{OT}_\varepsilon(a, b) = \max_{f, g}\ \langle f, a\rangle + \langle g, b\rangle - \varepsilon\sum_{ij}a_ib_j\Big(e^{(f_i + g_j - C_{ij})/\varepsilon} - 1\Big).$$

compare topic 15: the hard constraint $f_i + g_j \le C_{ij}$ has become a smooth exponential penalty for violating it. setting the derivative with respect to $f_i$ to zero gives exactly the $f$-update of idea 2, and likewise for $g$. so **Sinkhorn is block coordinate ascent on the dual**: each half-step maximizes the dual exactly over one block of variables, the dual value can only go up, and at convergence the mass term vanishes (the plan's entries sum to 1) and

$$\mathrm{OT}_\varepsilon(a, b) = \langle f, a\rangle + \langle g, b\rangle,$$

equal to the primal value at the converged plan (strong duality). weak duality holds everywhere: any $(f, g)$ gives a dual value below any feasible plan's primal value.

how far is $\mathrm{OT}_\varepsilon$ from the exact $W$? the entropic plan is feasible for the LP, so its transport cost $\langle C, P_\varepsilon\rangle$ is at least $W$; and comparing it with the exact plan in the regularized objective gives (on paper 2)

$$0 \le \langle C, P_\varepsilon\rangle - W \le \varepsilon\,\big(H(a) + H(b)\big) \le \varepsilon\log(nm).$$

so to approximate $W$ to accuracy $\delta$ you need $\varepsilon \sim \delta/\log n$, and then about $1/\varepsilon$ iterations.

> **spec.** `entropic_primal(P, a, b, C, eps)`: $\langle C, P\rangle + \varepsilon\,\mathrm{KL}(P\,\|\,ab^\top)$ with the generalized KL above and $0\log0 = 0$, for any nonnegative `P`. `entropic_dual(f, g, a, b, C, eps)`: the dual objective above. both return floats.

> **▶ build: stage 3.** implement `entropic_primal` and `entropic_dual`.

### idea 4: debiasing: the Sinkhorn divergence

$\mathrm{OT}_\varepsilon$ is not a distance: $\mathrm{OT}_\varepsilon(a, a) > 0$, because the regularized plan between a distribution and itself is blurred, and blurring costs transport. worse, $\mathrm{OT}_\varepsilon(\,\cdot\,, \nu)$ isn't minimized at $\nu$. by the **envelope theorem**, the gradient of $\mathrm{OT}_\varepsilon$ with respect to a source point $x_i$ ignores how the plan changes and is $\sum_jP_{ij}\nabla_xc(x_i, y_j)$; for the squared cost, divided by the point's mass $a_i$, that's $2\big(x_i - T(x_i)\big)$ with $T$ the barycentric map of the entropic plan. gradient descent moves each point to a blurred *average* of its targets, which lies inside the target's convex hull: a ring gets fitted by a smaller ring.

the **Sinkhorn divergence** subtracts the two self-transport costs:

$$S_\varepsilon(\mu, \nu) = \mathrm{OT}_\varepsilon(\mu, \nu) - \tfrac12\mathrm{OT}_\varepsilon(\mu, \mu) - \tfrac12\mathrm{OT}_\varepsilon(\nu, \nu).$$

Feydy et al. (2019) proved that with the KL-to-$a\otimes b$ convention, $S_\varepsilon$ is nonnegative, zero only when $\mu = \nu$, convex in each argument and metrizes weak convergence, for costs like $\|x - y\|^p$ with $p = 1, 2$. it **interpolates** between optimal transport and kernel methods: as $\varepsilon \to 0$ it tends to $W_p^p$, and as $\varepsilon \to \infty$ it tends to half an MMD with kernel $-\|x - y\|^p$ (for the squared cost, simply $\|\bar x - \bar y\|^2$, the squared distance between the means; on paper 4). for the squared cost it's also exact on translations: $S_\varepsilon(\mu, \mu + t) = \|t\|^2$ at every $\varepsilon$ (on paper 3), while $\mathrm{OT}_\varepsilon(\mu, \mu + t) = \|t\|^2 + \mathrm{OT}_\varepsilon(\mu, \mu)$.

> **spec.** `sinkhorn_divergence(X, Y, eps, *, a=None, b=None, p=2, max_iters=1000, tol=1e-9)`: for points `X` (n, d) and `Y` (m, d) (1-D arrays are points on a line) with weights `a`, `b` (uniform if None) and cost $\|x - y\|^p$, return $S_\varepsilon$ as a float, where each $\mathrm{OT}_\varepsilon$ is the dual value at the potentials your `sinkhorn_log` returns (called with the given `max_iters` and `tol`).

> **🔮 predict:** at large $\varepsilon$, which is bigger: the biased $\mathrm{OT}_\varepsilon(\mu, \nu)$ or the true $W_2^2(\mu, \nu)$? and $S_\varepsilon(\mu, \nu)$?

> **▶ build: stage 4.** implement `sinkhorn_divergence`.

## worked example

two points on each side, $a = b = (\tfrac12, \tfrac12)$, $C = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}$ (staying is free, swapping costs 1), $\varepsilon = 1$.

**Sinkhorn, by hand.** $K = \begin{pmatrix}1 & e^{-1}\\ e^{-1} & 1\end{pmatrix}$; from $v = \mathbf 1$, $Kv = (1 + e^{-1})\mathbf 1 = 1.3679\cdot\mathbf 1$, so $u = 0.3655\cdot\mathbf 1$; then $K^\top u = 0.5\cdot\mathbf 1$ and $v = \mathbf 1$ again. by symmetry it has already converged:

$$P = \begin{pmatrix}p & \tfrac12 - p\\ \tfrac12 - p & p\end{pmatrix},\qquad p = \frac{1}{2(1 + e^{-1/\varepsilon})} = \tfrac12\sigma(1/\varepsilon) = 0.36553 .$$

the cross-ratio check: $p^2/(\tfrac12 - p)^2 = e^{2/\varepsilon}$, as idea 1 says. at $\varepsilon = 0.1$, $p = 0.49998$: almost the exact plan (the identity). at $\varepsilon = 1$ a quarter of the mass, $1 - 2p = 0.269$, still takes the costly swap.

**values.** the transport part is $\langle C, P\rangle = 0.26894$ and $\mathrm{KL}(P\,\|\,ab^\top) = \sum_{ij}P_{ij}\log(4P_{ij}) = 0.11094$, so the primal is $0.37989$. by symmetry the potentials are equal constants, $f = g = c$ with $\tfrac14e^{2c} = p$, so $c = \tfrac12\log(4p) = 0.18994$, and the dual value is $\langle f, a\rangle + \langle g, b\rangle = 2c = 0.37989$: the same number. the exact $W$ is 0 (nobody needs to move), so here $\mathrm{OT}_1 = 0.38$ is pure bias, below the bound $\varepsilon(H(a) + H(b)) = 2\log 2 = 1.39$. and since $\mu = \nu$, the Sinkhorn divergence is exactly 0.

your `sinkhorn` (and the stage 2 checks) must reproduce $p = 0.36553$.

## on paper

1. **the scaling form and the dual.** form the Lagrangian of the entropic problem with multipliers $f$, $g$, minimize over $P > 0$, and derive both $P_{ij} = a_ib_je^{(f_i + g_j - C_{ij})/\varepsilon}$ and the dual objective.

<details><summary>solution</summary>

$\mathcal L = \sum_{ij}\big[C_{ij}P_{ij} + \varepsilon(P_{ij}\log\tfrac{P_{ij}}{a_ib_j} - P_{ij} + a_ib_j)\big] + \sum_if_i(a_i - \sum_jP_{ij}) + \sum_jg_j(b_j - \sum_iP_{ij})$. setting $\partial\mathcal L/\partial P_{ij} = C_{ij} + \varepsilon\log\tfrac{P_{ij}}{a_ib_j} - f_i - g_j = 0$ gives $P_{ij} = a_ib_je^{(f_i + g_j - C_{ij})/\varepsilon}$. substituting back, $\varepsilon P\log\tfrac{P}{ab} = P(f_i + g_j - C_{ij})$, so the $C$ terms and the multiplier terms cancel except $\langle f, a\rangle + \langle g, b\rangle$, leaving $\langle f, a\rangle + \langle g, b\rangle - \varepsilon\sum_{ij}(P_{ij} - a_ib_j)$, which is the dual with $P_{ij}$ written out.

</details>

2. **the entropic bias bound.** let $P^\star$ be an optimal LP plan and $P_\varepsilon$ the entropic one. show $0 \le \langle C, P_\varepsilon\rangle - W \le \varepsilon(H(a) + H(b))$, with $H(a) = -\sum_ia_i\log a_i$.

<details><summary>solution</summary>

the lower bound: $P_\varepsilon \in U(a, b)$, and $W$ is the minimum of $\langle C, P\rangle$ over $U(a, b)$. the upper bound: $P_\varepsilon$ minimizes the regularized objective, so $\langle C, P_\varepsilon\rangle + \varepsilon\mathrm{KL}(P_\varepsilon\|ab^\top) \le W + \varepsilon\mathrm{KL}(P^\star\|ab^\top)$. KL is nonnegative, and on $U(a, b)$, $\mathrm{KL}(P\|ab^\top) = H(a) + H(b) - H(P)$ with $H(P) \ge 0$ for a discrete distribution. so $\langle C, P_\varepsilon\rangle - W \le \varepsilon\mathrm{KL}(P^\star\|ab^\top) \le \varepsilon(H(a) + H(b)) \le \varepsilon\log(nm)$.

</details>

3. **translations.** with the squared cost, let $\nu$ be $\mu$ translated by $t$ (same weights). show $\mathrm{OT}_\varepsilon(\mu, \nu) = \mathrm{OT}_\varepsilon(\mu, \mu) + \|t\|^2$ and hence $S_\varepsilon(\mu, \nu) = \|t\|^2$.

<details><summary>solution</summary>

$\|x_i - x_j - t\|^2 = \|x_i - x_j\|^2 - 2\langle x_i, t\rangle + 2\langle x_j, t\rangle + \|t\|^2$. the extra terms depend on $i$ alone or $j$ alone, so on $U(a, a)$ they add the constant $-2\langle\bar x, t\rangle + 2\langle\bar x, t\rangle + \|t\|^2 = \|t\|^2$ to every plan's cost (with $\bar x = \sum_ia_ix_i$): the same plan is optimal and the value shifts by $\|t\|^2$. translation doesn't change $\mathrm{OT}_\varepsilon(\nu, \nu) = \mathrm{OT}_\varepsilon(\mu, \mu)$, so the self-terms cancel and $S_\varepsilon = \|t\|^2$.

</details>

4. **the large-$\varepsilon$ limit.** show that as $\varepsilon \to \infty$, $S_\varepsilon(\mu, \nu) \to \|\bar x - \bar y\|^2$ for the squared cost.

<details><summary>solution</summary>

as $\varepsilon \to \infty$ the KL term dominates and $P_\varepsilon \to ab^\top$ (the only plan with zero KL), with $\varepsilon\mathrm{KL} \to 0$, so $\mathrm{OT}_\varepsilon(\mu, \nu) \to \sum_{ij}a_ib_j\|x_i - y_j\|^2 = \mathbb E\|X - Y\|^2$ for independent $X \sim \mu$, $Y \sim \nu$. that's $\operatorname{tr}\mathrm{Cov}(X) + \operatorname{tr}\mathrm{Cov}(Y) + \|\bar x - \bar y\|^2$, while the self-terms tend to $2\operatorname{tr}\mathrm{Cov}(X)$ and $2\operatorname{tr}\mathrm{Cov}(Y)$. half of each cancels the covariances: $S_\infty = \|\bar x - \bar y\|^2$. only the means are compared, which is the MMD with the (degenerate) kernel $-\|x - y\|^2$.

</details>

## experiment

experiment 1 sweeps $\varepsilon$ over three orders of magnitude on a small problem and measures iterations, time, how blurred the plan is, the bias against your exact LP, and when the naive version breaks. experiment 2 races Sinkhorn against your topic 15 LP as $n$ grows, on speed and accuracy. experiment 3 fits a point cloud to a ring by gradient descent on the biased cost and on the debiased divergence.

## check your understanding

1. why does a larger $\varepsilon$ make Sinkhorn converge faster?

<details><summary>answer</summary>

a large $\varepsilon$ makes $K$ close to a constant matrix: every row "sees" every column, so one rescaling spreads information across the whole problem and the iteration is a strong contraction. with small $\varepsilon$, $K$ is nearly sparse (only the cheapest routes survive), each update changes the potentials only locally, and corrections propagate slowly through chains of near-optimal routes.

</details>

2. the naive and log-domain versions perform the same iterations. why does only one fail?

<details><summary>answer</summary>

they compute the same numbers in exact arithmetic, but the naive one stores $e^{-C/\varepsilon}$ and the scalings $u, v$, which span hundreds of orders of magnitude at small $\varepsilon$ and leave float64's range. the log-domain version stores $f, g$ and $C$, which are of the size of the costs, and only exponentiates differences shifted by their maximum.

</details>

3. Sinkhorn stops when the row marginals are within `tol`. why check the rows and not the columns?

<details><summary>answer</summary>

the last update in each iteration rescales the columns, so after it the column sums are exact by construction. the row sums are what the last update broke, so they're the only meaningful measure of how far from the fixed point you are.

</details>

4. a colleague fits a generative model by minimizing $\mathrm{OT}_\varepsilon(\text{model}, \text{data})$ with a large $\varepsilon$ for speed. what will the samples look like?

<details><summary>answer</summary>

shrunk towards the data's mean: each sample is pulled to a blurred average of the data points it's coupled to, which lies inside the data's convex hull, so the model underestimates the spread (and fine structure). the debiased $S_\varepsilon$ removes this pull because its self-term pushes the samples apart by the same blur; that's what experiment 3 shows.

</details>

5. why is $\langle f, a\rangle + \langle g, b\rangle$ a better number to report than the transport part $\langle C, P_\varepsilon\rangle$ when you want $\mathrm{OT}_\varepsilon$, and when is the transport part the better estimate of $W$?

<details><summary>answer</summary>

$\langle f, a\rangle + \langle g, b\rangle$ is the value of the regularized problem (at convergence), the quantity whose gradient the envelope theorem gives and the one that debiasing is defined for. the transport part is the LP cost of a feasible plan, so it's an *upper bound* on $W$, within $\varepsilon(H(a) + H(b))$; for a single estimate of $W$ at small $\varepsilon$ it's the natural choice, but the debiasing guarantees of idea 4 are proved for the entropic value, not for it.

</details>

## common bugs

- **the kernel's sign or scale**: `exp(C / eps)` or `exp(-C)`; the plan then has the wrong cross-ratios, and the checks say so.
- **forgetting the weights in the log domain**: dropping $\log b_j$ inside the $f$-update gives the plan for uniform weights; the marginals come out wrong for any non-uniform $a$ or $b$.
- **checking convergence on the columns**: they're always exact after a $v$- or $g$-update, so the loop stops after one iteration.
- **forming `exp(-C / eps)` anywhere** in the log-domain version, even "just to build $P$ at the end": compute $P$ from $f + g - C$ directly.
- **the transport part instead of the dual value** in the divergence: $\langle C, P_\varepsilon\rangle$ is a different quantity (sometimes called the "sharp" Sinkhorn cost), and Feydy et al.'s guarantees (positivity, convexity) are proved for the full entropic value.
- **forgetting the halves** in $S_\varepsilon$: subtracting both self-terms in full makes the divergence negative.

## going deeper

- Cuturi (2013), "Sinkhorn distances: lightspeed computation of optimal transport", *NeurIPS*. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)
- Peyré & Cuturi (2019), *Computational Optimal Transport*, chapter 4 (entropic regularization, Sinkhorn, log-domain stabilization, convergence). [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
- Sinkhorn & Knopp (1967), "Concerning nonnegative matrices and doubly stochastic matrices", *Pacific Journal of Mathematics* 21(2). [doi:10.2140/pjm.1967.21.343](https://doi.org/10.2140/pjm.1967.21.343)
- Feydy, Séjourné, Vialard, Amari, Trouvé & Peyré (2019), "Interpolating between optimal transport and MMD using Sinkhorn divergences", *AISTATS*. [arXiv:1810.08278](https://arxiv.org/abs/1810.08278)
- Genevay, Peyré & Cuturi (2018), "Learning generative models with Sinkhorn divergences", *AISTATS*. [arXiv:1706.00292](https://arxiv.org/abs/1706.00292)
- Altschuler, Weed & Rigollet (2017), "Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration", *NeurIPS*. [arXiv:1705.09634](https://arxiv.org/abs/1705.09634)
- Schmitzer (2019), "Stabilized sparse scaling algorithms for entropy regularized transport problems", *SIAM Journal on Scientific Computing* 41(3). log-domain absorption and $\varepsilon$-scaling, which speed up small-$\varepsilon$ Sinkhorn. [arXiv:1610.06519](https://arxiv.org/abs/1610.06519)

## connections

- ← builds on: the Kantorovich LP, potentials and barycentric maps (topic 15); KL and entropy (ch 05); `logsumexp` (ch 00); coordinate ascent and duality (ch 06).
- → used in: topic 17 samples minibatch couplings from your `sinkhorn_log` plans and turns the debiased divergence into a differentiable loss. after this topic you may use library Sinkhorn solvers (POT's `ot.sinkhorn`, GeomLoss's `SamplesLoss`, OTT-JAX), since you've built the algorithm; mind their conventions (Cuturi's $-\varepsilon H(P)$ penalty, costs scaled by $\tfrac12$, normalized $\varepsilon$).
