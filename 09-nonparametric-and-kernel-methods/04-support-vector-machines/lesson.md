# support vector machines

## why this matters

when two classes can be separated by a line, infinitely many lines do it. a perceptron (chapter 08) stops at the first one it finds, which may pass a hair's breadth from a training point, so the slightest noise flips the prediction. the **support vector machine** picks the separating line with the widest empty band around it: the **maximum margin**. intuitively, a wide margin is robust; formally, margin bounds generalization in a way that doesn't depend on the dimension of the features (Vapnik, 1995; chapter 13).

two more ideas make the SVM one of the most influential models of the 1990s and 2000s. a **soft margin** with the hinge loss lets it tolerate overlapping classes, with one knob, $C$, trading margin width against training errors. and its **dual** problem touches the data only through inner products, so the kernel trick of topic 03 turns it into a nonlinear classifier for free. the solution depends on only a subset of the training points, the **support vectors**: delete every other point and you get exactly the same classifier.

the dual is a quadratic program with $n$ variables. this topic's main build is the algorithm that made SVMs practical on large data: **sequential minimal optimization** (Platt, 1998), which solves the QP by repeatedly optimizing just two variables at a time, in closed form.

## warm-up: predict first

1. two training points, $(0, 0)$ in class $-1$ and $(2, 0)$ in class $+1$. what's the maximum-margin boundary, and how wide is the margin?
2. you remove a training point that is far from the boundary and refit. does the SVM change?
3. what happens to the boundary as $C \to \infty$ on overlapping classes? as $C \to 0$?
4. why does SMO update *two* variables at a time, never one?
5. an RBF SVM with a huge $\gamma$: how many training points will be support vectors?

## prerequisites check

- [ ] you can write a Lagrangian for an equality- and inequality-constrained problem and read off KKT conditions (→ ch 06 · topic 06)
- [ ] you know convexity, and why a convex QP has no bad local optima (→ ch 06 · topic 01)
- [ ] you can compute Gram matrices and use `get_kernel` (→ topic 03)
- [ ] you've written a `Classifier` with `classes_` and a `decision_function` (→ ch 07 · topics 01 and 03)
- [ ] you know the perceptron, or at least a linear classifier $\mathrm{sign}(w^\top x + b)$ (→ ch 08 · topic 03)

## core ideas

### idea 1: the margin, the slack, and the hinge

a linear classifier predicts $\mathrm{sign}(f(x))$ with $f(x) = w^\top x + b$, for labels $y \in \{-1, +1\}$. the distance from a point $x_i$ to the boundary $f = 0$ is $|f(x_i)|/\|w\|$, and the point is correctly classified when $y_i f(x_i) > 0$. $w$ and $b$ can be rescaled together without moving the boundary, so fix the scale by asking the closest points to have $y_i f(x_i) = 1$. then the margin, the distance from the boundary to the closest points, is $1/\|w\|$, and maximizing it means minimizing $\|w\|$:

$$\min_{w, b}\ \tfrac12\|w\|^2 \quad \text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 \ \text{ for all } i.$$

that's the **hard-margin** SVM. it has no solution when the classes overlap. the **soft margin** gives each point a slack $\xi_i \ge 0$ for how far it falls short of the margin, and charges $C$ per unit of slack:

$$\min_{w, b, \xi}\ \tfrac12\|w\|^2 + C\sum_i \xi_i \quad \text{s.t.}\quad y_i f(x_i) \ge 1 - \xi_i,\ \ \xi_i \ge 0.$$

at the optimum each slack is as small as allowed, $\xi_i = \max(0, 1 - y_i f(x_i))$, so the problem is unconstrained regularized loss minimization,

$$\min_{w, b}\ \tfrac12\|w\|^2 + C\sum_i \underbrace{\max\big(0,\ 1 - y_i f(x_i)\big)}_{\text{hinge loss}}.$$

the **hinge loss** is zero for points beyond the margin on the correct side ($y f \ge 1$), and grows linearly as a point moves into the margin or across the boundary. it's a convex upper bound on the 0–1 loss. compared with logistic regression's loss $\log(1 + e^{-yf})$, it's exactly zero for confidently correct points, which is where the sparsity of the solution comes from. $C$ is the inverse of a regularization strength: large $C$ punishes every margin violation (narrow margin, low bias, high variance; $C \to \infty$ recovers the hard margin when it exists), small $C$ prefers a wide margin and tolerates errors (warm-up 3).

> **▶ build: stage 1.** implement `hinge_loss(y, scores)` (elementwise, labels in $\{-1, +1\}$) and `primal_objective(w, b, X, y, C)`.

### idea 2: the dual, the KKT conditions, and support vectors

introduce multipliers $\alpha_i \ge 0$ for the margin constraints and $\mu_i \ge 0$ for $\xi_i \ge 0$, and set the Lagrangian's derivatives to zero (on paper 1). you get $w = \sum_i \alpha_i y_i x_i$, the constraint $\sum_i \alpha_i y_i = 0$, and $0 \le \alpha_i \le C$. substituting back gives the **dual**:

$$\max_{\alpha}\ \sum_i \alpha_i - \frac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\, x_i^\top x_j \quad \text{s.t.}\quad 0 \le \alpha_i \le C,\ \ \sum_i \alpha_i y_i = 0.$$

two things to notice. the data appear only through inner products $x_i^\top x_j$, so replace them with any kernel $K_{ij} = k(x_i, x_j)$: the classifier becomes $f(x) = \sum_i \alpha_i y_i\, k(x_i, x) + b$, linear in the kernel's feature space and nonlinear in $x$ (it's the representer theorem of topic 03 in action). and the dual objective is concave when $K$ is PSD, so it has no bad local maxima, and at the optimum it equals the primal objective (strong duality: the gap is zero).

the **KKT conditions** say how each $\alpha_i$ relates to its point's margin $m_i = y_i f(x_i)$:

| $\alpha_i$ | margin | the point is |
|---|---|---|
| $0$ | $m_i \ge 1$ | beyond the margin: it doesn't affect the solution |
| $0 < \alpha_i < C$ | $m_i = 1$ | exactly on the margin: a *free* support vector |
| $C$ | $m_i \le 1$ | inside the margin or misclassified: a *bound* support vector |

the points with $\alpha_i > 0$ are the **support vectors**, and only they appear in $w$ and $f$. points with $\alpha_i = 0$ could be deleted without changing anything (warm-up 2). the free support vectors also give $b$: for any of them, $y_i f(x_i) = 1$ pins it down.

these conditions are also the stopping test for a solver. given the current $\alpha$ and $f$, an example **violates KKT** if $m_i < 1$ while $\alpha_i < C$ (it should be pushed up) or $m_i > 1$ while $\alpha_i > 0$ (it should be released), by more than a tolerance.

> **▶ build: stage 2.** implement `dual_objective(alpha, K, y)` and `kkt_violations(alpha, y, f, C, tol=1e-3)`: with $r_i = y_i f_i - 1$, a boolean mask of $(r_i < -\text{tol} \text{ and } \alpha_i < C)$ or $(r_i > \text{tol} \text{ and } \alpha_i > 0)$.

> **🔮 predict:** on two well-separated blobs of 100 points each, what fraction of the points will be support vectors with a linear kernel? and with an RBF kernel whose $\gamma$ is enormous?

### idea 3: SMO: two variables at a time

the dual has $n$ variables, a box $[0, C]^n$ and one equality constraint. general QP solvers need the $n \times n$ matrix in memory and $O(n^3)$ time. SMO instead picks two variables $\alpha_i, \alpha_j$, holds all others fixed, and solves the two-variable problem **analytically**. why two (warm-up 4)? because $\sum_k \alpha_k y_k = 0$: changing one $\alpha$ alone would break it. with two, the constraint says $y_i\alpha_i + y_j\alpha_j$ stays constant, so the pair moves along a line, and the box clips that line to a segment $[L, H]$ for $\alpha_j$:

$$y_i \ne y_j:\ L = \max(0, \alpha_j - \alpha_i),\ H = \min(C, C + \alpha_j - \alpha_i); \qquad y_i = y_j:\ L = \max(0, \alpha_i + \alpha_j - C),\ H = \min(C, \alpha_i + \alpha_j).$$

along the line the dual is a 1-D quadratic in $\alpha_j$ with second derivative $\eta = 2K_{ij} - K_{ii} - K_{jj} = -\|\varphi(x_i) - \varphi(x_j)\|^2 \le 0$ (on paper 2). with the errors $E_k = f(x_k) - y_k$, its unconstrained maximum and the clipped update are

$$\alpha_j^{\text{new}} = \mathrm{clip}\!\left(\alpha_j - \frac{y_j(E_i - E_j)}{\eta},\ L,\ H\right), \qquad \alpha_i^{\text{new}} = \alpha_i + y_i y_j(\alpha_j - \alpha_j^{\text{new}}).$$

if $L = H$ there's nowhere to move; if $\eta \ge 0$ (two identical points, $\eta = 0$) the simplified algorithm skips the pair; if $\alpha_j$ barely changes, it reports no progress. round-off can leave $\alpha_i^{\text{new}}$ at $10^{-17}$ when it should be exactly 0 (or $C - 10^{-17}$): snap it to the bound, or the KKT test will keep flagging the point forever.

then the bias. choose $b$ so that the KKT condition holds for a point that's free after the step: with $\Delta_i = \alpha_i^{\text{new}} - \alpha_i$ and $\Delta_j$ likewise,

$$b_1 = b - E_i - y_i\Delta_i K_{ii} - y_j\Delta_j K_{ij}, \qquad b_2 = b - E_j - y_i\Delta_i K_{ij} - y_j\Delta_j K_{jj},$$

and $b^{\text{new}} = b_1$ if $0 < \alpha_i^{\text{new}} < C$, else $b_2$ if $0 < \alpha_j^{\text{new}} < C$, else their average. ($b_1$ makes $f(x_i) = y_i$ exactly, which is what a free support vector needs.)

> **▶ build: stage 3.** implement `smo_pair_update(i, j, alpha, b, K, y, C, E)` as a pure function: it returns `(new_alpha, new_b)` with `new_alpha` a *copy*, or `None` when `i == j`, $L \ge H$, $\eta \ge 0$, or $|\alpha_j^{\text{new}} - \alpha_j| < 10^{-5}(\alpha_j^{\text{new}} + \alpha_j + 10^{-5})$. snap $\alpha_i^{\text{new}}$ to 0 or $C$ when it's within $10^{-12}C$ of the bound.

### idea 4: the whole algorithm

SMO repeats pair updates until nothing violates KKT:

1. build $K$ once (with `get_kernel`), set $\alpha = 0$, $b = 0$, and the error cache $E = f - y = -y$.
2. **sweep** over $i = 1..n$. skip $i$ unless it violates KKT (by `tol`). pick a partner $j$: Platt's heuristic is the $j$ that maximizes $|E_i - E_j|$, because the step size is proportional to $E_i - E_j$. if that pair makes no progress, try one random $j \ne i$ (drawn from `as_rng(rng)`, so runs are reproducible).
3. after a successful update, refresh the whole cache in $O(n)$: $E \mathrel{+}= \Delta_i y_i K_{:,i} + \Delta_j y_j K_{:,j} + \Delta b$.
4. count consecutive sweeps that changed nothing; stop after `max_passes` of them, or after `max_iters` sweeps in total, **whichever comes first**. the cap matters: on a degenerate problem (duplicate points with opposite labels, a non-PSD "kernel") the loop might never settle, and a solver must stop anyway.

the $b$ of the last pair update reflects only that pair. so once the loop stops, recompute the bias from all the KKT conditions with the given `final_bias`: the average of $y_i - \sum_j\alpha_j y_j K_{ij}$ over the free support vectors (each of them should satisfy $y_i f(x_i) = 1$ exactly), or, if none is free, the midpoint of the range of $b$ that the bounded points allow. at the end, the support vectors are the points with $\alpha_i > 10^{-8}$, `dual_coef_` holds $\alpha_i y_i$ for them, and `decision_function` evaluates $\sum_{i \in SV}\alpha_i y_i k(x_i, x) + b$ against the support vectors only, which is what makes prediction cheap when the solution is sparse. labels can be any two values: `classes_[1]` plays $+1$.

each sweep costs $O(n)$ per update for the cache, and the number of sweeps depends on the data and $C$ (large $C$ is slower). the full Gram matrix costs $O(n^2)$ memory, fine for a few thousand points; libsvm caches only the rows it needs, which takes SMO to hundreds of thousands. this "simplified" SMO has no convergence guarantee, but it stops near the optimum in practice, and the checks compare your dual objective with libsvm's.

an SVM doesn't output probabilities. its margin $f(x)$ ranks points well but isn't calibrated; `predict_proba` raises, and if you need probabilities you calibrate the margin on held-out data (Platt scaling, chapter 07's calibration ideas).

> **▶ build: stage 4.** implement `SVC.fit` (the loop above; set `classes_`, `alpha_`, `support_`, `support_vectors_`, `dual_coef_`, `intercept_` and `n_iter_`, the number of sweeps; more or fewer than two classes, or $C \le 0$, raise `ValueError`), `decision_function` and `predict`.

> **🔮 predict:** you fit an SVM, delete every training point that isn't a support vector, and refit. what changes?

## worked example

two points: $x_1 = (0, 0)$ with $y_1 = -1$ and $x_2 = (2, 0)$ with $y_2 = +1$, linear kernel, $C = 10$. the Gram matrix is $K = \begin{pmatrix}0 & 0\\ 0 & 4\end{pmatrix}$.

start from $\alpha = (0, 0)$, $b = 0$: then $f = 0$ and $E = f - y = (1, -1)$. take the pair $(i, j) = (1, 2)$:

- labels differ, so $L = \max(0, 0 - 0) = 0$ and $H = \min(10, 10 + 0) = 10$.
- $\eta = 2K_{12} - K_{11} - K_{22} = 0 - 0 - 4 = -4$.
- $\alpha_2^{\text{new}} = 0 - \frac{(+1)(1 - (-1))}{-4} = 0.5$, inside $[0, 10]$, and $\alpha_1^{\text{new}} = 0 + (-1)(+1)(0 - 0.5) = 0.5$.
- $b_1 = 0 - 1 - (-1)(0.5)(0) - (1)(0.5)(0) = -1$, and $b_2 = 0 + 1 - (-1)(0.5)(0) - (1)(0.5)(4) = -1$. both $\alpha$'s are free, so $b = -1$.

the classifier is $w = \sum_i\alpha_i y_i x_i = 0.5 \cdot (-1)(0, 0) + 0.5 \cdot (+1)(2, 0) = (1, 0)$ with $b = -1$: $f(x) = x_1 - 1$, the boundary $x_1 = 1$ halfway between the points, and margin $1/\|w\| = 1$ on each side (warm-up 1). the dual objective is $\sum\alpha - \frac12\|w\|^2 = 1 - 0.5 = 0.5$, and the primal is $\frac12\|w\|^2 + C \cdot 0 = 0.5$: zero duality gap, so one step solved it. with $C = 0.2$ instead, the step is clipped at $H = 0.2$, both $\alpha$'s sit at the bound, and $b = \frac{b_1 + b_2}{2} = \frac{-1 + 0.2}{2} = -0.4$. the tests check both.

**a step where only one variable ends free.** three points, linear kernel: $x_1 = (1, 0)$ with $y_1 = -1$, $x_2 = (2, 1)$ and $x_3 = (0, 2)$ with $y_2 = y_3 = +1$, so $K = \begin{pmatrix}1 & 2 & 0\\ 2 & 5 & 2\\ 0 & 2 & 4\end{pmatrix}$. start mid-run from $\alpha = (0.2, 0.1, 0.1)$ (it satisfies $\sum\alpha_i y_i = 0$), $b = 0$ and $C = 0.5$: then $f = K(\alpha \odot y) = (0, 0.3, 0.6)$ and $E = f - y = (1, -0.7, -0.4)$. update the pair $(1, 2)$:

- labels differ: $L = \max(0, 0.1 - 0.2) = 0$, $H = \min(0.5, 0.5 + 0.1 - 0.2) = 0.4$; $\eta = 2 \cdot 2 - 1 - 5 = -2$.
- $\alpha_2 - y_2(E_1 - E_2)/\eta = 0.1 + 1.7/2 = 0.95$, clipped to $H = 0.4$; then $\alpha_1 = 0.2 + (-1)(+1)(0.1 - 0.4) = 0.5 = C$.
- with $\Delta_1 = \Delta_2 = 0.3$: $b_1 = 0 - 1 - (-1)(0.3)(1) - (1)(0.3)(2) = -1.3$ and $b_2 = 0 + 0.7 - (-1)(0.3)(2) - (1)(0.3)(5) = -0.2$.
- $\alpha_1$ is at the bound and $\alpha_2 = 0.4$ is free, so $b = b_2 = -0.2$: it's the free variable's condition, $f(x_2) = y_2$, that pins $b$ down. (averaging would give $-0.75$, the wrong one $-1.3$.) the tests check this step too.

## on paper

1. **deriving the dual.** write the Lagrangian of the soft-margin primal with multipliers $\alpha_i \ge 0$ (margin constraints) and $\mu_i \ge 0$ ($\xi_i \ge 0$). set its gradients in $w$, $b$, $\xi$ to zero and derive the dual problem and the box $0 \le \alpha_i \le C$.

<details><summary>solution</summary>

$\mathcal L = \frac12\|w\|^2 + C\sum\xi_i - \sum\alpha_i\big(y_i(w^\top x_i + b) - 1 + \xi_i\big) - \sum\mu_i\xi_i$. $\partial_w$: $w = \sum\alpha_i y_i x_i$. $\partial_b$: $\sum\alpha_i y_i = 0$. $\partial_{\xi_i}$: $C - \alpha_i - \mu_i = 0$, so $\alpha_i = C - \mu_i \le C$. substituting, the $\xi$ terms cancel and $b$ drops out: $\mathcal L = \frac12\|w\|^2 - \sum_{ij}\alpha_i\alpha_j y_i y_j x_i^\top x_j + \sum\alpha_i = \sum\alpha_i - \frac12\sum_{ij}\alpha_i\alpha_j y_i y_j x_i^\top x_j$, to be maximized over $0 \le \alpha_i \le C$ with $\sum\alpha_i y_i = 0$.

</details>

2. **the SMO step.** with $s = y_i y_j$ and $\alpha_i = \gamma - s\alpha_j$ (the constraint), write the dual as a function of $\alpha_j$ alone and show that its second derivative is $\eta = 2K_{ij} - K_{ii} - K_{jj}$ and its maximizer is $\alpha_j - y_j(E_i - E_j)/\eta$.

<details><summary>solution</summary>

the part of the dual that depends on the pair is $\alpha_i + \alpha_j - \frac12 K_{ii}\alpha_i^2 - \frac12 K_{jj}\alpha_j^2 - sK_{ij}\alpha_i\alpha_j - y_i\alpha_i v_i - y_j\alpha_j v_j$, with $v_k = \sum_{l \ne i,j}\alpha_l y_l K_{kl}$. substituting $\alpha_i = \gamma - s\alpha_j$ and differentiating twice gives $-K_{ii} - K_{jj} + 2K_{ij} = \eta$. setting the first derivative to zero and expressing $v_k$ through $f(x_k) = v_k + \alpha_i y_i K_{ki} + \alpha_j y_j K_{kj} + b$ gives $\eta(\alpha_j^{\text{new}} - \alpha_j) = y_j(E_i - E_j)$, i.e. $\alpha_j^{\text{new}} = \alpha_j - y_j(E_i - E_j)/\eta$ (dividing by $\eta < 0$). since $\eta = -\|\varphi(x_i) - \varphi(x_j)\|^2$, it's negative unless the two points coincide in feature space.

</details>

3. **sparsity and leave-one-out.** show that removing a point with $\alpha_i = 0$ leaves the solution unchanged. then argue that the leave-one-out error of an SVM is at most (number of support vectors)/$n$.

<details><summary>solution</summary>

the optimal $\alpha$ (with $\alpha_i = 0$) is feasible for the problem without point $i$, and it still satisfies every KKT condition there, so it's still optimal: same $w$, same $b$. for leave-one-out: if a non-support vector is left out, the classifier doesn't change, and it classified that point correctly (its margin was $\ge 1$). so only support vectors can be leave-one-out errors, and the error is at most $\#SV/n$. a sparse solution is evidence of good generalization.

</details>

4. **the hinge and the margin.** a point has margin $m = y f(x) = 0.3$. what's its hinge loss, and which case of the KKT table can it be in?

<details><summary>solution</summary>

$\max(0, 1 - 0.3) = 0.7$. it's inside the margin (correctly classified, but with $m < 1$), so at the optimum $\alpha_i = C$: a bound support vector, with slack $\xi_i = 0.7$.

</details>

## experiment

`lab.py` trains your `SVC`:

1. a grid of $C$ and $\gamma$ on two noisy moons: the boundaries, the support vectors, and train and test accuracy.
2. delete the non-support vectors and refit; then delete one support vector.
3. a real task, telling 4s from 9s in MNIST: test error against the support-vector bound of on paper 3.

## check your understanding

1. why does the SVM's solution depend only on the support vectors, while logistic regression's depends on every point?

<details><summary>answer</summary>

the hinge loss is exactly zero (with zero gradient) for points beyond the margin, so they exert no force on the solution: $\alpha_i = 0$. the logistic loss is positive for every point, however confidently it's classified, so every point pulls a little.

</details>

2. what does $\gamma$ do to an RBF SVM, and how does it interact with $C$?

<details><summary>answer</summary>

$\gamma$ sets the kernel's reach. large $\gamma$: each support vector influences only its immediate neighbourhood, the boundary can wrap around single points, and almost every point becomes a support vector (overfitting). small $\gamma$: the boundary is nearly linear. $C$ limits how hard the fit tries to classify every training point: large $C$ with large $\gamma$ memorizes; small $C$ with small $\gamma$ underfits. they're tuned together, on a log grid, by cross-validation.

</details>

3. your SMO loop runs forever on a dataset with two identical points carrying opposite labels. why, and what saves you?

<details><summary>answer</summary>

for that pair $\eta = 0$, so the pair update is skipped, yet both points violate KKT (no $f$ can give both margin $\ge 1$), so they're selected on every sweep. if other pairs keep making tiny changes, the "sweeps without change" counter may never reach `max_passes`. the `max_iters` cap on the total number of sweeps guarantees termination.

</details>

4. you scale every feature by 10. what happens to an RBF SVM with fixed $\gamma$?

<details><summary>answer</summary>

squared distances grow by 100, so it behaves like the original data with $\gamma$ multiplied by 100: kernel values collapse towards 0 for all but the nearest pairs, and the model overfits. SVMs need standardized features, like every distance-based method.

</details>

5. why can't you read class probabilities off `decision_function`?

<details><summary>answer</summary>

the margin's scale is set by the optimization (closest points at $|f| = 1$), not by how often points at a given margin are positive. points far beyond the margin all have zero loss, so nothing in training calibrates large $|f|$. you can fit a monotone map from $f$ to probabilities on held-out data (Platt scaling or isotonic regression).

</details>

6. how does the hinge loss compare with the logistic loss for a point with margin $m = 5$? with $m = -5$?

<details><summary>answer</summary>

$m = 5$: hinge 0, logistic $\log(1 + e^{-5}) \approx 0.0067$. $m = -5$: hinge 6, logistic $\approx 5.0067$. both grow linearly for badly misclassified points (so both are fairly robust to outliers, unlike the squared loss), and they differ mainly for confident correct points.

</details>

## common bugs

- **labels in $\{0, 1\}$.** every formula assumes $y \in \{-1, +1\}$. symptom: the dual constraint $\sum\alpha_i y_i = 0$ forces all class-0 alphas to be irrelevant, and the boundary is nonsense.
- **a stale error cache.** forgetting the $\Delta b$ term, or updating $E$ with the old alphas. symptom: SMO "converges" to a dual objective noticeably below the optimum, and KKT checks fail at the end.
- **mutating `alpha` inside the pair update.** symptom: `None` returns still leave partial changes behind, and the dual constraint drifts.
- **round-off at the bounds.** $\alpha = 10^{-17}$ counts as "$> 0$" in the KKT test. symptom: the solver never finishes its passes, or the final KKT check flags many points beyond the margin. snap to the bounds.
- **no iteration cap.** symptom: a frozen notebook on degenerate data. always cap the number of sweeps.
- **evaluating $f$ with all training points.** correct, but wasteful: use only the support vectors.

## going deeper

- Cortes & Vapnik (1995), "Support-vector networks", *Machine Learning* 20(3): 273–297. the soft-margin SVM. [doi:10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
- Boser, Guyon & Vapnik (1992), "A training algorithm for optimal margin classifiers", *COLT*. the kernelized maximum-margin classifier. [doi:10.1145/130385.130401](https://doi.org/10.1145/130385.130401)
- Platt (1998), "Sequential minimal optimization: a fast algorithm for training support vector machines", Microsoft Research technical report MSR-TR-98-14. [microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
- Keerthi, Shevade, Bhattacharyya & Murthy (2001), "Improvements to Platt's SMO algorithm for SVM classifier design", *Neural Computation* 13(3): 637–649. the maximal-violating-pair selection libsvm builds on. [doi:10.1162/089976601300014493](https://doi.org/10.1162/089976601300014493)
- Chang & Lin (2011), "LIBSVM: a library for support vector machines", *ACM TIST* 2(3). [doi:10.1145/1961189.1961199](https://doi.org/10.1145/1961189.1961199)
- Schölkopf & Smola (2002), *Learning with Kernels*, MIT Press, chapters 7 and 10 (SV classification, implementation). [doi:10.7551/mitpress/4175.001.0001](https://doi.org/10.7551/mitpress/4175.001.0001)

## connections

- ← builds on: Lagrangians, KKT conditions and convex QPs (ch 06), kernels and the representer theorem (topic 03), linear classifiers and the perceptron (ch 08), the estimator protocol (ch 07).
- → used in: margin-based generalization bounds (ch 13), the hinge loss in structured prediction and as a GAN loss (ch 21), calibration of margins (ch 07, ch 28), and the comparison with kernel ridge and GPs, which use the squared loss instead (topics 05–06).
