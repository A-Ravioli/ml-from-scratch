# matrix calculus

## why this matters

training a model means following a gradient downhill. a linear regression's loss is $\|Xw - y\|^2$, a neural network layer computes $\sigma(XW)$, a Gaussian's log-likelihood contains $\log\det\Sigma$. to train any of these you need derivatives with respect to vectors and matrices, and you need them to be *right*. a gradient with one transposed matrix doesn't crash. it quietly trains a worse model, or one that diverges three days into a run.

this topic gives you two tools that the rest of the course leans on. the first is a **calculus for vectors and matrices**: a short list of rules (differentials, the trace trick, "the gradient has the shape of the variable") that turn most ML gradients into a few lines of algebra. the second is the **gradient check**: compare your formula with finite differences, numerically, before you trust it. `numerical_gradient` and `check_gradient` become `mlfs.numgrad`, and every later chapter uses them to test its derivatives, all the way up to your autograd engine in chapter 14.

## warm-up: predict first

1. what's the gradient of $f(x) = x^\top Ax$? (careful: $A$ isn't necessarily symmetric.)
2. what shape is the gradient of a scalar loss with respect to a $3 \times 5$ weight matrix?
3. you estimate a derivative by $(f(x + h) - f(x - h))/2h$. is $h = 10^{-15}$ a good choice?
4. a gradient check passes at $x = 0$. does that tell you your gradient is right?
5. the derivative of $\log x$ is $1/x$. what do you think the gradient of $\log\det X$ is?

## prerequisites check

- [ ] you can take partial derivatives and apply the chain rule (→ ch 01 · topic 02)
- [ ] you know finite differences and their $O(h)$ / $O(h^2)$ errors in 1-D (→ ch 01 · topic 02)
- [ ] you know $\operatorname{tr}(AB) = \operatorname{tr}(BA)$ and can multiply matrices fluently (→ ch 02 · topic 02)
- [ ] you can compute a stable softmax and logsumexp (→ ch 00 · topic 03)
- [ ] you can solve linear systems with your `solve` (→ ch 02 · topic 03)

## core ideas

### idea 1: numerical gradients

for $f: \mathbb{R}^n \to \mathbb{R}$, the gradient collects the partial derivatives, $\nabla f(x) = (\partial f/\partial x_1, \dots, \partial f/\partial x_n)$. each partial is an ordinary 1-D derivative along a coordinate axis. so we can estimate it by nudging one coordinate at a time:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + he_i) - f(x - he_i)}{2h}.$$

Taylor's theorem says why the **central** difference is the one to use: the even-order terms cancel,

$$f(x \pm he_i) = f(x) \pm h\,\partial_if + \tfrac{h^2}{2}\partial_i^2f \pm \tfrac{h^3}{6}\partial_i^3f + \dots \quad\Longrightarrow\quad \frac{f(x + he_i) - f(x - he_i)}{2h} = \partial_if + O(h^2),$$

while the forward difference $(f(x + he_i) - f(x))/h$ is only $O(h)$.

**how small should $h$ be?** smaller isn't better (warm-up 3). $f$ is computed with a relative rounding error of about $\varepsilon \approx 10^{-16}$, so the numerator carries an absolute error of about $\varepsilon|f|$, and dividing by $h$ amplifies it. the total error is about

$$\underbrace{c_1h^2}_{\text{truncation}} + \underbrace{c_2\,\varepsilon/h}_{\text{rounding}},$$

minimized near $h \approx \varepsilon^{1/3} \approx 10^{-5}$, where the error is about $\varepsilon^{2/3} \approx 10^{-11}$. (for forward differences, $h \approx \sqrt\varepsilon \approx 10^{-8}$ and error $\approx 10^{-8}$.) at $h = 10^{-15}$, rounding dominates completely. experiment 1 traces the whole U-shaped curve.

nothing here needs $x$ to be a vector. for a matrix variable $X$ (or any array), nudge each entry in turn. the result has $X$'s shape. iterate with `np.ndindex(X.shape)` so one function handles every shape.

> **▶ build: stage 1.** implement `numerical_gradient(f, X, *, h=1e-6)`: central differences entry by entry, the result shaped like `X`, `X` left unmodified. (work on a float copy, and restore each entry after nudging it.)

### idea 2: checking a gradient

to check an analytic gradient $g$ against the numerical one $g_\text{fd}$, use a **relative** error, so the test doesn't depend on the scale of $f$:

$$\text{err} = \frac{\|g - g_\text{fd}\|}{\max(\|g\| + \|g_\text{fd}\|,\ 10^{-12})}.$$

the denominator is symmetric (neither gradient is presumed right), and the floor keeps it finite when both gradients are zero. in double precision with $h = 10^{-6}$, a correct gradient typically gives $10^{-10}$ to $10^{-8}$. a bug usually gives $10^{-2}$ or more. values in between deserve suspicion: a kink (like ReLU at 0) near the test point, or float32 data.

**choosing test points** matters as much as the formula. many wrong gradients agree with the right one at special points: at $x = 0$, with symmetric or identity matrices, with all-equal entries, with square shapes where a transpose still type-checks. check at random points, with random non-symmetric matrices of non-square shapes. experiment 2 shows a textbook bug sailing through three out of four "reasonable" checks.

> **▶ build: stage 2.** implement `check_gradient(f, grad_f, X, *, h=1e-6)`, returning the relative error above as a float.

### idea 3: gradients with respect to vectors

the cleanest way to find a gradient is through the **differential**: perturb $x$ by $dx$ and keep the first-order part,

$$f(x + dx) = f(x) + \underbrace{\nabla f(x)^\top dx}_{df} + o(\|dx\|).$$

whatever multiplies $dx$ in $df$ (as $g^\top dx$) is the gradient. this sidesteps index gymnastics, and it fixes the convention used throughout this course: **the gradient of a scalar has the same shape as the variable** (warm-up 2: $3 \times 5$).

- **linear:** $f = a^\top x$. then $df = a^\top dx$, so $\nabla f = a$.
- **quadratic:** $f = x^\top Ax$. then $df = dx^\top Ax + x^\top A\,dx = x^\top A^\top dx + x^\top A\,dx = \big((A + A^\top)x\big)^\top dx$, so $\nabla f = (A + A^\top)x$. this is $2Ax$ only for symmetric $A$ (warm-up 1).
- **least squares:** $f = \|Ax - b\|^2 = r^\top r$ with $r = Ax - b$. then $df = 2r^\top dr = 2r^\top A\,dx$, so $\nabla f = 2A^\top(Ax - b)$. setting it to zero gives topic 04's normal equations.
- **norm:** $f = \|x\| = (x^\top x)^{1/2}$. then $df = \tfrac12(x^\top x)^{-1/2}\,2x^\top dx$, so $\nabla f = x/\|x\|$: a unit vector, undefined at $0$, where the norm has a corner.

**jacobians and the chain rule.** for a vector-valued map $F: \mathbb{R}^n \to \mathbb{R}^m$, the derivative is the $m \times n$ **Jacobian** $J_F$, with $(J_F)_{ij} = \partial F_i/\partial x_j$: the matrix that maps a small input change to the output change, $dF = J_F\,dx$. for $F(x) = Ax - b$, $J_F = A$. compositions multiply Jacobians: if $f(x) = g(F(x))$, then

$$df = J_g\,dF = J_g\,J_F\,dx, \qquad J_{g\circ F} = J_g\,J_F.$$

a scalar function $f$ has a $1 \times n$ Jacobian, a row, and the gradient is its transpose: $\nabla f = J_f^\top$. so for a scalar loss $f = g(F(x))$,

$$\nabla f = J_F^\top\,\nabla g\big(F(x)\big).$$

least squares is exactly this: $g(r) = \|r\|^2$ has $\nabla g = 2r$, $F(x) = Ax - b$ has $J_F = A$, so $\nabla f = A^\top\cdot 2r$. every gradient in this topic, and all of backpropagation (chapter 14), is this rule applied link by link: a gradient is pulled back through each map by multiplying with that map's *transposed* Jacobian. (it's also why the gradient has the shape of the variable while the Jacobian has shape outputs × inputs.)

the chain rule is built in to the differential method too. in the least-squares case we wrote $df$ in terms of $dr$, then $dr$ in terms of $dx$, and multiplied. a **shape check** catches most slips: in $2A^\top(Ax - b)$, $A^\top$ is $n \times m$ and $Ax - b$ is $m$-long, so the result is $n$-long like $x$. writing $2A(Ax - b)$ would only "work" for square $A$, and it'd be wrong.

> **🔮 predict:** central differences have error $O(h^2)$. for the quadratic and least-squares losses above, how big is the truncation error, for any $h$?

> **▶ build: stage 3.** implement `grad_linear(a, x)`, `grad_quadratic(A, x)`, `grad_least_squares(A, b, x)` and `grad_norm(x)`. derive each on paper first.

### idea 4: gradients with respect to matrices

for a scalar function of a matrix $X$, the gradient $G$ (same shape as $X$) is defined by

$$df = \sum_{ij}G_{ij}\,dX_{ij} = \operatorname{tr}(G^\top dX).$$

so the recipe is: **write $df$ as $\operatorname{tr}(\text{something}\cdot dX)$; the gradient is the transpose of the something.** three rules do almost all the work:

$$\operatorname{tr}(ABC) = \operatorname{tr}(CAB) = \operatorname{tr}(BCA),\qquad \operatorname{tr}(A^\top) = \operatorname{tr}(A),\qquad a^\top b = \operatorname{tr}(ba^\top)\ \text{for vectors}.$$

- **trace:** $f = \operatorname{tr}(AX)$. then $df = \operatorname{tr}(A\,dX)$, so the "something" is $A$, and $\nabla f = A^\top$.
- **bilinear:** $f = a^\top Xb$. then $df = a^\top dX\,b = \operatorname{tr}(ba^\top dX)$, so $\nabla f = ab^\top$, a rank-one gradient. this is exactly why a linear layer's weight gradient, summed over a batch, is $\sum_i \delta_ix_i^\top$ (chapter 14).
- **matrix least squares:** $f = \|XW - Y\|_F^2 = \operatorname{tr}(R^\top R)$ with $R = XW - Y$. then $df = 2\operatorname{tr}(R^\top X\,dW)$, so $\nabla_W f = 2X^\top(XW - Y)$, the vector formula applied to each column of $W$ at once.
- **log-determinant:** a classic result (Jacobi's formula) says $d\det X = \det X\cdot\operatorname{tr}(X^{-1}dX)$. so $d\log|\det X| = \operatorname{tr}(X^{-1}dX)$, and $\nabla\log|\det X| = X^{-\top}$: the matrix version of $(\log x)' = 1/x$, with a transpose (warm-up 5). it appears in every Gaussian log-likelihood (chapter 04) and in normalizing flows (chapter 21). compute $X^{-\top}$ by *solving* $X^\top G = I$ with your `solve`, not by inverting entry by entry.

> **▶ build: stage 4.** implement `grad_trace(A, X)`, `grad_bilinear(a, b, X)`, `grad_matrix_least_squares(X, Y, W)` and `grad_logdet(X)`.

### idea 5: the gradients of classification losses

two losses dominate classification, and their gradients share one beautiful form.

**softmax cross-entropy.** logits $z \in \mathbb{R}^k$, true class $y$. the probabilities are $p = \operatorname{softmax}(z)$, $p_j = e^{z_j}/\sum_l e^{z_l}$, and the loss is

$$f(z) = -\log p_y = \operatorname{logsumexp}(z) - z_y.$$

the derivative of $\operatorname{logsumexp}(z) = \log\sum_l e^{z_l}$ is $e^{z_j}/\sum_l e^{z_l} = p_j$, and the derivative of $z_y$ is $e_y$. so

$$\nabla_z f = p - e_y \quad(\text{predicted probabilities minus the one-hot target}).$$

no softmax Jacobian needed: going through logsumexp skips it. compute $p$ stably, subtracting $\max z$ before exponentiating, or logits around 1000 will overflow.

**logistic regression.** labels $y_i \in \{0, 1\}$, score $z_i = x_i^\top w$, probability $\sigma(z_i)$. the mean negative log-likelihood is

$$f(w) = \frac1n\sum_i\big[\log(1 + e^{z_i}) - y_iz_i\big].$$

since $\frac{d}{dz}\log(1 + e^z) = \sigma(z)$, each term's derivative in $z_i$ is $\sigma(z_i) - y_i$, and the chain rule through $z = Xw$ gives

$$\nabla_w f = \frac1nX^\top\big(\sigma(Xw) - y\big).$$

prediction minus target again, pulled back through the linear map by $X^\top$. this pattern (it holds for every "canonical" generalized linear model, chapter 08) is why these losses are so pleasant to optimize.

> **🔮 predict:** logits $z = (1000, 0)$, true class $y = 1$. what should the gradient be? what does `np.exp(z) / np.exp(z).sum()` give you?

> **▶ build: stage 5.** implement `grad_softmax_cross_entropy(z, y)` (stable for huge logits) and `grad_logistic(X, y, w)` (mean, not sum).

## worked example

$f(x) = \|Ax - b\|^2$ with

$$A = \begin{pmatrix}1&2\\3&4\end{pmatrix},\qquad b = \begin{pmatrix}1\\1\end{pmatrix},\qquad x = \begin{pmatrix}1\\-1\end{pmatrix}.$$

**analytic.** $Ax = (1 - 2,\ 3 - 4) = (-1, -1)$, so $r = Ax - b = (-2, -2)$ and $f = 8$. then

$$\nabla f = 2A^\top r = 2\begin{pmatrix}1&3\\2&4\end{pmatrix}\begin{pmatrix}-2\\-2\end{pmatrix} = 2\begin{pmatrix}-8\\-12\end{pmatrix} = \begin{pmatrix}-16\\-24\end{pmatrix}.$$

**numerical, by hand, for the first coordinate.** $A(x + he_1) - b = r + h\,a_1 = (-2 + h,\ -2 + 3h)$, so

$$f(x + he_1) = (-2 + h)^2 + (-2 + 3h)^2 = 8 - 16h + 10h^2.$$

the central difference is $\frac{(8 - 16h + 10h^2) - (8 + 16h + 10h^2)}{2h} = -16$, **exactly**, for every $h$. the $h^2$ terms cancel. central differences are exact on quadratics (up to rounding), which answers the predict callout in idea 3. a forward difference would give $-16 + 10h$.

the tests check $(-16, -24)$, and compare all ten of your gradients with finite differences at random, non-symmetric, non-square points.

## on paper

derive each of these before (or while) coding the matching stage. the solutions use the differential method from ideas 3–4.

1. **quadratic with a symmetric part.** show that $x^\top Ax = x^\top A_sx$ with $A_s = \frac12(A + A^\top)$, and use it to re-derive $\nabla(x^\top Ax) = (A + A^\top)x$. what is the Hessian?

<details><summary>solution</summary>

$x^\top Ax$ is a scalar, so it equals its transpose $x^\top A^\top x$. averaging the two gives $x^\top A_sx$. for symmetric $A_s$, $\nabla(x^\top A_sx) = 2A_sx = (A + A^\top)x$. the Hessian is $A + A^\top$, which is constant. the antisymmetric part of $A$ is invisible to the quadratic form.

</details>

2. **$\nabla_W\|XW - Y\|_F^2$ by the trace trick.** fill in the steps: $f = \operatorname{tr}(R^\top R)$, $df = \operatorname{tr}(dR^\top R + R^\top dR) = 2\operatorname{tr}(R^\top dR)$, $dR = X\,dW$.

<details><summary>solution</summary>

$\operatorname{tr}(dR^\top R) = \operatorname{tr}((dR^\top R)^\top) = \operatorname{tr}(R^\top dR)$, so $df = 2\operatorname{tr}(R^\top X\,dW)$. matching $\operatorname{tr}(G^\top dW)$ gives $G^\top = 2R^\top X$, i.e. $G = 2X^\top R = 2X^\top(XW - Y)$. shapes: $X^\top$ is $d \times n$ and $R$ is $n \times k$, so $G$ is $d \times k$, like $W$. ✓

</details>

3. **Jacobi's formula.** show that $\det(X + dX) = \det X\cdot\det(I + X^{-1}dX)$ and $\det(I + E) = 1 + \operatorname{tr}E + O(\|E\|^2)$, hence $d\log|\det X| = \operatorname{tr}(X^{-1}dX)$ and $\nabla\log|\det X| = X^{-\top}$.

<details><summary>solution</summary>

the first identity is multiplicativity of det: $X + dX = X(I + X^{-1}dX)$. for the second, expand $\det(I + E)$ as a sum over permutations. the identity permutation contributes $\prod_i(1 + E_{ii}) = 1 + \sum_iE_{ii} + O(E^2)$, and every other permutation uses at least two off-diagonal entries, so it's $O(E^2)$. so $\log|\det(X + dX)| = \log|\det X| + \operatorname{tr}(X^{-1}dX) + O(dX^2)$. matching $\operatorname{tr}(G^\top dX)$ gives $G^\top = X^{-1}$, so $G = X^{-\top}$.

</details>

4. **softmax Jacobian, for comparison.** show that $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$, i.e. $J = \operatorname{diag}(p) - pp^\top$, and verify that $J^\top(-e_y/p_y) = p - e_y$, the chain rule through the softmax giving the same answer as idea 5.

<details><summary>solution</summary>

$p_i = e^{z_i}/S$ with $S = \sum_l e^{z_l}$, so $\partial p_i/\partial z_j = \delta_{ij}e^{z_i}/S - e^{z_i}e^{z_j}/S^2 = p_i\delta_{ij} - p_ip_j$. the loss $-\log p_y$ has gradient $-e_y/p_y$ with respect to $p$. the chain rule gives $J^\top(-e_y/p_y)$. $J$ is symmetric, and $J(-e_y/p_y)$ has entries $-(p_i\delta_{iy} - p_ip_y)/p_y = p_i - \delta_{iy}$. so it's $p - e_y$. ✓ (going through logsumexp is shorter and avoids dividing by a possibly tiny $p_y$.)

</details>

## experiment

`lab.py` investigates three things with your code:

1. the error of your `numerical_gradient` (and of a forward difference) against your `grad_logistic`, as the step $h$ shrinks from $10^{-1}$ to $10^{-15}$.
2. whether your `check_gradient` catches a classic transposition bug at four different kinds of test point.
3. the running time of a numerical gradient against an analytic one as the number of parameters grows, extrapolated to a modern network.

## check your understanding

1. why does the gradient of $f(X)$ have the same shape as $X$, while a Jacobian of $F: \mathbb{R}^n \to \mathbb{R}^m$ is $m \times n$?

<details><summary>answer</summary>

a convention, chosen so that gradient descent $X \leftarrow X - \eta\nabla f$ type-checks. the gradient is defined by $df = \langle\nabla f, dX\rangle$, a pairing with something of $X$'s shape. a Jacobian maps input perturbations to output perturbations, $dF = J\,dx$, so it's $m \times n$. for scalar $f$ of a vector, $\nabla f = J^\top$.

</details>

2. `check_gradient` returns $3 \times 10^{-6}$. right or wrong?

<details><summary>answer</summary>

suspicious, not conclusive. common benign causes: a kink (ReLU, abs, max) within $h$ of the test point, float32 arithmetic, or a badly scaled $f$ (very large values make $\varepsilon|f|/h$ big). rerun at a few other random points with $h \in \{10^{-4}, 10^{-5}, 10^{-6}\}$. a true bug stays large and doesn't move much with $h$. truncation error changes like $h^2$.

</details>

3. why is the numerical gradient useless for *training*, even though it's perfect for *testing*?

<details><summary>answer</summary>

it costs $2n$ evaluations of the loss for $n$ parameters (experiment 3). that's fine for testing a formula on a small problem, and hopeless for a model with millions of parameters at every step. analytic gradients, and later reverse-mode autodiff, cost a small constant times one evaluation.

</details>

4. for $f(x) = \|x\|$, what goes wrong with the numerical gradient at $x = 0$?

<details><summary>answer</summary>

$\|x\|$ has a cone-shaped kink at 0 and isn't differentiable there. the central difference gives $(h - h)/(2h) = 0$ in each coordinate, which is a valid *subgradient* but not a gradient, while `grad_norm` divides by zero. gradient checks at nondifferentiable points are meaningless.

</details>

5. $\nabla_X\operatorname{tr}(AXB) = ?$ (use the cyclic rule).

<details><summary>answer</summary>

$df = \operatorname{tr}(A\,dX\,B) = \operatorname{tr}(BA\,dX)$, so $\nabla f = (BA)^\top = A^\top B^\top$.

</details>

6. in idea 5, why is $\sigma(Xw) - y$ multiplied by $X^\top$ and not by $X$?

<details><summary>answer</summary>

the chain rule through $z = Xw$: $df = \sum_i(\sigma(z_i) - y_i)\,dz_i/n = (\sigma(z) - y)^\top X\,dw/n$, so the gradient is $X^\top(\sigma(z) - y)/n$. the shapes force it too: $X^\top$ is $d \times n$, the residual is $n$-long, and the gradient must be $d$-long.

</details>

7. (the predict callout in idea 5.) logits $z = (1000, 0)$ and true class $y = 1$: what's the gradient, and what does the naive softmax give?

<details><summary>answer</summary>

$p = \operatorname{softmax}(z) = (1/(1 + e^{-1000}), e^{-1000}/(1 + e^{-1000})) \approx (1, 0)$, so $\nabla_z f = p - e_1 \approx (1, -1)$: push the wrong logit down, the right one up. naively, `np.exp(1000)` overflows to `inf`, and `inf / inf` gives `nan`, so the gradient is `[nan, 0]` (with an overflow warning). subtracting $\max z$ first computes `exp([0, -1000]) = [1, 0]` exactly, and the answer comes out right.

</details>

8. for $F: \mathbb{R}^3 \to \mathbb{R}^2$ and $g: \mathbb{R}^2 \to \mathbb{R}$, what are the shapes of $J_F$, $J_g$ and $\nabla(g \circ F)$?

<details><summary>answer</summary>

$J_F$ is $2 \times 3$, $J_g$ is $1 \times 2$, and $J_{g\circ F} = J_gJ_F$ is $1 \times 3$, so $\nabla(g\circ F) = J_F^\top\nabla g$ is a length-3 vector, the shape of the input.

</details>

## common bugs

- **mutating `X` inside `numerical_gradient`.** symptom: the gradient is fine, but the caller's array has changed, or later entries are computed at a shifted point. perturb a copy and restore each entry.
- **integer arrays.** `X[i] += 1e-6` on an `int` array does nothing. symptom: an all-zero numerical gradient. convert with `np.array(X, dtype=float)`.
- **$2Ax$ for $x^\top Ax$.** symptom: passes for symmetric $A$, fails otherwise. use $(A + A^\top)x$.
- **missing transposes in matrix gradients.** `grad_trace` returning $A$ instead of $A^\top$, `grad_logdet` returning $X^{-1}$ instead of $X^{-\top}$. symptom: shape-correct but wrong for non-symmetric inputs. check with non-symmetric, non-square test matrices.
- **sum versus mean.** the logistic loss here is a mean. symptom: a gradient exactly $n$ times too large. a learning rate tuned for one convention diverges under the other.
- **overflowing softmax.** symptom: `nan` gradients for large logits. subtract `max(z)` first.

## going deeper

- Petersen & Pedersen (2012), *The Matrix Cookbook*. a reference table of matrix derivatives. check your derivations against it, but derive first. [matrixcookbook.com](https://www.matrixcookbook.com)
- Parr & Howard (2018), "The matrix calculus you need for deep learning". a gentle, example-driven introduction. [arXiv:1802.01528](https://arxiv.org/abs/1802.01528)
- Magnus & Neudecker (2019), *Matrix Differential Calculus with Applications in Statistics and Econometrics*, 3rd ed., Wiley. the rigorous theory of the differential method used here. [doi:10.1002/9781119541219](https://doi.org/10.1002/9781119541219)
- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer, §8.1 (finite-difference derivative approximations and the choice of $h$). [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Baydin, Pearlmutter, Radul & Siskind (2018), "Automatic differentiation in machine learning: a survey", *JMLR* 18(153): 1–43. where experiment 3 leads. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)

## connections

- ← builds on: derivatives and finite differences (ch 01 · topic 02), stable softmax (ch 00 · topic 03), `solve` (topic 03), least squares (topic 04).
- → used in: gradient checks in every later chapter via `mlfs.numgrad`, numerical Jacobians and Hessians (ch 03), gradient descent and Newton's method (ch 06), linear and logistic regression (ch 08), backpropagation and `gradcheck` for your autograd engine (ch 14).
