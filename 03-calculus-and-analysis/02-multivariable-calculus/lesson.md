# multivariable calculus

## why this matters

a model with a million parameters has a loss surface in a million-and-one dimensions. you can't draw it, but you need to know three things about it at every point: which way is downhill, how steeply, and how the surface curves. those are the **gradient**, the **directional derivative**, and the **Hessian**. a layer of a network is a map from vectors to vectors, and to train through it you need its **Jacobian**, the matrix of every output's sensitivity to every input. this topic is the calculus of all four.

the one-variable calculus you know mostly carries over, with two genuinely new phenomena. first, *direction*: at a point there are infinitely many directions to move, and the slope depends on which one you pick. second, *curvature can disagree with itself*: a surface can curve up in one direction and down in another (a saddle), which is exactly the kind of point that slows down neural network training. you'll build finite-difference Jacobians and Hessians that become the first half of **`mlfs.autodiff`**, the reference that every exact derivative in this course (topic 04, chapter 14) is checked against, and a classifier that tells minima from maxima from saddles.

## warm-up: predict first

1. at a point where $\nabla f = (3, 4)$, what's the largest slope you can find by walking in some direction, and which direction gives it?
2. a function $f: \mathbb{R}^5 \to \mathbb{R}^3$. what shape is its Jacobian?
3. is $\partial^2 f/\partial x\,\partial y$ always equal to $\partial^2 f/\partial y\,\partial x$?
4. your gradient checks used $h = 10^{-6}$. is that a good step for estimating second derivatives?
5. at a critical point, the Hessian has eigenvalues $3$, $-1$ and $0$. minimum, maximum, or saddle?

## prerequisites check

- [ ] you can take partial derivatives and use the 1-D chain rule (→ ch 01 · topic 02)
- [ ] you can compute a numerical gradient with `mlfs.numgrad.numerical_gradient`, and you know why central differences are $O(h^2)$ (→ ch 02 · topic 07)
- [ ] you know that a symmetric matrix has real eigenvalues and orthonormal eigenvectors, and what $x^\top Ax > 0$ means (→ ch 02 · topic 05)
- [ ] you can multiply matrices and reason about their shapes (→ ch 02 · topic 02)

## core ideas

### idea 1: slopes in every direction

for $f: \mathbb{R}^n \to \mathbb{R}$, the **partial derivative** $\partial f/\partial x_i$ is the ordinary derivative along coordinate axis $i$, with the other coordinates frozen. collected into a vector, they form the **gradient** $\nabla f(x)$. you built a numerical gradient in chapter 02; now we ask what it's *for*.

walk from $x$ along a direction $v$: the function along that line is $\phi(t) = f(x + tv)$, an ordinary 1-D function. its slope at $t = 0$ is the **directional derivative**

$$D_vf(x) = \lim_{t \to 0}\frac{f(x + tv) - f(x)}{t} = \nabla f(x)\cdot v.$$

the second equality is the chain rule ($\phi'(0) = \sum_i \partial_i f\,v_i$), and it's what "$f$ is differentiable" means in several variables: near $x$, $f$ is well approximated by a *linear* function, $f(x + d) = f(x) + \nabla f(x)\cdot d + o(\|d\|)$, so every directional slope comes from one vector. (having all partial derivatives isn't enough: $f(x, y) = xy/(x^2 + y^2)$, with $f(0, 0) = 0$, has both partials at the origin, both zero, but along the diagonal it's constant at $\tfrac12$, so it isn't even continuous there. the functions in ML are differentiable almost everywhere, and the exceptions, like ReLU at 0, are kinks you'll learn to handle.)

two geometric facts follow from $D_vf = \nabla f\cdot v$:

- **steepest ascent.** over unit vectors $v$, Cauchy–Schwarz gives $\nabla f\cdot v \le \|\nabla f\|$, with equality exactly for $v = \nabla f/\|\nabla f\|$. the gradient points uphill, and its length is the steepest slope (warm-up 1: slope 5, direction $(0.6, 0.8)$). $-\nabla f$ is the steepest way down, which is why gradient descent follows it.
- **level sets.** along a curve on which $f$ is constant, the slope is zero, so the curve's tangent $t$ has $\nabla f\cdot t = 0$: **the gradient is perpendicular to the level sets**. where the contour lines are packed together, $f$ changes fast and the gradient is long.

"steepest" quietly depends on how you measure step length: per unit of Euclidean distance *in your coordinates*. rescale one coordinate and the steepest direction changes, even though the function hasn't. experiment 2 makes this visible, and it's the first hint of why Newton's method, which doesn't care about coordinates, beats gradient descent on badly scaled problems.

> **▶ build: stage 1.** implement `directional_derivative(f, x, v, *, h)` (central difference along $v$, not normalized) and `steepest_direction(f, x, *, h)`, returning the unit gradient direction and the slope $\|\nabla f\|$ (raise `ValueError` at a flat point). use `numerical_gradient` from `mlfs.numgrad`.

### idea 2: the Jacobian

a map $F: \mathbb{R}^n \to \mathbb{R}^m$ has $m$ output functions, each with $n$ partial derivatives. arrange them in an $m \times n$ matrix, row $i$ for output $i$, column $j$ for input $j$:

$$J_F(x) = \begin{pmatrix}\partial F_1/\partial x_1 & \cdots & \partial F_1/\partial x_n\\ \vdots & & \vdots\\ \partial F_m/\partial x_1 & \cdots & \partial F_m/\partial x_n\end{pmatrix}, \qquad F(x + d) \approx F(x) + J_F(x)\,d.$$

the Jacobian is the *best linear approximation* of $F$ near $x$: zoom in far enough and every differentiable map looks like a matrix (warm-up 2: $3 \times 5$, outputs × inputs). a scalar function's Jacobian is a single row, $\nabla f^\top$.

**the chain rule is matrix multiplication.** if $x \mapsto F(x) \mapsto G(F(x))$, linearize each step: a small change $d$ becomes $J_Fd$ after $F$, and that becomes $J_G(J_Fd)$ after $G$. so

$$J_{G\circ F}(x) = J_G(F(x))\,J_F(x).$$

that's the whole multivariable chain rule. a deep network is a long composition, so its Jacobian is a long product of layer Jacobians. evaluating that product right to left, starting from the input, is forward-mode differentiation (topic 04). evaluating it left to right, starting from the scalar loss, is backpropagation (chapter 14). which order is cheaper depends only on the shapes.

**computing it numerically.** column $j$ of $J_F$ is the derivative of $F$ along $e_j$, so one central difference per input gives a whole column:

$$J_F(x)\,e_j \approx \frac{F(x + he_j) - F(x - he_j)}{2h}.$$

$2n$ evaluations of $F$ for the whole matrix, regardless of $m$. a numerical Jacobian is the honest, slow, always-available reference: if an exact derivative disagrees with it at a random point, the exact derivative is wrong.

> **▶ build: stage 2.** implement `numerical_jacobian(f, x, *, h=1e-6)`, shape $(m, n)$, one column per input. a scalar-valued `f` gives a $(1, n)$ result. don't modify `x`, and make integer inputs work.

### idea 3: the Hessian

the second derivatives of $f: \mathbb{R}^n \to \mathbb{R}$ form the $n \times n$ **Hessian**, $H_{ij} = \partial^2 f/\partial x_i\,\partial x_j$. it's the Jacobian of the gradient. it describes **curvature**: along a direction $v$, the second derivative of $\phi(t) = f(x + tv)$ at 0 is

$$\phi''(0) = v^\top H(x)\,v.$$

so $H$ is a machine that takes a direction and returns the curvature along it. for a quadratic $f(x) = \tfrac12x^\top Ax + b^\top x$ with symmetric $A$, $H = A$ everywhere. near any point, $f$ looks like such a quadratic (topic 03 makes that precise).

**symmetry.** if the second partials are continuous, the order of differentiation doesn't matter (Schwarz's theorem, also called Clairaut's): $H_{ij} = H_{ji}$ (warm-up 3: yes, for every function you'll meet in this course; counterexamples need deliberately pathological constructions). a symmetric Hessian has real eigenvalues $\lambda_1, \dots, \lambda_n$ and orthonormal eigenvectors: the **principal curvatures** and the directions in which they occur. $v^\top Hv$ ranges between $\lambda_{\min}$ and $\lambda_{\max}$ over unit $v$.

**computing it numerically.** difference $f$ twice. on the diagonal, the familiar second difference; off the diagonal, a four-point stencil that measures how the $x_i$-slope changes as you move in $x_j$:

$$H_{ii} \approx \frac{f(x + he_i) - 2f(x) + f(x - he_i)}{h^2},\qquad H_{ij} \approx \frac{f(x + he_i + he_j) - f(x + he_i - he_j) - f(x - he_i + he_j) + f(x - he_i - he_j)}{4h^2}.$$

both are exact for quadratics and have $O(h^2)$ truncation error in general. but now the rounding error is about $\varepsilon|f|/h^2$, because you divide by $h^2$. balancing $c\,h^2$ against $\varepsilon/h^2$ puts the best step near $h \approx \varepsilon^{1/4} \approx 10^{-4}$, with an error around $\sqrt\varepsilon \approx 10^{-8}$. at the gradient's $h = 10^{-6}$, rounding is amplified by $10^{12}$ and you keep only about four digits (warm-up 4). that's why `numerical_hessian` defaults to $h = 10^{-4}$.

> **🔮 predict:** you could also get a Hessian by applying `numerical_jacobian` to an *exact* gradient function. what step size would suit that route, and roughly how many correct digits would you get?

> **▶ build: stage 3.** implement `numerical_hessian(f, x, *, h=1e-4)` with the two stencils above. compute each off-diagonal pair once and store it in both places, so the result is exactly symmetric.

### idea 4: critical points and the second-derivative test

a **critical point** has $\nabla f(x^*) = 0$: every directional slope vanishes. it could be a minimum, a maximum, or neither. near $x^*$, the linear term is gone, so the quadratic term decides:

$$f(x^* + d) \approx f(x^*) + \tfrac12d^\top H(x^*)\,d.$$

write $d$ in the Hessian's eigenvector basis, $d = \sum_ic_iq_i$; then $d^\top Hd = \sum_i\lambda_ic_i^2$. so:

- all $\lambda_i > 0$: $f$ rises in every direction. a strict local **minimum**.
- all $\lambda_i < 0$: a strict local **maximum**.
- some positive, some negative: up in some directions, down in others. a **saddle**. this is decided by the signs alone, whatever the other eigenvalues are.
- otherwise (some $\lambda_i = 0$ and no sign mix): **degenerate**. the quadratic term is flat in some direction, and higher-order terms decide. $x^4 + y^2$ and $x^3 + y^2$ have the same Hessian at the origin; one has a minimum there, the other doesn't.

warm-up 5: eigenvalues $3, -1, 0$ are a saddle (the signs mix). in two dimensions you may have learned the test as "$\det H < 0$ means saddle". in three dimensions that fails: $-I_3$ has $\det = -1$ and is a maximum. use eigenvalues.

saddles matter in ML far more than the textbook suggests. for a random high-dimensional function, a critical point with all $n$ curvatures positive is rare: most critical points of neural-network losses are saddles, and the flat, slowly escaping regions around them, not bad local minima, are what slow training down (chapter 06 · topic 07, chapter 19).

in floating point, "$\lambda = 0$" has to mean "$|\lambda|$ is tiny compared with the Hessian's scale". `classify_critical_point` uses the threshold $s = \text{tol}\cdot\max(1, \max_i|\lambda_i|)$.

> **🔮 predict:** $f(x, y) = xy$ has a critical point at the origin, and $\partial^2f/\partial x^2 = \partial^2f/\partial y^2 = 0$ there. is it degenerate?

> **▶ build: stage 4.** implement `classify_critical_point(f, x, *, tol=1e-6)` with your `numerical_hessian` and `np.linalg.eigvalsh`, returning `"minimum"`, `"maximum"`, `"saddle"` or `"degenerate"`.

## worked example

take $f(x, y) = x^3 - 3x + y^2$.

**gradient and a directional derivative.** $\nabla f = (3x^2 - 3,\ 2y)$. at $(2, 1)$ that's $(9, 2)$. along $v = (1, 1)$ the slope is $9 + 2 = 11$ (not normalized: along $(1, 1)/\sqrt2$ it's $11/\sqrt2 \approx 7.78$). the steepest ascent direction is $(9, 2)/\sqrt{85} \approx (0.976, 0.217)$, with slope $\sqrt{85} \approx 9.22$.

**critical points.** $\nabla f = 0$ gives $x = \pm1$, $y = 0$.

**Hessian and classification.** $H = \begin{pmatrix}6x & 0\\ 0 & 2\end{pmatrix}$. at $(1, 0)$: eigenvalues $6, 2$, a **minimum** ($f = -2$). at $(-1, 0)$: eigenvalues $-6, 2$, a **saddle** ($f = 2$): along $x$ it's a local maximum of the cubic, along $y$ a parabola opening up.

**a Jacobian.** the polar-coordinate map $F(r, \theta) = (r\cos\theta,\ r\sin\theta)$ has

$$J_F = \begin{pmatrix}\cos\theta & -r\sin\theta\\ \sin\theta & r\cos\theta\end{pmatrix},\qquad J_F(2, \pi/2) = \begin{pmatrix}0 & -2\\ 1 & 0\end{pmatrix}.$$

column 1 says: increasing $r$ at $\theta = \pi/2$ moves the point straight up. column 2 says: increasing $\theta$ moves it left, twice as fast, because it's on a circle of radius 2. ($\det J_F = r$: topic 06 needs this.)

the tests check all of these numbers.

## on paper

1. **steepest ascent.** prove that among unit vectors $v$, $D_vf(x) = \nabla f(x)\cdot v$ is largest for $v = \nabla f/\|\nabla f\|$, and smallest for its negative. what's the set of directions with zero slope?

<details><summary>solution</summary>

Cauchy–Schwarz: $|\nabla f\cdot v| \le \|\nabla f\|\|v\| = \|\nabla f\|$, with equality iff $v$ is parallel to $\nabla f$. so the maximum $\|\nabla f\|$ is attained at $v = \nabla f/\|\nabla f\|$, the minimum $-\|\nabla f\|$ at its negative. the zero-slope directions are those with $\nabla f\cdot v = 0$: the hyperplane orthogonal to the gradient, which is the tangent space of the level set.

</details>

2. **a chain rule by matrices.** let $F(x, y) = (x^2y,\ x + y)$ and $g(u, w) = uw$. compute $\nabla(g\circ F)$ at $(1, 2)$ twice: by multiplying Jacobians, and by expanding $g(F(x, y))$ first.

<details><summary>solution</summary>

$J_F = \begin{pmatrix}2xy & x^2\\ 1 & 1\end{pmatrix} = \begin{pmatrix}4 & 1\\ 1 & 1\end{pmatrix}$ at $(1, 2)$, and $F(1, 2) = (2, 3)$, so $J_g = (w, u) = (3, 2)$. the product is $(3, 2)\begin{pmatrix}4 & 1\\ 1 & 1\end{pmatrix} = (14, 5)$. directly: $g(F) = x^2y(x + y) = x^3y + x^2y^2$, with gradient $(3x^2y + 2xy^2,\ x^3 + 2x^2y) = (6 + 8, 1 + 4) = (14, 5)$. ✓

</details>

3. **the four-point stencil.** show that the off-diagonal stencil is exact for $f(x) = x_ix_j$ (it returns 1), and that for any smooth $f$ its leading error term is $O(h^2)$. then estimate the best $h$ when $f$ is computed with relative error $\varepsilon$.

<details><summary>solution</summary>

for $f = x_ix_j$: $(x_i + h)(x_j + h) - (x_i + h)(x_j - h) - (x_i - h)(x_j + h) + (x_i - h)(x_j - h) = 4h^2$, so the stencil gives $4h^2/4h^2 = 1$. for general $f$, Taylor-expand each of the four values to fourth order: every term odd in $h_i$ or in $h_j$ cancels by symmetry, the $h_ih_j$ term survives with coefficient $4h^2\partial_{ij}f$, and the next surviving terms are $O(h^4)$, so after dividing by $4h^2$ the error is $O(h^2)$. rounding contributes about $4\varepsilon|f|/(4h^2) = \varepsilon|f|/h^2$. minimizing $ch^2 + \varepsilon|f|/h^2$ gives $h^4 = \varepsilon|f|/c$, i.e. $h \approx \varepsilon^{1/4} \approx 10^{-4}$ for well-scaled $f$, and an error near $2\sqrt{c\varepsilon|f|} \sim 10^{-8}$.

</details>

4. **Hessian of a quadratic form.** show that the Hessian of $f(x) = x^\top Ax$ is $A + A^\top$ for any square $A$, and that $f$ has a strict minimum at 0 iff $A + A^\top$ is positive definite.

<details><summary>solution</summary>

$\nabla f = (A + A^\top)x$ (ch 02 · topic 07), and its Jacobian is the constant matrix $A + A^\top$. $f(0) = 0$ and $f(d) = d^\top Ad = \tfrac12d^\top(A + A^\top)d$ exactly, so $f(d) > 0$ for all $d \ne 0$ iff $A + A^\top$ is positive definite. here the second-derivative test is exact, because $f$ *is* its own quadratic model.

</details>

## experiment

`lab.py` investigates three things with your code:

1. the angle between your steepest-ascent directions and the level sets of a bumpy 2-D function, and where the slopes are largest.
2. what happens to "the direction of steepest descent" when you measure one coordinate in different units, compared with Newton's direction.
3. the accuracy of your `numerical_hessian` as the step $h$ shrinks, against differencing an exact gradient.

## check your understanding

1. why is the directional derivative linear in $v$, and what assumption does that need?

<details><summary>answer</summary>

$D_vf = \nabla f\cdot v$, which is linear in $v$. that formula needs $f$ to be differentiable at $x$ (well approximated by a linear function), not just to have partial derivatives: $xy/(x^2 + y^2)$ has both partials at 0 (both zero), but along any direction off the axes the directional derivative doesn't even exist: on the ray $tv$ the function is the constant $v_1v_2/\|v\|^2$ for $t \ne 0$ and jumps to 0 at $t = 0$.

</details>

2. a contour plot shows contour lines spaced 1 mm apart in one region and 1 cm apart in another, for equally spaced levels. where is the gradient larger, and by how much?

<details><summary>answer</summary>

in the tightly spaced region, about 10× larger: the change in $f$ between neighbouring contours is the same, and the slope across them is that change divided by their spacing. the gradient is perpendicular to the contours, with length inversely proportional to their spacing.

</details>

3. what are the shapes of $J_F$, $J_G$ and $J_{G\circ F}$ for $F: \mathbb{R}^4 \to \mathbb{R}^3$ and $G: \mathbb{R}^3 \to \mathbb{R}^2$? why can't the product be in the other order?

<details><summary>answer</summary>

$J_F$ is $3 \times 4$, $J_G$ is $2 \times 3$, and $J_{G\circ F} = J_GJ_F$ is $2 \times 4$. $J_FJ_G$ would multiply $3 \times 4$ by $2 \times 3$, which doesn't type-check: the shapes force the order, which matches "first $F$, then $G$".

</details>

4. (the predict callout in idea 3.) what step suits differencing an exact gradient, and how accurate is it?

<details><summary>answer</summary>

it's a first difference of an exactly computed function, so the usual central-difference analysis applies: $h \approx \varepsilon^{1/3} \approx 10^{-5}$, error around $\varepsilon^{2/3} \approx 10^{-11}$. that's about a thousand times better than second differences of $f$ (experiment 3), and why exact gradients plus a finite-difference Hessian is a strong practical combination.

</details>

5. (the predict callout in idea 4.) is the critical point of $f = xy$ degenerate?

<details><summary>answer</summary>

no. $H = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}$ has eigenvalues $\pm1$, so it's a saddle: $f$ rises along $(1, 1)$ and falls along $(1, -1)$. a zero diagonal says nothing about the eigenvalues; the curvature lives in the off-diagonal entries.

</details>

6. at a critical point the Hessian has eigenvalues $10^{-12}$ and $4$. what does `classify_critical_point` say, and is that the truth?

<details><summary>answer</summary>

with $s = 10^{-6}\cdot 4$, the first eigenvalue counts as zero, so "degenerate". the truth might be a minimum, but a curvature of $10^{-12}$ is below what second differences with $h = 10^{-4}$ can resolve (their error is around $10^{-8}$), so "can't tell from second derivatives" is the honest answer.

</details>

7. why is it surprising, and important, that most critical points of high-dimensional losses are saddles?

<details><summary>answer</summary>

intuition from 1-D and 2-D pictures says critical points are mostly minima and maxima. but a minimum needs all $n$ curvatures positive; if signs were like coin flips, that happens with probability $2^{-n}$. for neural networks, the empirical and theoretical picture (Dauphin et al., 2014) is that low-loss regions are full of saddles and flat directions. methods that only look at the gradient slow down near saddles, which motivates momentum, noise and curvature-aware methods.

</details>

## common bugs

- **normalizing $v$ in the directional derivative.** symptom: the slope along $(1, 1)$ comes out $\sqrt2$ times too small. the derivative along $v$ uses $v$ as given; normalize only when you want the slope per unit length.
- **a transposed Jacobian.** symptom: shape $(n, m)$, or correct values only for square maps. row $i$ is output $i$; stack the columns along `axis=1`.
- **an unsymmetric Hessian.** symptom: $H \ne H^\top$ by $10^{-9}$, and `eigvalsh` silently uses only one triangle. compute each pair once and write it to both $H_{ij}$ and $H_{ji}$.
- **$h = 10^{-6}$ for second differences.** symptom: Hessians correct to only about four digits. use $h \approx 10^{-4}$, or difference an exact gradient.
- **the 2-D determinant test in $n$ dimensions.** symptom: $-x^2 - y^2 - z^2$ classified as a saddle. use the signs of the eigenvalues.
- **calling a sign-mixed Hessian degenerate.** symptom: $x^2 - y^2 + z^4$ reported as degenerate. one positive and one negative eigenvalue already prove a saddle; check that first.

## going deeper

- Hubbard & Hubbard (2015), *Vector Calculus, Linear Algebra, and Differential Forms: A Unified Approach*, 5th ed., Matrix Editions, chapters 1–3. derivatives as linear maps, the chain rule, and the second-derivative test, done properly.
- Deisenroth, Faisal & Ong (2020), *Mathematics for Machine Learning*, Cambridge University Press, chapter 5 (vector calculus). gradients, Jacobians and Hessians with ML examples; free online. [mml-book.github.io](https://mml-book.github.io)
- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer, §8.1 (finite-difference Jacobians and Hessians, and the choice of step). [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Dauphin, Pascanu, Gulcehre, Cho, Ganguli & Bengio (2014), "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization", *NeurIPS*. why saddles, not local minima, dominate. [arXiv:1406.2572](https://arxiv.org/abs/1406.2572)

## connections

- ← builds on: partial derivatives (ch 01 · topic 02), `numerical_gradient` (ch 02 · topic 07), eigenvalues of symmetric matrices (ch 02 · topic 05).
- → used in: quadratic models built from your gradient and Hessian (topic 03), forward-mode autodiff, checked against your `numerical_jacobian` (topic 04), smoothness constants from Hessian eigenvalues (topic 05), Jacobian determinants in change of variables (topic 06), the Newton capstone, Newton's method and convexity (ch 06), backpropagation as a product of Jacobians (ch 14), saddles in deep learning (ch 06 · topic 07, ch 19).
