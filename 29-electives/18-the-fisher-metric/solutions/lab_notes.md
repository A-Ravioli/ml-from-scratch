# the fisher metric · lab notes

numbers are from the reference solution with the lab's defaults (one CPU thread, float64). all three experiments use synthetic data, so they're the same offline and online.

## experiment 1: how big is a step?

**what people usually predict:** steps of the same length change the predictions by roughly the same amount, within a factor of 2 or so; every step changes them a little.

**what happens:** the 200 steps of Euclidean length 0.05 move the predictions by a mean KL from $4.3\cdot10^{-7}$ to $3.8\cdot10^{-5}$, a factor of **89**. (with five different seeds for the data, model and directions, the factor ranges from 40 to 167.) the Fisher quadratic form predicts every one of them: KL / $\tfrac12t^2u^\top Fu$ lies between 0.966 and 1.049 at this step size, and the scatter plot lies on the diagonal. $F$ has 17 eigenvalues that are zero to machine precision, and the smallest of the rest is $3\cdot10^{-15}$ against a largest of 0.22.

growing the step along fixed directions:

| direction | $t$ | $\tfrac12t^2u^\top Fu$ | KL(old ‖ new) | KL(new ‖ old) |
|---|---|---|---|---|
| top eigenvector of $F$ | 0.0074 | $6.0\cdot10^{-6}$ | $5.99\cdot10^{-6}$ | $5.98\cdot10^{-6}$ |
| top eigenvector of $F$ | 0.21 | 0.00474 | 0.00468 | 0.00481 |
| top eigenvector of $F$ | 3 | 0.986 | 2.5 | 1.7 |
| random direction | 0.21 | $3.97\cdot10^{-5}$ | $4.41\cdot10^{-5}$ | $4.52\cdot10^{-5}$ |
| random direction | 3 | 0.0083 | 0.0435 | 0.0764 |
| softmax shift | any | 0 | 0 | 0 |

**why:** the KL caused by a step $\Delta$ is $\tfrac12\Delta^\top F\Delta$ to second order, and for a fixed $\lVert\Delta\rVert$ that ranges from the smallest to the largest eigenvalue of $F$ in the directions you happen to sample. random directions average over many eigenvalues, which is why their spread is "only" two orders of magnitude; the extremes, an eigenvector of the largest eigenvalue against one of the near-zero ones, differ by more than ten orders. most of the parameter space barely matters to the predictions, and a few directions matter a lot: that's exactly the situation where gradient descent's "same step length in every direction" is a poor choice, and the reason topic 19 preconditions by $F^{-1}$.

the two KL directions agree with each other and with the quadratic form for small steps because all three share value, gradient and Hessian at $\Delta = 0$ (on paper 3); they first differ at third order, which is why their ratio to the quadratic form drifts away from 1 roughly linearly in $t$ and they split apart visibly only once the KL is around $10^{-2}$. at $t = 3$ along the top eigenvector the true KL is 2.5 times the quadratic prediction: "KL ≈ Fisher quadratic form" is a statement about small steps, and a natural-gradient step that trusts it too far will overshoot (topic 19's damping and topic 20's KL clipping are the guards).

the softmax-shift direction (every output row moves by the same vector, every output bias by the same number) changes all logits by the same amount for every input, so the predictions don't change at all: the KL is 0 at every step size, not only to second order. it's a symmetry of the parameterization, and $F$ has a null space of exactly that dimension (16 + 1 = 17). the many tiny-but-nonzero eigenvalues are different: directions that do change the predictions, but almost imperceptibly on these 100 inputs (saturated units, nearly redundant units). both kinds make $F$ singular for practical purposes, which is why every natural-gradient method adds damping.

## experiment 2: is the empirical Fisher the Fisher?

**what people usually predict:** close at the fit in both worlds (the empirical Fisher is "the Fisher estimated from data"), and closer with more data.

**what happens:** relative difference $\lVert\hat F_{\text{emp}} - F\rVert_F/\lVert F\rVert_F$, median of three seeds (per-seed values in the lab):

| world | $N$ | at init | at the MLE |
|---|---|---|---|
| well-specified | 100 | 0.67 | 0.24 |
| well-specified | 300 | 0.70 | 0.14 |
| well-specified | 1,000 | 0.71 | 0.09 |
| misspecified | 100 | 0.73 | 0.51 |
| misspecified | 300 | 0.74 | 0.47 |
| misspecified | 1,000 | 0.74 | 0.52 |

at a random initialization the two disagree by about 70% in both worlds, whatever $N$. at the fit, the gap shrinks steadily with $N$ in the well-specified world (roughly like $1/\sqrt N$: 0.24, 0.14, 0.09) and doesn't shrink at all in the misspecified one (about 0.5 at every $N$; the seed spread at $N = 100$ is large, 0.29 to 1.06).

**why:** the empirical Fisher replaces the model's expectation over $y$ with the observed label. per example, the outer product $(e_{y_n} - p_n)(e_{y_n} - p_n)^\top$ has expectation $\operatorname{diag}(p_n) - p_np_n^\top$ only if $y_n$ is drawn from $p_n$. at initialization the model's predictions have nothing to do with the labels, so the residuals $e_{y_n} - p_n$ are large and systematic, and the empirical Fisher measures how wrong the model is, not its curvature. at the maximum-likelihood fit of a well-specified model, $p_\theta(y\mid x)$ approaches the true label distribution, so the expectation is right and what's left is sampling noise, which averages away with $N$. in the misspecified world, the model is overconfident far from the origin (where the true labels are coin tosses), so its residuals there are much larger than its own $p(1 - p)$ predicts; that bias doesn't average away. so you need both conditions, a well-specified model and enough data, and real networks are always misspecified to some degree and are mostly used far from their optimum (Kunstner et al., 2019, make this argument in detail).

for Adam: its second-moment estimate is often called a diagonal empirical Fisher, but it isn't even that: it averages the square of the *minibatch-mean* gradient over time, not the per-example squared gradients, so it mixes the empirical Fisher's diagonal with the squared full gradient and depends on the batch size (Kunstner, Balles & Hennig, 2019, make the same point). so whatever Adam's virtues (and they're real, as topic 20's project shows), "Adam approximates natural gradient" is at best true near the optimum of a well-specified model, which is where it matters least.

## experiment 3: the length of a coin

**what people usually predict:** that the whole family is infinitely long (the logit is unbounded), or that the length depends on the coordinates.

**what happens:** with the default sliders (0.01 to 0.99), the Euclidean lengths of the three descriptions are 0.98 (in $p$), 9.19 (in the logit) and 0.895 (in $\sqrt p$), while all three Fisher–Rao lengths are 2.7409, matching the closed form $2|\arcsin\sqrt{p_1} - \arcsin\sqrt{p_0}|$ to four digits. the whole family, from the logit $-21$ to $21$ ($p$ from $10^{-9}$ to $1 - 10^{-9}$), has length 3.1415, against $\pi = 3.1416$.

**why:** the three paths are the same curve on the manifold, described in different coordinates; the Fisher matrix in each coordinate system is the pullback of the others, so the length element $\sqrt{d\theta^\top F d\theta}$ and its integral agree (up to the midpoint rule's discretization error). Euclidean length has no such correction, so it depends on the description. the logit stretches the ends of the family to infinity, but its Fisher $p(1 - p)$ shrinks just as fast, so the integral converges: in $(\sqrt p, \sqrt{1 - p})$ coordinates the family is a quarter circle of radius 2, length $\pi$. with that picture, $(0.9, 0.99)$ is further apart (0.443) than $(0.5, 0.6)$ (0.201): near certainty, a small change in $p$ is a big change in what you'd observe.
