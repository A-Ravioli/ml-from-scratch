# the fisher metric

## why this matters

take a trained classifier and nudge its parameters by a small step of Euclidean length 0.05. how much do its predictions change? the honest answer is "it depends which way you step", and the dependence is extreme. in this topic's lab, 200 random steps of exactly the same length change the predictions by amounts that differ by a factor of nearly 100, and one direction changes the parameters without changing the predictions at all, however far you go. the parameter vector is a *description* of the model, and the Euclidean distance between two descriptions says little about the distance between the models they describe.

this matters every time you take a gradient step. "steepest descent" means "the direction that decreases the loss fastest per unit of step length", and the answer depends on how you measure length. gradient descent measures it in raw parameters, so its behaviour changes if you rescale a layer or switch from $\sigma$ to $\log\sigma$, although the set of models you can represent stays the same. information geometry measures length by how *distinguishable* two distributions are, and that measure is the **Fisher information metric**. it's the same object you met in chapter 05 as the curvature of the log-likelihood and the bound on estimator variance; here you'll treat it as a ruler. topic 19 turns the ruler into an optimizer (natural gradient), and topic 20 makes that optimizer cheap enough for real networks (K-FAC).

## warm-up: predict first

1. a coin's bias changes from 0.50 to 0.51, or from 0.98 to 0.99. which change is easier to detect from 1,000 flips?
2. you add 1 to every bias of a softmax classifier's output layer. how much do its predictions change?
3. KL divergence isn't symmetric. for a *small* change of parameters, does $\mathrm{KL}(p\,\|\,q)$ differ much from $\mathrm{KL}(q\,\|\,p)$?
4. the average of $\nabla\log p(y_n\mid x_n)\nabla\log p(y_n\mid x_n)^\top$ over a training set is often called "the Fisher". is it?
5. the logit of a coin runs from $-\infty$ to $+\infty$. is the "distance" from a fair coin to a two-headed coin infinite?

## prerequisites check

- [ ] you know the score, the Fisher information and its two definitions (→ ch 05 · topic 06)
- [ ] you know KL divergence and can expand a function to second order (→ ch 05 · topic 05, ch 03 · topic 03)
- [ ] you can get gradients and Hessians from autograd, and know what a Jacobian is (→ ch 14, ch 02 · topic 07)
- [ ] you know softmax classifiers and cross-entropy (→ ch 08 · topic 05)

## core ideas

### idea 1: distributions are points, and the Fisher is the ruler

a parametric model $\{p_\theta\}$ is a set of distributions labelled by coordinates $\theta \in \mathbb R^P$. information geometry treats this set as a **statistical manifold**: a curved space whose points are distributions, with $\theta$ as one of many possible coordinate systems on it. to measure lengths on a curved space you need a **metric**: a positive (semi)definite matrix $G(\theta)$ that says how long a small displacement $d\theta$ is, $ds^2 = d\theta^\top G(\theta)\,d\theta$. Euclidean geometry is $G = I$.

which metric? ask how easily you could *tell apart* $p_\theta$ and $p_{\theta + d\theta}$ from samples. the natural measure is the KL divergence, and expanding it for a small step gives (idea 2)

$$\mathrm{KL}(p_\theta\,\|\,p_{\theta + d\theta}) = \tfrac12\,d\theta^\top F(\theta)\,d\theta + O(\lVert d\theta\rVert^3),\qquad F(\theta) = \mathbb E_{x\sim p_\theta}\big[s(x)s(x)^\top\big],\quad s(x) = \nabla_\theta\log p_\theta(x).$$

the Fisher information $F(\theta)$ is the metric: $ds^2 = d\theta^\top F d\theta$ is (twice) the KL divergence of an infinitesimal step. this choice isn't arbitrary. Rao (1945) proposed it, and Chentsov (1982) proved that, up to a constant factor, it's the *only* metric on families of distributions that is unchanged by sufficient statistics (relabelling or merging outcomes in ways that lose no information). when people say "distance between distributions" locally, they mean this.

two models you'll use constantly have closed forms.

**the categorical, in logits.** $p = \operatorname{softmax}(z)$, $z \in \mathbb R^K$. the score of outcome $y$ is $\nabla_z\log p_y = e_y - p$ (on paper 1), and

$$F(z) = \mathbb E_y\big[(e_y - p)(e_y - p)^\top\big] = \operatorname{diag}(p) - pp^\top.$$

it's singular: $F\mathbf 1 = p - p(p^\top\mathbf 1) = 0$. adding a constant to every logit doesn't change the distribution (warm-up 2), so that direction has length zero. the Fisher metric is happy to be only semidefinite: it measures the model, and redundant coordinates show up as null directions.

**the Gaussian, in mean and log-std.** $x \sim \mathcal N(\mu, \sigma^2)$ with $\sigma = e^{\ell}$:

$$F(\mu, \ell) = \begin{pmatrix}1/\sigma^2 & 0\\0 & 2\end{pmatrix}.$$

moving the mean by $d\mu$ costs $d\mu/\sigma$ (in units of the current spread); changing $\log\sigma$ costs the same everywhere. that's why policy networks parameterize $\log\sigma$: the geometry is flat in that coordinate.

> **spec.** `categorical_fisher(logits)`: logits `(..., K)` → `(..., K, K)`, the matrix $\operatorname{diag}(p) - pp^\top$ for each row. `gaussian_fisher(mean, log_std)`: scalars (floats or 0-d tensors) → a `(2, 2)` float64 tensor, the Fisher matrix in $(\mu, \ell)$ coordinates.

> **▶ build: stage 1.** implement `categorical_fisher` and `gaussian_fisher`.

### idea 2: two definitions, and KL in the small

chapter 05 gave you two formulas for the Fisher information, the covariance of the score and the expected negative Hessian of the log-likelihood:

$$F(\theta) = \mathbb E_{p_\theta}\big[s s^\top\big] = -\mathbb E_{p_\theta}\big[\nabla^2_\theta\log p_\theta(x)\big].$$

they agree because the score has mean zero, $\mathbb E[s] = \sum_x\nabla p_\theta(x) = \nabla 1 = 0$, and differentiating that identity once more swaps one form for the other. note that both expectations are under the model's **own** distribution $p_\theta$, not under data. that detail is the subject of idea 3.

now the claim that made $F$ a metric. write $D(\theta) = \mathrm{KL}(p_{\theta_0}\,\|\,p_\theta) = \mathbb E_{p_{\theta_0}}[\log p_{\theta_0} - \log p_\theta]$. at $\theta = \theta_0$ it is 0, its gradient $-\mathbb E_{p_{\theta_0}}[s] = 0$, and its Hessian is $-\mathbb E_{p_{\theta_0}}[\nabla^2\log p_\theta]\big|_{\theta_0} = F(\theta_0)$ (on paper 3). so near $\theta_0$ the KL is a quadratic bowl whose shape is the Fisher. the same is true of the reverse divergence $\mathrm{KL}(p_\theta\,\|\,p_{\theta_0})$: the two directions agree to second order and differ at third (warm-up 3). the asymmetry of KL is a large-step phenomenon.

this gives you two ways to compute $F$ for any small model, both with autograd. **by enumeration**: if $x$ takes finitely many values, sum $p_\theta(x)\,s(x)s(x)^\top$ over them. **as a KL Hessian**: if you can write the KL in closed form, differentiate it twice. the checks use each one to test the other.

> **🔮 predict:** a model on outcomes $\{0, 1, \ldots, 6\}$ is Binomial$(6, \sigma(\theta))$ with a logit parameter $\theta$. what is its Fisher information at $\theta = 0$? (it's a number you can get from idea 1's Bernoulli without any sums.)

> **spec.** `fisher_exact(log_prob, theta, support)`: `log_prob(theta, x)` returns a 0-d tensor $\log p_\theta(x)$ differentiable in `theta` (a 1-D tensor of $P$ parameters); `support` is an iterable over **every** outcome. returns $\sum_x p_\theta(x)\,s(x)s(x)^\top$, `(P, P)`. work on a detached copy of `theta`. `kl_hessian(kl, theta0)`: `kl(theta0, theta)` returns a 0-d tensor $\mathrm{KL}(p_{\theta_0}\,\|\,p_\theta)$. returns the Hessian of $\theta \mapsto$ `kl(theta0, theta)` at $\theta = \theta_0$, `(P, P)`.

> **▶ build: stage 2.** implement `fisher_exact` and `kl_hessian`.

### idea 3: the Fisher of a network, and its impostor

a classifier is a *conditional* model $p_\theta(y\mid x)$; the inputs come from data and aren't modelled. its Fisher averages the conditional Fisher over the inputs, with the label drawn from the **model**:

$$F(\theta) = \frac1N\sum_{n=1}^N\mathbb E_{y\sim p_\theta(\cdot\mid x_n)}\big[g_{n,y}\,g_{n,y}^\top\big],\qquad g_{n,y} = \nabla_\theta\log p_\theta(y\mid x_n).$$

for a softmax output with logits $z_n = f(x_n; \theta)$, the chain rule gives $g_{n,y} = J_n^\top(e_y - p_n)$ with $J_n = \partial z_n/\partial\theta$ the $K\times P$ Jacobian of the logits, so

$$F(\theta) = \frac1N\sum_n J_n^\top\big(\operatorname{diag}(p_n) - p_np_n^\top\big)J_n.$$

read it as idea 1's categorical Fisher, *pulled back* through the network (idea 4 makes "pulled back" precise). this is also the **generalized Gauss–Newton** matrix of the cross-entropy loss (Schraudolph, 2002; Martens, 2020): for a loss that is convex in the outputs, the GGN keeps the curvature of the loss and drops the curvature of the network. the Fisher is positive semidefinite by construction, unlike the Hessian of a nonconvex loss, which is a big part of why it makes a good preconditioner. for softmax regression ($z = Wx + b$) the network is linear, nothing is dropped, and $F$ equals the Hessian of the mean cross-entropy, whatever the labels are.

now the impostor. put the **observed** labels in place of the expectation and you get the **empirical Fisher**

$$\hat F_{\text{emp}}(\theta) = \frac1N\sum_n g_{n,y_n}g_{n,y_n}^\top,$$

the uncentred covariance of the per-example gradients. it's cheap (you already have those gradients), and it's what many methods that claim to approximate the Fisher actually compute; Adam's second-moment estimate is a diagonal, minibatch, running-average cousin of it. Kunstner, Balles & Hennig (2019) catalogue why that's risky. $\hat F_{\text{emp}}$ depends on the labels and on how wrong the model is; it has rank at most $N$ where $F$ has rank up to $N(K - 1)$; and $\mathbb E[\hat F_{\text{emp}}] = F$ needs two things at once: the labels must really come from $p_\theta(y\mid x)$ (a well-specified model, at its true parameter), and you must average over enough data. far from that, the empirical Fisher mostly measures the size of the gradient, not the curvature. the lab measures how far apart the two are.

> **spec.** `true_fisher(model, X)`: `model` is a `torch.nn.Module` mapping `(N, d)` → logits `(N, K)`; `X` is `(N, d)`. returns $F(\theta)$ above, `(P, P)`, with parameters flattened in `model.parameters()` order (each tensor `.reshape(-1)`, concatenated). `empirical_fisher(model, X, y)`: the same layout, with the observed integer labels `y` `(N,)`. neither may change the model's parameters. sizes are small (tens of examples, at most a few hundred parameters), so per-example loops are fine.

> **▶ build: stage 3.** implement `true_fisher` and `empirical_fisher`. (a `logit_jacobian` helper returning the `(N, K, P)` Jacobians is a suggestion, not a requirement.)

### idea 4: changing coordinates, and lengths that don't care

a manifold doesn't come with preferred coordinates. you can describe a coin by $p$, by its logit $t$, or by $s = \sqrt p$; a Gaussian by $(\mu, \sigma)$ or $(\mu, \log\sigma)$. a geometric quantity should not care which you pick. how does the Fisher matrix change? if $\theta = \theta(\psi)$ with Jacobian $J = \partial\theta/\partial\psi$, the score transforms by the chain rule, $s_\psi = J^\top s_\theta$, so

$$F_\psi = \mathbb E\big[J^\top s_\theta s_\theta^\top J\big] = J^\top F_\theta\,J.$$

this is the **pullback** of the metric (on paper 4). the matrix entries change; the length element doesn't: a displacement $d\psi$ moves $\theta$ by $d\theta = J\,d\psi$, and $d\psi^\top F_\psi d\psi = d\theta^\top F_\theta d\theta$. idea 3 was a pullback too, from the logits to the network's weights.

lengths of curves are built from length elements. a path $\theta(\tau)$ from one distribution to another has Fisher–Rao length

$$L = \int\sqrt{\dot\theta(\tau)^\top F(\theta(\tau))\,\dot\theta(\tau)}\;d\tau \approx \sum_t\sqrt{\Delta_t^\top F(m_t)\,\Delta_t},$$

with $\Delta_t$ the steps between consecutive points and $m_t$ their midpoints. the same curve described in any other coordinates has the same length, and the length doesn't depend on how fast you traverse it. the shortest such path between two distributions is a **geodesic**, and its length is the Fisher–Rao distance. for the Bernoulli family, write $p = \sin^2\varphi$: then $F_\varphi = 4$ (on paper 4), the family is a quarter circle of radius 2, and the distance from Bernoulli($p_0$) to Bernoulli($p_1$) is

$$d(p_0, p_1) = 2\,\big|\arcsin\sqrt{p_1} - \arcsin\sqrt{p_0}\big|.$$

the whole family, from $p = 0$ to $p = 1$, has length $\pi$ (warm-up 5). near $p = 0$ or 1, small changes in $p$ are easy to detect (a coin with $p = 0.98$ versus 0.99 shows up as a doubling of the tails count), so they count as long; near $p = \tfrac12$ the same change in $p$ is short (warm-up 1).

> **🔮 predict:** at fixed mean, a Gaussian's $\sigma$ grows from 1 to 8. in $(\mu, \sigma)$ coordinates the Fisher is $\operatorname{diag}(1/\sigma^2, 2/\sigma^2)$. what's the length of this path, and would it change if the points along the path were spaced unevenly?

> **spec.** `pullback_fisher(fisher, jacobian)`: `fisher` `(P_θ, P_θ)`, `jacobian` $=\partial\theta/\partial\psi$ `(P_θ, P_ψ)` → $J^\top FJ$, `(P_ψ, P_ψ)`. `fisher_rao_length(fisher_fn, path)`: `path` is `(T, P)` with $T \ge 2$; `fisher_fn(theta)` takes a 1-D tensor `(P,)` and returns `(P, P)`. returns the midpoint-rule sum above as a Python float.

> **▶ build: stage 4.** implement `pullback_fisher` and `fisher_rao_length`.

## worked example

**a categorical.** logits $z = (0, 0, \log 2)$ give $p = (\tfrac14, \tfrac14, \tfrac12)$, so

$$F = \operatorname{diag}(p) - pp^\top = \frac1{16}\begin{pmatrix}3 & -1 & -2\\-1 & 3 & -2\\-2 & -2 & 4\end{pmatrix}.$$

each row sums to 0 (the softmax-shift null direction). the step $dz = (0.01, 0, 0)$ has $\tfrac12 dz^\top F dz = \tfrac12\cdot\tfrac3{16}\cdot10^{-4} = 9.375\cdot10^{-6}$, and the exact KL from $p$ to $\operatorname{softmax}(z + dz)$ is $9.39\cdot10^{-6}$: the quadratic form is off by 0.2%, the size of the cubic term at this step.

**a Gaussian.** $\mathcal N(1, 2^2)$ in $(\mu, \ell)$ coordinates: $F = \operatorname{diag}(1/4, 2)$. in $(\mu, \sigma)$ coordinates it's $\operatorname{diag}(1/\sigma^2, 2/\sigma^2) = \operatorname{diag}(1/4, 1/2)$; pulling that back through $\sigma = e^\ell$, with $J = \operatorname{diag}(1, \sigma) = \operatorname{diag}(1, 2)$, gives $J^\top\operatorname{diag}(\tfrac14, \tfrac12)J = \operatorname{diag}(\tfrac14, 4\cdot\tfrac12) = \operatorname{diag}(\tfrac14, 2)$, as it should.

**a pullback by hand.** $F = \operatorname{diag}(2, 3)$, $J = \begin{pmatrix}1 & 1\\0 & 2\end{pmatrix}$: $J^\top FJ = \begin{pmatrix}2 & 2\\2 & 14\end{pmatrix}$.

**a length.** from Bernoulli(0.1) to Bernoulli(0.9): $2(\arcsin\sqrt{0.9} - \arcsin\sqrt{0.1}) = 2(1.2490 - 0.3218) = 1.8546$. the logit runs from $-2.197$ to $2.197$ (Euclidean length 4.39) and $p$ from 0.1 to 0.9 (length 0.8), yet with their own Fisher metrics both descriptions measure 1.8546.

your code must reproduce these numbers.

## on paper

1. **the categorical Fisher.** show that $\nabla_z\log\operatorname{softmax}(z)_y = e_y - p$ and that $\mathbb E_{y\sim p}[(e_y - p)(e_y - p)^\top] = \operatorname{diag}(p) - pp^\top$.

<details><summary>solution</summary>

$\log p_y = z_y - \log\sum_k e^{z_k}$, so $\partial\log p_y/\partial z_j = \mathbb 1[j = y] - p_j$. then $\mathbb E[(e_y - p)(e_y - p)^\top] = \mathbb E[e_ye_y^\top] - \mathbb E[e_y]p^\top - p\,\mathbb E[e_y]^\top + pp^\top = \operatorname{diag}(p) - pp^\top - pp^\top + pp^\top$. it's also the Hessian of $\log\sum_k e^{z_k}$, the categorical's log-partition function, as chapter 05's exponential-family result says it must be.

</details>

2. **the Gaussian Fisher in $(\mu, \log\sigma)$.** with $z = (x - \mu)/\sigma$, show the score is $(z/\sigma,\ z^2 - 1)$ and derive $F = \operatorname{diag}(1/\sigma^2, 2)$.

<details><summary>solution</summary>

$\log p = -\ell - \tfrac12(x - \mu)^2e^{-2\ell} + c$. $\partial_\mu = (x - \mu)/\sigma^2 = z/\sigma$; $\partial_\ell = -1 + (x - \mu)^2/\sigma^2 = z^2 - 1$. with $z \sim \mathcal N(0, 1)$: $\mathbb E[z^2]/\sigma^2 = 1/\sigma^2$; $\mathbb E[z(z^2 - 1)]/\sigma = 0$ (odd moments vanish); $\mathbb E[(z^2 - 1)^2] = \mathbb E[z^4] - 2\mathbb E[z^2] + 1 = 3 - 2 + 1 = 2$.

</details>

3. **KL is locally the Fisher quadratic form.** show that $D(\theta) = \mathrm{KL}(p_{\theta_0}\,\|\,p_\theta)$ has $D(\theta_0) = 0$, $\nabla D(\theta_0) = 0$ and $\nabla^2D(\theta_0) = F(\theta_0)$. then show the reverse divergence $R(\theta) = \mathrm{KL}(p_\theta\,\|\,p_{\theta_0})$ has the same three properties.

<details><summary>solution</summary>

$\nabla D = -\mathbb E_{\theta_0}[\nabla\log p_\theta]$, zero at $\theta_0$ because the score has mean zero. $\nabla^2D = -\mathbb E_{\theta_0}[\nabla^2\log p_\theta]$, which at $\theta_0$ is the second form of $F$. for $R(\theta) = \sum_x p_\theta(\log p_\theta - \log p_{\theta_0})$: $\nabla R = \sum_x\nabla p_\theta(\log p_\theta - \log p_{\theta_0}) + \sum_x\nabla p_\theta$; the second sum is $\nabla 1 = 0$ and the first vanishes at $\theta_0$. differentiating again, every term containing $(\log p_\theta - \log p_{\theta_0})$ vanishes at $\theta_0$ and what's left is $\sum_x\nabla p\,\nabla\log p^\top = \mathbb E[ss^\top] = F$. the two divergences first differ in their third derivatives.

</details>

4. **pullbacks and the Bernoulli circle.** (a) prove $F_\psi = J^\top F_\theta J$. (b) for the Bernoulli in $p$, $F = 1/(p(1 - p))$; pull it back through $p = \sin^2\varphi$ and integrate to get the distance formula of idea 4.

<details><summary>solution</summary>

(a) $\log p_{\theta(\psi)}(x)$ has gradient $J^\top s_\theta(x)$ in $\psi$, so $F_\psi = \mathbb E[J^\top s_\theta s_\theta^\top J] = J^\top F_\theta J$ ($J$ doesn't depend on $x$). (b) $dp/d\varphi = 2\sin\varphi\cos\varphi$, so $F_\varphi = 4\sin^2\varphi\cos^2\varphi/(\sin^2\varphi\cos^2\varphi) = 4$: a constant metric, so the length is $\int 2\,d\varphi = 2|\varphi_1 - \varphi_0|$ with $\varphi = \arcsin\sqrt p$. as $p$ runs over $[0, 1]$, $\varphi$ runs over $[0, \pi/2]$, so the total length is $\pi$. equivalently, $(\sqrt p, \sqrt{1 - p})$ lies on the unit circle, and the Fisher metric is 4 times the Euclidean metric on it: a circle of radius 2.

</details>

## experiment

`lab.py` investigates, with your Fisher code:

1. how much a fixed-size parameter step changes a classifier's predictions, in different directions, and how well $\tfrac12\Delta^\top F\Delta$ predicts the KL as the step grows.
2. how close the empirical Fisher is to the true Fisher, at initialization and at the maximum-likelihood fit, for well-specified and misspecified models.
3. the Fisher–Rao length of a path between two coins, in three coordinate systems.

## check your understanding

1. why is the Fisher metric only positive *semi*definite for a softmax classifier, and is that a problem for using it as a metric?

<details><summary>answer</summary>

any parameter change that leaves every $p_\theta(\cdot\mid x)$ unchanged has length zero. for softmax, shifting all logits by the same amount is one (17 dimensions for a 16-unit last layer: each class row moves by the same vector, and each bias by the same number). networks have other symmetries too. the metric is degenerate on the parameters but not on the distributions: it measures the model, and the degenerate directions are exactly the redundant ones. for optimization it means $F$ must be damped before it's inverted (topic 19).

</details>

2. (warm-up 1) which change of a coin's bias is easier to detect, 0.50 → 0.51 or 0.98 → 0.99?

<details><summary>answer</summary>

the second. $F(p) = 1/(p(1 - p))$ is 4 at $p = 0.5$ and about 68 at $p = 0.985$, so the second step is about $\sqrt{68/4} \approx 4.1$ times longer (the exact Fisher–Rao distances are 0.0200 and 0.0835). intuitively, the tails count halves (from about 20 to about 10 per 1,000 flips), which is far easier to see than 500 heads becoming 510.

</details>

3. (the predict callout in idea 2.) what is the Fisher of Binomial$(6, \sigma(\theta))$ at $\theta = 0$?

<details><summary>answer</summary>

$6\cdot\tfrac12\cdot\tfrac12 = 1.5$. the binomial is 6 independent Bernoulli flips, Fisher information adds over independent observations, and the Bernoulli in its logit has Fisher $p(1 - p)$ (idea 1's categorical with $K = 2$, restricted to one logit).

</details>

4. why is the Fisher of a network computed with labels sampled from the model rather than with the training labels?

<details><summary>answer</summary>

because it's the metric of the model's family of distributions: it measures how distinguishable $p_\theta$ is from $p_{\theta + d\theta}$, which involves only the model, not the data. the KL between two models of $y\mid x$ is an expectation under the model. the training labels enter the *gradient* of the loss, not the geometry. using them instead gives the empirical Fisher, which coincides with $F$ only in expectation, and only for a well-specified model at its true parameter.

</details>

5. the Fisher of a softmax-regression model equals the Hessian of its cross-entropy loss. does that hold for an MLP?

<details><summary>answer</summary>

no. the Hessian of the loss through a nonlinear network has an extra term, $\sum_n\sum_k(p_{nk} - \mathbb 1[y_n = k])\nabla^2_\theta z_{nk}$, the residuals times the curvature of the network. the Fisher (= generalized Gauss–Newton) drops it. the dropped term can be indefinite, so the Hessian can have negative eigenvalues while $F$ never does. at a perfect fit (residuals zero) the two agree.

</details>

6. you compute the Fisher–Rao length of a path in two coordinate systems and get different answers. name two likely bugs.

<details><summary>answer</summary>

(1) the Fisher in the second coordinates isn't the pullback (for example, you used $J F J^\top$, or forgot the Jacobian entirely and reused the first coordinates' formula). (2) the two paths aren't the same curve: transforming the endpoints and drawing a straight line in each coordinate system gives different curves, which genuinely have different lengths. the length is invariant to *re-describing* a curve, not to choosing a new one.

</details>

7. (the predict callout in idea 4.) how long is the path that grows $\sigma$ from 1 to 8?

<details><summary>answer</summary>

$\int_1^8\sqrt{2}/\sigma\,d\sigma = \sqrt2\log 8 \approx 2.94$. it's $\sqrt 2$ times the change in $\log\sigma$, the coordinate in which the metric is constant. uneven spacing doesn't change it (up to discretization error): a length is a property of the curve, not of its parameterization.

</details>

## common bugs

- **diag(p) without $-pp^\top$.** symptom: a full-rank "Fisher" whose rows don't sum to zero; KL predictions off by 10–50%.
- **the Fisher for $\sigma$ when you parameterize $\log\sigma$.** symptom: the second diagonal entry is $2/\sigma^2$ instead of 2.
- **forgetting to weight by $p_\theta(x)$ when enumerating.** symptom: a matrix that's too large by roughly the number of outcomes, and doesn't match the KL Hessian.
- **training labels in `true_fisher`.** symptom: it matches `empirical_fisher` exactly; its rank is at most $N$.
- **parameter order.** symptom: blocks of your Fisher matrix are right but in the wrong places. flatten in `model.parameters()` order (weight before bias, layer by layer), each with `.reshape(-1)`.
- **summing $\Delta^\top F\Delta$ instead of its square root.** symptom: lengths that shrink as you add more points to the path.

## going deeper

- Amari (1998), "Natural gradient works efficiently in learning", *Neural Computation* 10(2). [doi:10.1162/089976698300017746](https://doi.org/10.1162/089976698300017746)
- Amari & Nagaoka (2000), *Methods of Information Geometry*, AMS/Oxford. chapters 2–3 for the Fisher metric and Chentsov's theorem.
- Rao (1945), "Information and the accuracy attainable in the estimation of statistical parameters", *Bulletin of the Calcutta Mathematical Society* 37. the Fisher information as a Riemannian metric, and geodesic distance.
- Martens (2020), "New insights and perspectives on the natural gradient method", *JMLR* 21(146). [arXiv:1412.1193](https://arxiv.org/abs/1412.1193). the Fisher, the generalized Gauss–Newton matrix, and the empirical Fisher, carefully distinguished.
- Kunstner, Balles & Hennig (2019), "Limitations of the empirical Fisher approximation for natural gradient descent", *NeurIPS*. [arXiv:1905.12558](https://arxiv.org/abs/1905.12558)
- Schraudolph (2002), "Fast curvature matrix-vector products for second-order gradient descent", *Neural Computation* 14(7). [doi:10.1162/08997660260028683](https://doi.org/10.1162/08997660260028683)
- Nielsen (2020), "An elementary introduction to information geometry", *Entropy* 22(10). [arXiv:1808.08271](https://arxiv.org/abs/1808.08271)

## connections

- ← builds on: the score, Fisher information and exponential families (ch 05 · topic 06), KL divergence (ch 05 · topic 05), Taylor expansion (ch 03), Jacobians and autograd (ch 02 · topic 07, ch 14), softmax classifiers (ch 08 · topic 05). you used the KL Hessian once already, as TRPO's constraint (ch 24 · topic 04).
- → used in: natural gradient descent (topic 19) preconditions by $F^{-1}$; K-FAC (topic 20) approximates the network Fisher of idea 3 layer by layer; elastic weight consolidation (topic 13, in the continual-learning block) uses its diagonal as parameter importance.
