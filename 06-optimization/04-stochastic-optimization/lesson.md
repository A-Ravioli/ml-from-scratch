# stochastic optimization

## why this matters

every method so far computes the exact gradient of the loss at every step. for a model trained on a million examples, that's a million per-example gradients before you move once. stochastic gradient descent (SGD) moves after looking at a **minibatch** of 32: a noisy, unbiased estimate of the gradient at a 30,000th of the cost. it's noisy enough that it never quite settles, and cheap enough that it wins anyway. it's how essentially every neural network has been trained.

around that core idea grew a family: momentum, per-coordinate step sizes (Adagrad, RMSProp), their combination (Adam), and a fix for how Adam interacts with weight decay (AdamW). they look like a zoo of formulas, but they share one interface, and that interface is the most-reused code in this course. from chapter 08 on, every model trained by gradients (logistic regression, your NumPy neural network, the transformers) is trained by the `Optimizer` classes you write here, through `mlfs.optim`. so this topic is about two things: getting the update rules exactly right (they must match PyTorch's), and understanding what the noise does, when adaptivity helps, and when it quietly hurts.

## warm-up: predict first

1. SGD with a constant learning rate on a strongly convex problem. does the loss converge to the minimum?
2. you double the batch size. by what factor does the variance of the minibatch gradient shrink? what if the batch is the whole dataset?
3. Adam's first step, for a parameter whose gradient is $10^{-6}$. how far does it move with `lr=1e-3`?
4. you rotate the coordinate system of a problem by a random rotation. does SGD's behaviour change? does Adam's?
5. `opt.step()` writes `p.data = p.data - lr * g`. the model holds a reference to the old array. what does the model see after training?

## prerequisites check

- [ ] you've built `heavy_ball` and know how momentum accumulates a velocity (→ topic 03)
- [ ] you know gradient descent's rates and the role of $\kappa$ (→ topic 02)
- [ ] you know expectation, variance, and the variance of a mean of independent samples (→ ch 04 · topic 03)
- [ ] you can seed randomness with `as_rng` and write generators with `yield` (→ ch 00 · topics 05 and 01)
- [ ] you know the difference between rebinding a name and mutating an array in place (→ ch 00 · topic 02)

## core ideas

### idea 1: the parameter protocol, and minibatches

a training loop has three moving parts: parameters, gradients, and an update rule. `mlfs.optim` fixes how they talk to each other, once, for the whole course:

- a **parameter** is any object with two attributes: `.data`, a float array of weights, and `.grad`, an array of the same shape holding the current gradient, or `None` meaning "no gradient this step". `Param` is the simplest such object (it stores a float64 *copy* of what you give it). chapter 14's `Tensor` will be another.
- an **optimizer** is constructed with an iterable of parameters and a learning rate, `opt = SGD(params, lr=0.1)`. it keeps them in a list `opt.params`, its learning rate in a plain attribute `opt.lr` (so a scheduler in chapter 15 can change it between steps), and any per-parameter state (momentum buffers, running averages) in `opt.state[i]`, keyed by the parameter's position in the list.
- `opt.zero_grad()` sets every `.grad` to `None`. `opt.step()` updates every parameter whose `.grad` isn't `None`, **in place**: `p.data -= ...`, never `p.data = ...`. a model holds references to its weight arrays; rebinding `p.data` to a new array leaves the model holding the old one, and training silently does nothing (warm-up 5). and `step()` never modifies `p.grad`, which belongs to whoever computed it.

the loop that uses them is always the same: for each minibatch, `zero_grad()`, compute the loss and write gradients into `.grad`, `step()`.

**minibatches.** the loss is an average, $f(w) = \frac1n\sum_{i=1}^n\ell_i(w)$. a minibatch $B$ of $b$ examples drawn uniformly gives the estimate $g_B = \frac1b\sum_{i\in B}\nabla\ell_i(w)$, which is **unbiased**: $\mathbb E[g_B] = \nabla f(w)$. in practice you don't sample batches independently; you shuffle the data once per **epoch** and cut the permutation into consecutive chunks, so every example is used exactly once per epoch (sampling without replacement). if $b$ doesn't divide $n$, the last batch is smaller, or you drop it (`drop_last`). a generator is the natural way to write it: it yields one batch of indices at a time.

> **▶ build: stage 1.** implement `Param`, the `Optimizer` base class (`__init__` and `zero_grad`; `step` is for subclasses), and `minibatch_indices(n, batch_size, *, shuffle, drop_last, rng)`, a generator of index arrays for one epoch, shuffled with `as_rng(rng)`.

### idea 2: SGD, momentum, and why the learning rate must decay

**SGD** is gradient descent with $g_B$ in place of $\nabla f$: $w \leftarrow w - \eta g_B$. the noise changes the convergence story completely. near the minimum the true gradient is tiny, but $g_B$ isn't: its noise has a variance $\sigma^2/b$ that doesn't vanish at $w^*$. so with a constant $\eta$, SGD doesn't converge; it bounces around the minimum in a cloud whose size is proportional to $\eta$ (idea 5 makes this exact; warm-up 1). to converge, the learning rate must decay: the classical Robbins–Monro conditions are $\sum_k\eta_k = \infty$ (you can still travel any distance) and $\sum_k\eta_k^2 < \infty$ (the accumulated noise is finite). with $\eta_k \propto 1/k$ on a strongly convex problem, $\mathbb E[f(w_k)] - f^* = O(1/k)$; for convex problems with $\eta_k \propto 1/\sqrt k$ and averaging, $O(1/\sqrt k)$. both are *sublinear*: SGD trades the linear rate of gradient descent for steps that are $n/b$ times cheaper, and for large datasets that's a bargain, because you rarely need the loss to 10 digits.

**momentum** in PyTorch's form keeps a buffer $v$ per parameter:

$$v \leftarrow \mu v + g,\qquad w \leftarrow w - \eta v,$$

with $v = g$ on the parameter's first step. this is the heavy ball of topic 03 with the learning rate pulled out ($\eta v$ is the step). the **Nesterov** variant steps with $g + \mu v$ instead of $v$: the same look-ahead correction, written without an explicit look-ahead point. with noisy gradients momentum has a second job: averaging $1/(1 - \mu)$ recent gradients also averages out some of their noise.

**weight decay** in SGD is L2 regularization: add $\lambda w$ to the gradient, which minimizes $f(w) + \frac\lambda2\|w\|^2$. compute it as a new array, $g = $ `p.grad + wd * p.data`; writing `p.grad += ...` would corrupt the caller's gradient.

> **🔮 predict:** SGD with momentum $0.9$ and $\eta = 0.1$, starting at 0, with a constant gradient of 1. where is $w$ after three steps?

> **▶ build: stage 2.** implement `SGD(params, lr, momentum, nesterov, weight_decay)`: the momentum buffer lives in `self.state[i]`, parameters with `grad is None` are skipped (their buffers don't advance), and `nesterov=True` without momentum raises `ValueError`.

### idea 3: one step size per coordinate: Adagrad and RMSProp

topic 02's wall was the condition number: one learning rate for directions with very different curvatures. if the badly scaled directions happen to be the **coordinate axes** (features measured in different units, embeddings of rare and common words), there's a cheap fix: give every coordinate its own step size, smaller where gradients are large.

**Adagrad** divides each coordinate's step by the root of its accumulated squared gradients:

$$s \leftarrow s + g\odot g,\qquad w \leftarrow w - \eta\,\frac{g}{\sqrt s + \epsilon}$$

(elementwise). a coordinate that has seen large gradients takes small steps; a rarely active one (a rare feature) keeps a large step for the few times it gets a gradient, which is why Adagrad was designed for sparse problems. its first step is $\eta\,g/|g| = \eta\,\text{sign}(g)$ in every coordinate, whatever the gradient's scale. but $s$ only grows, so the steps shrink like $1/\sqrt k$ forever: fine for convex problems, too timid for long nonconvex training.

**RMSProp** replaces the sum with an exponential moving average, so old gradients are forgotten:

$$v \leftarrow \alpha v + (1 - \alpha)\,g\odot g,\qquad w \leftarrow w - \eta\,\frac{g}{\sqrt v + \epsilon}.$$

with $v$ starting at 0 and $\alpha = 0.99$, the first update has $v = 0.01g^2$ and moves $\eta g/(0.1|g|) = 10\eta$: an unintended large first step, because the average hasn't warmed up. Adam fixes exactly that.

this per-coordinate scaling is a **diagonal preconditioner**: it approximates rescaling each axis by an estimate of its curvature. that works when the ill-conditioning is aligned with the axes. a rotation mixes the coordinates, and a diagonal rescaling can no longer undo it (warm-up 4; lab experiment 1).

> **▶ build: stage 3.** implement `Adagrad(params, lr, eps)` and `RMSProp(params, lr, alpha, eps)`, with their running sums in `self.state[i]`.

### idea 4: Adam and AdamW

**Adam** combines momentum (a moving average $m$ of gradients) with RMSProp's scaling (a moving average $v$ of squared gradients), and corrects both for starting at zero:

$$m \leftarrow \beta_1m + (1 - \beta_1)g,\quad v \leftarrow \beta_2v + (1 - \beta_2)g\odot g,\quad \hat m = \frac{m}{1 - \beta_1^t},\quad \hat v = \frac{v}{1 - \beta_2^t},\quad w \leftarrow w - \eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon}.$$

why the correction: $m_t = (1 - \beta_1)\sum_{j\le t}\beta_1^{t-j}g_j$, and if the gradients have a constant mean, $\mathbb E[m_t] = (1 - \beta_1^t)\,\mathbb E[g]$ (on paper 2). dividing by $1 - \beta_1^t$ removes the bias towards zero in early steps. the step counter $t$ counts the steps where *this* parameter had a gradient, so it lives in `self.state[i]`. the result is a step of size roughly $\eta$ per coordinate, in the direction of the recent average gradient, normalized by its recent magnitude: the first step is exactly $\eta\,\text{sign}(g)$ (warm-up 3: $10^{-3}$, even for a gradient of $10^{-6}$). Adam is close to scale-invariant per coordinate, which is why one learning rate ($10^{-3}$ or $3\times10^{-4}$) works for so many problems.

**weight decay, coupled versus decoupled.** `Adam(weight_decay=λ)` adds $\lambda w$ to the gradient before everything else (coupled L2, like SGD). but then the decay term is divided by $\sqrt{\hat v}$ along with the rest: weights with large gradients get *less* regularization per step, weights with small gradients more. Loshchilov & Hutter (2019) proposed **decoupled** weight decay, AdamW: shrink the weights directly, then do a plain Adam step on the loss gradient,

$$w \leftarrow (1 - \eta\lambda)\,w,\qquad\text{then the Adam update with } g = \nabla f.$$

for plain SGD the two are the same thing ($w - \eta(g + \lambda w) = (1 - \eta\lambda)w - \eta g$), but not for Adam, and AdamW is what large models are trained with (chapter 27's `train_lm` uses it). on paper 4 shows that their fixed points differ.

> **🔮 predict:** two parameters start at 2. one gets gradient 0 and `Adam(lr=0.1, weight_decay=0.01)`; the other gets gradient 0 and `AdamW(lr=0.1, weight_decay=0.01)`. where is each after one step?

> **▶ build: stage 4.** implement `Adam(params, lr, betas, eps, weight_decay)` (coupled) and `AdamW(params, lr, betas, eps, weight_decay)` (decoupled), with per-parameter `step`, `exp_avg` and `exp_avg_sq` in `self.state[i]`.

### idea 5: how noisy is a minibatch, and where does SGD stall?

two formulas make the noise quantitative.

**the variance of a minibatch gradient.** let $g_1, \dots, g_n$ be the per-example gradients at some $w$, with mean $\bar g$ and spread $S = \frac1n\sum_i\|g_i - \bar g\|^2$. for a batch of $b$ drawn without replacement (on paper 1),

$$\mathbb E\,\|g_B - \bar g\|^2 = \frac{n - b}{n - 1}\cdot\frac{S}{b}.$$

the $S/b$ is the familiar "variance of a mean" (warm-up 2: double the batch, halve the variance); the factor $\frac{n - b}{n - 1}$ is the **finite-population correction**, which drives the noise to exactly 0 when $b = n$ (the full batch is the true gradient).

**the noise floor.** take a quadratic $f(x) = \sum_i\frac{a_i}2x_i^2$ and gradients with independent noise of variance $s_i$ along coordinate $i$. constant-step SGD is then $x_i \leftarrow (1 - \eta a_i)x_i - \eta\xi_i$, and its stationary second moment solves $v_i = (1 - \eta a_i)^2v_i + \eta^2s_i$ (on paper 3). so the expected excess loss settles at

$$\mathbb E[f(x) - f^*] = \sum_i\frac{a_i}{2}v_i = \sum_i\frac{\eta\,s_i}{2(2 - \eta a_i)} \approx \frac\eta4\sum_is_i\quad\text{for small }\eta a_i.$$

the floor is proportional to the learning rate and to the gradient noise, which is proportional to $1/b$. that's the quantitative version of "decay the learning rate", and it suggests that what matters is roughly $\eta/b$: double the batch and you can double the learning rate for the same noise level (the "linear scaling rule" used to train with huge batches, up to the point where $\eta a_i$ approaches the stability limit 2).

> **▶ build: stage 5.** implement `minibatch_gradient_variance(per_example_grads, batch_size)` and `sgd_noise_floor(lr, curvatures, noise_vars)`. a check runs your `SGD` on a noisy 400-dimensional quadratic and compares where it settles with your formula.

## worked example

**SGD with momentum.** $\eta = 0.1$, $\mu = 0.9$, constant gradient 1, from $w = 0$. the buffer is $v = 1$, then $0.9 + 1 = 1.9$, then $1.71 + 1 = 2.71$, so $w = -0.1, -0.29, -0.561$. with `nesterov=True` the steps use $g + \mu v$: $1 + 0.9 = 1.9$, then $1 + 0.9 \cdot 1.9 = 2.71$, so $w = -0.19, -0.461$. with `weight_decay=0.5` at $w = (2, -4)$ and gradient $(0, 1)$, the decayed gradient is $(1, -1)$ and one step with $\eta = 0.1$ gives $(1.9, -3.9)$.

**the first adaptive steps.** from $w = (1, 1)$ with gradient $(100, 0.01)$ and $\eta = 0.1$: Adagrad moves both coordinates by exactly $0.1$ (to $(0.9, 0.9)$), and with the same gradient again by $0.1/\sqrt2 = 0.0707$. RMSProp ($\eta = 0.01$, $\alpha = 0.99$) with gradient 3 has $v = 0.09$ and moves $0.01 \cdot 3/0.3 = 0.1$, ten times its learning rate. Adam ($\eta = 0.1$) with gradient $\pm s$ for any scale $s$ has $\hat m = g$, $\hat v = g^2$ after one step and moves by $0.1$ against the sign of the gradient.

**decay, coupled and decoupled.** $w = 2$, gradient 0, $\lambda = 0.01$, $\eta = 0.1$. Adam: $g = 0.02$, and its first step is $\eta\,\text{sign}(g)$: $w = 1.9$, a huge decay caused by a tiny penalty, because the adaptive scaling normalized it. AdamW: $w \leftarrow 2(1 - 0.001) = 1.998$, and the Adam part has $g = 0$: nothing more.

**minibatch variance.** four 1-D per-example gradients $1, 2, 3, 6$: $\bar g = 3$, $S = \frac14(4 + 1 + 0 + 9) = 3.5$. for $b = 2$ the formula gives $\frac{2}{3}\cdot\frac{3.5}{2} = 1.1667$. check by enumeration: the six pairs have means $1.5, 2, 3.5, 2.5, 4, 4.5$, with squared deviations from 3 of $2.25, 1, 0.25, 0.25, 1, 2.25$, average $7/6 = 1.1667$. ✓

## on paper

1. **minibatch variance.** let $B$ be a uniformly random subset of size $b$ of $\{1, \dots, n\}$, and $d_i = g_i - \bar g$ (so $\sum_id_i = 0$). show $\mathbb E\|g_B - \bar g\|^2 = \frac{n - b}{n - 1}\frac{S}{b}$. hint: $g_B - \bar g = \frac1b\sum_i\mathbb 1[i \in B]d_i$, with $P(i \in B) = \frac bn$ and $P(i, j \in B) = \frac{b(b - 1)}{n(n - 1)}$ for $i \ne j$.

<details><summary>solution</summary>

$\mathbb E\|g_B - \bar g\|^2 = \frac1{b^2}\Big(\sum_iP(i \in B)\|d_i\|^2 + \sum_{i\ne j}P(i, j \in B)\,d_i^\top d_j\Big)$. since $\sum_id_i = 0$, $\sum_{i\ne j}d_i^\top d_j = \|\sum_id_i\|^2 - \sum_i\|d_i\|^2 = -nS$. so the expression is $\frac1{b^2}\Big(\frac bn nS - \frac{b(b - 1)}{n(n - 1)}nS\Big) = \frac Sb\Big(1 - \frac{b - 1}{n - 1}\Big) = \frac Sb\cdot\frac{n - b}{n - 1}$.

</details>

2. **bias correction.** with $m_0 = 0$ and $m_t = \beta m_{t-1} + (1 - \beta)g_t$, show $m_t = (1 - \beta)\sum_{j=1}^t\beta^{t-j}g_j$ and, if every $g_j$ has mean $\gamma$, $\mathbb E[m_t] = (1 - \beta^t)\gamma$. why does this matter more for $v$ (with $\beta_2 = 0.999$) than for $m$ (with $\beta_1 = 0.9$)?

<details><summary>solution</summary>

unrolling the recursion gives the sum; its weights $(1 - \beta)\beta^{t-j}$ add up to $(1 - \beta)\frac{1 - \beta^t}{1 - \beta} = 1 - \beta^t$, so $\mathbb E[m_t] = (1 - \beta^t)\gamma$. the bias $\beta^t$ decays in about $1/(1 - \beta)$ steps: 10 for $\beta_1 = 0.9$, 1,000 for $\beta_2 = 0.999$. without the correction, $\hat v$ would be far too small for the first thousand steps, and steps far too large (RMSProp's first-step problem, made worse).

</details>

3. **the noise floor.** for $x_{k+1} = (1 - \eta a)x_k - \eta\xi_k$ with $\xi_k$ independent, mean 0, variance $s$, show that $v_k = \mathbb E[x_k^2]$ converges to $\frac{\eta s}{a(2 - \eta a)}$ when $0 < \eta a < 2$, and hence $\mathbb E[\tfrac a2x^2] \to \frac{\eta s}{2(2 - \eta a)}$.

<details><summary>solution</summary>

$x_k$ depends only on $\xi_0, \dots, \xi_{k-1}$, so it's independent of $\xi_k$ and the cross term vanishes: $v_{k+1} = (1 - \eta a)^2v_k + \eta^2s$. this is a linear recursion with factor $(1 - \eta a)^2 < 1$, converging to the fixed point $v = \frac{\eta^2s}{1 - (1 - \eta a)^2} = \frac{\eta^2s}{\eta a(2 - \eta a)} = \frac{\eta s}{a(2 - \eta a)}$. multiply by $a/2$.

</details>

4. **where Adam and AdamW stop.** consider a deterministic problem (exact gradients) and suppose each method converges to a point $w$ where its update is zero. show that for Adam with coupled decay, $\nabla f(w) + \lambda w = 0$ (the ridge stationarity condition). for AdamW, with $\epsilon$ negligible, show the fixed point satisfies $\lambda w_j = -\nabla_jf(w)/|\nabla_jf(w)|$ for each coordinate with a nonzero gradient. what does that say about AdamW's regularization?

<details><summary>solution</summary>

at a fixed point the moving averages equal their inputs: $\hat m = g$, $\hat v = g\odot g$. Adam's update is zero iff $\hat m = 0$, i.e. $g = \nabla f + \lambda w = 0$. AdamW's update is $\eta(\lambda w + \hat m/(\sqrt{\hat v} + \epsilon))$, with $g = \nabla f$, so the fixed point needs $\lambda w_j = -g_j/(|g_j| + \epsilon)$. with $\epsilon \to 0$ that's $\lambda w_j = -\text{sign}(g_j)$, which needs $|w_j| = 1/\lambda$, or else $g_j \approx 0$ (within $\epsilon$). so in the deterministic limit AdamW's decay doesn't minimize $f + \frac\lambda2\|w\|^2$: coordinates either sit at a stationary point of $f$ or are clipped near $\pm1/\lambda$. with noisy minibatch gradients, $\sqrt{\hat v}$ is dominated by the noise level and the picture changes again. the lesson is that "weight decay" and "L2 penalty" are different things once the optimizer is adaptive.

</details>

## experiment

`lab.py` investigates three things with your optimizers:

1. when Adam beats SGD with momentum, and when it doesn't: an ill-conditioned quadratic aligned with the axes, and the same quadratic randomly rotated.
2. the noise floor of constant-step SGD on a linear regression, for several learning rates and batch sizes, against your `sgd_noise_floor`, and what a decaying learning rate does.
3. an underdetermined regression (more weights than examples) that SGD and Adam both fit perfectly: do they find the same weights, and do they generalize equally well?

## check your understanding

1. why must `step()` update `p.data` in place, and never modify `p.grad`?

<details><summary>answer</summary>

the model (or a later chapter's `Tensor`) holds a reference to the weight array; `p.data = new_array` rebinds the optimizer's attribute but leaves every other reference pointing at the old, untrained weights. `p.grad` belongs to whoever computed it (a backward pass may accumulate into it, a logger may read it); an optimizer that edits it in place, e.g. `p.grad += wd * p.data`, corrupts it.

</details>

2. (the predict callout in idea 2.) SGD with momentum 0.9, constant gradient 1, $\eta = 0.1$: where after three steps?

<details><summary>answer</summary>

$-0.561$: the buffer grows $1, 1.9, 2.71$, heading for $1/(1 - 0.9) = 10$, at which point each step is 10 times a plain SGD step.

</details>

3. why does `zero_grad` set gradients to `None` rather than to zeros?

<details><summary>answer</summary>

`None` means "this parameter didn't get a gradient this step", and the optimizer skips it entirely: no weight decay, no momentum, no Adam step-count increment. a zero gradient would still decay the weights and advance the state, changing a frozen parameter. (PyTorch made the same choice for the same reason.)

</details>

4. Adam is "almost scale-invariant per coordinate". what does that mean, and when is it a disadvantage?

<details><summary>answer</summary>

multiplying a coordinate's gradients by a constant $c$ multiplies $\hat m$ by $c$ and $\sqrt{\hat v}$ by $|c|$, so the step is unchanged (up to $\epsilon$): the step size is set by $\eta$, not by the gradient's units. that's great for badly scaled features. it's a disadvantage when the gradient's size carries information: Adam takes steps of about $\eta$ even where the gradient is tiny because you're near the optimum, so it needs a decaying learning rate to converge precisely, and it treats every coordinate as equally important, which biases which solution it finds (lab experiment 3).

</details>

5. your training loss plateaus at a noisy level with constant-step SGD. name two ways to go lower, and what each costs.

<details><summary>answer</summary>

lower the learning rate (or decay it): the floor is proportional to $\eta$, but progress per step slows down. increase the batch size: the gradient noise shrinks like $1/b$ (and to zero at $b = n$), but each step costs more. averaging the iterates (Polyak–Ruppert averaging) is a third, nearly free option.

</details>

6. why does `Adam` keep a step count per parameter rather than one for the optimizer?

<details><summary>answer</summary>

the bias correction $1 - \beta^t$ must use the number of updates *that parameter's* averages have received. a parameter that first gets a gradient at the optimizer's 1,000th step has $m = v = 0$; with a global $t = 1000$ the correction would be about 1 instead of $1 - \beta$, and its first steps would be far too small (for $m$) and wildly unbalanced.

</details>

7. (the predict callout in idea 4.) Adam versus AdamW, one step from 2 with a zero gradient?

<details><summary>answer</summary>

Adam: the coupled decay makes the gradient $0.01 \times 2 = 0.02$, and Adam's first step is $\eta\,\text{sign}(g) = 0.1$ whatever its size, so $w = 1.9$: a tiny penalty produced a big step. AdamW: $w = 2(1 - 0.1 \times 0.01) = 1.998$, and the Adam part does nothing. (worked example.)

</details>

## common bugs

- **rebinding instead of updating in place.** symptom: the optimizer's own `p.data` changes but the model's weights don't. use `-=`, `*=`.
- **`p.grad += wd * p.data`.** symptom: the gradient the caller sees has changed, and repeated steps compound the decay. build a new array.
- **treating `grad is None` as zero.** symptom: frozen parameters still decay, and Adam's step count advances for parameters that never trained. skip them.
- **one momentum buffer or step count for all parameters.** symptom: parameters of the same shape interfere. key the state by the parameter's index.
- **caching the learning rate in `__init__`.** symptom: a scheduler changes `opt.lr` and nothing happens. read `self.lr` inside `step()`.
- **`eps` inside the square root, or no bias correction.** symptom: small but systematic mismatches with PyTorch (the checks compare against its rules), or huge early steps.
- **`np.asarray` in `Param`.** symptom: training a `Param` changes the array you built it from. `np.array(..., dtype=np.float64)` always copies.

## going deeper

- Robbins & Monro (1951), "A stochastic approximation method", *Annals of Mathematical Statistics* 22(3). where decaying step sizes come from. [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
- Bottou, Curtis & Nocedal (2018), "Optimization methods for large-scale machine learning", *SIAM Review* 60(2). SGD's theory and practice, including the noise floor. [arXiv:1606.04838](https://arxiv.org/abs/1606.04838)
- Duchi, Hazan & Singer (2011), "Adaptive subgradient methods for online learning and stochastic optimization", *JMLR* 12. Adagrad. [jmlr.org/papers/v12/duchi11a](https://jmlr.org/papers/v12/duchi11a.html)
- Kingma & Ba (2015), "Adam: a method for stochastic optimization", *ICLR*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Loshchilov & Hutter (2019), "Decoupled weight decay regularization", *ICLR*. AdamW. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Wilson, Roelofs, Stern, Srebro & Recht (2017), "The marginal value of adaptive gradient methods in machine learning", *NeurIPS*. what lab experiment 3 shows. [arXiv:1705.08292](https://arxiv.org/abs/1705.08292)
- Goyal et al. (2017), "Accurate, large minibatch SGD: training ImageNet in 1 hour". the linear scaling rule in practice. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)

## connections

- ← builds on: heavy ball and Nesterov momentum (topic 03), gradient descent and conditioning (topic 02), variance of averages (ch 04 · topic 03), `as_rng` and generators (ch 00).
- → used in: `SoftmaxRegression` with minibatch SGD (ch 08), `reparam_vi` with `Adam` over `Param`s (ch 12), `mlfs.nn.fit` with `minibatch_indices` and the full `zero_grad → backward → step` loop (ch 14), learning-rate schedules, `get_lr`/`set_lr`, gradient clipping and the LR range test (ch 15), DP-SGD (ch 28), and `ZeroAdamW` (ch 26), which must reproduce your `AdamW` exactly.
