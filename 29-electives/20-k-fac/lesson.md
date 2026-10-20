# K-FAC

## why this matters

topic 19's natural gradient reached a training loss in 4 steps that took Adam 30, and still lost on wall-clock time, because each step solved a linear system in the Fisher with 10 rounds of conjugate gradients. that was a 25,000-parameter network on 1,000 images. chapter 15's MNIST network has 235,000 parameters, and its first layer alone has a $201{,}000 \times 201{,}000$ Fisher block: about 160 GB in float32, and far more arithmetic to invert than anyone will spend on one step. exact natural gradient is out of reach for real networks, and conjugate gradients only partly rescue it.

**K-FAC** (Kronecker-factored approximate curvature; Martens & Grosse, 2015) makes the natural gradient cheap with two approximations. it ignores the curvature *between* layers (a block-diagonal Fisher), and it approximates each layer's block as a **Kronecker product** of two small matrices: one built from the layer's inputs, one from the gradients flowing back into its outputs. for that first layer the two factors are $785 \times 785$ and $256 \times 256$, and inverting them costs less than a single forward pass on a large batch. the result is an optimizer with natural gradient's per-step behaviour at a few times the cost of an Adam step. it spawned a family (KFC for convolutions, EKFAC) and related Kronecker-factored preconditioners (Shampoo), and is one of the few second-order methods people actually use for deep networks. in this topic you build it, and in the lab you run the information-geometry block's project: K-FAC against a carefully tuned Adam on chapter 15's network, counting both steps and seconds.

## warm-up: predict first

1. $A$ is $785\times785$ and $G$ is $256\times256$. how do you apply $(G\otimes A)^{-1}$ to a vector without forming the $201{,}000\times201{,}000$ matrix, and what does it cost?
2. a linear layer $s = W\bar a$ on one example. what's the rank of the gradient $\partial\log p/\partial W$?
3. on a batch of **one** example, is K-FAC's Kronecker approximation of the layer's Fisher block exact, approximate, or meaningless?
4. you add the damping $\lambda I$ to the Kronecker product by adding something to each factor instead. what extra terms do you get?
5. K-FAC needs fewer steps than Adam. will it also be faster in wall-clock time on a laptop CPU?

## prerequisites check

- [ ] you know the Fisher of a network, $\frac1N\sum_nJ_n^\top H_nJ_n$, and why it uses labels sampled from the model (→ topic 18)
- [ ] you know natural gradient descent, damping, and the cost of the exact step (→ topic 19)
- [ ] you can write forward hooks and use `torch.autograd.grad` on intermediate tensors (→ ch 15)
- [ ] you've built `mlfs.torchnn.MLP` and trained it on MNIST (→ ch 15 · topic 07, ch 15 capstone)

## core ideas

### idea 1: Kronecker products are cheap to invert

the **Kronecker product** of an $m\times m$ matrix $G$ and an $n\times n$ matrix $A$ is the $mn\times mn$ block matrix whose $(i, j)$ block is $G_{ij}A$. it acts on matrices: if you flatten an $m\times n$ matrix $V$ row by row into a vector $\operatorname{vec}(V)$, then (on paper 1)

$$(G\otimes A)\operatorname{vec}(V) = \operatorname{vec}(GVA^\top),\qquad (G\otimes A)^{-1} = G^{-1}\otimes A^{-1}.$$

so applying the inverse of a Kronecker product to a gradient matrix $V$ is $G^{-1}VA^{-1}$ (for symmetric $A$): two small inverses and two matrix products, $O(m^3 + n^3 + mn(m + n))$ instead of $O(m^3n^3)$ (warm-up 1). for $m = 256$, $n = 785$ that's about $10^9$ operations against $10^{16}$.

**damping.** the natural gradient needs $(F + \lambda I)^{-1}$, and $G\otimes A + \lambda I$ is no longer a Kronecker product. K-FAC's fix (Martens & Grosse, 2015) is **factored Tikhonov damping**: damp each factor, with $\gamma = \sqrt\lambda$,

$$\big(G + \tfrac{\gamma}{\pi}I\big)\otimes\big(A + \pi\gamma\,I\big) = G\otimes A + \lambda I + \gamma\Big(\pi\,G\otimes I + \tfrac1\pi\,I\otimes A\Big),$$

which is $G\otimes A + \lambda I$ plus two positive semidefinite cross terms (on paper 3). the free constant $\pi$ is chosen to make the cross terms as small as possible, and the answer is to give each factor damping in proportion to its own scale:

$$\pi = \sqrt{\frac{\operatorname{tr}(A)/\dim A}{\operatorname{tr}(G)/\dim G}}.$$

so if $A$'s eigenvalues average 100 times $G$'s, $A$ gets 10 times $\sqrt\lambda$ and $G$ gets a tenth of it. the cross terms make factored damping stronger than plain $\lambda I$, which is part of why K-FAC's best damping values differ from exact natural gradient's.

> **🔮 predict:** with $A = 4I_3$, $G = I_2$ and $\lambda = 0.01$, what gets added to each factor? (the worked example has the answer.)

> **spec.** `damped_inverses(A, G, damping)`: symmetric PSD factors `A` `(n, n)` and `G` `(m, m)` → the pair $\big((A + \pi\sqrt\lambda I)^{-1}, (G + \tfrac{\sqrt\lambda}{\pi}I)^{-1}\big)$ with $\pi$ as above; for `damping == 0` just $(A^{-1}, G^{-1})$. don't modify the inputs; work in their dtype. `kron_precondition(grad, A_inv, G_inv)`: `grad` `(m, n)` → `G_inv @ grad @ A_inv`.

> **▶ build: stage 1.** implement `damped_inverses` and `kron_precondition`.

### idea 2: where the two factors come from

take a linear layer with weights $W$ (out × in) and bias $b$, and fold the bias in: $\bar a = (a, 1)$ and $\bar W = [W\mid b]$, so $s = \bar W\bar a$. for one example and one label $y$, the chain rule gives

$$\frac{\partial\log p(y\mid x)}{\partial\bar W} = g\,\bar a^\top,\qquad g = \frac{\partial\log p(y\mid x)}{\partial s},$$

an outer product (rank 1, warm-up 2) of the backpropagated gradient $g$ at the layer's output and the layer's input $\bar a$. flattened row by row it's $g\otimes\bar a$, so the layer's Fisher block is

$$F_\ell = \mathbb E\big[(g\otimes\bar a)(g\otimes\bar a)^\top\big] = \mathbb E\big[gg^\top\otimes\bar a\bar a^\top\big]\ \approx\ \mathbb E[gg^\top]\otimes\mathbb E[\bar a\bar a^\top] = G_\ell\otimes A_\ell,$$

where the expectation is over the examples and over labels $y$ drawn from the model (topic 18). the single approximation is the step "expectation of a Kronecker product ≈ Kronecker product of expectations", which treats the activations and the backpropagated gradients as independent. it's exact when there's nothing to average (one example and one label: warm-up 3, on paper 2), and for a single example with the exact expectation over labels too, because $\bar a$ is then fixed. with many examples it's a genuine approximation; the lab measures how rough. $A_\ell$ is the (uncentred) covariance of the layer's inputs; $G_\ell$ is the Fisher with respect to the layer's *outputs*. both are small.

for the output layer of a softmax classifier $g = e_y - p$, so with the exact expectation over $y$, $G = \frac1N\sum_n\operatorname{diag}(p_n) - p_np_n^\top$: topic 18's categorical Fisher, averaged. for hidden layers, $g$ comes from one backward pass per label. you'll implement two modes: the **exact expectation** over the $K$ classes ($K$ backward passes, weighting example $n$'s gradient for class $k$ by $\sqrt{p_{nk}}$ before forming $G$), which is deterministic and what the checks use, and **given labels**, one backward pass, used with labels sampled from the model (an unbiased one-sample estimate of the true Fisher, as Martens & Grosse do) or with the data labels (the empirical Fisher, topic 18's impostor).

practicalities: forward hooks on each `nn.Linear` can record its input and its output tensor; `torch.autograd.grad(objective, [outputs...])` then returns every layer's $g$ at once, without touching `.grad`.

> **spec.** `kfac_factors(model, X, labels=None)`: `model` maps `(N, d)` → logits `(N, K)`. returns a list with one `(A, G)` pair per `nn.Linear` in `model.modules()` order: $A = \frac1N\sum_n\bar a_n\bar a_n^\top$ `(in + 1, in + 1)`, with the 1 appended **last**; $G = \frac1N\sum_n\sum_y w_{ny}\,g_{ny}g_{ny}^\top$ `(out, out)`, where $g_{ny} = \partial\log p(y\mid x_n)/\partial s_n$. with `labels=None`, sum over all $K$ classes with $w_{ny} = p(y\mid x_n)$; with integer `labels` `(N,)`, use only $y = $ `labels[n]` with weight 1. remove your hooks afterwards and leave every `.grad` untouched.

> **▶ build: stage 2.** implement `kfac_factors`.

### idea 3: one K-FAC step, and how far to trust it

with the factors in hand, each layer's natural-gradient step is $U_\ell = G_\ell^{-1}\,[\nabla W_\ell\mid\nabla b_\ell]\,A_\ell^{-1}$ (damped), and the update is $\bar W_\ell \leftarrow \bar W_\ell - \eta U_\ell$. the gradients are those of the mean loss, already in `.grad`.

how big should the step be? topic 18 showed that $\tfrac12\Delta^\top F\Delta$ predicts the KL divergence a step causes, as long as the step is small. for the K-FAC step $\Delta = -\eta U$, and with $FU \approx \nabla L$ (that's what preconditioning does), the predicted KL is about $\tfrac12\eta^2\sum_\ell\langle U_\ell, \nabla_\ell\rangle$. **KL clipping** (Ba, Grosse & Martens, 2017) shrinks the step whenever that estimate is too large:

$$\nu = \min\Big(1,\ \sqrt{\frac{\kappa}{\eta^2\sum_\ell\langle U_\ell, \nabla_\ell\rangle}}\Big),\qquad \bar W_\ell \leftarrow \bar W_\ell - \eta\,\nu\,U_\ell,$$

with $\langle\cdot,\cdot\rangle$ the elementwise (Frobenius) inner product. it's a trust region measured in distribution space, the same idea as TRPO's KL constraint (ch 24) and as topic 19's motivation for damping. in practice $\kappa$ becomes the main step-size knob: when the clip is active, $\eta$ cancels out of the update.

> **spec.** `kfac_step(model, factors, *, lr, damping, kl_clip=None)`: `model` has only `nn.Linear` layers with parameters, with gradients already in `.grad`; `factors` is a list like `kfac_factors` returns. for each layer, $U = $ `kron_precondition(`$[\nabla W\mid\nabla b]$`, *damped_inverses(A, G, damping))`; $\nu$ as above when `kl_clip` is set (and the sum is positive), else 1. update `weight` and `bias` in place by $-\eta\nu U$ (the last column of $U$ is the bias). returns $\nu$ as a float.

> **▶ build: stage 3.** implement `kfac_step`.

### idea 4: the optimizer

a usable K-FAC adds four things to `kfac_step`.

- **running averages.** a minibatch's factors are noisy. keep exponential moving averages, $A \leftarrow \varepsilon A + (1 - \varepsilon)A_{\text{new}}$ (and the same for $G$), starting from the first minibatch's factors.
- **sampled labels.** the true Fisher needs labels drawn from the model; one sample per example (one extra backward pass) is an unbiased estimate, and it's what you'll use by default. the empirical Fisher (the minibatch's own labels) is an option, for the lab.
- **amortization.** computing the factors costs a forward and backward pass plus the outer products; inverting them costs $O(n^3)$. both can be done less often than every step: factors every $T_{\text{stat}}$ steps, inverses every $T_{\text{inv}}$ steps (reused in between).
- **momentum.** heavy-ball momentum on the preconditioned, clipped step, $v \leftarrow \mu v + \nu U$, $\bar W \leftarrow \bar W - \eta v$.

Martens & Grosse's original algorithm also adapted the damping with a Levenberg–Marquardt rule and chose the step size and momentum from a quadratic model each step. the simpler recipe here (fixed damping, KL clipping, plain momentum) follows later practice (Ba et al., 2017; Grosse & Martens, 2016). per step, K-FAC costs one forward and backward for the gradient, one forward and backward with sampled labels for the factors (on statistics steps), the outer products $\bar a\bar a^\top$ and $gg^\top$, two small matrix products per layer to precondition, and an amortized share of the inverses. whether that beats Adam is the lab's question (warm-up 5).

> **🔮 predict:** chapter 15's network on MNIST, batch 256. how much more does one K-FAC step cost than one Adam step: about the same, 2–5 times, or 20 times?

> **spec.** `KFAC(model, lr=0.1, damping=1e-2, *, momentum=0.9, ema_decay=0.95, stat_every=1, inv_every=10, kl_clip=1e-3, fisher="sample", seed=None)`: `fisher` must be `"sample"`, `"exact"` or `"empirical"` (else `ValueError`). attributes: `steps` (starts at 0), `factors` and `inverses` (lists of pairs, `None` before the first step). `step(X, y)`, at step number `steps`: if `steps % stat_every == 0`, compute `kfac_factors` on the minibatch, with labels sampled from `softmax(model(X))` by `torch.multinomial` with a `torch.Generator` seeded from `seed` (default 0) for `"sample"`, `labels=None` for `"exact"`, `y` for `"empirical"`, and set `factors` to them (first time) or fold them into the running averages; if `steps % inv_every == 0` (or there are no inverses yet), recompute `inverses` with `damped_inverses`; compute the mean cross-entropy and its gradients; precondition with the stored inverses; clip as in `kfac_step`; update the momentum buffers and the parameters; increment `steps`. returns the loss before the update as a float.

> **▶ build: stage 4.** implement `KFAC`.

## worked example

**a Kronecker solve.** $G = \operatorname{diag}(2, 1)$, $A = \operatorname{diag}(1, 4)$, $V = \begin{pmatrix}1 & 2\\3 & 4\end{pmatrix}$: $G^{-1}VA^{-1}$ divides the rows by $(2, 1)$ and the columns by $(1, 4)$, giving $\begin{pmatrix}0.5 & 0.25\\3 & 1\end{pmatrix}$, which is what solving the $4\times4$ system $(G\otimes A)x = \operatorname{vec}(V)$ gives.

**factored damping.** $A = 4I_3$ (average eigenvalue 4), $G = I_2$ (average 1), $\lambda = 0.01$: $\pi = \sqrt{4/1} = 2$, so $A$ gets $\pi\sqrt\lambda = 0.2$ and $G$ gets $\sqrt\lambda/\pi = 0.05$: $A^{-1}_{\text{damped}} = I/4.2$, $G^{-1}_{\text{damped}} = I/1.05$.

**one example, exactly.** an output layer with input $x = 2$, so $\bar a = (2, 1)$ and $A = \bar a\bar a^\top = \begin{pmatrix}4 & 2\\2 & 1\end{pmatrix}$, and logits giving $p = (\tfrac14, \tfrac14, \tfrac12)$, so $G = \operatorname{diag}(p) - pp^\top = \frac1{16}\begin{pmatrix}3 & -1 & -2\\-1 & 3 & -2\\-2 & -2 & 4\end{pmatrix}$ (topic 18's worked example). the layer's exact $6\times6$ Fisher block is $G\otimes A$: for instance its entry for $(W_{11}, W_{11})$ is $G_{11}A_{11} = \tfrac3{16}\cdot4 = 0.75$, and for $(W_{11}, b_2)$ it's $G_{12}A_{12} = -\tfrac1{16}\cdot2 = -0.125$.

**a KL clip.** $\eta = 0.5$, $\sum_\ell\langle U_\ell, \nabla_\ell\rangle = 0.4$, $\kappa = 10^{-3}$: the predicted KL of the full step is $\tfrac12\cdot0.25\cdot0.4 = 0.05$, and $\nu = \sqrt{10^{-3}/(0.25\cdot0.4)} = 0.1$.

your code must reproduce these numbers.

## on paper

1. **the Kronecker identities.** with row-major $\operatorname{vec}$, show $(G\otimes A)\operatorname{vec}(V) = \operatorname{vec}(GVA^\top)$ and $(G\otimes A)(G^{-1}\otimes A^{-1}) = I$.

<details><summary>solution</summary>

row-major $\operatorname{vec}(V)$ lists $V_{11}, \ldots, V_{1n}, V_{21}, \ldots$: index $(i, j) \mapsto (i - 1)n + j$. the $(i, j)$ block of $G\otimes A$ is $G_{ij}A$, so the $i$-th block row of the product is $\sum_jG_{ij}A\,V_{j,:}^\top$, i.e. row $i$ of $GVA^\top$. the mixed-product rule $(G\otimes A)(G'\otimes A') = GG'\otimes AA'$ (block $(i, k)$ is $\sum_jG_{ij}G'_{jk}AA'$) gives $(G\otimes A)(G^{-1}\otimes A^{-1}) = I\otimes I = I$. eigenvalues multiply too: the eigenvalues of $G\otimes A$ are all products $\lambda_i(G)\lambda_j(A)$.

</details>

2. **the layer gradient and when K-FAC is exact.** show $\partial\log p/\partial\bar W = g\bar a^\top$ and that its row-major flattening is $g\otimes\bar a$. conclude that for a single example, $\mathbb E_y[(g\otimes\bar a)(g\otimes\bar a)^\top] = \mathbb E_y[gg^\top]\otimes\bar a\bar a^\top$ exactly.

<details><summary>solution</summary>

$s_i = \sum_j\bar W_{ij}\bar a_j$, so $\partial\log p/\partial\bar W_{ij} = g_i\bar a_j$. row-major flattening puts $g_i\bar a_j$ at index $(i - 1)(n) + j$, which is the $(i, j)$ entry of $g\otimes\bar a$. then $(g\otimes\bar a)(g\otimes\bar a)^\top = gg^\top\otimes\bar a\bar a^\top$ (mixed product). with one example, $\bar a$ doesn't depend on the label $y$ (it's computed before the output), so the expectation over $y$ only touches $gg^\top$. with several examples, $\frac1N\sum_n G_n\otimes A_n \ne (\frac1N\sum G_n)\otimes(\frac1N\sum A_n)$ in general.

</details>

3. **factored damping and the choice of $\pi$.** expand $(G + \tfrac{\gamma}{\pi}I)\otimes(A + \pi\gamma I)$ with $\gamma = \sqrt\lambda$, and show that choosing $\pi$ to minimize the trace of the extra terms beyond $G\otimes A + \lambda I$ gives $\pi = \sqrt{(\operatorname{tr}A/\dim A)/(\operatorname{tr}G/\dim G)}$.

<details><summary>solution</summary>

$(G + \tfrac\gamma\pi I)\otimes(A + \pi\gamma I) = G\otimes A + \pi\gamma\,G\otimes I + \tfrac\gamma\pi\,I\otimes A + \gamma^2I$. the extra terms are $\gamma(\pi\,G\otimes I + \tfrac1\pi I\otimes A)$, both PSD, with trace $\gamma(\pi\operatorname{tr}(G)\dim A + \tfrac1\pi\dim G\operatorname{tr}(A))$. minimizing $\pi u + v/\pi$ over $\pi > 0$ gives $\pi = \sqrt{v/u} = \sqrt{\dim G\operatorname{tr}A/(\operatorname{tr}G\dim A)}$, which is the formula. so $A$ gets $\pi\gamma = \gamma\sqrt{\bar\lambda_A/\bar\lambda_G}$ and $G$ gets $\gamma\sqrt{\bar\lambda_G/\bar\lambda_A}$: each factor is damped in proportion to its own average eigenvalue (Martens & Grosse use the same choice).

</details>

4. **the KL estimate behind the clip.** for the undamped K-FAC step $\Delta_\ell = -\eta(G_\ell\otimes A_\ell)^{-1}\nabla_\ell$ and a block-diagonal Fisher equal to the Kronecker approximation, show that $\tfrac12\Delta^\top F\Delta = \tfrac12\eta^2\sum_\ell\langle U_\ell, \nabla_\ell\rangle$.

<details><summary>solution</summary>

with $F$ block diagonal, $\Delta^\top F\Delta = \sum_\ell\Delta_\ell^\top(G_\ell\otimes A_\ell)\Delta_\ell = \eta^2\sum_\ell U_\ell^\top(G_\ell\otimes A_\ell)(G_\ell\otimes A_\ell)^{-1}\nabla_\ell = \eta^2\sum_\ell\langle U_\ell, \nabla_\ell\rangle$, in vectorized form. with damping or with the true $F$ it's an approximation, which is fine for a safety clip. requiring it to be at most $\kappa$ (dropping the $\tfrac12$, a convention) gives $\nu$.

</details>

## experiment

`lab.py` investigates, with your K-FAC:

1. how close the Kronecker blocks are to the exact Fisher blocks of a small MLP, and how much of the exact natural gradient's benefit K-FAC's direction keeps.
2. **the project:** K-FAC against a tuned Adam on chapter 15's MNIST network, with a fixed tuning protocol, three seeds, steps and wall-clock time to a common training loss, and test accuracy; plus K-FAC with the empirical Fisher and a cheaper K-FAC that refreshes its curvature less often.

## check your understanding

1. which two approximations turn the Fisher of a network into K-FAC, and which one is responsible for the cheap inverse?

<details><summary>answer</summary>

(1) block-diagonal by layer: the curvature between different layers is dropped. (2) within a block, $\mathbb E[gg^\top\otimes\bar a\bar a^\top] \approx \mathbb E[gg^\top]\otimes\mathbb E[\bar a\bar a^\top]$. the second makes each block a Kronecker product, which is what lets you invert it through two small matrices. the first lets you treat the layers separately. (Martens & Grosse also propose a block-tridiagonal version that keeps neighbouring layers' interactions.)

</details>

2. (warm-up 3.) why is K-FAC exact for a single example?

<details><summary>answer</summary>

the per-example score of a layer is $g\otimes\bar a$, and the input $\bar a$ doesn't depend on the label. the expectation over labels only acts on $gg^\top$, so the block is exactly $\mathbb E_y[gg^\top]\otimes\bar a\bar a^\top$. the approximation enters only when you average over examples, whose $\bar a$ and $g$ vary together.

</details>

3. why must the bias be appended to the input, and does it matter where?

<details><summary>answer</summary>

the bias gradient is $g$ itself, i.e. $g\cdot1$: appending a constant 1 to $\bar a$ puts the bias in the same Kronecker structure as the weights, so one pair of factors covers $[W\mid b]$. the position only has to be consistent: the spec appends it last, so the bias update is the last column of $U$. put it first in $A$ but read the last column, and you update the biases with weight curvature.

</details>

4. what would go wrong if you used the data labels instead of sampled labels for $G$ (the empirical Fisher)?

<details><summary>answer</summary>

$G$ would measure the gradients' size on the data rather than the model's curvature, inheriting the empirical Fisher's problems from topics 18 and 19: tiny steps far from a good fit, and a scale set by the residuals near it. how much that matters in practice is an empirical question; the lab's project measures it on MNIST.

</details>

5. why does K-FAC update its inverses only every $T_{\text{inv}}$ steps, and what's the risk?

<details><summary>answer</summary>

inverting the factors is the most expensive part (cubic in the layer widths). the factors change slowly, especially as running averages, so stale inverses are still good preconditioners. the risk is that the curvature changes faster than the inverses are refreshed (early in training, or after a learning-rate change), and the preconditioner then mis-scales the step; damping and the KL clip limit the damage.

</details>

6. in the KL clip, why can the step size $\eta$ "cancel out" when the clip is active?

<details><summary>answer</summary>

if the clip is active, $\nu = \sqrt\kappa/(\eta\sqrt{\sum\langle U, \nabla\rangle})$, so $\eta\nu = \sqrt{\kappa/\sum\langle U, \nabla\rangle}$, independent of $\eta$. the update is then the K-FAC direction scaled to a predicted KL of $\kappa$: a pure trust-region step. that's why $\kappa$, not $\eta$, is the knob to tune once the clip binds.

</details>

## common bugs

- **$A$ and $G$ swapped, or $V$ transposed.** symptom: shape errors for non-square layers, or a silently wrong update for square ones. $G$ (output side, out × out) multiplies on the left, $A$ (input side) on the right.
- **inverted $\pi$.** symptom: the large factor gets almost no damping and the small one too much; training is unstable at damping values that should be safe.
- **the bias in the wrong place.** symptom: the one-example exactness check fails only in the last row and column of $A$.
- **gradients of the mean loss in $G$.** symptom: $G$ is $N^2$ times too small. $G$ needs per-example gradients of $\log p$: backpropagate the *sum* over the batch (each $s_n$ depends only on example $n$), then average the outer products.
- **hooks left registered.** symptom: memory grows and every later forward pass is slower; or a second call collects twice as many tensors as layers.
- **sampling labels without a seeded generator.** symptom: runs that can't be reproduced, and checks that pass or fail at random.

## going deeper

- Martens & Grosse (2015), "Optimizing neural networks with Kronecker-factored approximate curvature", *ICML*. [arXiv:1503.05671](https://arxiv.org/abs/1503.05671). the original: the Kronecker approximation and when it's accurate, factored damping, adaptive damping, and the block-tridiagonal variant.
- Grosse & Martens (2016), "A Kronecker-factored approximate Fisher matrix for convolution layers", *ICML*. [arXiv:1602.01407](https://arxiv.org/abs/1602.01407). KFC: the same idea for conv layers, which chapter 16's networks would need.
- Ba, Grosse & Martens (2017), "Distributed second-order optimization using Kronecker-factored approximations", *ICLR*. [OpenReview](https://openreview.net/forum?id=SkkTMpjex). KL clipping and asynchronous inverses.
- George, Laurent, Bouthillier, Ballas & Vincent (2018), "Fast approximate natural gradient descent in a Kronecker-factored eigenbasis", *NeurIPS*. [arXiv:1806.03884](https://arxiv.org/abs/1806.03884). EKFAC.
- Gupta, Koren & Singer (2018), "Shampoo: preconditioned stochastic tensor optimization", *ICML*. [arXiv:1802.09568](https://arxiv.org/abs/1802.09568). a Kronecker-factored preconditioner built from gradients rather than the Fisher.
- Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue & Grosse (2019), "Which algorithmic choices matter at which batch sizes? Insights from a noisy quadratic model", *NeurIPS*. [arXiv:1907.04164](https://arxiv.org/abs/1907.04164). why K-FAC's advantage grows with the batch size.
- Martens (2020), "New insights and perspectives on the natural gradient method", *JMLR* 21(146). [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

## connections

- ← builds on: the network Fisher and the empirical Fisher (topic 18), natural gradient, damping and trust regions (topic 19), `mlfs.torchnn.MLP` and the MNIST pipeline (ch 15), forward hooks and autograd (ch 15), TRPO's KL constraint (ch 24 · topic 04).
- → used in: the chapter capstone's elective portfolio (this block's project). KFC and EKFAC extend the same factorization to convolutional networks (ch 16); the factored Fisher also appears as a Laplace-approximation posterior for Bayesian deep learning (Ritter, Botev & Barber, 2018) and in EWC-style continual learning (topic 13).
