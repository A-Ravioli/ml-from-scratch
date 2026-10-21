# OT in machine learning

## why this matters

optimal transport enters machine learning in three roles. as a **distance** between a model and the data, it gives a generator a useful gradient even when the two don't overlap: that was the WGAN. as a **coupling**, it decides which noise sample a generative model should turn into which data point, and a good pairing makes the model's paths straight, so it can sample in one or two steps instead of a hundred. as a **map**, it carries labelled data from one domain onto another.

in all three, real systems work on **minibatches** of a few hundred points, solved exactly (Hungarian) or approximately (Sinkhorn) inside a training loop. that makes OT cheap enough to use, and it changes what it computes: a minibatch distance is biased, a minibatch coupling is only locally optimal. this topic builds one tool for each role on top of your topic 15 and 16 solvers, and the lab ends with the block's **project**: replace the random pairing of noise and data in your chapter 22 flow matching with a minibatch OT coupling, and measure what it does to the straightness of the paths and to sample quality at few sampling steps, against the honest floor of topic 15.

## warm-up: predict first

1. in chapter 22, a flow trained with independent noise–data pairs is sampled with a single Euler step. where do the samples land?
2. pairing noise and data by optimal transport within each batch of $B$ points: as $B$ grows, what happens to the paths? does the distribution the flow generates change?
3. two batches of 256 points are drawn from the *same* distribution. is their exact $W_1$ zero? is their debiased Sinkhorn divergence?
4. a WGAN critic is trained to maximize $\mathbb E_\mu f - \mathbb E_\nu f$ with a gradient penalty. can its estimate exceed the true $W_1$?
5. labelled data from one domain, unlabelled data from a domain rotated by 90°. can optimal transport carry the labels across correctly?

## prerequisites check

- [ ] exact OT, Kantorovich potentials and barycentric maps (→ topic 15); Sinkhorn in the log domain and the Sinkhorn divergence (→ topic 16)
- [ ] flow matching, `fit_flow_matching(..., coupling=...)`, `sample_flow` and `straightness` (→ ch 22 · topic 05)
- [ ] the WGAN critic, `critic_loss` and `gradient_penalty` (→ ch 21 · topic 06)
- [ ] autograd and `.detach()` (→ ch 15)

## core ideas

### idea 1: minibatch OT couplings for flow matching

**flow matching, recalled.** draw noise $x_0 \sim \mathcal N(0, I)$ and data $x_1$, set $x_t = (1 - t)x_0 + tx_1$, and regress a network $v_\theta(x_t, t)$ on the velocity $x_1 - x_0$ of the straight line. the minimizer is the **conditional expectation** $v^\star(x, t) = \mathbb E[x_1 - x_0 \mid x_t = x]$: the average velocity of all the training lines passing through $x$ at time $t$. integrating $\dot x = v^\star(x, t)$ from noise gives samples from the data distribution.

**why the paths curve.** with **independent** pairs, every noise point is paired with a random data point, so lines from everywhere to everywhere pass through each $x$, and their average velocity points towards the middle of the data. at $t = 0$ it's exactly $\mathbb E[x_1] - x_0$ (ch 22, on paper 4): one Euler step sends every sample to the data mean. the trajectories of the ODE can't cross (they're solutions of the same ODE), but the training lines do, everywhere, and the difference between the two is the curvature that costs you sampling steps.

**straight by design.** the pairing is a **coupling** $\pi$ of the noise and data distributions, and *any* coupling gives a valid objective: the learned flow still carries noise to data (ch 22 · topic 05, on paper 1; here, on paper 1 checks that minibatch pairs are such a coupling). so choose the coupling whose lines don't cross. for the squared cost, the optimal transport plan does exactly that: its pairs are *cyclically monotone*, and two straight lines between OT pairs never meet at an intermediate time (on paper 2). then each $x$ at time $t$ lies on one training line, $v^\star$ is that line's velocity, the flow's trajectories *are* the lines, and one Euler step is exact. this is McCann's (1997) displacement interpolation, and the dynamic view of optimal transport (Benamou & Brenier, 2000).

you can't solve OT between a Gaussian and the whole data set, but you can solve it **within each training batch**: draw $B$ noise samples and $B$ data points, find the permutation of the noise that minimizes $\sum_k\|x_0^{\sigma(k)} - x_1^k\|^2$ (topic 15: for uniform equal-size batches the optimal plan is a permutation, and the Hungarian algorithm finds it), and train on those pairs. that's **minibatch OT conditional flow matching** (Tong et al., 2024; Pooladian et al., 2023). $B = 1$ is the independent coupling; as $B$ grows the coupling approaches the true OT plan and the paths straighten. the price is an $O(B^3)$ assignment per training step, done on the CPU. (the `torchcfm` package's OT sampler instead draws $B$ pairs *with replacement* from the plan; returning a permutation, as here and in Pooladian et al., uses every noise sample exactly once.)

> **spec.** `minibatch_ot_coupling(x0, x1)`: for tensors of the same shape `(B, ...)` (`ValueError` otherwise), return `x0` reordered, `x0[sigma]`, so that pairing `out[k]` with `x1[k]` minimizes the total squared Euclidean distance over all permutations (flatten the non-batch dimensions; use `scipy.optimize.linear_sum_assignment`). it plugs into chapter 22's `fit_flow_matching(..., coupling=minibatch_ot_coupling)`. `pairing_cost(x0, x1)`: the mean over $k$ of $\|x_0^k - x_1^k\|^2$ (summed over non-batch dimensions), a float.

> **🔮 predict:** training batches of 256 are split into blocks of $k$ points, each block re-paired by exact OT. how does straightness change from $k = 8$ to $k = 128$? and quality with 64 sampling steps?

> **▶ build: stage 1.** implement `minibatch_ot_coupling` and `pairing_cost`.

### idea 2: entropic couplings

an exact assignment is a hard, discontinuous decision: move one noise point slightly and its partner can jump to another mode. sampling the pairs from an **entropic** plan softens it. solve topic 16's problem between the batch's noise and data with uniform weights, then for each data point $x_1^k$ draw its noise partner from column $k$ of the plan,

$$\Pr\big[\text{partner of } x_1^k = x_0^i\big] = \frac{P_{ik}}{\sum_{i'}P_{i'k}} .$$

as $\varepsilon \to 0$ this recovers the exact coupling; as $\varepsilon \to \infty$ the plan becomes $\tfrac1{B^2}\mathbf 1\mathbf 1^\top$ and every partner is equally likely, the independent coupling. in between, it's a family of couplings with a knob. Tong et al. (2024) show that with $\varepsilon = 2\sigma^2$ and noisy interpolants of variance $\sigma^2$ the learned model approximates a **Schrödinger bridge**, the entropic counterpart of the OT flow. the draws are made with replacement, so unlike the exact coupling some noise samples are used twice in a batch and others not at all; the marginals are still right on average.

> **spec.** `sinkhorn_coupling(x0, x1, eps, *, generator=None, max_iters=1000, tol=1e-9)`: with $C_{ik} = \|x_0^i - x_1^k\|^2$ (flattened, float64) and your topic 16 `ot16.sinkhorn_log` with uniform weights and the given `max_iters` and `tol`, form $Q$ = the plan with each column divided by its sum, transposed (row $k$ is the distribution of the partner of $x_1^k$), draw `idx = torch.multinomial(Q, 1, replacement=True, generator=generator)[:, 0]`, and return `x0[idx]`. `ValueError` for mismatched shapes.

> **▶ build: stage 2.** implement `sinkhorn_coupling`.

### idea 3: OT as a loss: WGAN revisited

chapter 21's WGAN trains a **critic** $f$ to maximize $\mathbb E_\mu f - \mathbb E_\nu f$ over (approximately) 1-Lipschitz networks: by Kantorovich–Rubinstein duality (topic 15, on paper 3) the supremum is $W_1$, and the generator descends the critic's slope. two approximations hide in that sentence. the critic is restricted to a network class and trained by SGD, so it under-estimates the supremum; and the Lipschitz constraint is only a soft penalty on random interpolates, so the critic can also overshoot it. it's an estimate with no guarantee in either direction.

with topics 15 and 16 you can instead compute the transport problem **on each minibatch**: exactly (the Hungarian algorithm) or, smoothly, with Sinkhorn (Genevay et al., 2018). its gradient comes from the **envelope theorem**: the value of an optimization problem changes with a parameter only through the parameter's direct effect, at the fixed optimum. for the entropic dual, $\partial\,\mathrm{OT}_\varepsilon/\partial C_{ij} = P_{ij}$ (on paper 3), so

$$\nabla_{x_i}\mathrm{OT}_\varepsilon = \sum_jP_{ij}\,\nabla_xc(x_i, y_j) .$$

in an autograd framework: run Sinkhorn on the **detached** cost to get the potentials $f, g$, then build the dual objective $\langle f, a\rangle + \langle g, b\rangle - \varepsilon\sum_{ij}a_ib_j(e^{(f_i + g_j - C_{ij})/\varepsilon} - 1)$ from the detached potentials and a **differentiable** cost $C(x, y)$. its value is $\mathrm{OT}_\varepsilon$ at convergence, and backpropagation gives exactly the envelope gradient, without differentiating through hundreds of Sinkhorn iterations. debias as in topic 16 by subtracting half of each self-term; in $\mathrm{OT}_\varepsilon(x, x)$ the points appear in both slots of the cost, and autograd handles both.

there's one bias debiasing can't remove: **minibatch bias**. the expected OT between two batches of $B$ points is not the OT between the distributions, and it isn't zero when the distributions are equal (topic 15's experiment 3; Fatras et al., 2020). the lab measures how large it is at $B = 256$, for exact transport and for the Sinkhorn divergence at several $\varepsilon$.

> **spec.** `sinkhorn_loss(x, y, eps, *, debiased=True, max_iters=1000, tol=1e-9)`: for tensors `x` (n, d) and `y` (m, d) with uniform weights and squared Euclidean cost, return a 0-dim tensor (in the inputs' dtype) equal to $\mathrm{OT}_\varepsilon(x, y)$, or $S_\varepsilon(x, y) = \mathrm{OT}_\varepsilon(x, y) - \tfrac12\mathrm{OT}_\varepsilon(x, x) - \tfrac12\mathrm{OT}_\varepsilon(y, y)$ if `debiased`, each term being the entropic dual objective at the potentials from `ot16.sinkhorn_log` (run in float64 on the detached cost, with the given `max_iters` and `tol`), with the potentials detached and the cost differentiable, so that gradients with respect to `x` and `y` are the envelope-theorem gradients.

> **🔮 predict:** two batches of 256 points from the *same* 2-D distribution: is the debiased Sinkhorn divergence between them close to 0?

> **▶ build: stage 3.** implement `sinkhorn_loss`.

### idea 4: OT for domain adaptation

a classifier trained on a **source** domain (labelled: simulated images, last year's sensor) is applied to a **target** domain whose inputs have shifted (real images, a recalibrated sensor), with few or no target labels. Courty et al. (2017) proposed to transport the source samples onto the target samples, then train on the transported source with its original labels: if the shift is a smooth deformation, the cheapest way to move one cloud onto the other is likely to move each source point to where its counterpart lives. with a plan $P$ between the $n_s$ source and $n_t$ target points (uniform weights, squared cost), each source point is mapped by topic 15's **barycentric map**, $T(x_i) = \sum_jP_{ij}y_j/\sum_jP_{ij}$. with an exact plan and equal sizes, $T$ sends each source point onto one target point; with an entropic plan, onto a weighted average of nearby target points, which smooths out sampling noise.

the method's assumption is also its weakness: optimal transport never sees a label. it moves mass as cheaply as possible, and if the cheapest matching isn't the label-preserving one (a large rotation, classes of different sizes in the two domains), labels get carried to the wrong places. remedies add label information to the cost: a group-sparsity penalty that keeps each target point from receiving mass from several classes (Courty et al., 2017), or a joint cost on features and predicted labels (JDOT; Courty et al., 2017b).

> **spec.** `ot_domain_adaptation(Xs, Xt, *, eps=None, max_iters=1000)`: NumPy arrays `Xs` (n_s, d) and `Xt` (n_t, d), uniform weights, squared Euclidean cost; with `eps=None` the plan from your topic 15 `ot15.kantorovich_lp`, otherwise from `ot16.sinkhorn_log` (`max_iters` as given, `tol=1e-9`); returns the `(n_s, d)` barycentric images `ot15.barycentric_map(P, Xt)`.

> **▶ build: stage 4.** implement `ot_domain_adaptation`.

## worked example

**a coupling by hand.** noise $x_0 = \{(0, 0), (0, 2)\}$, data $x_1 = \{(2, 2), (2, 0)\}$, in that order. the given (independent) pairing sends $(0, 0) \to (2, 2)$ and $(0, 2) \to (2, 0)$: both lines have squared length 8, so the pairing cost is 8, and they **cross** at $(1, 1)$ at $t = \tfrac12$. the training velocities there are $(2, 2)$ and $(2, -2)$, so the regression target at $(1, 1)$, their average, is $(2, 0)$: the learned flow moves horizontally there, and neither particle follows its line. the OT pairing swaps the noise: `minibatch_ot_coupling` returns $\{(0, 2), (0, 0)\}$, pairing $(0, 2) \to (2, 2)$ and $(0, 0) \to (2, 0)$, two parallel lines of squared length 4: pairing cost 4, no crossing, and the velocity field $(2, 0)$ is exactly the lines'.

**the entropic version.** with $C_{ik} = \|x_0^i - x_1^k\|^2 = \begin{pmatrix}8 & 4\\ 4 & 8\end{pmatrix}$ and uniform weights, the plan is symmetric with off-diagonal mass $q$ and cross-ratio $\big((\tfrac12 - q)/q\big)^2 = e^{-(8 + 8 - 4 - 4)/\varepsilon}$, so the probability that $x_1^0 = (2, 2)$ gets the "right" partner $(0, 2)$ is $2q = \sigma(4/\varepsilon)$: 0.731 at $\varepsilon = 4$, 0.982 at $\varepsilon = 1$.

**a loss by hand.** for a cloud and its translate by $t$, topic 16 showed $S_\varepsilon(x, x + t) = \|t\|^2$ at every $\varepsilon$. with $n$ points of weight $1/n$, the gradient with respect to each $x_i$ is $-2t/n$: the loss pulls the cloud rigidly towards its target.

your `minibatch_ot_coupling`, `pairing_cost` and `sinkhorn_loss` must reproduce these numbers.

## on paper

1. **minibatch pairs form a coupling.** draw a noise batch $x_0^1, \dots, x_0^B$ i.i.d. from $p_0$ and a data batch $x_1^1, \dots, x_1^B$ i.i.d. from $p_1$, let $\sigma$ be any permutation computed from both batches (for instance the OT assignment), and pick a slot $K$ uniformly at random, independently of everything. show that the training pair $(x_0^{\sigma(K)}, x_1^K)$ has marginals $p_0$ and $p_1$, so it's a valid coupling for flow matching (ch 22, on paper 1), and that $B = 1$ gives the independent coupling. does a single slot $k$ of the returned noise, $x_0^{\sigma(k)}$, have law $p_0$?

<details><summary>solution</summary>

$x_1^K$ is a uniformly chosen element of an i.i.d. batch, with $K$ independent of the batch, so it has law $p_1$. for the noise, condition on both batches: $\sigma$ is then fixed and a bijection, so $\sigma(K)$ is uniform on $\{1, \dots, B\}$, and $\Pr[x_0^{\sigma(K)} \in A \mid \text{batches}] = \frac1B\sum_i\mathbf 1[x_0^i \in A]$, whose expectation is $p_0(A)$. with $B = 1$ the only permutation is the identity, and $x_0$, $x_1$ are drawn independently. a *fixed* slot is still marginally $p_0$ for the OT assignment: it treats data slots symmetrically and the batch is i.i.d., so every slot has the same law as the $K$-uniform pair. what changes is the dependence: given $x_1^k$ it isn't $p_0$, and that's the whole point. a $\sigma$ that treats slots asymmetrically ("smallest noise in slot 1") can break even the marginal.

</details>

2. **OT lines don't cross.** pairs $(x_0, x_1)$ and $(y_0, y_1)$ from an optimal plan for the squared cost satisfy $\|x_0 - x_1\|^2 + \|y_0 - y_1\|^2 \le \|x_0 - y_1\|^2 + \|y_0 - x_1\|^2$. show that the segments $(1 - t)x_0 + tx_1$ and $(1 - t)y_0 + ty_1$ can't meet at any $t \in (0, 1)$ unless the pairs are equal.

<details><summary>solution</summary>

expanding the squares, the inequality is equivalent to $\langle x_0 - y_0,\ x_1 - y_1\rangle \ge 0$. if the segments meet at time $t$, $(1 - t)(x_0 - y_0) = -t(x_1 - y_1)$, so $\langle x_0 - y_0, x_1 - y_1\rangle = -\tfrac{1 - t}{t}\|x_0 - y_0\|^2 \le 0$. both together force $x_0 = y_0$, and then $x_1 = y_1$. an exact minibatch coupling is optimal *for its batch*, so lines from the same batch never cross; lines from different batches still can, which is why a larger $k$ straightens more.

</details>

3. **the envelope theorem, and a factor of two.** let $D(f, g; C)$ be topic 16's entropic dual and $(f^\star, g^\star)$ its maximizer. show $\partial D(f^\star, g^\star; C)/\partial C_{ij} = P_{ij}$ (with $f^\star, g^\star$ held fixed), and that this is the derivative of $\mathrm{OT}_\varepsilon(C) = \max_{f,g}D$. what goes wrong if you instead write the loss as $\langle a, T_C(g^\star)\rangle + \langle b, T_C(f^\star)\rangle$, with $T_C$ the soft $c$-transform, and backpropagate through both transforms?

<details><summary>solution</summary>

$D$ depends on $C$ only through $-\varepsilon\sum_{ij}a_ib_j(e^{(f_i + g_j - C_{ij})/\varepsilon} - 1)$, whose derivative in $C_{ij}$ is $a_ib_je^{(f_i + g_j - C_{ij})/\varepsilon} = P_{ij}$. since $\nabla_{f,g}D = 0$ at the maximizer, $\frac{d}{dC}\max D = \partial_CD$ there (the envelope theorem). for the other expression: each soft $c$-transform satisfies $\partial\langle a, T_C(g)\rangle/\partial C_{ij} = P_{ij}$ at the fixed point, and so does the other, so the sum has derivative $2P_{ij}$: the gradient is doubled, while the value is right. (keeping one transform and the other potential detached, $\langle a, T_C(g^\star)\rangle + \langle b, g^\star\rangle$, the semi-dual, is also correct.)

</details>

4. **barycentric maps keep the target's mean.** with uniform weights and any plan $P \in U(a, b)$, show $\frac1{n_s}\sum_iT(x_i) = \frac1{n_t}\sum_jy_j$.

<details><summary>solution</summary>

with $a_i = 1/n_s$, $\sum_jP_{ij} = 1/n_s$, so $T(x_i) = n_s\sum_jP_{ij}y_j$ and $\frac1{n_s}\sum_iT(x_i) = \sum_{ij}P_{ij}y_j = \sum_jb_jy_j = \frac1{n_t}\sum_jy_j$, because the columns of $P$ sum to $b_j = 1/n_t$. the mapped source always sits on the target's centre of mass; at $\varepsilon \to \infty$ it collapses onto it.

</details>

## experiment

experiment 1 revisits the WGAN: a learned critic against exact $W_1$ between samples, and minibatch transport (exact, and your Sinkhorn divergence) against the full-sample values, including for two samples of the same distribution. experiment 2 adapts a nearest-neighbour classifier between two-moons domains rotated by 20° to 90°. experiment 3 is the block's **project**: flow matching on the 8 Gaussians with the independent coupling and minibatch OT couplings of three block sizes, three training seeds each, measuring pairing cost, straightness and sample quality at 1 to 64 Euler steps against the sampling floor.

## check your understanding

1. why does the flow-matching loss *not* go to zero with independent pairs, and why does it get much smaller with OT pairs?

<details><summary>answer</summary>

the loss is a regression of $x_1 - x_0$ on $(x_t, t)$; its minimum is the average conditional variance of $x_1 - x_0$ given $x_t$. with independent pairs many lines with different velocities pass through each point, so that variance is large and irreducible (ch 22). with OT pairs within a batch the lines don't cross, so given $x_t$ the velocity is nearly determined: the conditional variance, and the loss, shrink towards 0 as the block size grows.

</details>

2. a minibatch OT coupling gives straight paths but the samples after 64 steps aren't better than with the independent coupling. why isn't that a contradiction?

<details><summary>answer</summary>

ch 22's on paper 1, and this topic's: every coupling defines a flow from the same noise to the same data, so in the limit of a perfect network and many steps they all generate the data. the coupling changes the path, which matters when you integrate with few steps. at many steps, quality is limited by how well the network fits its velocity field, and the OT field can be *harder* to fit: the OT map from a Gaussian onto separated modes is discontinuous at the boundaries between the regions sent to each mode.

</details>

3. why is the debiased Sinkhorn divergence between two minibatches from the same distribution not zero?

<details><summary>answer</summary>

debiasing removes the entropic bias, $\mathrm{OT}_\varepsilon(\mu, \mu) > 0$: $S_\varepsilon(\mu, \mu) = 0$ for the *same* distribution. two minibatches are two *different* empirical distributions, and $S_\varepsilon$ correctly reports a positive distance between them. how large depends on $\varepsilon$: near 0 it behaves like $W_2^2$, with sampling error $\sim n^{-2/d}$; for large $\varepsilon$ like an MMD, with error $\sim n^{-1/2}$ in any dimension (Genevay et al., 2019).

</details>

4. a WGAN-GP critic reports a slightly *larger* value than the exact $W_1$ between the same samples. how is that possible?

<details><summary>answer</summary>

the gradient penalty only pushes the gradient norm towards 1 on random interpolates between real and fake points; elsewhere (and even there, softly) the critic can be steeper than 1-Lipschitz, and a function that isn't 1-Lipschitz can exceed the Kantorovich–Rubinstein supremum. the estimate also depends on training time and on the network class, which pull it the other way.

</details>

5. when would you choose `sinkhorn_coupling` over `minibatch_ot_coupling` for flow matching?

<details><summary>answer</summary>

when the exact assignment is too slow (the Hungarian algorithm is $O(B^3)$ and sequential, while Sinkhorn is $O(B^2)$ per iteration and runs on a GPU), when you want a knob between independent and OT pairings, or when you want the Schrödinger-bridge interpretation with noisy interpolants. the cost is less straightness at a given batch size, plus Sinkhorn's own iterations.

</details>

6. domain adaptation by OT makes accuracy on a 90°-rotated target go from 0.27 to about 0.52. is that a success?

<details><summary>answer</summary>

it's better than doing nothing (0.27 is worse than chance: the rotation swapped the moons' positions), but 0.52 is chance level for two balanced classes, far from the 0.999 of a classifier with target labels. the cheapest transport onto a strongly rotated target doesn't follow the rotation, so the labels arrive mixed. a large shift needs label information in the coupling, or a few target labels.

</details>

## common bugs

- **the inverse permutation.** `linear_sum_assignment(C)` on $C[k, i] = \|x_1^k - x_0^i\|^2$ returns, for each data point $k$, the noise index for slot $k$: use it directly. on $C[i, k]$ (noise rows) you'd need its inverse. a shuffled copy of `x0` as `x1` must come back exactly.
- **sampling rows instead of columns** in the entropic coupling: you then draw a data partner for each *noise* point but put it in the data point's slot. the pairs no longer follow the plan.
- **backpropagating through Sinkhorn's iterations or both $c$-transforms**: slow, memory-hungry, or a doubled gradient. build the dual from detached potentials and a differentiable cost.
- **dropping the mass term** of the dual in the loss: at convergence it's 0, so the value looks right, but before convergence the value is wrong and the gradient misses $P$ entirely.
- **comparing couplings on one seed, or on a raw sample-to-sample $W_2$ without its floor.** the per-seed spread of the 64-step quality is as large as the differences between couplings; only paired, multi-seed comparisons above the floor support a claim.

## going deeper

- Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf & Bengio (2024), "Improving and generalizing flow-based generative models with minibatch optimal transport", *TMLR*. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
- Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman & Chen (2023), "Multisample flow matching: straightening flows with minibatch couplings", *ICML*. [arXiv:2304.14772](https://arxiv.org/abs/2304.14772)
- Liu, Gong & Liu (2023), "Flow straight and fast: learning to generate and transfer data with rectified flow", *ICLR*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- McCann (1997), "A convexity principle for interacting gases", *Advances in Mathematics* 128(1). displacement interpolation. [doi:10.1006/aima.1997.1634](https://doi.org/10.1006/aima.1997.1634)
- Benamou & Brenier (2000), "A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem", *Numerische Mathematik* 84(3). [doi:10.1007/s002110050002](https://doi.org/10.1007/s002110050002)
- Arjovsky, Chintala & Bottou (2017), "Wasserstein generative adversarial networks", *ICML*. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875); Gulrajani, Ahmed, Arjovsky, Dumoulin & Courville (2017), "Improved training of Wasserstein GANs", *NeurIPS*. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
- Genevay, Peyré & Cuturi (2018), "Learning generative models with Sinkhorn divergences", *AISTATS*. [arXiv:1706.00292](https://arxiv.org/abs/1706.00292); Genevay, Chizat, Bach, Cuturi & Peyré (2019), "Sample complexity of Sinkhorn divergences", *AISTATS*. [arXiv:1810.02733](https://arxiv.org/abs/1810.02733)
- Fatras, Zine, Flamary, Gribonval & Courty (2020), "Learning with minibatch Wasserstein: asymptotic and gradient properties", *AISTATS*. [arXiv:1910.04091](https://arxiv.org/abs/1910.04091)
- Courty, Flamary, Tuia & Rakotomamonjy (2017), "Optimal transport for domain adaptation", *IEEE TPAMI* 39(9). [arXiv:1507.00504](https://arxiv.org/abs/1507.00504); Courty, Flamary, Habrard & Rakotomamonjy (2017b), "Joint distribution optimal transportation for domain adaptation", *NeurIPS*. [arXiv:1705.08848](https://arxiv.org/abs/1705.08848)

## connections

- ← builds on: exact OT and barycentric maps (topic 15); log-domain Sinkhorn and the debiased divergence (topic 16); flow matching and straightness (ch 22 · topic 05); WGAN-GP (ch 21 · topic 06).
- → used in: the chapter capstone's portfolio entry for this block takes the project's paired, per-seed numbers. after this block you may use POT (`ot.emd`, `ot.sinkhorn`, `ot.da`), GeomLoss's `SamplesLoss` and the `torchcfm` package's OT-CFM samplers, since you've built each piece.
