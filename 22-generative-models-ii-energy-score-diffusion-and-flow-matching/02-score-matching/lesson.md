# score matching

## why this matters

topic 01's energy-based model had a strange training loop: every gradient step ran an MCMC sampler, the loss's quality depended on how well that sampler had mixed, and the model ended up fitted to its sampler rather than to the data. and all of that machinery existed to serve one quantity. Langevin dynamics never looks at the energy, only at its gradient, the score $\nabla_x \log p(x)$.

so why learn an energy at all? train a network $s_\theta(x)$ to output the score *directly*, by regression. the obstacle is obvious: regression needs targets, and the true score of the data distribution is exactly what you don't know. this topic is about two ways around that obstacle, both simple enough to fit in a few lines. **sliced score matching** turns the unknown target into a random projection of the network's own Jacobian. **denoising score matching** adds a little Gaussian noise to each data point and regresses on the direction back to the clean point, which is known. the second one, scaled up to a whole range of noise levels and paired with a sampler that walks down those levels, was the first generative model to rival GANs on images (Song & Ermon, 2019), and it's a diffusion model in all but name.

## warm-up: predict first

1. the score of $\mathcal{N}(\mu, s^2)$ is $-(x - \mu)/s^2$. what's the score of a mixture $0.2\,\mathcal{N}(-5, 1) + 0.8\,\mathcal{N}(5, 1)$ at $x = -5$? does it depend on the weights 0.2 and 0.8?
2. you blur the data with noise of standard deviation $\sigma$. what's the score of the blurred distribution if the data were $\mathcal{N}(0, s^2)$?
3. a perfect score network trained by denoising has a training loss of exactly 0, small but positive, or large?
4. you learn a score from samples, then run Langevin dynamics from random starting points far from the data. where is the learned score least trustworthy?
5. Langevin with the *exact* score of the two-component mixture above, chains started uniformly in $[-8, 8]$: what fraction ends up near $-5$?

## prerequisites check

- [ ] the score and Langevin dynamics, `langevin_dynamics` (→ topic 01)
- [ ] integration by parts in one and several variables, the divergence theorem (→ ch 03 · topic 04)
- [ ] conditional expectations and the Gaussian posterior (→ ch 04 · topic 04; ch 12 · topic 01)
- [ ] `torch.autograd.grad` with `create_graph=True` for second derivatives (→ ch 15 · topic 07)
- [ ] the trace of a matrix as $\mathbb{E}[v^\top A v]$ for random $v$ (→ ch 02 · topic 05)

## core ideas

### idea 1: denoising score matching

the ideal objective is the **Fisher divergence** between the model's score and the data's,

$$J(\theta) = \tfrac12\,\mathbb{E}_{x \sim p}\,\big\lVert s_\theta(x) - \nabla_x \log p(x)\big\rVert^2,$$

which can't be computed because $\nabla\log p$ is unknown. now perturb every data point with Gaussian noise, $\tilde x = x + \sigma z$ with $z \sim \mathcal{N}(0, I)$. the perturbed data have a density $q_\sigma(\tilde x) = \int \mathcal{N}(\tilde x; x, \sigma^2 I)\,p(x)\,dx$, a blurred version of $p$, and Vincent (2011) showed that matching *its* score is equivalent, up to a constant independent of $\theta$, to a regression with a known target (on paper 2):

$$\tfrac12\,\mathbb{E}_{\tilde x \sim q_\sigma}\big\lVert s_\theta(\tilde x) - \nabla\log q_\sigma(\tilde x)\big\rVert^2 = \tfrac12\,\mathbb{E}_{x, z}\Big\lVert s_\theta(x + \sigma z) - \underbrace{\Big(-\frac{\tilde x - x}{\sigma^2}\Big)}_{=\,-z/\sigma}\Big\rVert^2 + C.$$

the target $-(\tilde x - x)/\sigma^2$ is the score of the noise kernel: it points from the noisy point straight back to the clean one. a single noisy point could have come from many clean ones, so the network can't hit every target; the best it can do is their average, and that average is the true score of $q_\sigma$. the same fact, read the other way, is **Tweedie's formula**: $\mathbb{E}[x \mid \tilde x] = \tilde x + \sigma^2 \nabla\log q_\sigma(\tilde x)$. a score model *is* a denoiser (on paper 3).

two consequences you'll measure. first, the loss can never reach 0: the leftover is the irreducible uncertainty about which clean point produced $\tilde x$. for Gaussian data $\mathcal{N}(\mu, s^2 I)$ in $D$ dimensions, the exact score of the blurred data is $-(\tilde x - \mu)/(s^2 + \sigma^2)$, and the loss at that optimum, with the weighting below, is $\frac12 D\, s^2/(s^2 + \sigma^2)$. second, you learn the score of the *blurred* distribution, not of $p$: small $\sigma$ means a small bias but noisy, high-variance targets.

to train one network for several noise levels (idea 3), the losses at different $\sigma$ must be on the same scale. the target $-z/\sigma$ has size $\sim 1/\sigma$, so the squared error is $\sim 1/\sigma^2$; multiplying by $\lambda(\sigma) = \sigma^2$ makes every level count equally:

$$\mathcal{L}_\text{DSM} = \frac1B \sum_b \tfrac12\,\sigma_b^2\,\big\lVert s_\theta(\tilde x_b, \sigma_b) + z_b/\sigma_b\big\rVert^2.$$

> **spec · stage 1.** `dsm_loss(score_fn, x, sigma, *, generator=None)`: `x` `(B, ...)`; `sigma` a float or a `(B,)` tensor (one level per example). draw `z = torch.randn(x.shape, generator=generator)` (CPU, moved to `x`'s device), form $\tilde x = x + \sigma z$ (broadcasting `sigma` over the non-batch dims), call `score_fn(x_tilde, sigma_vec)` with `sigma_vec` a `(B,)` tensor even if `sigma` was a float, and return the mean over the batch of $\frac12\sigma_b^2 \sum (s - \text{target})^2$, the sum over every non-batch dimension, with target $-(\tilde x - x)/\sigma^2$. this is `mlfs.diffusion.dsm_loss`.

> **▶ build: stage 1.** implement `dsm_loss` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** Gaussian data with $s^2 = (0.25, 1)$ in two dimensions, $\sigma = 0.5$: what's the loss of the *exact* noisy score? (it isn't zero.)

### idea 2: sliced score matching

can you match the score of $p$ itself, without blurring? Hyvärinen (2005) integrated the cross term of the Fisher divergence by parts (on paper 1):

$$J(\theta) = \mathbb{E}_{x \sim p}\Big[\operatorname{tr}\big(\nabla_x s_\theta(x)\big) + \tfrac12\lVert s_\theta(x)\rVert^2\Big] + C,$$

no unknown score anywhere. the catch is the trace of the Jacobian: $D$ diagonal entries, each a separate backward pass, which is 3,072 passes for a CIFAR image. **sliced score matching** (Song, Garg, Shi & Ermon, 2019) uses the Hutchinson trick: for a random vector $v$ with $\mathbb{E}[vv^\top] = I$, $\mathbb{E}_v[v^\top A v] = \operatorname{tr}(A)$, and $v^\top \nabla_x s\,v = v^\top \nabla_x (v^\top s)$ needs only *one* backward pass, of the scalar $v^\top s$:

$$\mathcal{L}_\text{SSM} = \mathbb{E}_{x}\,\mathbb{E}_v\Big[v^\top \nabla_x\big(v^\top s_\theta(x)\big) + \tfrac12\big(v^\top s_\theta(x)\big)^2\Big].$$

(using $\frac12(v^\top s)^2$ in place of $\frac12\lVert s\rVert^2$ is the paper's version; its expectation over $v$ is the same, $\frac12\lVert s\rVert^2$.) the loss contains a gradient, so training it needs a gradient of a gradient: compute $\nabla_x(v^\top s)$ with `create_graph=True`, or the trace term silently stops contributing to the parameter gradient. for a linear score $s(x) = -Ax$ the expected loss is exactly $\mathbb{E}[\frac12\lVert Ax\rVert^2] - \operatorname{tr}(A)$, which the checks use.

SSM learns the score of the clean data, without blur; its price is a noisier estimator and a double backward pass. DSM's targets are cheap and its blur is often a feature (idea 3). the rest of the chapter uses DSM and its descendants.

> **spec · stage 2.** `sliced_score_matching_loss(score_fn, x, *, n_projections=1, generator=None)`: `score_fn(x)` maps `(B, ...)` to the same shape (no noise level). detach a copy of `x` with `requires_grad_`; draw `v = torch.randn((n_projections, *x.shape), generator=generator)` once; for each projection, per example, $v^\top \nabla_x(v^\top s) + \frac12 (v^\top s)^2$ (sums over the non-batch dims); return the mean over projections and the batch, differentiable in the score's parameters. this is `mlfs.diffusion.sliced_score_matching_loss`.

> **▶ build: stage 2.** implement `sliced_score_matching_loss`.

### idea 3: one network, many noise levels

two failures of a single-$\sigma$ score model show up as soon as you sample with it (Song & Ermon, 2019):

- **the score is wrong where there's no data.** the loss is an expectation over the data distribution, so the network is only ever corrected where samples land. far from the data, where Langevin chains *start*, the learned score is extrapolation.
- **Langevin can't get the proportions right.** in a mixture with well-separated modes, the score near each mode is just that component's score: the weights only enter through the region between the modes, where the density is tiny (warm-up question 1). chains started at random settle into whichever mode they're closest to and practically never cross, so the fraction in each mode is set by the starting points, not by the weights.

blurring fixes both. with a large $\sigma$ the blurred density fills the space, so the score is learned (and accurate) everywhere, and the modes merge so that chains can move between them. with a small $\sigma$ the blurred density is close to the data. so learn them all: a **noise-conditional score network** $s_\theta(x, \sigma)$ trained with DSM on a ladder of levels $\sigma_1 > \sigma_2 > \dots > \sigma_L$, usually geometric ($\sigma_i / \sigma_{i+1}$ constant), each example getting a random level. a useful design choice from NCSNv2 (Song & Ermon, 2020): since the true score has magnitude about $1/\sigma$, let the network output $f_\theta(x, \log\sigma)/\sigma$, so the raw network works on a constant scale.

for Gaussian data the target is known in closed form at every level, $-(\tilde x - \mu)/(s^2 + \sigma^2)$, and the checks compare your trained network with it.

> **spec · stage 3.**
> - `geometric_sigmas(sigma_max, sigma_min, n)`: a float32 tensor `(n,)` from `sigma_max` down to `sigma_min` with a constant ratio.
> - `NoiseConditionalScore(data_dim=2, *, hidden_dim=128, n_layers=3)`: an MLP with `n_layers` SiLU hidden layers of width `hidden_dim` on the concatenation $[x, \log\sigma]$ (input width `data_dim + 1`), output width `data_dim`, divided by $\sigma$. `forward(x, sigma)`: `x` `(B, data_dim)`, `sigma` a float or `(B,)`.
> - `fit_dsm(model, X, sigmas, *, n_iters=2000, batch_size=256, lr=1e-3, seed=0)`: `g = torch.Generator().manual_seed(seed)`, `torch.optim.Adam(lr)`; per step, `X[torch.randint(N, (batch_size,), generator=g)]`, one level per example `sigmas[torch.randint(len(sigmas), (batch_size,), generator=g)]`, and `dsm_loss(model, batch, sigma_vec, generator=g)`. returns the `n_iters` losses as floats.

> **▶ build: stage 3.** implement `geometric_sigmas`, `NoiseConditionalScore` and `fit_dsm`.

### idea 4: annealed Langevin dynamics

sampling walks down the ladder: run Langevin at $\sigma_1$, where the score is good everywhere and the modes are connected; use the result as the starting point at $\sigma_2$; and so on down to $\sigma_L$. each level's chains start close to that level's distribution, so a few dozen steps per level suffice. the step size shrinks with the level, in proportion to $\sigma_i^2$ (the scale of the blurred distribution's local variance):

$$\alpha_i = \epsilon\,\frac{\sigma_i^2}{\sigma_L^2}, \qquad x \leftarrow x + \frac{\alpha_i}{2}\,s_\theta(x, \sigma_i) + \sqrt{\alpha_i}\,z,$$

with $\epsilon$ the step at the smallest level (Song & Ermon, 2019). this sampler is the bridge to diffusion: replace the ladder by a continuum of levels and the Langevin steps by the exact reverse of a noising process, and you have topics 03 and 04.

> **spec · stage 4.** `annealed_langevin(score_fn, x0, sigmas, *, steps_per_sigma=100, eps=2e-05, generator=None)`: for each $\sigma_i$ in `sigmas` in the given (decreasing) order, `steps_per_sigma` Langevin steps with $\alpha_i = \epsilon\,\sigma_i^2/\sigma_L^2$ ($\sigma_L$ = the last entry), calling `score_fn(x, sigma_vec)` with `sigma_vec = torch.full((B,), sigma_i)` and drawing `z = torch.randn(x.shape, generator=generator)` per step; no gradients; doesn't modify `x0`; returns the final `x`. this is `mlfs.diffusion.annealed_langevin`.

> **▶ build: stage 4.** implement `annealed_langevin`.

> **🔮 predict:** with the exact scores of the 0.2 / 0.8 mixture, annealed Langevin from $\sigma = 20$ down to $\sigma = 0.1$: what fraction of chains ends near $-5$ now?

## worked example

**one DSM term by hand.** one-dimensional data $\mathcal{N}(0, 1)$ and $\sigma = 0.5$. a clean point $x = 1$ with noise draw $z = 0.4$ gives $\tilde x = 1.2$ and target $-z/\sigma = -0.8$. the blurred data are $\mathcal{N}(0, 1 + 0.25)$, so the exact score there is $-1.2/1.25 = -0.96$, and this example's loss is $\frac12 \cdot 0.25 \cdot (-0.96 + 0.8)^2 = 0.0032$. a different clean point could have produced the same $\tilde x$ with a target anywhere around $-0.96$: averaged over all of them, the loss at the exact score is $\frac12 \cdot \frac{1}{1.25} = 0.4$, the irreducible floor $\frac12 s^2/(s^2 + \sigma^2)$.

**the floor in two dimensions.** with $s^2 = (0.25, 1)$ and $\sigma = 0.5$: $\frac12(0.25/0.5 + 1/1.25) = \frac12(0.5 + 0.8) = 0.65$. the checks compute your `dsm_loss` at the exact score on 40,000 points and expect 0.65; they also check that multiplying the exact score by 0.8 or 1.2 makes it worse.

**sliced score matching on a linear score.** for $s(x) = -Ax$ with $A = \begin{pmatrix} 2 & 0.5 \\ 0.5 & 1\end{pmatrix}$: $\nabla_x s = -A$, so the trace term is $-\operatorname{tr}A = -3$ for every $x$, and the loss is $\mathbb{E}[\frac12\lVert Ax\rVert^2] - 3$. with 2,000 projections, your estimate must land within a few percent of it.

## on paper

1. **Hyvärinen's trick in one dimension.** expand $\frac12\mathbb{E}_p[(s_\theta - (\log p)')^2]$ and show that the cross term $-\mathbb{E}_p[s_\theta\,(\log p)']$ equals $\mathbb{E}_p[s_\theta']$, assuming $p(x)s_\theta(x) \to 0$ as $|x| \to \infty$.

<details><summary>solution</summary>

$\mathbb{E}_p[s_\theta (\log p)'] = \int s_\theta\,p\,\frac{p'}{p}\,dx = \int s_\theta\,p'\,dx = [s_\theta p]_{-\infty}^{\infty} - \int s_\theta'\,p\,dx = -\mathbb{E}_p[s_\theta']$. so the objective is $\mathbb{E}_p[\frac12 s_\theta^2 + s_\theta'] + \frac12\mathbb{E}_p[((\log p)')^2]$, and the last term doesn't depend on $\theta$. in $D$ dimensions the same integration by parts (the divergence theorem) gives $\operatorname{tr}\nabla_x s_\theta$.

</details>

2. **denoising = matching the blurred score.** show that $\mathbb{E}_{q_\sigma}\lVert s - \nabla\log q_\sigma\rVert^2$ and $\mathbb{E}_{x, \tilde x}\lVert s(\tilde x) - \nabla_{\tilde x}\log q(\tilde x \mid x)\rVert^2$ differ by a constant.

<details><summary>solution</summary>

expand both squares. the $\lVert s\rVert^2$ terms are the same expectation over $\tilde x \sim q_\sigma$. the cross terms: $\mathbb{E}_{q_\sigma}[s^\top\nabla\log q_\sigma] = \int s^\top\nabla q_\sigma\,d\tilde x = \int s^\top \int p(x)\nabla_{\tilde x} q(\tilde x \mid x)\,dx\,d\tilde x = \mathbb{E}_{x, \tilde x}[s(\tilde x)^\top \nabla_{\tilde x}\log q(\tilde x \mid x)]$, using $\nabla q = q\nabla \log q$ twice. the remaining terms don't involve $s$. so both objectives have the same minimizer, $s^* = \nabla\log q_\sigma$.

</details>

3. **Tweedie's formula and the loss floor.** using the previous result, show $\nabla\log q_\sigma(\tilde x) = (\mathbb{E}[x \mid \tilde x] - \tilde x)/\sigma^2$, and that the weighted loss at the optimum is $\frac12\sum_d \operatorname{Var}(z_d \mid \tilde x)$ averaged over $\tilde x$. evaluate it for $\mathcal{N}(\mu, s^2)$ data.

<details><summary>solution</summary>

the regression's minimizer is the conditional mean of the target: $s^*(\tilde x) = \mathbb{E}[-(\tilde x - x)/\sigma^2 \mid \tilde x] = (\mathbb{E}[x \mid \tilde x] - \tilde x)/\sigma^2$. with the $\sigma^2$ weight, the loss is $\frac12\mathbb{E}\lVert \sigma s + z\rVert^2 = \frac12\mathbb{E}\lVert z - \mathbb{E}[z \mid \tilde x]\rVert^2$ (since $\sigma s^* = -\mathbb{E}[z \mid \tilde x]$), the expected conditional variance. for Gaussian data, $(z, \tilde x)$ are jointly Gaussian with $\operatorname{Cov}(z, \tilde x) = \sigma$ and $\operatorname{Var}\tilde x = s^2 + \sigma^2$ per dimension, so $\operatorname{Var}(z \mid \tilde x) = 1 - \sigma^2/(s^2 + \sigma^2) = s^2/(s^2 + \sigma^2)$, and the floor is $\frac12 D\, s^2/(s^2 + \sigma^2)$: near $\frac12 D$ when $\sigma \ll s$, near 0 when $\sigma \gg s$.

</details>

4. **Hutchinson's estimator.** show $\mathbb{E}_v[v^\top A v] = \operatorname{tr}A$ for $v \sim \mathcal{N}(0, I)$, and that $v^\top(\nabla_x s)v = v^\top\nabla_x(v^\top s)$ when $v$ doesn't depend on $x$.

<details><summary>solution</summary>

$\mathbb{E}[v^\top A v] = \mathbb{E}[\operatorname{tr}(A vv^\top)] = \operatorname{tr}(A\,\mathbb{E}[vv^\top]) = \operatorname{tr}A$. and $\nabla_x(v^\top s) = (\nabla_x s)^\top v$ (a vector-Jacobian product, one backward pass), so $v^\top\nabla_x(v^\top s) = v^\top(\nabla_x s)^\top v = v^\top(\nabla_x s)v$. Rademacher $v$ (random $\pm 1$) also works and has lower variance; the spec uses Gaussians.

</details>

## experiment

`lab.py` uses your losses and samplers on 2-D data with known scores:

1. Langevin dynamics with the exact score of a lopsided mixture, and annealed Langevin with the exact blurred scores.
2. where a learned score is accurate, and where it isn't.
3. DSM at one noise level against many, and against sliced score matching.

## check your understanding

1. why does denoising score matching need no knowledge of $\nabla \log p$?

<details><summary>answer</summary>

its target is the score of the Gaussian noise kernel, $-(\tilde x - x)/\sigma^2$, known because you added the noise yourself. averaging that target over all clean points consistent with $\tilde x$ produces the true blurred score; the regression does the averaging for you.

</details>

2. what does a score network learned with DSM at noise level $\sigma$ sample, if you run exact Langevin with it?

<details><summary>answer</summary>

the blurred distribution $q_\sigma = p * \mathcal{N}(0, \sigma^2 I)$, not $p$. for Gaussian data that's variance $s^2 + \sigma^2$ instead of $s^2$. that's why the last level of the annealing ladder is small, and why NCSN-style samplers often end with one Tweedie denoising step $x + \sigma^2 s(x, \sigma)$.

</details>

3. why weight the DSM loss by $\sigma^2$?

<details><summary>answer</summary>

the target $-z/\sigma$ has magnitude about $1/\sigma$, so unweighted squared errors at small $\sigma$ are $1/\sigma^2$ times larger and dominate the gradient of a multi-level loss. with $\lambda = \sigma^2$ the loss at every level is $\frac12\lVert\sigma s + z\rVert^2$, on the scale of a unit-variance noise prediction: the same quantity a diffusion model's $\epsilon$-prediction loss measures (topic 03).

</details>

4. what breaks if you compute $\nabla_x(v^\top s)$ in sliced score matching without `create_graph=True`?

<details><summary>answer</summary>

the gradient is returned as a constant, so the trace term contributes nothing to the parameter gradient. the network then minimizes only $\frac12(v^\top s)^2$, whose minimum is $s = 0$: training drives the score to zero.

</details>

5. Langevin with the exact score gets the mixture's proportions wrong. what makes annealing fix it?

<details><summary>answer</summary>

at large $\sigma$ the blurred mixture is nearly one wide blob whose shape still reflects the weights (more mass on the heavier side), and Langevin mixes across it quickly. as $\sigma$ shrinks, the chains are already distributed between the modes in the right proportions and only need to sharpen locally; no chain has to cross a low-density gap.

</details>

6. why does the score network output $f(x, \log\sigma)/\sigma$ instead of $f(x, \sigma)$?

<details><summary>answer</summary>

the target magnitude varies by orders of magnitude across the ladder (from about $1/\sigma_1$ to $1/\sigma_L$); dividing by $\sigma$ makes the raw output $f$ about unit-sized at every level, which makes it far easier to learn. conditioning on $\log\sigma$ spreads the geometric ladder evenly over the network's input.

</details>

## common bugs

- **the target's sign.** symptom: Langevin samples explode outward; the target is $-(\tilde x - x)/\sigma^2$, pointing from the noisy point back to the clean one.
- **one $\sigma$ for the whole batch when you meant one per example.** symptom: fine for a float, but with a `(B,)` tensor you broadcast it over the wrong dimension (a `(B,)` against `(B, D)` silently broadcasts along $D$ when $B = D$).
- **forgetting the $\sigma^2$ weight.** symptom: the multi-level loss is dominated by the smallest level, and large-$\sigma$ scores are poor, which ruins the start of annealed sampling.
- **detaching inside sliced score matching.** symptom: the score collapses to zero during training (see question 4).
- **annealed step size inverted.** symptom: huge steps at the smallest level, samples blown apart at the end; $\alpha_i \propto \sigma_i^2$, largest at the start.
- **a sampler that builds a graph.** symptom: memory grows with the number of steps; sample under `torch.no_grad()`.

## going deeper

- Hyvärinen (2005), "Estimation of non-normalized statistical models by score matching", *JMLR* 6. [jmlr.org/papers/v6/hyvarinen05a.html](https://jmlr.org/papers/v6/hyvarinen05a.html)
- Vincent (2011), "A connection between score matching and denoising autoencoders", *Neural Computation* 23(7). [doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
- Song, Garg, Shi & Ermon (2019), "Sliced score matching: a scalable approach to density and score estimation", *UAI*. [arXiv:1905.07088](https://arxiv.org/abs/1905.07088)
- Song & Ermon (2019), "Generative modeling by estimating gradients of the data distribution", *NeurIPS*. NCSN and annealed Langevin. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
- Song & Ermon (2020), "Improved techniques for training score-based generative models", *NeurIPS*. NCSNv2: noise ladders, $1/\sigma$ output scaling, the final denoising step. [arXiv:2006.09011](https://arxiv.org/abs/2006.09011)
- Efron (2011), "Tweedie's formula and selection bias", *JASA* 106(496). [doi:10.1198/jasa.2011.tm11181](https://doi.org/10.1198/jasa.2011.tm11181)

## connections

- ← builds on: the score and Langevin dynamics (topic 01), integration by parts (ch 03), Gaussian conditioning (ch 04, ch 12), Hutchinson trace estimation (ch 02).
- → used in: DDPM, whose noise-prediction loss is denoising score matching on a continuum of noise levels with $\epsilon = -\sigma s$ (topic 03); the probability-flow ODE and reverse-time SDE, which consume exactly these scores (topic 04); guidance, which adds the score of a classifier (topic 06).
