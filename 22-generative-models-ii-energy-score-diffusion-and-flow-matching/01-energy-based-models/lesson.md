# energy-based models

## why this matters

every model in chapter 21 paid for its likelihood with an architectural straitjacket. a PixelCNN must be causal, a flow must be invertible with a cheap Jacobian, a VAE must come with an encoder and settle for a bound. what if you dropped all of that, took *any* network $E_\theta(x)$ that outputs one number, and declared

$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}, \qquad Z_\theta = \int e^{-E_\theta(x)}\,dx?$$

low energy means likely, high energy means unlikely, and the network is free to be any shape you like. this is an **energy-based model** (EBM), the Boltzmann distribution of statistical physics with a neural network for the energy. the price is $Z_\theta$: an integral over every possible image, hopeless to compute. you can't evaluate $p_\theta(x)$, you can't sample by inverting anything, and the maximum-likelihood gradient seems to need $Z_\theta$ too.

this topic shows how to live without $Z_\theta$. the trick is that the *gradient* of the log-density, $\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$, doesn't involve $Z_\theta$ at all, and that gradient is enough to sample (Langevin dynamics) and, with samples, to train (contrastive divergence). this gradient, the **score**, is the thread that runs through the whole chapter: score matching learns it directly (topic 02), diffusion models learn it at every noise level (topics 03–04), and flow matching learns a close relative (topic 05).

## warm-up: predict first

1. you double every energy, $E \to 2E$. what happens to the distribution $p \propto e^{-E}$? to its modes?
2. a sampler moves $x$ a little uphill in $\log p$ at every step and adds a bit of noise. without the noise, where does it end up? with too much noise?
3. Langevin dynamics with a step size $h$ samples a Gaussian of variance 1. will the samples' variance be exactly 1, a little more, or a little less?
4. the maximum-likelihood gradient of an EBM is "push the energy of data down, push the energy of the model's own samples up". what happens when the model's samples look exactly like the data?
5. you train an EBM with samplers that run only 20 steps from noise. after training you run the same sampler for 2,000 steps. better samples, or worse?

## prerequisites check

- [ ] Markov chains, stationary distributions, Brownian motion (→ ch 04 · topic 07)
- [ ] Metropolis–Hastings and acceptance ratios (→ ch 12 · topic 05); Hamiltonian Monte Carlo's use of $\nabla \log p$ (→ ch 12 · topic 06)
- [ ] exponential families: the maximum-likelihood gradient is "data moments minus model moments" (→ ch 05 · topic 06)
- [ ] `torch.autograd.grad` with respect to inputs, `torch.enable_grad` (→ ch 15 · topic 07); Adam (→ ch 06 · topic 04)
- [ ] chapter 21's view of likelihood, sampling and evaluation (→ ch 21 · topic 01)

## core ideas

### idea 1: the score, and Langevin dynamics

take logs: $\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$. the normalizer is a constant in $x$, so it disappears when you differentiate with respect to the input:

$$s_\theta(x) := \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x).$$

the **score** $s(x)$ is a vector field that points towards higher probability, computed with one backward pass. how do you turn a vector field into samples? gradient ascent alone climbs to the nearest mode and stops: you'd get the single most likely image, every time. the fix is to add exactly the right amount of noise. **Langevin dynamics** iterates

$$x_{k+1} = x_k + \frac{h}{2}\, s(x_k) + \sqrt{h}\, z_k, \qquad z_k \sim \mathcal{N}(0, I),$$

with a small step size $h$. the drift pulls samples up the density, the noise spreads them out, and the balance between the two is the target distribution itself. in continuous time ($h \to 0$) this is the stochastic differential equation $dX_t = \tfrac12 \nabla \log p(X_t)\,dt + dW_t$, whose stationary distribution is exactly $p$ (on paper 1). the $\tfrac12$ against the $\sqrt h$ is not a convention you can tweak: halve the drift or double the noise and the stationary distribution changes.

for a finite step, the discretization is slightly wrong. on a Gaussian $\mathcal{N}(0, s^2)$ the score is $-x/s^2$, the update becomes $x_{k+1} = (1 - \tfrac{h}{2s^2})x_k + \sqrt h z_k$, a linear (AR(1)) recursion, and its stationary variance comes out as $s^2 / (1 - h/(4s^2))$ instead of $s^2$ (on paper 2). this is the **unadjusted Langevin algorithm** (ULA): fast, simple and biased by an amount that grows with $h$.

> **spec · stage 1.**
> - `score_from_energy(energy_fn)`: returns a function `score_fn(x)` $= -\nabla_x \sum_b E(x)_b$ (same shape as `x`, detached), where `energy_fn` maps `(B, ...)` to energies `(B,)`. it must work when called inside `torch.no_grad()`: turn gradients on locally with `torch.enable_grad()`.
> - `langevin_dynamics(score_fn, x0, *, step_size=0.01, n_steps=100, generator=None)`: `n_steps` updates $x \leftarrow x + \frac h2 s(x) + \sqrt h\,z$, one draw `z = torch.randn(x.shape, generator=generator)` per step (on the CPU, moved to `x`'s device); doesn't modify `x0`; builds no graph; returns the final `x`. this is `mlfs.diffusion.langevin_dynamics`.

> **▶ build: stage 1.** implement `score_from_energy` and `langevin_dynamics` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** Langevin dynamics on $\mathcal{N}(0, 1)$ with step $h = 1$: what variance do the samples settle at? and with $h = 4$?

### idea 2: removing the bias with a Metropolis correction

chapter 12 gave you a way to make any proposal exact: accept or reject it with the Metropolis–Hastings ratio. a Langevin step is a proposal from a Gaussian centred at $x + \frac h2 s(x)$,

$$q(x' \mid x) = \mathcal{N}\big(x';\ x + \tfrac h2 s(x),\ hI\big), \qquad \log q(x' \mid x) = -\frac{\lVert x' - x - \frac h2 s(x)\rVert^2}{2h} + \text{const},$$

and the proposal is not symmetric ($q(x' \mid x) \ne q(x \mid x')$ because the drift differs at the two points), so the full Hastings ratio is needed:

$$\log \alpha = \big[E(x) - E(x')\big] + \log q(x \mid x') - \log q(x' \mid x).$$

accept with probability $\min(1, \alpha)$, otherwise stay put. this is **MALA**, the Metropolis-adjusted Langevin algorithm (Roberts & Tweedie, 1996). it samples $p$ exactly for any step size: large steps get rejected more often instead of biasing the result. note what it needs: energy *differences* $E(x) - E(x')$, which the unknown $Z_\theta$ cancels out of, just like in chapter 12's random-walk Metropolis.

why not always use MALA? on a trained EBM in high dimension the acceptance rate collapses unless $h$ is tiny, and every step needs an extra energy evaluation. practitioners mostly run unadjusted Langevin and accept the bias; you'll see in the lab what that bias does on a mixture.

> **spec · stage 2.** `mala(energy_fn, x0, *, step_size=0.01, n_steps=100, generator=None)`, with `energy_fn` mapping `(B, ...)` to `(B,)`. per step: `z = torch.randn(x.shape, generator=generator)`, proposal $x' = x - \frac h2 \nabla E(x) + \sqrt h z$, then `u = torch.rand(B, generator=generator)`; row $b$ moves to $x'_b$ if $\log u_b < E(x_b) - E(x'_b) + \log q(x_b \mid x'_b) - \log q(x'_b \mid x_b)$ and stays otherwise. returns `(x, acceptance_rate)`: the final `x` (detached) and the fraction of accepted moves over all rows and steps, a python float.

> **▶ build: stage 2.** implement `mala`.

### idea 3: contrastive divergence

now training. the log-likelihood of one example is $\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$, and differentiating $\log Z_\theta = \log \int e^{-E_\theta(x')}dx'$ gives an expectation under the model (on paper 3):

$$\nabla_\theta \log Z_\theta = -\mathbb{E}_{x' \sim p_\theta}\big[\nabla_\theta E_\theta(x')\big],$$

so the gradient of the negative log-likelihood over the data is

$$\nabla_\theta\,\mathbb{E}_\text{data}\big[-\log p_\theta(x)\big] = \mathbb{E}_{x \sim \text{data}}\big[\nabla_\theta E_\theta(x)\big] - \mathbb{E}_{x' \sim p_\theta}\big[\nabla_\theta E_\theta(x')\big].$$

read it as a tug of war: lower the energy where the data are, raise it where the model's own samples are. at the optimum the two expectations balance, the model's samples are statistically indistinguishable from the data (as far as the gradients of $E$ can tell), and the gradient vanishes. for an exponential family (chapter 5), where $E_\theta(x) = -\theta^\top T(x)$, this is exactly "match the expected sufficient statistics".

the second expectation needs samples from the model, which is what Langevin gives you. a minibatch estimate is the **contrastive divergence** loss (Hinton, 2002) whose gradient is the formula above, with fakes produced by MCMC and treated as constants:

$$\mathcal{L}_\text{CD} = \frac1B\sum_b E_\theta(x_b) - \frac1B\sum_b E_\theta(\tilde x_b) + \alpha\Big(\overline{E_\theta(x)^2} + \overline{E_\theta(\tilde x)^2}\Big).$$

the $\alpha$ term is a practical addition (Du & Mordatch, 2019): energies are only defined up to a constant, and without it they drift to huge magnitudes. it keeps both near zero.

the energy network itself can be anything with a scalar output, with one caution: Langevin uses its *input gradient*, so the activations should be smooth. a ReLU MLP's energy is piecewise linear; its score is piecewise constant and its curvature is zero everywhere, which makes for a jerky sampler.

> **spec · stage 3.**
> - `EnergyMLP(data_dim=2, *, hidden_dim=128, n_layers=3)`: an `nn.Module`, an MLP with `n_layers` hidden layers of width `hidden_dim` and smooth activations (SiLU, softplus or tanh), then a linear layer to one output; `forward(x)` maps `(B, data_dim)` to energies `(B,)`.
> - `ebm_loss(energy_real, energy_fake, *, alpha=0.1)`: the formula above, means over the batch. this is `mlfs.diffusion.ebm_loss`.

> **▶ build: stage 3.** implement `EnergyMLP` and `ebm_loss`.

> **🔮 predict:** an EBM whose energy is $\tfrac12((x - \mu)/s)^2$ with learnable $\mu$ and $s$ is trained by contrastive divergence on data from $\mathcal{N}(1, 0.5^2)$. where do $\mu$ and $s$ end up, if the MCMC samples are good?

### idea 4: where the negative samples come from

the theory says "sample from $p_\theta$". the practice is a compromise, because a fully converged MCMC run per gradient step is unaffordable, and the energy changes after every step anyway. three strategies:

- **CD-$k$ (Hinton, 2002):** start chains at the data and run $k$ steps. cheap, but the fakes never leave the neighbourhood of the data, so spurious low-energy regions far away are never discovered and never corrected.
- **persistent CD (Tieleman, 2008):** keep a set of chains alive across training iterations, continuing each one from where it stopped. because the energy changes slowly, the chains stay close to equilibrium, and over many iterations they effectively run for thousands of steps. a **replay buffer** (Du & Mordatch, 2019) is a persistent pool of samples: each iteration picks a random subset, runs $k$ Langevin steps from them, writes them back, and re-initializes a small fraction (say 5%) to fresh noise so new regions keep being explored.
- **short-run MCMC (Nijkamp et al., 2019):** start *every* chain from noise and run a fixed, small number of steps. it trains fine, and the samples at exactly that number of steps can look good, but the energy it learns is not a faithful density: it's whatever makes that particular short sampler land on the data. run the chain longer and it may drift somewhere else. one thing no training scheme pins down well is how mass is shared between well-separated modes: chains essentially never cross between them during training, so the relative depths of the wells get little or no gradient. the lab measures this for short-run and persistent training.

which the checks can tell apart, cleanly: give the EBM a Gaussian energy with learnable mean and spread. with persistent chains, the fixed point of contrastive divergence is the maximum-likelihood estimate, so the learned mean and standard deviation match the data's (on paper 4); with chains restarted from noise every iteration, the chains never reach the model distribution and the learned spread comes out wrong.

> **spec · stage 4.** `train_ebm(energy, X, *, n_iters=500, batch_size=128, lr=1e-3, langevin_steps=20, step_size=0.01, buffer_size=10000, reinit_prob=0.05, alpha=0.1, seed=0)`, with `energy` an `nn.Module` mapping `(B, D)` to `(B,)` and `X` an array or tensor `(N, D)`.
> - all randomness from `g = torch.Generator().manual_seed(seed)`; `torch.optim.Adam(energy.parameters(), lr=lr)`.
> - the buffer: `buffer_size` points uniform on the data's bounding box (per coordinate, from `X.min(0)` to `X.max(0)`).
> - each of `n_iters` iterations: a real batch `X[torch.randint(N, (batch_size,), generator=g)]`; `batch_size` buffer indices (`torch.randint`); each selected point replaced by fresh box noise with probability `reinit_prob`; `langevin_dynamics(score_from_energy(energy), ...)` for `langevin_steps` steps of `step_size` with generator `g`; the results written back into the buffer; one Adam step on `ebm_loss(energy(real), energy(fakes), alpha=alpha)`.
> - returns a dict with lists of floats `loss`, `e_real`, `e_fake` (the batch-mean energies), one entry per iteration, and `buffer`, the final buffer tensor `(buffer_size, D)`.

> **▶ build: stage 4.** implement `train_ebm`.

## worked example

**one Langevin step.** target $\mathcal{N}(0, 1)$ (score $-x$), $h = 0.5$, $x = 2$, noise draw $z = 0.3$: $x' = 2 + 0.25 \cdot (-2) + \sqrt{0.5} \cdot 0.3 = 1.5 + 0.212 = 1.712$.

**its stationary variance.** the recursion is $x' = 0.75\,x + \sqrt{0.5}\,z$; a stationary variance $v$ must satisfy $v = 0.5625\,v + 0.5$, so $v = 0.5/0.4375 = 1.143$, which is $1/(1 - 0.5/4)$ as the general formula says. at $h = 1$ it's $4/3$; at $h = 2$, $v = 2$; at $h = 4$ the recursion is $x' = -x + 2z$ and the variance grows without bound. the checks run your sampler on 20,000 chains and compare with these numbers, so a missing $\frac12$ (giving $0.667$ at $h = 0.5$) or noise $\sqrt{2h}$ (giving $2.29$) fails.

**the CD loss.** real energies $(1, -2, 0.5)$, fake energies $(3, 1)$, $\alpha = 0.3$: $\bar E_\text{real} = -0.167$, $\bar E_\text{fake} = 2$, $\overline{E^2_\text{real}} = 1.75$, $\overline{E^2_\text{fake}} = 5$, so $\mathcal{L} = -0.167 - 2 + 0.3 \cdot 6.75 = -0.142$. its gradient is $+\frac13$ (plus the regularizer's share) for each real energy and $-\frac12$ for each fake: training pushes real energies down and fake energies up.

## on paper

1. **Langevin's stationary distribution.** for the SDE $dX = \frac12 \nabla\log p(X)\,dt + dW$, the density $\rho_t$ of $X_t$ evolves by the Fokker–Planck equation $\partial_t \rho = -\nabla\cdot\big(\tfrac12 \rho\,\nabla\log p\big) + \tfrac12\Delta\rho$. show that $\rho = p$ is stationary.

<details><summary>solution</summary>

with $\rho = p$: $\rho\,\nabla\log p = p\,\nabla p / p = \nabla p$, so the drift term is $-\nabla\cdot(\tfrac12\nabla p) = -\tfrac12\Delta p$, which cancels the diffusion term $\tfrac12\Delta p$ exactly. so $\partial_t \rho = 0$. the factor $\tfrac12$ on the drift matches the $\tfrac12$ in front of the Laplacian that a unit-variance Brownian motion produces; with drift $\nabla \log p$ instead, the stationary density would be $p^2$ (normalized), a sharpened version.

</details>

2. **the bias of unadjusted Langevin.** for $p = \mathcal{N}(0, s^2)$ and step $h$, derive the stationary variance of $x_{k+1} = x_k + \frac h2 s(x_k) + \sqrt h z_k$, and the largest step for which it exists.

<details><summary>solution</summary>

$x_{k+1} = a x_k + \sqrt h z_k$ with $a = 1 - h/(2s^2)$. stationarity: $v = a^2 v + h$, so $v = h/(1 - a^2)$. with $1 - a^2 = h/s^2 - h^2/(4s^4) = (h/s^2)(1 - h/(4s^2))$: $v = s^2/(1 - h/(4s^2))$. it needs $|a| < 1$, i.e. $h < 4s^2$. the bias is first order in $h$: about $+h/4$ for $s = 1$. in $D$ dimensions with different scales, the narrowest direction sets the stability limit and the widest one the mixing time: the reason samplers struggle on badly conditioned targets, as gradient descent does (chapter 6).

</details>

3. **the log-partition gradient.** show $\nabla_\theta \log Z_\theta = -\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x)]$.

<details><summary>solution</summary>

$\nabla_\theta \log Z_\theta = \frac{1}{Z_\theta}\int \nabla_\theta e^{-E_\theta(x)}dx = \frac{1}{Z_\theta}\int -\nabla_\theta E_\theta(x)\,e^{-E_\theta(x)}dx = -\int \nabla_\theta E_\theta(x)\,p_\theta(x)\,dx$ (assuming you can swap derivative and integral). the expectation is under the model, which is why training needs samples from it.

</details>

4. **CD on a Gaussian energy.** with $E(x) = \frac12((x - \mu)/s)^2$ (one dimension), find the fixed points of the exact contrastive-divergence gradient (data expectation minus model expectation).

<details><summary>solution</summary>

$\partial_\mu E = -(x - \mu)/s^2$ and $\partial_{\log s} E = -((x - \mu)/s)^2$. the gradient vanishes when $\mathbb{E}_\text{data}[x - \mu] = \mathbb{E}_{p_\theta}[x - \mu] = 0$, so $\mu = \mathbb{E}_\text{data}[x]$, and when $\mathbb{E}_\text{data}[(x - \mu)^2] = \mathbb{E}_{p_\theta}[(x - \mu)^2] = s^2$, so $s^2$ = the data variance. if the "model samples" are biased (chains that haven't converged, or ULA's step-size bias), the fixed point moves: the model is adjusted until its *sampler's* output matches the data, not the model itself.

</details>

## experiment

`lab.py` uses your sampler and training loop:

1. the bias of unadjusted Langevin against step size, and what MALA does about it.
2. EBMs on 2-D data trained with short-run and with persistent chains, and what happens when you run their samplers longer than they were trained with.
3. what the learned energy looks like as a density.

## check your understanding

1. why can Langevin dynamics sample from $p_\theta$ without knowing $Z_\theta$?

<details><summary>answer</summary>

it only uses $\nabla_x \log p_\theta = -\nabla_x E_\theta$, and the normalizer is constant in $x$, so its gradient is zero. MALA additionally uses energy differences $E(x) - E(x')$, from which $\log Z_\theta$ cancels.

</details>

2. what goes wrong if you drop the noise from Langevin dynamics? if you double it?

<details><summary>answer</summary>

without noise it's gradient ascent on $\log p$: every chain climbs to a local mode and stops, so you get a handful of points, not samples. doubling the noise *variance* (or halving the drift) changes the stationary distribution to a flattened one ($p^{1/2}$, normalized, in the small-step limit); doubling the noise *amplitude* quadruples the variance and gives $p^{1/4}$. either way: too spread out.

</details>

3. MALA is exact. why do people still run unadjusted Langevin on EBMs?

<details><summary>answer</summary>

the acceptance rate of MALA falls quickly with dimension unless the step is tiny (it scales like $h \propto D^{-1/3}$ for nice targets), each step needs the energy at the proposal as well, and with a learned energy the samples are only as good as the energy anyway. for training, a cheap biased sampler is often good enough, as long as you remember the model is fitted to that sampler's output.

</details>

4. the CD gradient is zero for a model that is wrong. how is that possible?

<details><summary>answer</summary>

the gradient is zero when the data and the *negative samples* agree in expectation of $\nabla_\theta E$. if the negative samples come from short, unconverged chains, they agree with the data while the model's true distribution $e^{-E}/Z$ doesn't. the model has learned to make its sampler produce data, not to be the data density.

</details>

5. why does a replay buffer re-initialize a few percent of its chains every iteration?

<details><summary>answer</summary>

persistent chains can get stuck in a few modes; regions of low energy that no chain visits are never pushed up, so spurious modes can survive there. fresh noise keeps exploring the space and finds them.

</details>

6. why do EBM energies need the $\alpha$ regularizer while a flow's log-likelihood doesn't?

<details><summary>answer</summary>

$p \propto e^{-E}$ is unchanged by $E \to E + c$, and the CD loss is unchanged too (the constant cancels between real and fake terms), so nothing anchors the energies' level, and small asymmetries can make them drift without bound. a flow's density is normalized by construction, so there's no free constant.

</details>

## common bugs

- **gradients through the sampler.** symptom: memory grows with `langevin_steps` and training is slow; the fakes carry a graph back through every step. detach (or `torch.no_grad()` around the loop with `torch.enable_grad()` only inside the score).
- **`torch.autograd.grad` inside `no_grad`.** symptom: "element 0 of tensors does not require grad". the score function must turn gradients on locally.
- **a sign error in the score.** symptom: samples fly away to infinity; the score is $-\nabla E$, so the drift lowers the energy.
- **forgetting to write the Langevin samples back into the buffer.** symptom: the chains never progress beyond `langevin_steps` steps from noise; persistent training silently becomes short-run training.
- **accumulating the CD gradient into the fakes.** symptom: the fakes' energies are pushed up *and* the samples move; compute `energy(fake)` on detached samples.
- **ReLU energies.** symptom: samples with streaky, axis-aligned artifacts and a sampler that doesn't care about curvature.

## going deeper

- Hinton (2002), "Training products of experts by minimizing contrastive divergence", *Neural Computation* 14(8). [doi:10.1162/089976602760128018](https://doi.org/10.1162/089976602760128018)
- Tieleman (2008), "Training restricted Boltzmann machines using approximations to the likelihood gradient", *ICML*. persistent CD. [doi:10.1145/1390156.1390290](https://doi.org/10.1145/1390156.1390290)
- Roberts & Tweedie (1996), "Exponential convergence of Langevin distributions and their discrete approximations", *Bernoulli* 2(4). MALA. [doi:10.2307/3318418](https://doi.org/10.2307/3318418)
- Du & Mordatch (2019), "Implicit generation and modeling with energy-based models", *NeurIPS*. replay buffer and regularization. [arXiv:1903.08689](https://arxiv.org/abs/1903.08689)
- Nijkamp, Hill, Zhu & Wu (2019), "Learning non-convergent non-persistent short-run MCMC toward energy-based model", *NeurIPS*. [arXiv:1904.09770](https://arxiv.org/abs/1904.09770)
- Song & Kingma (2021), "How to train your energy-based models". a survey. [arXiv:2101.03288](https://arxiv.org/abs/2101.03288)
- LeCun, Chopra, Hadsell, Ranzato & Huang (2006), "A tutorial on energy-based learning", in *Predicting Structured Data*, MIT Press. [yann.lecun.com/exdb/publis/pdf/lecun-06.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)

## connections

- ← builds on: Markov chains and Brownian motion (ch 04 · topic 07), Metropolis–Hastings and HMC (ch 12 · topics 05–06), exponential families and moment matching (ch 05 · topic 06), the likelihood-based models of ch 21.
- → used in: score matching, which learns $\nabla_x \log p$ without ever forming an energy or running MCMC during training (topic 02); every diffusion sampler in this chapter, which is Langevin dynamics run while the noise level shrinks (topics 02–04); and EBM-style rerankers and verifiers in later chapters.
