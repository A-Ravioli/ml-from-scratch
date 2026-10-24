# the SDE view & fast sampling

## why this matters

a DDPM trained with 1,000 steps needs 1,000 sequential network calls per sample. for a CIFAR-sized U-Net that's minutes per batch on a laptop, and for the latent video models of today, unthinkable. the obvious fix, "just take bigger steps", fails for ancestral sampling: each step's mean and noise are tied to the small $\beta_t$ of that step, and skipping steps breaks the Gaussian approximation the sampler relies on.

the way out is to stop thinking of diffusion as a chain of 1,000 discrete steps and see it as the discretization of a *continuous* process (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole, 2021). the forward process becomes a stochastic differential equation; its time reversal is another SDE that needs only the score; and, remarkably, there is a *deterministic* ODE whose solutions have exactly the same marginal distributions at every time, the **probability-flow ODE**. a deterministic ODE can be solved with the large-step, high-order methods you built in chapter 20. DDIM (Song, Meng & Ermon, 2021), discovered from a different angle, turns out to be the simplest such solver. none of this needs retraining: it's the same network from topic 03, sampled differently.

## warm-up: predict first

1. a stochastic process and a deterministic ODE have the same distribution at every time. can a single trajectory of the ODE look like a trajectory of the SDE?
2. you take a trained DDPM and sample it with a deterministic method in 2 steps instead of 1,000. what do the samples look like?
3. a fourth-order solver costs 4 network calls per step. with a budget of 20 calls, is 5 RK4 steps better or worse than 20 Euler steps?
4. with a *perfect* noise predictor, is a 10-step deterministic sampler exact?
5. which is more forgiving of an imperfect network: a sampler that injects noise at every step, or a deterministic one?

## prerequisites check

- [ ] the DDPM forward process, $\bar\alpha_t$, the noise predictor and `ddpm_sample` (→ topic 03)
- [ ] scores and Tweedie's formula (→ topic 02)
- [ ] fixed-step ODE solvers (Euler, midpoint, RK4) and `mlfs.ode.odeint` (→ ch 20 · topic 06)
- [ ] Brownian motion and the idea of an SDE (→ ch 04 · topic 07); the Fokker–Planck equation (→ topic 01, on paper 1)

## core ideas

### idea 1: diffusion in continuous time

in the limit of many small steps, the forward chain $x_t = \sqrt{1 - \beta_t}\,x_{t-1} + \sqrt{\beta_t}\,z$ becomes the **variance-preserving SDE**

$$dx = -\tfrac12\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW,$$

with a noise rate $\beta(t)$; its marginals are exactly topic 03's, $x_t = \sqrt{\bar\alpha(t)}\,x_0 + \sqrt{1 - \bar\alpha(t)}\,\epsilon$ with $\bar\alpha(t) = \exp(-\int_0^t\beta)$, i.e. $\beta(t) = -\frac{d}{dt}\log\bar\alpha(t)$ (on paper 2). to use a network trained on a discrete schedule, define the continuous $\bar\alpha(t)$ by interpolating $\log\bar\alpha$ linearly between the grid points $t_k = k/T$; then $\beta(t)$ is piecewise constant, $T(\log\bar\alpha_k - \log\bar\alpha_{k+1})$ on segment $k$. the network's input is the same $t \in [0, (T-1)/T]$ as in training, now evaluated in between grid points too (it generalizes smoothly, since it saw a fine grid).

two facts from the theory of diffusion processes then give you two samplers:

- **the reverse-time SDE** (Anderson, 1982). running the forward SDE backwards in time is again an SDE, $dx = [f(x, t) - g(t)^2\nabla\log p_t(x)]\,dt + g(t)\,d\bar W$ with $f = -\frac12\beta x$ and $g^2 = \beta$; it needs only the score of the noisy marginal $p_t$.
- **the probability-flow ODE.** the deterministic ODE

$$\frac{dx}{dt} = f(x, t) - \tfrac12 g(t)^2\,\nabla\log p_t(x) = -\tfrac12\beta(t)\big(x + \nabla\log p_t(x)\big)$$

moves probability mass exactly the way the SDE does, so if you start it from $p_T$ and run it backwards, its solution at every $t$ is distributed as $p_t$ (on paper 1). individual trajectories are smooth and deterministic, not noisy; only the *distribution* matches.

the score is the network in disguise: $\nabla\log p_t(x) \approx -\epsilon_\theta(x, t)/\sqrt{1 - \bar\alpha(t)}$ (topic 03's Tweedie connection).

> **spec · stage 1.**
> - `alpha_bar_at(schedule, t)`: $\bar\alpha(t)$ for a tensor `t` of any shape: $\log\bar\alpha$ interpolated linearly between $t_k = k/T$ ($k = 0, \dots, T-1$), `t` clamped to $[0, (T-1)/T]$.
> - `beta_at(schedule, t)`: $\beta(t) = -\frac{d}{dt}\log\bar\alpha(t) = T(\log\bar\alpha_k - \log\bar\alpha_{k+1})$ on $[k/T, (k+1)/T)$, same clamping, the last segment at $t = (T-1)/T$.
> - `eps_to_score(eps, alpha_bar)`: $-\epsilon/\sqrt{1 - \bar\alpha}$, with `alpha_bar` a float or a `(B,)` tensor broadcast over the non-batch dims.
> - `pf_ode_drift(model, schedule, x, t, y=None)`: with `t` `(B,)`, $-\frac12\beta(t)\,(x + s)$ where $s$ = `eps_to_score(model(x, t, y), alpha_bar_at(schedule, t))`.

> **▶ build: stage 1.** implement `alpha_bar_at`, `beta_at`, `eps_to_score` and `pf_ode_drift` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: DDIM

DDIM gets to a deterministic sampler without any SDE. its observation: the training loss only ever uses the marginals $q(x_t \mid x_0)$, never the step-to-step transitions, so *any* process with the same marginals is compatible with the trained network. among them is one where $x_{t'}$ (for any earlier $t'$, not just $t - 1$) is a deterministic function of $x_t$ and $x_0$:

$$\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}, \qquad x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat x_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\,\epsilon_\theta(x_t, t) + \sigma z.$$

in words: predict the clean image, then re-noise it to level $t'$ using the *same* noise direction the network just inferred. with $\sigma = 0$ ($\eta = 0$) the sampler is deterministic, and nothing ties $t'$ to $t - 1$: DDIM visits an evenly spaced subsequence of the training steps, $\tau_1 < \dots < \tau_S$, and jumps between them; the last jump goes to $\bar\alpha = 1$, i.e. returns $\hat x_0$. the parameter $\eta$ puts noise back in,

$$\sigma = \eta\,\sqrt{\frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t}}\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}},$$

and at $\eta = 1$ with every step visited it is ancestral sampling with the posterior variance $\tilde\beta_t$.

why does it work so well? divide the update by $\sqrt{\bar\alpha_{t'}}$ and write $\bar x = x/\sqrt{\bar\alpha}$, $\sigma(t) = \sqrt{(1 - \bar\alpha)/\bar\alpha}$: the $\eta = 0$ step becomes $\bar x_{t'} = \bar x_t + (\sigma_{t'} - \sigma_t)\,\epsilon_\theta(x_t, t)$, one **Euler step** of the probability-flow ODE written in those coordinates (on paper 4). in them the ODE is nearly straight, which is why a first-order method with big steps is already decent.

> **spec · stage 2.**
> - `ddim_timesteps(n_train, n_steps)`: `torch.linspace(0, n_train - 1, n_steps).round().long()`, except that a single step is `[n_train - 1]` (start at the noisiest step and jump straight to $\hat x_0$; `linspace` would give `[0]`, a sampler that never denoises).
> - `ddim_sample(model, schedule, shape, *, n_steps=50, eta=0.0, y=None, generator=None, device='cpu')`: `x = torch.randn(shape, generator=generator)` (CPU, then `device`); for the taus from the last to the first: `eps = model(x, torch.full((B,), tau / T), y)`, $\hat x_0$, $\bar\alpha_{t'}$ of the previous tau (1 after the first), $\sigma$ as above, the update, and `z = torch.randn(shape, generator=generator)` only when $\sigma > 0$. no gradients; eval mode for an `nn.Module`, restored afterwards. this is `mlfs.diffusion.ddim_sample`.

> **▶ build: stage 2.** implement `ddim_timesteps` and `ddim_sample`.

> **🔮 predict:** on Gaussian data $\mathcal{N}(0, 0.5^2)$ with the *exact* noise predictor, what standard deviation do 10 deterministic DDIM steps produce? and 2 steps?

### idea 3: the probability-flow ODE with a real solver

with the drift of idea 1 you can hand the sampling problem to any ODE solver: start at $t_{\max} = (T-1)/T$ from $x \sim \mathcal{N}(0, I)$ and integrate $dx/dt = $ `pf_ode_drift` backwards to $t = 0$ with `mlfs.ode.odeint`. the cost is counted in **network function evaluations** (NFE): Euler uses 1 per step, midpoint 2, RK4 4. at $t = 0$ the sample still carries the step-0 noise ($\bar\alpha_0 = 1 - \beta_0$), so finish with one denoising step, $\hat x_0 = (x - \sqrt{1 - \bar\alpha_0}\,\epsilon_\theta(x, 0))/\sqrt{\bar\alpha_0}$, the same as DDIM's last jump.

two separate errors limit any sampler: the **discretization error** of the solver (which shrinks with more steps, at a rate set by its order), and the **model error** of the network (which doesn't). on Gaussian data with the exact predictor only the first exists, and the ODE is linear: it maps $x_T$ to $x_T\sqrt{v_0/v_T}$ with $v_t = \bar\alpha(t)s^2 + 1 - \bar\alpha(t)$, followed by the denoising step. your solver must reproduce that map number for number. (one practical wrinkle: $\beta(t)$ is piecewise constant, with a kink at every grid point, which limits how much RK4's high order can buy you on a 1,000-step schedule.)

the same ODE gives something the SDE can't: an **exact likelihood**. it's a continuous normalizing flow (chapter 20's neural ODE), so the instantaneous change-of-variables formula $\frac{d}{dt}\log p_t(x(t)) = -\nabla\cdot(\text{drift})$ yields $\log p_0(x)$ by integration, with a Hutchinson estimate of the divergence (topic 02). that's how diffusion models report bits/dim.

> **spec · stage 3.** `probability_flow_sample(model, schedule, shape, *, n_steps=50, method='rk4', y=None, generator=None, device='cpu')`: `x = torch.randn(shape, generator=generator)`; integrate `pf_ode_drift` from $t = (T-1)/T$ to 0 with `mlfs.ode.odeint` in exactly `n_steps` steps of `method`, passing the model `t` as a `(B,)` tensor; then the denoising step at $t = 0$. no gradients; eval mode, restored. this is `mlfs.diffusion.probability_flow_sample`.

> **▶ build: stage 3.** implement `probability_flow_sample`.

> **🔮 predict:** a perfect network and a budget of about 20 network calls: which is more accurate, the PF-ODE with 19 Euler steps or with 4 RK4 steps?

### idea 4: the reverse-time SDE

the other sampler from idea 1 integrates the reverse SDE with **Euler–Maruyama**: a drift step plus $\sqrt{\beta h}$ worth of fresh noise, going backwards from $t_{\max}$ to 0 in steps of $h$:

$$x \leftarrow x + h\,\beta(t)\big(\tfrac12 x + \nabla\log p_t(x)\big) + \sqrt{\beta(t)\,h}\;z.$$

compared with the ODE, its drift pulls twice as hard towards high density ($g^2$ instead of $\frac12 g^2$ on the score), and the noise pushes back out. that combination is *contracting*: errors made early (by the solver, or by an imperfect network) get partly forgotten, because the injected noise re-randomizes what doesn't fit, and the strong drift pulls it back onto the data. the price is that stochastic samplers need more steps to converge than deterministic ones of the same order. ancestral DDPM sampling is a cousin of this sampler, on the discrete grid.

> **spec · stage 4.** `reverse_sde_sample(model, schedule, shape, *, n_steps=100, y=None, generator=None, device='cpu')`: `x = torch.randn(shape, generator=generator)`; `n_steps` Euler–Maruyama steps of size $h = t_{\max}/n_{\text{steps}}$ from $t = t_{\max}$ downwards (at $t_j = t_{\max} - jh$), with $\beta$ from `beta_at`, the score from `eps_to_score` and one `z = torch.randn(shape, generator=generator)` per step; then the denoising step at $t = 0$. no gradients; eval mode, restored.

> **▶ build: stage 4.** implement `reverse_sde_sample`.

## worked example

**Gaussian data, exact predictor.** data $\mathcal{N}(0, 0.5^2)$, the default linear schedule ($T = 1000$, $\bar\alpha_{999} = 4.04 \times 10^{-5}$). the exact map of the PF-ODE plus the final denoising step multiplies $x_T$ by $0.49991$, so the samples have standard deviation 0.5 (to four digits; the denoising step's shrinkage at $\bar\alpha_0 = 0.9999$ is invisible).

**DDIM with the same predictor.** DDIM's per-step factor is linear in $x$ (on paper 3), so the whole sampler multiplies $x_T$ by a product of factors you can compute by hand. for the default schedule:

| DDIM steps | 2 | 5 | 10 | 20 | 50 | 100 | 1000 |
|---|---|---|---|---|---|---|---|
| output std (target 0.5) | 0.012 | 0.225 | 0.362 | 0.431 | 0.472 | 0.486 | 0.4985 |

even with a perfect network, few-step DDIM *shrinks* this distribution: with 2 steps it jumps from $\bar\alpha = 4 \times 10^{-5}$ straight to $\hat x_0$, and the best guess of $x_0$ from almost pure noise is the mean. the checks compare your `ddim_sample` with this product number for number (at 10 steps), and your `probability_flow_sample` with the exact ODE map.

**NFE accounting.** for a budget of about 50 network calls: DDPM can't (it needs $T$); DDIM 50 steps; PF-ODE with Euler 49 steps (+1 denoising call), midpoint 24 (+1), RK4 12 (+1).

## on paper

1. **same marginals.** the density of the SDE $dx = f\,dt + g\,dW$ evolves by $\partial_t p = -\nabla\cdot(f p) + \frac12 g^2\Delta p$. show that the ODE $\dot x = f - \frac12 g^2\nabla\log p_t$ produces the same $\partial_t p$.

<details><summary>solution</summary>

an ODE $\dot x = u(x, t)$ transports density by the continuity equation $\partial_t p = -\nabla\cdot(u p)$. with $u = f - \frac12 g^2\nabla\log p$: $u p = f p - \frac12 g^2\nabla p$ (since $p\nabla\log p = \nabla p$), so $-\nabla\cdot(u p) = -\nabla\cdot(f p) + \frac12 g^2\Delta p$. the two evolutions agree for every $t$, so starting from the same $p_T$ they have the same marginals at every time.

</details>

2. **the VP-SDE's marginals.** for $dx = -\frac12\beta(t)x\,dt + \sqrt{\beta(t)}\,dW$ (one dimension), find ODEs for the mean $m(t)$ and variance $v(t)$ of $x_t$ given $x_0$, and solve them.

<details><summary>solution</summary>

$\dot m = -\frac12\beta m$ and $\dot v = -\beta v + \beta$ (the drift shrinks the variance at twice the rate it shrinks the mean; the noise adds $\beta\,dt$). with $\bar\alpha(t) = e^{-\int_0^t\beta}$: $m = \sqrt{\bar\alpha}\,x_0$ and $v = 1 - \bar\alpha$ (check: $\dot v = \beta\bar\alpha = -\beta(1 - \bar\alpha) + \beta$). these are topic 03's marginals.

</details>

3. **DDIM shrinks Gaussians.** for data $\mathcal{N}(0, s^2)$ and the exact predictor, show that one $\eta = 0$ DDIM step from $\bar\alpha$ to $\bar\alpha'$ multiplies $x$ by $c = \big(\sqrt{\bar\alpha\bar\alpha'}\,s^2 + \sqrt{(1 - \bar\alpha)(1 - \bar\alpha')}\big)/v$ with $v = \bar\alpha s^2 + 1 - \bar\alpha$, and that $c^2 v \le v'$, with equality only if $s = 1$ or $\bar\alpha = \bar\alpha'$.

<details><summary>solution</summary>

with $\epsilon^* = \sqrt{1 - \bar\alpha}\,x/v$: $\hat x_0 = \sqrt{\bar\alpha}\,s^2 x/v$, and $x' = \sqrt{\bar\alpha'}\hat x_0 + \sqrt{1 - \bar\alpha'}\,\epsilon^* = c\,x$. if $x \sim \mathcal{N}(0, v)$ then $\operatorname{Var}x' = c^2 v = (a\cdot b)^2/v$ with $a = (\sqrt{\bar\alpha}s, \sqrt{1 - \bar\alpha})$, $b = (\sqrt{\bar\alpha'}s, \sqrt{1 - \bar\alpha'})$, $\lVert a\rVert^2 = v$, $\lVert b\rVert^2 = v'$. by Cauchy–Schwarz $(a\cdot b)^2 \le v v'$, so $c^2 v \le v'$, with equality iff $a \parallel b$: $s = 1$ or $\bar\alpha = \bar\alpha'$. big jumps lose variance whenever the data's scale differs from the noise's.

</details>

4. **DDIM is Euler.** with $\bar x = x/\sqrt{\bar\alpha}$ and $\sigma = \sqrt{(1 - \bar\alpha)/\bar\alpha}$, show that the $\eta = 0$ DDIM update is $\bar x' = \bar x + (\sigma' - \sigma)\,\epsilon_\theta(x, t)$.

<details><summary>solution</summary>

divide $x' = \sqrt{\bar\alpha'}\hat x_0 + \sqrt{1 - \bar\alpha'}\,\epsilon$ by $\sqrt{\bar\alpha'}$: $\bar x' = \hat x_0 + \sigma'\epsilon$. and $\hat x_0 = (x - \sqrt{1 - \bar\alpha}\,\epsilon)/\sqrt{\bar\alpha} = \bar x - \sigma\epsilon$. so $\bar x' = \bar x + (\sigma' - \sigma)\epsilon$: an Euler step of $d\bar x/d\sigma = \epsilon_\theta$, which is the probability-flow ODE in these coordinates (Karras et al., 2022 build their whole sampler design on this form).

</details>

## experiment

`lab.py` compares your samplers on two problems:

1. a 2-D Gaussian mixture with the *exact* noise predictor (computed in closed form), so the only error is the sampler's: quality against the number of network calls.
2. the same samplers on a small diffusion model trained by your topic 03 code on digits, scored with the course FID.
3. deterministic against stochastic sampling when the network is imperfect.

## check your understanding

1. how can a deterministic ODE produce *random* samples?

<details><summary>answer</summary>

the randomness is all in the starting point $x_T \sim \mathcal{N}(0, I)$. the ODE is a fixed, invertible map from noise to data (a continuous normalizing flow); different starting points give different samples, and the map is built so that pushing $\mathcal{N}(0, I)$ through it gives the data distribution.

</details>

2. why can't you just run ancestral DDPM sampling on every 20th step?

<details><summary>answer</summary>

each ancestral step's mean and variance are derived for one small step $\beta_t$; across 20 steps the true reverse transition is not well approximated by a Gaussian with those parameters, and using $\beta_t$ of a single step as if it were the variance of 20 gives far too little noise. respacing (Nichol & Dhariwal, 2021) recomputes the betas for the subsequence, which works but is a different sampler.

</details>

3. why does few-step DDIM shrink the Gaussian, even with a perfect network?

<details><summary>answer</summary>

each step re-noises the *predicted* $\hat x_0 = \mathbb{E}[x_0 \mid x_t]$, a posterior mean, which has less variance than $x_0$; the re-noising uses the inferred $\epsilon$ instead of fresh noise, so the lost variance is never restored. in the many-step limit the error vanishes (it's an Euler discretization of the exact ODE); with few steps it's the solver's discretization error, and it shows up as blur and loss of diversity.

</details>

4. which sampler would you use with a 20-call budget and an excellent network? with a mediocre network?

<details><summary>answer</summary>

excellent network: a deterministic solver (DDIM or a higher-order ODE solver), since the only error left is discretization, and deterministic solvers converge fastest. mediocre network: some stochasticity ($\eta > 0$ or an SDE sampler) often helps, because the noise lets the sampler correct for the network's systematic errors, but it needs more calls to converge; with only 20 calls, that tradeoff has to be measured.

</details>

5. what does the PF-ODE give you that DDPM sampling doesn't?

<details><summary>answer</summary>

determinism (the same noise always gives the same image, so you can encode an image to its noise by integrating forwards, interpolate between noises, edit), free choice of solver and step count, and exact likelihoods via the instantaneous change of variables.

</details>

## common bugs

- **integrating in the wrong direction.** symptom: samples blow up; the drift is integrated from $t_{\max}$ *down* to 0 (`odeint` with a decreasing time grid).
- **mixing step indices and times.** symptom: DDIM visits steps 0..999 but tells the network $t$ = 0..999 instead of $\tau/T$.
- **the last DDIM jump.** symptom: the output is still slightly noisy; after the first tau, $\bar\alpha_{\text{prev}} = 1$, so the last update returns $\hat x_0$.
- **drawing noise when $\eta = 0$.** symptom: the deterministic sampler consumes random numbers, so two runs with the same generator but different `n_steps` don't start from the same $x_T$ alignment; draw $z$ only when $\sigma > 0$.
- **a float32 rounding of the step count.** symptom: `odeint` takes $n + 1$ steps instead of $n$ (since $t_{\max}/(t_{\max}/n)$ comes out as $n.0000001$); give the step size a tiny slack.
- **DDIM from a schedule that ends at $\bar\alpha \approx 0$.** symptom: samples explode to huge values. the cosine schedule's last $\bar\alpha$ is about $10^{-9}$ (its last $\beta$ is clipped at 0.999), and $\hat x_0 = (x - \sqrt{1 - \bar\alpha}\,\epsilon)/\sqrt{\bar\alpha}$ multiplies the network's error by $1/\sqrt{\bar\alpha} \approx 2 \times 10^4$. image pipelines clip $\hat x_0$ to the data range; the samples here don't, so pair DDIM with schedules whose last $\bar\alpha$ isn't tiny (the linear one ends at $4 \times 10^{-5}$).
- **forgetting the final denoising step.** symptom: hard to see by eye, but your NFE count is off by one and the samples keep the step-0 noise.

## going deeper

- Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021), "Score-based generative modeling through stochastic differential equations", *ICLR*. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Song, Meng & Ermon (2021), "Denoising diffusion implicit models", *ICLR*. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Anderson (1982), "Reverse-time diffusion equation models", *Stochastic Processes and their Applications* 12(3). [doi:10.1016/0304-4149(82)90051-5](https://doi.org/10.1016/0304-4149(82)90051-5)
- Karras, Aittala, Aila & Laine (2022), "Elucidating the design space of diffusion-based generative models", *NeurIPS*. the $\sigma$-parametrized ODE, Heun's method, and what matters in samplers. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Lu, Zhou, Bao, Chen, Li & Zhu (2022), "DPM-Solver: a fast ODE solver for diffusion probabilistic model sampling in around 10 steps", *NeurIPS*. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)
- Grathwohl, Chen, Bettencourt, Sutskever & Duvenaud (2019), "FFJORD: free-form continuous dynamics for scalable reversible generative models", *ICLR*. likelihoods of continuous flows. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)

## connections

- ← builds on: DDPM and its noise predictor (topic 03), scores (topic 02), ODE solvers and neural ODEs (ch 20 · topic 06), Brownian motion (ch 04).
- → used in: flow matching, which learns an ODE directly and designs it to be straight (topic 05); guidance, which modifies the drift (topic 06); the capstone's DDPM-vs-flow-matching comparison at equal numbers of network calls; consistency models and distillation, which compress the ODE into one or two steps.
