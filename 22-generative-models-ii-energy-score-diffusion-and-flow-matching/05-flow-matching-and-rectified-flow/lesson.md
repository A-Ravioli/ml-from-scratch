# flow matching & rectified flow

## why this matters

topic 04 found a deterministic ODE hiding inside every diffusion model and sampled it with big steps. but that ODE was an accident of the noising process: its trajectories curve, so even a good solver needs tens of steps, and few-step samples come out blurry. if an ODE is what you want, why not *design* it? choose the paths you'd like samples to travel from noise to data (straight lines are the obvious choice, since a straight path is integrated exactly by a single Euler step), and train a network to produce the velocity field that moves probability along them.

that is **flow matching** (Lipman, Chen, Ben-Hamu, Nickel & Le, 2023; Liu, Gong & Liu, 2023; Albergo & Vanden-Eijnden, 2023). like denoising score matching, it replaces an intractable regression target with a tractable conditional one: draw noise $x_0$ and data $x_1$, put a point on the line between them, and regress the network's output on the direction of that line, $x_1 - x_0$. no noise schedule, no ELBO, no SDE; just a squared error. it trains as stably as diffusion and samples better in few steps, which is why it powers Stable Diffusion 3, Flux and most recent image and video models. **rectified flow** then makes the paths straighter by retraining on the model's own noise–sample pairs, until one or two Euler steps produce decent samples.

## warm-up: predict first

1. a network is trained to regress $x_1 - x_0$ from $x_t = (1 - t)x_0 + t x_1$ with $x_0$ noise and $x_1$ data drawn independently. can its loss reach zero?
2. every training path is a straight line. are the sampling trajectories of the trained ODE straight too?
3. on one-dimensional Gaussian data $\mathcal{N}(\mu, s^2)$, where does the exact flow send the noise value $x_0 = 1$?
4. Euler's method with one step is exact for which ODEs?
5. you draw 10,000 noise vectors, run them through the trained model, and retrain on those (noise, sample) pairs. what changes, and what can't get better?

## prerequisites check

- [ ] ODE solvers and `mlfs.ode.odeint` (→ ch 20 · topic 06); normalizing flows and the change of variables (→ ch 21 · topic 07)
- [ ] denoising score matching: a conditional target with the same minimizer (→ topic 02)
- [ ] the probability-flow ODE and NFE accounting (→ topic 04)
- [ ] `TimeMLP` and the chapter's time convention (→ topic 03)
- [ ] conditional expectation as the minimizer of squared error (→ ch 04 · topic 03)

## core ideas

### idea 1: conditional flow matching

a time-dependent velocity field $v(x, t)$ defines an ODE $\dot x = v(x, t)$ that moves points, and with them probability densities. flow matching asks for a field whose flow carries the noise distribution at $t = 0$ to the data distribution at $t = 1$ (the opposite direction from diffusion's time: here **$t = 0$ is noise and $t = 1$ is data**). which field? build it from paths. pair each noise sample $x_0 \sim \mathcal{N}(0, I)$ with a data point $x_1$ and connect them with a straight line,

$$x_t = (1 - t)\,x_0 + t\,x_1, \qquad \frac{d x_t}{dt} = x_1 - x_0.$$

(Lipman et al. call this the **conditional optimal-transport path**: given its endpoints, each sample moves along the straight, constant-speed path that optimal transport between two points would take. the *marginal* flow it produces is not an optimal-transport map in general; idea 4 and the OT elective are about getting closer to one.)

at time $t$ many lines pass through the same point $x$, going in different directions. the field that moves the *distribution* of $x_t$ correctly (the **marginal velocity**) is their average (on paper 1):

$$v^*(x, t) = \mathbb{E}\big[x_1 - x_0 \mid x_t = x\big].$$

you can't compute that expectation, but you don't need to: a least-squares regression onto the per-sample targets $x_1 - x_0$ has exactly this minimizer, the same trick as denoising score matching (on paper 2). that is the **conditional flow matching** loss:

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t \sim \mathcal{U}[0, 1],\ x_0 \sim \mathcal{N}(0, I),\ x_1 \sim p_\text{data}}\ \big\lVert v_\theta(x_t, t) - (x_1 - x_0)\big\rVert^2.$$

also like DSM, the loss never reaches zero: the leftover is the variance of the line directions through each point. for one-dimensional Gaussian data $\mathcal{N}(\mu, s^2)$ everything is explicit. $(x_1 - x_0, x_t)$ is jointly Gaussian, which gives

$$v^*(x, t) = \mu + \frac{t s^2 - (1 - t)}{(1 - t)^2 + t^2 s^2}\,(x - t\mu), \qquad \operatorname{Var}(x_1 - x_0 \mid x_t) = \frac{s^2}{(1 - t)^2 + t^2 s^2},$$

so the irreducible loss is $\int_0^1 s^2/((1 - t)^2 + t^2 s^2)\,dt = s\pi/2$ per dimension: $\pi/2 \approx 1.571$ for $s = 1$, $\pi/4 \approx 0.785$ for $s = 0.5$ (on paper 3). the network sees the pair $(x, t)$; in this chapter's convention it's called `model(x, t, y)` with `t` a `(B,)` tensor in $[0, 1]$. note that `x0` can be *given* instead of sampled: that's the hook for pairing noise and data cleverly (idea 4, and the optimal-transport elective).

> **spec · stage 1.** `flow_matching_loss(model, x1, *, x0=None, y=None, generator=None)`: if `x0` is `None`, draw `x0 = torch.randn(x1.shape, generator=generator)` first; then `t = torch.rand(B, generator=generator)` (CPU draws, moved to `x1`'s device); $x_t = (1 - t)x_0 + t x_1$ with `t` broadcast over the non-batch dims; return the mean squared error between `model(x_t, t, y)` and $x_1 - x_0$ (a mean over all elements). this is `mlfs.diffusion.flow_matching_loss`.

> **▶ build: stage 1.** implement `flow_matching_loss` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** you train a good network on 2-D data. its loss plateaus. at what value: near 0, or near 1, or near 3?

### idea 2: sampling is an ODE solve

to sample, draw $x_0 \sim \mathcal{N}(0, I)$ and integrate $\dot x = v_\theta(x, t)$ from $t = 0$ to $t = 1$ with any solver: this is chapter 20's `odeint` again, with Euler, midpoint or RK4 and $n$ equal steps (NFE $= n$, $2n$, $4n$). there's no final denoising step and no schedule: the time grid is uniform.

are the trajectories straight? the training paths are, but the marginal field averages over crossing lines, so its integral curves bend. the 1-D Gaussian case shows it in closed form: for $\mu = 0$ the ODE is $\frac{d}{dt}\log x = \frac{d}{dt}\frac12\log\big((1 - t)^2 + t^2 s^2\big)$, so

$$x_t = x_0\,\sqrt{(1 - t)^2 + t^2 s^2},$$

which ends at the right place, $x_1 = s\,x_0$ (and in general $\mu + s x_0$, the monotone map from $\mathcal{N}(0, 1)$ to $\mathcal{N}(\mu, s^2)$), but not along the straight line: at $t = \frac12$ with $s = 0.5$ it's at $0.56\,x_0$, not $0.75\,x_0$. one Euler step from $x_0$ would land at $x_0 + v^*(x_0, 0) = \mu$, the mean, for every starting point: the one-step sample has collapsed. curvature is what forces extra steps. the checks compare your sampler with the exact map $\mu + s x_0$, with each solver's order of accuracy, and with its call counts.

> **spec · stage 2.** `sample_flow(model, shape, *, n_steps=100, method='euler', x0=None, y=None, generator=None, device='cpu')`: start from `x0` (a copy), or `torch.randn(shape, generator=generator)` if `None`; integrate $\dot x = $ `model(x, t, y)` from $t = 0$ to 1 with `mlfs.ode.odeint` in exactly `n_steps` steps of `method` (`'euler'`, `'midpoint'`, `'rk4'`), passing `t` as a `(B,)` tensor; no gradients; eval mode for an `nn.Module`, restored afterwards; returns $x_1$. this is `mlfs.diffusion.sample_flow`.

> **▶ build: stage 2.** implement `sample_flow`.

### idea 3: training, with or without a coupling

the training loop is the plain regression: minibatches of data, the loss of idea 1, Adam. one option matters for the rest of the topic: where $x_0$ comes from. in standard CFM it's fresh noise, independent of $x_1$ (the **independent coupling**). but nothing in the derivation requires independence: any joint distribution of $(x_0, x_1)$ with the right marginals defines straight-line paths and a valid marginal field. if the pairs are chosen so that lines don't cross, the conditional targets through each point agree, the irreducible variance vanishes, and the marginal paths *are* the straight lines. the extreme case is a deterministic coupling, $x_1 = T(x_0)$; then the loss can reach zero.

> **spec · stage 3.** `fit_flow_matching(model, X, *, X0=None, coupling=None, n_iters=2000, batch_size=256, lr=1e-3, seed=0)`: `g = torch.Generator().manual_seed(seed)`, `torch.optim.Adam(lr)`, train mode; per step `idx = torch.randint(N, (batch_size,), generator=g)`, `x1 = X[idx]`, and `x0 = X0[idx]` if pairs `X0` are given (the *same* indices); if a `coupling` is given (a function `(x0_batch, x1_batch) -> x0_batch` reordered, re-solved for every batch, as minibatch optimal transport does), `x0 = coupling(torch.randn(x1.shape, generator=g), x1)`; otherwise `x0=None` so the loss draws noise with `g`; then `flow_matching_loss(model, x1, x0=..., generator=g)`. `ValueError` if both `X0` and `coupling` are given. returns the losses as floats.

> **▶ build: stage 3.** implement `fit_flow_matching`.

### idea 4: rectified flow

where do non-crossing pairs come from? from the trained model itself (Liu, Gong & Liu, 2023). its ODE is a deterministic, invertible map from noise to data, and ODE trajectories can't cross (two trajectories through the same point at the same time would be the same trajectory). so:

1. train a flow with the independent coupling (the "1-rectified flow");
2. draw fresh noise $x_0$ and integrate the trained ODE to get $x_1 = \Phi(x_0)$: these **reflow pairs** have the data distribution (as well as the model learned it) and a deterministic coupling;
3. train a new flow on the pairs (the "2-rectified flow").

the new flow's straight training paths come from a coupling whose lines rarely cross, so its trajectories are much straighter, and few-step Euler sampling improves dramatically. reflow never improves the *distribution* (its data are the old model's samples, errors included), and each round costs a full sampling pass over the training set, so one or two rounds is typical. Liu et al. prove that rectification never increases any convex transport cost $\mathbb{E}\,c(x_1 - x_0)$: the coupling moves towards optimal transport.

to measure straightness, compare the velocity along a trajectory with its overall displacement:

$$S = \frac1n\sum_{k=0}^{n-1}\mathbb{E}\,\big\lVert (x_\text{end} - x_0) - v(x_k, t_k)\big\rVert^2,$$

along the model's own Euler path $x_k$ at $t_k = k/n$. a perfectly straight flow (constant velocity along every path) has $S = 0$.

> **spec · stage 4.**
> - `make_reflow_pairs(model, n, shape, *, n_steps=100, y=None, generator=None, device='cpu')`: `shape` is one example's shape (an int means one dimension); `x0 = torch.randn((n, *shape), generator=generator)` and `x1 = sample_flow(model, (n, *shape), n_steps=n_steps, x0=x0, y=y, device=device)` (Euler, the default); returns `(x0, x1)`. this is `mlfs.diffusion.make_reflow_pairs`.
> - `straightness(model, x0, *, n_steps=100, y=None)`: Euler-integrate from `x0` in `n_steps` steps ($x_{k+1} = x_k + v(x_k, t_k)/n$) and return $S$ above as a python float (squared norms summed over the non-batch dims, averaged over the batch and the steps); eval mode, no gradients.

> **▶ build: stage 4.** implement `make_reflow_pairs` and `straightness`.

> **🔮 predict:** after one round of reflow, how many Euler steps does the new flow need to match the quality of the original flow with 16 steps?

## worked example

**one loss term.** $x_0 = -0.8$, $x_1 = 1.4$, $t = 0.25$: $x_t = 0.75 \cdot (-0.8) + 0.25 \cdot 1.4 = -0.25$ and the target is $1.4 - (-0.8) = 2.2$. a network output of 1.9 contributes $(1.9 - 2.2)^2 = 0.09$.

**the exact field for $\mathcal{N}(1, 0.5^2)$.** at $t = 0$, $v^*(x, 0) = 1 - x$: from any noise value, head for the mean. at $t = 1$, $v^*(x, 1) = 1 + (x - 1) \cdot 0.25/0.25 = x$, which is $\mathbb{E}[x_1 - x_0 \mid x_1 = x] = x - 0$. the ODE carries $x_0 = 1$ to $1 + 0.5 = 1.5$ and $x_0 = -2$ to $0$. the checks run your `sample_flow` on this field with RK4 and 20 steps and require agreement with $\mu + s x_0$ to $10^{-3}$, and Euler's error > midpoint's > RK4's.

**the loss floor.** $\int_0^1 \frac{0.25}{(1 - t)^2 + 0.25\,t^2}\,dt = 0.5\cdot\frac\pi2 = 0.785$ for $s = 0.5$ (on paper 3). the checks evaluate your loss at the exact field on 100,000 points and expect 0.785.

**straightness by hand.** the field $v(x, t) = 0.5\,x$ with $n = 2$ Euler steps from $x_0 = 1$: $x_1 = 1.25$, $x_2 = 1.5625$, displacement $0.5625$; velocities $0.5$ and $0.625$; $S = \frac12\big((0.5625 - 0.5)^2 + (0.5625 - 0.625)^2\big) = 0.0039$.

## on paper

1. **the marginal velocity.** let $p_t$ be the density of $x_t = (1 - t)x_0 + t x_1$. show that $v^*(x, t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$ satisfies the continuity equation $\partial_t p_t + \nabla\cdot(p_t v^*) = 0$, so its flow transports $p_0$ to $p_1$.

<details><summary>solution</summary>

for a test function $\varphi$: $\frac{d}{dt}\mathbb{E}[\varphi(x_t)] = \mathbb{E}[\nabla\varphi(x_t)\cdot(x_1 - x_0)] = \mathbb{E}\big[\nabla\varphi(x_t)\cdot\mathbb{E}[x_1 - x_0 \mid x_t]\big] = \int \nabla\varphi\cdot v^*\,p_t\,dx = -\int\varphi\,\nabla\cdot(p_t v^*)\,dx$. the left side is $\int\varphi\,\partial_t p_t\,dx$, and since this holds for every $\varphi$, $\partial_t p_t = -\nabla\cdot(p_t v^*)$: the continuity equation of the ODE $\dot x = v^*$.

</details>

2. **why the conditional target works.** show that $\mathbb{E}\lVert v(x_t, t) - (x_1 - x_0)\rVert^2 = \mathbb{E}\lVert v(x_t, t) - v^*(x_t, t)\rVert^2 + C$ with $C$ independent of $v$.

<details><summary>solution</summary>

add and subtract $v^*$: the cross term $\mathbb{E}[(v - v^*)\cdot(v^* - (x_1 - x_0))]$ vanishes, because conditionally on $x_t$ the first factor is fixed and the second has mean zero. so the CFM loss is the "marginal" flow matching loss plus $C = \mathbb{E}\lVert v^* - (x_1 - x_0)\rVert^2 = \mathbb{E}\operatorname{tr}\operatorname{Cov}(x_1 - x_0 \mid x_t)$, the irreducible part.

</details>

3. **the floor in closed form.** with $\mu = 0$, evaluate $\int_0^1 \frac{s^2}{(1 - t)^2 + t^2 s^2}dt$ for $s > 0$. what is it for $s = 1$?

<details><summary>solution</summary>

complete the square: $(1 - t)^2 + t^2 s^2 = (1 + s^2)t^2 - 2t + 1 = \frac{((1 + s^2)t - 1)^2 + s^2}{1 + s^2}$. with $u = ((1 + s^2)t - 1)/s$, $du = \frac{1 + s^2}{s}dt$, the integrand becomes $\frac{s^2(1 + s^2)}{s^2(u^2 + 1)}\cdot\frac{s}{1 + s^2}du = \frac{s\,du}{u^2 + 1}$, so the integral is $s\,[\arctan u]_{-1/s}^{s} = s(\arctan s + \arctan(1/s)) = s\pi/2$. for $s = 1$ that's $\pi/2$: even when noise and data have the *same* distribution, where the identity map would be a perfect flow, independent pairing leaves a loss of 1.57 per dimension, and the learned field is not the identity: at $t = 0$ it points every $x_0$ towards 0.

</details>

4. **one Euler step from the independent coupling.** show that one Euler step of the exact marginal field from $x_0$ lands at $\mathbb{E}[x_1]$, whatever $x_0$, and that one Euler step of a flow trained on a deterministic coupling $x_1 = T(x_0)$ (with zero loss) lands at $T(x_0)$.

<details><summary>solution</summary>

at $t = 0$, $x_t = x_0$ and $v^*(x_0, 0) = \mathbb{E}[x_1 - x_0 \mid x_0] = \mathbb{E}[x_1] - x_0$ under independence, so $x_0 + v^*(x_0, 0) = \mathbb{E}[x_1]$: every one-step sample is the data mean. with a deterministic coupling and straight non-crossing paths, $v^*(x_t, t) = T(x_0) - x_0$ along each path, constant in $t$, so a single step of size 1 lands exactly on $T(x_0)$.

</details>

## experiment

`lab.py` trains your flows on 2-D data:

1. independent-coupling flow matching against one round of reflow: sample quality against the number of Euler steps, and straightness.
2. what the trajectories look like.
3. the loss you can't get rid of.

## check your understanding

1. flow matching and diffusion both learn a field that an ODE solver integrates. what's actually different?

<details><summary>answer</summary>

the path from noise to data. diffusion's paths are fixed by the noise schedule ($\sqrt{\bar\alpha}x_0 + \sqrt{1 - \bar\alpha}\epsilon$, with curved coefficients and a $t \to $ pure-noise endpoint reached only asymptotically), and its network predicts noise; flow matching picks straight-line interpolation with exact endpoints and predicts velocity. with Gaussian paths the two are interconvertible (for the linear path, $\epsilon = x_t - t\,v$ up to the reversed time convention and $\hat x_1 = x_t + (1 - t)v$), so the differences are in weighting across times and in how curved the resulting ODE is.

</details>

2. why can't ODE trajectories cross, and why does that make reflow work?

<details><summary>answer</summary>

an ODE with a (locally Lipschitz) velocity field has a unique solution through each point at each time, so two trajectories that meet coincide from then on. the trained flow's map $x_0 \mapsto x_1$ therefore pairs noise and data without crossings, and the straight lines between those pairs cross far less than independent pairs do; training on them gives a field whose trajectories are nearly those lines.

</details>

3. a 2-rectified flow samples in one step nearly as well as the 1-rectified flow does in 16. why not keep reflowing?

<details><summary>answer</summary>

each round trains on the previous model's samples, so modelling errors accumulate (the distribution can only get worse), and each round costs a full sampling pass plus a training run. the straightness gain is largest in the first round; after one or two, distillation into a one-step model is the usual next move.

</details>

4. why is `x0` an argument of `flow_matching_loss`?

<details><summary>answer</summary>

to support couplings other than independent noise: reflow pairs, minibatch optimal-transport pairs (the OT elective), or paired data (image-to-image translation). the loss is the same; only where $x_0$ comes from changes.

</details>

5. the CFM loss of a model is 1.6 per dimension. is it a bad model?

<details><summary>answer</summary>

not necessarily: the floor depends on the data and the coupling ($\pi/2$ per dimension for standard normal data with independent noise). the loss tells you about the regression, not directly about sample quality; compare against the floor where you know it, or measure samples.

</details>

## common bugs

- **reversing the time convention.** symptom: samples come out as noise; in this chapter $t = 0$ is noise and $t = 1$ is data for flow matching (the opposite of diffusion's $t$).
- **regressing on $x_1$ instead of $x_1 - x_0$.** symptom: training converges, sampling diverges; the network must output a velocity.
- **different indices for the pairs.** symptom: with `X0` given, the loss stays at the independent-coupling floor; use the *same* `idx` for both arrays.
- **broadcasting `t`.** symptom: images `(B, C, H, W)` times a `(B,)` `t` either fails or broadcasts along `W`; reshape to `(B, 1, 1, 1)`.
- **reflow pairs with too few solver steps.** symptom: the 2-rectified flow learns the 1-rectified flow's discretization errors; generate pairs with enough steps (or a higher-order solver).
- **sampling in train mode.** symptom: dropout noise in the velocity, jagged trajectories.

## going deeper

- Lipman, Chen, Ben-Hamu, Nickel & Le (2023), "Flow matching for generative modeling", *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Liu, Gong & Liu (2023), "Flow straight and fast: learning to generate and transfer data with rectified flow", *ICLR*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Albergo & Vanden-Eijnden (2023), "Building normalizing flows with stochastic interpolants", *ICLR*. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)
- Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf & Bengio (2024), "Improving and generalizing flow-based generative models with minibatch optimal transport", *TMLR*. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
- Esser et al. (2024), "Scaling rectified flow transformers for high-resolution image synthesis", *ICML*. Stable Diffusion 3. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
- Lipman, Havasi, Holderrieth, Shaul, Le, Karrer, Chen, Lopez-Paz, Ben-Hamu & Gat (2024), "Flow matching guide and code". [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)

## connections

- ← builds on: ODE solvers (ch 20), normalizing flows (ch 21), denoising score matching (topic 02), the probability-flow ODE (topic 04), `TimeMLP` (topic 03).
- → used in: conditional generation and guidance, where the same `cfg_model` works on velocities (topic 06); latent flow matching with a DiT (topic 07); the capstone's flow-matching model; the optimal-transport elective (topics 15–17), which replaces the independent coupling by minibatch OT pairs through `x0=`.
