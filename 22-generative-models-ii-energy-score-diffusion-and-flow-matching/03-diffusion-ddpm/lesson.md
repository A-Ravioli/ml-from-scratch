# diffusion (DDPM)

## why this matters

topic 02 ended with a recipe that works but feels improvised: a hand-picked ladder of noise levels, a network conditioned on $\sigma$, Langevin at each level with a step-size rule and a number of steps you tune by eye. **denoising diffusion probabilistic models** (Sohl-Dickstein et al., 2015; Ho, Jain & Abbeel, 2020) turn the same idea into a clean latent-variable model. fix a *forward process* that destroys an image in $T$ small Gaussian steps, from data to pure noise, with nothing to learn. then train a network to run it backwards, one step at a time. the training objective falls out of the variational bound you met in chapter 12, and after some algebra it's almost embarrassingly simple: add noise to an image, ask the network what noise you added, and penalize the squared error.

this is the model family behind DALL·E 2, Imagen, Stable Diffusion and most image, audio and video generators since 2021. in this topic you'll build all of it: the noise schedule, the networks that know what time it is, the loss, the sampler, and a U-Net big enough for CIFAR-10. the price you'll notice immediately: a sample costs one network evaluation *per step*, a thousand of them by default. topic 04 is about getting that down to 20.

## warm-up: predict first

1. you add independent Gaussian noise of variance 0.01 to an image 1,000 times, shrinking the image a little each time so the total variance stays at 1. how much of the original image is left?
2. after $t$ such steps, do you have to simulate all $t$ of them to get a noisy image, or is there a shortcut?
3. the network is trained to predict the noise $\epsilon$ that was added. at which noise levels is that easy, and at which is it hard?
4. a perfectly trained noise predictor on Gaussian data: is its loss 0?
5. why must a denoiser be told the noise level, instead of working it out from the image?

## prerequisites check

- [ ] denoising score matching, Tweedie's formula, noise ladders (→ topic 02)
- [ ] the ELBO and KL divergences between Gaussians (→ ch 12 · topic 07; ch 21 · topic 03)
- [ ] products and conditionals of Gaussians (→ ch 04 · topic 04)
- [ ] residual blocks, group/batch normalization and the U-Net (→ ch 16 · topics 04 and 06), sinusoidal position features (→ ch 18 · topic 02)
- [ ] training loops, Adam, `nn.Module` train/eval modes (→ ch 15)

## core ideas

### idea 1: the forward process and its shortcut

pick a schedule of small variances $\beta_0, \dots, \beta_{T-1}$ and define a Markov chain that, at every step, shrinks the image slightly and adds a little noise:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\,x_{t-1},\ \beta_t I\big).$$

the shrink factor $\sqrt{1 - \beta_t}$ is chosen so that a unit-variance input stays unit-variance ("variance preserving"). because every step is linear-Gaussian, the composition of $t$ steps is Gaussian too, and you never need to simulate the chain (on paper 1):

$$q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\ (1 - \bar\alpha_t) I\big), \qquad \alpha_t = 1 - \beta_t, \quad \bar\alpha_t = \prod_{s \le t}\alpha_s,$$

so a noisy version at any step is one line: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$. $\bar\alpha_t$ is the fraction of the signal's variance left at step $t$; the **signal-to-noise ratio** $\bar\alpha_t / (1 - \bar\alpha_t)$ falls from huge to almost zero. (the indexing here starts at $t = 0$, the first noisy step, with $\bar\alpha_0 = \alpha_0$; papers usually count from 1. the network is told the time as $t / T \in [0, 1)$.)

the schedule is a design choice with visible consequences:

- **linear** (Ho et al., 2020): $\beta_t$ from $10^{-4}$ to $0.02$ over $T = 1000$ steps. by the end $\bar\alpha \approx 4 \times 10^{-5}$: pure noise, as the sampler needs. but the image is mostly destroyed long before the end, so many steps are spent on noise.
- **cosine** (Nichol & Dhariwal, 2021): define $\bar\alpha$ directly, $\bar\alpha_t = f(t + 1)/f(0)$ with $f(u) = \cos^2\big(\frac{u/T + 0.008}{1.008}\cdot\frac\pi2\big)$, and recover $\beta_t = 1 - \bar\alpha_t/\bar\alpha_{t-1}$, clipped at 0.999 (the last step would otherwise be exactly 1). it destroys information more evenly over time.

> **spec · stage 1.** `DiffusionSchedule(n_steps=1000, *, beta_start=0.0001, beta_end=0.02, schedule='linear')`, a plain class with attributes `n_steps` and float32 tensors `betas`, `alphas` $= 1 - \beta$, `alpha_bars` $= \operatorname{cumprod}(\alpha)$, all `(n_steps,)` (compute in float64, then cast). `'linear'`: `betas = linspace(beta_start, beta_end, n_steps)`; `'cosine'`: as above (the `beta_*` arguments ignored); anything else raises `ValueError`. `q_sample(x0, t, noise=None)`: `t` int64 `(B,)` steps, `noise` standard normal if `None`; returns $\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\text{noise}$, broadcast over the non-batch dims. `to(device)` moves the three tensors and returns `self`. this is `mlfs.diffusion.DiffusionSchedule`.

> **▶ build: stage 1.** implement `DiffusionSchedule` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** with the default linear schedule, at which fraction of the 1,000 steps is $\bar\alpha_t$ down to 1% (an SNR of 0.01)? and with the cosine schedule?

### idea 2: networks that know what time it is

the same network handles every noise level, so it must be told the level. the standard encoding is borrowed from transformers' positions (chapter 18): map the time to a vector of sines and cosines at geometrically spaced frequencies,

$$\phi(t)_k = \cos(1000\,t\,f_k), \quad \phi(t)_{k + d/2} = \sin(1000\,t\,f_k), \qquad f_k = M^{-k/(d/2)},\ k < d/2,$$

with $M = 10{,}000$ the maximum period. the factor 1,000 is there so that $t \in [0, 1]$ spans the same range as the integer steps of the original paper. a small MLP on top (Linear, SiLU, Linear) lets the network shape the embedding. low frequencies tell coarse times apart, high frequencies fine ones, and every time gets a distinct, smoothly varying code.

for 2-D experiments the denoiser is an MLP that adds a projection of the time embedding to every hidden layer. for conditional models (topic 06) it also adds a learned class embedding, from a table with one extra row: label $K$ (for $K$ classes) means "no label", the **null label**, and that is what a missing `y` means.

> **spec · stage 2.**
> - `sinusoidal_features(t, dim, *, max_period=10000.0)`: `t` `(B,)` → `(B, dim)`, cosines in the first half, sines in the second, as above; a zero column is appended when `dim` is odd.
> - `TimeEmbedding(dim, *, max_period=10000.0)`: `sinusoidal_features` then an MLP `Linear(dim, dim)`, SiLU, `Linear(dim, dim)`; `(B,)` → `(B, dim)`. this is `mlfs.diffusion.TimeEmbedding`.
> - `TimeMLP(data_dim=2, *, hidden_dim=256, n_layers=3, time_dim=64, n_classes=None)`: `forward(x, t, y=None)` maps `(B, data_dim)` to `(B, data_dim)`; a `TimeEmbedding(time_dim)`, plus `nn.Embedding(n_classes + 1, time_dim)` when `n_classes` is set (`y=None` means the null label `n_classes`; passing `y` to an unconditional model raises `ValueError`); an MLP with `n_layers` hidden layers of width `hidden_dim` that uses the conditioning (internal design yours). this is `mlfs.diffusion.TimeMLP`.

> **▶ build: stage 2.** implement `sinusoidal_features`, `TimeEmbedding` and `TimeMLP`.

### idea 3: the loss: predict the noise

the generative model runs the chain backwards: $p(x_{T-1}) = \mathcal{N}(0, I)$ and Gaussian steps $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$. like a VAE with $T$ layers of latents and a fixed encoder, it's trained by maximizing the ELBO, which decomposes into one KL per step between the model's step and the **forward posterior** $q(x_{t-1} \mid x_t, x_0)$, which is Gaussian and known in closed form (on paper 2):

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(\tilde\mu_t(x_t, x_0),\ \tilde\beta_t I\big), \qquad \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t.$$

a KL between Gaussians with equal variances is a squared distance between their means, so each term asks $\mu_\theta(x_t, t)$ to match $\tilde\mu_t(x_t, x_0)$. now substitute $x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$: the target mean becomes (on paper 3)

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\epsilon\Big),$$

where $x_t$ is known to the network and only $\epsilon$ isn't. so parametrize $\mu_\theta$ the same way with a network $\epsilon_\theta(x_t, t)$ that *predicts the noise*, and every KL becomes a weighted squared error between $\epsilon_\theta$ and $\epsilon$. Ho et al. found that dropping the weights works better for sample quality:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{t \sim \mathcal{U}\{0, \dots, T-1\},\ x_0,\ \epsilon}\ \big\lVert \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon,\ t\big) - \epsilon\big\rVert^2.$$

this is denoising score matching again: by Tweedie, the optimal predictor is $\epsilon^*(x_t, t) = -\sqrt{1 - \bar\alpha_t}\,\nabla\log q_t(x_t)$, a scaled score of the noisy marginal, and the loss is DSM with topic 02's $\sigma^2$ weighting, in rescaled coordinates. like DSM it never reaches zero: for Gaussian data $\mathcal{N}(\mu, s^2)$ the optimal predictor is $\epsilon^*(x_t) = \sqrt{1 - \bar\alpha}\,(x_t - \sqrt{\bar\alpha}\,\mu)/(\bar\alpha s^2 + 1 - \bar\alpha)$ and its loss is $\mathbb{E}_t\big[\bar\alpha_t s^2/(\bar\alpha_t s^2 + 1 - \bar\alpha_t)\big]$ per dimension (on paper 4). at high noise $\epsilon$ is nearly all there is in $x_t$, so it's easy to predict; at low noise it's a tiny perturbation of the image, and telling it apart from the image's own fine detail is hard.

> **spec · stage 3.**
> - `ddpm_loss(model, schedule, x0, *, y=None, generator=None)`: draw `t = torch.randint(n_steps, (B,), generator=generator)` first, then `noise = torch.randn(x0.shape, generator=generator)` (both on the CPU, moved to `x0`'s device); return the mean squared error between `model(q_sample(x0, t, noise), t.float() / n_steps, y)` and `noise`. this is `mlfs.diffusion.ddpm_loss`.
> - `fit_ddpm(model, schedule, X, *, n_iters=2000, batch_size=256, lr=1e-3, seed=0)`: `g = torch.Generator().manual_seed(seed)`, `torch.optim.Adam(lr)`, the model in train mode; per step `X[torch.randint(N, (batch_size,), generator=g)]` and `ddpm_loss(..., generator=g)`; returns the losses as floats.

> **▶ build: stage 3.** implement `ddpm_loss` and `fit_ddpm`.

### idea 4: ancestral sampling

sampling runs the learned reverse chain: start from pure noise, and at each step move to the predicted mean and add fresh noise,

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\epsilon_\theta(x_t, t)\Big) + \sigma_t z, \qquad z \sim \mathcal{N}(0, I),$$

with no noise at the last step. two choices of $\sigma_t^2$ are standard: $\beta_t$ (the forward step's variance, used here) and $\tilde\beta_t$ (the posterior's); with $T = 1000$ they give nearly the same samples. notice that the update is "Langevin-like": $-\epsilon_\theta/\sqrt{1 - \bar\alpha}$ is the score, so each step is a small move up the noisy density plus noise, at a noise level that shrinks as you go. unlike annealed Langevin, though, the step sizes and the noise are not tuned by hand: they're fixed by the forward process, which is exactly what makes the sampler correct.

if the network were perfect, would the samples be exact? for Gaussian data you can check, because $\epsilon^*$ is known: plug it in, and your sampler must return $\mathcal{N}(\mu, s^2)$. the checks do exactly that. the cost is the elephant in the room: $T$ sequential network evaluations per sample.

samplers evaluate the network in **eval mode** (dropout off) and without gradients, and should leave the model the way they found it: a sampler that calls `model.eval()` and never calls `model.train()` again silently turns off dropout for the rest of training.

> **spec · stage 4.** `ddpm_sample(model, schedule, shape, *, y=None, generator=None, device='cpu')`: `x = torch.randn(shape, generator=generator)` (CPU, then `device`); for $i = T-1, \dots, 0$: `eps = model(x, torch.full((B,), i / T), y)`, the mean above, then add $\sqrt{\beta_i}\,z$ with `z = torch.randn(shape, generator=generator)` for $i > 0$ only. no gradients; if `model` is an `nn.Module`, evaluate it in eval mode and restore its previous mode afterwards. returns $x$. this is `mlfs.diffusion.ddpm_sample`.

> **▶ build: stage 4.** implement `ddpm_sample`.

> **🔮 predict:** you train a 2-D diffusion model with $T = 10$ steps instead of 1,000 (and a schedule rescaled to reach pure noise). each sample is 100 times cheaper. how much worse will the samples be?

### idea 5: a U-Net that knows the time

for images the denoiser is a U-Net (chapter 16): a downsampling path, a bottleneck, and an upsampling path with skip connections that carry fine detail across. the skip connections matter more than usual: at low noise the output is almost the input, and the network only has to predict a small correction, so a direct path for the full-resolution input is essential. the diffusion version adds three things:

- **residual blocks** (GroupNorm, SiLU, 3×3 convolution, twice, plus a skip), with the time embedding projected and *added to the feature map* inside every block, so every layer knows the noise level;
- **self-attention** at the lowest resolution, which gives every pixel a view of the whole image (a denoiser that only sees a local patch can't decide what object the patch belongs to);
- **dropout** in the residual blocks, which matters on small datasets like CIFAR-10;
- and, for class-conditional models, a class embedding added to the time embedding (the same null-label convention as `TimeMLP`).

a widely used trick is to zero-initialize the final convolution, so the untrained network predicts $\epsilon = 0$ and starts with a loss of exactly 1. `ResBlock` and `SelfAttention2d` are given in the notebook; the assembly is yours.

> **spec · stage 5.** `TimeUNet(in_channels=3, *, base_channels=64, channel_mults=(1, 2, 2), n_classes=None, dropout=0.0)`: `forward(x, t, y=None)` maps `(B, C, H, W)` to the same shape. one resolution level per entry of `channel_mults` (width `base_channels * mult`), downsampling by 2 between levels, so $H$ and $W$ must be divisible by $2^{\text{len}(\text{channel\_mults}) - 1}$ (`ValueError` otherwise); residual blocks with the time (+ class) embedding and `dropout`; global context at the lowest resolution (the reference uses self-attention; the checks require every output pixel to depend on every input pixel); a mirrored up path with skip connections; one class table `nn.Embedding(n_classes + 1, ·)` when `n_classes` is set (null label and `ValueError` as for `TimeMLP`). this is `mlfs.diffusion.TimeUNet`.

> **▶ build: stage 5.** implement `TimeUNet`.

## worked example

**a three-step schedule.** $\beta = (0.1, 0.2, 0.3)$: $\alpha = (0.9, 0.8, 0.7)$ and $\bar\alpha = (0.9, 0.72, 0.504)$. a pixel $x_0 = 1$ with noise $\epsilon = -0.5$ at step $t = 1$: $x_1 = \sqrt{0.72} - 0.5\sqrt{0.28} = 0.849 - 0.265 = 0.584$. the network is told $t = 1/3$.

**one reverse step.** at $t = 1$ with $x_1 = 0.584$ and a perfect noise prediction $\epsilon_\theta = -0.5$: the mean is $(0.584 - \frac{0.2}{\sqrt{0.28}}(-0.5))/\sqrt{0.8} = (0.584 + 0.189)/0.894 = 0.864$, and with $z$ drawn, $x_0 = 0.864 + \sqrt{0.2}\,z$. (with $\sigma_t^2 = \tilde\beta_1 = 0.2 \cdot 0.1/0.28 = 0.071$ instead, the noise would be $0.27 z$ rather than $0.45 z$.) the checks run your sampler on a three-step schedule with a toy network and compare every number, including where the noise draws go.

**the default schedule's endpoint.** $\log\bar\alpha_{999} = \sum_t \log(1 - \beta_t) \approx -\sum_t \beta_t = -1000 \cdot 0.01005 \approx -10.05$ (the second-order terms take it to about $-10.1$): $\bar\alpha_{999} \approx 4 \times 10^{-5}$, so $x_{999}$ keeps $\sqrt{\bar\alpha} \approx 0.6\%$ of the image's amplitude.

**the loss floor.** for 1-D Gaussian data with $s = 0.5$, a schedule with 100 steps (β from $10^{-3}$ to 0.2): averaging $\bar\alpha s^2 / (\bar\alpha s^2 + 1 - \bar\alpha)$ over the 100 steps gives the smallest possible loss, 0.171. the checks compute it from your schedule and require the exact predictor to reach it, and your trained `TimeMLP` to come within 0.03.

## on paper

1. **the shortcut.** show by induction that $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1 - \bar\alpha_t)I)$.

<details><summary>solution</summary>

true for $t = 0$. if $x_{t-1} = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1 - \bar\alpha_{t-1}}\,\epsilon'$, then $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\,\epsilon'' = \sqrt{\bar\alpha_t}x_0 + \big(\sqrt{\alpha_t(1 - \bar\alpha_{t-1})}\,\epsilon' + \sqrt{1 - \alpha_t}\,\epsilon''\big)$. the bracket is a sum of independent zero-mean Gaussians with variance $\alpha_t - \bar\alpha_t + 1 - \alpha_t = 1 - \bar\alpha_t$.

</details>

2. **the forward posterior.** use Bayes' rule, $q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)$, to find its mean and variance (one dimension is enough).

<details><summary>solution</summary>

both factors are Gaussian in $x_{t-1}$: precision $\alpha_t/\beta_t$ from the first (mean $x_t/\sqrt{\alpha_t}$) and $1/(1 - \bar\alpha_{t-1})$ from the second (mean $\sqrt{\bar\alpha_{t-1}}x_0$). precisions add: $\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}} = \frac{\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t}{\beta_t(1 - \bar\alpha_{t-1})} = \frac{1 - \bar\alpha_t}{\beta_t(1 - \bar\alpha_{t-1})}$, so $\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t$. the mean is the precision-weighted average: $\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}x_t$.

</details>

3. **from $x_0$ to $\epsilon$.** substitute $x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ into $\tilde\mu_t$ and simplify to $\frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\epsilon\big)$.

<details><summary>solution</summary>

the $x_0$ coefficient times $1/\sqrt{\bar\alpha_t}$ is $\frac{\beta_t}{\sqrt{\alpha_t}(1 - \bar\alpha_t)}$ (using $\bar\alpha_t = \alpha_t\bar\alpha_{t-1}$). the coefficient of $x_t$ becomes $\frac{\beta_t + \alpha_t(1 - \bar\alpha_{t-1})}{\sqrt{\alpha_t}(1 - \bar\alpha_t)} = \frac{1 - \bar\alpha_t}{\sqrt{\alpha_t}(1 - \bar\alpha_t)} = \frac{1}{\sqrt{\alpha_t}}$, and the coefficient of $\epsilon$ is $-\frac{\beta_t}{\sqrt{\alpha_t}(1 - \bar\alpha_t)}\sqrt{1 - \bar\alpha_t} = -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar\alpha_t}}$.

</details>

4. **the optimal noise predictor for Gaussian data.** for $x_0 \sim \mathcal{N}(\mu, s^2)$, find $\mathbb{E}[\epsilon \mid x_t]$ and the expected squared error of the best predictor at step $t$.

<details><summary>solution</summary>

$(\epsilon, x_t)$ is jointly Gaussian with $\operatorname{Cov}(\epsilon, x_t) = \sqrt{1 - \bar\alpha}$ and $\operatorname{Var}(x_t) = \bar\alpha s^2 + 1 - \bar\alpha =: v$. so $\mathbb{E}[\epsilon \mid x_t] = \sqrt{1 - \bar\alpha}\,(x_t - \sqrt{\bar\alpha}\mu)/v$ and $\operatorname{Var}(\epsilon \mid x_t) = 1 - (1 - \bar\alpha)/v = \bar\alpha s^2/v$. near $t = 0$ ($\bar\alpha \to 1$) the floor is close to 1 for any $s$ much larger than the noise: predicting the noise is nearly hopeless; near $t = T$ it goes to 0.

</details>

## experiment

`lab.py` uses your schedule, networks, loss and sampler:

1. what the linear and cosine schedules do to a digit, and how much signal survives at each step.
2. the loss as a function of the noise level, on data that lie on a curve.
3. the number of diffusion steps against sample quality and cost.

## check your understanding

1. why is the forward process given $\sqrt{1 - \beta_t}$ shrinking instead of just adding noise?

<details><summary>answer</summary>

without shrinking, the variance grows to $1 + \sum\beta_t$ and the endpoint depends on the data's scale; with it, $x_T$ is (nearly) $\mathcal{N}(0, I)$ whatever the data, which is the known distribution the sampler starts from. this is the "variance preserving" choice; the "variance exploding" alternative (topic 02's ladder) adds noise without shrinking and starts sampling from a very wide Gaussian.

</details>

2. why is the network trained on random $t$ instead of running through the chain step by step?

<details><summary>answer</summary>

the shortcut $q(x_t \mid x_0)$ gives a noisy sample at any step in one line, and the ELBO is a sum of independent per-step terms, so a random $t$ per example is an unbiased estimate of the full sum, with no sequential simulation and every step trained equally often.

</details>

3. a model predicts $\epsilon$; another predicts $x_0$; a third predicts the score. are they different models?

<details><summary>answer</summary>

not in what they can represent: $x_0 = (x_t - \sqrt{1 - \bar\alpha}\,\epsilon)/\sqrt{\bar\alpha}$ and $\nabla\log q_t = -\epsilon/\sqrt{1 - \bar\alpha}$ convert one into another. they differ in how the regression is weighted across noise levels (since $\hat x_0 - x_0 = -\sqrt{(1 - \bar\alpha)/\bar\alpha}\,(\hat\epsilon - \epsilon)$, an MSE on $x_0$ is an MSE on $\epsilon$ divided by the SNR) and in numerical behaviour at the ends: $\epsilon$-prediction is ill-conditioned for recovering $x_0$ at high noise. the "v-prediction" $v = \sqrt{\bar\alpha}\,\epsilon - \sqrt{1 - \bar\alpha}\,x_0$ (Salimans & Ho, 2022) balances the two and is close to the velocity of topic 05.

</details>

4. why does the loss at small $t$ stay high even for a perfect model?

<details><summary>answer</summary>

at small $t$ the noise is a tiny perturbation, and from $x_t$ alone you can't tell which part of the fine detail is noise and which is image (the image distribution has detail at every scale). the conditional variance of $\epsilon$ given $x_t$ stays near 1 there (on paper 4); the loss is high because the task is impossible, not because the network is bad.

</details>

5. what happens if the sampler leaves the model in eval mode?

<details><summary>answer</summary>

if you sample during training (for a progress picture, or an FID estimate), every later training step runs with dropout off and batch-norm statistics frozen: training silently changes. samplers must restore the mode they found.

</details>

6. why add the time embedding inside every residual block, instead of once at the input?

<details><summary>answer</summary>

the right operation depends strongly on the noise level everywhere in the network (at high noise, sketch global structure; at low noise, polish texture). an input-only time channel is diluted by the first few layers; adding it in every block lets each layer modulate its features with the time directly, the same reason DiT's adaLN conditions every block (topic 07).

</details>

## common bugs

- **off-by-one in $\bar\alpha$.** symptom: `alpha_bars[0] == 1`, so step 0 adds no noise and the sampler divides by zero at $t = 0$ (its coefficient $\beta_0/\sqrt{1 - \bar\alpha_0}$ becomes $\beta_0/0$).
- **feeding the integer step instead of $t / T$.** symptom: a model trained on 0..999 and sampled with 0..1 (or vice versa) outputs nonsense at sampling time while the training loss looked fine.
- **broadcasting $\bar\alpha_t$ wrongly.** symptom: `alpha_bars[t]` of shape `(B,)` against images `(B, C, H, W)` raises an error or, worse, broadcasts along $W$ when $B = W$. reshape to `(B, 1, 1, 1)`.
- **adding noise at the last step.** symptom: samples are visibly grainy; the final step returns the mean.
- **sampling with gradients on.** symptom: memory grows with $T$ until it runs out at a few hundred steps.
- **an EMA of the weights that starts averaging at step 0.** symptom: early in training the averaged model samples far worse than the raw one (a decay of 0.999 still holds 22% of the random initialization after 1,500 steps, and a noise predictor that's slightly too small is amplified by $1/\sqrt{\bar\alpha}$ at high noise); warm the decay up, e.g. $\min(0.999, (1 + k)/(10 + k))$ at step $k$. the capstone measures this.
- **forgetting `model.eval()` / restoring `model.train()`.** symptom: noisy samples from dropout, or training that changes after the first progress picture.

## going deeper

- Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli (2015), "Deep unsupervised learning using nonequilibrium thermodynamics", *ICML*. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
- Ho, Jain & Abbeel (2020), "Denoising diffusion probabilistic models", *NeurIPS*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021), "Improved denoising diffusion probabilistic models", *ICML*. the cosine schedule and learned variances. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Kingma, Salimans, Poole & Ho (2021), "Variational diffusion models", *NeurIPS*. the ELBO in terms of the SNR. [arXiv:2107.00630](https://arxiv.org/abs/2107.00630)
- Salimans & Ho (2022), "Progressive distillation for fast sampling of diffusion models", *ICLR*. v-prediction. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
- Luo (2022), "Understanding diffusion models: a unified perspective". a careful derivation of everything in this topic. [arXiv:2208.11970](https://arxiv.org/abs/2208.11970)

## connections

- ← builds on: denoising score matching and noise ladders (topic 02), the ELBO (ch 12, ch 21), Gaussian conditioning (ch 04), U-Nets (ch 16), sinusoidal features (ch 18).
- → used in: fast samplers for the same trained networks (topic 04), class conditioning and guidance (topic 06), latent diffusion (topic 07), the chapter capstone on CIFAR-10, and the R2 milestone (reproducing DDPM at small scale).
