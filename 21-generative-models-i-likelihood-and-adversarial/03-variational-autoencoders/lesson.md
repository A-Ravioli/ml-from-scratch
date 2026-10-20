# variational autoencoders

## why this matters

an autoregressive model (topic 02) has no idea what a digit *is*. it predicts pixel 412 from pixels 1 to 411 and never forms a compact summary like "a slanted 7 with a thin stroke". a **latent-variable model** does: it tells a two-step story, first draw a small code $z \sim p(z)$ (the "idea" of the image), then draw the image from $p_\theta(x \mid z)$, and a neural network decoder turns codes into pixels. the code space becomes something you can walk through, edit, and compress into, and chapter 22's latent diffusion models run their whole generative process inside it.

the price is the likelihood. $p_\theta(x) = \int p_\theta(x \mid z)\,p(z)\,dz$ is an integral over every possible code, with a neural network inside, and no algorithm computes it exactly. you met the way out in chapter 12: variational inference maximizes a lower bound, the **ELBO**, using an approximate posterior $q(z \mid x)$. the **variational autoencoder** (Kingma & Welling, 2014; Rezende, Mohamed & Wierstra, 2014) adds two ideas that made this work at scale: an **encoder** network that outputs $q(z \mid x)$ for any $x$ in one pass (amortized inference, instead of an optimization per data point), and the **reparameterization trick**, which lets gradients flow through the random sampling of $z$. you'll build both, the MLP and convolutional VAEs the rest of the course reuses, a tighter likelihood estimate than the ELBO, and the diagnostics for the VAE's best-known failure: latent dimensions that the model simply stops using.

## warm-up: predict first

1. you train an autoencoder (no randomness, no KL) and decode a point halfway between two codes. what do you expect to see?
2. you need $\nabla_\phi\, \mathbb{E}_{z \sim q_\phi}[f(z)]$. why can't you just sample $z$ and backpropagate through $f(z)$?
3. a VAE reports a negative ELBO of 100 nats on test digits. is its true NLL above or below 100?
4. you give a VAE 32 latent dimensions. will it use all of them?
5. you multiply the KL term by $\beta = 4$. do reconstructions get sharper or blurrier?

## prerequisites check

- [ ] the ELBO, mean-field variational inference and the reparameterization idea (→ ch 12 · topic 07)
- [ ] latent variables and EM; probabilistic PCA (→ ch 11 · topics 02–03)
- [ ] KL divergence between Gaussians (→ ch 05 · topic 05); the multivariate Gaussian (→ ch 04 · topic 04)
- [ ] binary cross-entropy with logits (→ ch 08 · topic 04); convolutions and transposed convolutions (→ ch 16 · topics 01, 06)
- [ ] bits per dimension and likelihood bounds (→ topic 01)

## core ideas

### idea 1: the reparameterization trick, and a KL in closed form

recall the bound. for any distribution $q(z \mid x)$,

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q}\big[\log p_\theta(x \mid z)\big] - \operatorname{KL}\big(q(z \mid x) \,\|\, p(z)\big)}_{\text{ELBO}} + \operatorname{KL}\big(q(z \mid x) \,\|\, p_\theta(z \mid x)\big).$$

the last KL is $\ge 0$ and unknowable, so the ELBO is a lower bound on $\log p_\theta(x)$, tight when $q$ equals the true posterior. a VAE uses a diagonal Gaussian $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \operatorname{diag}\sigma^2_\phi(x))$ whose mean and variance come from an encoder network, and a standard normal prior $p(z) = \mathcal{N}(0, I)$.

training needs the gradient of the ELBO with respect to the encoder's parameters $\phi$, and $\phi$ sits in the *distribution* the expectation is taken over. sampling $z$ and backpropagating doesn't work: the sampling step has no derivative with respect to $\phi$. the **reparameterization trick** moves the randomness out of the way. draw parameter-free noise $\epsilon \sim \mathcal{N}(0, I)$ and write

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \sigma = \exp(\tfrac12 \log\sigma^2).$$

now $z$ is a deterministic, differentiable function of $\phi$ given $\epsilon$, so $\nabla_\phi\,\mathbb{E}_\epsilon[f(z)] = \mathbb{E}_\epsilon[\nabla_\phi f(\mu + \sigma\epsilon)]$, and one sample gives an unbiased, low-variance gradient estimate. the encoder outputs $\log\sigma^2$ (called `logvar`) rather than $\sigma$, so it can be any real number.

the KL term needs no sampling at all. between a diagonal Gaussian and $\mathcal{N}(0, I)$ it has a closed form (on paper 1), summed over latent coordinates:

$$\operatorname{KL}\big(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)\big) = \tfrac12\big(\mu^2 + \sigma^2 - 1 - \log\sigma^2\big).$$

it's zero exactly when $\mu = 0$ and $\sigma = 1$: a code that says nothing about $x$.

> **spec · stage 1.**
> - `reparameterize(mu, logvar, *, generator=None)`: `mu + exp(logvar / 2) * eps` with `eps = torch.randn(mu.shape, generator=generator)` (moved to `mu`'s device), any shape. this is `mlfs.generative.reparameterize`.
> - `kl_standard_normal(mu, logvar)`: the closed-form KL summed over **every non-batch dimension** → `(B,)` (so it works for flat `(B, L)` and spatial `(B, C, h, w)` latents). this is `mlfs.generative.kl_standard_normal`.

> **▶ build: stage 1.** implement `reparameterize` and `kl_standard_normal` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: the VAE

put the pieces together. the encoder maps $x$ to $(\mu, \log\sigma^2)$; you sample $z$ with the trick; the decoder maps $z$ to the parameters of $p_\theta(x \mid z)$; the loss is the negative ELBO, one sample per data point:

$$\mathcal{L}(x) = \underbrace{-\log p_\theta(x \mid z)}_{\text{reconstruction}} + \beta\,\underbrace{\operatorname{KL}\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{rate}}, \qquad z = \mu + \sigma\epsilon,$$

with $\beta = 1$ for the true ELBO. the two terms pull in opposite directions. reconstruction wants each $q(z \mid x)$ to be a sharp, distinctive code; the KL wants every $q(z \mid x)$ to look like the prior. the compromise is a code space where nearby codes decode to similar images (the noise in $z$ forces it) and all codes live in one region around the origin (the KL forces it), which is why you can interpolate and sample.

the decoder's output depends on the **likelihood**:

- **Bernoulli** (binary or binarized pixels): the decoder outputs logits $\ell$, and $-\log p(x \mid z) = \sum_d \operatorname{BCE}(\ell_d, x_d)$. this is an exact discrete likelihood for binary $x$. it's also widely used on grayscale values in $[0, 1]$ as a reconstruction loss, but then it is no longer a normalized likelihood, and bits/dim computed from it mean nothing.
- **Gaussian** with unit variance (real-valued data): the decoder outputs a mean $m$, and $-\log p(x \mid z) = \sum_d \tfrac12 (x_d - m_d)^2 + \tfrac12\log 2\pi$. include the constant, so that the loss really is a negative log-density.

**$\beta$-VAE** (Higgins et al., 2017) sets $\beta \ne 1$. $\beta > 1$ pushes harder towards the prior, trading reconstruction for a smoother, sometimes more disentangled code; $\beta < 1$ does the reverse. only $\beta = 1$ is a bound on the likelihood, so compare likelihoods at $\beta = 1$ (or with idea 3's estimate).

> **spec · stage 2.**
> - `reconstruction_nll(x_out, x, likelihood)`: $-\log p(x \mid z)$ per example `(B,)`, summed over every non-batch dimension: `"bernoulli"` (`x_out` are logits; `F.binary_cross_entropy_with_logits`, targets may be in $[0, 1]$) or `"gaussian"` (`x_out` are means, unit variance, including $\tfrac12\log 2\pi$ per dimension); `ValueError` otherwise.
> - `VAE(in_dim, latent_dim, *, hidden_dims=(512, 256), likelihood="bernoulli")`, an `nn.Module` with MLP encoder and decoder (internal design yours; `ValueError` for an unknown likelihood). `encode(x)` → `(mu, logvar)`, each `(B, latent_dim)`; `decode(z)` → logits or means `(B, in_dim)`; `forward(x)` → `(x_out, mu, logvar)` decoding a **sample** `z = reparameterize(mu, logvar)` (global RNG); `loss(x, *, beta=1.0)` → `(loss, recon, kl)` from **one** call to `forward`: batch means of the per-example `reconstruction_nll` and `kl_standard_normal`, `loss = recon + beta * kl`; `sample(n, *, generator=None)` decodes `z = torch.randn((n, latent_dim), generator=generator)` under `no_grad` and returns probabilities (`sigmoid`, Bernoulli) or means. this is `mlfs.generative.VAE`.

> **▶ build: stage 2.** implement `reconstruction_nll` and `VAE`.

> **🔮 predict:** you train a VAE with a 32-dimensional latent space on MNIST at $\beta = 1$. how many of the 32 coordinates end up carrying information about the input?

### idea 3: a tighter estimate of the likelihood

the ELBO is a bound, and its gap, $\operatorname{KL}(q(z \mid x) \| p(z \mid x))$, can be several nats. to compare a VAE fairly with models that compute exact likelihoods (topic 02's MADE and PixelCNN, topic 07's flows), you want something tighter. **importance sampling** gives it. since $p(x) = \mathbb{E}_{z \sim q(z \mid x)}\big[p(x \mid z)\,p(z) / q(z \mid x)\big]$, draw $k$ samples $z_1, \dots, z_k$ from $q$ and average the **importance weights** $w_j = p(x \mid z_j)\,p(z_j) / q(z_j \mid x)$:

$$\hat{\mathcal{L}}_k(x) = \log \frac{1}{k}\sum_{j=1}^{k} w_j.$$

by Jensen's inequality, $\mathbb{E}[\hat{\mathcal{L}}_k] \le \log p(x)$, so $-\hat{\mathcal{L}}_k$ is still an upper bound on the NLL. for $k = 1$ it is exactly the ELBO (in expectation), and Burda, Grosse & Salakhutdinov (2016) proved that it tightens monotonically as $k$ grows and converges to $\log p(x)$. the encoder only needs to put mass where the posterior is; the weights correct for the mismatch.

compute it in log space. $\log w_j = \log p(x \mid z_j) + \log p(z_j) - \log q(z_j \mid x)$, where each term is a sum over dimensions *with all its constants* (the $\tfrac12\log 2\pi$ of each Gaussian density, the reconstruction's constants), and then $\hat{\mathcal{L}}_k = \operatorname{logsumexp}_j(\log w_j) - \log k$. a missing constant doesn't cancel here, because the three densities have different dimensions.

> **spec · stage 3.** `iwae_nll(vae, x, *, k=64, generator=None, chunk_size=None)`: $-\hat{\mathcal{L}}_k(x)$ per example `(B,)`, for any model with `encode`, `decode` and a `likelihood` attribute (the `VAE` and stage 4's `ConvVAE`, whose log-densities sum over every latent dimension), with all of `eps = torch.randn((k, *mu.shape), generator=generator)` drawn first and $z = \mu + \sigma\epsilon$; `chunk_size` samples per example are decoded at a time (`None`: all $k$ at once), which saves memory and doesn't change the result; no gradients. as $k \to \infty$ it must converge to the exact NLL (the checks compute it by quadrature for a 2-D latent space).

> **▶ build: stage 3.** implement `iwae_nll`.

> **🔮 predict:** on binarized MNIST, how much tighter than the ELBO will the $k = 100$ importance-weighted bound be: a tenth of a nat, one nat, or five?

### idea 4: convolutional VAEs with spatial latents

an MLP VAE squeezes an image into one vector and has to relearn every pattern at every position. for images, a convolutional encoder and decoder are better, and a **spatial latent** is better still for some uses: instead of one vector, the code is a small image of shape $(C_z, H/4, W/4)$, each latent "pixel" describing a patch. this is what latent diffusion (chapter 22, Rombach et al., 2022) needs: a compressed image-like space where a U-Net can run 16 times cheaper than in pixel space.

`ConvVAE` uses two stride-2 convolutions in the encoder (each halving the resolution) and two stride-2 transposed convolutions in the decoder, and outputs $2C_z$ channels split into $\mu$ and $\log\sigma^2$. everything else (the loss, the KL summed over all latent positions, sampling from $\mathcal{N}(0, I)$ of the latent shape) is the same as for the MLP VAE. its default likelihood is Gaussian, and chapter 22 trains it with a small $\beta$, because a unit-variance Gaussian on pixel values gives a weak reconstruction signal next to the KL.

> **spec · stage 4.** `ConvVAE(in_channels=3, *, latent_channels=4, base_channels=64, likelihood="gaussian")`, an `nn.Module`: `encode(x)` for `(B, C, H, W)` with $H, W$ divisible by 4 (`ValueError` otherwise) → `(mu, logvar)` of shape `(B, latent_channels, H/4, W/4)`; `decode(z)` → `(B, C, 4h, 4w)` means (Gaussian) or logits (Bernoulli); `forward`, `loss(x, *, beta=1.0)` as for `VAE`; `sample(n, img_size, *, generator=None)` with `img_size` an int or `(H, W)`, decoding `torch.randn((n, latent_channels, H // 4, W // 4), generator=generator)`. this is `mlfs.generative.ConvVAE`.

> **▶ build: stage 4.** implement `ConvVAE`.

### idea 5: posterior collapse and active units

look at the ELBO per latent coordinate. coordinate $i$ costs $\operatorname{KL}(q(z_i \mid x) \| \mathcal{N}(0, 1))$ nats per example and pays only if it improves the reconstruction by more than that. a coordinate that isn't worth its price gets switched off completely: $q(z_i \mid x) = \mathcal{N}(0, 1)$ for every $x$, KL exactly 0, and the decoder learns to ignore it. that's **posterior collapse** at the level of single dimensions, and it's the norm, not an exception: a VAE with 32 latent dimensions on MNIST puts most of its information into a handful of coordinates, while the others drift towards the prior (fully, given enough training). with a powerful decoder (an autoregressive one, say), the whole latent space can collapse, because the decoder can model $x$ well without any code (Bowman et al., 2016; Chen et al., 2017).

two diagnostics. **active units** (Burda et al., 2016): coordinate $i$ is active if its encoder mean varies across inputs, $\operatorname{Var}_x\big(\mathbb{E}_{q}[z_i \mid x]\big) > 0.01$. a collapsed coordinate has the same mean (0) for every input. the count is a blunt instrument (a coordinate on its way to collapse can still pass the threshold), so look also at the **KL per dimension**, averaged over data: informative coordinates carry a nat or more each, collapsing ones a small fraction of a nat, and the profile shows exactly how the information is spread. larger $\beta$ raises the price of every nat, and fewer coordinates survive.

> **spec · stage 5.**
> - `active_units(vae, x, *, threshold=0.01)`: `(count, variances)` where `variances` is a float64 ndarray with one entry per latent coordinate (flattened, so spatial latents work), the population variance (`ddof = 0`) over the batch of the encoder mean, and `count` the number above `threshold`. uses only `vae.encode(x)`.
> - `kl_per_dim(vae, x)`: the batch mean of each coordinate's KL to $\mathcal{N}(0, 1)$, a float64 ndarray (its sum is the mean KL).

> **▶ build: stage 5.** implement `active_units` and `kl_per_dim`.

## worked example

**one example's ELBO by hand.** a 2-D latent, encoder output $\mu = (1, 0)$, $\log\sigma^2 = (0, \log 0.25)$. the KL: coordinate 1 gives $\tfrac12(1 + 1 - 1 - 0) = 0.5$; coordinate 2 gives $\tfrac12(0 + 0.25 - 1 - \log 0.25) = \tfrac12(-0.75 + 1.386) = 0.318$: total **0.818 nats**. with $\epsilon = (0.5, -2)$, the reparameterized sample is $z = (1 + 1 \cdot 0.5,\ 0 + 0.5 \cdot (-2)) = (1.5, -1)$. if the decoder's Bernoulli probabilities for a 3-pixel image $x = (1, 0, 1)$ are $(0.9, 0.2, 0.6)$, the reconstruction term is $-\ln 0.9 - \ln 0.8 - \ln 0.6 = 0.105 + 0.223 + 0.511 = 0.839$ nats, and the one-sample negative ELBO is $0.839 + 0.818 = 1.657$ nats ($\beta = 1$), or $0.839 + 4 \times 0.818 = 4.11$ at $\beta = 4$.

**gradients through the trick.** with $z = \mu + e^{\frac12 \log\sigma^2}\epsilon$: $\partial z/\partial\mu = 1$ and $\partial z / \partial \log\sigma^2 = \tfrac12 \sigma\epsilon$. at $\log\sigma^2 = \log 4$ and $\epsilon = 0.3$: $\tfrac12 \cdot 2 \cdot 0.3 = 0.3$.

**importance weights.** a model with $\log p(x \mid z_j) + \log p(z_j) - \log q(z_j \mid x) = (-90, -88, -95)$ for three samples: the ELBO estimate is their mean, $-91$; the $k = 3$ estimate is $\operatorname{logsumexp}(-90, -88, -95) - \ln 3 = -87.87 - 1.10 = -88.97$, tighter by 2 nats, dominated by the best sample. the checks reproduce the KL formula, the gradients, and the bound's convergence to the exact value.

## on paper

1. **the Gaussian KL.** derive $\operatorname{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \tfrac12(\mu^2 + \sigma^2 - 1 - \log\sigma^2)$.

<details><summary>solution</summary>

$\operatorname{KL} = \mathbb{E}_q[\log q(z) - \log p(z)]$ with $\log q(z) = -\tfrac12\log 2\pi\sigma^2 - (z - \mu)^2/(2\sigma^2)$ and $\log p(z) = -\tfrac12\log 2\pi - z^2/2$. under $q$, $\mathbb{E}[(z - \mu)^2] = \sigma^2$ and $\mathbb{E}[z^2] = \mu^2 + \sigma^2$. so $\operatorname{KL} = -\tfrac12\log\sigma^2 - \tfrac12 + \tfrac12(\mu^2 + \sigma^2)$, which is the formula. it's minimized (at 0) by $\mu = 0$, $\sigma^2 = 1$: the derivative in $\sigma^2$ is $\tfrac12(1 - 1/\sigma^2)$.

</details>

2. **the ELBO's gap.** show that $\log p(x) - \text{ELBO} = \operatorname{KL}(q(z \mid x) \| p(z \mid x))$, and explain why maximizing the ELBO over $\phi$ (the encoder) and over $\theta$ (the decoder) does two different things.

<details><summary>solution</summary>

$\log p(x) = \mathbb{E}_q[\log p(x)] = \mathbb{E}_q\big[\log \frac{p(x, z)}{p(z \mid x)}\big] = \mathbb{E}_q\big[\log\frac{p(x, z)}{q(z \mid x)}\big] + \mathbb{E}_q\big[\log\frac{q(z \mid x)}{p(z \mid x)}\big]$. the first term is the ELBO ($\log p(x,z) = \log p(x \mid z) + \log p(z)$ splits it into reconstruction minus the KL to the prior), the second is the KL to the posterior. $\log p(x)$ doesn't depend on $\phi$, so maximizing the ELBO over $\phi$ only shrinks the gap (better inference). over $\theta$, it raises the likelihood *and* can shrink the gap by making the true posterior easier to match, a bias towards models whose posteriors look Gaussian.

</details>

3. **importance weighting is a bound.** show $\mathbb{E}[\hat{\mathcal{L}}_k] \le \log p(x)$, and that $\hat{\mathcal{L}}_1$ averages to the ELBO.

<details><summary>solution</summary>

each $w_j$ has $\mathbb{E}_q[w_j] = \int q \cdot \frac{p(x \mid z)p(z)}{q} dz = p(x)$, so $\mathbb{E}[\frac1k\sum_j w_j] = p(x)$, and by Jensen ($\log$ is concave) $\mathbb{E}[\log \frac1k\sum w_j] \le \log \mathbb{E}[\frac1k\sum w_j] = \log p(x)$. for $k = 1$, $\mathbb{E}[\log w_1] = \mathbb{E}_q[\log p(x \mid z) + \log p(z) - \log q(z \mid x)]$, which is the ELBO.

</details>

4. **when does a coordinate switch off?** a toy model: coordinate $i$ with encoder mean $\mu_i(x)$ of spread $s$ over the data and a fixed posterior variance $\sigma^2$. its average KL is about $\tfrac12(s^2 + \sigma^2 - 1 - \log\sigma^2)$. at $\beta = 4$, how many nats of reconstruction must this coordinate save per example to be worth keeping at $s = 1$, $\sigma^2 = 0.1$?

<details><summary>solution</summary>

KL $\approx \tfrac12(1 + 0.1 - 1 + 2.303) = 1.20$ nats; at $\beta = 4$ the coordinate costs $4.8$ nats in the loss, so it must lower the reconstruction term by more than 4.8 nats per example on average, or the optimizer is better off collapsing it (KL 0). at $\beta = 1$ the bar is 1.2 nats. raising $\beta$ raises the bar for every coordinate, and the weakest ones fall first, which is exactly the active-unit curve in the lab.

</details>

## experiment

`lab.py` trains your VAE on binarized MNIST:

1. latent-space interpolation versus pixel-space interpolation.
2. a $\beta$ sweep: reconstruction, KL, active units, likelihood (with your `iwae_nll`) and FID.
3. a 2-D latent space: where the codes of real digits sit relative to the prior.

## check your understanding

1. why does a plain autoencoder interpolate badly, while a VAE interpolates well?

<details><summary>answer</summary>

nothing in an autoencoder's loss constrains the codes between training points, so the decoder can do anything there, and codes of different digits may sit far apart with garbage in between. a VAE's noise forces nearby codes to decode to similar images, and the KL packs all codes into one region around the origin, so the straight line between two codes passes through codes the decoder was trained on.

</details>

2. what would go wrong if the encoder output $\sigma$ directly (through a linear layer) instead of $\log\sigma^2$?

<details><summary>answer</summary>

$\sigma$ must be positive, and a linear output can be negative or zero; the KL contains $\log\sigma^2$, which explodes as $\sigma \to 0$. predicting $\log\sigma^2$ makes every real output valid and keeps the parameterization numerically stable (exponentiating only where needed).

</details>

3. why must you include the $\tfrac12\log 2\pi$ constants in `iwae_nll` but can drop them from a training loss?

<details><summary>answer</summary>

a constant doesn't change gradients, so training is unaffected. but `iwae_nll` reports a number, $-\log p(x)$, and the log-weight combines three densities of different dimensions ($x$, and twice $z$): dropping the constants changes the result by $\tfrac12 D \log 2\pi$ (for a Gaussian decoder) or by the latent constants, which don't cancel. you'd report a wrong likelihood.

</details>

4. a VAE's ELBO says 98 nats, its IWAE estimate with $k = 1000$ says 91, and a PixelCNN reports 82 (exact). what can you conclude?

<details><summary>answer</summary>

the VAE's true NLL is at most 91 (the IWAE estimate is still an upper bound), and the gap to the ELBO (7 nats) is the price of its Gaussian approximate posterior. even with the tighter estimate, it doesn't reach the PixelCNN; but you can't rule out that the true NLL is somewhat below 91. the comparison is fair only because all three are bounds or exact values on the same binarized data.

</details>

5. why is the Bernoulli loss on grayscale (non-binary) MNIST not a likelihood, even though it trains fine?

<details><summary>answer</summary>

a Bernoulli distribution assigns probability only to 0 and 1. for $x = 0.4$, $x\log p + (1 - x)\log(1 - p)$ is a cross-entropy, but it doesn't integrate (or sum) to 1 over the possible values of $x$, so the resulting "NLL" isn't the negative log of any probability. it's a fine reconstruction loss; it's not something to report in bits/dim (the continuous Bernoulli, Loaiza-Ganem & Cunningham 2019, fixes the normalization).

</details>

6. why do collapsed latent coordinates have KL exactly 0 rather than "small"?

<details><summary>answer</summary>

if a coordinate's information isn't worth its KL, the optimum for that coordinate is $q(z_i \mid x) = p(z_i)$ for every $x$, where its KL is 0 and the decoder ignores it. any partial use costs KL without enough reconstruction benefit, so gradient descent keeps pushing it towards the prior; after a finite budget you see small, still-shrinking KLs rather than exact zeros.

</details>

7. what does $\beta < 1$ do to samples from the prior, even as reconstructions improve?

<details><summary>answer</summary>

the codes of training images drift away from the prior (the KL is cheap), so the aggregate posterior covers the prior less well: large regions of $\mathcal{N}(0, I)$ decode to things the decoder never trained on. reconstructions get sharper while samples from the prior get worse, and the lab measures both.

</details>

## common bugs

- **`exp(logvar)` as the standard deviation.** symptom: the sample variance is the square of what it should be; the KL and the sampling disagree.
- **averaging the KL over latent dimensions** (or the reconstruction over pixels) instead of summing. symptom: the KL term is 32 times too weak (or the reconstruction 784 times too weak), and the balance of the ELBO is wrong; likelihoods are nonsense.
- **decoding $\mu$ in `forward`.** symptom: training works like an autoencoder, the noise never reaches the decoder, and samples from the prior look poor.
- **`loss` calling `forward` twice** (once for the reconstruction, once for the KL). symptom: the terms come from different samples of $z$; the checks can't match your loss to your own forward pass.
- **forgetting constants or the $-\log k$ in the IWAE estimate.** symptom: the estimate doesn't match the exact likelihood, or gets *worse* as $k$ grows.
- **sampling logits instead of probabilities.** symptom: Bernoulli "samples" with values outside $[0, 1]$.

## going deeper

- Kingma & Welling (2014), "Auto-encoding variational Bayes", *ICLR*. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Rezende, Mohamed & Wierstra (2014), "Stochastic backpropagation and approximate inference in deep generative models", *ICML*. [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)
- Burda, Grosse & Salakhutdinov (2016), "Importance weighted autoencoders", *ICLR*. the $k$-sample bound and active units. [arXiv:1509.00519](https://arxiv.org/abs/1509.00519)
- Higgins et al. (2017), "β-VAE: learning basic visual concepts with a constrained variational framework", *ICLR*. [openreview.net/forum?id=Sy2fzU9gl](https://openreview.net/forum?id=Sy2fzU9gl)
- Bowman et al. (2016), "Generating sentences from a continuous space", *CoNLL*. posterior collapse with strong decoders. [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)
- Kingma & Welling (2019), "An introduction to variational autoencoders", *Foundations and Trends in Machine Learning*. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)
- Rombach et al. (2022), "High-resolution image synthesis with latent diffusion models", *CVPR*. why spatial latents. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

## connections

- ← builds on: variational inference and the reparameterization trick (ch 12 · topic 07), EM and probabilistic PCA (ch 11), Gaussian KLs (ch 05), bits/dim and likelihood bounds (topic 01), convolutions and transposed convolutions (ch 16).
- → used in: discrete latents and VQ-VAE (topic 04), the chapter capstone (the latent-variable family, graded by IWAE likelihood and FID), latent diffusion on your `ConvVAE` (ch 22 · topic 07), and representation learning (ch 25).
