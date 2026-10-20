# GANs

## why this matters

every model so far in this chapter is trained by maximum likelihood (or a bound on it), and maximum likelihood has a habit: it would rather put a little probability everywhere than miss any data point, because missing one costs $-\log 0 = \infty$. on images, that habit shows up as blur: a VAE hedges between all plausible digits and draws the average. what if you dropped likelihood entirely and asked only for samples that are *indistinguishable from real data*?

that's the idea of the **generative adversarial network** (Goodfellow et al., 2014). a **generator** $G$ maps noise $z \sim \mathcal{N}(0, I)$ to samples; a **discriminator** $D$ is a classifier trained to tell real data from generated samples; and the generator is trained to fool the discriminator. no density, no likelihood, no encoder: just a learned critic that keeps finding what's wrong with the samples. from 2014 to about 2021, GANs produced the sharpest images in machine learning (StyleGAN's faces are the famous example). they're also notoriously hard to train: the two networks play a game rather than minimize a loss, and the game can cycle, stall, or **collapse** onto a few modes of the data while ignoring the rest. in this topic you'll build the losses, one step of the game, the DCGAN architecture, and the metric that makes mode collapse visible; topic 06 stabilizes it.

## warm-up: predict first

1. the discriminator is a binary classifier. what loss would you train it with, and what is its best possible output for a point where real and generated data are equally dense?
2. the generator wants $D(G(z))$ to be high. early in training, $D(G(z)) \approx 0.001$. which gives a larger gradient: minimizing $\log(1 - D(G(z)))$ or maximizing $\log D(G(z))$?
3. a GAN is trained on a mixture of 8 gaussians. after many steps, will it cover all 8?
4. why can't you compute a GAN's test likelihood?
5. in one training step, should the discriminator see the generator's samples with or without gradients flowing back into the generator?

## prerequisites check

- [ ] logistic regression, binary cross-entropy and logits (→ ch 08 · topic 04)
- [ ] KL divergence, and the Jensen–Shannon divergence as a symmetrized KL (→ ch 05 · topic 05)
- [ ] `detach()`, optimizers and `zero_grad` (→ ch 15 · topic 07); `mlfs.torchnn.MLP` (→ ch 15 · topic 07)
- [ ] transposed convolutions and batch norm (→ ch 16 · topic 06, ch 15 · topic 02)
- [ ] precision and recall for generative models (→ topic 01)

## core ideas

### idea 1: the minimax game, and the loss that actually trains

the discriminator outputs a logit $\ell = D(x)$ and the probability $\sigma(\ell)$ that $x$ is real. it is trained with ordinary binary cross-entropy: real data labelled 1, generated data labelled 0,

$$\mathcal{L}_D = -\mathbb{E}_{x \sim p_\text{data}}\big[\log \sigma(D(x))\big] - \mathbb{E}_{z}\big[\log\big(1 - \sigma(D(G(z)))\big)\big].$$

Goodfellow et al. framed the whole thing as a minimax game, $\min_G \max_D V(D, G)$ with $V = -\mathcal{L}_D$. for a fixed generator, the best discriminator is (on paper 1)

$$D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_g(x)},$$

and plugging it in shows that the generator then minimizes $2\,\operatorname{JS}(p_\text{data} \,\|\, p_g) - \log 4$: the Jensen–Shannon divergence between the data and the model, minimized exactly when $p_g = p_\text{data}$, where $D^* = 1/2$ everywhere.

the generator's half of the minimax game is to *minimize* $\mathbb{E}_z[\log(1 - \sigma(D(G(z))))]$, the **saturating** loss. it has a fatal flaw. early in training, the generator's samples are bad, the discriminator rejects them confidently, $\sigma(\ell) \to 0$, and the gradient of $\log(1 - \sigma(\ell))$ with respect to $\ell$ is $-\sigma(\ell) \to 0$ (on paper 2): the generator learns nothing exactly when it most needs to. the paper's fix, now universal, is the **non-saturating** loss: *maximize* $\log \sigma(D(G(z)))$, i.e. minimize $-\log\sigma(\ell)$, whose gradient $-(1 - \sigma(\ell))$ is close to $-1$ when the generator is losing. both losses have the same fixed point; the non-saturating one has useful gradients along the way.

in code, never compute $\log\sigma(\ell)$ directly (it underflows for $\ell \ll 0$). use the identities $-\log\sigma(\ell) = \operatorname{softplus}(-\ell)$ and $-\log(1 - \sigma(\ell)) = \operatorname{softplus}(\ell)$, which `F.binary_cross_entropy_with_logits` uses internally.

> **spec · stage 1.**
> - `discriminator_loss(real_logits, fake_logits)`: the mean BCE-with-logits of `real_logits` against 1 plus the mean of `fake_logits` against 0 (each a mean over its own batch), a scalar, finite for logits of $\pm 80$. this is `mlfs.generative.discriminator_loss`.
> - `generator_loss(fake_logits, *, non_saturating=True)`: `non_saturating=True`: mean $-\log\sigma(\ell)$; `False`: mean $\log(1 - \sigma(\ell))$ (the minimax loss, which the generator minimizes). stable for large logits. this is `mlfs.generative.generator_loss`.

> **▶ build: stage 1.** implement `discriminator_loss` and `generator_loss` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** after a 50-step head start, the discriminator rejects the generator's samples with $D(G(z)) \approx 0.01$. how many times smaller is the generator's gradient under the saturating loss than under the non-saturating one, and will the saturating GAN still learn?

### idea 2: one step of the game

GANs are trained by alternating gradient steps: one (or a few) for the discriminator, then one for the generator. one step of the simplest version:

1. draw a batch of real data $x$ and two batches of noise $z_1, z_2$.
2. **discriminator update:** compute $\mathcal{L}_D$ on $D(x)$ and $D(G(z_1))$, with $G(z_1)$ **detached**: the discriminator's loss should update only the discriminator, and detaching saves the backward pass through $G$. step the discriminator's optimizer.
3. **generator update:** compute the generator loss on $D(G(z_2))$, with fresh noise and the *just-updated* discriminator, and backpropagate through $D$ into $G$. step only the generator's optimizer. $D$'s parameters receive gradients too, but they're never stepped by this optimizer and are zeroed before the next discriminator update.

notice what's not here: a single objective that decreases. $\mathcal{L}_D$ and the generator loss pull against each other, and neither curve tells you whether the samples are improving. a falling discriminator loss can mean a better discriminator or a worse generator. you have to look at samples, or measure them (topic 01).

> **spec · stage 2.** `gan_step(G, D, opt_g, opt_d, real, *, latent_dim, non_saturating=True, generator=None)`: draw `z1` then `z2`, each `torch.randn((B, latent_dim), generator=generator)` with `B = real.shape[0]`; update `D` once on `discriminator_loss(D(real), D(G(z1).detach()))`; then update `G` once on `generator_loss(D(G(z2)), non_saturating=non_saturating)`; `D` may return logits of shape `(B,)` or `(B, 1)`. returns `(d_loss, g_loss)` as python floats.

> **▶ build: stage 2.** implement `gan_step`.

### idea 3: DCGAN

early GANs used MLPs and produced blurry, unstable results on images. **DCGAN** (Radford, Metz & Chintala, 2016) found a recipe of architectural choices that made convolutional GANs train reliably, and much of it is still standard:

- **generator:** project $z$ with a linear layer to a small spatial map (here $4 \cdot$`base` channels at $1/4$ of the image size), then upsample with **stride-2 transposed convolutions**, each followed by batch norm and ReLU, and finish with a convolution and **tanh**, so images live in $[-1, 1]$ (scale your data to match).
- **discriminator:** stride-2 convolutions instead of pooling, **LeakyReLU(0.2)** activations (a plain ReLU zeroes the gradient for half the inputs, and the generator learns only through $D$'s gradients), batch norm everywhere except the first layer, and a linear layer to one logit.
- **initialization** of all convolution and linear weights from $\mathcal{N}(0, 0.02)$, batch-norm scales from $\mathcal{N}(1, 0.02)$; Adam with learning rate $2 \times 10^{-4}$ and $\beta_1 = 0.5$ (a smaller momentum, because the target keeps moving).

two stride-2 layers in each network mean the image size must be divisible by 4: $28 \times 28$ MNIST goes $7 \to 14 \to 28$, CIFAR's $32 \times 32$ goes $8 \to 16 \to 32$. the notebook gives you the initialization helper `dcgan_init`.

> **spec · stage 3.**
> - `DCGANGenerator(latent_dim=100, out_channels=3, *, base_channels=64, img_size=32)`: `z` `(B, latent_dim)` → images `(B, C, img_size, img_size)` in $[-1, 1]$ (tanh), upsampling with stride-2 transposed convolutions and batch norm; `ValueError` unless `img_size % 4 == 0`; DCGAN initialization. this is `mlfs.generative.DCGANGenerator`.
> - `DCGANDiscriminator(in_channels=3, *, base_channels=64, img_size=32)`: images → logits `(B,)`, downsampling with stride-2 convolutions, every activation a `LeakyReLU(0.2)`; `ValueError` unless `img_size % 4 == 0`; DCGAN initialization. this is `mlfs.generative.DCGANDiscriminator`.

> **▶ build: stage 3.** implement `DCGANGenerator` and `DCGANDiscriminator`.

### idea 4: mode collapse, and how to see it

suppose the generator finds one kind of sample the current discriminator can't reject, say, perfect 1s. its loss is happy: every sample fools $D$. nothing in one gradient step asks it to also produce 7s. the discriminator then learns that 1s are now suspicious (most 1s it sees are fake), the generator moves to the next blind spot, and the pair can chase each other from mode to mode without ever covering the distribution. that's **mode collapse** (Goodfellow, 2016), and it's the GAN's signature failure: the samples look great and the diversity is gone. it happens because the alternating gradient game isn't the minimax problem it's derived from: the theory assumes the discriminator is optimal at every generator step, and in practice it's a few gradient steps behind.

on real data, collapse is hard to see by eye, which is why topic 01 built recall. on a **toy distribution with known modes** it can be counted exactly. the classic is 8 gaussians on a circle: a generated sample is **high quality** if it lies within 3 standard deviations of the nearest mode, and a mode is **covered** if enough of the samples land on it with high quality (Srivastava et al., 2017). a healthy generator covers all 8 with most samples of high quality; a collapsed one covers 1 or 2 with high quality; a blurry one covers everything with low quality.

> **spec · stage 4.** `mode_coverage(samples, centers, std, *, n_std=3.0, min_frac=0.01)`: NumPy `samples` `(n, d)` and `centers` `(K, d)` → `(n_covered, frac_high_quality, counts)`: a sample is high quality if its distance to its **nearest** centre is $\le$ `n_std * std`; `counts` (int ndarray `(K,)`) counts the high-quality samples per centre; a centre is covered if `counts >= min_frac * n` (with $n$ all samples); `frac_high_quality` is a float, `n_covered` an int.

> **▶ build: stage 4.** implement `mode_coverage`.

> **🔮 predict:** a GAN trained on 8 gaussians for 5,000 steps: will the number of modes it covers go up steadily and end at 8? or will it jump around?

## worked example

**losses by hand.** a discriminator outputs logits $(2, -1)$ on two real images and $(-3, 0)$ on two fakes. $\sigma(2) = 0.881$, $\sigma(-1) = 0.269$: the real term is $-\tfrac12(\ln 0.881 + \ln 0.269) = \tfrac12(0.127 + 1.313) = 0.720$. $\sigma(-3) = 0.047$, $\sigma(0) = 0.5$: the fake term is $-\tfrac12(\ln 0.953 + \ln 0.5) = \tfrac12(0.049 + 0.693) = 0.371$. $\mathcal{L}_D = 1.091$. a discriminator that says $1/2$ everywhere scores $2\ln 2 = 1.386$, the value at the game's equilibrium.

**the two generator losses.** on the fake logits $(-3, 0)$: non-saturating $\tfrac12(\operatorname{softplus}(3) + \operatorname{softplus}(0)) = \tfrac12(3.049 + 0.693) = 1.871$; saturating $\tfrac12(\ln(1 - 0.047) + \ln 0.5) = -0.371$. their gradients with respect to the logit $-3$: non-saturating $-(1 - \sigma(-3))/2 = -0.476$, saturating $-\sigma(-3)/2 = -0.024$, 20 times smaller. at a logit of $-6$, the ratio is 400.

**mode coverage.** centres $(0, 0)$, $(10, 0)$, $(0, 10)$ with std $0.1$ (radius $0.3$); six samples $(0.1, 0)$, $(0.2, 0.1)$, $(9.9, 0)$, $(5, 5)$, $(0, 0.29)$, $(0, 10.31)$. the high-quality ones are the first three and the fifth (the sixth is $0.31$ from its centre): counts $(3, 1, 0)$, a high-quality fraction of $4/6$. with `min_frac = 0.3` a centre needs $1.8$ samples: only centre 0 is covered. the checks reproduce these numbers.

## on paper

1. **the optimal discriminator.** for a fixed generator with density $p_g$, show that $D^*(x) = p_\text{data}(x) / (p_\text{data}(x) + p_g(x))$ maximizes $V(D) = \mathbb{E}_{p_\text{data}}[\log D] + \mathbb{E}_{p_g}[\log(1 - D)]$, and that $V(D^*) = 2\operatorname{JS}(p_\text{data} \| p_g) - \log 4$.

<details><summary>solution</summary>

$V = \int p_\text{data}(x)\log D(x) + p_g(x)\log(1 - D(x))\,dx$. pointwise, $a\log y + b\log(1 - y)$ is maximized at $y = a/(a + b)$ (set the derivative $a/y - b/(1-y)$ to 0). plugging in, with $m = (p_\text{data} + p_g)/2$: $V(D^*) = \mathbb{E}_{p_\text{data}}[\log \frac{p_\text{data}}{2m}] + \mathbb{E}_{p_g}[\log\frac{p_g}{2m}] = \operatorname{KL}(p_\text{data} \| m) + \operatorname{KL}(p_g \| m) - 2\log 2 = 2\operatorname{JS} - \log 4$. it's minimized over $p_g$ at $p_g = p_\text{data}$ (JS = 0), where $D^* = 1/2$.

</details>

2. **saturation.** with $D = \sigma(\ell)$, compute $\frac{d}{d\ell}\log(1 - \sigma(\ell))$ and $\frac{d}{d\ell}[-\log\sigma(\ell)]$, and their values when $D(G(z)) = 0.001$.

<details><summary>solution</summary>

$\frac{d}{d\ell}\log(1 - \sigma(\ell)) = -\sigma(\ell)$ and $\frac{d}{d\ell}(-\log\sigma(\ell)) = -(1 - \sigma(\ell))$. at $\sigma = 0.001$: $-0.001$ versus $-0.999$, a factor of 999. both push the logit up (the generator minimizes either loss), but the minimax version barely pushes when the generator is losing badly; the non-saturating version pushes hardest there.

</details>

3. **JS can't see far.** two distributions with disjoint supports, $p_\text{data} = \delta_0$ and $p_g = \delta_\theta$ on the real line. compute $\operatorname{JS}(p_\text{data} \| p_g)$ as a function of $\theta$ and explain what it means for the generator's gradient.

<details><summary>solution</summary>

for $\theta \ne 0$ the supports are disjoint, $m$ puts mass $\tfrac12$ on each point, and $\operatorname{KL}(\delta_0 \| m) = \log 2$, likewise for $\delta_\theta$: $\operatorname{JS} = \log 2$ for every $\theta \ne 0$, and 0 at $\theta = 0$. it's constant, so its gradient with respect to $\theta$ is zero: an optimal discriminator gives the generator no information about *which direction* to move. real image distributions live on thin manifolds that rarely overlap at first, which is exactly the regime where this happens; topic 06's Wasserstein distance fixes it.

</details>

4. **why alternating steps can cycle.** consider the bilinear game $\min_x \max_y xy$ with simultaneous gradient steps of size $\eta$: $x \leftarrow x - \eta y$, $y \leftarrow y + \eta x$. show that $x^2 + y^2$ grows every step.

<details><summary>solution</summary>

$x'^2 + y'^2 = (x - \eta y)^2 + (y + \eta x)^2 = (1 + \eta^2)(x^2 + y^2)$. the iterates spiral outward around the equilibrium $(0, 0)$ instead of converging: gradient descent on a game is not gradient descent on a loss. alternating updates spiral less but still orbit. mode hopping in GANs is a high-dimensional cousin of this rotation.

</details>

## experiment

`lab.py` trains GANs with your code:

1. an MLP GAN on 8 gaussians, three seeds: modes covered during training.
2. the same with the saturating loss: gradients at the start, and where training ends up.
3. your DCGAN on MNIST: samples, FID, and which digits it draws.

## check your understanding

1. why is the discriminator's loss not a useful measure of sample quality?

<details><summary>answer</summary>

it measures how well *the current* discriminator separates *the current* samples from data. a low value can mean a strong discriminator or a weak generator; a value near $2\ln 2$ can mean a perfect generator or a discriminator that has stopped learning. the two networks move together, so neither loss is anchored to the data distribution.

</details>

2. why detach $G(z_1)$ in the discriminator update?

<details><summary>answer</summary>

the discriminator's loss should change only the discriminator. without the detach, the backward pass also computes gradients for the generator's parameters (wasted work), and if the generator's gradients aren't zeroed before its own update, they'd push the generator in the *wrong* direction (helping the discriminator).

</details>

3. why does the generator step use fresh noise $z_2$ and the updated discriminator?

<details><summary>answer</summary>

fresh noise gives an independent estimate of the generator's loss (reusing $z_1$ would correlate the two updates); the updated discriminator is the best current critic, so the generator responds to the latest information. both are conventions with small effects, but fixing them makes runs reproducible and comparable.

</details>

4. a GAN's samples have precision 0.95 and recall 0.2 on MNIST. which failure is that, and why do GANs suffer it more than VAEs?

<details><summary>answer</summary>

mode collapse: excellent samples from a small part of the distribution. a VAE's likelihood term punishes missing any training image (each must be reconstructed), so it covers everything and blurs. a GAN's generator is only punished for samples the discriminator rejects; it's never punished directly for the modes it doesn't produce.

</details>

5. why tanh at the generator's output and LeakyReLU in the discriminator?

<details><summary>answer</summary>

tanh bounds the outputs to $[-1, 1]$, matching data scaled to that range, so the generator can't win by producing out-of-range values. the generator's only learning signal is the gradient of $D$ with respect to its input; a ReLU discriminator passes no gradient where its units are off, so LeakyReLU keeps a small gradient everywhere.

</details>

6. how can you evaluate a model whose likelihood you can't compute?

<details><summary>answer</summary>

with sample-based metrics: FID and precision/recall in a feature space (topic 01), mode counts on toy data where the modes are known, a classifier's label distribution on the samples, and human inspection. none is a likelihood, and each has blind spots, which is why the capstone reports several.

</details>

## common bugs

- **`torch.log(torch.sigmoid(l))`.** symptom: `-inf` and `nan` losses as soon as the discriminator becomes confident.
- **forgetting `.detach()` and `zero_grad()`.** symptom: the generator's update includes the discriminator loss's gradient, which pushes it the wrong way; training stalls or diverges.
- **one optimizer for both networks.** symptom: each update moves both networks; the discriminator step also trains the generator to help the discriminator.
- **images in $[0, 1]$ with a tanh generator.** symptom: the generator can't produce the data's black (it can, but only half its range is used) and the discriminator spots fakes by their value range.
- **judging training by the losses.** symptom: a "converged" loss curve with collapsed samples; always look at samples and a coverage metric.
- **the saturating loss.** symptom: the generator barely moves at the start; samples stay noise for a long time.

## going deeper

- Goodfellow et al. (2014), "Generative adversarial nets", *NeurIPS*. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Goodfellow (2016), "NIPS 2016 tutorial: generative adversarial networks". mode collapse and the non-saturating loss. [arXiv:1701.00160](https://arxiv.org/abs/1701.00160)
- Radford, Metz & Chintala (2016), "Unsupervised representation learning with deep convolutional generative adversarial networks", *ICLR*. DCGAN. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Metz, Poole, Pfau & Sohl-Dickstein (2017), "Unrolled generative adversarial networks", *ICLR*. mode collapse on 8 gaussians. [arXiv:1611.02163](https://arxiv.org/abs/1611.02163)
- Srivastava, Valkov, Russell, Gutmann & Sutton (2017), "VEEGAN: reducing mode collapse in GANs using implicit variational learning", *NeurIPS*. mode and quality counts. [arXiv:1705.07761](https://arxiv.org/abs/1705.07761)
- Karras, Laine & Aila (2019), "A style-based generator architecture for generative adversarial networks", *CVPR*. StyleGAN. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)

## connections

- ← builds on: logistic regression and cross-entropy (ch 08), KL and JS divergences (ch 05), training loops and `MLP` (ch 15), transposed convolutions and batch norm (ch 15–16), sample-based metrics (topic 01).
- → used in: Wasserstein GANs, gradient penalties and spectral normalization (topic 06), the chapter capstone (the adversarial family), adversarial losses inside other models (VQGAN, neural codecs in the audio elective), and adversarial examples, a different use of the same gradients (ch 28).
