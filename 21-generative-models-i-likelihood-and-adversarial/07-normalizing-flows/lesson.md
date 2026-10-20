# normalizing flows

## why this matters

each family so far gives up something. autoregressive models (topic 02) have exact likelihoods but sample one pixel at a time. VAEs (topic 03) sample in one pass but only bound the likelihood, and need an encoder that approximates the posterior. GANs (topics 05–06) sample in one pass and have no likelihood at all. is there a model with an exact likelihood, one-pass sampling, *and* an exact encoder?

a **normalizing flow** is one. it's an invertible network $f$ that maps data $x$ to a latent $z = f(x)$ of the same dimension, with $z$ standard normal. sampling runs it backwards, $x = f^{-1}(z)$; encoding runs it forwards; and the likelihood follows from the change-of-variables formula you derived in chapter 3:

$$\log p_X(x) = \log p_Z\big(f(x)\big) + \log\Big|\det \frac{\partial f}{\partial x}\Big|.$$

the catch is the determinant. for a generic $D \times D$ Jacobian it costs $O(D^3)$ (a $784$-dimensional MNIST image: hundreds of millions of operations per example per step), and generic networks aren't invertible anyway. flows are the art of designing layers that are invertible by construction, with Jacobians whose determinant is cheap. the **affine coupling layer** of RealNVP (Dinh, Sohl-Dickstein & Bengio, 2017) does it with a trick so simple it seems like cheating: leave half the coordinates unchanged, and transform the other half elementwise, with scales and shifts computed from the unchanged half. its Jacobian is triangular, and its log-determinant is a sum. you'll build it, stack it into RealNVP, train it by maximum likelihood, and learn the preprocessing that turns its density on 8-bit images into bits/dim you can compare with every other model in the chapter.

## warm-up: predict first

1. you scale a 1-D density by mapping $z \mapsto x = 2z$. what happens to the density at each point?
2. a layer leaves $x_1$ unchanged and sets $y_2 = x_2 \cdot e^{s(x_1)} + t(x_1)$ for arbitrary networks $s$, $t$. is it invertible? what is its Jacobian determinant?
3. why doesn't it matter that $s$ and $t$ are not invertible themselves?
4. a flow is trained on 8 separate gaussian blobs. can it put zero density between them?
5. a flow models pixels in $[0, 1]$. why might it need a transform to $(-\infty, \infty)$ first, and what does that do to the likelihood?

## prerequisites check

- [ ] change of variables for densities, Jacobian determinants (→ ch 03 · topic 06)
- [ ] determinants of triangular matrices; `slogdet` (→ ch 02 · topic 03)
- [ ] maximum likelihood (→ ch 05 · topic 02); Adam and training loops (→ ch 15)
- [ ] dequantization and bits per dimension (→ topic 01); MADE's autoregressive masks (→ topic 02)
- [ ] mode coverage on 8 gaussians (→ topic 05)

## core ideas

### idea 1: the affine coupling layer

pick a binary **mask** $m \in \{0, 1\}^D$. the coordinates with $m_i = 1$ pass through unchanged; the others are scaled and shifted by amounts that depend only on the unchanged ones:

$$y = x_m + (1 - m) \odot \big(x \odot e^{s(x_m)} + t(x_m)\big), \qquad x_m = m \odot x,$$

with $s, t: \mathbb{R}^D \to \mathbb{R}^D$ arbitrary neural networks (they see $x_m$, whose masked-out entries are zero). three facts make it a flow layer:

- **invertible, in closed form.** $y_m = x_m$, so from $y$ you can recompute $s(x_m) = s(y_m)$ and $t(y_m)$, then undo the affine map: $x = y_m + (1 - m) \odot (y - t(y_m)) \odot e^{-s(y_m)}$. $s$ and $t$ themselves are never inverted, so they can be as complicated as you like.
- **a triangular Jacobian.** order the coordinates with the unchanged ones first: $\partial y / \partial x = \begin{pmatrix} I & 0 \\ \ast & \operatorname{diag}(e^{s}) \end{pmatrix}$. the $\ast$ block (how the transformed half depends on the conditioning half, through $s$ and $t$) is complicated, but a triangular matrix's determinant is the product of its diagonal (on paper 1).
- **a log-determinant that's a sum.** $\log|\det| = \sum_{i: m_i = 0} s_i(x_m)$, computed for free during the forward pass. the inverse's log-determinant is its negative.

in practice $s$ is bounded, e.g. $s = c \odot \tanh(\tilde s)$ with a learned scale $c$, so $e^{s}$ can't explode early in training, and the last layer of the $s, t$ network is often initialized to zero so every coupling starts as the identity.

> **spec · stage 1.**
> - `standard_normal_log_prob(z)`: $\sum_i \log\mathcal{N}(z_i; 0, 1)$ over every non-batch dimension, `(B,)`.
> - `AffineCoupling(dim, mask, *, hidden_dim=128)`, an `nn.Module` with a buffer `mask` of shape `(dim,)` (0/1 floats) and networks computing $s$ and $t$ from $x_m$ (internal design yours). `forward(x)`: `(B, dim)` → `(y, log_det)` with $y$ as above and `log_det` `(B,)` $= \sum_{i: m_i = 0} s_i(x_m)$; `inverse(y)` → `(x, log_det_inverse)`, exact, with `log_det_inverse` $= -\sum_{i: m_i = 0} s_i(y_m)$. this is `mlfs.generative.AffineCoupling`.

> **▶ build: stage 1.** implement `standard_normal_log_prob` and `AffineCoupling` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: RealNVP

one coupling layer leaves half the coordinates untouched, so stack several and **alternate the masks**: layer $k$ uses $m_i = (i + k) \bmod 2$, so the coordinates transformed by one layer condition the next. a composition of invertible maps is invertible (apply the inverses in reverse order), and the log-determinants of a composition add up:

$$z = f_K \circ \dots \circ f_1(x), \qquad \log\Big|\det\frac{\partial z}{\partial x}\Big| = \sum_{k=1}^{K} \log\Big|\det\frac{\partial f_k}{\partial h_{k-1}}\Big|.$$

with a standard normal base distribution, the exact log-likelihood is `standard_normal_log_prob(z) + log_det`, and sampling is `inverse(randn)`. both cost one pass through the network, the property that autoregressive models lack. (the RealNVP paper also uses checkerboard and channel masks for images, multi-scale architectures, and batch norm inside the flow; the masks here are the 1-D version of the checkerboard.)

an aside on the cousins: a coupling layer transforms half the coordinates given the other half; a **masked autoregressive flow** (MAF, Papamakarios et al., 2017) transforms coordinate $i$ given *all* coordinates before it, using a MADE (topic 02) to compute every $s_i, t_i$ in one pass. its log-determinant is also a sum, its density evaluation is one pass, but its inverse (sampling) is sequential, $D$ passes. coupling layers trade some flexibility per layer for one-pass sampling.

> **spec · stage 2.** `RealNVP(dim, *, n_layers=8, hidden_dim=128)`, an `nn.Module` with `self.layers` an `nn.ModuleList` of `n_layers` `AffineCoupling` layers, layer $k$ with mask $m_i = (i + k) \bmod 2$. `forward(x)` → `(z, log_det)` (data to latent, `log_det` `(B,)` the sum of the layers'); `inverse(z)` → `x` (the layers' inverses in reverse order); `log_prob(x)` → `(B,)` exact log-density under a standard normal base; `sample(n, *, generator=None)` → `inverse(torch.randn((n, dim), generator=generator))` without gradients, shape `(n, dim)`. this is `mlfs.generative.RealNVP`.

> **▶ build: stage 2.** implement `RealNVP`.

> **🔮 predict:** the 8-gaussian mixture has a known density, so its true test NLL is known (about $-1.08$ nats). how close will an 8-layer RealNVP get: within 0.01 nats, 0.1, or 1?

### idea 3: maximum likelihood, and what it optimizes

with an exact log-likelihood, training is just minimizing the average negative log-likelihood of the data, $-\frac1N\sum_n \log p_\theta(x_n)$, by stochastic gradient descent. no adversary, no bound, no encoder: the most straightforward objective of any model in this chapter, and a loss curve that means exactly what it says.

what does it optimize? minimizing the expected NLL minimizes $\operatorname{KL}(p_\text{data} \| p_\theta)$, the "forward" KL. it heavily punishes $p_\theta(x) \approx 0$ where data exist (the log goes to $-\infty$), and only mildly punishes mass where there is no data. so likelihood-trained models are **mode-covering**: they'd rather spread mass over gaps between modes than miss a mode, the opposite failure from topic 05's mode-collapsing GANs. a flow adds a second reason to leak mass into gaps: it's a continuous bijection of a connected Gaussian blob, so it can't tear the blob into disconnected islands; it can only stretch thin "bridges" between them (on paper 3). the lab measures how thin.

> **spec · stage 3.** `fit_flow(flow, X, *, n_steps=2000, batch_size=256, lr=1e-3, seed=0)`: `torch.optim.Adam(lr)` on the mean of `-flow.log_prob(batch)`, each batch `X[torch.randint(len(X), (batch_size,), generator=g)]` with `g = torch.Generator().manual_seed(seed)`; `X` an array or tensor `(N, dim)`; returns the list of `n_steps` losses (python floats).

> **▶ build: stage 3.** implement `fit_flow`.

> **🔮 predict:** a RealNVP is trained on 8 well-separated gaussian blobs. what fraction of its samples will land *between* the blobs?

### idea 4: flows on images: dequantization and the logit transform

two pieces of plumbing turn a flow on images into an honest likelihood model, and both change the log-likelihood in ways you must account for.

**dequantization** (topic 01): pixel values are integers, so add uniform noise and scale to $[0, 1)$, $u = (x + \epsilon)/L$, and use the bound $-\log P(x) \le -\log p(u) + D\log L$.

**the logit transform.** a flow's base is a Gaussian on all of $\mathbb{R}^D$, but $u$ lives in a bounded box; near the box's edges the flow would need infinitely steep maps. so map $u$ to an unbounded space first, keeping away from 0 and 1 with a small $\alpha$:

$$p = \alpha + (1 - 2\alpha)\,u, \qquad y = \operatorname{logit}(p) = \log p - \log(1 - p).$$

this is itself an invertible elementwise map, so it's just one more flow layer with its own log-determinant, $\sum_i \log\frac{1 - 2\alpha}{p_i (1 - p_i)}$ (on paper 4). the flow models $y$; the density of $u$ is $\log p(u) = \log p_\text{flow}(y) + \log|\det \partial y/\partial u|$; then the dequantization bound and bits/dim. forget the transform's log-determinant and you'll report a likelihood for the wrong variable, typically off by several bits per dimension.

> **spec · stage 4.**
> - `logit_transform(x, *, alpha=0.05)`: `x` in $[0, 1)$, any shape `(B, ...)` → `(y, log_det)` with $y = \operatorname{logit}(\alpha + (1 - 2\alpha)x)$ and `log_det` `(B,)` the sum over non-batch dims of $\log\frac{1 - 2\alpha}{p(1 - p)}$.
> - `flow_bits_per_dim(flow, x_levels, *, n_levels=256, alpha=0.05, generator=None)`: per-example bits/dim `(B,)` of integer images `x_levels` `(B, ...)`: `dequantize` (topic 01, with `n_levels` and `generator`) → `logit_transform` → `flow.log_prob(y)` with `y` in the data's own shape (flatten images yourself for a flow on vectors, keep them as images for an image flow), plus the transform's `log_det` → `discrete_nll` with `n_levels` → `bits_per_dim` over $D$ = the number of dimensions per example; no gradients.

> **▶ build: stage 4.** implement `logit_transform` and `flow_bits_per_dim`.

## worked example

**a coupling layer by hand.** $D = 2$, mask $m = (1, 0)$, input $x = (0.5, 2)$. the networks output $s(x_m) = (\cdot, \ln 3)$ and $t(x_m) = (\cdot, -1)$ (only the second entries matter). then $y_1 = 0.5$, $y_2 = 2 \cdot 3 - 1 = 5$, and $\log|\det J| = \ln 3 = 1.099$. inverse: $y_m = (0.5, 0)$ gives the same $s$ and $t$, and $x_2 = (5 - (-1)) / 3 = 2$. ✓.

**the log-likelihood.** if this were the whole flow, $z = (0.5, 5)$ and $\log p(x) = \log\mathcal{N}(0.5) + \log\mathcal{N}(5) + \ln 3 = (-1.044) + (-13.419) + 1.099 = -13.364$ nats: the stretch by a factor 3 *raises* the density of $x$ relative to its latent (the density at $x$ is the latent density times the factor by which $f$ expands volume around $x$).

**the logit transform.** $\alpha = 0.05$, a pixel with $u = 0.5$: $p = 0.5$, $y = 0$, and its log-derivative is $\ln\frac{0.9}{0.25} = 1.281$; a nearly black pixel $u = 0.002$: $p = 0.0518$, $y = -2.907$, log-derivative $\ln\frac{0.9}{0.0518 \cdot 0.9482} = 2.908$. dark and bright pixels get stretched most, because the logit is steep near the edges. the checks reproduce these formulas, and that a randomly initialized RealNVP's density integrates to 1.

## on paper

1. **the coupling Jacobian.** write the Jacobian of an affine coupling layer with $D = 4$ and mask $(1, 1, 0, 0)$ as a block matrix, and show $\log|\det J| = s_3(x_m) + s_4(x_m)$.

<details><summary>solution</summary>

with $y_{1,2} = x_{1,2}$ and $y_j = x_j e^{s_j(x_1, x_2)} + t_j(x_1, x_2)$ for $j = 3, 4$: $J = \begin{pmatrix} I_2 & 0 \\ A & \operatorname{diag}(e^{s_3}, e^{s_4}) \end{pmatrix}$ where $A_{jk} = x_j e^{s_j}\partial_k s_j + \partial_k t_j$. block lower triangular, so $\det J = \det I_2 \cdot \det\operatorname{diag}(e^{s_3}, e^{s_4}) = e^{s_3 + s_4}$; the block $A$ never enters. $\log|\det J| = s_3 + s_4$.

</details>

2. **composition.** show that if $f = f_2 \circ f_1$ with both invertible, then $\log|\det J_f(x)| = \log|\det J_{f_1}(x)| + \log|\det J_{f_2}(f_1(x))|$, and that the inverse's log-determinant is the negative of the forward one.

<details><summary>solution</summary>

the chain rule gives $J_f(x) = J_{f_2}(f_1(x))\,J_{f_1}(x)$, and $\det(AB) = \det A \det B$, so the log-determinants add. for the inverse, $f^{-1}(f(x)) = x$ gives $J_{f^{-1}}(f(x))\,J_f(x) = I$, so $\det J_{f^{-1}}(f(x)) = 1/\det J_f(x)$ and the logs are negatives.

</details>

3. **why flows leak mass between modes.** let $f: \mathbb{R}^2 \to \mathbb{R}^2$ be a continuous bijection with continuous inverse, and $p_Z$ a Gaussian (positive everywhere). show that the flow's density $p_X$ is positive everywhere, and explain why a path between two modes of $p_X$ must carry probability mass.

<details><summary>solution</summary>

$p_X(x) = p_Z(f(x))\,|\det J_f(x)|$. for a differentiable bijection with differentiable inverse, $\det J_f \ne 0$, and $p_Z > 0$, so $p_X(x) > 0$ everywhere: no exact zeros between modes. to make the density between two modes tiny, $f$ must compress the region between them into a very small part of latent space (tiny $|\det J_f|$ there), which takes large, sharp derivatives. a finite network with bounded scales (the $\tanh$-bounded $s$) can only approximate that, so it leaves thin, low-but-nonzero "bridges" whose total mass shrinks only slowly with depth.

</details>

4. **the logit transform's log-determinant.** derive $\frac{d}{du}\operatorname{logit}(\alpha + (1 - 2\alpha)u)$ and its log for a $D$-dimensional image.

<details><summary>solution</summary>

$\frac{d}{dp}\log\frac{p}{1 - p} = \frac1p + \frac{1}{1 - p} = \frac{1}{p(1 - p)}$ and $\frac{dp}{du} = 1 - 2\alpha$, so the derivative is $\frac{1 - 2\alpha}{p(1 - p)}$. the map is elementwise, so its Jacobian is diagonal and $\log|\det| = \sum_i \log\frac{1 - 2\alpha}{p_i(1 - p_i)}$. each term is positive whenever $\alpha < 3/8$ (the minimum of $1/(p(1-p))$ is 4, and $4(1 - 2\alpha) > 1$ for $\alpha < 3/8$): the transform expands volume, so it raises $\log p(u)$ relative to $\log p(y)$.

</details>

## experiment

`lab.py` trains your RealNVP on 2-D toys:

1. exact likelihoods: the density integrates to 1, and the gap to the true NLL of the 8 gaussians.
2. how much mass a flow leaks between separated modes.
3. depth: 2 to 16 coupling layers.

## check your understanding

1. why can $s$ and $t$ be arbitrary networks, even non-invertible ones?

<details><summary>answer</summary>

inverting the coupling layer never inverts $s$ or $t$: they're evaluated on $x_m = y_m$, which the inverse already knows (it passes through unchanged). invertibility comes from the structure (identity on half, an elementwise affine map with nonzero scale $e^s$ on the other half), not from the networks.

</details>

2. what would go wrong with a single coupling layer, or with all layers using the same mask?

<details><summary>answer</summary>

the masked coordinates would never be transformed: $z_m = x_m$ exactly, so their marginal stays whatever it was in the data and must already be standard normal, and they'd never depend on the others. alternating masks let every coordinate be transformed, conditioned on coordinates transformed earlier.

</details>

3. a flow and a VAE both map data to latents. what's different about their latents?

<details><summary>answer</summary>

a flow's latent has the same dimension as the data and the map is a deterministic bijection: every $x$ has exactly one $z$, and $x$ can be recovered exactly. a VAE's latent is lower-dimensional and stochastic: $q(z \mid x)$ is a distribution, decoding loses information, and the likelihood is only bounded. flows pay for exactness with full-dimensional latents and constrained architectures.

</details>

4. why is maximum likelihood "mode-covering", and why does that matter for samples?

<details><summary>answer</summary>

the forward KL $\mathbb{E}_{p_\text{data}}[\log p_\text{data} - \log p_\theta]$ is infinite if $p_\theta = 0$ anywhere the data has mass, so the model must cover every mode. it costs little to put mass where the data has none, so likelihood models over-spread: samples include implausible in-between points (blur in VAEs, bridges in flows). GANs fail the other way.

</details>

5. a flow reports $-1{,}000$ nats per MNIST image on data scaled to $[0, 1]$, without dequantization. what's wrong with comparing that to a PixelCNN?

<details><summary>answer</summary>

it's a continuous density on discrete data: without dequantization the flow can concentrate mass on the 256 values of each pixel and push the density towards infinity, and without the $D \log 256$ term it's measured on the wrong scale. a negative NLL in nats per image is a red flag. dequantize, add the logit transform's log-determinant, add $D\log L$, then convert to bits/dim.

</details>

6. why bound $s$ with a $\tanh$?

<details><summary>answer</summary>

$e^{s}$ appears in the forward pass and $e^{-s}$ in the inverse. an unbounded $s$ can blow up early in training, making $y$ or the inverse overflow and the log-determinant huge, which destabilizes the likelihood. a bounded $s$ (with a learned scale) keeps each layer's volume change moderate; depth provides the rest.

</details>

## common bugs

- **summing $s$ over all coordinates.** symptom: the log-determinant includes the masked coordinates, where the map is the identity; the density no longer integrates to 1.
- **computing $s, t$ from the full $x$.** symptom: the inverse can't reproduce them (it doesn't know the transformed half), so `inverse(forward(x)) != x`.
- **the inverse in the wrong order.** symptom: `y * exp(-s) - t` instead of `(y - t) * exp(-s)`; reconstructions are off by $t(1 - e^{-s})$.
- **forgetting to reverse the layers in `inverse`.** symptom: sampling produces garbage while `log_prob` looks fine.
- **reporting `log_prob` of logit-transformed data as the likelihood of the pixels.** symptom: bits/dim several bits too good (or negative); add the transform's log-determinant and the dequantization term.
- **an untrained-looking flow that never moves.** symptom: zero-initialized $s, t$ layers combined with a mistake that blocks their gradients (e.g. a `detach` in the coupling); the loss is constant.

## going deeper

- Dinh, Sohl-Dickstein & Bengio (2017), "Density estimation using Real NVP", *ICLR*. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- Dinh, Krueger & Bengio (2015), "NICE: non-linear independent components estimation", *ICLR workshop*. additive couplings. [arXiv:1410.8516](https://arxiv.org/abs/1410.8516)
- Kingma & Dhariwal (2018), "Glow: generative flow with invertible 1x1 convolutions", *NeurIPS*. [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)
- Papamakarios, Pavlakou & Murray (2017), "Masked autoregressive flow for density estimation", *NeurIPS*. [arXiv:1705.07057](https://arxiv.org/abs/1705.07057)
- Papamakarios, Nalisnick, Rezende, Mohamed & Lakshminarayanan (2021), "Normalizing flows for probabilistic modeling and inference", *JMLR* 22. a survey. [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)
- Cornish, Caterini, Deligiannidis & Doucet (2020), "Relaxing bijectivity constraints with continuously indexed normalising flows", *ICML*. why flows struggle with disconnected supports. [arXiv:1909.13833](https://arxiv.org/abs/1909.13833)

## connections

- ← builds on: change of variables and Jacobians (ch 03 · topic 06), determinants (ch 02), maximum likelihood (ch 05), dequantization and bits/dim (topic 01), MADE (topic 02), mode coverage (topic 05).
- → used in: the chapter capstone (the exact-likelihood latent family on MNIST), continuous normalizing flows and neural ODEs (ch 20 · topic 06), flow matching, which trains continuous flows without simulating them (ch 22 · topic 05), and density models for anomaly detection and simulation-based inference.
