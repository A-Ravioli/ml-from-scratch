# 21 · generative models I: likelihood & adversarial · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. what is bits per dimension, and what does a uniform model of 8-bit images score?

<details><summary>answer</summary>

the NLL in nats per example divided by $D \ln 2$, with $D$ the number of pixels times channels: the average number of bits per dimension an arithmetic coder driven by the model would need. a uniform distribution over 256 levels scores exactly 8 bits/dim (2 levels: 1 bit/dim).

</details>

2. why must a continuous density be evaluated on *dequantized* data to be compared with a discrete model, and what term does the comparison need?

<details><summary>answer</summary>

on integer data, a density can put spikes at the integers and grow without bound. adding uniform noise $u \in [0, 1)$ and scaling by $1/L$ gives each discrete image a box of volume $L^{-D}$; by Jensen, $-\log P(x) \le \mathbb{E}_u[-\log p((x + u)/L)] + D \log L$. without the $D\log L$ term the "NLL" is measured on the wrong scale (8 bits/dim too good for $L = 256$).

</details>

3. write the Fréchet distance between two Gaussians and say why $(C_1 C_2)^{1/2}$ is computed through $C_1^{1/2} C_2 C_1^{1/2}$.

<details><summary>answer</summary>

$\|\mu_1 - \mu_2\|^2 + \operatorname{Tr}(C_1 + C_2 - 2(C_1C_2)^{1/2})$. $C_1C_2$ is not symmetric, but it is similar to the symmetric PSD matrix $C_1^{1/2}C_2C_1^{1/2}$, which has the same eigenvalues; so the trace of the square root is the sum of square roots of eigenvalues from two `eigh` calls, with no complex arithmetic.

</details>

4. why is FID between two samples of the same distribution not zero, and how does $\text{FID}_\infty$ deal with it?

<details><summary>answer</summary>

the means and covariances are estimated from finite samples, and their estimation noise adds a positive bias that shrinks roughly like $1/n$ (in 128 dimensions with $n = 50$, the covariance estimate isn't even full rank). $\text{FID}_\infty$ computes FID at several $n$, fits $a + b/n$, and reports the intercept.

</details>

5. a model draws perfect digits of only three classes. what do precision and recall say?

<details><summary>answer</summary>

precision high (its samples fall inside the real data's $k$-NN balls), recall low (most real digits aren't inside any ball around its samples). FID alone would only say "worse".

</details>

6. how do MADE's degree masks guarantee that output $d$ depends only on $x_{<d}$?

<details><summary>answer</summary>

inputs have degrees $1..D$, hidden units degrees in $[1, D - 1]$; a connection from degree $a$ to a hidden unit of degree $b$ exists iff $b \ge a$, and to output $d$ (degree $d$) iff $d > b$. along any path degrees never decrease and the last step is strict, so a path from $x_j$ to output $d$ exists only if $j < d$.

</details>

7. why does a PixelCNN use a type-A mask in its first layer and type-B masks after it?

<details><summary>answer</summary>

the first layer sees the image itself, so pixel $(r, c)$'s output must exclude input $(r, c)$ (type A). after it, the feature at $(r, c)$ depends only on earlier pixels, so later layers may include the centre (type B) without leaking the current pixel.

</details>

8. state the ELBO and explain the reparameterization trick.

<details><summary>answer</summary>

$\log p(x) \ge \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \operatorname{KL}(q(z \mid x) \| p(z))$. to differentiate the expectation with respect to the encoder, write $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$: the randomness no longer depends on the parameters, so gradients pass through $\mu$ and $\sigma$.

</details>

9. what is posterior collapse in a VAE, and how do you detect it?

<details><summary>answer</summary>

latent coordinates whose posterior equals the prior for every input (KL 0), so they carry no information and the decoder ignores them; it happens when a coordinate's KL costs more than it saves in reconstruction (more often with large $\beta$ or powerful decoders). detect it with the per-dimension KL and with active units: coordinates whose encoder mean varies across inputs by more than 0.01.

</details>

10. how does a VQ-VAE pass gradients through the argmin, and what do its two extra loss terms do?

<details><summary>answer</summary>

the straight-through estimator: $z_q = z_e + \operatorname{sg}(z_q - z_e)$ uses the code's value forwards and copies the gradient to $z_e$ unchanged backwards. the codebook loss $\|\operatorname{sg}(z_e) - e\|^2$ moves codes towards the encoder outputs assigned to them; the commitment loss $\beta\|z_e - \operatorname{sg}(e)\|^2$ keeps encoder outputs near their codes.

</details>

11. why does the saturating GAN generator loss stall early in training, and what replaces it?

<details><summary>answer</summary>

with $D = \sigma(\ell)$, $\frac{d}{d\ell}\log(1 - \sigma(\ell)) = -\sigma(\ell)$, which vanishes when the discriminator confidently rejects the samples ($D(G(z)) \to 0$), exactly at the start. the non-saturating loss $-\log D(G(z))$ has gradient $-(1 - \sigma(\ell))$, close to $-1$ there.

</details>

12. what is mode collapse, and why does the alternating GAN game produce it?

<details><summary>answer</summary>

the generator covers only some modes of the data with good samples. each generator step moves samples towards whatever the *current* discriminator accepts; nothing in one step rewards covering all modes, and the discriminator is never optimal, so the pair can chase each other between modes or settle on a few.

</details>

13. give the dual form of the Wasserstein-1 distance, and three ways to keep a critic 1-Lipschitz.

<details><summary>answer</summary>

$W_1 = \sup_{\|f\|_L \le 1}\mathbb{E}_p[f] - \mathbb{E}_q[f]$. weight clipping (crude, the estimate is scaled down), a gradient penalty $\lambda\,\mathbb{E}[(\|\nabla f(\hat x)\| - 1)^2]$ on interpolates (WGAN-GP), and spectral normalization of every layer (a loose, product-of-layers bound).

</details>

14. why is the log-determinant of an affine coupling layer cheap, and why can $s$ and $t$ be arbitrary networks?

<details><summary>answer</summary>

the masked half passes through unchanged and the other half is transformed elementwise with scales depending only on the masked half, so the Jacobian is triangular with diagonal $(1, \dots, 1, e^{s_i})$: $\log|\det| = \sum s_i$. inverting the layer never inverts $s$ or $t$, only the elementwise affine map, because their input passes through unchanged.

</details>

15. compare the failure modes of a maximum-likelihood model and a GAN on a mixture of separated modes.

<details><summary>answer</summary>

maximum likelihood minimizes $\operatorname{KL}(p_\text{data} \| p_\theta)$, which is infinite if any data region gets zero density: it covers every mode and leaks mass between them (blur in VAEs, bridges in flows: high recall, lower precision). a GAN is punished only for samples the discriminator rejects: it can drop modes while keeping its samples sharp (high precision, low recall).

</details>

## 2. by hand

1. **likelihood bookkeeping.** (a) a flow reports $\log p(y) = 2{,}500$ nats on dequantized $28 \times 28$ binary images scaled to $[0, 1)$ ($L = 2$). what is its bound in bits/dim? (b) the same flow worked in logit space after `logit_transform` with $\alpha = 0.05$: for one pixel with dequantized value $u = 0.25$, compute $y$ and its contribution to the log-determinant. (c) a PixelCNN on the same data has an NLL of 80 nats per image: which model is better, and is the comparison fair?

<details><summary>solution</summary>

(a) the discrete bound is $-2{,}500 + 784\ln 2 = -2{,}500 + 543.4 = -1{,}956.6$ nats, which is negative: impossible for a probability $\le 1$, so the reported log-density must be wrong (most likely missing the transform's log-determinant or computed on the wrong scale). a correct bound is always $\ge 0$. (b) $p = 0.05 + 0.9 \cdot 0.25 = 0.275$, $y = \ln(0.275/0.725) = -0.969$, log-derivative $\ln\frac{0.9}{0.275 \cdot 0.725} = \ln 4.514 = 1.507$. (c) the PixelCNN: $80/(784 \ln 2) = 0.147$ bits/dim, exact. a flow's dequantized bound is an upper bound, so the comparison is fair in one direction: if the PixelCNN's exact value beats the flow's bound, the flow may still be better than its bound suggests, but it can't be shown to be.

</details>

2. **a VAE step and its bound.** a 2-D latent, encoder output $\mu = (0, 1)$, $\log\sigma^2 = (\log 0.5, 0)$. (a) compute the KL to $\mathcal{N}(0, I)$. (b) with $\epsilon = (2, -1)$, give the reparameterized $z$ and $\partial z / \partial\mu$, $\partial z / \partial \log\sigma^2$. (c) three importance samples have log-weights $(-50, -52, -60)$: give the ELBO estimate and the $k = 3$ estimate.

<details><summary>solution</summary>

(a) coordinate 1: $\tfrac12(0 + 0.5 - 1 - \ln 0.5) = \tfrac12(-0.5 + 0.693) = 0.097$; coordinate 2: $\tfrac12(1 + 1 - 1 - 0) = 0.5$; total $0.597$ nats. (b) $\sigma = (0.707, 1)$, $z = (0 + 0.707 \cdot 2, 1 - 1) = (1.414, 0)$; $\partial z/\partial\mu = (1, 1)$; $\partial z/\partial\log\sigma^2 = \tfrac12\sigma\epsilon = (0.707, -0.5)$. (c) the ELBO estimate is the mean, $-54$; the $k = 3$ estimate is $\operatorname{logsumexp}(-50, -52, -60) - \ln 3 = -49.873 - 1.099 = -50.97$, three nats tighter.

</details>

3. **two games.** (a) a discriminator outputs logits $(1, 3)$ on real images and $(-2, 0)$ on fakes: compute the discriminator loss. (b) the generator's samples get logit $-5$: give the per-sample gradient of the saturating and non-saturating losses with respect to the logit. (c) for point masses $\delta_0$ and $\delta_3$, give $W_1$ and the value of the critic $f(x) = -x$, and check that it's optimal.

<details><summary>solution</summary>

(a) real: $\tfrac12(\operatorname{softplus}(-1) + \operatorname{softplus}(-3)) = \tfrac12(0.313 + 0.049) = 0.181$; fake: $\tfrac12(\operatorname{softplus}(-2) + \operatorname{softplus}(0)) = \tfrac12(0.127 + 0.693) = 0.410$; total $0.591$. (b) $\sigma(-5) = 0.0067$: saturating $-\sigma = -0.0067$, non-saturating $-(1 - \sigma) = -0.9933$, 148 times larger. (c) $W_1 = 3$; $f(0) - f(3) = 0 - (-3) = 3$ and $f$ is 1-Lipschitz, so it attains the supremum.

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild the **VAE** (topic 03, stages 1–3): `reparameterize`, `kl_standard_normal`, `reconstruction_nll`, the `VAE` class and `iwae_nll`. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=21-generative-models-i-likelihood-and-adversarial/03-variational-autoencoders
mkdir -p /tmp/rebuild-21
git show HEAD:$T/exercise.py > /tmp/rebuild-21/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-21/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-21/solutions/` and run `python scripts/stubify.py /tmp/rebuild-21`.) open `/tmp/rebuild-21/exercise.py` with `marimo edit`, implement stages 1–3 from memory, and run the checks against your copy:

```bash
python -m pytest /tmp/rebuild-21/test_exercise.py -k "stage1 or stage2 or stage3"
```

you pass if every check is green within the hour, including the importance-weighted bound's convergence to the exact likelihood.

**harder variant** (for testing out of the chapter with confidence): within 120 minutes in total, also rebuild topic 04's `nearest_code` and `VectorQuantizer` (stages 1–2), topic 05's losses and `gan_step` (stages 1–2), and topic 07's `AffineCoupling` and `RealNVP` (stages 1–2), each in its own scratch directory the same way:

```bash
python -m pytest /tmp/rebuild-21-vq/test_exercise.py -k "stage1 or stage2"
python -m pytest /tmp/rebuild-21-gan/test_exercise.py -k "stage1 or stage2"
python -m pytest /tmp/rebuild-21-flow/test_exercise.py -k "stage1 or stage2"
```

(run them from the repo root: the notebooks import earlier `mlfs` modules, which the tests resolve to the reference solutions, so only the code you rebuilt is being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each part of a problem separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the idea, and the topic lessons are organized by those ideas.
