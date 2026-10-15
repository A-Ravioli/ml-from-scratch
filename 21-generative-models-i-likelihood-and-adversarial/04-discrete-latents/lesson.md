# discrete latents

## why this matters

a VAE's code is a vector of real numbers, and its prior $\mathcal{N}(0, I)$ is a poor description of where real codes live (topic 03's lab showed the gaps). language, on the other hand, is made of discrete symbols, and models of discrete sequences are the most powerful generative models we have (chapter 18). what if an image could be turned into a short sequence of symbols, a few dozen "visual words" from a learned vocabulary? then compression is just counting symbols, and generation can be handed to any model of discrete sequences: a PixelCNN in 2017, a transformer in DALL·E (2021), the token-based audio and video models of today.

that's the **vector-quantized VAE** (van den Oord, Vinyals & Kavukcuoglu, 2017). an encoder maps each patch of the image to a vector; the vector is replaced by the nearest entry of a learned **codebook** of $K$ vectors; a decoder reconstructs the image from the chosen entries. the image becomes a grid of integers in $\{0, \dots, K-1\}$. two problems have to be solved to train it. "replace by the nearest entry" is an argmin, which has no useful gradient: you'll build the **straight-through estimator** that trains the encoder anyway. and codebooks love to **collapse**: most entries are never chosen, never trained, and the "$K = 512$" model quietly uses a few dozen codes. you'll measure that and build the standard fix, then find out in the lab how much it actually buys.

## warm-up: predict first

1. $z_q = \text{codebook}[\arg\min_k \|z_e - e_k\|]$. what is $\partial z_q / \partial z_e$?
2. if the decoder's gradient can't pass through the quantizer, which parts of the model learn anything?
3. a VQ-VAE turns a $28 \times 28$ digit into a $7 \times 7$ grid of codes from a codebook of 256. how many bits per image is that?
4. you initialize a codebook of 512 entries. after training, how many do you expect to be in use?
5. a VQ-VAE has no KL term. what keeps it from being a plain autoencoder, and how do you sample from it?

## prerequisites check

- [ ] k-means and the nearest-centroid assignment (→ ch 11 · topic 01)
- [ ] the VAE, the ELBO and the reparameterization trick (→ topic 03)
- [ ] `detach()` and how autograd treats it (→ ch 14 · topic 03, ch 15 · topic 07)
- [ ] entropy in nats and bits (→ ch 05 · topic 05); convolutions and transposed convolutions (→ ch 16)
- [ ] autoregressive priors over discrete symbols (→ topic 02)

## core ideas

### idea 1: vector quantization

**vector quantization** is k-means used as a code: keep a codebook $\{e_1, \dots, e_K\} \subset \mathbb{R}^D$ and represent any vector $z$ by the index of its nearest entry,

$$k^\star(z) = \arg\min_k \|z - e_k\|^2.$$

for $N$ vectors and $K$ codes that's an $N \times K$ matrix of squared distances, computed in one shot by expanding the square:

$$\|z_n - e_k\|^2 = \|z_n\|^2 - 2\, z_n^\top e_k + \|e_k\|^2,$$

a matrix product plus two broadcast vectors, with no loop over $n$. the index costs $\log_2 K$ bits. in a VQ-VAE, the encoder produces one $D$-dimensional vector per position of a downsampled grid (a $7 \times 7$ grid for a $28 \times 28$ image), and each is quantized independently: the image becomes 49 indices, $49 \log_2 K$ bits.

> **spec · stage 1.** `nearest_code(z, codebook)`: `z` `(N, D)`, `codebook` `(K, D)` → int64 indices `(N,)` of the nearest code in squared Euclidean distance, ties to the **lowest** index (`argmin` does this), vectorized through the $N \times K$ distance matrix.

> **▶ build: stage 1.** implement `nearest_code` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: straight-through gradients and the two losses

the forward pass of the quantizer is $z_q = e_{k^\star(z_e)}$: piecewise constant in $z_e$. its derivative is 0 almost everywhere (moving $z_e$ a little doesn't change which code is nearest) and undefined on the boundaries. if you backpropagate through it honestly, the encoder receives no gradient from the reconstruction at all.

the **straight-through estimator** (Bengio, Léonard & Courville, 2013) pretends the quantizer is the identity in the backward pass: use $z_q$'s value going forward and copy the gradient from $z_q$ to $z_e$ unchanged going back. in code, a one-liner with `detach` (which blocks gradients):

$$z_{\text{st}} = z_e + \operatorname{sg}(z_q - z_e),$$

where $\operatorname{sg}$ is stop-gradient. its value is $z_q$; its gradient with respect to $z_e$ is the identity. the estimate is biased, but it points the encoder in the direction that would have helped if $z_e$ itself had been decoded, and since $z_q$ is near $z_e$, that's a good direction.

the straight-through path never trains the **codebook**: no gradient reaches $e_k$ through it. VQ-VAE adds two loss terms, each with a stop-gradient that decides who moves:

$$\mathcal{L}_\text{VQ} = \underbrace{\|\operatorname{sg}(z_e) - z_q\|^2}_{\text{codebook loss}} + \beta\,\underbrace{\|z_e - \operatorname{sg}(z_q)\|^2}_{\text{commitment loss}}.$$

the codebook loss moves each chosen code towards the encoder outputs assigned to it (exactly a k-means update, done by gradient descent). the commitment loss keeps the encoder's outputs close to their codes, so they don't drift off while the codebook chases them. with $\beta = 0.25$ (the paper's default), the codebook moves faster than the encoder. both terms have the same *value*, so $\mathcal{L}_\text{VQ}$'s value is $(1 + \beta)\,\|z_e - z_q\|^2$, but their gradients go to different places: the encoder gets $\beta \cdot 2(z_e - e)$, the code gets $2(e - z_e)$ (per element, divided by the number of elements for a mean). the full training loss is the reconstruction error plus $\mathcal{L}_\text{VQ}$.

notice what's missing: a KL term. the VQ-VAE's "posterior" is a one-hot choice and its prior is uniform over the codes, so the KL is the constant $\log K$ per position and drops out of training. a VQ-VAE is trained like an autoencoder; it becomes a generative model only once you fit a **prior** over the code grids (a PixelCNN over the $7 \times 7$ indices, say) and sample from it.

> **spec · stage 2.** `VectorQuantizer(n_codes, code_dim, *, beta=0.25)`, an `nn.Module` with `self.codebook = nn.Embedding(n_codes, code_dim)` initialized uniform in $[-1/K, 1/K]$. `forward(z_e)`: `z_e` `(B, D, *spatial)` (images `(B, D, H, W)`, sequences such as audio `(B, D, T)`) → `(z_q, vq_loss, indices)`: `indices` `(B, *spatial)` int64, the nearest code for each vector `z_e[b, :, ...]` at each position; `z_q` of `z_e`'s shape with the selected codes' values and the straight-through gradient; `vq_loss = mse(z_q, sg(z_e)) + beta * mse(z_e, sg(z_q))`, means over all elements. this is `mlfs.generative.VectorQuantizer`.

> **▶ build: stage 2.** implement `VectorQuantizer`.

> **🔮 predict:** a VQ-VAE with a codebook of 256 entries is trained on MNIST with nothing but these two losses. how many of the 256 codes will it end up using?

### idea 3: codebook collapse, and dead-code restarts

here's how a codebook dies. the codes start in a tiny ball around the origin, while the encoder's outputs start wherever its random initialization puts them, typically concentrated in a few directions. on the first batches, only the codes that happen to point the right way win any argmin, and only those receive codebook-loss gradient and move towards the data, which makes them win even more often (rich get richer). a code that never wins never gets a gradient, never moves, and so never wins later either: it's **dead**. the effective codebook is the set of live codes, and it can be a small fraction of $K$.

two numbers measure it. the fraction of codes used at least once, and the **perplexity** of the code distribution: if code $k$ is used with frequency $p_k$, the perplexity is $e^{H(p)} = \exp(-\sum_k p_k \log p_k)$, the number of equally used codes that would have the same entropy. $K$ codes used uniformly give a perplexity of $K$; one code gives 1. its log base 2 is the number of bits per position the model actually spends.

the standard fix is blunt: every so often, find the codes that weren't used in the last stretch of training and **restart** them at randomly chosen encoder outputs from the current batch (Dhariwal et al., 2020; Williams et al., 2020). a restarted code sits right on top of real data, so it wins at least that vector's argmin and starts receiving gradient. other fixes initialize the codebook by k-means on encoder outputs, update it by an exponential moving average instead of gradients, or project codes to a low-dimensional space before the lookup.

> **spec · stage 3.**
> - `codebook_usage(indices, n_codes)`: `(perplexity, fraction_used)`, two floats: $\exp$ of the entropy (in nats) of the empirical distribution of `indices` (any shape) over `n_codes` codes, unused codes contributing 0, and the fraction of codes used at least once.
> - `restart_dead_codes(vq, z_e, counts, *, threshold=1, generator=None)`: every code with `counts < threshold` is replaced, in place, by an encoder output vector; `z_e` `(B, D, *spatial)` is flattened to its $N$ vectors `z_e[b, :, ...]` in row-major order of `(b, *positions)` (a 2-D `z_e` is taken to be already flattened, `(N, D)`), and the rows are drawn with `torch.randint(N, (n_dead,), generator=generator)`, assigned to the dead codes in increasing order of code index. returns `n_dead`; live codes are untouched.

> **▶ build: stage 3.** implement `codebook_usage` and `restart_dead_codes`.

> **🔮 predict:** you restart dead codes every 50 steps. the number of codes in use goes from a few dozen to almost all 256. what happens to the reconstruction error?

### idea 4: the VQ-VAE

the full model: a convolutional encoder that downsamples by 4 (two stride-2 convolutions) to a grid of `code_dim`-dimensional vectors, the vector quantizer, and a convolutional decoder that upsamples by 4 (two stride-2 transposed convolutions) back to the image. `forward` returns the reconstruction, the VQ loss and the indices; training minimizes $\|x - \hat x\|^2 + \mathcal{L}_\text{VQ}$. two more methods turn it into a tokenizer: `encode(x)` returns only the integer grid, and `decode_indices` maps a grid of indices back to an image (codebook lookup, then the decoder). `decode_indices(encode(x))` must reproduce `forward(x)`'s reconstruction exactly: the indices are the whole story.

a trained VQ-VAE plus a prior over its index grids is a generative model with an honest likelihood (for the indices), a compressed representation, and a tokenization that any sequence model can use. the audio elective builds a neural audio codec this way, stacking several quantizers in a residual chain (each quantizing what the previous ones missed).

> **spec · stage 4.** `VQVAE(in_channels=3, *, n_codes=512, code_dim=64, base_channels=64)`, an `nn.Module` with `self.vq` a `VectorQuantizer(n_codes, code_dim)`, an encoder that downsamples by 4 and a decoder that upsamples by 4 (internal design yours). `forward(x)`: `(B, C, H, W)` with $H, W$ divisible by 4 → `(x_recon (B, C, H, W), vq_loss, indices (B, H/4, W/4))`; `encode(x)` → `indices`; `decode_indices(indices)` → images `(B, C, 4h, 4w)`, equal to `forward`'s reconstruction for `encode(x)`. the decoder's last layer has no activation (train with MSE). this is `mlfs.generative.VQVAE`.

> **▶ build: stage 4.** implement `VQVAE`.

## worked example

**quantizing by hand.** codebook $e_0 = (0, 0)$, $e_1 = (2, 0)$, $e_2 = (0, 2)$, $e_3 = (2, 0)$ (a duplicate of $e_1$). the vector $(0.9, 0)$ has squared distances $0.81, 1.21, 4.81, 1.21$: code 0. $(1, 0)$ is at distance 1 from both $e_0$ and $e_1$: the tie goes to code 0. $(1.6, 0.1)$: distances $2.57, 0.17, 6.17, 0.17$, a tie between 1 and 3, so code 1, and code 3 can never win anything: a duplicate code is dead from birth. $(0.2, 1.5)$: code 2.

**the losses and their gradients.** one encoder output $z_e = (1, 1)$ assigned to $e = (0, 2)$: the squared error per element is $(1, 1)$, mean $1$. $\mathcal{L}_\text{VQ} = 1 + 0.25 \cdot 1 = 1.25$. gradients (mean over the 2 elements): encoder $\beta \cdot 2(z_e - e)/2 = 0.25 \cdot (1, -1) = (0.25, -0.25)$; code $2(e - z_e)/2 = (-1, 1)$. gradient descent moves the code towards $z_e$ four times as fast as it moves $z_e$ towards the code.

**perplexity.** code indices $[3, 3, 3, 5]$ over $K = 6$: frequencies $0.75, 0.25$, entropy $-(0.75 \ln 0.75 + 0.25 \ln 0.25) = 0.562$ nats, perplexity $e^{0.562} = 1.75$; 2 of 6 codes used. the checks reproduce all three examples.

**rate.** a $7 \times 7$ grid with $K = 256$: $49 \times 8 = 392$ bits per image, against $6{,}272$ for the raw 8-bit image: 16 times smaller, before any entropy coding.

## on paper

1. **the straight-through gradient.** for $z_{\text{st}} = z_e + \operatorname{sg}(z_q - z_e)$, show that the forward value is $z_q$ and that $\partial \mathcal{L}/\partial z_e = \partial \mathcal{L}/\partial z_{\text{st}}$ for any downstream loss $\mathcal{L}$.

<details><summary>solution</summary>

forward: $z_e + (z_q - z_e) = z_q$, since sg is the identity on values. backward: sg has zero derivative, so $\partial z_{\text{st}}/\partial z_e = I + 0 = I$, and the chain rule gives $\partial\mathcal{L}/\partial z_e = (\partial z_{\text{st}}/\partial z_e)^\top \partial\mathcal{L}/\partial z_{\text{st}} = \partial\mathcal{L}/\partial z_{\text{st}}$. the codebook gets nothing through this path, because $z_q$ appears only inside sg.

</details>

2. **the codebook loss is k-means.** fix the assignments. show that a gradient step of size $\eta$ on the codebook loss (sum over the $n_k$ vectors assigned to code $k$, not the mean) moves $e_k$ towards the mean of its assigned vectors, and that $\eta = 1/(2n_k)$ lands exactly on it.

<details><summary>solution</summary>

$\nabla_{e_k} \sum_{n: k^\star(n) = k} \|z_n - e_k\|^2 = -2\sum_n (z_n - e_k) = -2n_k(\bar z_k - e_k)$. the step $e_k \leftarrow e_k + 2\eta n_k (\bar z_k - e_k)$ moves towards the mean $\bar z_k$, and with $\eta = 1/(2n_k)$ it gives $e_k = \bar z_k$: the k-means centroid update. the EMA variant of VQ-VAE does this update directly instead of by gradient descent.

</details>

3. **perplexity and bits.** show that the perplexity of a distribution over $K$ codes is at most $K$, with equality iff the distribution is uniform, and convert a perplexity of 20 into bits per position.

<details><summary>solution</summary>

$H(p) \le \log K$ with equality iff $p$ is uniform (Gibbs' inequality, or KL$(p \| \text{uniform}) = \log K - H(p) \ge 0$). so $e^{H} \le K$. a perplexity of 20 is $\log_2 20 = 4.32$ bits per position: a codebook of 256 entries used with perplexity 20 behaves, entropy-wise, like a uniform codebook of 20.

</details>

4. **why codes die, in low and high dimensions.** the codebook is initialized uniformly in $[-1/K, 1/K]^D$ and the encoder outputs are far from it ($\|z\| \gg 1/K$). (a) argue that only codes on the convex hull of the codebook's tiny cloud can ever win, and that in $D = 2$ most codes start dead. (b) why does that argument fail for $D = 16$, and what kills codes there instead?

<details><summary>solution</summary>

(a) for $\|z\| \gg 1/K$, the nearest code to $z$ is the one maximizing $z^\top e_k - \|e_k\|^2/2 \approx z^\top e_k$: the code furthest in the direction of $z$. only codes on the convex hull of the cloud are extreme in some direction, so interior codes never win. in 2-D the hull of $K$ random points has only a handful of vertices (about 15 of 256 for a uniform square), so most codes start dead. (b) in high dimensions almost every point of a random cloud is a hull vertex (in 16-D, all 256 are), so every code *could* win for some direction. what kills codes there is that the encoder's outputs don't point in all directions: they occupy a small region, so only the codes extreme in those few directions win at first, and the codebook loss then pulls the winners towards the data, making them win more (rich get richer) while the rest never move. restarts, k-means initialization, or initializing codes at encoder outputs all avoid this.

</details>

## experiment

`lab.py` trains your VQ-VAE on MNIST:

1. how many of 256 codes are in use, and how that evolves during training.
2. the same models with dead-code restarts.
3. codebooks of 2, 16 and 256 codes: bits per image against reconstruction quality.

## check your understanding

1. why does the encoder need the straight-through estimator, while the codebook doesn't?

<details><summary>answer</summary>

the reconstruction's gradient reaches $z_q$, but the argmin blocks it from $z_e$; straight-through copies it across. the codebook is trained by its own loss term, $\|\operatorname{sg}(z_e) - z_q\|^2$, which depends on $e_k$ directly (as the value looked up), so it has an ordinary gradient.

</details>

2. what would happen without the commitment loss?

<details><summary>answer</summary>

the encoder's outputs would be free to drift and grow, since only the reconstruction (through straight-through) acts on them; the codebook would chase them, and the gap $\|z_e - z_q\|$ could grow, making the straight-through approximation (decode $z_q$, update as if $z_e$ were decoded) poor. the commitment term keeps encoder outputs near their codes so the codebook can keep up.

</details>

3. why is the value of `vq_loss` equal to $(1 + \beta)$ times the MSE, even though the two terms play such different roles?

<details><summary>answer</summary>

stop-gradient doesn't change values, only gradients: $\|\operatorname{sg}(z_e) - z_q\|^2$ and $\|z_e - \operatorname{sg}(z_q)\|^2$ are numerically the same number. what differs is where each term's gradient goes (codebook vs encoder). the number you log is not informative about which side is moving.

</details>

4. a VQ-VAE with $K = 512$ reports a perplexity of 30. how many bits per position does it use, and what does that mean for a prior fitted over its codes?

<details><summary>answer</summary>

about $\log_2 30 = 4.9$ bits per position, out of a nominal 9. a prior over the codes only needs to model the ~30 codes in use; the rest are wasted parameters. you could shrink the codebook, or revive the dead codes, to use the capacity you're paying for.

</details>

5. how do you generate new images with a trained VQ-VAE?

<details><summary>answer</summary>

the VQ-VAE alone has no useful prior (uniform over codes, independently per position, would give garbage). fit an autoregressive model (a PixelCNN, or a transformer over the flattened grid) to the index grids of the training set, sample a grid from it, and call `decode_indices`. the likelihood of an image is then the prior's likelihood of its codes (plus the reconstruction term).

</details>

6. why is a duplicate code dead from the start?

<details><summary>answer</summary>

any vector equally close to two identical codes goes to the lower index (ties to the lowest), and any vector closer to one is closer to both. so the higher-index duplicate never wins an argmin, never receives a gradient, and never moves away from its twin.

</details>

## common bugs

- **returning `z_q` without the straight-through trick.** symptom: the encoder's gradient is zero; reconstruction improves only through the decoder and the codebook, and stalls.
- **detaching everything** (`z_q.detach()`). symptom: same as above, and the decoder still trains, which hides the bug in the loss curve.
- **putting $\beta$ on the codebook term.** symptom: the encoder pulls hard on codes that barely move; codes lag behind and the gap $\|z_e - z_q\|$ stays large.
- **forgetting a stop-gradient in the codebook term.** symptom: the encoder gets an extra gradient towards the codes with weight 1 instead of $\beta$.
- **flattening `(B, D, H, W)` without moving `D` last.** symptom: "vectors" made of values from different positions; quantization error is huge and indices don't match positions.
- **a loop over vectors in `nearest_code`.** symptom: training runs 100× slower than it should.

## going deeper

- van den Oord, Vinyals & Kavukcuoglu (2017), "Neural discrete representation learning", *NeurIPS*. the VQ-VAE. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- Bengio, Léonard & Courville (2013), "Estimating or propagating gradients through stochastic neurons for conditional computation". the straight-through estimator. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- Razavi, van den Oord & Vinyals (2019), "Generating diverse high-fidelity images with VQ-VAE-2", *NeurIPS*. hierarchical codes and PixelCNN priors. [arXiv:1906.00446](https://arxiv.org/abs/1906.00446)
- Dhariwal et al. (2020), "Jukebox: a generative model for music". random restarts of dead codes. [arXiv:2005.00341](https://arxiv.org/abs/2005.00341)
- Esser, Rombach & Ommer (2021), "Taming transformers for high-resolution image synthesis", *CVPR*. VQGAN with a transformer prior. [arXiv:2012.09841](https://arxiv.org/abs/2012.09841)
- Zeghidour et al. (2021), "SoundStream: an end-to-end neural audio codec", *IEEE/ACM TASLP*. residual vector quantization. [arXiv:2107.03312](https://arxiv.org/abs/2107.03312)

## connections

- ← builds on: k-means (ch 11 · topic 01), the VAE (topic 03), `detach` and autograd (ch 14–15), entropy (ch 05), autoregressive models (topic 02).
- → used in: discrete tokenizers for image and audio generation (residual VQ in the audio elective), autoregressive priors over codes, transformer models of discrete tokens (ch 18), and the comparison of latent spaces in the capstone's write-up.
