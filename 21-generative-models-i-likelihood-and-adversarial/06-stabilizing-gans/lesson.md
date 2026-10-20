# stabilizing GANs

## why this matters

topic 05's GANs hopped between modes, collapsed, and gave you loss curves that said nothing about sample quality. part of the trouble is the game itself, but part is the distance the original GAN implicitly minimizes. on paper 3 of topic 05 you saw that the Jensen–Shannon divergence between two distributions with disjoint supports is the constant $\log 2$, however far apart they are: an optimal discriminator separates them perfectly and tells the generator nothing about which way to move. real images live on thin manifolds in pixel space, so at the start of training the real and generated distributions almost never overlap, which is exactly that regime.

Arjovsky, Chintala & Bottou (2017) proposed a different distance: the **Wasserstein-1** or **earth mover's** distance, the minimum cost of moving the generated probability mass onto the data. it grows smoothly as the distributions separate, so it has a useful gradient everywhere. its dual form turns the discriminator into a **critic** that must be 1-Lipschitz, and *how* you enforce that constraint turned out to matter as much as the distance itself: the original weight clipping was crude, the **gradient penalty** of WGAN-GP (Gulrajani et al., 2017) worked much better, and **spectral normalization** (Miyato et al., 2018) became the cheap default in large GANs. you'll build all three, and the WGAN-GP training loop, and measure in the lab what each one really constrains.

## warm-up: predict first

1. two point masses at $0$ and $\theta$. what is their Wasserstein-1 distance, and its gradient with respect to $\theta$?
2. a critic must satisfy $|f(x) - f(y)| \le \|x - y\|$. if you clip every weight of a 3-layer MLP to $\pm 0.01$, is it 1-Lipschitz? is it *too* constrained?
3. the gradient penalty pushes $\|\nabla f\|$ towards 1. at which points $x$ is it evaluated?
4. spectral normalization divides each weight matrix by its largest singular value. how do you get that singular value cheaply at every step?
5. the WGAN critic is trained for 5 steps per generator step. why more than one?

## prerequisites check

- [ ] Lipschitz continuity, and the Lipschitz constant of a composition (→ ch 03 · topic 05)
- [ ] singular values, the spectral norm and power iteration (→ ch 02 · topics 05–06)
- [ ] linear programming duality, or at least the idea of a dual problem (→ ch 06 · topic 06)
- [ ] the GAN game, `gan_step` and mode coverage (→ topic 05)
- [ ] `torch.autograd.grad` with `create_graph=True` (→ ch 15 · topic 07)

## core ideas

### idea 1: the Wasserstein distance and its critic

the **Wasserstein-1** distance between distributions $p$ and $q$ is the cost of the cheapest transport plan:

$$W_1(p, q) = \inf_{\gamma \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \gamma}\big[\|x - y\|\big],$$

where a plan $\gamma$ is a joint distribution with marginals $p$ and $q$ (how much mass moves from each $x$ to each $y$). between point masses at 0 and $\theta$ it's $|\theta|$, with gradient $\pm 1$ everywhere: it *sees* how far apart the distributions are, which JS can't (on paper 1).

the infimum over plans is intractable for images, but the **Kantorovich–Rubinstein duality** rewrites it as a supremum over functions:

$$W_1(p, q) = \sup_{\|f\|_L \le 1}\ \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{y \sim q}[f(y)],$$

over all 1-Lipschitz functions $f$ (those with $|f(x) - f(y)| \le \|x - y\|$). the maximizing $f$ is a **critic**: a scalar score, high on real data and low on generated data, whose slope is at most 1. a WGAN trains a neural critic to maximize this difference (so it *minimizes* `critic_loss = fake.mean() - real.mean()`), and the generator to minimize the critic's estimate, i.e. to raise its own scores: `wgan_generator_loss = -fake.mean()`. no logs, no sigmoids: the generator's gradient on each sample's score is a constant.

the critic must be (close to) 1-Lipschitz, or the supremum is infinite: scale any separating $f$ by 1,000 and the difference grows 1,000-fold. the original WGAN enforced it bluntly, by **clipping** every weight to $[-c, c]$ after each critic step ($c = 0.01$). bounded weights give a bounded Lipschitz constant, but the bound can be far below 1, and clipping biases the critic towards very simple functions; Gulrajani et al. showed it also leads to exploding or vanishing gradients in deeper critics.

> **spec · stage 1.**
> - `critic_loss(real_scores, fake_scores)`: `fake_scores.mean() - real_scores.mean()`. this is `mlfs.generative.critic_loss`.
> - `wgan_generator_loss(fake_scores)`: `-fake_scores.mean()`. this is `mlfs.generative.wgan_generator_loss`.
> - `clip_weights_(module, c=0.01)`: clamp every parameter of `module` (weights and biases) to $[-c, c]$ in place, without gradients; return `module`.

> **▶ build: stage 1.** implement `critic_loss`, `wgan_generator_loss` and `clip_weights_` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** the generator's samples sit further and further outside the data. for each of the saturating GAN loss, the non-saturating GAN loss and the Wasserstein loss, does the gradient a sample receives shrink, stay the same, or grow with the distance?

### idea 2: the gradient penalty

a differentiable $f$ is 1-Lipschitz iff $\|\nabla f(x)\| \le 1$ everywhere. and the optimal critic has more structure: on the lines connecting coupled points of the optimal transport plan, its gradient has norm exactly 1 (on paper 2). **WGAN-GP** (Gulrajani et al., 2017) turns that into a soft constraint. sample points on lines between real and generated samples and penalize gradient norms that differ from 1:

$$\mathcal{L}_\text{critic} = \mathbb{E}[f(\tilde x)] - \mathbb{E}[f(x)] + \lambda\, \mathbb{E}_{\hat x}\Big[\big(\|\nabla_{\hat x} f(\hat x)\|_2 - 1\big)^2\Big], \qquad \hat x = \epsilon x + (1 - \epsilon)\tilde x,\ \epsilon \sim U[0, 1],$$

with one $\epsilon$ per pair of a real $x$ and a generated $\tilde x$, and $\lambda = 10$. the penalty is two-sided (it pulls norms *up* to 1 as well as down), which matches the optimal critic and works better in practice than a one-sided $\max(0, \|\nabla f\| - 1)^2$.

computing it needs a gradient *of* a gradient. `torch.autograd.grad(f(x_hat).sum(), x_hat, create_graph=True)` returns $\nabla_{\hat x} f$ as a tensor that is itself part of the graph, so the penalty can be backpropagated into the critic's parameters. for images, the norm is over all of a sample's pixels and channels, and $\epsilon$ has shape $(B, 1, 1, 1)$.

> **spec · stage 2.** `gradient_penalty(critic, real, fake, *, generator=None)`: `eps = torch.rand((B, 1, ..., 1), generator=generator)` (one per example, broadcast over the other dims), $\hat x = \epsilon\,\text{real} + (1 - \epsilon)\,\text{fake}$, the gradient of `critic(x_hat).sum()` with respect to $\hat x$ (with `create_graph=True`), its 2-norm over all non-batch dims, and the mean of $(\text{norm} - 1)^2$: a scalar, differentiable with respect to the critic's parameters. this is `mlfs.generative.gradient_penalty`.

> **▶ build: stage 2.** implement `gradient_penalty`.

### idea 3: spectral normalization

a third way to bound the Lipschitz constant: control each layer. a linear map $x \mapsto Wx$ has Lipschitz constant $\sigma(W)$, its largest singular value (the spectral norm), and ReLU and LeakyReLU are 1-Lipschitz. a composition's constant is at most the product of its parts' constants. so if every layer's weight is divided by its own spectral norm, $\bar W = W / \sigma(W)$, the whole network is at most 1-Lipschitz, with no penalty term and no extra backward pass.

computing $\sigma(W)$ exactly (an SVD) at every step would be expensive. **power iteration** gets it cheaply: keep a vector $u$ between steps and alternate

$$v \leftarrow \frac{W^\top u}{\|W^\top u\|}, \qquad u \leftarrow \frac{W v}{\|W v\|}, \qquad \sigma \approx u^\top W v.$$

each iteration multiplies the error by the squared ratio of the two largest singular values, $(\sigma_2/\sigma_1)^2$ (on paper 4); since $W$ changes only slightly per training step and $u$ is carried over, **one** iteration per step is enough (Miyato et al., 2018). $u$ and $v$ are treated as constants (no gradient flows through the iteration), but $\sigma = u^\top W v$ is differentiable in $W$, with $\partial\sigma/\partial W = u v^\top$, and that gradient is part of what makes SN train well. convolution weights $(C_\text{out}, C_\text{in}, k, k)$ are reshaped to $(C_\text{out}, C_\text{in}k^2)$ first (strictly, that's not the convolution's exact spectral norm, but it's the standard practical choice).

the product-of-layers bound is loose: a network whose every layer has norm 1 usually has a Lipschitz constant well below 1, because the directions that maximize one layer's gain are not the ones the next layer amplifies. SN-GANs pair it with the ordinary GAN losses as often as with Wasserstein ones.

> **spec · stage 3.**
> - `spectral_normalize(weight, u, n_power_iterations=1)`: with `W = weight.reshape(out, -1)`, under `no_grad`: `v = normalize(W^T u)`, then `n_power_iterations` times `v = normalize(W^T u)`, `u = normalize(W v)` (`F.normalize`); then `sigma = u · (W v)` with gradient through `W`; returns `(weight / sigma, u, sigma)` with `u` the updated (gradient-free) vector. this is `mlfs.generative.spectral_normalize`.
> - `SNLinear(in_features, out_features, bias=True, *, n_power_iterations=1)`: an `nn.Linear` with a buffer `u` (a random unit vector of size `out_features`); `forward` uses `spectral_normalize(weight, u, n_power_iterations)`'s weight and stores the new `u` in training mode only.

> **▶ build: stage 3.** implement `spectral_normalize` and `SNLinear`.

### idea 4: training a WGAN-GP

a Wasserstein critic is only useful if it's close to the supremum: the generator's gradient comes from the critic's slope, and a half-trained critic's slope points in the wrong directions. so WGANs train the critic for **`n_critic` = 5 steps per generator step**, each on a fresh real batch and fresh noise, and use Adam with a small $\beta_1$ (Gulrajani et al. used $\beta = (0.5, 0.9)$ or even $\beta_1 = 0$). and because the critic estimates a distance, its value is finally a useful training curve: $-$`critic_loss` should fall as the generator improves (the lab checks how well).

no batch norm in the critic: the gradient penalty is defined per sample, and batch norm makes each sample's score depend on the rest of the batch. layer norm or no normalization is the usual choice.

> **spec · stage 4.** `train_wgan(G, critic, sample_real, *, n_steps, latent_dim, n_critic=5, gp_lambda=10.0, lr=1e-4, betas=(0.5, 0.9), generator=None, callback=None)`: create `torch.optim.Adam(lr, betas)` for each network; for each of `n_steps` generator steps: `n_critic` times { `real = sample_real()`, `z = torch.randn((B, latent_dim), generator=generator)`, `fake = G(z).detach()`, loss `critic_loss(critic(real), critic(fake)) + gp_lambda * gradient_penalty(critic, real, fake, generator=generator)`, one critic step }, then one generator step on `wgan_generator_loss(critic(G(z)))` with a fresh `z`. scores may be `(B,)` or `(B, 1)`. returns a dict of lists with one entry per generator step: `'critic_loss'`, `'gp'` and `'wasserstein'` (= $-$`critic_loss`), from the last critic step. if `callback` is given, call `callback(step, history)` after each generator step (`step` = 1, 2, …), so you can monitor training.

> **▶ build: stage 4.** implement `train_wgan`.

> **🔮 predict:** you train three critics, with weight clipping, a gradient penalty and spectral normalization, to estimate the Wasserstein distance between two fixed distributions whose true distance is 1. will the three estimates agree?

## worked example

**a Wasserstein distance by hand.** real data: point masses of weight $\tfrac12$ at $0$ and $2$; generated: $\tfrac12$ at $1$ and $\tfrac12$ at $5$. the cheapest plan sends $1 \to 0$ (cost 1) and $5 \to 2$ (cost 3), total $\tfrac12 \cdot 1 + \tfrac12 \cdot 3 = 2$ (the crossed plan $1 \to 2$, $5 \to 0$ costs $\tfrac12 \cdot 1 + \tfrac12 \cdot 5 = 3$). the dual: $f(x) = -x$ is 1-Lipschitz and gives $\mathbb{E}_\text{real}[f] - \mathbb{E}_\text{gen}[f] = -1 - (-3) = 2$, matching: it's an optimal critic. the JS divergence between these two distributions is $\log 2$ (disjoint supports), the same as if the generated points were at $10^6$.

**the gradient penalty for a linear critic.** $f(x) = w^\top x$ with $w = (2, -1, 2)$: $\nabla f = w$ everywhere, $\|w\| = 3$, so the penalty is $(3 - 1)^2 = 4$ at every interpolate, and its gradient with respect to $w$ is $2(3 - 1)\,w/3 = (8/3, -4/3, 8/3)$, pulling the norm down towards 1.

**power iteration.** $W = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$, $u = (1, 1)/\sqrt 2$. $W^\top u = (3, 1)/\sqrt 2$, so $v = (3, 1)/\sqrt{10}$; $Wv = (9, 1)/\sqrt{10}$, so $u = (9, 1)/\sqrt{82}$; $\sigma = u^\top W v = (81 + 1)/\sqrt{820} \approx 2.86$, within 5% of the true $\sigma = 3$ after one iteration (and $2.998$ after two). the checks reproduce each formula.

## on paper

1. **Wasserstein versus JS for point masses.** compute $W_1(\delta_0, \delta_\theta)$ and $\operatorname{JS}(\delta_0 \| \delta_\theta)$ as functions of $\theta$, and the gradient each gives a generator that controls $\theta$.

<details><summary>solution</summary>

the only plan moves all mass from $\theta$ to 0: $W_1 = |\theta|$, gradient $\operatorname{sign}(\theta)$, pointing the generator straight at the data from any distance. $\operatorname{JS} = \log 2$ for every $\theta \ne 0$ and 0 at $\theta = 0$: flat, zero gradient everywhere except a discontinuity. a model trained on JS gets no signal until its samples already overlap the data.

</details>

2. **the optimal critic's slope.** in one dimension, $p = \delta_0$ and $q = \delta_2$. show that $f(x) = -x$ (any shift) is an optimal critic, that $|f'| = 1$ on the segment $[0, 2]$, and that a critic with $|f'| < 1$ on the segment gives a smaller value.

<details><summary>solution</summary>

$f(0) - f(2) = 2 = W_1$, and $|f(x) - f(y)| = |x - y|$, so it's 1-Lipschitz and attains the bound. for any 1-Lipschitz $f$, $f(0) - f(2) = -\int_0^2 f'(t)\,dt \le \int_0^2 |f'(t)|\,dt$, which equals 2 only if $|f'| = 1$ (with the right sign) almost everywhere on $[0, 2]$. so the optimal critic has unit slope along the transport path; a critic with slope $s < 1$ on the segment underestimates $W_1$ by the factor $s$. that's why the penalty targets norm 1 at interpolates, not just $\le 1$.

</details>

3. **clipping's Lipschitz bound.** a ReLU MLP $2 \to 256 \to 256 \to 1$ has every weight in $[-0.01, 0.01]$. bound its Lipschitz constant, using $\sigma(W) \le \|W\|_F$.

<details><summary>solution</summary>

$\|W\|_F \le 0.01\sqrt{mn}$: $0.01\sqrt{512} = 0.226$, $0.01\sqrt{65536} = 2.56$, $0.01\sqrt{256} = 0.16$. the product is at most $0.093$: the critic's slope can't exceed about 0.09, so its estimate of $W_1$ is at most a tenth of the truth (in practice much less: the clipped weights aren't aligned to reach the bound). the estimate is still roughly *proportional* to the distance, which is why the original WGAN trained at all.

</details>

4. **power iteration converges.** write $u_0 = \sum_i a_i u_i$ in the basis of left singular vectors of $W$ (singular values $\sigma_1 > \sigma_2 \ge \dots$). show that after $k$ iterations $u_k \propto \sum_i a_i \sigma_i^{2k} u_i$, and conclude how fast $\sigma$'s estimate converges.

<details><summary>solution</summary>

one iteration maps $u \mapsto WW^\top u / \|\cdot\|$, and $WW^\top u_i = \sigma_i^2 u_i$. so $u_k \propto \sum_i a_i \sigma_i^{2k} u_i = \sigma_1^{2k}(a_1 u_1 + \sum_{i > 1} a_i (\sigma_i/\sigma_1)^{2k} u_i)$: the other components shrink like $(\sigma_2/\sigma_1)^{2k}$. the error in $\sigma$ shrinks at least that fast (a Rayleigh-quotient-like estimate, quadratically better). when $W$ changes slowly between training steps, the carried-over $u$ starts close to $u_1$, so one iteration per step keeps up.

</details>

## experiment

`lab.py` puts the three Lipschitz methods and the WGAN-GP loop to work on 8 gaussians:

1. three critics estimating a known Wasserstein distance.
2. the gradient each generator loss gives a sample, as the samples move away from the data.
3. a GAN and a WGAN-GP trained on the same data, three seeds each: coverage, sharpness, and the critic's estimate as a loss curve.

## check your understanding

1. why does the Wasserstein generator loss have no logarithm or sigmoid?

<details><summary>answer</summary>

the critic outputs a score, not a probability: $W_1$'s dual is a plain difference of expected scores. so the generator just raises the expected score of its samples, and its gradient on each score is a constant $-1/B$; it never saturates. the critic's Lipschitz constraint, not a sigmoid, keeps the scores bounded relative to distances.

</details>

2. why is the gradient penalty evaluated at interpolates between real and generated samples, rather than at real or generated samples only?

<details><summary>answer</summary>

the constraint matters where the critic's slope drives the generator: along the paths from generated to real data, where the optimal critic has unit gradient. enforcing it everywhere is impossible with samples; enforcing it only at the data would leave the region in between, which the generator has to cross, unconstrained.

</details>

3. a WGAN critic with weight clipping estimates $W_1 = 0.02$ when the true value is 1. is the WGAN broken?

<details><summary>answer</summary>

not necessarily: clipping makes the critic $K$-Lipschitz for some small unknown $K$, so its estimate is roughly $K \cdot W_1$. the generator's gradient still points the right way (scaled by $K$, which Adam's normalization partly undoes). but the estimate isn't a distance you can report, and the bias towards simple critics hurts sample quality, which is what the gradient penalty fixed.

</details>

4. why must the critic be trained for several steps per generator step?

<details><summary>answer</summary>

the generator's gradient is the slope of the critic at its samples, and the dual formula is only the Wasserstein distance at the *optimal* critic. a stale or under-trained critic has slopes that don't point along transport paths. 5 critic steps per generator step keep it near its optimum as the generator moves; unlike the original GAN, a strong critic doesn't starve the generator of gradients, so there's no reason to hold it back.

</details>

5. why is batch norm avoided in a WGAN-GP critic?

<details><summary>answer</summary>

the gradient penalty constrains each sample's gradient $\nabla_{\hat x} f(\hat x)$, which assumes $f(\hat x)$ depends on $\hat x$ alone. with batch norm, each output depends on the whole batch's statistics, so the per-sample gradient mixes in other samples and the penalty no longer constrains what it's supposed to.

</details>

6. spectral normalization makes every layer 1-Lipschitz. why might an SN critic still underestimate $W_1$?

<details><summary>answer</summary>

the network's Lipschitz constant is at most the product of the layers' constants, and usually much less, because each layer's maximal-gain direction isn't aligned with the next layer's. so the SN critic's slope along transport paths tends to be well below 1, and its estimate of the dual objective is well below $W_1$. for an ordinary (non-Wasserstein) GAN loss that doesn't matter: SN is used there as a regularizer of the discriminator's smoothness.

</details>

## common bugs

- **`autograd.grad` without `create_graph=True`.** symptom: the penalty's value is right, but `backward()` doesn't reach the critic's parameters; the constraint has no effect and the critic's gradients blow up.
- **one $\epsilon$ per element instead of per example.** symptom: the interpolates aren't on lines between samples; the penalty constrains the wrong points.
- **a one-sided or squared-norm penalty.** symptom: $(\|g\|^2 - 1)^2$ or $\max(0, \|g\| - 1)^2$ in place of $(\|g\| - 1)^2$; the checks' closed forms disagree.
- **power iteration with gradients.** symptom: backpropagating through $u$ and $v$ makes training slow and unstable; compute them under `no_grad`.
- **updating $u$ in eval mode.** symptom: evaluating the model changes it; two evaluations of the same input give different outputs.
- **sigmoid on the critic.** symptom: a "WGAN" whose critic outputs probabilities; the Wasserstein estimate is bounded by 1 and saturates like a GAN.

## going deeper

- Arjovsky, Chintala & Bottou (2017), "Wasserstein generative adversarial networks", *ICML*. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
- Arjovsky & Bottou (2017), "Towards principled methods for training generative adversarial networks", *ICLR*. disjoint supports and vanishing gradients. [arXiv:1701.04862](https://arxiv.org/abs/1701.04862)
- Gulrajani, Ahmed, Arjovsky, Dumoulin & Courville (2017), "Improved training of Wasserstein GANs", *NeurIPS*. WGAN-GP. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
- Miyato, Kataoka, Koyama & Yoshida (2018), "Spectral normalization for generative adversarial networks", *ICLR*. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
- Villani (2009), *Optimal Transport: Old and New*, Springer, chapter 5 (Kantorovich duality). [doi:10.1007/978-3-540-71050-9](https://doi.org/10.1007/978-3-540-71050-9)
- Mescheder, Geiger & Nowozin (2018), "Which training methods for GANs do actually converge?", *ICML*. gradient penalties and convergence. [arXiv:1801.04406](https://arxiv.org/abs/1801.04406)

## connections

- ← builds on: the GAN game (topic 05), Lipschitz continuity (ch 03), singular values and power iteration (ch 02), duality (ch 06), higher-order autograd (ch 15).
- → used in: the capstone's adversarial family, optimal transport (elective 15–17), adversarial losses in neural audio codecs (audio elective), and Lipschitz constraints for robustness (ch 28).
