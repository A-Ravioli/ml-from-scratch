# non-contrastive methods

## why this matters

SimCLR needs negatives: without them, "make the two views of an image agree" has a trivial solution, an encoder that ignores its input and outputs the same vector for every image. every pair agrees perfectly; the loss is at its minimum; the representation is worthless. negatives prevent this by also demanding that different images *disagree*. but negatives are expensive (they need big batches or a memory queue) and they're conceptually awkward: two photos of different cats count as a pair that should be pushed apart.

in 2020 two papers showed that you can drop the negatives entirely. BYOL (Grill et al.) trains an online network to predict the output of a slowly moving copy of itself. SimSiam (Chen & He) showed that even the slow copy isn't needed: two identical branches, a small **predictor** on one side, and a **stop-gradient** on the other are enough. both match or beat SimCLR. nobody predicted this would work, and the reason it does is still a research topic. in this topic you build both, learn to *measure* collapse rather than guess at it, and in the lab you remove the stop-gradient and watch the representation fold up while the loss reports perfect success.

## warm-up: predict first

1. a network outputs the same unit vector for every input. what's its SimSiam loss (negative cosine between the views)? is that good or bad for the loss?
2. how would you detect that a batch of 256 embeddings has collapsed, without looking at any downstream accuracy?
3. BYOL's target network is an exponential moving average of the online network with momentum 0.99. roughly how many steps back does it "remember"?
4. can embeddings collapse *partially*, so that they vary, but only along a few directions?
5. the SimSiam loss with a stop-gradient is not the gradient of any single function of the weights. what could "optimising" it then mean?

## prerequisites check

- [ ] you have `SimCLRAugment` and the two-view training setup (→ topic 02)
- [ ] you know the singular value decomposition and what rank means (→ ch 02 · topic 06)
- [ ] you know entropy (→ ch 05)
- [ ] you can use `.detach()` and know how gradients flow through a computation graph (→ ch 14, ch 15)
- [ ] you have `ema_update` and know an exponential moving average (→ ch 15 · topic 07, ch 24 · topic 01)

## core ideas

### idea 1: agreement without negatives, and the stop-gradient

the non-contrastive recipe: two augmented views $x_1, x_2$ of an image go through the same encoder $f$ and projector $g$, giving $z_1 = g(f(x_1))$ and $z_2 = g(f(x_2))$. a small MLP, the **predictor** $q$, maps one view's projection to a guess of the other's: $p_1 = q(z_1)$. the loss compares the guess with the target by cosine similarity, symmetrised over the two views:

$$\mathcal L_\text{SimSiam} = -\tfrac12\big[\cos(p_1, \operatorname{sg}(z_2)) + \cos(p_2, \operatorname{sg}(z_1))\big], \qquad \cos(a, b) = \frac{a^\top b}{\|a\|\|b\|},$$

averaged over the batch. $\operatorname{sg}$ is the **stop-gradient**: in the forward pass it's the identity, in the backward pass it's a wall (`.detach()` in PyTorch). the target side receives no gradient from this term; only the prediction side does. its minimum value is $-1$.

BYOL writes the same idea as a squared error between normalised vectors: $\|\bar p - \bar z\|^2 = 2 - 2\cos(p, z)$ for unit vectors $\bar p, \bar z$, with its target $z$ coming from a separate **target network** (idea 4) and also detached:

$$\mathcal L_\text{BYOL}(p, z) = \frac1B\sum_{i}\big(2 - 2\cos(p_i, \operatorname{sg}(z_i))\big).$$

why the stop-gradient matters: without it, the gradient pulls *both* sides towards each other, and the fastest way to make every pair agree is to make every output the same (on paper 1). with it, each step moves the prediction towards a target that is held fixed for that step. Chen & He interpret this as alternating optimisation, like k-means or EM: one sub-problem sets the targets (the "representation of each image", averaged over augmentations), the other trains the network to predict them; with the target held fixed, a constant output is no longer an easy win. the predictor is essential too: in the lab, removing it collapses the projections even with the stop-gradient in place. why exactly these two ingredients suffice is an open question (Tian, Chen & Ganguli, 2021, analyse the linear case).

> **spec.** `simsiam_loss(p1, p2, z1, z2)`: tensors `(B, d)`; returns the scalar $-\frac12[\operatorname{mean}_i\cos(p_{1,i}, z_{2,i}) + \operatorname{mean}_i\cos(p_{2,i}, z_{1,i})]$ with `z1` and `z2` detached. `byol_loss(p, z)`: returns the scalar $\operatorname{mean}_i(2 - 2\cos(p_i, z_i))$ with `z` detached. neither may send any gradient to its `z` arguments.

> **▶ build: stage 1.** implement `simsiam_loss` and `byol_loss`.

### idea 2: measuring collapse

a collapsed representation can't be seen in the loss (it's at its best) or in the loss curve (it looks like fast convergence). you need to measure the embeddings directly. two measures, for a batch $Z \in \mathbb R^{n \times d}$:

**per-dimension std of the normalised embeddings** (the SimSiam paper's monitor). L2-normalise each row, compute each dimension's standard deviation over the batch, and average over dimensions. complete collapse (every row the same) gives 0. if the rows were spread uniformly over the unit sphere, each coordinate would have variance $1/d$, so the std sits near $1/\sqrt d$: about 0.125 for $d = 64$. it's cheap enough to log every step.

**effective rank.** collapse can also be *dimensional*: the embeddings vary, but only within a low-dimensional subspace, so most of the $d$ dimensions carry nothing (Jing et al., 2022). the std above barely notices if the few remaining directions are well spread. the singular values $\sigma_1 \ge \dots \ge \sigma_d$ of $Z$ do: normalise them into a distribution $p_k = \sigma_k/\sum_j\sigma_j$ and take the exponential of its entropy,

$$\operatorname{erank}(Z) = \exp\Big(-\sum_k p_k\log p_k\Big).$$

this is Roy & Vetterli's (2007) effective rank, used for SSL monitoring as "RankMe" (Garrido et al., 2023). it's 1 for a rank-one matrix and $k$ for $k$ equal nonzero singular values, and it varies smoothly in between (on paper 3). like RankMe, we apply it to the raw embeddings, without centring or normalising the rows.

> **spec.** `collapse_metrics(z)`: `z` a tensor `(n, d)` (it may require grad; don't build a graph). returns a dict of Python floats: `'std'`, the mean over dimensions of the population standard deviation (divide by $n$) of the L2-normalised rows; and `'effective_rank'`, $\exp(-\sum_kp_k\log p_k)$ with $p_k = \sigma_k/\sum_j\sigma_j$ from the singular values of the raw `z` (add $10^{-12}$ to each $p_k$ before the log so zero singular values are harmless). compute in float64 on the CPU.

> **🔮 predict:** an untrained CNN's 64 features, on 5,000 MNIST digits: is their effective rank close to 64? (the lab prints it.)

> **▶ build: stage 2.** implement `collapse_metrics`.

### idea 3: SimSiam

SimSiam is the minimal version, and you'll build it with a switch to remove its key ingredient. the encoder is any network producing `feature_dim` features. the **projector** is a two-layer MLP with BatchNorm on its output: Linear → BN → ReLU → Linear → BN (`feature_dim` → `proj_dim` → `proj_dim`; the paper's CIFAR version uses two layers, the ImageNet one three). the **predictor** is a bottleneck MLP, Linear → BN → ReLU → Linear (`proj_dim` → `pred_dim` → `proj_dim`; the paper uses $2048 \to 512 \to 2048$).

both views go through the same encoder, projector and predictor (the network is "Siamese": two branches, shared weights). the loss is `simsiam_loss(p1, p2, z1, z2)`. the switch, `stop_gradient=False`, computes the same expression without detaching anything, so you can reproduce the paper's most famous ablation.

> **spec.** `SimSiam(encoder, feature_dim, *, proj_dim=128, pred_dim=32, stop_gradient=True)` is an `nn.Module` with attributes `encoder` (the module you pass in), `projector = nn.Sequential(Linear(feature_dim, proj_dim, bias=False), BatchNorm1d(proj_dim), ReLU(), Linear(proj_dim, proj_dim, bias=False), BatchNorm1d(proj_dim))`, `predictor = nn.Sequential(Linear(proj_dim, pred_dim, bias=False), BatchNorm1d(pred_dim), ReLU(), Linear(pred_dim, proj_dim))` and `stop_gradient`. `forward(x1, x2)` returns `(loss, z1)`: the SimSiam loss of the two views (through `simsiam_loss` when `stop_gradient` is true; otherwise the identical expression with nothing detached), and view 1's projection $z_1$ (for monitoring).

> **▶ build: stage 3.** implement `SimSiam`.

### idea 4: BYOL, and targets that move slowly

BYOL came first, and uses an extra stabiliser. there are two networks. the **online** network (encoder, projector, predictor) is trained by gradient descent. the **target** network (encoder and projector only, no predictor) is never trained directly; after every step its weights move a little towards the online network's:

$$\xi \leftarrow m\,\xi + (1 - m)\,\theta,$$

with momentum $m$ around 0.99–0.996. the online network predicts the target network's projection of the *other* view: $\mathcal L = \mathcal L_\text{BYOL}(p_1, z'_2) + \mathcal L_\text{BYOL}(p_2, z'_1)$, where $z'$ comes from the target. it's the same exponential moving average you used for DQN's soft target updates (ch 24 · topic 01) and for diffusion models' weight averaging: a target that changes slowly is a stable thing to regress towards. averaging over the last $\sim 1/(1 - m)$ steps (on paper 2) also smooths away the noise of individual updates.

BYOL's own ablation reported collapse at $m = 0$. SimSiam then showed that a stop-gradient copy of the *current* network, with a predictor, trains fine: the momentum is a stabiliser that helps, not the thing that prevents collapse. in the lab you can try both. **DINO** (Caron et al., 2021) is a third member of the family: a momentum teacher and a student, but with softmax outputs over many prototypes and a cross-entropy loss; it avoids collapse by **centring** the teacher's outputs (subtracting a running mean, so no single prototype can win every time) and **sharpening** them (a low temperature, so the outputs can't become uniform). no negatives, no predictor.

the target network's BatchNorm running statistics are copied from the online network rather than averaged (that's what `ema_update` does with buffers).

> **spec.** `BYOL(encoder, feature_dim, *, proj_dim=128, hidden_dim=256, momentum=0.99)` is an `nn.Module` with `encoder` (yours), `projector = Sequential(Linear(feature_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, proj_dim))`, `predictor` of the same form (`proj_dim` → `hidden_dim` → `proj_dim`), `target_encoder` and `target_projector` as `copy.deepcopy`s of the online ones with `requires_grad=False`, and `momentum`. `forward(x1, x2)` returns `(byol_loss(p1, t2) + byol_loss(p2, t1), z1)` with $t = $ target projector(target encoder(x)) computed without grad. `update_target()` applies `ema_update(target, online, decay=momentum)` to the encoder and the projector pairs.

> **▶ build: stage 4.** implement `BYOL`.

### idea 5: training, and what to watch

the loop is topic 02's with a different model: two augmentations per batch, `loss, z1 = model(v1, v2)`, a gradient step, and, for BYOL, `update_target()` right *after* the optimiser step (so the target follows the weights the step produced). because collapse is silent in the loss, log a collapse metric every step: the std of the normalised $z_1$ is cheap enough.

what to watch for: the loss racing to its minimum ($-1$ for SimSiam, $0$ for BYOL) while the std of $z$ heads to 0 is complete collapse. a healthy SimSiam run plateaus well above $-1$ with the std near $1/\sqrt{d}$.

> **spec.** `train_ssl(model, images, *, epochs=1, batch_size=256, lr=1e-3, strength=0.5, seed=0, device='cpu')` trains any model with `model(x1, x2) -> (loss, z1)` on `images` `(N, C, H, W)` in $[0, 1]$. `torch.manual_seed(seed)` first; move the model to `device`; Adam (`lr`) over the parameters that require grad; `SimCLRAugment(H, strength=strength)` for the views. each epoch: `model.train()`, a fresh permutation from `np.random.default_rng(seed)` (one generator per run), `N // bs` full batches with `bs = min(batch_size, N)`; per batch, step on `model(augment(xb), augment(xb))`'s loss, then call `model.update_target()` if the model has one. returns `{'loss': [...], 'z_std': [...]}`, one entry per step, with `z_std = collapse_metrics(z1)['std']`.

> **🔮 predict:** you train SimSiam with `stop_gradient=False`. sketch the loss curve and the std-of-$z$ curve over 150 steps.

> **▶ build: stage 5.** implement `train_ssl`.

## worked example

**the losses.** a batch of two: $p_1 = ((1, 0), (0, 2))$, $p_2 = ((1, 1), (3, 0))$, $z_1 = ((2, 0), (1, 1))$, $z_2 = ((1, 1), (0, -1))$. then $\cos(p_1, z_2) = (1/\sqrt2, -1)$, with batch mean $-0.1464$, and $\cos(p_2, z_1) = (1/\sqrt2, 1/\sqrt2)$, with mean $0.7071$. so $\mathcal L_\text{SimSiam} = -\frac12(-0.1464 + 0.7071) = -0.2803$. BYOL's loss of $p_1$ against $z_2$ is $\frac12[(2 - 2/\sqrt2) + (2 + 2)] = 2.2929$: the second pair points in exactly opposite directions, which costs the maximum, 4.

**the EMA.** a target weight of 1.0, an online weight of 3.0 and $m = 0.9$: after `update_target()` the target is $0.9 \cdot 1.0 + 0.1 \cdot 3.0 = 1.2$. with $m = 0$ it would jump to 3.0; with $m = 1$ it would never move.

**the metrics.** rows $(1, 0)$ and $(0, 1)$, half each: already unit length; each coordinate is 0 half the time and 1 half the time, so its population std is 0.5, and `std` = 0.5; the two singular values are equal, so the effective rank is 2. the matrix with rows $(3, 0)$ and $(0, 1)$ has singular values 3 and 1, so $p = (0.75, 0.25)$, entropy $0.5623$ nats, effective rank $e^{0.5623} = 1.755$: "between one and two directions".

your code must reproduce these numbers.

## on paper

1. **why the trivial solution wins without a stop-gradient.** show that any network whose output $p = z = c$ is constant achieves $\mathcal L_\text{SimSiam} = -1$, the global minimum. then, for the loss without stop-gradient, write the gradient with respect to $z_2$ and explain why gradient descent makes $z_1$ and $z_2$ approach each other *from both sides*, and why that makes collapse an attractor.

<details><summary>solution</summary>

$\cos(c, c) = 1$ for every pair, so the loss is $-\frac12(1 + 1) = -1$, and since $\cos \le 1$ that's the minimum. without the stop-gradient, the term $-\cos(p_1, z_2)$ has gradient with respect to $z_2$ equal to $-\frac{1}{\|z_2\|}(\bar p_1 - (\bar p_1^\top\bar z_2)\bar z_2)$: it rotates $z_2$ towards $p_1$, while the gradient with respect to $p_1$ rotates $p_1$ towards $z_2$. every pair of views is pulled together from both ends, for every image; the cheapest way to satisfy all pairs at once is to ignore the input, and nothing in the loss pushes back. with the stop-gradient, $z_2$ is a constant in each step: the network can only move $p_1$, so it must learn to *predict* $z_2$ from $x_1$, and $z_2$ itself changes only as a side effect of updating the shared weights.

</details>

2. **the EMA's memory.** unroll $\xi_t = m\xi_{t-1} + (1 - m)\theta_t$ to write $\xi_t$ as a weighted sum of past online weights. show the weights sum to $1 - m^{t}$ (plus the initial value's share $m^t$) and that their mean age is about $m/(1 - m)$ steps. evaluate for $m = 0.99$ and $m = 0.996$.

<details><summary>solution</summary>

$\xi_t = m^t\xi_0 + (1 - m)\sum_{k=0}^{t-1}m^k\theta_{t-k}$. the weights $(1 - m)m^k$ sum to $(1 - m)\frac{1 - m^t}{1 - m} = 1 - m^t$. as $t \to \infty$ the mean age is $\sum_k k(1 - m)m^k = m/(1 - m)$: 99 steps for $m = 0.99$, 249 for $m = 0.996$. so BYOL's target lags the online network by a couple of hundred steps; with the lab's 156 steps and $m = 0.99$, the target is still noticeably pulled towards the initialisation ($0.99^{156} \approx 0.21$ of its weight).

</details>

3. **effective rank.** show that $1 \le \operatorname{erank}(Z) \le \operatorname{rank}(Z)$, with equality on the right exactly when the nonzero singular values are all equal. what is the effective rank of a matrix with singular values $(1, 1, \epsilon)$ as $\epsilon \to 0$?

<details><summary>solution</summary>

the entropy of a distribution over $r$ outcomes with nonzero mass is between 0 (all mass on one) and $\log r$ (uniform), so $\exp(H)$ is between 1 and $r = \operatorname{rank}(Z)$, the upper bound reached exactly for the uniform distribution, i.e. equal singular values. for $(1, 1, \epsilon)$, $p \to (\frac12, \frac12, 0)$ and the effective rank tends continuously to 2, unlike the rank, which jumps from 3 to 2 only when $\epsilon$ is exactly 0. that continuity is what makes it a useful monitor of *partial* (dimensional) collapse.

</details>

## experiment

`lab.py` investigates, with your code, on unlabelled MNIST:

1. SimSiam with and without the stop-gradient (two seeds each): the loss, the collapse metric over training, kNN accuracy and the effective rank of the encoder's features.
2. one more ingredient of your choice: removing the predictor, or BYOL with and without its momentum target.

## check your understanding

1. why can't you detect collapse from the loss curve?

<details><summary>answer</summary>

collapse *is* the loss's global minimum: a collapsed run shows a fast, smooth decrease to $-1$ (or 0 for BYOL), which looks like successful training. you need a metric of the embeddings themselves (std, rank) or a downstream evaluation.

</details>

2. SimCLR doesn't need a stop-gradient or a predictor. why not?

<details><summary>answer</summary>

its negatives make collapse expensive: if every image maps to the same point, every negative is as similar as the positive, and the loss sits at $\log(2B - 1)$, far from its minimum. the uniformity pressure comes from the loss itself.

</details>

3. the std of normalised $z$ is 0.12 (healthy, for $d = 64$), but the effective rank of $z$ is 3. what's going on?

<details><summary>answer</summary>

dimensional collapse: the embeddings are spread out, but within a three-dimensional subspace. the per-dimension std can stay sizeable if those few directions are mixed into every coordinate. the representation is using 3 of its 64 dimensions.

</details>

4. why does BYOL's target network not get a predictor?

<details><summary>answer</summary>

the target provides the regression target $z'$; only the online branch has to *predict*. the asymmetry (predictor on one side, stop-gradient and averaging on the other) is what keeps the two branches from simply agreeing on a constant.

</details>

5. in the lab, removing the predictor collapses $z$, but the encoder's kNN accuracy stays above a random network's (0.80 against 0.77, one seed). how can both be true?

<details><summary>answer</summary>

collapse can happen in the projector alone: its final BatchNorm's scale parameters can shrink so every $z$ becomes (nearly) the same vector, while the encoder's features $h$, earlier in the network, still carry what was learned before the projector stopped passing gradient information through. monitoring only $z$ overstates the damage to $h$; monitoring only downstream accuracy hides the failure of the method.

</details>

6. why update BYOL's target after the optimiser step rather than before?

<details><summary>answer</summary>

so the target follows the weights that the step produced. updating before the step makes the target trail by one extra step; with $m = 0$ it means the target is the *previous* network rather than the current one, which is not what SimSiam's $m = 0$ setting compares against.

</details>

## common bugs

- **forgetting `.detach()` on the target.** training "works" (the loss falls fast) and the representation collapses.
- **comparing each prediction with its own view's projection** ($p_1$ with $z_1$): the predictor learns the identity and nothing is learned about invariance.
- **target network sharing parameters with the online one** (no `deepcopy`): the EMA is a no-op and the "target" receives gradients.
- **target parameters left with `requires_grad=True`:** the optimiser may update them, and gradients may flow into them.
- **monitoring only the loss:** collapse looks like convergence.
- **effective rank from squared singular values or from centred embeddings:** a different number from the spec's (and from RankMe), with a different meaning.

## going deeper

- Grill et al. (2020), *Bootstrap your own latent: a new approach to self-supervised learning* (BYOL). [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- Chen & He (2021), *Exploring simple Siamese representation learning* (SimSiam). [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)
- Caron et al. (2021), *Emerging properties in self-supervised vision transformers* (DINO). [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Tian, Chen & Ganguli (2021), *Understanding self-supervised learning dynamics without contrastive pairs*. [arXiv:2102.06810](https://arxiv.org/abs/2102.06810)
- Jing, Vincent, LeCun & Tian (2022), *Understanding dimensional collapse in contrastive self-supervised learning*. [arXiv:2110.09348](https://arxiv.org/abs/2110.09348)
- Garrido, Balestriero, Najman & LeCun (2023), *RankMe: assessing the downstream performance of pretrained self-supervised representations by their rank*. [arXiv:2210.02885](https://arxiv.org/abs/2210.02885)
- Roy & Vetterli (2007), *The effective rank: a measure of effective dimensionality*. [EUSIPCO 2007](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf)
- Bardes, Ponce & LeCun (2022), *VICReg*: preventing collapse with explicit variance and covariance terms instead. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)

## connections

- ← builds on: two-view training and augmentations (topic 02), the evaluation harness (topic 01), EMA target networks (ch 24 · topic 01, `mlfs.torchnn.ema_update`), SVD (ch 02).
- → used in: masked modeling (topic 04) avoids collapse differently, by reconstructing pixels; the capstone can swap SimCLR for your SimSiam as a stretch goal; effective rank is a general diagnostic for any representation, including the ones you'll probe in ch 28 · topic 01.
