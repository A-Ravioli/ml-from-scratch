# contrastive learning

## why this matters

labels are expensive; images are free. a self-supervised method needs a training signal that comes from the images themselves, and the most successful idea of the last decade is almost embarrassingly simple: take a photo, make two randomly distorted copies of it (crop it differently, flip it, change its colours), and train a network so the two copies get similar embeddings, while copies of *different* photos get dissimilar ones. nobody tells the network what's in the picture. to recognise that two heavily cropped, recoloured views come from the same image, it has to pick up on what survives the distortions: shapes, parts, textures, objects.

this is **contrastive learning**, and SimCLR (Chen et al., 2020) is its cleanest form: an encoder, a small projection head, a batch of augmented pairs, and one loss, **InfoNCE**. with a 4× wider ResNet-50 and a lot of compute, a linear probe on SimCLR features matched the accuracy of a standard supervised ResNet-50 trained with all ImageNet labels. in this topic you build all four pieces, and in the lab you find out why the loss value on its own is uninterpretable, and what the temperature in that loss does.

## warm-up: predict first

1. a batch has $B = 256$ images, so $2B = 512$ views. for one view, how many candidates must the loss pick its partner out from, and what's the loss of an encoder that guesses at random?
2. why compare *cosine* similarities rather than dot products?
3. the colour-jitter augmentation is turned off. what shortcut could the network find to match two crops of the same photo without learning anything about objects?
4. SimCLR computes the loss on $z = g(h)$, the output of a projection head, but evaluates $h$. why not just put the loss on $h$?
5. temperature $T = 0.05$ or $T = 1$: which one makes the loss care most about the single most confusing negative?

## prerequisites check

- [ ] you know softmax cross-entropy and its gradient, including logits with $-\infty$ entries (→ ch 08 · topic 05, ch 14)
- [ ] you know data augmentation as a regulariser (→ ch 15 · topic 03)
- [ ] you can batch per-sample operations without loops, including `F.grid_sample` style resampling (→ ch 16)
- [ ] you have `extract_features`, `knn_eval` and `linear_probe` (→ topic 01)

## core ideas

### idea 1: InfoNCE, picking your partner out of a crowd

take a batch of $B$ images and augment each one twice, giving $2B$ views; encode them into embeddings $z_1, \dots, z_{2B}$, where views $i$ and $i + B$ (mod $2B$) come from the same image. for each view $i$, its partner $p(i)$ is the **positive**, and the other $2B - 2$ views are **negatives**. the loss turns "recognise your partner" into a classification problem with $2B - 1$ choices:

$$\ell_i = -\log\frac{\exp(s_{i,p(i)}/T)}{\sum_{j \ne i}\exp(s_{ij}/T)}, \qquad s_{ij} = \frac{z_i^\top z_j}{\|z_i\|\,\|z_j\|}, \qquad \mathcal L = \frac{1}{2B}\sum_{i=1}^{2B}\ell_i.$$

this is cross-entropy with logits $s_{ij}/T$ and target $p(i)$, over every view in the batch except $i$ itself (its similarity with itself is always 1, a free answer the loss must not count). SimCLR calls it **NT-Xent** (normalised, temperature-scaled cross-entropy); it's an instance of **InfoNCE** (van den Oord et al., 2018), whose name comes from a classical idea, noise-contrastive estimation, and from a bound: minimising it maximises a lower bound on the mutual information between the two views, $I \ge \log(2B - 1) - \mathcal L$ (on paper 3).

three design choices are packed in. **cosine similarity** puts every embedding on the unit sphere, so the network can't satisfy the loss by inflating norms. the **temperature** $T$ sets how peaked the softmax is; similarities live in $[-1, 1]$, so without it ($T = 1$) the logits span only 2 units and the loss can never be small. and **every other view in the batch is a negative**, including the other view of the same side ($z_1[j]$ for $z_1[i]$), so a batch of $B$ supplies $2B - 2$ negatives for free.

in code: concatenate the two sides into a $(2B, d)$ matrix, normalise rows, take all pairwise similarities divided by $T$, set the diagonal to $-\infty$, and call cross-entropy with targets $(i + B) \bmod 2B$. no loops.

> **spec.** `info_nce_loss(z1, z2, *, temperature=0.1)`: `z1`, `z2` are `(B, d)` tensors where `z1[i]` and `z2[i]` are two views of image $i$. returns the scalar tensor $\mathcal L$ above over all $2B$ views (cosine similarity, self-similarities excluded, mean over views). differentiable; with $B = 1$ the loss is 0.

> **▶ build: stage 1.** implement `info_nce_loss`.

### idea 2: augmentations define what is learned

the loss only says "views of the same image should match". *which* views you generate decides what the network must become invariant to, and so what it learns. Chen et al. ran the ablation: random crops alone, or colour distortion alone, give poor representations; the **composition** of random crop and colour distortion is what works, and colour distortion has to be strong.

the reason is shortcuts. two crops of the same photo share their colour histogram almost exactly; without colour jitter, a network can match them by comparing histograms and never look at shape. randomising brightness, contrast and saturation (and sometimes dropping colour entirely) removes that shortcut. random crops do the opposite job: they create pairs that show *different parts* of the scene (a dog's head and its tail), or the whole and a zoomed detail, so matching them requires knowing that these parts go together.

SimCLR's pipeline, which you'll build for batches of tensors (each image gets its own random draw):

1. **random resized crop:** pick an area fraction uniformly in $[0.2, 1]$ (SimCLR uses $[0.08, 1]$ on ImageNet; $[0.2, 1]$ is the usual choice for $32 \times 32$ images) and an aspect ratio log-uniform in $[3/4, 4/3]$; width $= \sqrt{\text{area}\cdot\text{ratio}}$ and height $= \sqrt{\text{area}/\text{ratio}}$ as fractions of the image (clipped at 1); place the box uniformly at random inside the image, and resize it to `size` × `size` with bilinear interpolation. one call to `F.affine_grid` + `F.grid_sample` does the whole batch.
2. **horizontal flip** with probability 0.5 (fold it into the same affine transform).
3. **colour jitter** with probability 0.8: brightness (multiply by $b$), contrast (blend with the image's mean gray level: $(x - \bar g)c + \bar g$), saturation (blend with each pixel's gray value: $(x - g)\sigma + g$), in that order, each factor uniform in $[1 - 0.8s, 1 + 0.8s]$ for strength $s$, clamping to $[0, 1]$ after each step. gray is the luma $0.299R + 0.587G + 0.114B$. (SimCLR also jitters hue by $0.2s$; we leave it out, since hue rotation is fiddly and adds little at this scale.)
4. **grayscale** with probability 0.2: replace every channel by the luma.

> **spec.** `SimCLRAugment(size=32, *, strength=0.5)`; calling it on `x` `(B, C, H, W)` (floats in $[0, 1]$, $C \in \{1, 3\}$) returns `(B, C, size, size)` in $[0, 1]$, same dtype and device, with steps 1–4 above drawn independently per image from torch's global generator (so `torch.manual_seed` makes it reproducible). colour steps leave single-channel images' channel count alone. calling it twice gives two independent views.

> **🔮 predict:** a constant-colour image (every pixel $(0.3, 0.4, 0.5)$) goes through the pipeline. which steps can change it, and what stays exactly the same?

> **▶ build: stage 2.** implement `SimCLRAugment`.

### idea 3: the projection head

SimCLR doesn't apply the loss to the encoder's output $h = f(x)$. it adds a small MLP, the **projection head** $z = g(h)$, computes the loss on $z$, and throws $g$ away after training. the ablation in the paper is striking: a linear probe on $h$ beats a linear probe on $z$ by more than 10 points, and using a nonlinear head beats using none by about as much.

the explanation: the loss *demands* invariance to the augmentations. the last layers before the loss give in to that demand completely, discarding colour, orientation and position, because the loss punishes any difference between views. some of that information is useful downstream (is the bird red? which way is the car facing?). with a head in place, the head absorbs the invariance and $h$, one step removed, keeps more of the information. you'll see in topic 03 that the head (and a second one, the predictor) matters for other reasons too.

the head is an MLP: Linear → BatchNorm → ReLU → Linear, typically $2048 \to 2048 \to 128$ for a ResNet-50, and we'll default to a hidden width of 512 and an output of 128.

> **spec.** `ProjectionHead(in_dim, hidden_dim=512, out_dim=128)` is an `nn.Module` holding `self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))`; `forward(x)` maps `(B, in_dim)` to `(B, out_dim)`.

> **▶ build: stage 3.** implement `ProjectionHead`.

### idea 4: SimCLR, and why batch size and temperature matter

the training loop: sample a batch, augment it twice, encode both views, project, compute InfoNCE, step. there are no labels anywhere, and no special tricks: the encoder and head are trained jointly with one optimiser (SimCLR used LARS with huge learning rates; Adam is fine at our scale).

two hyperparameters dominate.

**batch size** sets the number of negatives. more negatives make the classification problem harder and the mutual-information bound looser at the top ($\log(2B - 1)$ grows), and SimCLR's accuracy kept improving up to batches of 8,192 (with 100 epochs; the advantage shrinks with longer training). it also changes the *scale* of the loss: an untrained encoder scores about $\log(2B - 1)$, so a loss value means nothing without the batch size next to it. at a fixed number of epochs, small batches also take more steps, which can matter more on small data.

**temperature** controls how the loss distributes its attention over negatives. the gradient of $\ell_i$ with respect to a negative's similarity $s_{ij}$ is $p_{ij}/T$, where $p_{ij}$ is that negative's softmax probability (on paper 2). at small $T$ the softmax is peaked, so almost all the push goes to the few negatives most similar to the anchor, the *hard negatives*. at large $T$ every negative is pushed about equally and the loss can't get low even with perfect positives, because similarities in $[-1, 1]$ divided by $T = 1$ can't separate the partner from 510 negatives decisively. SimCLR's ablations found values between 0.1 and 0.5 best, with the best value depending on the batch size and training length.

> **spec.** `train_simclr(encoder, feature_dim, images, *, epochs=1, batch_size=256, temperature=0.1, lr=1e-3, strength=0.5, hidden_dim=512, proj_dim=128, seed=0, device='cpu')`: `images` is an array or tensor `(N, C, H, W)` in $[0, 1]$. call `torch.manual_seed(seed)` first, then build `ProjectionHead(feature_dim, hidden_dim, proj_dim)` and `SimCLRAugment(H, strength=strength)`, move encoder and head to `device`, and optimise both with Adam (`lr`). each epoch: put both in train mode, shuffle with a fresh permutation from `np.random.default_rng(seed)` (one generator for the whole run), and take `N // bs` full batches with `bs = min(batch_size, N)`; per batch, augment twice, compute `info_nce_loss(head(encoder(v1)), head(encoder(v2)), temperature=temperature)` and step. the encoder is trained in place. returns `(head, history)` with `history['loss']` a list of per-step floats.

> **🔮 predict:** you train with $T = 0.05$, $0.2$ and $1$ for the same number of steps. which ends with the lowest loss? which with the best kNN accuracy? are those the same answer?

> **▶ build: stage 4.** implement `train_simclr`.

## worked example

**the loss by hand.** $B = 2$, $z_1 = ((1, 0), (0, 1))$, $z_2 = ((1, 0), (0, 1))$: the two views of each image are identical, and the two images are orthogonal. stack them: $v_1 = (1, 0)$, $v_2 = (0, 1)$, $v_3 = (1, 0)$, $v_4 = (0, 1)$, with partners $1 \leftrightarrow 3$ and $2 \leftrightarrow 4$. with $T = 1$, view 1 sees similarities $s_{12} = 0$, $s_{13} = 1$ (its partner), $s_{14} = 0$, and $s_{11}$ is excluded:

$$\ell_1 = -\log\frac{e^1}{e^0 + e^1 + e^0} = \log\Big(1 + \frac{2}{e}\Big) = 0.5514.$$

by symmetry every view has the same loss, so $\mathcal L = 0.5514$. two classic mistakes give different numbers: leaving the self-similarity $s_{11} = 1$ in the denominator gives $\log\big((2e + 2)/e\big) = \log(2 + 2/e) = 1.0064$, and using only the other side's views as negatives (the $B \times B$ version, which is what CLIP does in topic 05) gives $\log(1 + 1/e) = 0.3133$. at $T = 0.1$ the same embeddings give $\log(1 + 2e^{-10}) \approx 9 \times 10^{-5}$: the temperature is what lets a perfect solution score near zero.

**the jitter, by hand.** the colour $c = (0.3, 0.4, 0.5)$ has luma $0.299 \cdot 0.3 + 0.587 \cdot 0.4 + 0.114 \cdot 0.5 = 0.3815$. brightness $b = 1.2$ gives $(0.36, 0.48, 0.60)$, luma $0.4578$. for a constant image, contrast blends with the image's mean gray, which is that same luma, so it scales the colour's distance from gray by $c$ and keeps the luma; saturation does the same with factor $\sigma$. so after all three, the luma is $1.2 \times 0.3815$ and the distance from gray has been scaled by $b\,c\,\sigma$: the checks read your brightness factors back from exactly this.

## on paper

1. **chance level.** show that if all $2B - 1$ candidate similarities of a view are equal, $\ell_i = \log(2B - 1)$. use Jensen's inequality to show that for *any* similarities with the positive no larger than the average negative, $\ell_i \ge \log(2B - 1)$.

<details><summary>solution</summary>

with all logits equal, the softmax is uniform over $2B - 1$ candidates and $\ell_i = -\log\frac{1}{2B - 1}$. in general, $\ell_i = -a_+ + \log\big(e^{a_+} + \sum_{j \in N}e^{a_j}\big)$ with $a = s/T$ and $N$ the $2B - 2$ negatives. by Jensen (convexity of $\exp$), $\sum_{j \in N}e^{a_j} \ge (2B - 2)e^{\bar a_N}$, so $\ell_i \ge \log\big(1 + (2B - 2)e^{\bar a_N - a_+}\big) \ge \log(2B - 1)$ when $a_+ \le \bar a_N$. the inequality is strict when the negatives' logits are spread: this is why random embeddings score *above* $\log(2B - 1)$ at small temperatures in the lab, where $s/T$ has a large spread.

</details>

2. **the gradient, and what the temperature does.** writing $p_{ij} = e^{s_{ij}/T}/\sum_{k \ne i}e^{s_{ik}/T}$, show that $\partial\ell_i/\partial s_{ij} = p_{ij}/T$ for a negative $j$ and $(p_{i,p(i)} - 1)/T$ for the positive. describe the distribution of "push" over negatives as $T \to 0$ and as $T$ grows.

<details><summary>solution</summary>

$\ell_i = -s_{i,p(i)}/T + \log\sum_{k \ne i}e^{s_{ik}/T}$. differentiating, the log-sum-exp contributes $p_{ij}/T$ for every $j$, and the first term contributes $-1/T$ for the positive. the positive is pulled with strength $(1 - p_{i,p(i)})/T$, and each negative pushed with a strength proportional to its own softmax weight. as $T \to 0$ the weights concentrate on the most similar negatives (hard-negative mining, built in). as $T$ grows they flatten, every negative is pushed about equally (pushing away from the negatives' mean), and once $T$ is large the overall $1/T$ makes every gradient small. Wang & Liu (2021) show that small temperatures push the embedding towards more uniformity but less tolerance of semantically similar negatives.

</details>

3. **a mutual-information bound (sketch).** the InfoNCE paper shows $I(v; v') \ge \log N - \mathcal L_N$ for $N$ candidates per anchor, at the optimum of the critic. what does that say about the largest mutual information InfoNCE can certify with $B = 256$? why does this make people say contrastive learning needs large batches, and why is that only part of the story?

<details><summary>solution</summary>

since $\mathcal L_N \ge 0$, the bound can never certify more than $\log N = \log(2B - 1) \approx 6.2$ nats at $B = 256$, however much information the views share. larger batches raise the ceiling. but the bound's tightness isn't what makes representations good: Tschannen et al. (2020) show that encoders with higher MI bounds can have *worse* downstream accuracy, and that the inductive biases of the architecture and the augmentations do much of the work. large batches mostly help by supplying more (and harder) negatives per step.

</details>

## experiment

`lab.py` investigates, with your code:

1. what does an InfoNCE loss value mean? the loss of random and of "perfect" embeddings for batch sizes from 16 to 1,024 and four temperatures.
2. SimCLR on unlabelled MNIST at three temperatures and two seeds: kNN and linear-probe accuracy, and the alignment and uniformity of the embeddings.

## check your understanding

1. why must the self-similarity be excluded from the denominator rather than just left in?

<details><summary>answer</summary>

$s_{ii} = 1$ always (after normalisation), the largest possible value. left in, it dominates the softmax at small temperatures, so the positive can never have most of the probability mass: the loss floor rises to about $\log 2$ even for perfect embeddings, and the gradient keeps "pushing away" from a copy of the anchor itself, which does nothing useful.

</details>

2. why does SimCLR evaluate $h$ and not $z$?

<details><summary>answer</summary>

$z$ is trained to be invariant to every augmentation, so it discards information (colour, orientation) the loss calls nuisance but downstream tasks may need. $h$, one nonlinear layer earlier, keeps more of it. the projection head is a buffer between "what the pretext task needs" and "what the representation should keep".

</details>

3. you double the batch size and the loss goes up from 5.0 to 5.6. did training get worse?

<details><summary>answer</summary>

not necessarily: chance level went from $\log(2B - 1)$ to about $\log(4B - 1)$, an increase of $\log 2 \approx 0.69$. compare losses at the same batch size, or report $\log(2B - 1) - \mathcal L$, the MI lower bound.

</details>

4. which augmentation would you remove to make SimCLR learn colour-sensitive features, and what shortcut might then appear?

<details><summary>answer</summary>

the colour jitter and the grayscale step. then crops of the same image share their colour statistics, and the network can match views by colour histogram alone, which is easy and doesn't require recognising objects; representations usually get much worse on object classification.

</details>

5. why does every image in a batch need its own random crop and jitter, rather than one draw per batch?

<details><summary>answer</summary>

with one draw per batch, the two views of every image differ by the *same* transformation, and all images in a view share it. the network could learn the batch's transformation once instead of becoming invariant to transformations; and negatives would share the same distortion as the anchor, giving a shortcut (match the distortion) that tells it nothing about content.

</details>

6. why put BatchNorm in the projection head?

<details><summary>answer</summary>

empirically, it stabilises training and helps accuracy (the SimCLR v2 and MoCo v3 heads use it). BatchNorm also couples the examples in a batch, which turned out to matter a lot for non-contrastive methods (topic 03), where it was once suspected of being a hidden source of negatives.

</details>

## common bugs

- **self-similarity left in the softmax.** the loss can't go below about $\log 2$, even on perfect embeddings; the worked example gives 1.006 instead of 0.551.
- **only cross-view negatives.** the $B \times B$ CLIP-style loss used for SimCLR: half the negatives are missing. worked example: 0.313.
- **forgetting to normalise.** the network inflates norms instead of learning; the loss can drop to 0 without learning anything.
- **one random draw per batch** in the augmentation (a `torch.rand(1)` where `torch.rand(B)` belongs). identical images in a batch come out identical.
- **the same augmented tensor used for both views.** the loss falls to near 0 almost immediately, and the representation learns nothing.
- **comparing loss values across batch sizes or temperatures.** chance level and the achievable floor both move.

## going deeper

- Chen, Kornblith, Norouzi & Hinton (2020), *A simple framework for contrastive learning of visual representations* (SimCLR). [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- van den Oord, Li & Vinyals (2018), *Representation learning with contrastive predictive coding* (InfoNCE). [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
- He, Fan, Wu, Xie & Girshick (2020), *Momentum contrast for unsupervised visual representation learning* (MoCo): negatives from a queue instead of the batch. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
- Wang & Isola (2020), *Understanding contrastive representation learning through alignment and uniformity on the hypersphere*. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
- Wang & Liu (2021), *Understanding the behaviour of contrastive loss*: the temperature and hard negatives. [arXiv:2012.09740](https://arxiv.org/abs/2012.09740)
- Tschannen, Djolonga, Rubenstein, Gelly & Lucic (2020), *On mutual information maximization for representation learning*. [arXiv:1907.13625](https://arxiv.org/abs/1907.13625)

## connections

- ← builds on: cross-entropy (ch 08, ch 14), data augmentation (ch 15 · topic 03), CNN encoders (ch 16), the evaluation harness (topic 01).
- → used in: topic 03 removes the negatives; topic 05's CLIP is InfoNCE between an image and its caption; the capstone pretrains a ResNet with your `train_simclr` pieces and measures what it buys at 1%, 10% and 100% of the labels.
