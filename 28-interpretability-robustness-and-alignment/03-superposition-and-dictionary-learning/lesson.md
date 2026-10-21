# superposition & dictionary learning

## why this matters

topic 02 found an induction head by reading attention patterns, and attention heads are a convenient unit: each one does something you can often name. neurons are not. look at the MLP neurons of a language model and many of them are **polysemantic**: one fires for academic citations, Korean text and HTTP requests. if the natural unit of a model's computation isn't the neuron, what is it?

the **superposition hypothesis** (Elhage et al., 2022) offers an answer. a model wants to represent far more features (properties of the input it finds useful) than it has dimensions. if features are **sparse**, rarely active at the same time, it can store them as *almost* orthogonal directions, more directions than dimensions, and tolerate a little interference when two happen to be active together. then each neuron is a mixture of many features, and the features are directions in activation space that no single neuron lines up with.

if that's right, interpretability needs a way to find the directions. **sparse autoencoders** (SAEs) do it by learning an overcomplete dictionary: write each activation vector as a sparse combination of many more directions than it has dimensions. Bricken et al. (2023) and Templeton et al. (2024) trained SAEs on real language models and found millions of features that do look interpretable ("the Golden Gate Bridge", "code with a security vulnerability"), and that can be turned up to steer the model's behaviour. in this topic you build the toy model in which superposition was first studied, watch it appear as features get sparser, and train SAEs that recover the hidden features, including the strange way their answers depend on how many features you allow them.

## warm-up: predict first

1. a linear map squeezes 20 features through 5 dimensions and back. how many features can it reconstruct perfectly?
2. now add a ReLU after the map back, and make the features rare (each active 1% of the time). how many features do you guess a trained model will keep?
3. two features are stored in the same dimension, pointing in opposite directions. when only one is active, can a ReLU read it back without error?
4. you train an SAE with 4 features and one with 16 on the same activations. will the 16-feature dictionary contain the 4-feature one plus 12 more?

## prerequisites check

- [ ] you know PCA: the best rank-$k$ linear reconstruction keeps the top $k$ variance directions (→ ch 11 · 03)
- [ ] you know L1 regularisation promotes sparsity, and why (→ ch 08 · 03, the lasso)
- [ ] you know autoencoders (→ ch 21 · 02)
- [ ] you can write a training loop with Adam in PyTorch (→ ch 15 · 07)
- [ ] you can read activations out of a model (→ topic 02, `run_with_cache`)

## core ideas

### idea 1: features, and why sparsity matters

call a **feature** any property of the input that the model represents with a direction in some activation space: "this token is French", "there is a curve in this patch". in real models we don't know the features, which is the whole difficulty. so start with a world where we do: synthetic inputs $x \in \mathbb{R}^n$ whose coordinates *are* the features. each $x_i$ is zero with probability $S$ (the **sparsity**) and uniform on $[0, 1)$ otherwise, independently.

sparsity is the key variable. with $S = 0$ every feature is active in every input; with $S = 0.99$ a typical input has $n/100$ active features, and two given features are active together only $10^{-4}$ of the time. a model that stores two features in overlapping directions pays for the overlap only when both are active, so the rarer co-activation is, the cheaper sharing becomes.

> **spec.** `sparse_feature_batch(n, n_features, sparsity, *, generator=None)`: a float32 tensor `(n, n_features)` whose entries are independently uniform on $[0, 1)$ and then set to zero with probability `sparsity`; all randomness from `generator`; `ValueError` unless $0 \le$ `sparsity` $\le 1$.

> **▶ build: stage 1.** implement `sparse_feature_batch`.

### idea 2: the toy model of superposition

Elhage et al.'s toy model compresses $n$ features into $m < n$ hidden dimensions with a matrix $W \in \mathbb{R}^{m \times n}$ and reconstructs them with its transpose, a bias and a ReLU:

$$h = W x, \qquad x' = \mathrm{ReLU}(W^\top h + b) = \mathrm{ReLU}(W^\top W x + b).$$

column $W_i \in \mathbb{R}^m$ is feature $i$'s direction in the hidden space. the reconstruction of feature $i$ is $\mathrm{ReLU}(\|W_i\|^2 x_i + \sum_{j \ne i} (W_i \cdot W_j) x_j + b_i)$: its own signal plus **interference** from every other active feature that isn't orthogonal to it. two facts make superposition possible. the ReLU can throw away *negative* interference, and a negative bias can absorb small positive interference. so if features rarely co-occur, packing them at obtuse angles costs little. a purely linear model can't do this: without the ReLU, the best it can do is PCA, keeping $m$ features and dropping the rest (warm-up 1).

features differ in how much they matter. the model is trained with an **importance**-weighted loss, importance $I_i$ for feature $i$, so that it has to decide what to drop.

> **spec.** `ToySuperposition(n_features, n_hidden, *, importance=None)`: an `nn.Module` with parameters `W` `(n_hidden, n_features)` (Xavier-normal init) and `b` `(n_features,)` (zeros), and a buffer `importance` `(n_features,)` (ones by default; `ValueError` for the wrong shape); attributes `n_features`, `n_hidden`. `forward(x)`: `(B, n_features)` → $\mathrm{ReLU}(x W^\top W + b)$.

> **▶ build: stage 2.** implement `ToySuperposition`.

### idea 3: training it, and measuring the geometry

the loss is the importance-weighted squared error, averaged over a batch:

$$L = \mathbb{E}_x\Big[\sum_i I_i\, (x_i - x'_i)^2\Big].$$

train it on a fresh batch of sparse features every step (there's no fixed dataset: the distribution is the data). the lab's result is a sequence of **phase changes**. with dense features the model learns $W^\top W$ = a projection onto the $m$ most important features and ignores the rest. as sparsity rises, less important features are added, first as **antipodal pairs** (two features sharing a dimension in opposite directions, so that whichever is active, the other's reconstruction goes negative and is cut off by the ReLU), then in more crowded arrangements.

to describe the arrangement, Elhage et al. defined each feature's **dimensionality**

$$D_i = \frac{\|W_i\|^2}{\sum_j (\hat W_i \cdot W_j)^2}, \qquad \hat W_i = W_i / \|W_i\|.$$

the numerator is how strongly feature $i$ is represented; the denominator counts how many features (weighted by their overlap with $i$'s direction) share that direction, including $i$ itself. a feature with its own dimension has $D_i = 1$; each of an antipodal pair, $\tfrac12$; each of five features arranged as a regular pentagon in a 2-D plane, $\tfrac25$ (on paper 2). a feature that isn't represented ($W_i = 0$) gets 0.

> **spec.** `superposition_loss(model, x)`: the scalar tensor $\frac{1}{B}\sum_b \sum_i I_i (x_{bi} - \text{model}(x)_{bi})^2$ using `model.importance`. `train_toy_model(model, sparsity, *, steps=3000, batch_size=1024, lr=1e-3, generator=None)`: Adam with learning rate `lr`; each of `steps` steps draws a NEW `sparse_feature_batch(batch_size, model.n_features, sparsity, generator=generator)` and takes one step on `superposition_loss`; returns the list of per-step losses (floats). `feature_dimensionality(W)`: `W` `(n_hidden, n_features)` → `(n_features,)` tensor of $D_i$, 0 for zero columns (no NaN).

> **🔮 predict:** 5 features, 2 dimensions, importance $0.8^i$. how many features does a trained model represent with dense features? with sparsity 0.9?

> **▶ build: stage 3.** implement `superposition_loss`, `train_toy_model` and `feature_dimensionality`.

### idea 4: sparse autoencoders

in a real model you don't have $W$, only activations $h$. if superposition is true, each $h$ is approximately a sparse, non-negative combination of feature directions $d_j$: $h \approx \sum_j f_j d_j + \text{bias}$, with most $f_j = 0$. finding the $d_j$ and $f_j$ is **dictionary learning**, an old problem from signal processing and neuroscience (Olshausen & Field, 1996, found that a sparse code for natural image patches looks like the receptive fields of the visual cortex).

a sparse autoencoder solves it with one hidden layer much wider than the input:

$$f = \mathrm{ReLU}\big((h - b_\text{dec}) W_\text{enc} + b_\text{enc}\big), \qquad \hat h = f D^\top + b_\text{dec},$$

trained to reconstruct $h$ while keeping $f$ sparse:

$$L = \mathbb{E}\big[\|\hat h - h\|_2^2\big] + \lambda\, \mathbb{E}\big[\|f\|_1\big].$$

the columns of $D \in \mathbb{R}^{d \times F}$ are the dictionary, one direction per learned feature. they're kept at **unit norm**, because otherwise the L1 penalty can be dodged: shrink $f$ by a factor $c$ and grow $D$'s column by $c$, and the reconstruction is unchanged while $\|f\|_1$ drops. here the decoder uses $D = W_\text{dec}$ with each column divided by its norm, so the constraint holds by construction. subtracting $b_\text{dec}$ before encoding centres the data on the point the decoder adds back. initialising $W_\text{enc}$ as a copy of the decoder makes each feature start out "reading" the same direction it "writes".

> **spec.** `SparseAutoencoder(d_in, d_hidden, *, l1_coeff=0.001)`: parameters `W_enc` `(d_in, d_hidden)`, `b_enc` `(d_hidden,)`, `W_dec` `(d_in, d_hidden)`, `b_dec` `(d_in,)`; `W_dec` starts with random unit-norm columns, `W_enc` as a copy of it, biases at zero; attribute `l1_coeff`. `dictionary()`: `W_dec` with every column divided by its norm. `encode(x)` = $\mathrm{ReLU}((x - b_\text{dec}) W_\text{enc} + b_\text{enc})$, `(B, d_hidden)`; `decode(f)` = $f\,\text{dictionary}()^\top + b_\text{dec}$; `forward(x)` → `(x_hat, f, loss)` with `loss` = mean over the batch of $\|\hat x - x\|^2$ (summed over `d_in`) + `l1_coeff` × mean over the batch of $\|f\|_1$.

> **▶ build: stage 4.** implement `SparseAutoencoder`.

### idea 5: training a dictionary, and judging it

train the SAE on a large set of cached activations, with minibatches sampled at random. then ask three questions:

- **is it sparse?** $L_0$: the average number of active features per input.
- **does it reconstruct?** the **fraction of variance unexplained**, $\text{FVU} = \sum \|h - \hat h\|^2 / \sum \|h - \bar h\|^2$ (0 is perfect, 1 is no better than predicting the mean). in real models the decisive test is downstream: splice $\hat h$ back into the model and measure how much the loss rises.
- **are its features alive?** a **dead** feature never fires on the data; dead features are wasted capacity, and a common failure.

$\lambda$ trades the first two against each other, and neither tells you whether the features are *right*. in a toy world you can check: the **mean max cosine similarity** (MMCS) between the learned dictionary and the true directions, averaged over true features, measures how many were found. the lab shows the catch. with too little sparsity pressure the SAE reconstructs perfectly with a dictionary that matches nothing in particular (an overcomplete dictionary can reconstruct a vector in infinitely many ways); with too much, it starts dropping information. and when features have structure, "the true features" may depend on how many you ask for: a narrow SAE learns a general feature that a wide SAE **splits** into specific ones (Bricken et al., 2023).

> **spec.** `train_sae(sae, acts, *, steps=2000, batch_size=256, lr=1e-3, generator=None)`: Adam with `lr`; each step samples `batch_size` row indices of `acts` with `torch.randint(..., generator=generator)` and steps on the SAE's loss; returns the per-step losses (floats). `mmcs(learned, true)`: `learned` `(d, m)`, `true` `(d, n)`; for each column of `true` the largest cosine similarity with a column of `learned`, averaged over the $n$ true columns; a float. `sae_stats(sae, acts)`: `{"l0": mean number of features > 0 per row, "dead": fraction of features that are never > 0, "fvu": the formula above}`, floats, without gradients.

> **🔮 predict:** an SAE with 40 features on 5-dimensional activations reconstructs them with FVU 0.001 at $\lambda = 10^{-4}$. will its dictionary match the true features?

> **▶ build: stage 5.** implement `train_sae`, `mmcs` and `sae_stats`.

## worked example

**the toy loss.** three features with importance $(1, 0.5, 0)$, $W = 0$ and $b = (0.2, 0, 0)$, so every output is $(0.2, 0, 0)$. for $x = (1, 1, 1)$ the weighted error is $1 \cdot 0.8^2 + 0.5 \cdot 1^2 + 0 = 1.14$; for $x = (0, 0.5, 0)$ it's $1 \cdot 0.2^2 + 0.5 \cdot 0.25 = 0.165$. the loss is the mean, **0.6525**.

**feature dimensionality.** $W = \begin{pmatrix} 1 & -1 & 0 \\ 0 & 0 & 2 \end{pmatrix}$: features 0 and 1 are an antipodal pair, feature 2 is alone. for feature 0: $\|W_0\|^2 = 1$, and $\hat W_0 \cdot W_j = (1, -1, 0)$, so the denominator is $1 + 1 + 0 = 2$ and $D_0 = \tfrac12$. the same for feature 1, and $D_2 = 4/4 = 1$. scale the pair to lengths 3 and 0.5: $D_0 = 9/(9 + 0.25) = 0.973$, $D_1 = 0.25/9.25 = 0.027$; the strong feature owns the dimension.

**SAE statistics.** an SAE with features $f = (\mathrm{ReLU}(h_0), \mathrm{ReLU}(-h_0), 0)$ and dictionary columns $e_0, e_0, e_1$ reconstructs $\hat h = (|h_0|, 0)$. on the four rows $(1, 0), (2, 1), (-1, 0), (0, 0)$: $L_0 = 3/4$, one of three features is dead, and FVU $= (0 + 1 + 4 + 0) / 5.75 = 0.87$ (it gets $h_0$'s sign wrong and ignores $h_1$).

your functions must reproduce 0.6525, $(\tfrac12, \tfrac12, 1)$, and $(0.75, \tfrac13, 0.87)$.

## on paper

1. **why the ReLU matters.** without the ReLU and bias, $x' = W^\top W x$ with $W \in \mathbb{R}^{m \times n}$. show that $x' $ can equal $x$ for at most $m$ linearly independent directions, so a linear model can't use superposition.

<details><summary>solution</summary>

$W^\top W$ has rank at most $m$, so its image is at most $m$-dimensional; $x' = x$ requires $x$ in that image. with dense features the optimal $W^\top W$ is the projection onto the $m$ most important (variance-weighted) features, as in PCA; every other feature is reconstructed as a mixture of the kept ones, or dropped. a nonlinearity is what lets the model trade occasional interference for capacity.

</details>

2. **the pentagon.** five unit vectors at angles $2\pi k/5$ in a plane. compute $D_i$.

<details><summary>solution</summary>

$\hat W_i \cdot W_j = \cos(2\pi(j - i)/5)$. $\sum_j \cos^2(2\pi j/5) = \sum_j \tfrac12(1 + \cos(4\pi j/5)) = \tfrac52 + \tfrac12 \cdot 0 = \tfrac52$ (the cosines of the five 5th roots of unity, doubled in angle, sum to 0). so $D_i = 1/\tfrac52 = \tfrac25$. in general $k \ge 3$ unit vectors evenly spread in a plane give $D = 2/k$, and with $k = 2$ the antipodal pair gives $\tfrac12$.

</details>

3. **when is an antipodal pair worth it?** features 1 and 2 share a dimension: $W_1 = -W_2 = u$, $\|u\| = 1$, $b = 0$. what's the reconstruction error when (a) only feature 1 is active, (b) both are active with values $x_1, x_2$? with sparsity $S$, how often does (b) happen?

<details><summary>solution</summary>

(a) $x'_1 = \mathrm{ReLU}(x_1) = x_1$ and $x'_2 = \mathrm{ReLU}(-x_1) = 0$: perfect. (b) $x'_1 = \mathrm{ReLU}(x_1 - x_2)$ and $x'_2 = \mathrm{ReLU}(x_2 - x_1)$: the smaller one is lost entirely and the larger one is reduced, error $\min(x_1, x_2)^2 \cdot 2$ roughly. both are active with probability $(1-S)^2$, which is $10^{-2}$ at $S = 0.9$ and $10^{-4}$ at $S = 0.99$, against the alternative of never representing feature 2 at all. the denser the features, the worse the deal.

</details>

4. **why unit-norm decoder columns?** show that without the constraint, the SAE loss can be made arbitrarily close to the pure reconstruction loss without making $f$ any sparser.

<details><summary>solution</summary>

replace $W_\text{dec}$'s column $j$ by $c\, d_j$ and $f_j$ by $f_j / c$ (by scaling $W_\text{enc}$'s column and $b_\text{enc}$ by $1/c$). the reconstruction $f_j d_j$ is unchanged, the L1 term is divided by $c$. as $c \to \infty$ the penalty vanishes while the number of active features stays the same: the L1 "sparsity" would be measuring the scale, not the sparsity.

</details>

## experiment

the lab trains the toy model of Elhage et al. across sparsities (20 features, 5 dimensions) and counts how many features it keeps, plots the feature dimensionalities you computed to see where they cluster, trains your SAE on the toy model's hidden activations to see whether it recovers the hidden features and how that depends on `l1_coeff`, and finally trains SAEs of increasing width on data with a hierarchy of features to watch features split.

## check your understanding

1. what is the difference between a polysemantic neuron and superposition?

<details><summary>answer</summary>

polysemanticity is an observation about neurons: one unit responds to several unrelated things. superposition is a hypothesis about representations: more features than dimensions, stored as non-orthogonal directions. superposition implies polysemantic neurons (no neuron can align with more than a few of the overcomplete directions), but neurons can also be polysemantic for other reasons.

</details>

2. why does the toy model learn *negative* interference ($W_i \cdot W_j < 0$) between features it superposes?

<details><summary>answer</summary>

positive interference adds a false positive to feature $i$ whenever $j$ is active, and the ReLU passes it through. negative interference pushes $x'_i$ below zero, and the ReLU removes it: when one of the pair is active alone (by far the common case for sparse features), the other's reconstruction is exactly zero.

</details>

3. a model's features have importance $0.7^i$. in which order does it add features as sparsity rises, and why?

<details><summary>answer</summary>

by importance: the most important features get dedicated dimensions first, and the least important ones are added (in superposition) only when sparsity makes interference cheap enough to be worth the small gain they bring. at every sparsity the represented features are a prefix of the importance order.

</details>

4. why is FVU alone a poor way to choose an SAE's `l1_coeff`?

<details><summary>answer</summary>

FVU only gets better as the penalty shrinks, and the best FVU is achieved by dense, uninterpretable codes: with more features than dimensions there are infinitely many exact reconstructions. you need a sparsity measure ($L_0$) too, and ideally a downstream measure (the model's loss with the reconstruction spliced in) or, in a toy setting, the match with the true features.

</details>

5. a feature in a 4-feature SAE corresponds to four features in a 16-feature SAE. which one is "real"?

<details><summary>answer</summary>

the question may not have an answer: both are good descriptions at different resolutions. the data has a hierarchy, and the dictionary size decides which level an SAE spends its features on. this is feature splitting; it suggests treating SAE features as a useful basis, not as the unique "true" variables of the model.

</details>

6. why are SAEs trained on cached activations rather than end-to-end with the model?

<details><summary>answer</summary>

the goal is to explain the model as it is, so its weights stay frozen; the SAE is a measurement device. caching (topic 02's `run_with_cache`) lets you collect millions of activation vectors once and train dictionaries of several sizes on them cheaply.

</details>

## common bugs

- **inverted sparsity.** `x * (rand < sparsity)` keeps features with probability $S$ instead of zeroing them. check the zero fraction.
- **one fixed batch.** drawing the data once outside the training loop trains on a few hundred points instead of the distribution; the loss looks fine and the geometry is wrong.
- **forgetting the importance.** an unweighted loss makes every feature equally worth keeping and removes the ordering the lab relies on.
- **un-normalised decoder.** the L1 penalty then shrinks $f$ and grows $W_\text{dec}$, and "sparsity" means nothing.
- **NaN dimensionalities.** a dropped feature has $W_i = 0$ and $D_i = 0/0$; return 0.
- **MMCS the wrong way round.** averaging over learned features rewards a dictionary with many copies of one easy feature; average over the *true* ones.

## going deeper

- Elhage et al. (2022), "Toy models of superposition", Anthropic. [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
- Bricken et al. (2023), "Towards monosemanticity: decomposing language models with dictionary learning", Anthropic. [transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- Cunningham et al. (2023), "Sparse autoencoders find highly interpretable features in language models". [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
- Templeton et al. (2024), "Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet", Anthropic. [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- Olshausen & Field (1996), "Emergence of simple-cell receptive field properties by learning a sparse code for natural images", *Nature* 381. [doi:10.1038/381607a0](https://doi.org/10.1038/381607a0)
- Gao et al. (2024), "Scaling and evaluating sparse autoencoders" (TopK SAEs, dead latents, downstream-loss evaluation). [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)

## connections

- ← builds on: PCA (ch 11 · 03), the lasso's L1 sparsity (ch 08), autoencoders (ch 21), reading activations with `run_with_cache` (topic 02).
- → used in: chapter 30's interpretability projects start from `SparseAutoencoder` on your Llama's residual stream; the capstone finds that a transformer trained on modular addition uses only a handful of Fourier features, a case where the model's features are *not* in superposition and can be read directly.
