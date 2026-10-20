# sets

## why this matters

a lidar scan is a list of 3-D points. a particle-physics event is a list of measured particles. a patient's record is a list of lab results; a shopping basket is a list of items. in each case the list is really a **set**: the order the elements happen to arrive in carries no information, and the number of elements varies from one example to the next.

feed such a list to an ordinary network and two things go wrong. first, it needs a fixed input size, so you pad or truncate. second, and worse, it treats the order as meaningful. a set of 10 points can be listed in $10! = 3.6$ million orders, and an MLP on the concatenated list sees 3.6 million different inputs that should all get the same answer. it can only learn that from data, approximately, one ordering at a time. topic 01's group averaging would fix it exactly, at 3.6 million forward passes per example.

this topic builds the alternative: networks that are permutation-invariant (or equivariant) by construction and handle any set size, at the cost of one forward pass. the architecture is almost embarrassingly simple: process every element with the same small network, then combine the results with a symmetric operation like a sum. the interesting questions are which symmetric operation, what it can and can't represent, and what changes when the combination is learned (attention) instead of fixed. the lab answers them on three tiny tasks, and one of the answers surprises nearly everybody.

## warm-up: predict first

1. you represent a set by the **mean** of its elements' feature vectors. can that representation tell the set $\{3\}$ from the set $\{3, 3\}$? the **sum**? the **max**?
2. a batch holds sets of sizes 2, 5 and 3, padded with zeros to size 5. you take the max over each padded set. when does the padding change the answer?
3. how many free parameters does a linear map $\mathbb{R}^n \to \mathbb{R}^n$ have that commutes with every permutation of its $n$ inputs: $n^2$, $n$, $2$, or $1$?
4. attention pooling computes a weighted average of the elements with weights from a softmax. can a weighted average count how many elements a set has?
5. a network is trained only on sets of 1 to 10 elements. on a set of 25, which do you trust more: a sum-pooled network or a max-pooled one?

## prerequisites check

- [ ] you can test permutation invariance with `equivariance_error` and random permutations (→ topic 01)
- [ ] you can build an `MLP` and train it with Adam in PyTorch (→ ch 15 · topic 07)
- [ ] you know what a softmax does and why subtracting the max makes it stable (→ ch 08 · topic 05)
- [ ] you've seen scaled dot-product attention (→ ch 18 · topic 01; idea 4 below is self-contained)
- [ ] you can use boolean masks and broadcasting in PyTorch (→ ch 00 · topic 02, the NumPy version)

## core ideas

### idea 1: padded batches and symmetric pooling

sets in a batch have different sizes, but tensors are rectangular. the standard representation pads every set to the largest size $N$ in the batch and carries a boolean **mask**: a tensor `x` of shape `(B, N, F)` and `mask` of shape `(B, N)`, where `mask[b, i]` is `True` if row $i$ of set $b$ is a real element. the padded rows contain whatever happens to be there (zeros, garbage, the previous batch's values), and **no output may depend on them**. that's a correctness requirement, not a nicety: a model whose output changes when you change the padding gives a different prediction for the same set depending on what else is in the batch.

the building block of every set network is a **symmetric pooling** that reduces a set of vectors to one vector without looking at the order:

| pooling | formula (over real elements) | sees | blind to |
|---|---|---|---|
| sum | $\sum_i h_i$ | the multiset: values *and* multiplicities | nothing, in principle |
| mean | $\frac{1}{n}\sum_i h_i$ | the distribution of values | the set size $n$ |
| max | $\max_i h_i$ (per feature) | which values occur | multiplicities, size |

masking each one takes care. for the sum, multiply by the mask (padding contributes 0). for the mean, also divide by the number of *real* elements, not by $N$. for the max, the tempting `(h * mask).max(1)` is wrong: if all real values of a feature are negative, a padded 0 wins. fill the padding with $-\infty$ before the max instead. and decide what an **empty set** pools to: here, 0 for all three (without care the mean divides by zero and the max returns $-\infty$).

> **spec · stage 1.** `masked_pool(h, mask=None, aggregation="sum")`: `h` `(B, N, F)`, `mask` `(B, N)` bool (`None`: every row is real) → `(B, F)`, pooling over the real rows with `aggregation` in `{"sum", "mean", "max"}` (`ValueError` otherwise). the result must not depend on the values in padded rows; an empty set pools to 0 (no `nan`, no `-inf`); gradients flow to the real rows only.

> **▶ build: stage 1.** implement `masked_pool` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: DeepSets, the sum decomposition

here's the whole architecture: apply the same network $\phi$ to every element, pool, then apply a second network $\rho$ to the pooled vector:

$$f(X) = \rho\Big(\bigoplus_{i=1}^{n} \phi(x_i)\Big), \qquad \bigoplus \in \{\textstyle\sum, \text{mean}, \max\}.$$

it's invariant for the reason topic 01 gave: $\phi$ applied elementwise is permutation-equivariant, the pool is invariant, and $\rho$ after an invariant is still invariant. it handles any set size, since $\phi$ doesn't care how many elements there are.

the surprise is that with the sum, this simple form is *universal*. Zaheer et al. (2017) proved that when elements come from a countable set (say, integers, or atoms of a few types), **every** permutation-invariant function of finite sets can be written as $\rho(\sum_i \phi(x_i))$. the idea (on paper 1): choose $\phi$ so the sum is injective, i.e. different sets give different sums. then the sum is a lossless code for the set, and $\rho$ can decode it and compute anything. for continuous elements the story is subtler: Wagstaff et al. (2019) showed that exact representation of all continuous invariant functions on sets of up to $N$ real numbers needs a latent dimension (the width of $\phi$'s output) of at least $N$. in practice, a latent width of a few dozen goes a long way, but "universal" doesn't mean "easy to learn", and the pooling you choose decides what's easy.

> **spec · stage 2.** `DeepSets(in_dim, hidden_dim, out_dim, *, aggregation="sum")`, an `nn.Module`: $\phi$ maps `in_dim` → `hidden_dim` and $\rho$ maps `hidden_dim` → `out_dim` (each an MLP with at least one hidden layer; `mlfs.torchnn.MLP` is fine); stores `self.aggregation` (`ValueError` for anything but sum/mean/max). `forward(x, mask=None)`: `(B, N, in_dim)` → `(B, out_dim)`, exactly invariant to permuting each set, independent of padding. this is `mlfs.gnn.DeepSets`.

> **▶ build: stage 2.** implement `DeepSets`.

> **🔮 predict:** a DeepSets with *max* aggregation is trained to output the number of elements in a set. what's the best it can do?

### idea 3: equivariant layers for sets

DeepSets processes each element in isolation until the very last pool: no element ever learns about the others before the summary is taken. for a per-element output (segment each point of a point cloud, score each item in a basket) or a deeper model, you want layers that let elements interact while staying **equivariant**: permute the input rows, and the output rows permute the same way.

what does the most general equivariant *linear* layer look like? for one channel, a linear map $L: \mathbb{R}^n \to \mathbb{R}^n$ is permutation-equivariant if and only if $L = \lambda I + \gamma \mathbf{1}\mathbf{1}^\top$ (topic 01, on paper 4): each output is "$\lambda$ times my own value plus $\gamma$ times the sum of everyone's". two numbers, whatever $n$ is. with $F_\text{in}$ input and $F_\text{out}$ output channels, each pair of channels gets its own two numbers, so the layer is

$$y_i = x_i \Lambda + \operatorname{pool}(X)\,\Gamma + b, \qquad \Lambda, \Gamma \in \mathbb{R}^{F_\text{in} \times F_\text{out}},$$

where the theorem says pool = sum, and in practice mean (better-behaved across set sizes) or max work too (max makes the layer nonlinear, but still equivariant). stacking these layers with nonlinearities in between, then pooling, is the GDL blueprint for sets. note how few parameters there are: $2F_\text{in}F_\text{out} + F_\text{out}$ regardless of the set size, which is the whole point of building in the symmetry.

padding needs one more rule here: the pool must skip padded rows, and padded rows of the *output* should be zero, so that stacking layers can't leak garbage back in through the next layer's pool.

> **spec · stage 3.** `PermEquivariant(in_dim, out_dim, *, aggregation="mean")`, an `nn.Module`: `forward(x, mask=None)` maps `(B, N, in_dim)` → `(B, N, out_dim)` with $y_i = x_i\Lambda + \operatorname{pool}(X)\Gamma + b$, the pool over real rows (`masked_pool`); padded output rows are exactly 0; `ValueError` for an unknown aggregation.

> **▶ build: stage 3.** implement `PermEquivariant`.

### idea 4: pooling by attention

sum, mean and max are fixed choices. **attention** makes the pooling learned. scaled dot-product attention takes queries $Q \in \mathbb{R}^{M \times d}$, keys $K \in \mathbb{R}^{N \times d}$ and values $V \in \mathbb{R}^{N \times d_v}$ and returns, for each query, a weighted average of the values:

$$\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\Big(\frac{QK^\top}{\sqrt d}\Big)V,$$

the softmax taken over the $N$ keys. the $\sqrt d$ keeps the scores' variance near 1 when entries of $Q$ and $K$ have unit variance, so the softmax doesn't saturate. permuting the keys and values together permutes the terms of each weighted sum: the output is invariant to the order of the set. with padding, padded keys must get weight exactly 0: set their scores to $-\infty$ before the softmax. if a set has *no* real elements, every score is $-\infty$, the softmax computes $0/0$, and `nan` appears; worse, it appears in the backward pass even if you overwrite the forward value afterwards. the safe way is to avoid all-$-\infty$ rows before the softmax, then zero the result.

**pooling by multi-head attention** (PMA; Lee et al., 2019) uses $k$ learned **seed vectors** $S \in \mathbb{R}^{k \times d}$ as the queries, and the set elements as keys and values:

$$\operatorname{PMA}(X) = \operatorname{Attn}(S W_q,\ X W_k,\ X W_v)\, W_o.$$

with **multiple heads**, the $d$ dimensions are split into $h$ groups of $d/h$; each head runs its own attention on its slice, and the results are concatenated before $W_o$. each head can attend to different elements, one to the largest, another to a typical one. the set transformer stacks self-attention blocks (each element attends to all others, an equivariant layer that costs $O(N^2)$) before a PMA.

one property of every softmax-weighted average deserves a warning label: list every element of a set twice, and every weight halves, so the output **doesn't change** (on paper 3). attention pooling, like the mean, can't see how many copies of anything there are, and so can't count. the stage 4 checks test exactly this.

> **spec · stage 4.**
> - `masked_attention(q, k, v, mask=None)`: $\operatorname{softmax}(qk^\top/\sqrt d)\,v$ over the real keys; `q` `(B, M, d)`, `k` `(B, N, d)`, `v` `(B, N, dv)`, `mask` `(B, N)` → `(B, M, dv)`. a set with no real elements gives 0, with finite gradients.
> - `AttentionPool(dim, *, heads=1, n_seeds=1)`, an `nn.Module`: learned seeds `(n_seeds, dim)` and query/key/value/output projections `dim → dim` (`ValueError` unless `heads` divides `dim`). `forward(x, mask=None)`: `(B, N, dim)` → `(B, n_seeds, dim)`, each head attending over the real elements with `masked_attention`.

> **▶ build: stage 4.** implement `masked_attention` and `AttentionPool`.

> **🔮 predict:** attention pooling with 4 heads, trained on "the sum of the set". how close does it get, compared with sum pooling?

## worked example

one set with two real elements and one padded row, `h = [[1, -2], [3, 4], [100, 100]]`, `mask = [True, True, False]`:

- sum: $(1 + 3,\ -2 + 4) = (4, 2)$.
- mean: divide by the **2** real elements: $(2, 1)$. dividing by 3 would give $(1.33, 0.67)$, a classic bug.
- max: $(3, 4)$. had the padding been filled with zeros and the second feature's real values been $-2$ and $-4$, an unmasked max would return 0, a value that isn't in the set.

your `masked_pool` must reproduce these exactly (they're stage 1's first check).

now attention with one query $q = (1, 0)$ over two keys $k_1 = (2, 0)$, $k_2 = (0, 2)$ in $d = 2$, values $v_1 = 10$, $v_2 = 20$. scores: $q \cdot k_1 / \sqrt 2 = 1.414$, $q \cdot k_2/\sqrt 2 = 0$. softmax: $e^{1.414}/(e^{1.414} + 1) = 0.804$ and $0.196$. output: $0.804 \cdot 10 + 0.196 \cdot 20 = 11.96$. list both elements twice: scores $(1.414, 0, 1.414, 0)$, weights $(0.402, 0.098, 0.402, 0.098)$, output $11.96$ again.

## on paper

1. let elements come from a countable set, enumerated as $c: \mathcal{X} \to \{0, 1, 2, \dots\}$. show that $\phi(x) = 2^{-c(x)}$ makes $X \mapsto \sum_{x \in X}\phi(x)$ injective on finite **sets** (no repeats), so every invariant function is $\rho \circ \sum \phi$. how would you change $\phi$ for **multisets** with at most $m$ copies of each element?

<details><summary>solution</summary>

$\sum_{x \in X} 2^{-c(x)}$ is a number whose binary expansion has a 1 in position $c(x)$ for each $x \in X$ and 0 elsewhere. a finite set gives a finite binary expansion, which is unique, so two different sets give different sums. define $\rho$ on the image by $\rho(\sum_{x \in X}\phi(x)) = f(X)$; it's well defined because the sum is injective. for multisets with multiplicities at most $m$, use base $m + 1$: $\phi(x) = (m+1)^{-c(x)}$, so each "digit" holds a count from 0 to $m$ without carrying. (this $\phi$ is wildly discontinuous, which is why the continuous case needs the latent-dimension bound instead.)

</details>

2. give two different sets (or multisets) that mean pooling maps to the same vector for *every* $\phi$, and two that max pooling does. what does this say about a mean-pooled network asked for the sum, and a max-pooled network asked for the count?

<details><summary>solution</summary>

mean: $\{a\}$ and $\{a, a\}$, or any multiset and its double, give $\phi(a)$ both times. max: $\{a, b\}$ and $\{a, a, b\}$ give $\max(\phi(a), \phi(b))$ both times. so mean pooling can only compute functions of the *empirical distribution*, and the sum ($n$ times the mean) isn't one; max pooling can only compute functions of the *support* (which values occur), and the count isn't one. no amount of training fixes this: the information is gone after the pool.

</details>

3. show that softmax attention pooling gives the same output for a set $X$ and for the multiset $X \cup X$ (every element twice), for any query.

<details><summary>solution</summary>

with scores $s_i$ the weights are $w_i = e^{s_i}/\sum_j e^{s_j}$. in $X \cup X$ every score appears twice, so the denominator doubles and each copy gets $w_i/2$; the two copies of $v_i$ together contribute $2 \cdot (w_i/2) v_i = w_i v_i$. the output $\sum_i w_i v_i$ is unchanged. any normalized weighting has this property, which is why attention pooling, like the mean, can't count.

</details>

4. compare the cost of a DeepSets forward pass with a self-attention layer on a set of $N$ elements with $d$ features. where's the crossover with $d = 64$?

<details><summary>solution</summary>

DeepSets: $\phi$ on each element is $O(Nd^2)$, the pool $O(Nd)$, $\rho$ $O(d^2)$: linear in $N$. self-attention: the projections are $O(Nd^2)$ and the score matrix and weighted sum $O(N^2 d)$. the quadratic term dominates once $N^2 d > N d^2$, i.e. $N > d = 64$. that's why the set transformer's ISAB attends through $m \ll N$ "inducing points" for $O(Nmd)$, and why PMA (with $k$ seeds) is $O(Nkd)$.

</details>

## experiment

the lab trains your `DeepSets` with sum, mean and max aggregation, and a `phi → AttentionPool → rho` model, on three tasks over sets of 1 to 10 numbers: the sum of the set, its max, and the count of elements above 0.5. you'll find out which aggregations can learn which tasks, what happens when the trained models meet sets of 20 to 30 elements, and whether learned attention pooling beats the fixed choices. predict before you run each one.

## check your understanding

1. why must padded rows of a `PermEquivariant` layer's output be set to zero?

<details><summary>answer</summary>

the next layer pools over its input with the same mask, so padded values wouldn't enter *that* pool. but anything that reads the tensor without the mask (a loss, a `sum` someone writes later, a layer that forgets the mask) would see them. zeroing keeps the invariant "padded rows carry nothing" true after every layer, so a single forgotten mask can't silently leak garbage.

</details>

2. is a DeepSets with max aggregation still permutation-invariant if $\phi$ is a different network for each position $i$?

<details><summary>answer</summary>

no. permuting the input would send each element through a different $\phi$, changing the pooled vector. the sharing of $\phi$ across elements is exactly what makes $\phi$ equivariant, the same "weight sharing" that makes a convolution translation-equivariant.

</details>

3. a DeepSets is trained on sets of sizes 1–10 with sum pooling. why might its predictions on a set of size 100 be poor even for a task it learned perfectly?

<details><summary>answer</summary>

the pooled vector $\sum\phi(x_i)$ grows with the set size, so at size 100 $\rho$ is evaluated on inputs far outside anything it saw in training, and an MLP's behaviour there is just whatever its last linear pieces extrapolate to. tasks whose answer really does scale linearly with the sum (the sum itself) may survive; tasks where $\rho$ had to undo the size (a max computed from a sum) don't. mean pooling keeps the pooled vector in range but loses the size; normalizing by $\sqrt n$ or feeding $n$ separately are common compromises.

</details>

4. what's the most general permutation-*invariant* linear map $\mathbb{R}^n \to \mathbb{R}$?

<details><summary>answer</summary>

$x \mapsto \gamma \sum_i x_i$ (plus a constant, for an affine map). invariance means $w^\top P x = w^\top x$ for every permutation, so $w$ has equal entries. this is the "sum" in the sum decomposition, and the reason invariant linear readouts on sets and graphs are sums or means.

</details>

5. why does attention scale scores by $1/\sqrt d$?

<details><summary>answer</summary>

if $q$ and $k$ have independent entries with mean 0 and variance 1, $q \cdot k$ has variance $d$. without scaling, for large $d$ the scores spread over many units, the softmax becomes nearly one-hot, and its gradients vanish. dividing by $\sqrt d$ restores unit variance.

</details>

6. you want a set network that can count *and* weight elements by relevance. how could you combine sum pooling and attention?

<details><summary>answer</summary>

pool with both and concatenate: `concat(sum_i phi(x_i), PMA(phi(X)))`; or use unnormalized attention (sigmoid gates instead of a softmax: $\sum_i \sigma(s_i) v_i$), which is a learned weighted *sum* and does see multiplicities; or append the set size $n$ (or $\log n$) as a feature after the pool.

</details>

## common bugs

- **dividing by $N$ instead of the number of real elements**: symptom: the mean of a set depends on how much padding its batch has; stage 1's hand example gives $(1.33, 0.67)$ instead of $(2, 1)$.
- **masking a max by multiplying**: symptom: features whose real values are all negative pool to exactly 0. fill padded rows with $-\infty$ (and handle empty sets).
- **`nan` from an empty set's softmax**: symptom: loss becomes `nan` after one step, only in batches that contain an empty set. overwriting the forward output isn't enough; the backward pass still goes through $0/0$. keep rows finite before the softmax.
- **mask shape**: symptom: a shape error, or worse, a silent broadcast of a `(B, N)` mask against `(B, N, F)` along the wrong axis. use `mask.unsqueeze(-1)` for features and `mask.unsqueeze(1)` for the attention score matrix `(B, M, N)`.
- **mixing heads and batch when reshaping**: symptom: invariance holds but different sets in a batch influence each other's outputs. split `(B, N, h·d)` into `(B, N, h, d)`, move heads next to the batch, and undo exactly the same steps afterwards.
- **a per-position $\phi$**: symptom: permutation tests fail by large amounts. $\phi$ must be the *same* module applied to every element (a `Linear` on the last axis does this automatically).

## going deeper

- Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov & Smola (2017), "Deep sets", *NeurIPS*. the sum decomposition and the equivariant layer. [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
- Qi, Su, Mo & Guibas (2017), "PointNet: deep learning on point sets for 3D classification and segmentation", *CVPR*. a max-pooled set network for point clouds. [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)
- Lee, Lee, Kim, Kosiorek, Choi & Teh (2019), "Set transformer: a framework for attention-based permutation-invariant neural networks", *ICML*. SAB, ISAB and PMA. [arXiv:1810.00825](https://arxiv.org/abs/1810.00825)
- Wagstaff, Fuchs, Engelcke, Posner & Osborne (2019), "On the limitations of representing functions on sets", *ICML*. why the latent dimension must be at least the set size. [arXiv:1901.09006](https://arxiv.org/abs/1901.09006)
- Vaswani et al. (2017), "Attention is all you need", *NeurIPS*. scaled dot-product and multi-head attention. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## connections

- ← builds on: invariance, equivariance and the tester (topic 01), MLPs and training in PyTorch (ch 15 · topic 07), softmax (ch 08), attention (ch 18).
- → used in: graphs are sets with extra structure: message passing (topic 03) is DeepSets applied to each node's set of neighbours, with exactly these aggregations, and topic 04 shows that the choice of aggregation decides what a GNN can tell apart. `masked_pool`-style readouts become `global_add_pool`/`global_mean_pool`. attention pooling returns as graph attention (GAT, topic 03).
