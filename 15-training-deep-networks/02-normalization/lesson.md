# normalization

## why this matters

topic 01 chose the initial weights so that every layer's activations start at a sensible scale. but initialization is a promise about step 0 only. the moment training starts, every weight matrix changes, and so does the scale of everything downstream of it. with a large learning rate, one unlucky step can make a layer's outputs ten times larger, the next layer's ten times larger again, and the loss overflows. with a small one, the network is safe but slow. deep networks before 2015 lived between those two failure modes, with learning rates tuned by hand to a narrow window.

Ioffe and Szegedy (2015) proposed something blunt: if you want the activations to have mean 0 and variance 1, *measure* them and divide. a **normalization layer** standardizes its input on every forward pass, then lets the network rescale it with two learnable parameters. it costs almost nothing, it let networks train with learning rates 5 to 30 times larger, and some version of it sits in almost every modern architecture: batch norm in convolutional networks, layer norm and RMSNorm in transformers. this topic builds all three, including their backward passes, which you'll derive once (one formula serves all three) and plug into your engine with `Tensor.from_op`. the lab then shows you the price: a batch-norm network's answer for one example depends on the other examples in the batch, and forgetting `model.eval()` can turn 86% accuracy into 16%.

## warm-up: predict first

1. you standardize a batch of numbers to mean 0 and variance 1, then add a constant to every input. what happens to the outputs? and to the gradient with respect to the inputs, summed over the batch?
2. batch norm is applied to a batch of exactly two examples. what are the two normalized values of any feature?
3. a `Linear` layer is followed by batch norm. what does the `Linear`'s bias do?
4. at test time you have one example. how can batch norm normalize it?
5. layer norm normalizes over the features of each example rather than over the batch. which of the problems in questions 2 and 4 does it have?

## prerequisites check

- [ ] you can backpropagate through a mean and a variance by hand (→ ch 03 · topic 02, ch 14 · topic 01)
- [ ] you can extend your engine with `Tensor.from_op(data, parents, vjp)` (→ ch 14 · topic 03)
- [ ] your `Module` discovers `Parameter` attributes and switches `.training` with `train()`/`eval()` (→ ch 14 · topic 04)
- [ ] you know the difference between the biased ($1/n$) and unbiased ($1/(n-1)$) variance (→ ch 05 · topic 02)
- [ ] you know exponential moving averages, $m \leftarrow (1 - \alpha)m + \alpha x$ (→ ch 06 · topic 04)

## core ideas

### idea 1: standardize, and its backward pass

all three layers in this topic are built on one operation: **standardize** a set of numbers along some axis,

$$\mu = \frac1m\sum_{i=1}^m x_i,\qquad \sigma^2 = \frac1m\sum_{i=1}^m (x_i - \mu)^2,\qquad \hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}},$$

where the axis is the batch (batch norm) or the features (layer norm). $\epsilon$ (`eps`, $10^{-5}$ by default) keeps the division finite when all $m$ numbers are equal. the variance is the **biased** one, dividing by $m$: that's what makes $\hat x$ have variance exactly $\sigma^2/(\sigma^2 + \epsilon) \approx 1$.

you could build this from your `Tensor`'s `mean`, `-`, `**` and `/`, and your engine would differentiate it. but that records half a dozen nodes and stores their intermediates, and the gradient that comes out is a sum of several pieces you never see. the gradient has a short closed form; deriving it is worth doing once. write $s = (\sigma^2 + \epsilon)^{-1/2}$ (`inv_std`) and $g_i = \partial L/\partial \hat x_i$. every $x_j$ affects $\hat x_i$ directly, through $\mu$ and through $\sigma^2$:

$$\frac{\partial \hat x_i}{\partial x_j} = s\,\delta_{ij} - \frac sm - \frac{s}{m}\hat x_i\hat x_j$$

(the last term uses $\partial\sigma^2/\partial x_j = 2(x_j - \mu)/m$ and $\partial s/\partial\sigma^2 = -s^3/2$; on paper 1 fills in the steps). summing $g_i\,\partial\hat x_i/\partial x_j$ over $i$:

$$\boxed{\frac{\partial L}{\partial x_j} = s\Big(g_j - \overline{g} - \hat x_j\,\overline{g\hat x}\Big)}$$

where the bars are means over the normalized axis. it reads naturally. $g$ minus its mean: shifting all inputs by a constant doesn't change $\hat x$, so the gradient can't have a component along "all ones". minus $\hat x$ times $\overline{g\hat x}$: scaling the inputs doesn't change $\hat x$ either, so the gradient can't have a component along $\hat x$. what's left is $g$ projected onto the directions that *do* change the output, times $s$. the whole backward pass needs only $\hat x$ and $s$ from the forward pass.

> **▶ build: stage 1.** implement `standardize(x, axis, eps=1e-5)`, returning `(xhat, inv_std)` with `inv_std` kept with `keepdims=True`, and `standardize_backward(g, xhat, inv_std, axis)`.

### idea 2: batch norm

**batch normalization** standardizes each feature (column) of a batch $X \in \mathbb R^{N \times C}$ over the batch, then applies a learnable per-feature scale and shift:

$$y_{ic} = \gamma_c\,\hat x_{ic} + \beta_c,\qquad \hat x_{:,c} = \operatorname{standardize}(x_{:,c}).$$

$\gamma$ (`weight`, initialized to ones) and $\beta$ (`bias`, zeros) let the network undo the normalization if it wants: with $\gamma = \sigma$ and $\beta = \mu$ the layer is the identity. so normalization doesn't restrict what the network can represent; it changes how the parameters control it. the scale and mean of the layer's output are now set directly by $\gamma$ and $\beta$, not by a fragile interaction of every weight upstream.

**the backward pass** is idea 1 with axis 0 and $g = \gamma \odot \partial L/\partial y$, plus the parameters' gradients, which are those of an elementwise affine map: $\partial L/\partial\gamma_c = \sum_i \frac{\partial L}{\partial y_{ic}}\hat x_{ic}$ and $\partial L/\partial\beta_c = \sum_i \frac{\partial L}{\partial y_{ic}}$. build the output as **one** `Tensor.from_op` node whose parents are `(x, weight, bias)` and whose `vjp` returns those three gradients. that's how every framework implements normalization layers: one fused operation with a hand-written backward.

**evaluation is different.** at test time there may be a single example, and even with a batch you don't want an example's prediction to depend on which other examples happen to share it. so during training batch norm also keeps **running estimates** of each feature's mean and variance, exponential moving averages over training batches:

$$\text{running\_mean} \leftarrow (1 - m)\,\text{running\_mean} + m\,\mu_\mathcal{B},\qquad \text{running\_var} \leftarrow (1 - m)\,\text{running\_var} + m\,\tfrac{N}{N-1}\sigma^2_\mathcal{B},$$

with `momentum` $m = 0.1$. note PyTorch's convention: momentum is the weight of the *new* value (the opposite of the optimizer's momentum), and the running variance uses the **unbiased** estimate, because it estimates the population's variance, not the batch's. in eval mode the layer uses these fixed statistics, and becomes a plain per-feature affine map, $y = \gamma(x - \text{running\_mean})/\sqrt{\text{running\_var} + \epsilon} + \beta$, whose input gradient is simply $g\,\gamma\,s$. the running statistics are **buffers**: plain ndarrays, not `Parameter`s, so the optimizer never touches them and `parameters()` doesn't list them. `running_mean` starts at 0 and `running_var` at 1; update them in place. in training mode a batch of one can't be normalized (its variance is 0), so refuse it with a `ValueError`, as PyTorch does.

two practical consequences. a `Linear` layer followed by batch norm doesn't need a bias: whatever constant it adds is subtracted again by the mean (warm-up 3; $\beta$ takes over its role). and the `model.train()`/`model.eval()` switch you built in chapter 14 finally matters: forget `eval()` and test-time predictions are normalized with the statistics of the test batch, which is the lab's experiment 2.

> **🔮 predict:** a batch-norm layer receives a batch of 2. what is $\hat x$ for each feature, whatever the inputs are? what's the gradient with respect to the inputs? (the worked example has the answer.)

> **▶ build: stage 2.** implement `BatchNorm1d(num_features, *, eps=1e-5, momentum=0.1)` for inputs of shape `(N, C)`.

### idea 3: layer norm

**layer normalization** (Ba, Kiros & Hinton, 2016) standardizes along the other axis: over the $C$ features of each example, separately for every example:

$$y_{i,:} = \gamma \odot \operatorname{standardize}(x_{i,:}) + \beta.$$

for an input with more axes, such as a sequence $(B, T, C)$, it normalizes the last axis only, so every token is standardized over its own features. because the statistics come from one example, layer norm has none of batch norm's batch-size problems: it behaves identically in training and evaluation, works with a batch of one, keeps no running statistics, and an example's output never depends on its neighbours. that's why it's the normalization of recurrent networks and transformers (chapter 18), where batches are small, sequences have varying lengths, and generation runs one token at a time.

what it gives up is per-feature calibration. batch norm makes every feature have mean 0 and variance 1 *across the data*; layer norm makes every *example* have mean 0 and variance 1 *across its features*, so one feature with an unusually large value shrinks all the others in that example. the backward pass is idea 1 along axis $-1$, and $\gamma$'s and $\beta$'s gradients sum over every leading axis (batch, and time if there is one).

> **▶ build: stage 3.** implement `LayerNorm(num_features, *, eps=1e-5)` for inputs of shape `(..., C)`.

### idea 4: RMSNorm

**RMSNorm** (Zhang & Sennrich, 2019) asks whether layer norm needs to subtract the mean at all, and answers no: dividing by the root mean square is enough to fix the scale, and the re-centring contributes little. it drops the mean and the bias:

$$y = x \odot r \odot \gamma,\qquad r = \Big(\frac1C\sum_{c=1}^C x_c^2 + \epsilon\Big)^{-1/2}.$$

it's cheaper (one reduction instead of two), and it's the normalization in Llama-style language models (chapter 27, where your `mlfs.torchnn.RMSNorm` from topic 07 is used). its default $\epsilon$ is $10^{-6}$.

its backward pass is a smaller cousin of idea 1. with $u = g \odot \gamma$ (the gradient arriving at $x \odot r$), and since $\partial r/\partial x_j = -r^3 x_j/C$,

$$\frac{\partial L}{\partial x_j} = r\,u_j - x_j\,r^3\,\overline{u x},\qquad \frac{\partial L}{\partial\gamma} = \sum_{\text{leading axes}} g \odot x \odot r.$$

the first term is the direct path; the second removes the component along $x$ itself, because RMSNorm is (up to $\epsilon$) invariant to rescaling an example: there's no mean term, because it isn't invariant to shifts.

> **🔮 predict:** RMSNorm keeps the mean. if every input is shifted by $+3$, does the output stay centred?

> **▶ build: stage 4.** implement `RMSNorm(num_features, *, eps=1e-6)`.

## worked example

one feature, a batch of three: $x = (0, 1, 5)$, $\epsilon = 0$, and an upstream gradient $g = \partial L/\partial\hat x = (1, 0, 0)$ (the loss cares only about the first example's normalized value).

**forward.** $\mu = 2$, $\sigma^2 = (4 + 1 + 9)/3 = 4.6667$, $s = 1/\sqrt{4.6667} = 0.46291$, so

$$\hat x = (-2, -1, 3) \times 0.46291 = (-0.92582,\ -0.46291,\ 1.38873).$$

**backward.** $\overline g = 1/3$ and $\overline{g\hat x} = -0.92582/3 = -0.30861$, so

$$\frac{\partial L}{\partial x} = 0.46291\Big((1, 0, 0) - 0.33333 + 0.30861\,(-0.92582, -0.46291, 1.38873)\Big) = 0.46291\,(0.38095,\ -0.47619,\ 0.09524) = (0.17635,\ -0.22043,\ 0.04409).$$

the gradient sums to zero (shift invariance) and $\sum_j x_j\,\partial L/\partial x_j = -0.22043 + 0.22043 = 0$ (scale invariance). and look at the signs: increasing $x_1 = 0$ raises $\hat x_1$, as expected, but increasing $x_2 = 1$ *lowers* it. raising $x_2$ raises the mean, and since $x_2$ is below the mean it also shrinks the spread; both push $\hat x_1 = (0 - \mu)/\sigma$, already negative, further down. every example's normalized value depends on every other example in the batch.

**the extreme case, a batch of two.** for any two distinct values, $\hat x = (-1, +1)$: one is one standard deviation below the mean, the other one above. the output carries no information about the input except which of the two is larger, and the gradient with respect to $x$ is exactly zero. batch norm on tiny batches throws information away; the lab measures how much. the tests check this example, the zero gradient for two, and every layer against PyTorch.

## on paper

1. **derive the standardize backward pass.** show that $\partial\mu/\partial x_j = 1/m$, $\partial\sigma^2/\partial x_j = 2(x_j - \mu)/m$, and hence $\partial\hat x_i/\partial x_j = s\delta_{ij} - s/m - (s/m)\hat x_i\hat x_j$. then derive the boxed formula.

<details><summary>solution</summary>

$\partial\mu/\partial x_j = 1/m$ is immediate. $\sigma^2 = \frac1m\sum_i(x_i - \mu)^2$, so $\partial\sigma^2/\partial x_j = \frac2m\sum_i(x_i - \mu)(\delta_{ij} - \frac1m) = \frac2m(x_j - \mu) - \frac{2}{m^2}\sum_i(x_i - \mu) = \frac2m(x_j - \mu)$, since the deviations sum to zero. with $\hat x_i = (x_i - \mu)s$ and $s = (\sigma^2 + \epsilon)^{-1/2}$, $\partial s/\partial x_j = -\frac12 s^3\cdot\frac2m(x_j - \mu) = -\frac{s^2}{m}\hat x_j$. so $\partial\hat x_i/\partial x_j = (\delta_{ij} - \frac1m)s + (x_i - \mu)\cdot(-\frac{s^2}{m}\hat x_j) = s\delta_{ij} - \frac sm - \frac sm\hat x_i\hat x_j$. then $\sum_i g_i\,\partial\hat x_i/\partial x_j = s g_j - \frac sm\sum_i g_i - \frac sm\hat x_j\sum_i g_i\hat x_i = s(g_j - \overline g - \hat x_j\overline{g\hat x})$.

</details>

2. **why the unbiased running variance?** a batch of $N$ examples from a population with variance $v$: show that $E[\sigma^2_\mathcal{B}] = \frac{N-1}{N}v$. by what factor would a running average of biased batch variances underestimate $v$ for $N = 4$?

<details><summary>solution</summary>

$E[\sum_i(x_i - \bar x)^2] = \sum_i E[x_i^2] - NE[\bar x^2] = N(v + \mu^2) - N(v/N + \mu^2) = (N - 1)v$, so $E[\sigma^2_\mathcal{B}] = \frac{N-1}{N}v$. for $N = 4$ that's $0.75v$: at test time every feature would be divided by $\sqrt{0.75} = 0.87$ times its true std, i.e. scaled up by 15% relative to what training saw. the factor $N/(N-1)$ removes the bias. (it's a minor effect for $N = 64$, and one of the major ones for $N = 4$: experiment 3.)

</details>

3. **a linear layer followed by batch norm is scale-invariant.** let $y = \operatorname{BN}(Wx)$ in training mode with $\epsilon = 0$. show that $\operatorname{BN}(cWx) = \operatorname{BN}(Wx)$ for $c > 0$, and that the gradient $\partial L/\partial W$ is orthogonal to $W$ (in the sense $\sum_{ij}W_{ij}\,\partial L/\partial W_{ij} = 0$) and scales like $1/c$.

<details><summary>solution</summary>

scaling a column of pre-activations by $c$ scales its mean and its std by $c$, so $\hat x$ is unchanged. since $L(cW) = L(W)$ for all $c > 0$, differentiating with respect to $c$ at $c = 1$ gives $\sum_{ij}W_{ij}\,\partial L/\partial W_{ij} = 0$. and differentiating $L(cW) = L(W)$ with respect to $W$ gives $c\,\nabla L(cW) = \nabla L(W)$, so $\nabla L(cW) = \nabla L(W)/c$. consequence: if a step makes $W$ larger, the next gradients get smaller and the *relative* change $\|\Delta W\|/\|W\|$ shrinks like $1/\|W\|^2$. a too-large learning rate inflates $\|W\|$, which automatically lowers the effective learning rate: a built-in brake, and part of why batch norm tolerates large learning rates (the lab's experiment 1).

</details>

4. **RMSNorm's backward pass.** with $r = (\overline{x^2} + \epsilon)^{-1/2}$ over $C$ features and $y_i = x_i r\gamma_i$, derive $\partial L/\partial x_j$.

<details><summary>solution</summary>

$\partial r/\partial x_j = -\frac12 r^3\cdot\frac{2x_j}{C} = -\frac{r^3x_j}{C}$. with $u_i = g_i\gamma_i$, $L$ depends on $x$ through $v_i = x_ir$, and $\partial v_i/\partial x_j = r\delta_{ij} + x_i\,\partial r/\partial x_j = r\delta_{ij} - \frac{r^3}{C}x_ix_j$. so $\partial L/\partial x_j = \sum_i u_i(r\delta_{ij} - \frac{r^3}{C}x_ix_j) = ru_j - x_jr^3\,\overline{ux}$.

</details>

## experiment

`lab.py` uses your `BatchNorm1d` and `LayerNorm` inside 8-layer ReLU networks on MNIST:

1. a learning-rate sweep with plain SGD, with no normalization, batch norm and layer norm.
2. a trained batch-norm network scored in eval mode, then in training mode with test batches of different sizes, and with test batches sorted by class.
3. the three networks trained with batches of 4.

## check your understanding

1. why does batch norm's output have exactly zero gradient along "all ones" in each column?

<details><summary>answer</summary>

adding the same constant to every example of a feature changes the batch mean by that constant and leaves $\hat x$ unchanged, so the loss can't depend on that direction. the formula shows it as the $-\overline g$ term. in a network, this means the bias of a `Linear` before a batch norm gets exactly zero gradient.

</details>

2. why does batch norm need running statistics but layer norm doesn't?

<details><summary>answer</summary>

batch norm's statistics are over the batch, which doesn't exist (or shouldn't matter) at test time, so it needs population estimates collected during training. layer norm's statistics are over the features of a single example, which are available at test time exactly as in training.

</details>

3. you fine-tune a batch-norm network on a new dataset in eval mode, by mistake. what's different from training mode?

<details><summary>answer</summary>

the normalization uses the old running statistics (from the original data) and they're never updated, so the layer is a fixed affine map. gradients flow differently too: there's no $-\overline g$ and $-\hat x\overline{g\hat x}$ projection. sometimes this is done on purpose ("frozen batch norm") when batches are too small for reliable statistics.

</details>

4. the running mean uses `momentum=0.1`. roughly how many recent batches does it average over, and what happens if you evaluate after only 3 training steps?

<details><summary>answer</summary>

an exponential average with weight 0.1 has an effective window of about $1/0.1 = 10$ batches. after 3 steps it's still $0.9^3 = 73\%$ its initial value (mean 0, variance 1), so evaluation uses statistics far from the data's: a freshly-built batch-norm network can look much worse in eval mode than in training mode for the first few dozen steps.

</details>

5. a colleague puts dropout *before* batch norm. what goes wrong at test time?

<details><summary>answer</summary>

dropout changes the variance of its output between training (inverted dropout adds variance) and evaluation (none), so the running variance batch norm collects during training doesn't match the variance it sees at test time: a train/test shift. the usual order is batch norm, then activation, then dropout, or no dropout at all in batch-norm networks (topic 03).

</details>

6. why is the biased variance used to normalize but the unbiased one stored?

<details><summary>answer</summary>

normalizing with the biased variance makes the batch's $\hat x$ have variance exactly (up to $\epsilon$) one, which is what the layer promises. the running variance estimates the *population* variance, to be used on new data, and the unbiased estimator is the right one for that (on paper 2).

</details>

7. RMSNorm and layer norm give identical outputs when? 

<details><summary>answer</summary>

when the input already has zero mean along the normalized axis (then the mean square is the variance) and layer norm's bias is zero. transformers' residual streams are roughly centred, which is part of why dropping the mean costs so little.

</details>

## common bugs

- **normalizing the wrong axis.** batch norm over axis 1 is layer norm (and vice versa). symptom: every row has mean 0 when every column should, and an example's prediction suddenly doesn't depend on the batch (or does when it shouldn't).
- **the unbiased variance in the normalization.** symptom: outputs with variance $(N-1)/N$ instead of 1, and a mismatch with PyTorch around $10^{-2}$ for small batches.
- **running stats as Parameters.** symptom: the optimizer "trains" them (with weight decay!) and they drift; or `state_dict` and the parameter count include them. keep them plain arrays.
- **updating running stats in eval mode.** symptom: evaluating twice gives different answers, and validation slowly corrupts the statistics.
- **forgetting `model.eval()`.** symptom: test accuracy depends on the test batch size and on the order of the test set; with class-sorted batches it collapses (the lab).
- **a composite backward that misses a path.** writing the backward as "$g\gamma s$" only (ignoring that $\mu$ and $\sigma$ depend on $x$) gives the eval-mode gradient in training mode. symptom: gradcheck fails by a lot, training is unstable.

## going deeper

- Ioffe & Szegedy (2015), "Batch normalization: accelerating deep network training by reducing internal covariate shift", *ICML*. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Ba, Kiros & Hinton (2016), "Layer normalization". [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Zhang & Sennrich (2019), "Root mean square layer normalization", *NeurIPS*. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Santurkar, Tsipras, Ilyas & Madry (2018), "How does batch normalization help optimization?", *NeurIPS*. argues the benefit is a smoother loss landscape, not reduced covariate shift. [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
- Wu & He (2018), "Group normalization", *ECCV*. a middle ground for small batches. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
- van Laarhoven (2017), "L2 regularization versus batch and weight normalization". the scale invariance of on paper 3 and its effect on the learning rate. [arXiv:1706.05350](https://arxiv.org/abs/1706.05350)

## connections

- ← builds on: `Tensor.from_op` (ch 14 · topic 03), `Module`, `Parameter` and the train/eval switch (ch 14 · topic 04), biased and unbiased variance (ch 05), exponential moving averages (ch 06), the variance law (topic 01).
- → used in: the dropout-and-normalization interaction (topic 03), the gradient-parity test of `mlfs.torchnn.to_torch`, which converts your `BatchNorm1d` and `LayerNorm` with their running statistics (topic 07), `torchnn.MLP(norm=...)` and `torchnn.RMSNorm` (topic 07), the capstone's planted `eval()` bug, batch norm in CNNs and ResNets (ch 16), layer norm in transformers (ch 18), RMSNorm in Llama (ch 27).
