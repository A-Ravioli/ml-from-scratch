# modules, losses & training loops

## why this matters

with your `Tensor` you can already train a network: create a few weight tensors, write the forward pass by hand, call `backward`, update. try it for a three-layer network with biases and you'll be juggling six tensors, remembering which ones go to the optimizer, writing the loss's numerics again for the third time, and copying the same twelve-line loop into every experiment. then you'll want to save the trained weights, or evaluate on a test set without building a graph, or switch a later layer (dropout, batch norm: chapter 15) between its training and evaluation behaviour, and every one of those needs the same bookkeeping again.

every deep learning framework solves this the same way, and this topic builds that solution: a **`Module`** that owns its parameters and finds the parameters of its sub-modules, a small set of **layers** and **losses** written once with the numerics done right, and a **training loop** that's correct once and reused forever. the names and conventions are PyTorch's (`Linear` stores its weight as `(out, in)`, `Sequential`'s parameters are called `0.weight`, losses take logits), so that what you learn here transfers directly, and so that chapter 15 can port your `mlfs.nn` to PyTorch and check that the two agree to the last digit. you'll also meet the single most useful debugging trick in deep learning: before training on the whole dataset, make sure the model can memorize one batch.

## warm-up: predict first

1. a 10-class classifier at initialization predicts roughly uniform probabilities. what's its cross-entropy loss, as a number?
2. `nn.Linear(784, 128)`: how many parameters, and what's the weight's shape in PyTorch?
3. why do PyTorch's losses take *logits* instead of probabilities?
4. you forget `optimizer.zero_grad()` in your training loop. does training fail loudly, quietly, or not at all?
5. a network with 100,000 parameters is trained on a single batch of 32 images with *random* labels. can it reach 100% training accuracy on that batch?

## prerequisites check

- [ ] you have a working `Tensor` with `backward`, `no_grad`, `logsumexp` and integer-array indexing (→ ch 14 · topic 03)
- [ ] you know `mlfs.optim`'s optimizers (`SGD`, `Adam`) and the `.data`/`.grad` protocol they use (→ ch 06 · topic 04)
- [ ] you can derive the gradient of softmax cross-entropy, $p - e_y$ (→ ch 02 · topic 07)
- [ ] you know why a training/validation split is needed and what overfitting looks like (→ ch 07 · topics 02 and 04)
- [ ] you can write a stable softplus, $\log(1 + e^z)$ (→ ch 00 · topic 03)

## core ideas

### idea 1: modules own their parameters

a model is a function with state: its parameters. the `Module` abstraction bundles the two, and its whole job is **discovery**: given a model, find every trainable tensor inside it, however deeply nested, so that you can hand them to an optimizer (`Adam(model.parameters())`), save them (`state_dict`), or zero their gradients.

the rule is simple. a `Parameter` is a `Tensor` that always has `requires_grad=True`: it marks "this is a weight". a module's attributes are scanned in the order they were assigned (Python dicts remember insertion order, so `vars(self)` gives you that for free):

- a `Parameter` attribute named `w` is a parameter named `w`;
- a `Module` attribute named `block` contributes its own parameters, prefixed: `block.w`;
- a list or tuple of modules named `layers` contributes `layers.0.w`, `layers.1.w`, ….

anything else (arrays of constants, ints, configuration) is ignored. two more details: the same `Parameter` can appear twice (weight tying: one embedding matrix used for input and output), and it must be listed **once**, or the optimizer would update it twice per step. and the order must be deterministic, because optimizers keep per-parameter state (Adam's moments) by position in that list.

the rest of the class is small. `__call__` runs `forward`, so a module is used like a function. `train()`/`eval()` set a `.training` flag on the whole tree: some layers behave differently while training (dropout, batch norm: chapter 15), and the flag must reach every one of them. `zero_grad()` resets every `.grad` to `None`. `state_dict()` returns `{name: copy of data}`, and `load_state_dict` copies values back **in place**, into the existing arrays: anything that holds a reference to a parameter (an optimizer, a tied layer) keeps seeing the same object. it refuses mismatched names (`KeyError`) and shapes (`ValueError`), because silently broadcasting a `(1,)` array into a weight is exactly the kind of bug you'd never find.

> **▶ build: stage 1.** implement `Parameter` and `Module`: `__init__`, `__call__`, `named_parameters`, `parameters`, `train`, `eval`, `zero_grad`, `state_dict` and `load_state_dict`. (`_children` is given.)

### idea 2: layers

**`Linear`** computes $y = xW^\top + b$ for a batch $x$ of shape $(n, \text{in})$. storing $W$ as `(out, in)` rather than `(in, out)` is PyTorch's convention; it reads naturally ("row $i$ of $W$ is the weights of output unit $i$") and it's what you'll meet in every checkpoint file. its initialization matters more than it looks. with inputs of unit variance, each output is a sum of `in` terms, so its variance grows like `in` times the weights' variance; drawing weights from $U(-1/\sqrt{\text{in}}, 1/\sqrt{\text{in}})$ (variance $1/(3\,\text{in})$) keeps outputs at a sensible scale whatever the layer's width. that's PyTorch's default, and chapter 15 develops the theory (Xavier and He initialization) properly. draw the weight first, then the bias, from one `rng`, so that seeded models are reproducible.

**activations** become modules too (`ReLU`, `Tanh`, `Sigmoid`), with no parameters, so that a whole network can be written as a list of layers. that's **`Sequential`**: it stores layer $i$ as the attribute named `str(i)`, so discovery names its parameters `0.weight`, `0.bias`, `2.weight`, … (the activation at position 1 has none), exactly as `torch.nn.Sequential` does. `net[0]` is the first layer, `net[-1]` the last, and `net[:2]` a new `Sequential` that shares the same layer objects (handy for looking at hidden features).

> **🔮 predict:** a `Sequential(Linear(784, 128), ReLU(), Linear(128, 10))` network: how many parameters does `parameters()` return, and how many numbers do they hold in total?

> **▶ build: stage 2.** implement `Linear`, `ReLU`, `Tanh`, `Sigmoid` and `Sequential` (with `__getitem__` and `__len__`). layers accept an ndarray input by wrapping it as a constant `Tensor`.

### idea 3: losses that don't overflow

a loss turns predictions and targets into one number to minimize. three cover most of this course.

**mean squared error** for regression: $\frac1N\sum(\hat y - y)^2$ over all $N$ entries. the classic bug is shapes: a model that outputs `(n, 1)` against targets of shape `(n,)` broadcasts to an `(n, n)` matrix of all pairwise differences, and the "loss" becomes the distance from every prediction to every target (topic 03's lab trains a model to predict the mean that way). your `mse_loss` refuses mismatched shapes with a `ValueError` instead of broadcasting.

**cross-entropy** for $k$-class classification. the model outputs **logits** $z \in \mathbb R^k$ (unnormalized scores), the probabilities are $\operatorname{softmax}(z)$, and the loss for true class $y$ is

$$-\log\operatorname{softmax}(z)_y = \operatorname{logsumexp}(z) - z_y.$$

computing it from logits, in one expression, is the whole point (warm-up 3): $\operatorname{softmax}$ followed by $\log$ underflows to $\log 0 = -\infty$ as soon as a wrong class gets a probability below $10^{-308}$, while logsumexp minus a logit is exact for any logits. its gradient is $p - e_y$ (ch 02 · topic 07), and your engine produces that automatically from `logsumexp` and indexing. average over the batch: the gradient of a mean doesn't grow with the batch size, so the learning rate doesn't have to change when the batch does. at initialization the logits are small and $p \approx$ uniform, so the loss is about $\log k$, $\log 10 \approx 2.30$ for digits (warm-up 1): a free sanity check on every new model.

**binary cross-entropy with logits**, for one logit $z$ per example and labels $y \in \{0, 1\}$:

$$-y\log\sigma(z) - (1 - y)\log(1 - \sigma(z)) = \operatorname{softplus}(z) - yz,\qquad \operatorname{softplus}(z) = \log(1 + e^z).$$

compute softplus stably as a logsumexp of $(0, z)$: `stack([zeros, z], axis=-1).logsumexp(axis=-1)`. its gradient is $\sigma(z) - y$, the same "prediction minus target" pattern.

`MSELoss` and `CrossEntropyLoss` are module wrappers, so that a loss can be passed around like a layer.

> **▶ build: stage 3.** implement `mse_loss`, `cross_entropy`, `binary_cross_entropy_with_logits`, and the `MSELoss` and `CrossEntropyLoss` modules.

### idea 4: the training loop

every gradient-based training loop in this course has the same five lines at its heart:

```python
optimizer.zero_grad()          # 1. forget the previous step's gradients
loss = loss_fn(model(xb), yb)  # 2-3. forward pass and loss on one minibatch
loss.backward()                # 4. gradients of the loss for every parameter
optimizer.step()               # 5. update the parameters in place
```

wrapped in two loops: over epochs, and over the minibatches of one epoch (`mlfs.optim.minibatch_indices`, ch 06). the order matters. forget step 1 and gradients accumulate across steps (topic 03 made `.grad` accumulate on purpose), so each update uses the sum of every gradient so far: a strange, unstable momentum that trains badly but rarely crashes (warm-up 4, and the lab). `fit` adds the bookkeeping: `model.train()` at the start of each epoch, the epoch's training loss (the mean over all examples: weight each batch's mean loss by its size, since the last batch may be smaller), an optional validation loss after each epoch via `evaluate`, and a `callback(epoch, history)` that can stop training by returning `True` (chapter 15's early stopping plugs in here).

`evaluate` and `predict` run the model in **eval mode under `no_grad`**, in batches: no graph is recorded, so evaluating a test set costs only the forward pass, and the mode flag tells layers like dropout to switch off. they restore the model's previous mode afterwards, so calling them in the middle of training (as `fit` does for validation) doesn't leave the model in the wrong mode.

**the first thing to run with a new model: overfit one batch.** take a single batch of, say, 32 examples, and train on it alone for a few hundred steps. a network with far more parameters than examples should drive the loss to almost zero and get every example right, even with random labels (warm-up 5). if it can't, something in the pipeline is broken: gradients not flowing, the optimizer not stepping, the loss mismatched with the outputs. it takes seconds, and it catches most plumbing bugs before you waste an hour of training on them. (what it *doesn't* catch is the lab's surprise.)

> **🔮 predict:** your network passes the one-batch test. which of these bugs could still be present: a missing `zero_grad`, a missing activation function, a learning rate 1,000 times too small?

> **▶ build: stage 4.** implement `evaluate`, `predict` and `fit`.

## worked example

cross-entropy on a batch of three, with $k = 2$ classes:

$$Z = \begin{pmatrix}0&0\\2&0\\0&3\end{pmatrix},\qquad y = (0, 1, 1).$$

**loss.** row by row, $\operatorname{logsumexp}(z_i) - z_{i,y_i}$:

- row 1: $\log(e^0 + e^0) - 0 = \log 2 = 0.6931$ (a 50/50 guess);
- row 2: $\log(e^2 + 1) - 0 = 2.1269$ (confidently wrong: the model favours class 0);
- row 3: $\log(1 + e^3) - 3 = \log(1 + e^{-3}) = 0.0486$ (confidently right).

the mean is $\mathbf{0.9562}$. the confidently wrong example dominates: cross-entropy punishes confident mistakes far more than it rewards confident successes.

**gradient.** $(p_i - e_{y_i})/3$ row by row, with $p_2 = \operatorname{softmax}(2, 0) = (0.8808, 0.1192)$ and $p_3 = (0.0474, 0.9526)$:

$$\frac{\partial L}{\partial Z} = \frac13\begin{pmatrix}0.5 - 1 & 0.5\\0.8808 & 0.1192 - 1\\0.0474 & 0.9526 - 1\end{pmatrix} = \begin{pmatrix}-0.1667&0.1667\\0.2936&-0.2936\\0.0158&-0.0158\end{pmatrix}.$$

each row sums to zero (softmax can't change the total), and the biggest push goes to the confidently wrong row. the tests check the loss, this gradient, and that logits like $(1000, 0, -1000)$ give a finite loss of exactly 1000 for class 1.

## on paper

1. **parameter count.** how many parameters does an MLP with layer sizes $d_0, d_1, \dots, d_L$ (biases everywhere) have? evaluate it for the capstone's $784 \to 256 \to 128 \to 10$.

<details><summary>solution</summary>

$\sum_{l=1}^{L}(d_{l-1} + 1)d_l$. for $784 \to 256 \to 128 \to 10$: $785 \cdot 256 + 257 \cdot 128 + 129 \cdot 10 = 200{,}960 + 32{,}896 + 1{,}290 = 235{,}146$. (the predict callout in idea 2: four `Parameter`s holding $785 \cdot 128 + 129 \cdot 10 = 101{,}770$ numbers.)

</details>

2. **the variance of a linear layer's output.** inputs $x_j$ independent with mean 0 and variance 1, weights $W_{ij} \sim U(-a, a)$ independent of them, no bias. compute $\operatorname{Var}(y_i)$ for $y_i = \sum_{j=1}^{\text{in}} W_{ij}x_j$, and show that $a = 1/\sqrt{\text{in}}$ gives variance $1/3$ for any width.

<details><summary>solution</summary>

$\operatorname{Var}(U(-a, a)) = a^2/3$. each term $W_{ij}x_j$ has mean 0 and variance $E[W^2]E[x^2] = a^2/3$, and the terms are independent, so $\operatorname{Var}(y_i) = \text{in}\cdot a^2/3$. with $a = 1/\sqrt{\text{in}}$ that's $1/3$, independent of the width. (it shrinks by a factor of 3 per layer, plus whatever the activation does, which is why deep networks need the more careful schemes of chapter 15.)

</details>

3. **the binary loss.** show that $-y\log\sigma(z) - (1 - y)\log(1 - \sigma(z)) = \operatorname{softplus}(z) - yz$ and that its derivative is $\sigma(z) - y$.

<details><summary>solution</summary>

$-\log\sigma(z) = \log(1 + e^{-z}) = \operatorname{softplus}(z) - z$ and $-\log(1 - \sigma(z)) = \log(1 + e^z) = \operatorname{softplus}(z)$. so the loss is $y(\operatorname{softplus}(z) - z) + (1 - y)\operatorname{softplus}(z) = \operatorname{softplus}(z) - yz$. differentiating, $\operatorname{softplus}'(z) = e^z/(1 + e^z) = \sigma(z)$, so the derivative is $\sigma(z) - y$.

</details>

4. **why weight by batch size.** 25 examples in batches of 10, 10 and 5 with mean losses $\ell_1, \ell_2, \ell_3$. what's the epoch's mean loss, and how far off is the plain average of the three batch means?

<details><summary>solution</summary>

$(10\ell_1 + 10\ell_2 + 5\ell_3)/25$. the plain average $(\ell_1 + \ell_2 + \ell_3)/3$ counts each of the last batch's 5 examples twice as much as the others; it differs by $\frac{1}{15}(\ell_1 + \ell_2 - 2\ell_3)$, zero only if the last batch happens to have the average loss.

</details>

## experiment

`lab.py` uses your `mlfs.nn` for three experiments:

1. the one-batch sanity check, on 32 MNIST images with true and with random labels, for a correct training loop and four broken ones. which bugs does it catch?
2. the loss of an untrained network, as you scale its initial weights.
3. a learning-rate sweep with SGD: what happens when the learning rate is too large?

## check your understanding

1. why must `named_parameters` list a shared parameter only once?

<details><summary>answer</summary>

the optimizer would otherwise receive the same object twice and update it twice per step (with doubled weight decay and, for Adam, two separate sets of moments for the same weights). gradients from both uses are already summed into its single `.grad` by backward.

</details>

2. `load_state_dict` writes `p.data[...] = value` rather than `p.data = value`. what would break with the second?

<details><summary>answer</summary>

the arrays would be replaced, not filled. anything that held the old array (a view, a tied layer sharing the array, a cached copy in an optimizer or a checkpointing helper) would silently go on using the old values. writing in place keeps every reference valid. it also enforces the dtype of the existing array.

</details>

3. why does `evaluate` use `no_grad`? what would go wrong without it, and what wouldn't?

<details><summary>answer</summary>

the numbers would be identical. but every evaluation batch would build a graph and keep every intermediate activation alive until the loss tensor is dropped, costing memory and time for nothing. on a large model that's the difference between evaluation fitting in memory or not.

</details>

4. your 10-class model starts with a loss of 18. what does that tell you?

<details><summary>answer</summary>

a uniform guess scores $\log 10 = 2.30$. 18 means the initial logits are huge and confidently wrong on most examples: the weights are initialized at far too large a scale (the lab's experiment 2), or the inputs aren't normalized (pixels in 0–255 instead of 0–1).

</details>

5. why does `cross_entropy` take integer class labels rather than one-hot vectors?

<details><summary>answer</summary>

the loss only needs the true class's logit, which integer-array indexing picks out directly ($n$ numbers instead of an $n \times k$ matrix of mostly zeros). it's also PyTorch's convention. (one-hot or smoothed targets need a different formula: chapter 15's label smoothing.)

</details>

6. you pass the one-batch test with random labels. what have you shown, and what haven't you?

<details><summary>answer</summary>

that gradients flow to the parameters, the optimizer updates them, the loss is wired to the outputs, and the model has enough capacity to memorize 32 examples. you haven't shown that it generalizes, that the architecture is what you intended (a missing activation still passes), or that the learning rate is good for the full dataset.

</details>

7. `fit` calls `model.train()` at the start of every epoch rather than once at the start. why?

<details><summary>answer</summary>

validation (via `evaluate`) and any callback may switch the model to eval mode. restoring training mode at the top of every epoch keeps the next epoch correct regardless of what happened in between.

</details>

## common bugs

- **forgetting `zero_grad`.** symptom: training is erratic and stalls well above zero loss, but nothing crashes. the gradients of all previous steps pile up in `.grad`.
- **`(in, out)` weights.** symptom: everything works until you load PyTorch weights, or compare with chapter 15's port, and the shapes are transposed. store `(out, in)` and compute `x @ W.T`.
- **softmax, then log.** symptom: `nan` or `inf` losses once the model gets confident. compute cross-entropy as logsumexp minus the true logit.
- **an unweighted epoch mean.** symptom: the reported training loss is slightly off whenever the batch size doesn't divide the dataset, and doesn't match `evaluate` on the same data.
- **evaluating in training mode or with a graph.** symptom: memory grows during validation; later, in chapter 15, dropout stays on at test time and accuracy drops.
- **target shapes.** symptom: a regression that predicts the mean of $y$ for every input. compare shapes, don't broadcast.

## going deeper

- Paszke et al. (2019), "PyTorch: an imperative style, high-performance deep learning library", *NeurIPS*. the design of `nn.Module`, eager execution and the training loop you've rebuilt. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
- Karpathy (2019), "A recipe for training neural networks". the one-batch overfit test and a dozen more sanity checks, with reasons. [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
- Zhang, Bengio, Hardt, Recht & Vinyals (2017), "Understanding deep learning requires rethinking generalization", *ICLR*. networks memorize random labels on whole datasets, not just one batch. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, §6.2 (output units and cost functions: why cross-entropy with logits). [deeplearningbook.org](https://www.deeplearningbook.org)
- Prince (2023), *Understanding Deep Learning*, MIT Press, chapters 5–6 (loss functions and fitting models). [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)

## connections

- ← builds on: your `Tensor`, `no_grad` and `stack` (topic 03), `mlfs.optim`'s optimizers and `minibatch_indices` (ch 06 · topic 04), `as_rng` (ch 00 · topic 05), softmax cross-entropy and its gradient (ch 02 · topic 07), train/validation splits (ch 07).
- → used in: universal approximation experiments (topic 05), checkpointed layers (topic 06), the MNIST capstone, initializers, normalization layers, dropout, schedulers and diagnostics built on `Module` and `fit` (ch 15), the PyTorch port with a gradient-parity test (ch 15 · topic 07).
