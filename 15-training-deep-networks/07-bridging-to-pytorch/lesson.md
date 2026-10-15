# bridging to PyTorch

## why this matters

everything in chapters 14 and 15 runs on an engine you wrote: about 500 lines of NumPy that record operations, differentiate them and train networks. it's correct (you've checked it against finite differences hundreds of times), and it's surprisingly fast for small models. but chapters 16 to 28 train convolutional networks, recurrent networks, transformers and diffusion models, with operations you haven't written, on GPUs your NumPy can't use. that's what PyTorch is for. the course's rule is *build before you use*: you may use a PyTorch component once you've built it yourself. you have now built the core of PyTorch, so it's time to switch, carefully.

carefully means two things. first, a **map** between the two frameworks: which of your classes corresponds to which of theirs, where the conventions agree (by design, almost everywhere) and where they differ (float32 by default, devices, how modules register their parameters). second, **evidence**: a converted network should compute the same loss and the same gradients as yours, not approximately but to the last few digits of float64. that's the gradient-parity test, and it's the strongest check your engine will ever get. then you'll write the small toolkit, **`mlfs.torchnn`**, that every later chapter trains with: an `MLP`, an `RMSNorm`, data loaders, a `fit`/`evaluate` loop that works with your schedulers, and the exponential moving average of weights that chapters 21–27 rely on.

## warm-up: predict first

1. your `Linear` stores its weight as `(out_features, in_features)`. what shape is `torch.nn.Linear(784, 128).weight`?
2. your engine computes in float64. PyTorch's default is float32. how many significant decimal digits does each carry?
3. two programs compute the same matrix product in float64 with different libraries. can the results differ? by roughly how much?
4. `torch.nn.Module` finds its parameters without scanning `vars(self)`. how might it know about them?
5. a model's weights are averaged with `ema = 0.99 * ema + 0.01 * weights` after every step. over roughly how many recent steps is that an average?

## prerequisites check

- [ ] your `mlfs.tensor`, `mlfs.nn` and their conventions: `(out, in)` weights, `Sequential` names like `0.weight`, logits into losses (→ ch 14 · topics 03 and 04)
- [ ] your `BatchNorm1d`, `LayerNorm`, `RMSNorm` and `Dropout`, with running statistics as buffers (→ topics 02 and 03)
- [ ] your schedulers drive optimizers through `get_lr`/`set_lr` (→ topic 04)
- [ ] floating-point precision and rounding error (→ ch 00 · topic 03)

## core ideas

### idea 1: the map, and `to_torch`

your framework was built with PyTorch's conventions on purpose, so most of the map is one-to-one:

| yours (`mlfs`) | PyTorch | differences |
|---|---|---|
| `Tensor(data, requires_grad)` | `torch.tensor(data, requires_grad=...)` | default dtype float32, not float64; tensors live on a device |
| `.backward()`, `.grad` (accumulates), `no_grad()` | the same names | torch can also free the graph early, and `zero_grad()` sets grads to `None` by default, like yours |
| `Tensor.from_op(data, parents, vjp)` | `torch.autograd.Function` with `forward`/`backward` | the same idea: a custom node with a hand-written VJP |
| `Module`, `Parameter`, `train()`/`eval()`, `state_dict` | `nn.Module`, `nn.Parameter`, the same | torch registers parameters when they're assigned, not by scanning |
| `Linear` (weight `(out, in)`, $U(\pm 1/\sqrt{n_\text{in}})$ init) | `nn.Linear` | the same layout and the same init distribution |
| `Sequential` (layer $i$ named `str(i)`) | `nn.Sequential` | the same names: `0.weight`, `2.bias` |
| `BatchNorm1d` (running stats as plain arrays) | `nn.BatchNorm1d` (running stats as **buffers**) | buffers are registered tensors that move with the module and appear in its `state_dict` |
| `cross_entropy(logits, targets)` | `F.cross_entropy` | the same: logits in, integer targets, mean |
| `mlfs.optim.Adam(params, lr)`, `.lr` | `torch.optim.Adam(params, lr=...)`, `param_groups` | your `get_lr`/`set_lr` (topic 04) hide the difference |

two differences deserve a sentence each. **registration**: `nn.Module` overrides `__setattr__`, so `self.fc = nn.Linear(...)` records `fc` in an internal dictionary of sub-modules at the moment of assignment; your `Module` instead scans `vars(self)` whenever it's asked. both give the same names. **buffers**: state that isn't trained but belongs to the model (batch norm's running statistics) is registered with `register_buffer`, so `.to(device)`, `state_dict()` and `ema_update` (idea 5) all see it.

**`to_torch(module)`** walks an `mlfs` network and builds the equivalent PyTorch module, copying every weight: `Sequential` → `nn.Sequential` (recursively), `Linear` → `nn.Linear` (copy weight and bias as they are: same layout, no transpose), `ReLU`/`Tanh`/`Sigmoid` → the torch modules of the same names, your `Dropout(p)` → `nn.Dropout(p)`, `LayerNorm` and `BatchNorm1d` → the torch layers with the same `eps` (and `momentum`), copying `weight`, `bias` and, for batch norm, the running statistics into torch's buffers, and your `RMSNorm` → the torch `RMSNorm` of idea 3. everything is created in **float64**, so the comparison in idea 2 is exact; call `.float()` on the result to train in float32. the converted module is put in the same mode as the original (`train()` or `eval()`). dispatch on `type(module).__name__`, so the function works for your classes and for the reference ones alike; read sizes from the weight's shape (`n_out, n_in = weight.shape`); raise `TypeError` for anything you don't know how to convert.

> **▶ build: stage 1.** implement `to_torch(module)`.

### idea 2: gradient parity

a gradient check (chapter 14) compares your backward pass with finite differences, which agree to about $10^{-6}$ relative error at best, because finite differences are approximations. a **parity test** compares two exact implementations of the same mathematics, so the only differences left are rounding: in float64, about $10^{-16}$ relative to the numbers involved. if a single VJP anywhere in your engine were subtly wrong (a missing term in batch norm's backward, a broadcasting gradient summed over the wrong axis), parity would show it as a difference many orders of magnitude above $10^{-12}$.

**`gradient_parity(model, X, y)`** returns a dict with the largest absolute differences in the `"loss"`, the `"output"` and the `"grad"` (over every entry of every parameter, matched by name: the names agree by construction):

1. convert **first**: `theirs = to_torch(model)`, so both copies start from the same state (a batch-norm forward in training mode updates running statistics).
2. run your side on a **copy** (`copy.deepcopy(model)`), so the caller's model is left untouched: no gradients left behind, no running statistics moved.
3. the loss is `cross_entropy` for integer targets and `mse_loss` otherwise, on both sides.
4. backward on both, then compare.

why can't float64 results be *exactly* equal? floating-point addition isn't associative, and the two libraries add the terms of each matrix product in different orders (blocked, vectorized, in parallel). the lab trains the two copies side by side for 300 steps and watches how those $10^{-16}$ differences evolve.

> **🔮 predict:** you convert a network to float32 and repeat the parity test. what size of differences do you expect, and would a difference of $10^{-5}$ in a gradient mean your engine has a bug?

> **▶ build: stage 2.** implement `gradient_parity(model, X, y)`.

### idea 3: torch modules of your own

a PyTorch module is a subclass of `nn.Module` with an `__init__` that calls `super().__init__()` and assigns sub-modules and parameters, and a `forward`. three pieces of the toolkit:

- **`MLP(in_features, hidden_sizes, out_features, *, activation='relu', dropout=0.0, norm=None)`**: for every hidden size, `Linear → [norm] → activation → [dropout]`, then a plain `Linear` output, stored as `self.net = nn.Sequential(...)`. activations `'relu'`, `'gelu'`, `'tanh'`, `'silu'`; norms `None`, `'layer'` (`nn.LayerNorm`), `'batch'` (`nn.BatchNorm1d`); unknown names raise `ValueError`. putting the norm before the activation is the classic ordering of topic 02. it's the workhorse for every small network in later chapters: value functions in RL, probes on frozen features, the heads of larger models.
- **`RMSNorm(dim, eps=1e-6)`**: topic 02's RMSNorm as a torch module, with a `weight` of ones, $\epsilon$ inside the square root exactly as in topic 02. one twist: a **float16 or bfloat16** input is upcast to float32 (`x.float()`) for the normalization and the result cast back to `x`'s dtype before multiplying by the weight; float32 and float64 inputs are computed in their own dtype, so float64 parity with your `mlfs.train.RMSNorm` stays exact. large models train in 16-bit precision (chapter 26), where squaring and averaging activations loses too many digits, or overflows ($300^2$ is beyond float16's 65,504); doing that one reduction in float32 is cheap insurance. note the dtype rules: a half-precision input times the float32 `weight` gives a float32 result (cast the module with `.half()`/`.to(torch.bfloat16)` to keep everything in 16 bits, as Llama-style code does). chapter 27's Llama uses this module.
- **`count_parameters(model, trainable_only=True)`**: the sum of `p.numel()` over the parameters (only those with `requires_grad`, by default, so frozen layers don't count).

and one function for hardware. a module and the tensors it's applied to must live on the same **device** (`model.to(device)`, `x.to(device)`). **`get_device(prefer=None)`** returns `torch.device(prefer)` if you ask for one, else the best available: `'cuda'` (an NVIDIA GPU), then `'mps'` (Apple's GPU), then `'cpu'`.

> **▶ build: stage 3.** implement `RMSNorm`, `MLP`, `count_parameters` and `get_device`.

### idea 4: data and the training loop, PyTorch-style

PyTorch separates data from the loop. a `Dataset` knows how to return example $i$; a `DataLoader` batches, shuffles and (optionally) loads in parallel. for arrays already in memory, `TensorDataset(X, y)` is all you need. **`make_loader(X, y=None, batch_size=64, shuffle=True, seed=None)`** converts the arrays (features to float32; integer labels to int64, which `F.cross_entropy` requires; float targets to float32), and, if `seed` is given, passes a seeded `torch.Generator` so the shuffling is reproducible. without `y`, batches are 1-tuples `(x,)`.

**`fit(model, loss_fn, optimizer, train_loader, *, epochs=1, val_loader=None, scheduler=None, grad_clip=None, device='cpu', callback=None)`** is chapter 14's loop in PyTorch: move the model to the device; for each epoch, `model.train()`, then for each batch move it to the device, `optimizer.zero_grad()`, forward, loss, `backward()`, clip with `torch.nn.utils.clip_grad_norm_` if `grad_clip` is set (after `backward`, before `step`: topic 04), `optimizer.step()`, and `scheduler.step()` once per optimizer step. record the epoch's mean training loss (weighted by batch size, using `.item()` so no graph is kept alive), the validation loss if there's a `val_loader`, and the learning rate after the epoch; then `callback(epoch, history)` may stop training by returning `True` (your `EarlyStopping` from topic 03 plugs straight in). because the scheduler is only ever asked to `step()`, both PyTorch's schedulers and yours from topic 04 work.

**`evaluate(model, loader, loss_fn=None, *, device='cpu')`** runs under `torch.no_grad()` in eval mode, restores the previous mode afterwards, and returns a dict: `"loss"` (the mean over examples, if `loss_fn` is given) and `"accuracy"` (if the outputs are `(B, C)` and the targets are integers).

> **▶ build: stage 4.** implement `make_loader`, `evaluate` and `fit`.

### idea 5: exponential moving averages of weights

a constant learning rate leaves SGD bouncing around a minimum (topic 04, on paper 1). one cheap way to land closer to the middle of the bounce: keep a second copy of the model whose weights are an **exponential moving average** of the trained weights,

$$\theta_\text{ema} \leftarrow \delta\,\theta_\text{ema} + (1 - \delta)\,\theta,$$

updated after every optimizer step. with decay $\delta$, the average covers roughly the last $1/(1 - \delta)$ steps (100 for $\delta = 0.99$, 1,000 for $0.999$). it's Polyak averaging with a forgetting factor, and it's everywhere in modern deep learning: evaluation copies of image and diffusion models (chapters 21–22), the slowly moving **target networks** of DQN and BYOL (chapters 24 and 25, which call it a "soft update" with $\tau = 1 - \delta$), and the teacher in self-distillation.

**`ema_update(target, source, decay=0.999)`** does it in place under `torch.no_grad()`, pairing `target.parameters()` with `source.parameters()` in order, and **copies** the buffers (batch norm's running statistics describe the current weights' activations, so averaging them would describe a network that never existed). it's in place, so the target stays a leaf tensor and no graph is built.

> **🔮 predict:** a model is trained with a constant learning rate, and an EMA copy (decay 0.99) follows it. the EMA copy is never trained itself. will it score better or worse on the test set than the model it averages?

> **▶ build: stage 5.** implement `ema_update(target, source, decay=0.999)`.

## worked example

**counting parameters.** `MLP(784, [256, 128], 10, norm='batch')` is `Linear(784, 256)`, `BatchNorm1d(256)`, `ReLU`, `Linear(256, 128)`, `BatchNorm1d(128)`, `ReLU`, `Linear(128, 10)`:

$$\underbrace{784 \cdot 256 + 256}_{200{,}960} + \underbrace{2 \cdot 256}_{512} + \underbrace{256 \cdot 128 + 128}_{32{,}896} + \underbrace{2 \cdot 128}_{256} + \underbrace{128 \cdot 10 + 10}_{1{,}290} = 235{,}914.$$

the running statistics ($2 \times 256 + 2 \times 128$ numbers) are buffers, not parameters, so `count_parameters` doesn't count them. compare chapter 14's capstone network, $235{,}146$ parameters: batch norm adds 768.

**an EMA by hand.** a target parameter starts at 0 and the source sits at 1, with $\delta = 0.9$. after one update the target is $0.1$, after two $0.9 \cdot 0.1 + 0.1 = 0.19$, after three $0.271$: in general $1 - 0.9^k$. it takes $k = 22$ updates to get within 10% of the source, and an EMA with $\delta = 0.999$ takes 2,300. that lag is the point (it averages over the recent past) and the price (early in training the EMA copy is still near its initial weights, which is why its accuracy starts lower).

**precision.** float64 has a 52-bit mantissa, a machine epsilon of $2.2 \times 10^{-16}$, about 16 significant digits; float32 has 23 bits, $1.2 \times 10^{-7}$, about 7 digits. so parity in float64 should hold to $\sim 10^{-15}$ relative error, and a float32 copy differs from the float64 one by $\sim 10^{-7}$ from the start, just from rounding the weights. the tests ask for $10^{-10}$ in float64; the lab measures both.

## on paper

1. **why rounding differs between libraries.** compute $(1 + 10^{16}) - 10^{16}$ and $1 + (10^{16} - 10^{16})$ in float64. what does this say about summing the 784 terms of a dot product in two different orders?

<details><summary>solution</summary>

the spacing between float64 numbers near $10^{16}$ is 2, so $1 + 10^{16}$ rounds to $10^{16}$ and the first expression gives $0$; the second gives $1$. addition isn't associative: the result depends on the order. a dot product of 784 terms summed in different orders (a BLAS kernel sums in blocks and in parallel) typically differs by a few units in the last place, about $10^{-16}$ relative. both are correct to within rounding; neither is "the" answer.

</details>

2. **the effective window of an EMA.** show that the EMA is $\theta_\text{ema}^{(t)} = (1 - \delta)\sum_{k \ge 0}\delta^k\theta^{(t - k)}$ (ignoring the initial value), that the weights sum to 1, and that their mean age is $\delta/(1 - \delta)$.

<details><summary>solution</summary>

unrolling $\theta_\text{ema}^{(t)} = \delta\theta_\text{ema}^{(t-1)} + (1 - \delta)\theta^{(t)}$ gives the sum. the weights sum to $(1 - \delta)\sum_k\delta^k = 1$. the mean age is $(1 - \delta)\sum_k k\delta^k = (1 - \delta)\frac{\delta}{(1 - \delta)^2} = \frac{\delta}{1 - \delta}$: 99 steps for $\delta = 0.99$, about $1/(1 - \delta)$.

</details>

3. **EMA of a noisy random walk around a minimum.** suppose the trained weight is $\theta^{(t)} = \theta^* + \xi_t$ with independent noise of variance $s^2$. what's the variance of the EMA (in steady state)?

<details><summary>solution</summary>

$\operatorname{Var} = (1 - \delta)^2\sum_k\delta^{2k}s^2 = \frac{(1 - \delta)^2}{1 - \delta^2}s^2 = \frac{1 - \delta}{1 + \delta}s^2$. for $\delta = 0.99$ that's $s^2/199$: the EMA sits about $\sqrt{199} \approx 14$ times closer to $\theta^*$ than the bouncing weights. (real SGD noise is correlated from step to step, so the gain is smaller, but the direction is the same.)

</details>

4. **why float32 for the normalization.** in 16-bit floats (bfloat16 has an 8-bit mantissa, about 3 significant digits), the mean of $x^2$ over 4,096 features is accumulated term by term. roughly when does adding one more term stop changing the sum?

<details><summary>solution</summary>

with an 8-bit mantissa, adding a term smaller than about $2^{-8} \approx 0.4\%$ of the running sum doesn't change it. if every $x_i^2$ is about 1, the sum stops growing at about 256: the remaining 3,840 terms are lost and the "mean" is off by a factor of 16. accumulating in float32 (23 bits) keeps every term up to sums of millions, which is why `RMSNorm` upcasts 16-bit inputs to float32 and casts back.

</details>

## experiment

`lab.py` puts your two engines side by side:

1. an `mlfs` network and its `to_torch` copy trained in lockstep for 300 steps, in float64 and in float32.
2. the time of one training step in your engine, PyTorch on the CPU (float32 and float64) and PyTorch on the accelerator, across widths.
3. a model and the exponential moving average of its weights, scored on the test set after each epoch.

## check your understanding

1. why does `to_torch` create float64 modules when PyTorch's default is float32?

<details><summary>answer</summary>

to compare like with like: your engine computes in float64, so a float64 torch copy makes any difference beyond $\sim 10^{-15}$ a real discrepancy. in float32 every result differs by $\sim 10^{-7}$ from rounding alone, which would hide small bugs. for training afterwards, `.float()` converts it.

</details>

2. what does a passing parity test prove that a passing gradient check doesn't?

<details><summary>answer</summary>

a gradient check confirms your backward pass matches *your* forward pass, to finite-difference accuracy ($\sim 10^{-6}$); it can't catch a forward pass that computes the wrong function consistently (say, batch norm with the unbiased variance), and its tolerance hides small errors. parity compares the whole computation, forward and backward, with an independent implementation to $10^{-12}$: same function, same derivatives, same conventions.

</details>

3. why does `gradient_parity` convert the model *before* running the mlfs side, and run the mlfs side on a copy?

<details><summary>answer</summary>

a forward pass in training mode updates batch norm's running statistics; converting afterwards would give the torch copy different statistics than the mlfs copy started with (it doesn't matter for this pass's outputs, but it does for an eval-mode comparison later). running on a copy means the caller's model keeps its gradients (none) and its statistics exactly as they were: a test shouldn't change what it tests.

</details>

4. you call `fit` with `device='cuda'` but forget to move the validation data. what happens, and where?

<details><summary>answer</summary>

PyTorch raises a `RuntimeError` ("expected all tensors to be on the same device") the first time the model, now on the GPU, receives a CPU tensor, which is why `fit` and `evaluate` move every batch with `.to(device)`. unlike a silent dtype mismatch, device mismatches always fail loudly.

</details>

5. why does `fit` accumulate `loss.item()` instead of `loss`?

<details><summary>answer</summary>

`loss` is a tensor attached to the computation graph; adding it to a running total keeps every batch's graph (and all its activations) alive until the end of the epoch, a memory leak that grows with the epoch's length. `.item()` returns a Python float and lets the graph be freed.

</details>

6. `ema_update` copies buffers rather than averaging them. what would go wrong with averaged running statistics?

<details><summary>answer</summary>

running statistics estimate the mean and variance of each layer's activations *for the current weights*. the EMA's weights are an average of past weights, whose activations' statistics are not the average of the past statistics (the map from weights to activation statistics is nonlinear). copying the source's is an approximation too (the right statistics would be re-estimated by running data through the EMA model), but a much better one, and the convention in DQN/BYOL implementations.

</details>

## common bugs

- **transposing Linear weights.** symptom: a shape error, or worse, a square layer that silently computes $xW$ instead of $xW^\top$. both frameworks store `(out, in)`: copy as is.
- **dtype mismatches.** feeding a float64 tensor to a float32 model raises an error; integer labels as float32 make `F.cross_entropy` treat them as probabilities (or fail). `make_loader` converts both.
- **forgetting `model.eval()` / `torch.no_grad()` in evaluation.** symptom: dropout on at test time, batch norm using test-batch statistics, and graphs built for nothing. `evaluate` does both and restores the mode.
- **stepping the scheduler per epoch.** symptom: a per-step schedule that barely moves. `fit` steps it once per optimizer step.
- **timing the GPU without synchronizing.** symptom: impossibly fast GPU numbers; GPU calls are asynchronous, so time after `torch.cuda.synchronize()` (or `torch.mps.synchronize()`).
- **averaging buffers in an EMA.** symptom: an EMA model whose batch norm statistics don't match its weights, and an evaluation accuracy far below the trained model's.

## going deeper

- Paszke et al. (2019), "PyTorch: an imperative style, high-performance deep learning library", *NeurIPS*. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
- PyTorch documentation, "Autograd mechanics" and "Extending PyTorch" (custom `autograd.Function`s, the counterpart of your `Tensor.from_op`). [pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)
- Polyak & Juditsky (1992), "Acceleration of stochastic approximation by averaging", *SIAM Journal on Control and Optimization* 30(4): 838–855. [doi:10.1137/0330046](https://doi.org/10.1137/0330046)
- Izmailov, Podoprikhin, Garipov, Vetrov & Wilson (2018), "Averaging weights leads to wider optima and better generalization", *UAI*. stochastic weight averaging. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
- Goldberg (1991), "What every computer scientist should know about floating-point arithmetic", *ACM Computing Surveys* 23(1): 5–48. [doi:10.1145/103162.103163](https://doi.org/10.1145/103162.103163)
- Micikevicius et al. (2018), "Mixed precision training", *ICLR*. float16 training and where float32 is still needed. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

## connections

- ← builds on: your `mlfs.tensor` and `mlfs.nn` (ch 14), normalization layers and dropout (topics 02–03), schedulers, clipping and `get_lr`/`set_lr` (topic 04), early stopping as a callback (topic 03), floating point (ch 00 · topic 03).
- → used in: every chapter from 16 on trains with `mlfs.torchnn` (`MLP`, `make_loader`, `fit`, `evaluate`, `get_device`); the capstone's pipeline; `torchnn.RMSNorm` in Llama (ch 27); `ema_update` for DQN target networks (ch 24), BYOL (ch 25) and EMA models in diffusion (ch 22); mixed precision (ch 26).
