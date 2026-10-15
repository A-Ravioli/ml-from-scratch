# debugging training

## why this matters

when ordinary code is wrong, it usually crashes, or a test fails. when training code is wrong, it usually runs to completion and produces a model that's a bit worse than it should be, or no better than chance, and the only symptom is a number at the end. a learning rate ten times too large, a missing `zero_grad`, a bias initialized to $-3$, labels shuffled by a data-loading bug: every one of them produces a loss curve that goes *somewhere*, and a newcomer stares at it and adjusts the learning rate.

experienced practitioners debug training the way doctors diagnose: they take measurements whose normal ranges they know, and they know which disease produces which combination of abnormal readings. the loss curve is one reading. the others live inside the network: how large each layer's gradient is, how much each layer's weights change per step relative to their size, how many of its ReLUs are permanently off. this topic builds an instrument that records all of them at every step, and a `summary()` that turns the classic signatures into warnings. the lab then gives you six broken training scripts to diagnose, and shows you the two bugs no instrument of this kind can see.

## warm-up: predict first

1. a 10-class classifier's loss sits at 2.30 for the whole run. name two completely different bugs that produce exactly this.
2. a weight matrix has norm 10, and one SGD step changes it by a matrix of norm 0.01. is the learning rate too large, too small, or about right?
3. what does a ReLU unit that outputs 0 for every example in every batch contribute to the gradient of the weights *feeding into* it?
4. in a 10-layer sigmoid network, which layer's weights get the smallest gradient: the first or the last? by roughly how much?
5. can a diagnostic that only looks at the gradients and weights notice that the labels have been shuffled?

## prerequisites check

- [ ] you can read training and validation curves, and know what overfitting looks like (→ ch 07 · topic 04)
- [ ] you know why sigmoids saturate and ReLUs can die (→ ch 14 · topics 01 and 04, topic 01 of this chapter)
- [ ] you know the global gradient norm and gradient clipping (→ topic 04)
- [ ] your `Module` has `named_parameters()`, and layers are called through `__call__` → `forward` (→ ch 14 · topic 04)

## core ideas

### idea 1: what to measure, and where

**the loss curve first.** most bugs have a recognizable curve. flat at $\log k$ (2.30 for 10 classes) from the first step: nothing is being learned at all (dead network, learning rate far too small, a disconnected graph). falling, then jumping up and never recovering: an instability, usually the learning rate. falling fast to near zero while the validation loss rises: overfitting, or a leak. a curve whose noise grows over time: the steps are getting larger, not smaller. ch 14's two sanity checks still come first: the initial loss should be about $\log k$, and the network should be able to overfit a single batch.

**the global gradient norm**, $\|g\| = \sqrt{\sum_p\|g_p\|^2}$, is the one-number summary of the backward pass: spikes mean exploding gradients, a slow collapse towards zero means the network has stopped learning. `grad_norm(params)` computes it (as `clip_grad_norm` in topic 04 did), skipping parameters with no gradient.

**per layer.** the loss and the global norm can look fine while one layer is broken: a first layer whose gradient vanishes still lets the last layers fit something. so the instrument needs to see each layer separately, which means walking the model. **`named_modules(model)`** returns `(dotted name, module)` for the model itself (named `''`) and every sub-module, depth first, in assignment order, with the same naming rule as `named_parameters` (attributes, and lists or tuples of modules as `name.i`). for a `Sequential` that's `''`, `'0'`, `'1'`, …; it's how the logger will find every ReLU.

> **▶ build: stage 1.** implement `grad_norm(params)` and `named_modules(model)`.

### idea 2: two per-layer health numbers

**the update-to-weight ratio.** a gradient's size means little on its own: $\|g\| = 0.1$ is huge for a weight matrix of norm 0.01 and negligible for one of norm 100. what matters is how much a step changes the weights *relative to their size*:

$$\text{update ratio} = \frac{\|W_\text{after} - W_\text{before}\|}{\|W_\text{after}\|}.$$

it folds in the learning rate, the optimizer (momentum, Adam's normalization) and the gradient, and it's comparable across layers and models. Karpathy's rule of thumb is about $10^{-3}$ per step: much smaller ($10^{-5}$ and below) and the weights barely move, so the learning rate is too small or the gradient vanishes; much larger ($10^{-1}$) and every step rewrites the layer, so training is unstable. the lab checks the rule. `update_ratio(before, after)` returns 0.0 if both arrays are zero.

**the dead fraction.** a ReLU unit whose pre-activation is negative for every input outputs 0 and has gradient 0: it passes no signal forward and no gradient backward to the weights feeding it, and since nothing changes those weights, it stays dead. a few dead units are normal; most of a layer dead means the layer is gone. `dead_fraction(activations)` takes a layer's output for one batch, shape `(batch, units, ...)` (extra axes count as more units), and returns the fraction of units that are exactly zero for **every** example in the batch. a unit that's zero for some examples is doing its job: that's what ReLUs do.

> **🔮 predict:** in a healthy ReLU layer with kaiming init, about what fraction of the individual outputs in a batch are exactly zero? and what `dead_fraction` do you expect for that layer?

> **▶ build: stage 2.** implement `update_ratio(before, after)` and `dead_fraction(activations)`.

### idea 3: the logger

**`DiagnosticsLogger(model)`** watches an `mlfs.nn` model (for PyTorch, the same series come from `register_forward_hook` and the parameters' `.grad`). it is created once, after the model and before training, and its `log(step, loss, lr=None)` is called right after every `optimizer.step()`. it keeps everything in `history`, a dict of lists with one entry per `log` call:

| key | value at each log |
|---|---|
| `"step"`, `"loss"`, `"lr"` | the arguments (`loss` as a float, even if you pass a `Tensor`; `lr=None` as `nan`) |
| `"grad_norm"` | the global gradient norm |
| `"grad/<name>"` | $\|g\|$ for parameter `<name>` (`nan` if its `.grad` is `None`) |
| `"weight/<name>"` | $\|W\|$ |
| `"update/<name>"` | the update ratio since the previous `log` (for the first, since the logger was created) |
| `"dead/<relu name>"` | the dead fraction of that ReLU's output on the latest training batch |

the update ratio needs the weights as they were at the previous log, so the logger keeps a **copy** of every parameter's data (a reference would change along with the weights, and every ratio would be 0), taken at construction and refreshed at every `log`.

the dead fractions need something new: the logger has to see the *outputs* of every ReLU, which never leave the model's `forward`. frameworks solve this with **hooks**: PyTorch's `register_forward_hook` calls your function with each module's output. your `Module` has no hooks, but Python lets you do the same by hand: for every `ReLU` found by `named_modules`, replace that instance's `forward` attribute with a wrapper that calls the original and records `dead_fraction` of its output (the `_watch` helper is given: it writes into a dict `self._dead`, keyed by the ReLU's name, which your `__init__` creates before calling it; the given `_weights` helper reads `self.params`, the list from `named_parameters()`). since `Module.__call__` calls `self.forward(...)`, the instance attribute takes precedence over the class's method, and the model's results are unchanged. the wrapper records only in **training mode**, so validation passes (in eval mode, through your `evaluate` and `predict`) don't overwrite the training batch's numbers.

> **🔮 predict:** you forget to copy, and store `self._before[name] = p.data`. what do all the update ratios come out as, and why? (careful: `mlfs.optim` updates `p.data` in place.)

> **▶ build: stage 3.** implement `DiagnosticsLogger.__init__` and `DiagnosticsLogger.log`.

### idea 4: from numbers to a diagnosis

a dashboard is only useful if you know what to look for, so `summary()` encodes the classic signatures as rules. each warning is a string that **starts with its tag**, followed by a colon and the numbers that triggered it:

| tag | rule | typical causes |
|---|---|---|
| `non-finite` | some logged loss is `nan` or `inf` | learning rate too high, overflow, $\log 0$ |
| `exploding` | some gradient norm is more than 100× the median of all the norms logged before it, or is not finite (report the first such step) | learning rate too high, unnormalized inputs |
| `vanishing` | a weight matrix's median gradient norm is below $10^{-3}$ × the largest weight matrix's median | saturating activations, bad init, too deep without normalization |
| `update ratio` | a weight matrix's median update ratio is above $10^{-1}$ (`too large`) or below $10^{-5}$ (`too small`) | learning rate too high / too low, frozen layers |
| `dead relu` | the latest dead fraction of some ReLU is above 0.5 | large negative biases, an instability that pushed pre-activations negative |
| `plateau` | at least 10 logs, all finite, and the mean loss of the last quarter is above 0.95 × the first quarter's | learning rate too small, dead network, broken data |

"weight matrix" means a parameter with 2 or more dimensions; biases are left out of the per-layer rules, because their norms are naturally tiny and noisy. use medians, not means, for anything over time: one spike shouldn't change a verdict about 200 steps. a healthy run returns `[]`.

the rules are deliberately simple and their thresholds are conventions, not laws; real tools (TensorBoard, Weights & Biases dashboards) show the same series and leave the judgement to you. one family of bugs no rule of this kind can catch: anything that's wrong with the *data*. shuffled labels give perfectly healthy gradients, because the network is busily (and successfully) memorizing noise. only a comparison with held-out data shows it. `plot()` draws the series in four panels (loss; gradient norm per weight matrix; update ratio per weight matrix on a log scale, with the $10^{-3}$ line; dead fraction per ReLU) and returns the matplotlib `Figure`.

> **▶ build: stage 4.** implement `DiagnosticsLogger.summary` and `DiagnosticsLogger.plot`.

## worked example

**one step by hand.** a parameter $W = (3, 4)$ (norm 5) gets the gradient $g = (0, -5)$ and an SGD step with learning rate 0.1: $W \leftarrow (3, 4.5)$. the log records `grad/W` $= 5$, `weight/W` $= \|(3, 4.5)\| = 5.408$, and

$$\text{update/W} = \frac{\|(0, 0.5)\|}{5.408} = 0.0925.$$

that's about 100 times the rule of thumb: a step that changes this weight by 9% of its size. repeated every step, the weight would be rewritten every ten steps.

**a dead fraction by hand.** a ReLU's output on a batch of two examples:

$$\begin{pmatrix} 0 & 1 & 0 & 0\\ 0 & 2 & 3 & 0\end{pmatrix}.$$

units 1 and 4 are zero for both examples: dead. unit 3 is zero for the first example only: alive. the dead fraction is $2/4 = 0.5$, which is not above the `dead relu` threshold.

**a summary by hand.** a 40-step history with the loss falling from 2.0 to 0.5, gradient norms steady at 1, update ratios at $10^{-3}$ and a dead fraction of 0.1 triggers nothing. change only the loss to a constant 2.3: `plateau` (the last quarter's mean, 2.3, isn't below $0.95 \times 2.3$). change only the last gradient norm to 500: `exploding` ($500 > 100 \times 1$). change only the first layer's gradient norms to $10^{-5}$ while the others stay at 0.5: `vanishing` ($10^{-5} < 10^{-3} \times 0.5$). the tests check each rule this way, one at a time, and then on real healthy and broken runs.

## on paper

1. **SGD's update ratio.** for SGD with learning rate $\eta$, show that the update ratio of a layer is $\eta\|g\|/\|W_\text{after}\|$. a layer has $\|W\| \approx 10$ and $\|g\| \approx 1$; which learning rate gives the rule-of-thumb ratio?

<details><summary>solution</summary>

the step is $-\eta g$, so $\|W_\text{after} - W_\text{before}\| = \eta\|g\|$. for a ratio of $10^{-3}$: $\eta \cdot 1/10 = 10^{-3}$, so $\eta = 10^{-2}$. different layers with different $\|g\|/\|W\|$ get different ratios at the same learning rate, which is one reason the ratio is logged per layer.

</details>

2. **Adam's update ratio.** with Adam, each weight moves by about $\eta$ per step when its gradient is consistent (topic 04, on paper 3). estimate the update ratio of a $128 \times 128$ weight matrix with entries of size about $0.1$ at $\eta = 10^{-3}$.

<details><summary>solution</summary>

the step has $128^2$ entries of size $\le\eta$, so its norm is at most $128\eta$ (less when the per-weight steps are smaller, as they are once $\hat m$ averages noisy gradients). $\|W\| \approx 0.1 \times 128 = 12.8$. so the ratio is at most $128 \times 10^{-3}/12.8 = 10^{-2}$, in practice a few times $10^{-3}$: the lab measures $3.4 \times 10^{-3}$. it scales linearly with $\eta$, whatever the gradients' sizes.

</details>

3. **vanishing gradients through sigmoids.** the derivative of the sigmoid is at most $1/4$. bound the factor by which the gradient can shrink through 8 sigmoid layers with weights of norm about 1 per unit, and compare with the lab's 0.72 versus $8 \times 10^{-7}$.

<details><summary>solution</summary>

each layer multiplies the backward signal by $W^\top\operatorname{diag}(\sigma')$; with $\sigma' \le 1/4$ and weights that preserve norms, each layer shrinks it by at least 4: $4^{-8} = 1.5 \times 10^{-5}$ over 8 layers. the default init's weights *shrink* norms too (variance $1/(3n)$, topic 01), giving a factor $\sqrt{1/3} \times 1/4 \approx 0.14$ per layer: $0.14^8 \approx 1.6 \times 10^{-7}$. the lab's ratio, $8 \times 10^{-7}/0.72 \approx 10^{-6}$, is in that range.

</details>

4. **why a missing `zero_grad` grows the gradient norm.** without `zero_grad`, the `.grad` after step $t$ is $G_t = \sum_{s \le t}g_s$. if the per-step gradients have a common component $\bar g$ plus independent noise of variance $\sigma^2$ per coordinate, how do the mean and the noise of $G_t$ grow with $t$?

<details><summary>solution</summary>

$E[G_t] = t\bar g$ (linear growth) and the noise has variance $t\sigma^2$ per coordinate (standard deviation $\sqrt t\sigma$). as long as there's a consistent direction, $\|G_t\|$ grows linearly: a step $\eta G_t$ is like SGD with a learning rate growing linearly in $t$ along the consistent direction, a kind of momentum without decay. it works at a small $\eta$ for a while (the lab's run reaches 89%), and then the effective step becomes too large. a `summary()` rule for it: the gradient norm keeps rising over the run (e.g. the last quarter's median above 5× the first quarter's).

</details>

## experiment

`lab.py` uses your `DiagnosticsLogger` on small MNIST training runs:

1. six broken training scripts, in a hidden order: diagnose each from its summary, validation accuracy and dashboard.
2. Adam at five learning rates: validation accuracy against the median update ratio, to test the $10^{-3}$ rule of thumb.
3. the evidence that catches the two bugs the logger can't see.

## check your understanding

1. why is the update ratio a better health number than the gradient norm?

<details><summary>answer</summary>

the gradient norm's healthy range depends on the scale of the weights, the loss and the layer; the update ratio is dimensionless, includes the learning rate and the optimizer's transformation, and directly answers "how much is this layer changing?", which is comparable across layers, models and optimizers.

</details>

2. why does the logger wrap `forward` on the ReLU *instances* rather than editing the `ReLU` class?

<details><summary>answer</summary>

editing the class would affect every ReLU in every model in the process, including ones the logger isn't watching, and couldn't tell them apart by name. an instance attribute shadows the class's method for that one object only, so each watched ReLU records under its own name, and other models are untouched.

</details>

3. why does the logger record dead fractions only in training mode?

<details><summary>answer</summary>

validation and prediction run the same ReLUs on different data (often a much larger batch). if they recorded too, the latest number before a `log` might come from a validation pass, so the history would mix training and evaluation statistics depending on when you evaluated.

</details>

4. a run triggers `exploding` and `dead relu` together. which probably came first?

<details><summary>answer</summary>

the explosion. a huge gradient step pushes many pre-activations (through the weights and biases) far negative, killing those ReLUs; once dead, they pass no gradient, so after the spike the gradients become small and the run sits on a plateau. the lab's "lr too high" script has exactly this history: a gradient norm of $10^{33}$, then 91–100% dead units.

</details>

5. the logger reports nothing, the training loss falls, and the validation accuracy is 7%. what kinds of bugs remain?

<details><summary>answer</summary>

bugs in the data or the evaluation, not the optimization: labels misaligned with inputs (the lab's shuffled labels), a different preprocessing for validation than training, evaluating on the wrong split, or an accuracy function with a bug. compare training and validation loss on the same scale, and look at a few examples with their labels.

</details>

6. why use the median over the run for the update-ratio rule, not the last value?

<details><summary>answer</summary>

single steps are noisy (one batch can produce a large or tiny update), and the verdict should describe the run, not one step. the median is robust to a few spikes. the catch: after a network dies, its updates are 0 for the rest of the run, which drags the median down; the lab's Adam run at lr 0.1 has a median ratio of $10^{-3}$ for that reason, and it's the other warnings that give it away.

</details>

## common bugs

- **storing references instead of copies.** symptom: every update ratio is exactly 0, because `before` and `after` are the same (in-place updated) array.
- **logging before `optimizer.step()`.** symptom: the update ratios are one step behind (or 0 on the first log); log right after the step.
- **counting a unit as dead if it's zero for one example.** symptom: dead fractions near 0.5 for every healthy ReLU layer (about half of any ReLU's outputs are zero) and constant `dead relu` warnings.
- **recording dead fractions in eval mode.** symptom: the history jumps whenever you run validation.
- **including biases in the vanishing rule.** symptom: spurious `vanishing` warnings; bias gradients are naturally smaller than weight gradients.
- **treating the thresholds as truth.** a warning is evidence, not a verdict. confirm it on the dashboard, and remember the bugs the logger can't see.

## going deeper

- Karpathy (2019), "A recipe for training neural networks". the sanity checks, the update-ratio rule of thumb, and the habit of looking at everything. [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
- Glorot & Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks", *AISTATS*. per-layer activation and gradient statistics as a diagnostic, and sigmoid saturation. [proceedings.mlr.press/v9/glorot10a](https://proceedings.mlr.press/v9/glorot10a.html)
- Lu, Shin, Su & Karniadakis (2020), "Dying ReLU and initialization: theory and numerical examples", *Communications in Computational Physics* 28(5). [arXiv:1903.06733](https://arxiv.org/abs/1903.06733)
- Zhang, Bengio, Hardt, Recht & Vinyals (2017), "Understanding deep learning requires rethinking generalization", *ICLR*. networks fit random labels with healthy-looking training, which is why the shuffled-labels bug is invisible from the inside. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- PyTorch documentation, `torch.nn.Module.register_forward_hook`. the framework version of the instance-wrapping trick. [pytorch.org/docs/stable/generated/torch.nn.Module.html](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

## connections

- ← builds on: overfitting and validation curves (ch 07), dead ReLUs and the one-batch test (ch 14 · topic 04), initialization (topic 01), normalization (topic 02), learning rates, schedules and gradient norms (topic 04).
- → used in: comparing hyperparameter trials (topic 06), the capstone, where the logger is your first tool for finding the planted bugs, and every training run from chapter 16 on (with PyTorch, register hooks to log the same series).
