# 15 · training deep networks

> make deep networks actually train, and diagnose them when they don't.

chapter 14 ended with a three-layer network at 98% on MNIST, trained by code you wrote from the ground up. make it twenty layers deep and the same code stalls at 10%, with nothing crashing and nothing obviously wrong. this chapter is about the gap between a network that *can* represent a function and one that actually learns it, and about the habits that tell you which side of the gap you're on. you'll derive the initialization that keeps a signal alive through fifty layers, and see why that's not enough on its own; build batch, layer and RMS normalization with backward passes you derive by hand; regularize with weight decay (and find out why Adam and an L2 penalty don't mix), dropout, label smoothing, augmentation and early stopping; turn the learning rate into a schedule, find a good one in a hundred minibatches, and clip the gradients that would wreck it; build an instrument panel that diagnoses broken training runs from their gradients, weights and dead units; search hyperparameters with hyperband, and learn how much of any difference is just the seed; and finally move to PyTorch, proving with a gradient-parity test that your engine and theirs compute the same thing. the capstone hands you an MNIST pipeline that runs without errors and contains ten bugs. on the way you'll meet a PyTorch default init that makes every image look identical by layer 10, an L2 penalty that wipes out the weights it shouldn't touch, a batch-norm network that drops from 86% to 16% because nobody called `model.eval()`, a warmup that turns 10% into 89%, two broken scripts that look perfectly healthy to every diagnostic, a coin-flip's worth of agreement between single-seed comparisons, and a NumPy engine that keeps pace with PyTorch on small networks.

**you will build:** `calculate_fan`/`calculate_gain`, xavier and kaiming initializers, `deep_mlp` and `activation_stats` (topic 01); `standardize` and its backward pass, `BatchNorm1d`, `LayerNorm`, `RMSNorm` as single `Tensor.from_op` nodes (02); `l2_penalty`, `Dropout`, `label_smoothing_cross_entropy`, `random_crop`, `random_horizontal_flip`, `EarlyStopping` (03); `get_lr`/`set_lr`, `LRScheduler`, `StepLR`, `CosineSchedule` with warmup, `clip_grad_norm`, `lr_range_test`, `suggest_lr` (04); `grad_norm`, `named_modules`, `update_ratio`, `dead_fraction`, `DiagnosticsLogger` with forward hooks, a rule-based `summary()` and a dashboard (05); `random_search`, `successive_halving`, `hyperband`, `ExperimentLog` (06); `to_torch`, `gradient_parity`, and PyTorch's `MLP`, `RMSNorm`, `make_loader`, `fit`, `evaluate`, `ema_update` (07). modules: **`mlfs.train`** (topics 01–06, framework-agnostic where it can be: the schedulers, `EarlyStopping` and the searches drive PyTorch optimizers too) and **`mlfs.torchnn`** (topic 07), the PyTorch toolkit that chapters 16–28 train with.
**prerequisites:** chapter 14 (your `mlfs.tensor` with `Tensor.from_op`, and `mlfs.nn`'s `Module`, layers, losses and `fit`), SGD, momentum, Adam and AdamW (ch 06 · topic 04), validation sets, overfitting and grid search (ch 07), ridge regression (ch 08 · topic 02), the bootstrap and standard errors (ch 05), variances of sums and products (ch 04), and floating point (ch 00 · topic 03).
**time · scaffolding:** about 80 hours · L3 (independent): every function and method comes with its signature and a one-line docstring; the conventions (shapes, draw order, PyTorch's defaults, the logger's history keys, the summary's rules) are in the lessons. hints have two levels. the capstone gives you data loading and the evaluation cells; the pipeline is yours to repair.
**newly unlocked library components:** from chapter 16 on you may use, in PyTorch: `torch.nn.init.xavier_*`, `kaiming_*` and `calculate_gain` (topic 01); `nn.BatchNorm1d`/`2d`, `nn.LayerNorm`, `nn.RMSNorm` (02); `nn.Dropout`, `F.cross_entropy(..., label_smoothing=...)`, `torch.optim.AdamW` and optimizers' `weight_decay`, `torchvision.transforms.RandomCrop`/`RandomHorizontalFlip` (03); `torch.optim.lr_scheduler` (`StepLR`, `CosineAnnealingLR`, `LambdaLR` for warmup) and `torch.nn.utils.clip_grad_norm_` (04); `register_forward_hook` and logging tools such as TensorBoard (05); hyperparameter-search libraries' random search and hyperband schedulers, e.g. Optuna or Ray Tune (06); and `torch.utils.data.Dataset`/`DataLoader`/`TensorDataset`, the `torch.optim` optimizers you built in chapter 06, devices (`.to(device)`), and `torch.optim.swa_utils.AveragedModel` for weight averaging (07). inside `mlfs`, use `mlfs.torchnn` for training loops; `mlfs.train`'s NumPy layers stay the reference for what the PyTorch ones compute.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [initialization](01-initialization/) | `calculate_fan`, `calculate_gain`, `xavier_uniform_`/`xavier_normal_`, `kaiming_uniform_`/`kaiming_normal_`, `deep_mlp`, `activation_stats` → `mlfs.train` | kaiming keeps a 50-layer ReLU net's scale perfectly, yet every image ends up pointing the same way (cosine 0.993); PyTorch's default init loses the input entirely by layer 10 and can't train at depth 20 |
| 02 | [normalization](02-normalization/) | `standardize` + backward, `BatchNorm1d`, `LayerNorm`, `RMSNorm` → `mlfs.train` | batch norm still trains at 10× the plain network's breaking learning rate, layer norm doesn't; forgetting `model.eval()` with class-sorted test batches turns 86% into 16% |
| 03 | [regularization](03-regularization/) | `l2_penalty`, `Dropout`, `label_smoothing_cross_entropy`, `random_crop`, `random_horizontal_flip`, `EarlyStopping` → `mlfs.train` | Adam + L2 at λ = 1e-4 zeroes the dead-pixel weights within two epochs while barely touching the rest; label smoothing raises accuracy, yet in this small, high-accuracy setting it makes the model *under*confident and its calibration error goes up |
| 04 | [optimization in practice](04-optimization-in-practice/) | `get_lr`, `set_lr`, `LRScheduler`, `StepLR`, `CosineSchedule`, `clip_grad_norm`, `lr_range_test`, `suggest_lr` → `mlfs.train` | 50 warmup steps lift a layer-norm network from 10% to 89% at the same peak learning rate; the linear scaling rule holds to batch 512 and diverges at 2,048 |
| 05 | [debugging training](05-debugging-training/) | `grad_norm`, `named_modules`, `update_ratio`, `dead_fraction`, `DiagnosticsLogger` → `mlfs.train` | of six broken scripts, two (shuffled labels, no `zero_grad`) produce no warning at all, and the missing `zero_grad` run even beats correct SGD |
| 06 | [hyperparameters & experiments](06-hyperparameters-and-experiments/) | `random_search`, `run_bracket`, `successive_halving`, `hyperband`, `ExperimentLog` → `mlfs.train` | single-seed comparisons of two learning rates agree with the 10-seed means 55% of the time; data order causes as much spread as the initialization |
| 07 | [bridging to PyTorch](07-bridging-to-pytorch/) | `to_torch`, `gradient_parity`, `MLP`, `RMSNorm`, `count_parameters`, `get_device`, `make_loader`, `fit`, `evaluate`, `ema_update` → `mlfs.torchnn` | your engine and PyTorch stay within $5 \times 10^{-16}$ for 300 training steps; on small networks your NumPy engine is as fast as PyTorch, and the GPU is the slowest option |

## capstone

*fix the training run*: an MNIST pipeline in PyTorch (split, normalize, a kaiming-initialized MLP with dropout, a training loop with clipping, your `StepLR` and `EarlyStopping` on a torch optimizer, evaluation) that runs without errors and contains **10 planted bugs**. find each one with the checks and your diagnostics, fix it, and measure what it cost. the fixed pipeline reaches **0.981 test accuracy** (95% CI [0.978, 0.984]; three seeds: 0.981, 0.981, 0.983) in about 25 s on a laptop CPU; individual bugs range from invisible (a statistics leak, 0.982) to fatal (chance level, or `nan` everywhere). small CPU config: a few seconds on the offline synthetic stand-in.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (variance through depth for three inits, batch norm forward and backward on four numbers, hyperband's bracket arithmetic), and a 60-minute blind rebuild of normalization.

## notation

| symbol | meaning |
|---|---|
| $n_\text{in}$, $n_\text{out}$ (fan_in, fan_out) | inputs per output and outputs per input of a weight; times the kernel size for convolutions |
| $\operatorname{Var}(W)$, $\sigma$ | variance and standard deviation of the initial weights; gain $\times\,1/\sqrt{\text{fan}}$ for kaiming |
| $z_l$, $h_l$ | pre-activations and activations of layer $l$; $E[h^2]$ their second moment |
| $\mu$, $\sigma^2$, $\epsilon$ | mean, biased variance and stabilizer of a normalization; $s = (\sigma^2 + \epsilon)^{-1/2}$ |
| $\hat x$ | standardized input $(x - \mu)s$ |
| $\gamma$, $\beta$ | a normalization layer's learnable scale (`weight`) and shift (`bias`) |
| $m$ (momentum) | weight of the *new* batch statistic in batch norm's running averages |
| $p$ | dropout probability |
| $s$ (smoothing), $k$ | label-smoothing strength; number of classes |
| $\lambda$ | weight decay / L2 strength |
| $\eta_t$, $\eta_0$, $t_w$, $T$ | learning rate at step $t$, peak (base) learning rate, warmup steps, total steps |
| $c$ | gradient-clipping threshold (max global norm) |
| $\|g\|$ | global 2-norm of all gradients |
| update ratio | $\|W_\text{after} - W_\text{before}\|/\|W_\text{after}\|$ |
| $R$, $\eta$ (hyperband), $s_\text{max}$ | maximum budget, reduction factor, number of brackets − 1 |
| $\delta$ | EMA decay; $1/(1 - \delta)$ is its effective window |
| $B$ | batch size |

## how this connects

- builds on: your tensor engine and `mlfs.nn` (ch 14), stochastic optimizers (ch 06), the ML workflow: validation, overfitting, search (ch 07), ridge regression (ch 08), the bootstrap (ch 05), variances (ch 04), floating point (ch 00).
- used later in: every network from here on trains with `mlfs.torchnn` and this chapter's recipe: CNNs and ResNets with batch norm, augmentation and cosine schedules (ch 16), RNNs with gradient clipping (ch 17), transformers with layer norm, warmup and AdamW (ch 18), signal propagation and the NTK (ch 19), generative models with EMA weights (ch 21–22), DQN/BYOL target networks via `ema_update` (ch 24–25), mixed precision and `RMSNorm` (ch 26–27), calibration (ch 28), and multi-seed reporting (ch 30).

## core references

- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, chapters 7 (regularization) and 8 (optimization for training deep models). [deeplearningbook.org](https://www.deeplearningbook.org)
- Prince (2023), *Understanding Deep Learning*, MIT Press, chapters 7–9 (gradients and initialization, measuring performance, regularization) and 11 (residual networks and normalization). [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)
- He, Zhang, Ren & Sun (2015), "Delving deep into rectifiers", *ICCV*. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- Ioffe & Szegedy (2015), "Batch normalization", *ICML*. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Loshchilov & Hutter (2019), "Decoupled weight decay regularization", *ICLR*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Karpathy (2019), "A recipe for training neural networks". [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
