# 14 · neural networks from scratch

> build a working deep learning framework yourself. this is the centerpiece of the course.

PyTorch, JAX and TensorFlow are huge, but the idea at their core is small enough to build in a week: numbers (or arrays) that remember how they were computed, operations that know their own derivatives, and one sweep backwards through the record that gives the gradient of a loss with respect to every parameter at once. this chapter builds that idea from the ground up, entirely in NumPy. you start with neurons and XOR, deriving a network's gradients by hand until you never want to again; build a scalar reverse-mode engine in a hundred lines; scale it to a `Tensor` engine with broadcasting, reductions, matrix products and indexing, checked against finite differences and PyTorch; wrap it in modules, losses and a training loop with the same names and conventions as `torch.nn`; probe what one-hidden-layer networks can and can't do efficiently; and finally look at backpropagation from the outside, as vector-Jacobian products that cost little time and a lot of memory, and trade the memory back with gradient checkpointing. the capstone trains an MNIST classifier to over 97% test accuracy on your own framework, with no libraries beyond NumPy. on the way you'll meet a wrong gradient that still trains, a shape bug that teaches a regression to predict its mean and passes every gradient check, a learning rate so large it quietly kills every neuron, a 2-unit network that fails at XOR half the time, a deep network that represents a sawtooth exponentially better than a shallow one but can't learn it, and a checkpointed network that uses *more* memory than an unchecked one.

**you will build:** linear layers, ReLU and sigmoid, MLPs, a hand-wired XOR network and hand-derived backpropagation; `Value`, a scalar reverse-mode autodiff engine with an iterative topological sort; `Tensor`, a reverse-mode engine over NumPy arrays (about 25 operations, broadcasting-aware gradients, `no_grad`, `gradcheck`); `Module`, `Parameter`, `Linear`, activations, `Sequential`, MSE and cross-entropy losses, and the `fit`/`evaluate`/`predict` training loop; a constructive universal approximator, the deep sawtooth network and a linear-region counter; VJPs, reverse-mode Jacobians, graph memory accounting, `checkpoint` and $\sqrt L$ `checkpoint_sequential`. modules: **`mlfs.tensor`** (topics 03 and 06) and **`mlfs.nn`** (topic 04), on which chapter 15 builds its training utilities and its PyTorch port.
**prerequisites:** matrix calculus and gradient checks with `mlfs.numgrad` (ch 02 · topic 07), forward-mode autodiff with `Dual` (ch 03 · topic 04), the bootstrap (ch 05 · topic 03), stochastic optimization and the `Optimizer` family (ch 06 · topic 04), train/validation/test splits (ch 07), and NumPy broadcasting and stable numerics (ch 00). chapter 08 (linear models) helps but isn't required: logistic and softmax regression reappear here as one-layer networks.
**time · scaffolding:** about 85 hours · L3 (independent): every function and method comes with its signature and a one-line docstring; the exact conventions (shapes, the order of random draws, accumulation rules, PyTorch's naming) are in the lessons. hints have two levels. the capstone gives you data loading and the experiment cells; you write the pipeline.
**newly unlocked library components:** from chapter 15 on (PyTorch itself arrives in ch 15 · topic 07), you may use `torch.Tensor` and `torch.autograd` (`backward`, `torch.no_grad`, `torch.autograd.gradcheck`: you built them in topic 03), `torch.nn.Module`, `nn.Parameter`, `nn.Linear`, `nn.ReLU`, `nn.Tanh`, `nn.Sigmoid`, `nn.Sequential`, `nn.MSELoss`, `nn.CrossEntropyLoss`, `F.cross_entropy` and `F.binary_cross_entropy_with_logits` (topic 04), and `torch.utils.checkpoint`, `torch.autograd.functional.vjp`/`jacobian` and `torch.func.jvp` (topic 06). inside `mlfs`, chapter 15 builds initializers, normalization layers, dropout, schedulers and diagnostics on *your* `mlfs.nn` and `mlfs.tensor`, through the contracts in [`CONTRACTS.md`](../CONTRACTS.md), and checks your engine's gradients against PyTorch's.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [from linear models to neurons](01-from-linear-models-to-neurons/) | `linear_layer`, `relu`, `sigmoid`, `mlp_forward`, `collapse_linear`, a hand-wired XOR network, hand-derived gradients and `train_two_layer` | a 10-layer linear network equals one layer to $10^{-15}$; with 2 hidden units (enough to represent XOR) gradient descent solves it only 27 times in 50 |
| 02 | [scalar reverse-mode autodiff](02-scalar-reverse-mode-autodiff/) | `Value` (micrograd-style), `topological_order`, `value_and_grad`, `graph_size` | reverse mode is $O(n)$ and forward mode $O(n^2)$ for a gradient; a scalar engine is over 10,000× slower than NumPy on one layer |
| 03 | [a tensor autograd engine](03-a-tensor-autograd-engine/) | `unbroadcast`, `Tensor` (~25 ops), `concatenate`, `stack`, `where`, `no_grad`, `gradcheck` → `mlfs.tensor` | a bias gradient $n$ times too small still lowers the loss; an $(n, 1) - (n,)$ shape bug makes a regression predict its mean, and gradcheck passes |
| 04 | [modules, losses & training loops](04-modules-losses-and-training-loops/) | `Module`, `Parameter`, `Linear`, `ReLU`, `Tanh`, `Sigmoid`, `Sequential`, `mse_loss`, `cross_entropy`, `binary_cross_entropy_with_logits`, `fit`, `evaluate`, `predict` → `mlfs.nn` | 100,000 parameters memorize 32 random labels, and so does a network missing its activation; too large a learning rate doesn't explode, it kills 96% of the ReLUs and sits at $\log 10$ |
| 05 | [universal approximation](05-universal-approximation/) | `relu_interpolant`, `sawtooth_net`, `count_linear_regions`, `fit_function`, `max_error` | 10 units of depth match 33 units of width on a 32-piece sawtooth; random deep networks have *fewer* pieces than shallow ones; the deep network never learns the sawtooth |
| 06 | [backprop in depth](06-backprop-in-depth/) | `vjp`, `jacobian_reverse`, `graph_nbytes`, `checkpoint` → `mlfs.tensor`, `checkpoint_sequential` | forward mode wins only for very tall Jacobians; checkpointing as one segment uses *more* peak memory than none, 8 segments 2.6× less |

## capstone

*MNIST with your own framework*: a $784 \to 256 \to 128 \to 10$ MLP trained with your `Tensor`, `mlfs.nn` and Adam to **≥ 97% test accuracy** (the reference reaches 0.977–0.980), reported with bootstrap confidence intervals against softmax regression and the majority class. small CPU config: about a second on the offline synthetic stand-in; full config: about 40 seconds for 12 epochs on a laptop CPU.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (backprop through a scalar graph, a linear layer with softmax cross-entropy, unbroadcasting and a ReLU interpolant), and a 60-minute blind rebuild of the tensor engine's core.

## notation

| symbol | meaning |
|---|---|
| $x$, $X$ | one input (a vector), a batch of inputs (rows are examples, shape $(n, d)$) |
| $W$, $b$ | a layer's weights in torch layout, shape (out, in), and its bias, shape (out,) |
| $Z = XW^\top + b$ | pre-activations of a layer; $H = \phi(Z)$ its activations |
| $\phi$, $\sigma$ | an activation function; the logistic sigmoid |
| $\operatorname{relu}(z)$ | $\max(z, 0)$ |
| $z$ (in a classifier) | logits: unnormalized class scores |
| $\operatorname{softmax}(z)_j$ | $e^{z_j}/\sum_k e^{z_k}$ |
| $\operatorname{logsumexp}(z)$, $\operatorname{softplus}(z)$ | $\log\sum_k e^{z_k}$; $\log(1 + e^z)$ |
| $L$ | the scalar loss being differentiated (also the number of layers, where it says so) |
| $\bar v$ | the adjoint of $v$: $\partial L/\partial v$, with $v$'s shape |
| $J$, $J_f$ | a Jacobian, shape (outputs, inputs) |
| JVP, VJP | Jacobian-vector product $Ju$ (forward mode); vector-Jacobian product $J^\top v$ (reverse mode) |
| $\operatorname{unbroadcast}(G, s)$ | $G$ summed back to shape $s$: the transpose of broadcasting |
| $\eta$ | learning rate |
| $T$, $T^{\circ d}$ | the tent map $2\min(x, 1 - x)$ on $[0, 1]$; its $d$-fold composition (a sawtooth with $2^d$ pieces) |
| $s$ | the number of checkpointed segments |

## how this connects

- builds on: broadcasting and stable numerics (ch 00), matrix calculus and gradient checks (ch 02 · topic 07), forward-mode autodiff (ch 03 · topic 04), the bootstrap (ch 05), SGD, Adam and minibatches (ch 06 · topic 04), splits and honest evaluation (ch 07).
- used later in: chapter 15 builds initializers, normalization layers, regularization, schedulers, diagnostics and hyperparameter search on `mlfs.nn`, and ports it to PyTorch with a gradient-parity test; convolutions (ch 16), recurrent networks and backpropagation through time (ch 17), and transformers (ch 18) are new operations and modules on the same idea; deep learning theory (ch 19) returns to expressivity and trainability; ML systems (ch 26) to activation memory and recomputation.

## core references

- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, chapter 6 (deep feedforward networks, back-propagation). [deeplearningbook.org](https://www.deeplearningbook.org)
- Prince (2023), *Understanding Deep Learning*, MIT Press, chapters 3–7 (shallow and deep networks, losses, fitting, gradients). [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)
- Baydin, Pearlmutter, Radul & Siskind (2018), "Automatic differentiation in machine learning: a survey", *JMLR* 18(153): 1–43. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
- Griewank & Walther (2008), *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed., SIAM. [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
- Karpathy (2020), *micrograd*. [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Paszke et al. (2019), "PyTorch: an imperative style, high-performance deep learning library", *NeurIPS*. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
