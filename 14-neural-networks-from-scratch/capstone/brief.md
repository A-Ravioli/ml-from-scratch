# 14 · capstone: MNIST with your own framework

## goal

train a neural network to read handwritten digits, using nothing but NumPy and the framework you built in this chapter. every piece of the pipeline is yours: the `Tensor` that records the computation and differentiates it (topic 03), the `Linear` and `ReLU` layers and the `cross_entropy` loss (topic 04), the `fit` loop, and the Adam optimizer from chapter 06. the target is a multilayer perceptron with **at least 97% accuracy on the MNIST test set**, reported honestly: with a confidence interval, against baselines, with the test set used once.

this is the chapter's proof of concept. when it works, you'll have trained a network of about 235,000 parameters through 5 layers of your own code, and you'll know exactly what every line does. chapter 15 then teaches you to make deep networks train *well*, and checks your framework against PyTorch.

## data

MNIST (LeCun, Cortes & Burges): 60,000 training and 10,000 test images of handwritten digits, $28 \times 28$ grayscale, labels 0–9, loaded with `mlfs.datasets.mnist()` (downloaded once, about 11 MB, and cached under `~/.cache/mlfs/`). hold out 5,000 training images as a validation set for choosing the epoch; the test set is for the final number only. offline (and in the tests), a small synthetic stand-in with the same shapes and dtypes is used, which is much easier than MNIST: your write-up must use the real data.

## configurations

| config | where | model and training | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | $784 \to 64 \to 10$, 5 epochs, batch 64, on 1,800 synthetic images | about a second |
| **full** (your write-up) | default | $784 \to 256 \to 128 \to 10$, Adam ($10^{-3}$), batch 128, 12 epochs, keeping the epoch with the best validation accuracy | about 40 s with the reference framework (about 3 s per epoch on the laptop it was timed on), plus about 5 s for the softmax-regression baseline |

the notebook picks the config from `datasets.is_offline()`. your framework may be slower than the reference (a copy here, an extra pass there): a few minutes is fine; if an epoch takes more than about 30 s, profile it (prompt 5).

## deliverables

1. `capstone.py` with the four stages implemented and their checks green:
   - `preprocess(images)` → float64 $(n, 784)$ in $[0, 1]$; `split_validation(X, y, n_val, *, rng=None)` → `(X_train, y_train, X_val, y_val)`, validation = the first `n_val` indices of `rng.permutation(n)`.
   - `build_mlp(sizes, *, rng=None)` → a `Sequential` of `Linear(sizes[i], sizes[i + 1])` layers with `ReLU` between them (not after the last), all drawing their initial weights from one generator.
   - `accuracy(model, X, y)` and `accuracy_ci(model, X, y, *, n_boot=2000, alpha=0.05, rng=None)` → `(accuracy, lo, hi)`, a percentile bootstrap interval over per-example correctness, using your `bootstrap_ci` from chapter 05.
   - `train_classifier(model, X_train, y_train, X_val, y_val, *, epochs=10, batch_size=128, lr=1e-3, rng=None)`: Adam and cross-entropy through your `fit`, recording `history["val_acc"]` after every epoch, and restoring the weights of the best epoch (by validation accuracy) at the end.
2. the experiments in the notebook (given, running on your functions): the full-config MLP, a softmax-regression baseline (`build_mlp((784, 10))`) trained the same way, the majority-class baseline, the test accuracies with 95% intervals, a paired interval for the MLP's advantage, learning curves, and the misclassified digits.
3. the write-up at the end of the notebook, answering its five prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `train_classifier` on synthetic digits | $\ge$ 90% validation accuracy for a $784 \to 32 \to 10$ MLP in 4 epochs (median of 3 seeds) |
| MLP test accuracy, full config, real MNIST | **$\ge$ 0.97**, reported with a 95% bootstrap interval |
| baselines | softmax regression and majority class on the same test set, with the paired difference |

for reference, the reference solution's full config reaches a test accuracy of **0.977** (95% CI [0.974, 0.980]); softmax regression trained the same way reaches 0.927, the majority class 0.114, and the paired difference MLP − softmax regression is +0.051 [+0.046, +0.056]. with seeds 1–3 used for everything (the validation split, the initialization and the minibatch order) the test accuracy is 0.980, 0.980 and 0.979; keeping split seed 0 and varying only the initialization and minibatch seed gives 0.980, 0.979 and 0.978. every interval's lower end is above 0.975.

## stretch goals

- **beat 98%.** wider layers, a third hidden layer, a learning-rate decay (multiply `optimizer.lr` by 0.5 every few epochs, from a callback), or simple data augmentation (shift each training image by up to 2 pixels). report every attempt, including the ones that didn't help, and choose by validation accuracy only.
- **gradient check your network.** run your `gradcheck` on a $784 \to 16 \to 10$ network with the cross-entropy on 5 real images (after reducing the inputs to, say, 20 pixels so it's fast). which of your operations does it exercise?
- **checkpoint it.** train the full MLP with each layer wrapped in your `checkpoint` (topic 06) and measure the peak memory per step with `tracemalloc` at batch size 1,024, with and without.
- **profile your framework.** time the forward pass, backward pass and optimizer step for one batch, and find the slowest operation. try avoiding the copy in `Tensor.__init__` for results of operations, or caching `W.T`, and measure the speedup.
- **seed-to-seed variance.** train the full config with 5 different seeds and report the spread of test accuracies next to your bootstrap interval. which source of uncertainty is larger? (chapter 15 · topic 06 returns to this.)

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- LeCun, Bottou, Bengio & Haffner (1998), "Gradient-based learning applied to document recognition", *Proceedings of the IEEE* 86(11): 2278–2324. MNIST, and a table of MLP results on it (a 300-unit MLP: about 95.3–96.4%; a 500-150 MLP: about 97.1%). [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
- Kingma & Ba (2015), "Adam: a method for stochastic optimization", *ICLR*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall. percentile intervals and paired comparisons.
- Karpathy (2019), "A recipe for training neural networks". [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
