# debugging training · lab notes

numbers are for real MNIST (standardized pixels), measured with the reference solution: a $784 \to 128 \to 128 \to 10$ ReLU network, 200 steps of batch 64 on 2,000 training images, scored on 1,000 held-out images. offline (the synthetic stand-in that `verify.py` uses) the task is so easy that the healthy run reaches 100% and its gradients shrink towards zero; the dead-init, deep-sigmoid, lr-too-low and shuffled-labels scripts keep their signatures, the lr-too-high script happens to produce no warning, and the missing `zero_grad` diverges late (exploding at step 183) instead of training. that's a useful lesson in itself: signatures depend on the problem as well as the bug.

## experiment 1: six broken training scripts

**what people usually predict:** that each bug produces one clean, specific warning, and that every bug produces at least one.

**what happens:**

| script | bug | loss, first → last | validation accuracy | `summary()` tags |
|---|---|---|---|---|
| healthy | none (Adam, $10^{-3}$) | 2.32 → 0.08 | 0.938 | none |
| A | dead init | 2.31 → 2.29 | 0.107 | update ratio too small (×3), dead relu (100%, 100%), plateau |
| B | deep sigmoid | 2.36 → 2.30 | 0.096 | vanishing (×5), update ratio too small (×5), plateau |
| C | shuffled labels | 2.32 → 2.05 | 0.067 | **none** |
| D | lr too high | 2.32 → 2.30 | 0.096 | exploding (239 at step 3 against a median of 1.2 so far; it later peaks at $2 \times 10^{33}$), update ratio too small (×3), dead relu (91%, 100%) |
| E | no zero_grad | 2.32 → 0.12 | 0.888 | **none** |
| F | lr too low | 2.32 → 2.32 | 0.053 | update ratio too small (×3, ~$10^{-6}$), plateau |

four scripts are diagnosed by their warnings, though rarely by just one: a single bug sets off a chain of symptoms. two scripts look perfectly healthy to the logger.

**why:** **A**: biases of $-3$ against standardized inputs make every pre-activation negative, so every ReLU outputs 0 for every image; nothing flows backward to the hidden weights (update ratio exactly 0), and the output layer can only learn the class frequencies (a plateau at $\log 10$). **B**: each sigmoid layer shrinks the backward signal by a factor of about 0.14 with the default init (on paper 3), so the first layer's gradient is $10^{-6}$ of the last's; the first layers can't learn, and the network plateaus. **D**: the first few SGD steps at learning rate 2 produce a gradient norm of $10^{33}$ (exploding), which flings the pre-activations far negative and kills 91–100% of the ReLUs; after that nothing moves (update ratio 0) and the loss sits at $\log 10$. the explosion came first; the dead units are its aftermath. **F**: Adam's steps are about $\eta = 10^{-7}$ per weight, an update ratio of about $10^{-6}$, so nothing changes in 200 steps. **C** and **E** are silent because nothing is wrong with the optimization. with shuffled labels the network is successfully memorizing random labels (the training loss does fall, 2.32 → 2.05), with ordinary gradients and updates; only the validation accuracy (6.7%, below chance) gives it away. without `zero_grad`, the accumulated gradient acts like a momentum term with no decay; at learning rate 0.01 that's still stable over 200 steps, and the run actually reaches 88.8%, better than the same SGD *with* `zero_grad` (0.815, experiment 3). its signature, a gradient norm that grows steadily, isn't one of the six rules (on paper 4 suggests one).

## experiment 2: is 1e-3 really the right update ratio?

**what people usually predict:** exactly $10^{-3}$, or no idea.

**what happens:**

| Adam lr | median update ratio (2.weight) | validation accuracy | warnings |
|---|---|---|---|
| 1e-5 | 1.05e-04 | 0.403 | plateau |
| 1e-4 | 1.15e-03 | 0.904 | none |
| 1e-3 | 3.39e-03 | 0.938 | none |
| 1e-2 | 1.45e-02 | 0.904 | none |
| 1e-1 | 1.09e-03 | 0.103 | exploding, vanishing ×2, dead relu ×2 |

for the four stable runs the update ratio is almost exactly proportional to the learning rate (about $1.1$–$3.4 \times \eta$, with the larger ratios early in training), and accuracy after 200 steps peaks at a ratio of $3.4 \times 10^{-3}$, with ratios of $10^{-3}$ and $10^{-2}$ both somewhat worse and equally so. the run at lr 0.1 is the trap: it destroyed itself in the first steps, after which its weights stopped changing, so its *median* ratio is a harmless-looking $10^{-3}$. (offline, the easy synthetic task is learned at every learning rate from $10^{-4}$ to $10^{-1}$, and the median ratios *fall* at larger learning rates, $6 \times 10^{-5}$ at $10^{-3}$ and $4 \times 10^{-6}$ at $10^{-2}$, because the network solves the task within a few dozen steps and its gradients, and so its updates, shrink to almost nothing afterwards: at lr $10^{-2}$ the logger even reports "update ratio too small". a converged network and a stuck one look alike in this statistic; the loss tells them apart.)

**why:** an Adam step moves each weight by about $\eta$ (less when the gradient's sign is noisy), so $\|\Delta W\| \approx \eta\sqrt{\text{number of weights}}$ times a factor below 1, and the ratio is proportional to $\eta$ whatever the gradient's scale (on paper 2). so the rule of thumb is really a statement about learning rates in units of the weights' size, and the range from about $10^{-3}$ to $10^{-2}$ is fine for this model and this budget; it's a range to stay inside, not a target. for SGD the step is $\eta g$, so the ratio also depends on the gradient's scale, which changes during training and from layer to layer: that's where per-layer ratios are most informative. and the lr 0.1 run is the reminder that a summary statistic over a whole run can hide a disaster in the first ten steps: always look at the dashboard (and the other warnings) too.

## experiment 3: catching the silent bugs

**what people usually predict:** with shuffled labels the validation loss stays flat at $\log 10$; without `zero_grad` the gradient norm "grows" vaguely, or explodes.

**what happens:** with shuffled labels, the training loss falls from 2.32 to 2.05 while the validation loss *rises* from 2.31 to 2.52: worse than guessing uniformly. the healthy run's training and validation losses fall together (0.08 and 0.28). without `zero_grad`, the global gradient norm grows from 0.72 at step 1 to 29.2 at step 100 and 37.4 at step 200, roughly linearly at first; the healthy SGD run's stays between 0.7 and 1.6.

**why:** with labels shuffled independently of the images, there's nothing to learn that generalizes; the only way to lower the training loss is to memorize which random label goes with which training image. that memorization makes confident predictions on validation images too, and since those predictions are unrelated to the true labels, confident means wrong: the validation loss goes above $\log 10$. "does the validation loss track the training loss?" is the one check that catches data-pipeline bugs, which is why every serious training loop evaluates on held-out data from the first epoch. without `zero_grad`, `.grad` holds the sum of all gradients so far (on paper 4): the consistent part grows linearly with the step count, the noise only like $\sqrt t$, so the norm rises steadily until the steps it produces become large enough to change the gradients themselves (the growth slows down after about 50 steps). at a small learning rate this behaves like heavy momentum, which is why it trains; at a larger one, or for longer, it diverges. a rule like "the median gradient norm of the last quarter is 5× that of the first" would catch it.
