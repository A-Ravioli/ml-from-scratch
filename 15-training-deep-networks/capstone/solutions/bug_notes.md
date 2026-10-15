# 15 · capstone · answer key: the 10 bugs

read this after your own write-up. every measurement is a full-config run on real MNIST (55,000 training + 5,000 validation images, 10,000 test images; MLP (256, 128); seed 0) with **only that bug** re-introduced into the reference solution, measured on a laptop CPU (about 25 s per full run). the fixed pipeline scores **0.9810** (95% bootstrap CI [0.9783, 0.9837]); seeds 1 and 2 give 0.9808 and 0.9828, so seed-to-seed differences of about ±0.002 are noise. the pipeline as shipped (all 10 bugs) scores **0.098**, with a `nan` training loss from the first step.

| # | where | the bug | symptom | measured effect (test accuracy, seed 0) |
|---|---|---|---|---|
| 1 | `split_data` | `X = X[perm]` without `y = y[perm]`: the images are shuffled, the labels aren't | training loss stuck at 2.30 ($\log 10$), validation accuracy ~0.11; early stopping ends the run after 4 epochs | **0.114** [0.108, 0.120]: chance |
| 2 | `normalize` | per-pixel statistics, `std(axis=0)`: MNIST's border pixels are 0 in every image, so $0/0 = $ `nan` (and $x/0 = \infty$) | `RuntimeWarning: invalid value encountered in divide`; training loss `nan` from the first step | **0.098**: every prediction is class 0 (argmax of `nan`s) |
| 3 | `normalize` | the statistics are computed on the training **and** evaluation sets concatenated: leakage | none visible: the checks catch it by changing the evaluation set and watching the training data's normalization change | **0.982** [0.979, 0.985]: no measurable effect here (the mean and std of 55,000 versus 60,000 MNIST images are almost identical); it matters whenever the evaluation set differs from the training set |
| 4 | `build_model` | a `Softmax` after the last layer, while `F.cross_entropy` applies its own log-softmax | the training loss can't go below ~1.46 (it reaches 1.488 after 10 epochs, while accuracy is 97%): the "logits" are probabilities in $[0, 1]$, so the loss can never become confident | **0.9756** [0.9726, 0.9786]: −0.5 points, and much slower early training (validation 0.924 after epoch 1 versus 0.954) |
| 5 | `build_model` | `nn.init.normal_(std=1.0)` instead of kaiming: weights about 20× too large in the first layer ($1/\sqrt{2/784}$) | first-epoch training loss **57**; on standardized inputs the first layer's pre-activations have a std of about $\sqrt{784} = 28$ instead of $\sqrt 2$ | **0.226** [0.218, 0.234]: the network barely trains in 10 epochs |
| 6 | `train_one_epoch` | no `optimizer.zero_grad()`: `.grad` accumulates every step's gradient | training loss *rises* within the first epoch (7.9 → 12.7 → 13.5 by epoch 4); the gradient norm grows step after step | **0.101**: chance (at this learning rate and momentum, the accumulated "momentum" diverges) |
| 7 | `train_one_epoch` | `clip_grad_norm_` called before `loss.backward()`: it clips the gradients of the previous step (already set to `None` by `zero_grad`), so clipping never happens | none in this config: the clipping threshold (5) is never reached by the fixed pipeline's gradients, so the run is bit-for-bit identical | **0.9810**: identical to the fixed pipeline. but with bug 5 also present (huge initial gradients), working clipping gives 0.226 and this bug drops it to **0.114**: chance |
| 8 | `fit` | the scheduler, defined in **epochs** (`StepLR(step_size=4)`), is passed to `train_one_epoch`, which steps it after every **batch** | `history["lr"]` reads 0.05, $3 \times 10^{-34}$, $9 \times 10^{-67}$, …: the learning rate is halved every 4 batches; the loss freezes at ~0.98 after the first epoch | **0.769** [0.761, 0.778]: training effectively stops after about 40 steps |
| 9 | `evaluate` | no `model.eval()` and no `torch.no_grad()`: dropout stays on during evaluation | validation accuracy is about 1 point lower than it should be and changes from one call to the next; evaluation uses more memory (it builds graphs) | **0.9804** on the test set (scored correctly in eval mode by the given cell), but every *validation* number is wrong (best 0.970 versus 0.979), and early stopping, fed noisy numbers, stopped after 9 epochs |
| 10 | `fit` | `EarlyStopping(patience)` in its default `mode='min'` on an **accuracy** | the run always stops after `patience + 1 = 4` epochs, and the restored weights are epoch 1's (accuracy increasing never counts as an improvement) | **0.9582** [0.954, 0.962]: −2.3 points (the best validation accuracy seen was 0.973, but epoch 1's weights, 0.954, were restored) |

## ranked by cost and by visibility

by cost (fixed pipeline 0.981): 2 (`nan`, 0.098) ≈ 6 (0.101) ≈ 1 (0.114) > 5 (0.226) > 8 (0.769) > 10 (0.958) > 4 (0.976) > 9 (0.980 test, but wrong validation numbers) ≈ 3 (0.982) ≈ 7 (0.981).

by visibility, the order is almost reversed. bugs 2 and 6 announce themselves (a warning and a `nan`; a loss that rises). bug 1 looks exactly like a broken model or a learning rate that's too small (a plateau at $\log 10$), and the logger of topic 05 would report a healthy optimization. bug 8 is visible only if you plot or log the learning rate. bug 10 shows up as "training always stops at epoch 4". bugs 3, 7 and 9 change nothing you'd notice in this configuration and would ship: 3 becomes a real error the day the evaluation data is distributed differently from the training data, 7 the day a gradient spikes (it's the difference between 0.226 and chance in the presence of bug 5), and 9 silently corrupts every validation-based decision (early stopping, model selection, hyperparameter search). that's the lesson of the capstone: the bugs that cost the most are rarely the hardest to find, and the ones you'd never find without tests are exactly the ones that make your next experiment wrong.

## what makes each one loud

| # | a check that would have caught it at once |
|---|---|
| 1 | assert that the training loss falls below $\log 10$ within the first few hundred steps; overfit one batch (ch 14 · topic 04); look at a few images with their labels after any reshuffle |
| 2 | `assert np.isfinite(X).all()` after preprocessing; stop on the first non-finite loss |
| 3 | compute statistics inside a fitted transformer that only ever sees training data (ch 07 · topic 05's `StandardScaler`); a test like stage 1's |
| 4 | check the initial loss is about $\log 10$ *and* that the model's outputs aren't normalized; never put a softmax in a model trained with a logits-based loss |
| 5 | `activation_stats` at initialization (topic 01): the first layer's std should be about 1, not 20; the initial loss should be about $\log 10$, not 57 |
| 6 | log the gradient norm (topic 05): steady growth over steps is the signature |
| 7 | log `clip_grad_norm_`'s return value (topic 04): it's the pre-clip norm, and a value of 0 every step means it's looking at empty gradients |
| 8 | log the learning rate every epoch; assert the schedule's units (steps or epochs) where it's built |
| 9 | evaluate twice and assert identical results; `evaluate` should always set and restore the mode |
| 10 | assert the early stopper's mode matches the metric (`'max'` for accuracies); log `best_step` |
