# regularization

## why this matters

train chapter 14's MNIST network on 1,000 images instead of 55,000 and it reaches 100% training accuracy within a few epochs. the training loss keeps falling, towards $10^{-3}$. meanwhile the validation loss bottoms out around epoch 9 and then *rises*, while validation accuracy stalls at 92%. the network has more than enough capacity to memorize every training image, and that's what it does: past a point, every step makes it more confident about the training set and no better (in loss, worse) at anything else.

**regularization** is the toolkit for that situation: everything that makes a model generalize better without changing the data it's trained on. classical machine learning had one main tool, the norm penalty (ridge regression, chapter 08). deep learning kept it, found that it interacts in a surprising way with adaptive optimizers, and added others that only make sense for networks: dropping units at random, softening the targets, inventing new training examples by transforming old ones, and simply stopping at the right time. this topic builds all five. along the way you'll see that "weight decay" and "an L2 penalty" are the same thing for SGD and quite different things for Adam, which is why every modern training recipe says `AdamW`.

## warm-up: predict first

1. SGD with an L2 penalty $\frac\lambda2\|w\|^2$ on the loss: what does one step do to a weight whose loss gradient is zero?
2. Adam with the same penalty: a weight whose only gradient is the penalty, $\lambda w$. how big is Adam's step for it, roughly, whether $\lambda$ is $10^{-4}$ or $10^{-1}$?
3. dropout with $p = 0.5$ zeroes half the units during training. at test time all units are on. what must you do so the next layer sees inputs of the same scale?
4. with label smoothing, the target for the true class is 0.91 instead of 1 (10 classes, $s = 0.1$). what's the loss of a model that predicts probability 1 for the true class?
5. you watch the validation loss for early stopping and it starts rising at epoch 10, but the validation accuracy is still improving at epoch 15. which epoch's weights do you keep?

## prerequisites check

- [ ] you know ridge regression and why an L2 penalty shrinks weights (→ ch 08 · topic 02)
- [ ] you know how Adam normalizes each coordinate by $\sqrt{\hat v}$, and what `AdamW` changes (→ ch 06 · topic 04)
- [ ] you can explain overfitting with training and validation curves (→ ch 07 · topics 02 and 04)
- [ ] your `Module` has the `train()`/`eval()` switch and `fit` takes a `callback` (→ ch 14 · topic 04)
- [ ] you can compute cross-entropy from logits with `logsumexp` (→ ch 14 · topic 04)

## core ideas

### idea 1: weight decay and the L2 penalty

the oldest regularizer adds $\frac\lambda2\|w\|^2$ to the loss: of all the weight settings that fit the data about equally well, prefer the small ones. its gradient is $\lambda w$, so a plain SGD step becomes

$$w \leftarrow w - \eta(\nabla L + \lambda w) = (1 - \eta\lambda)\,w - \eta\nabla L.$$

the penalty *decays* every weight by the factor $1 - \eta\lambda$ at every step, independently of the data's gradient: that's why it's also called **weight decay**, and why `SGD(weight_decay=λ)` and "add $\frac\lambda2\|w\|^2$ to the loss" are the same algorithm. your `l2_penalty(params, weight_decay)` builds the penalty as a `Tensor`, so it can be added to any loss and differentiated with everything else.

for Adam they are *not* the same. Adam divides each coordinate's gradient by $\sqrt{\hat v}$, its running root-mean-square. with an L2 penalty, the penalty's gradient $\lambda w$ is divided too. look at the two extremes. for a weight whose loss gradient is large and noisy, $\sqrt{\hat v}$ is large, and $\lambda w/\sqrt{\hat v}$ is negligible: the weight is barely regularized. for a weight with no loss gradient at all (say, connected to an input pixel that's always zero), the only gradient is $\lambda w$, so $\hat m/\sqrt{\hat v} \approx \operatorname{sign}(w)$ and Adam takes a full step of size $\eta$ towards zero, whatever $\lambda$ is. so "L2 under Adam" regularizes weights roughly in inverse proportion to how much the data cares about them, which isn't what anyone intends. Loshchilov and Hutter (2019) proposed the fix that you implemented in chapter 06: **decouple** the decay from the gradient, $w \leftarrow w - \eta\lambda w$, then the Adam step on the loss alone. that's `AdamW`, and its $\lambda$ means what it says: every weight shrinks by the same fraction per step. (in practice, biases and normalization parameters are usually excluded from weight decay; they're few, and shrinking them doesn't reduce complexity.)

> **▶ build: stage 1.** implement `l2_penalty(params, weight_decay)`: $(\lambda/2)\sum_p\|p\|^2$ as a one-element `Tensor`.

### idea 2: dropout

**dropout** (Srivastava et al., 2014) zeroes each activation independently with probability $p$ on every training step. a different random sub-network is trained at every step, all sharing weights, so no unit can rely on any particular other unit being present; the network is pushed towards redundant, distributed features. at test time every unit is on, and the prediction behaves like an average over the exponentially many sub-networks.

the scale has to match between training and test. a unit kept with probability $1 - p$ contributes, on average, $(1 - p)$ times its value. **inverted dropout** fixes this during training rather than at test time: keep each entry with probability $1 - p$ and divide the kept ones by $1 - p$,

$$\operatorname{dropout}(x) = x \odot \frac{m}{1 - p},\qquad m_i \sim \operatorname{Bernoulli}(1 - p),$$

so $E[\operatorname{dropout}(x)] = x$, and evaluation is simply the identity. the backward pass multiplies by the same mask (`x * (mask / (1 - p))` with your `Tensor` gives it for free). conventions for your `Dropout(p=0.5, *, rng=None)`: draw the keep-mask as `self.rng.random(x.shape) >= p` from a generator stored once at construction, so a seeded model is reproducible and every call gets a fresh mask; return the input unchanged in eval mode or when $p = 0$; return zeros when $p = 1$ (without dividing by zero); refuse $p$ outside $[0, 1]$.

dropout is a regularizer with a cost: it adds noise to every gradient, so training is slower, and the training loss stays higher. it's used less in convolutional networks with batch norm (the two interact badly: the variance batch norm sees in training includes dropout's noise, which disappears at test time) and is common in the fully connected and attention layers of transformers.

> **🔮 predict:** with $p = 0.5$, what's the variance of a dropped-out unit whose input is the constant 1? does inverted dropout preserve the variance as well as the mean?

> **▶ build: stage 2.** implement `Dropout(p=0.5, *, rng=None)` as a `Module`.

### idea 3: label smoothing

cross-entropy against a one-hot target is minimized only when the model puts probability 1 on the true class, which needs the true logit to be infinitely larger than the others. on a training set the model can memorize, it keeps pushing the logits apart forever, and becomes extremely confident, including on the examples it will get wrong at test time. **label smoothing** (Szegedy et al., 2016) replaces the one-hot target with a softened distribution over $k$ classes:

$$q = (1 - s)\,e_y + \frac sk\mathbf 1,\qquad L = -\sum_j q_j\log p_j = (1 - s)\,\underbrace{(-\log p_y)}_{\text{ordinary CE}} + s\,\underbrace{\Big(-\frac1k\sum_j\log p_j\Big)}_{\text{CE against uniform}}.$$

the true class gets $1 - s + s/k$, every other class $s/k$. in logits, with $\operatorname{lse} = \operatorname{logsumexp}(z)$: $-\log p_y = \operatorname{lse} - z_y$ and $-\frac1k\sum_j\log p_j = \operatorname{lse} - \bar z$, so the whole loss is a combination of operations your `Tensor` already has and stays stable for any logits. its gradient with respect to the logits is $p - q$, the usual "prediction minus target", and it vanishes at $p = q$: when the true logit exceeds each other logit by exactly $\log\frac{1 - s + s/k}{s/k}$ (on paper 3). a finite target confidence, $0.91$ for $s = 0.1, k = 10$. the convention (PyTorch's `F.cross_entropy(..., label_smoothing=s)`) spreads $s$ over **all** $k$ classes, the true one included.

> **▶ build: stage 3.** implement `label_smoothing_cross_entropy(logits, targets, smoothing=0.1)`.

### idea 4: data augmentation

the most effective regularizer is more data. when you can't get more, you can often manufacture it: if a small shift or a mirror image doesn't change what an image shows, then every training image stands for many, and the network can be trained on random transformations of it, a new one every epoch. this bakes **invariances** into the training data that the architecture doesn't have. two classics for images in NCHW layout:

- **`random_crop(images, size, padding=...)`**: zero-pad every image by `padding` pixels on each side, then cut a random `size × size` window out of each one. with `size` equal to the original size it's a random translation by up to `padding` pixels, the standard CIFAR-10 augmentation (chapter 16). draw the offsets for the whole batch at once: `tops = rng.integers(0, H + 2·padding − size + 1, size=n)`, then `lefts` the same way, so seeded runs are reproducible.
- **`random_horizontal_flip(images, p=0.5)`**: mirror each image left to right (the last axis) with probability $p$, drawing `rng.random(n) < p`. return a new array, never modify the caller's.

augmentations encode assumptions about the task, and the wrong ones hurt. flipping is harmless for cats and trucks and wrong for digits (a mirrored 2 isn't a 2; a mirrored 7 isn't anything). augmentation is applied **only to training data**, freshly each epoch; applying it at evaluation time makes test scores noisy and lower (one of the capstone's planted bugs is a cousin of this).

> **▶ build: stage 4.** implement `random_crop(images, size, *, padding=0, rng=None)` and `random_horizontal_flip(images, *, p=0.5, rng=None)`.

### idea 5: early stopping

training longer always lowers the training loss; past some epoch, it raises the validation loss. **early stopping** watches a validation metric and stops when it hasn't improved for `patience` evaluations in a row, then keeps the weights from the best evaluation. it's cheap, it needs no tuning beyond the patience, and for linear models trained by gradient descent it's provably similar to an L2 penalty: the number of steps plays the role of $1/\lambda$ (Goodfellow et al., 2016, §7.8).

your `EarlyStopping(patience=5, *, min_delta=0.0, mode='min')` is a small state machine that knows nothing about models, so it works with any training loop, NumPy or PyTorch:

- `step(value)` takes the latest metric (anything `float()` accepts, including a torch scalar) and returns `True` once `patience` consecutive steps brought no improvement; it stays `True` after that. `patience` must be at least 1 (a `ValueError` otherwise: with 0 it would stop on the very first step).
- an improvement means beating the best value so far by more than `min_delta`: below `best - min_delta` for `mode='min'` (losses), above `best + min_delta` for `'max'` (accuracies). the first value always counts.
- it exposes `best` (`None` before the first step), `best_step` (0-based index of the best value, $-1$ before any), `should_stop`, and a counter of bad steps.

restoring the best weights is the caller's job (save a `state_dict` whenever `best_step` changes), because only the caller knows what the model is. plugged into chapter 14's `fit` it's one line: `callback=lambda epoch, h: es.step(h["val_loss"][-1])`.

> **🔮 predict:** the validation loss and the validation accuracy are both reasonable things to watch. will they tell an early stopper to stop at the same epoch? (experiment 3.)

> **▶ build: stage 5.** implement `EarlyStopping`.

## worked example

**label smoothing by hand.** three classes, logits $z = (2, 0, 0)$, true class 0, smoothing $s = 0.3$. the softmax is $p = (e^2, 1, 1)/(e^2 + 2) = (0.78699, 0.10651, 0.10651)$, so $\log p = (-0.23954, -2.23954, -2.23954)$. the smoothed target is $q = 0.7\,e_0 + 0.1 = (0.8, 0.1, 0.1)$, and

$$L = -(0.8 \cdot (-0.23954) + 0.1 \cdot (-2.23954) + 0.1 \cdot (-2.23954)) = 0.19164 + 0.44791 = 0.63954.$$

ordinary cross-entropy would be $0.23954$. the gradient $p - q = (-0.01301, 0.00651, 0.00651)$ still pushes the true logit up, but only gently: the optimal gap is $\log(0.8/0.1) = \log 8 = 2.079$, and this model's gap of 2 is almost there. with plain cross-entropy the gradient would be $(-0.21301, 0.10651, 0.10651)$, sixteen times larger, and it would never reach zero.

**early stopping by hand.** `EarlyStopping(patience=2)` fed the validation losses $1.0, 0.8, 0.85, 0.79, 0.9, 0.95$:

| step | value | best so far | improved? | bad steps | `step()` returns |
|---|---|---|---|---|---|
| 0 | 1.00 | 1.00 | yes (first) | 0 | False |
| 1 | 0.80 | 0.80 | yes | 0 | False |
| 2 | 0.85 | 0.80 | no | 1 | False |
| 3 | 0.79 | 0.79 | yes | 0 | False |
| 4 | 0.90 | 0.79 | no | 1 | False |
| 5 | 0.95 | 0.79 | no | 2 | **True** |

`best = 0.79`, `best_step = 3`: restore the weights saved after step 3. the tests check both examples.

## on paper

1. **L2 under Adam, at the two extremes.** ignore momentum and bias correction: Adam's step is $\eta g/\sqrt{v}$ with $v \approx E[g^2]$. (a) a weight whose only gradient is $\lambda w$: what's the step, and how many steps does it take to reach zero from $w_0$? (b) a weight whose loss gradient has RMS $\sigma \gg \lambda|w|$: what's the effective decay factor per step contributed by the penalty?

<details><summary>solution</summary>

(a) $g = \lambda w$, $v = \lambda^2w^2$, so the step is $\eta\lambda w/(\lambda|w|) = \eta\operatorname{sign}(w)$: a fixed step of $\eta$ towards zero, independent of $\lambda$. it takes about $|w_0|/\eta$ steps (20 for $w_0 = 0.02$, $\eta = 10^{-3}$: the lab's dead-pixel weights are gone within the first two epochs, for every $\lambda > 0$). (b) the penalty adds $\eta\lambda w/\sigma$ to the step, a decay factor $1 - \eta\lambda/\sigma$: weaker the larger the loss gradient's RMS. with AdamW the decay is $1 - \eta\lambda$ for every weight.

</details>

2. **inverted dropout's variance.** for a constant input $x$, compute the mean and variance of $x\,m/(1 - p)$ with $m \sim \operatorname{Bernoulli}(1 - p)$.

<details><summary>solution</summary>

$E = x(1 - p)/(1 - p) = x$. $E[(xm/(1 - p))^2] = x^2(1 - p)/(1 - p)^2 = x^2/(1 - p)$, so $\operatorname{Var} = x^2\frac{1}{1 - p} - x^2 = x^2\frac{p}{1 - p}$: equal to $x^2$ for $p = 0.5$. the mean is preserved, the variance is not; that's the train/test shift that makes dropout before batch norm a bad idea.

</details>

3. **the optimal logits under label smoothing.** show that $L(z) = -\sum_j q_j\log\operatorname{softmax}(z)_j$ is minimized when $\operatorname{softmax}(z) = q$, and that for $q = (1 - s + s/k, s/k, \dots, s/k)$ this means $z_y - z_j = \log\frac{1 - s + s/k}{s/k}$ for every $j \ne y$.

<details><summary>solution</summary>

$\partial L/\partial z_j = p_j - q_j$ (the softmax cross-entropy gradient with a general target, since $\sum q_j = 1$), which is zero iff $p = q$; $L$ is convex in $z$, so that's the minimum. $p_y/p_j = e^{z_y - z_j}$, so $z_y - z_j = \log(q_y/q_j) = \log\frac{1 - s + s/k}{s/k}$: $\log 91 = 4.51$ for $s = 0.1$, $k = 10$.

</details>

4. **early stopping as a penalty.** for least squares $L(w) = \frac12\|Xw - y\|^2$, gradient descent from $w_0 = 0$ with step $\eta$ gives, in the eigenbasis of $X^\top X$ (eigenvalues $\lambda_i$), $w_i^{(t)} = (1 - (1 - \eta\lambda_i)^t)\,w_i^*$. ridge regression with penalty $\alpha$ gives $w_i = \frac{\lambda_i}{\lambda_i + \alpha}w_i^*$. compare the two filters for $\lambda_i \gg \alpha$ and $\lambda_i \ll \alpha$ with $\alpha = 1/(\eta t)$.

<details><summary>solution</summary>

for $\lambda_i \gg 1/(\eta t)$: $(1 - \eta\lambda_i)^t \approx e^{-\eta\lambda_i t} \approx 0$, so the GD filter is $\approx 1$, and ridge's is $\lambda_i/(\lambda_i + \alpha) \approx 1$. for $\lambda_i \ll 1/(\eta t)$: $1 - (1 - \eta\lambda_i)^t \approx \eta\lambda_i t = \lambda_i/\alpha$, and ridge's is $\approx\lambda_i/\alpha$ too. both keep the well-determined directions and shrink the poorly determined ones; stopping after $t$ steps acts like a ridge penalty of $1/(\eta t)$.

</details>

## experiment

`lab.py` trains a 270,000-parameter network on 1,000 MNIST digits, to make overfitting easy, and uses your tools:

1. Adam with an L2 penalty versus AdamW, across weight-decay strengths, looking at the weights attached to pixels that never light up.
2. dropout with $p = 0, 0.2, 0.5$: training and validation curves.
3. label smoothing's effect on accuracy, confidence and calibration, and two `EarlyStopping` monitors that disagree.

## check your understanding

1. why is `SGD(weight_decay=λ)` identical to adding `l2_penalty(params, λ)` to the loss, while `AdamW(weight_decay=λ)` is not identical to `Adam` plus the penalty?

<details><summary>answer</summary>

SGD applies the same linear step to every gradient component, so adding $\lambda w$ to the gradient is the same as decaying $w$ by $\eta\lambda w$. Adam rescales each coordinate by $1/\sqrt{\hat v}$, which also rescales the penalty's gradient (and feeds it into the moments), so the decay a weight receives depends on its gradient history. AdamW applies the decay outside the rescaling.

</details>

2. dropout makes the training loss *higher*. why is that not a problem?

<details><summary>answer</summary>

the training loss is measured on a noisy sub-network, not on the model you'll deploy. what matters is the eval-mode loss on held-out data. a regularizer that made the training loss lower would be odd: its job is to stop the network from fitting the training set as well as it could.

</details>

3. why does a model trained with label smoothing never predict probability 1?

<details><summary>answer</summary>

the loss is minimized at $p = q$, which puts $1 - s + s/k < 1$ on the true class; pushing past it *raises* the loss (the gradient $p - q$ changes sign). so its confidence on training data saturates around $1 - s + s/k$, and on everything else is lower.

</details>

4. you augment MNIST with random horizontal flips and accuracy drops. why?

<details><summary>answer</summary>

the augmentation tells the network that a digit and its mirror image have the same label, which is false (2, 3, 4, 5, 7 and 9 look different or like nothing when mirrored). augmentations must be label-preserving for the task; for digits, small shifts and rotations are, flips aren't.

</details>

5. why does `EarlyStopping` not save the model itself?

<details><summary>answer</summary>

to stay framework-agnostic. it only sees numbers, so the same class works for `mlfs.nn`, PyTorch or anything else; the caller knows how to save and restore its own model (`state_dict` / `load_state_dict`) and does it when `best_step` changes.

</details>

6. what does `min_delta` protect you against?

<details><summary>answer</summary>

noise. a validation loss that wobbles by $10^{-4}$ would otherwise keep resetting the patience counter with "improvements" that mean nothing, and training would never stop (or stop late). with `min_delta`, only improvements larger than the noise count.

</details>

7. in the lab, Adam + L2 sends the dead-pixel weights to zero within two epochs. does that matter for the test set?

<details><summary>answer</summary>

on MNIST's test set, slightly: 202 of the 10,000 test images (2%) have ink in at least one of the 175 pixels that were blank in all 1,000 training images, and with zeroed weights the network ignores that ink. the bigger point is what it reveals: the strength of an L2 penalty under Adam varies wildly from weight to weight, unrelated to the $\lambda$ you chose.

</details>

## common bugs

- **dropout without the $1/(1 - p)$.** symptom: everything is fine in training, and test-time activations are $1/(1 - p)$ times larger than anything the network saw; accuracy drops in eval mode.
- **dropout left on at test time.** symptom: predictions change from one call to the next, and accuracy is lower than it should be. call `model.eval()` (your `predict` and `evaluate` do).
- **a new generator per call.** `Dropout` that creates `default_rng(seed)` inside `forward` draws the *same* mask every step. create it once in `__init__`.
- **label smoothing over the other classes only.** spreading $s$ over $k - 1$ classes instead of $k$ is a different (also reasonable) loss, but not PyTorch's; symptom: a mismatch of about $s/k$ in the target.
- **augmenting the evaluation data.** symptom: test accuracy fluctuates between runs and is lower than training curves suggest.
- **`EarlyStopping(mode='min')` on accuracy.** symptom: training stops after exactly `patience + 1` epochs, because accuracy going *up* never counts as an improvement. (the capstone plants it.)

## going deeper

- Loshchilov & Hutter (2019), "Decoupled weight decay regularization", *ICLR*. AdamW. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (2014), "Dropout: a simple way to prevent neural networks from overfitting", *JMLR* 15: 1929–1958. [jmlr.org/papers/v15/srivastava14a](https://jmlr.org/papers/v15/srivastava14a.html)
- Szegedy, Vanhoucke, Ioffe, Shlens & Wojna (2016), "Rethinking the inception architecture for computer vision", *CVPR*. introduces label smoothing (§7). [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- Müller, Kornblith & Hinton (2019), "When does label smoothing help?", *NeurIPS*. its effect on representations and calibration. [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)
- Shorten & Khoshgoftaar (2019), "A survey on image data augmentation for deep learning", *Journal of Big Data* 6: 60. [doi:10.1186/s40537-019-0197-0](https://doi.org/10.1186/s40537-019-0197-0)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, chapter 7 (regularization; §7.8 early stopping as L2). [deeplearningbook.org](https://www.deeplearningbook.org)

## connections

- ← builds on: ridge regression (ch 08), Adam and AdamW (ch 06 · topic 04), overfitting and validation (ch 07), `Module`, the train/eval switch and `fit`'s callback (ch 14 · topic 04), batch norm's train/test behaviour (topic 02).
- → used in: the learning-rate schedules that interact with weight decay (topic 04), the hyperparameter searches that tune these knobs (topic 06), `torchnn.MLP(dropout=...)` (topic 07), the capstone's early-stopping and augmentation bugs, CIFAR-10 augmentation (ch 16 · capstone), dropout in transformers (ch 18), calibration (ch 28).
