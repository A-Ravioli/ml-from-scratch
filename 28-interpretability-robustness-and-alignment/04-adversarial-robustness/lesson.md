# adversarial robustness

## why this matters

Szegedy et al. (2014) found something that still unsettles people: take an image a network classifies correctly, add a perturbation too small for a human to notice, and the network confidently says something else. the perturbation isn't random noise (random noise of the same size does nothing); it's chosen, using the model's own gradients, to push it across a decision boundary. these **adversarial examples** transfer between models, survive printing and photographing (Kurakin et al., 2017; stop signs with a few stickers read as speed limits, Eykholt et al., 2018), and exist for every kind of model people have looked at, from speech recognisers to the language models that topic 07 will try to keep aligned (where adversarial suffixes "jailbreak" safety training).

two lessons come out of this, and the topic is built around both. first, a model can be accurate on its test distribution and still be wrong almost everywhere *near* it: test accuracy measures the average case, an attacker picks the worst case. second, robustness is always robustness *to something*. a model trained to resist small pixel changes isn't thereby robust to fog, rotation or a new camera, and can even get worse at them. you'll build the two standard attacks (FGSM and PGD), adversarial training, and a small suite of natural corruptions, and measure what each defence buys and costs.

## warm-up: predict first

1. a linear model $w^\top x$ on 784 pixels. you may change every pixel by at most 0.1. by how much can you change its output, in terms of $w$?
2. one big step in the gradient's sign direction, or ten small ones projected back into the allowed region: which finds a worse input for the model, and why?
3. training on adversarial examples makes a model robust to them. does it cost anything on clean data?
4. a model robust to every perturbation with $\max_i |\delta_i| \le 0.2$. is it robust to lowering the image's contrast by half?

## prerequisites check

- [ ] you can compute the gradient of a loss with respect to the input, in eval mode, without touching parameter gradients (→ topic 01)
- [ ] you know projected gradient descent and projections onto simple sets (→ ch 06 · 05, `mlfs.optim`)
- [ ] you know $\ell_p$ norms and their dual norms ($\ell_\infty$ and $\ell_1$) (→ ch 02 · 01)
- [ ] you know what batch norm does differently in train and eval mode (→ ch 15 · 03)
- [ ] you know LeNet (→ ch 16 · 03)

## core ideas

### idea 1: threat models, and the fast gradient sign method

an attack needs a **threat model**: what the attacker may change. the classic one for images is an $\ell_p$ ball: the adversarial input $x'$ must satisfy $\|x' - x\|_p \le \epsilon$ (and stay a valid image, pixels in $[0, 1]$). the attacker wants the input in that ball with the highest loss:

$$x^\star = \arg\max_{\|x' - x\|_p \le \epsilon} \; \ell(f(x'), y).$$

Goodfellow, Shlens & Szegedy (2015) argued that adversarial examples come from models being *too linear*, not too nonlinear. replace the loss by its first-order approximation, $\ell(x + \delta) \approx \ell(x) + \nabla_x \ell^\top \delta$, and maximise over $\|\delta\|_\infty \le \epsilon$: the answer is $\delta = \epsilon\, \mathrm{sign}(\nabla_x \ell)$, and the loss rises by $\epsilon \|\nabla_x \ell\|_1$. that's the **fast gradient sign method** (FGSM). the rise scales with the $\ell_1$ norm of the gradient, which grows with the number of input dimensions: each pixel moves imperceptibly, but 784 tiny moves, all in the harmful direction, add up (warm-up 1: $\epsilon\|w\|_1$).

as with saliency (topic 01), compute the gradient in **eval mode** and with `torch.autograd.grad`, so the attack neither uses dropout nor leaves gradients on the parameters. and make it work under `torch.no_grad()`, because evaluation loops call attacks from inside one: wrap the gradient part in `torch.enable_grad()`.

> **spec.** `fgsm(model, x, y, epsilon, *, loss_fn=None, clamp=(0.0, 1.0))`: `loss_fn(model(x), y)` is the loss (default `F.cross_entropy` on logits); returns `clamp(x + epsilon * sign(grad_x loss))` (no clamping if `clamp` is None), detached. the model runs in eval mode and its mode is restored; parameter `.grad`s are untouched; `x` isn't modified; works inside `torch.no_grad()`.

> **▶ build: stage 1.** implement `fgsm`.

### idea 2: projected gradient descent

FGSM takes one step and trusts the linear approximation all the way to the edge of the ball. a network isn't linear at that scale, so **projected gradient descent** (PGD; Madry et al., 2018) takes many smaller steps and, after each one, projects back into the allowed set:

$$x^{t+1} = \Pi_{\mathcal{B}(x, \epsilon) \cap [0,1]^d}\big(x^t + \alpha\, s(\nabla_x \ell(x^t))\big).$$

the step direction $s$ is the steepest-ascent direction *for the norm*: $\mathrm{sign}(g)$ for $\ell_\infty$ and $g / \|g\|_2$ for $\ell_2$. the projections are simple: for $\ell_\infty$, clip $\delta = x' - x$ to $[-\epsilon, \epsilon]$ coordinate-wise; for $\ell_2$, rescale $\delta$ to norm $\epsilon$ if it's longer. then clamp to the valid pixel range (which keeps you inside the ball, because $x$ itself is a valid image). Madry et al. start from a **random point** in the ball (uniform per coordinate for $\ell_\infty$), which avoids getting stuck where the gradient at $x$ happens to be uninformative, and take steps small enough to settle but large enough to cross the ball several times (for MNIST, $\epsilon = 0.3$ with 40 steps of 0.01). this topic's default, $2.5\epsilon / n_\text{steps}$, is the rule of thumb used by the Madry lab's `robustness` library, not a setting from the paper.

PGD is the standard "strong first-order attack": empirically, restarts and more steps find only slightly higher losses. but every attack gives only an *upper bound* on robust accuracy. a defence that breaks gradients (non-differentiable preprocessing, randomness, saturated outputs) can make PGD fail without making the model robust; Athalye et al. (2018) found that 7 of the 9 defences published at ICLR 2018 relied on such obfuscated gradients, and circumvented 6 of them completely and 1 partially. when you evaluate a defence, attack it as hard as you'd attack your worst enemy's.

> **spec.** `pgd_attack(model, x, y, epsilon, *, alpha=None, n_steps=10, norm='linf', random_start=True, loss_fn=None, clamp=(0.0, 1.0), generator=None)`: `norm` in `{'linf', 'l2'}` (`ValueError` otherwise); `alpha` defaults to $2.5\epsilon / n_\text{steps}$. with `random_start`, start at $x$ plus uniform noise in $[-\epsilon, \epsilon]$ per coordinate (`linf`) or a uniformly random point of the $\ell_2$ ball (`l2`: random direction, radius $\epsilon u^{1/d}$), drawn from `generator`, projected; otherwise start at $x$. then `n_steps` times: step by `alpha` along $\mathrm{sign}(g)$ (`linf`) or $g / \|g\|_2$ per example (`l2`), project onto the $\epsilon$-ball around $x$, clamp. returns the detached result; eval mode (restored); works inside `torch.no_grad()`.

> **🔮 predict:** a 95%-accurate MNIST LeNet, $\epsilon = 0.2$ in $\ell_\infty$. what accuracy will it have under 10-step PGD?

> **▶ build: stage 2.** implement `pgd_attack`.

### idea 3: adversarial training

if the model's weakness is the worst case in the ball, train on the worst case. Madry et al. framed robust training as a min-max problem:

$$\min_\theta \; \mathbb{E}_{(x, y)}\Big[\max_{\|\delta\| \le \epsilon} \ell\big(f_\theta(x + \delta), y\big)\Big].$$

the inner max is approximated by PGD against the *current* model; the outer min by a gradient step on the loss at the PGD point (by Danskin's theorem, the gradient of a max is the gradient at the maximiser, so this is a valid descent direction for the robust loss). every training step costs $n_\text{steps} + 1$ forward-backward passes, which is why adversarial training is expensive.

two engineering details. craft the attack in **eval mode** (otherwise every PGD step updates batch-norm statistics with adversarial batches, and dropout makes the attack noisy) and take the update in **train mode**. and evaluate with an attack at least as strong as the one used in training: a model trained against 5-step PGD and evaluated with FGSM looks better than it is.

robustness isn't free. Tsipras et al. (2019) showed that on some distributions the most accurate classifier and the most robust one are necessarily different: accuracy can be gained from many weakly predictive features that a small perturbation can flip, and a robust model has to give those up. in practice robust models lose some clean accuracy, and their robustness doesn't extend beyond the training $\epsilon$. a surprising bonus: their input gradients look like the object (Tsipras et al. call them *perceptually aligned*), because the only features left are ones a human would also use.

> **spec.** `adversarial_training_step(model, optimizer, x, y, *, epsilon, alpha=None, n_steps=7, loss_fn=None, generator=None)`: makes `linf` examples with `pgd_attack(model, x, y, epsilon, alpha=alpha, n_steps=n_steps, loss_fn=loss_fn, generator=generator)` (eval mode), then in train mode computes `loss_fn(model(x_adv), y)` (default cross-entropy), `optimizer.zero_grad()`, backward, `optimizer.step()`; returns the loss as a float and leaves the model in train mode. `robust_accuracy(model, x, y, attack=None, *, batch_size=256)`: the fraction of rows the model classifies correctly on `attack(model, xb, yb)` (or on the clean rows if `attack` is None), batch by batch, predictions in eval mode; the model's mode is restored.

> **▶ build: stage 3.** implement `adversarial_training_step` and `robust_accuracy`.

### idea 4: natural distribution shift

an $\ell_\infty$ ball is a convenient threat model, not a description of the world. real inputs shift in other ways: sensor noise, lighting, blur, compression, viewpoint, a different hospital's scanner. Hendrycks & Dietterich (2019) turned 15 such **corruptions** at 5 severities into a benchmark (ImageNet-C) and found that accuracy under corruption tracks clean accuracy across architectures but falls far below it. this is **covariate shift**: $p(x)$ changes while $p(y \mid x)$ doesn't, and nothing in the training objective promises anything about the new $p(x)$.

how does adversarial training interact with it? Yin et al. (2019) found it helps against high-frequency corruptions like noise (an $\ell_\infty$ ball contains a lot of small high-frequency changes) and *hurts* against low-frequency ones like fog and contrast changes: the robust model relies more on low-frequency structure, so changing that structure hurts it more. the lab tests four simple corruptions on MNIST. the general lesson: name the threat model, and test outside it.

> **spec.** `corrupt(x, kind, severity, *, generator=None)`: images `(B, C, H, W)` in $[0, 1]$, `severity` $s \in \{0, \dots, 5\}$ (`ValueError` otherwise); `"gaussian_noise"`: $x + 0.08\,s\,z$ with $z \sim \mathcal{N}(0, 1)$ drawn from `generator`; `"contrast"`: $m + (x - m)(1 - 0.15 s)$ with $m$ each image's mean; `"brightness"`: $x + 0.1 s$; `"shift"`: move the image $s$ pixels right along the last axis, filling with zeros; unknown kinds raise `ValueError`; the result is clamped to $[0, 1]$, and `x` isn't modified. `corruption_accuracy(model, x, y, kinds, severities, *, generator=None)`: `{kind: ndarray of accuracies, one per severity}`, in eval mode (restored).

> **🔮 predict:** after adversarial training, will the model be more or less accurate on low-contrast digits?

> **▶ build: stage 4.** implement `corrupt` and `corruption_accuracy`.

## worked example

a two-class linear model with logits $(0,\; w^\top x)$, $w = (2, -1, 0.5)$, at $x = (0.5, 0.5, 0.5)$ with label 1. the logit of class 1 is $w^\top x = 0.75$, so $p_1 = \sigma(0.75) = 0.679$.

**FGSM.** the cross-entropy is $-\log p_1$, whose gradient with respect to $x$ is $-(1 - p_1)\, w = -0.321\, w$. its sign is $-\mathrm{sign}(w) = (-1, +1, -1)$. with $\epsilon = 0.1$: $x' = (0.4, 0.6, 0.4)$. the logit drops by exactly $\epsilon\|w\|_1 = 0.35$, to 0.40, and $p_1$ falls to 0.599. in 784 dimensions with weights of similar size, the same $\epsilon$ would move the logit about 260 times as much ($784/3$).

**an $\ell_2$ projection.** a PGD iterate with $\delta = (0.3, 0.4)$ and $\epsilon = 0.25$: $\|\delta\|_2 = 0.5 > 0.25$, so scale by $0.25 / 0.5$: $\delta = (0.15, 0.2)$. the $\ell_\infty$ projection of the same $\delta$ with $\epsilon = 0.25$ clips coordinate-wise: $(0.25, 0.25)$.

**a PGD trace.** maximise $-(x - 0.33)^2$ from $x = 0$ with $\epsilon = 1$, no random start, 4 steps. with $\alpha = 0.1$ the iterates are $0.1, 0.2, 0.3, 0.4$ (each step follows the sign of the gradient, which points towards 0.33). with the default $\alpha = 2.5 / 4 = 0.625$ they are $0.625, 0, 0.625, 0$: too large a step oscillates around the maximiser.

your `fgsm` must reproduce $(0.4, 0.6, 0.4)$, and your `pgd_attack` the final $0.4$ and $0$.

## on paper

1. **the dual norm.** show that $\max_{\|\delta\|_\infty \le \epsilon} g^\top \delta = \epsilon \|g\|_1$, attained at $\delta = \epsilon\,\mathrm{sign}(g)$, and that $\max_{\|\delta\|_2 \le \epsilon} g^\top \delta = \epsilon \|g\|_2$.

<details><summary>solution</summary>

$g^\top \delta = \sum_i g_i \delta_i \le \sum_i |g_i| |\delta_i| \le \epsilon \sum_i |g_i|$, with equality when every $\delta_i = \epsilon\,\mathrm{sign}(g_i)$. for $\ell_2$, Cauchy–Schwarz gives $g^\top \delta \le \|g\|_2 \|\delta\|_2 \le \epsilon\|g\|_2$, with equality at $\delta = \epsilon g / \|g\|_2$. this is why the $\ell_\infty$ step uses the sign and the $\ell_2$ step the normalised gradient; and since $\|g\|_1$ can be up to $\sqrt{d}$ times $\|g\|_2$, $\ell_\infty$ attacks get stronger with dimension.

</details>

2. **projections.** show that clipping each coordinate of $\delta$ to $[-\epsilon, \epsilon]$ is the Euclidean projection onto the $\ell_\infty$ ball, and that clamping $x + \delta$ to $[0, 1]^d$ afterwards can't leave the ball if $x \in [0, 1]^d$.

<details><summary>solution</summary>

the projection minimises $\|\delta' - \delta\|_2^2 = \sum_i (\delta'_i - \delta_i)^2$ subject to $|\delta'_i| \le \epsilon$; the problem separates over coordinates and each is solved by clipping. clamping moves each coordinate of $x + \delta$ towards the interval $[0, 1]$, which contains $x_i$, so $|x'_i - x_i|$ can only shrink. the same argument works coordinate-wise for $\ell_2$: every $|\delta_i|$ shrinks, so $\|\delta\|_2$ does too.

</details>

3. **Danskin, informally.** let $L(\theta) = \max_\delta \ell(\theta, \delta)$ and $\delta^\star(\theta)$ the maximiser. argue that $\nabla L(\theta) = \nabla_\theta \ell(\theta, \delta^\star)$, so adversarial training's update is a gradient step on the robust loss.

<details><summary>solution</summary>

$L(\theta + h) \ge \ell(\theta + h, \delta^\star) = L(\theta) + \nabla_\theta \ell(\theta, \delta^\star)^\top h + o(h)$, so $\nabla_\theta \ell(\theta, \delta^\star)$ is a subgradient; when the maximiser is unique and things are smooth, $L$ is differentiable with exactly that gradient (the change of $\delta^\star$ contributes nothing to first order, because $\delta^\star$ is a maximum). PGD only approximates $\delta^\star$, so in practice the update is an approximate descent direction, better the stronger the attack.

</details>

4. **why robust models lose accuracy, in a toy.** a label $y = \pm 1$; one feature $x_1 = y$ with probability 0.9 (otherwise $-y$), and 100 features $x_j \sim \mathcal{N}(\eta y, 1)$ with $\eta = 0.3$. what does a standard linear classifier rely on, and what happens to it under an $\ell_\infty$ attack with $\epsilon = 2\eta$?

<details><summary>solution</summary>

the average of the 100 weak features is $\mathcal{N}(\eta y, 1/100)$: mean 0.3, standard deviation 0.1, so its sign predicts $y$ with 99.9% accuracy, far better than $x_1$'s 90%. but an attacker moving each weak feature by $2\eta = 0.6$ against $y$ shifts their mean to $-0.3$, and the classifier becomes almost always wrong. a classifier using only $x_1$ (which an $\epsilon = 0.6$ attack can't flip, since it's $\pm 1$) is 90% accurate and stays 90% under attack. that's Tsipras et al.'s construction: accuracy and robustness pull towards different features.

</details>

## experiment

the lab attacks a LeNet trained on MNIST with FGSM and PGD across budgets and shows what the attacked digits look like; trains a second LeNet adversarially with your training step and compares the two clean and under attack, along with their input gradients; and tests both models on your four corruptions to see whether robustness to the $\ell_\infty$ ball carries over.

## check your understanding

1. why is random noise of the same $\ell_\infty$ size almost harmless while FGSM is devastating?

<details><summary>answer</summary>

random signs make $g^\top \delta$ a sum of terms with random signs, of size about $\epsilon \|g\|_2$; aligned signs give $\epsilon\|g\|_1$, which in $d$ dimensions is up to $\sqrt{d}$ times larger. the attack spends its budget only in directions that hurt.

</details>

2. a paper reports 90% accuracy under FGSM for its defence. why is that not enough?

<details><summary>answer</summary>

FGSM is a weak, single-step attack; a defence can resist it by making the loss surface locally flat or noisy (gradient masking) while remaining vulnerable to multi-step attacks, attacks without gradients, or attacks through a differentiable approximation. robustness claims need PGD with random restarts and, ideally, adaptive attacks designed against the defence.

</details>

3. why does PGD use a random start?

<details><summary>answer</summary>

at the clean point the loss is often small and locally flat or misleading (especially for adversarially trained models, where the gradient at $x$ is deliberately uninformative); starting elsewhere in the ball explores more of it, and repeated random restarts give a stronger attack.

</details>

4. why craft the adversarial examples in eval mode during adversarial training?

<details><summary>answer</summary>

in train mode every PGD forward pass would update batch-norm running statistics with adversarial batches ($n_\text{steps}$ extra updates per step) and dropout would randomise the attack's gradient. the attack should target the model as it will be used.

</details>

5. a model trained against $\epsilon = 0.2$ is 58% accurate at $\epsilon = 0.2$ and 12% at $\epsilon = 0.3$. what does that say about what was learned?

<details><summary>answer</summary>

robustness is specific to the training budget: the model has learned to be insensitive to changes of size 0.2 in the directions PGD finds, not to be correct in some absolute sense. beyond the budget it is again cheap to flip, though still much harder than the standard model.

</details>

6. why might adversarial training *hurt* accuracy under a contrast change?

<details><summary>answer</summary>

the robust model gives up small, high-frequency, easily-flipped features and leans more heavily on large-scale intensity structure; a contrast change scales exactly that structure. robustness to one family of perturbations shifts the model's reliance to features that another family attacks.

</details>

## common bugs

- **descending instead of ascending.** `x - eps * sign(g)` makes the input *easier*. the attack maximises the loss.
- **no projection after a step**, or projecting onto the ball around the *current* iterate instead of the original $x$.
- **the $\ell_2$ step with `sign`.** for $\ell_2$ the steepest direction is $g / \|g\|_2$, per example (normalise each row, not the whole batch).
- **attacking in train mode.** batch norm and dropout corrupt both the attack and the model's statistics.
- **forgetting `zero_grad`** in the training step: gradients from previous steps accumulate.
- **evaluating with a weaker attack than you trained with.** robust accuracy is only as good as the attack that measured it.

## going deeper

- Szegedy et al. (2014), "Intriguing properties of neural networks". [arXiv:1312.6199](https://arxiv.org/abs/1312.6199)
- Goodfellow, Shlens & Szegedy (2015), "Explaining and harnessing adversarial examples", *ICLR* (FGSM). [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
- Madry et al. (2018), "Towards deep learning models resistant to adversarial attacks", *ICLR* (PGD, adversarial training). [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
- Athalye, Carlini & Wagner (2018), "Obfuscated gradients give a false sense of security", *ICML*. [arXiv:1802.00420](https://arxiv.org/abs/1802.00420)
- Tsipras et al. (2019), "Robustness may be at odds with accuracy", *ICLR*. [arXiv:1805.12152](https://arxiv.org/abs/1805.12152)
- Hendrycks & Dietterich (2019), "Benchmarking neural network robustness to common corruptions and perturbations", *ICLR* (ImageNet-C). [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)
- Yin et al. (2019), "A Fourier perspective on model robustness in computer vision", *NeurIPS*. [arXiv:1906.08988](https://arxiv.org/abs/1906.08988)
- Eykholt et al. (2018), "Robust physical-world attacks on deep learning visual classification", *CVPR*. [arXiv:1707.08945](https://arxiv.org/abs/1707.08945)

## connections

- ← builds on: input gradients and eval-mode hygiene (topic 01), projected gradient methods (ch 06), LeNet (ch 16), the training loop and batch norm (ch 15).
- → used in: topic 05 asks whether models at least *know* when inputs are unusual (calibration, out-of-distribution detection); topic 07 meets adversarial optimisation again as reward hacking: an optimiser finding the inputs where a learned reward model is wrong.
