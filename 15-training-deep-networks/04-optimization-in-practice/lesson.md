# optimization in practice

## why this matters

chapter 06 treated the learning rate as a number you pick once. every serious training run you'll read about treats it as a *curve*: a few hundred steps ramping up from zero, a long plateau or a slow decline, and a final descent to almost nothing. the reasons are practical. a learning rate large enough to make fast progress early is too large to settle into a good minimum late, because SGD's noise keeps it bouncing around at a distance proportional to the step size. and at the very start, before the optimizer's statistics and the network's scale have settled, even a learning rate that will be fine later can wreck the weights in a few steps.

the other practical question is *which* learning rate. trying ten values with full training runs costs ten training runs. the **learning-rate range test** (Smith, 2017) gets a good answer from a single sweep of a hundred minibatches. and when one batch occasionally produces an enormous gradient (recurrent networks, chapter 17, do this routinely), **gradient clipping** keeps that one step from undoing everything. this topic builds all of these on top of chapter 06's optimizers. one design choice runs through it: the schedulers talk to the optimizer only through two small functions, `get_lr` and `set_lr`, so the same scheduler drives your `mlfs.optim.Adam` today and `torch.optim.AdamW` from topic 07 on.

## warm-up: predict first

1. SGD on a noisy quadratic with a constant learning rate converges to... the minimum? or to a cloud around it? how does the cloud's size depend on the learning rate?
2. Adam's very first update, with bias correction: how large is it for each weight, in terms of the learning rate and the gradient?
3. you clip gradients to a global norm of 1. a batch produces a gradient of norm 1,000. how large is the resulting SGD step, with learning rate 0.1?
4. you double the batch size and keep the number of epochs fixed. how many optimizer steps do you take now? what should happen to the learning rate?
5. a learning-rate sweep shows the loss falling fastest around $10^{-2}$ and exploding at $10^{-1}$. which learning rate would you train with?

## prerequisites check

- [ ] you know SGD, momentum and Adam, and that `mlfs.optim` optimizers expose a mutable `.lr` (→ ch 06 · topic 04)
- [ ] you know why SGD's minibatch gradient is noisy and what its variance depends on (→ ch 06 · topic 04)
- [ ] your `mlfs.nn` has `state_dict`/`load_state_dict` and the five-line training loop (→ ch 14 · topic 04)
- [ ] you know how layer norm rescales activations (→ topic 02)

## core ideas

### idea 1: the learning rate is a dial outside the optimizer

a schedule changes the learning rate between steps, so it needs to read and write the optimizer's learning rate. the two frameworks in this course store it differently. an `mlfs.optim` optimizer (chapter 06) has an attribute: `opt.lr`. a PyTorch optimizer has no such attribute; it has a list of **parameter groups**, dictionaries that each hold some parameters and their own hyperparameters, `opt.param_groups[0]["lr"]`. groups exist so that different parts of a model can train at different rates (a pretrained backbone slower than a new head, chapter 16).

two functions hide the difference:

- **`get_lr(optimizer)`**: `param_groups[0]["lr"]` if the optimizer has `param_groups`, else `optimizer.lr`.
- **`set_lr(optimizer, lr)`**: sets every group's `"lr"`, or `.lr`.

it's a small thing, but it's what makes everything else in this topic (and `EarlyStopping` from topic 03, and topic 06's searches) framework-agnostic: code that only ever calls these two functions works on both. it also means you should never create a `.lr` attribute on a torch optimizer: nothing reads it, and the bug is silent. one limitation to know: `set_lr` gives *every* parameter group the same learning rate, so these schedulers can't keep discriminative per-group rates (a pretrained backbone at a tenth of the head's rate, chapter 16). for that, use `torch.optim.lr_scheduler`, whose schedulers scale each group's own rate.

> **▶ build: stage 1.** implement `get_lr(optimizer)` and `set_lr(optimizer, lr)`.

### idea 2: schedules and warmup

**why decay.** with a constant learning rate, SGD on a noisy problem doesn't converge to the minimum; it converges to a *distribution* around it. on a one-dimensional quadratic with curvature $h$ and gradient noise of variance $\sigma^2$, the iterates settle with variance $\approx\eta\sigma^2/(2h)$ (on paper 1): halve the learning rate, halve the cloud. so you want a large $\eta$ early (fast progress while you're far away, where noise doesn't matter) and a small one late (a tight cloud). two shapes cover most practice:

- **`StepLR(optimizer, step_size, gamma=0.1)`**: multiply by $\gamma$ every `step_size` steps, $\eta_t = \eta_0\gamma^{\lfloor t/\text{step\_size}\rfloor}$. the classic ImageNet recipe (÷10 at epochs 30, 60, 90).
- **`CosineSchedule(optimizer, total_steps, warmup_steps=0, min_lr=0)`**: after warmup, follow half a cosine from $\eta_0$ down to $\eta_\text{min}$,

$$\eta_t = \eta_\text{min} + (\eta_0 - \eta_\text{min})\,\frac{1 + \cos(\pi\tau)}{2},\qquad \tau = \frac{t - t_w}{T - t_w},$$

and stay at $\eta_\text{min}$ from step $T$ on (Loshchilov & Hutter, 2017). it spends a long time near the peak and decays smoothly to the end, with only one number to choose, $T$.

**why warm up.** during the first steps, everything is at its least reliable. Adam's first update is $\eta\,\hat m/\sqrt{\hat v} = \eta\,g/|g| = \pm\eta$ for every weight, however small its gradient (on paper 3): a full-size step on every coordinate, in a direction estimated from one minibatch. in a deep network, a large learning rate applied that way can push the weights into a region from which training never recovers. **warmup** ramps the learning rate linearly from 0 to $\eta_0$ over the first $t_w$ steps, $\eta_t = \eta_0\,t/t_w$, so that the optimizer's statistics (and, with normalization, the scale of the weights) settle before full-size steps begin. it became standard with large-batch training (Goyal et al., 2017) and transformers (chapter 18), which almost never train without it. the lab measures when it matters.

**the protocol.** a scheduler is a function `lr_at(step)` plus a counter. **`LRScheduler(optimizer)`** records `base_lr = get_lr(optimizer)` (the peak, $\eta_0$), sets `last_step = 0`, and immediately sets the optimizer to `lr_at(0)`; `step()` increments `last_step` and sets `lr_at(last_step)`. call it **once per optimizer step**, after `optimizer.step()`. (PyTorch's schedulers are often stepped once per epoch; ours are per step, so `total_steps` counts optimizer steps.) with warmup, `lr_at(0) = 0`, so the very first optimizer step is a no-op: that's the convention of Hugging Face's `get_cosine_schedule_with_warmup`, and it costs one step out of thousands. subclasses store their own settings *before* calling `super().__init__`, because the base constructor calls `lr_at(0)`.

> **🔮 predict:** with `CosineSchedule(opt, total_steps=10, warmup_steps=2, min_lr=0.1)` and a base learning rate of 1, what is the learning rate at step 6, halfway through the cosine?

> **▶ build: stage 2.** implement `LRScheduler`, `StepLR` and `CosineSchedule`.

### idea 3: gradient clipping

some minibatches produce gradients orders of magnitude larger than usual: a rare example with a huge loss, a recurrent network unrolled through a region where its Jacobians are large (chapter 17), a loss with a nearly flat region followed by a cliff. one such step with a normal learning rate can throw the weights far away. **clipping by global norm** caps the size of the step without changing its direction:

$$\|g\| = \sqrt{\textstyle\sum_p\|g_p\|^2},\qquad g_p \leftarrow g_p\cdot\frac{c}{\|g\|}\quad\text{if }\|g\| > c,$$

where the norm runs over *all* parameters' gradients together. scaling every gradient by the same factor keeps their relative sizes, so the update points the same way; only its length is capped at $\eta c$ for SGD (warm-up 3: 0.1, however large the raw gradient). clipping each coordinate separately (`np.clip(g, -c, c)`) would change the direction, and is rarely what you want.

**`clip_grad_norm(params, max_norm)`** skips parameters whose `.grad` is `None`, scales the gradients **in place** by `max_norm / (norm + 1e-6)` when the norm exceeds `max_norm` (the $10^{-6}$ is PyTorch's guard against dividing by zero), and returns the norm *before* clipping. that return value is a free diagnostic: log it every step, and spikes in the gradient norm show you where training gets into trouble (topic 05). it goes between `loss.backward()` and `optimizer.step()`; called before `backward()` it clips the previous step's (zeroed) gradients, and does nothing. (the capstone plants that bug.)

> **▶ build: stage 3.** implement `clip_grad_norm(params, max_norm)`.

### idea 4: finding the learning rate, and the batch size

the **learning-rate range test** (Smith, 2017) trains for a short while with an exponentially increasing learning rate, one minibatch per step,

$$\eta_i = \eta_\text{min}\Big(\frac{\eta_\text{max}}{\eta_\text{min}}\Big)^{i/(n-1)},\qquad i = 0, \dots, n - 1,$$

and records the loss at each step. the curve has three regions: a flat start (the learning rate is too small to change anything in a few steps), a descent (learning happens, faster and faster), and an explosion (too large). your **`lr_range_test`** stops as soon as a loss exceeds 4 times the best loss so far (or isn't finite), since everything after the explosion is noise. it's written for `mlfs.nn` models (on a torch model, `state_dict()` returns references to the live tensors, so "restoring" it would silently restore nothing). because it's an experiment, not training, it must leave no trace: save the model's `state_dict` and the optimizer's learning rate at the start and restore them at the end (in a `finally`, so they're restored even if something fails), and for an `mlfs.optim` optimizer also its per-parameter `state` (Adam's moments). it returns the lists `(lrs, losses)` of the steps it took. minibatches come from `minibatch_indices`, starting a new shuffled epoch whenever one runs out.

reading the curve: the minimum of the loss is *not* the learning rate to train with. at the minimum, the loss has just been driven down by the last few steps at a rate that is about to blow up; training there for thousands of steps would be at the edge of stability. a common rule, and the one **`suggest_lr`** uses, is a tenth of the learning rate at the minimum of the (smoothed) curve. smooth first, because each point is one noisy minibatch: an exponential moving average with bias correction, `avg = beta*avg + (1-beta)*loss`, `smoothed_k = avg / (1 - beta**k)`, with `beta = 0.8`.

**the batch size and the learning rate are coupled.** a minibatch gradient is an average of $B$ per-example gradients, so its noise variance shrinks like $1/B$. larger batches give cleaner gradients but fewer steps per epoch. the **linear scaling rule** (Goyal et al., 2017): when you multiply the batch size by $k$, multiply the learning rate by $k$, so that one large-batch step does roughly the work of $k$ small-batch steps (on paper 4). it works up to a point (a "critical batch size", McCandlish et al., 2018), beyond which larger batches stop helping and the scaled learning rate becomes unstable without warmup. the lab finds that point for a small MLP.

> **🔮 predict:** in a range test, why does the loss usually stay flat for the first two or three orders of magnitude of the sweep, rather than falling slowly from the start?

> **▶ build: stage 4.** implement `lr_range_test` and `suggest_lr(lrs, losses, *, beta=0.8)`.

## worked example

**a cosine schedule with warmup**, base learning rate 1, `total_steps=10`, `warmup_steps=2`, `min_lr=0.1`:

| step | phase | formula | lr |
|---|---|---|---|
| 0 | warmup | $1 \cdot 0/2$ | 0.0 |
| 1 | warmup | $1 \cdot 1/2$ | 0.5 |
| 2 | cosine, $\tau = 0$ | $0.1 + 0.9 \cdot 1$ | 1.0 |
| 3 | $\tau = 1/8$ | $0.1 + 0.45(1 + \cos(\pi/8))$ | 0.96575 |
| 4 | $\tau = 1/4$ | $0.1 + 0.45(1 + 0.70711)$ | 0.86820 |
| 6 | $\tau = 1/2$ | $0.1 + 0.45$ | 0.55 |
| 8 | $\tau = 3/4$ | $0.1 + 0.45(1 - 0.70711)$ | 0.23180 |
| 10, 11, … | done | $\eta_\text{min}$ | 0.1 |

the constructor sets the optimizer to 0.0 (step 0); after the $k$-th call of `step()` the optimizer's learning rate is the row for step $k$. the predict callout's answer is 0.55: halfway through a cosine is halfway between the peak and the floor.

**clipping by hand.** two parameters with gradients $(3, 0)$ and $(4)$, and a third with no gradient. the global norm is $\sqrt{9 + 0 + 16} = 5$. with `max_norm=1`, both gradients are scaled by $1/5$ (more precisely $1/(5 + 10^{-6})$): $(0.6, 0)$ and $(0.8)$, a combined norm of 1, pointing the same way. `clip_grad_norm` returns 5. the tests check both examples, the schedulers against PyTorch's `StepLR` and `CosineAnnealingLR`, and clipping against `torch.nn.utils.clip_grad_norm_`.

## on paper

1. **SGD's noise floor.** minimize $f(w) = \frac h2w^2$ with noisy gradients $g_t = hw_t + \xi_t$, $E[\xi_t] = 0$, $\operatorname{Var}(\xi_t) = \sigma^2$, independent. show that with $0 < \eta h < 2$, $E[w_t^2]$ converges to $\frac{\eta\sigma^2}{h(2 - \eta h)} \approx \frac{\eta\sigma^2}{2h}$.

<details><summary>solution</summary>

$w_{t+1} = (1 - \eta h)w_t - \eta\xi_t$, so $E[w_{t+1}^2] = (1 - \eta h)^2E[w_t^2] + \eta^2\sigma^2$ (the cross term vanishes because $\xi_t$ is independent of $w_t$ with mean 0). the fixed point $v = (1 - \eta h)^2v + \eta^2\sigma^2$ gives $v = \frac{\eta^2\sigma^2}{1 - (1 - \eta h)^2} = \frac{\eta^2\sigma^2}{\eta h(2 - \eta h)} = \frac{\eta\sigma^2}{h(2 - \eta h)}$, and the recursion converges to it when $|1 - \eta h| < 1$. for small $\eta h$ it's $\eta\sigma^2/(2h)$: linear in the learning rate. decaying $\eta$ is how you shrink it.

</details>

2. **clipping preserves the direction.** show that clipping by global norm leaves $g/\|g\|$ unchanged and bounds the SGD step by $\eta c$. give a two-coordinate example where clipping each coordinate to $[-c, c]$ changes the direction.

<details><summary>solution</summary>

if $\|g\| > c$, the clipped gradient is $\alpha g$ with $\alpha = c/\|g\| > 0$, so it's a positive multiple of $g$ (same direction), with norm $c$; otherwise it's unchanged with norm $\le c$. the step is $\eta\alpha g$, of norm at most $\eta c$. coordinate clipping of $g = (10, 1)$ with $c = 1$ gives $(1, 1)$: the direction rotates from almost pure $x$ to $45°$.

</details>

3. **Adam's first step.** with $m_0 = v_0 = 0$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, compute $\hat m_1$, $\hat v_1$ and the first update. what does it mean for a weight whose gradient is $10^{-8}$?

<details><summary>solution</summary>

$m_1 = 0.1g$, $v_1 = 0.001g^2$; bias correction divides by $1 - 0.9 = 0.1$ and $1 - 0.999 = 0.001$: $\hat m_1 = g$, $\hat v_1 = g^2$. the update is $-\eta g/(|g| + \epsilon) \approx -\eta\operatorname{sign}(g)$. a weight with gradient $10^{-8}$ moves by almost a full $\eta$ (the $\epsilon = 10^{-8}$ halves it), exactly like a weight with gradient 1. with $\eta = 0.03$ and weights of size $\sim 0.1$ (kaiming, 128 wide), the first step changes every weight by 30% in a direction chosen by one minibatch: the reason for warmup.

</details>

4. **the linear scaling rule.** suppose the gradient barely changes over $k$ consecutive SGD steps. compare $k$ steps with batch size $B$ and learning rate $\eta$ with one step with batch size $kB$ and learning rate $k\eta$.

<details><summary>solution</summary>

$k$ small steps: $w_{t+k} \approx w_t - \eta\sum_{j<k}\frac1B\sum_{i \in \mathcal B_j}\nabla\ell_i(w_t)$, using the gradient at $w_t$ throughout (the assumption). one large step: $w_t - k\eta\frac{1}{kB}\sum_{i \in \cup\mathcal B_j}\nabla\ell_i(w_t)$, the same sum. so the two agree to first order. the assumption fails when $k\eta$ is large enough that the gradient changes a lot over the step (early in training, and at large $k$), which is where the rule breaks and warmup helps.

</details>

## experiment

`lab.py` uses your schedulers and range test on MNIST:

1. a range test for an 8-layer network with Adam, and a check of its suggestion by training at the suggestion, 10× less and 10× more.
2. Adam with and without linear warmup, across peak learning rates, for a plain and a layer-norm network.
3. batch sizes from 8 to 2,048, with a fixed learning rate and with the linear scaling rule.

## check your understanding

1. why must a scheduler call `set_lr` in its constructor, before any `step()`?

<details><summary>answer</summary>

so that the very first optimizer step uses `lr_at(0)`, not whatever learning rate the optimizer was created with. with warmup the difference is everything: the optimizer was created with the *peak* learning rate, and without the constructor call the first step would take it at full size, the very thing warmup exists to prevent.

</details>

2. you step a per-step `CosineSchedule(total_steps=10_000)` once per epoch instead of once per optimizer step, with 500 steps per epoch. what happens?

<details><summary>answer</summary>

after 20 epochs (10,000 optimizer steps) the schedule has advanced only 20 of its 10,000 steps: the learning rate is still essentially at its peak, and the run ends without ever decaying. the opposite mistake, stepping an epoch-based schedule per batch, decays it 500 times too fast. (the capstone plants one of them.)

</details>

3. `clip_grad_norm` returns the norm before clipping. why is that more useful than the norm after?

<details><summary>answer</summary>

after clipping, the norm is just `min(norm, max_norm)`: you know it already. the raw norm tells you how often clipping activates and by how much: if it's clipping on every step, `max_norm` is too small (or the learning rate too large) and clipping has silently become a learning-rate reduction; if it never clips, it's doing nothing.

</details>

4. why does the range test restore the weights afterwards, rather than keeping the progress it made?

<details><summary>answer</summary>

the last steps were taken at learning rates past the explosion: the weights are damaged, and Adam's moments are full of huge gradients. it would also make experiments depend on whether you ran a range test first. restoring makes it a pure measurement.

</details>

5. warmup makes the first steps smaller. why can that lead to a *better final* model, not just a slower start?

<details><summary>answer</summary>

because the damage done by a few early oversized steps isn't undone later: they can kill ReLUs, saturate units, or blow up a layer's scale so that normalization layers and Adam's statistics settle into a bad regime. the lab shows networks that end at 10% accuracy without warmup and 89% with it, at the same peak learning rate.

</details>

6. you double the batch size and the learning rate, and accuracy after a fixed number of epochs stays the same. you do it again, and again, and suddenly training diverges. what happened?

<details><summary>answer</summary>

the linear scaling rule assumes the gradient barely changes over the span of the big step. at large learning rates that's false, and the step exceeds what the loss's curvature allows (for a quadratic, $\eta > 2/h$ diverges). the batch size has passed the critical batch size: beyond it, larger batches no longer reduce useful noise, and the scaled learning rate is just too large. warmup extends the range a bit; nothing extends it forever.

</details>

## common bugs

- **setting `.lr` on a torch optimizer.** symptom: the "schedule" does nothing at all. torch reads `param_groups`.
- **storing settings after `super().__init__`.** symptom: `AttributeError` in `lr_at` during construction (the base constructor calls `lr_at(0)`).
- **stepping the scheduler at the wrong frequency.** symptom: a learning rate that never decays, or that collapses to almost zero after the first epoch. per optimizer step means per batch.
- **clipping before `backward()`.** symptom: clipping "works" (no errors) but never changes anything, and the gradient spikes it was meant to stop still happen.
- **per-parameter clipping when you meant global.** symptom: the update's direction changes; layers with large gradients are slowed relative to others.
- **training at the range test's minimum.** symptom: the first epoch looks great, then the loss jumps and never recovers. back off by about 10×.

## going deeper

- Smith (2017), "Cyclical learning rates for training neural networks", *WACV*. the learning-rate range test. [arXiv:1506.01186](https://arxiv.org/abs/1506.01186)
- Loshchilov & Hutter (2017), "SGDR: stochastic gradient descent with warm restarts", *ICLR*. the cosine schedule. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
- Goyal, Dollár, Girshick, Noordhuis, Wesolowski, Kyrola, Tulloch, Jia & He (2017), "Accurate, large minibatch SGD: training ImageNet in 1 hour". the linear scaling rule and gradual warmup. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- McCandlish, Kaplan, Amodei & OpenAI Dota Team (2018), "An empirical model of large-batch training". the gradient noise scale and the critical batch size. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
- Pascanu, Mikolov & Bengio (2013), "On the difficulty of training recurrent neural networks", *ICML*. exploding gradients and norm clipping. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
- Liu, Jiang, He, Chen, Liu, Gao & Han (2020), "On the variance of the adaptive learning rate and beyond", *ICLR*. why Adam needs warmup (RAdam). [arXiv:1908.03265](https://arxiv.org/abs/1908.03265)

## connections

- ← builds on: SGD's noise, momentum and Adam (ch 06 · topic 04), the training loop and `state_dict` (ch 14 · topic 04), layer norm (topic 02).
- → used in: gradient-norm diagnostics (topic 05), hyperparameter searches over the learning rate (topic 06), `torchnn.fit(scheduler=..., grad_clip=...)` driving torch optimizers through your `set_lr` (topic 07), the capstone's scheduler and clipping bugs, ResNet training recipes (ch 16), clipping in RNNs (ch 17), warmup + cosine for transformers (ch 18).
