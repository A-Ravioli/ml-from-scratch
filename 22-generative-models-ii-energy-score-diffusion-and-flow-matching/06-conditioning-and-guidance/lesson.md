# conditioning & guidance

## why this matters

an unconditional generator draws *some* image. you usually want a particular kind: a digit 7, a ship, "an astronaut riding a horse". the direct approach is to feed the label to the network, $\epsilon_\theta(x_t, t, y)$ or $v_\theta(x_t, t, y)$, and train on labelled pairs; nothing else changes. it works, and for a while that's how class-conditional diffusion was done. but conditional samples were disappointing: recognizably "about" the label on average, yet often ambiguous, mixed, or simply off-class. the model had learned $p(x \mid y)$, broad and diverse, when what people wanted was the *typical* example of $y$.

**guidance** is the fix, and it's the single most important trick in practical text-to-image systems: at sampling time, push the score further in the direction the label adds than the model itself suggests. the classic version uses a separate classifier's gradient (Dhariwal & Nichol, 2021). the version everyone uses now, **classifier-free guidance** (Ho & Salimans, 2021), needs no classifier at all: train one network to be both conditional and unconditional by randomly hiding the label, and at sampling time extrapolate from the unconditional prediction through the conditional one. one number, the guidance scale, then trades diversity for fidelity, and the lab measures that trade.

## warm-up: predict first

1. you train a network that sometimes sees the label and sometimes doesn't. what does it predict when it doesn't?
2. $\nabla\log p(x \mid y) - \nabla\log p(x)$ equals the gradient of the log of what?
3. a guidance scale of $w = 3$: samples more typical of their class, or less? more diverse, or less?
4. can strong guidance produce samples *outside* the region where the training data of that class lie?
5. if the label is never hidden during training, what goes wrong with classifier-free guidance?

## prerequisites check

- [ ] noise prediction as a scaled score; `ddpm_sample`, `ddim_sample` (→ topics 03–04)
- [ ] flow matching and `sample_flow` (→ topic 05)
- [ ] Bayes' rule for densities; softmax classifiers and their log-probabilities (→ ch 04; ch 08 · topic 03)
- [ ] the null-label convention of `TimeMLP` / `TimeUNet` (→ topic 03)
- [ ] precision–recall and FID as diversity/fidelity measures (→ ch 21 · topic 01)

## core ideas

### idea 1: one network, two models: label dropout

a conditional network $\epsilon_\theta(x_t, t, y)$ trained on pairs $(x_0, y)$ learns the noise predictor of $p(x \mid y)$: by Tweedie, $\epsilon^*(x_t, t, y) = -\sqrt{1 - \bar\alpha_t}\,\nabla\log p_t(x_t \mid y)$. now hide the label at random: with probability $p_\text{uncond}$ (typically 0.1–0.2), replace $y$ by the **null label** $\varnothing$ (here, the extra class index $K$). for those examples the best prediction is the average over all classes, the *unconditional* noise predictor, so the same network learns both

$$\epsilon_\theta(x_t, t, y) \approx -\sqrt{1 - \bar\alpha_t}\,\nabla\log p_t(x_t \mid y), \qquad \epsilon_\theta(x_t, t, \varnothing) \approx -\sqrt{1 - \bar\alpha_t}\,\nabla\log p_t(x_t),$$

at the cost of a slightly smaller effective dataset for each. that's all classifier-free guidance needs from training. if $p_\text{uncond} = 0$, the null row of the embedding table is never trained and the "unconditional" output is garbage.

> **spec · stage 1.** `drop_labels(y, p_uncond, null_label, *, generator=None)`: a new tensor where label $i$ becomes `null_label` when `torch.rand(len(y), generator=generator)[i] < p_uncond` (drawn on the CPU); same dtype and device as `y`, which is not modified. this is `mlfs.diffusion.drop_labels`.

> **▶ build: stage 1.** implement `drop_labels` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: classifier-free guidance

Bayes' rule splits the conditional score into the unconditional one plus the gradient of an (implicit) classifier:

$$\nabla\log p_t(x \mid y) = \nabla\log p_t(x) + \nabla\log p_t(y \mid x).$$

guidance amplifies the classifier part by a factor $w$:

$$\nabla\log p_t(x) + w\,\nabla\log p_t(y \mid x) = \nabla\log p_t(x) + w\big(\nabla\log p_t(x \mid y) - \nabla\log p_t(x)\big),$$

and since noise predictions are scaled scores, the same combination of the network's two outputs is

$$\tilde\epsilon(x, t, y) = \epsilon_\theta(x, t, \varnothing) + w\,\big(\epsilon_\theta(x, t, y) - \epsilon_\theta(x, t, \varnothing)\big).$$

$w = 0$ is the unconditional model, $w = 1$ the plain conditional one, $w > 1$ extrapolates past it. (conventions differ: some papers write $(1 + w)\epsilon_y - w\epsilon_\varnothing$, so their $w = 0$ is this lesson's $w = 1$.) at $w > 1$ the guided score is the score of $p_t(x)^{1 - w}p_t(x \mid y)^w$, which concentrates on points where $p(y \mid x)$ is large: typical, unambiguous members of the class, and fewer of them in the regions the classes share. for Gaussians you can see the extrapolation in closed form: the guided distribution is narrower than the class, and its mean overshoots *past* the class mean, away from the other classes (on paper 2).

because the combination is affine with weights summing to 1, it works unchanged for scores, noise predictions and flow-matching velocities (on paper 4), and with any sampler: guidance is a wrapper around the model. one practical point: the two predictions are needed for the same $(x, t)$ at every step, so evaluate them in **one batched call**, concatenating $[x, x]$ with labels $[y, \varnothing]$, which halves the overhead.

> **spec · stage 2.** `cfg_model(model, guidance_scale, null_label)` returns an `nn.Module` $g$ that holds `model` as a submodule (so `g.eval()` reaches it) with `g(x, t, y)` = $m(x, t, \varnothing) + w\,(m(x, t, y) - m(x, t, \varnothing))$ computed with one call `model(cat([x, x]), cat([t, t]), cat([y, null]))`, where `null` is `y` filled with `null_label`; `y=None` returns the unconditional output. this is `mlfs.diffusion.cfg_model`.

> **▶ build: stage 2.** implement `cfg_model` (and the module it returns).

> **🔮 predict:** two overlapping Gaussian classes in 2-D. as $w$ goes from 1 to 8, what happens to the samples' spread, and to their distance from the other class?

### idea 3: classifier guidance

the original guidance (Dhariwal & Nichol, 2021) used a real classifier instead of the implicit one: train $p_\phi(y \mid x_t, t)$ on *noisy* inputs at every noise level (an ordinary classifier never saw noisy images and gives meaningless gradients on them), and add its input gradient to an unconditional model's score:

$$\tilde\epsilon(x, t, y) = \epsilon_\theta(x, t) - w\,\sqrt{1 - \bar\alpha_t}\,\nabla_x\log p_\phi(y \mid x, t).$$

with the exact classifier and $w = 1$ this is, by Bayes' rule, *exactly* the conditional noise predictor; the checks verify it on a two-class Gaussian mixture where every term is known in closed form. in practice classifier guidance has two drawbacks that made classifier-free guidance win: a second network that must be trained on noisy data, and gradients of a classifier that can be "fooled" (an adversarial direction raises $p_\phi(y \mid x)$ without making the image look more like $y$). its advantage: any classifier, trained after the fact, can steer a frozen unconditional model. since samplers call the model under `torch.no_grad()`, the classifier gradient must turn gradients on locally, as topic 01's `score_from_energy` did.

> **spec · stage 3.** `classifier_guided_eps(eps_model, classifier, schedule, guidance_scale)` returns an `nn.Module` holding both networks whose forward `(x, t, y)` computes `eps_model(x, t)` (called *without* labels: an unconditional model, or a conditional one using its null label) and, unless `y is None`, subtracts $w\sqrt{1 - \bar\alpha(t)}\,\nabla_x\sum_b \log\operatorname{softmax}(\text{classifier}(x, t))_{b, y_b}$ (the classifier is called as `classifier(x, t)` and returns logits `(B, K)`), with $\bar\alpha(t)$ from topic 04's `alpha_bar_at` and the gradient computed under `torch.enable_grad()`.

> **▶ build: stage 3.** implement `classifier_guided_eps`.

### idea 4: training with label dropout

the training loop is the one you know, with two changes: labels go through `drop_labels` before every step, and the objective is a choice, `ddpm_loss` (noise prediction, needs a schedule) or `flow_matching_loss` (velocity). the capstone trains both on CIFAR-10 with the same network and budget and compares them under guidance.

> **spec · stage 4.** `fit_conditional(model, X, y, n_classes, *, objective='flow', schedule=None, p_uncond=0.1, n_iters=2000, batch_size=256, lr=1e-3, seed=0)`: `g = torch.Generator().manual_seed(seed)`, `torch.optim.Adam(lr)`, train mode; per step `idx = torch.randint(N, (batch_size,), generator=g)`, labels `drop_labels(y[idx], p_uncond, n_classes, generator=g)`, then `flow_matching_loss(model, X[idx], y=labels, generator=g)` for `'flow'` or `ddpm_loss(model, schedule, X[idx], y=labels, generator=g)` for `'ddpm'` (`ValueError` for any other objective, and for `'ddpm'` without a schedule). returns the losses as floats.

> **▶ build: stage 4.** implement `fit_conditional`.

> **🔮 predict:** you train with $p_\text{uncond} = 0$ (the label is never hidden), then sample with $w = 4$. what do you expect?

## worked example

**two classes in one dimension.** classes $\mathcal{N}(\pm 1, 0.5^2)$ with equal priors, a noise level with $\bar\alpha = 0.5$: each noisy class is $\mathcal{N}(\pm\sqrt{0.5}, v)$ with $v = 0.5 \cdot 0.25 + 0.5 = 0.625$. at $x = 0.2$:

- conditional predictors $\epsilon(x \mid y) = \sqrt{1 - \bar\alpha}\,(x - \sqrt{\bar\alpha}\mu_y)/v$: $\epsilon(\cdot \mid +) = -0.574$, $\epsilon(\cdot \mid -) = 1.026$;
- posteriors $p(+ \mid x) = 0.611$, $p(- \mid x) = 0.389$, so the unconditional predictor (their posterior-weighted average) is $\epsilon(\cdot \mid \varnothing) = 0.048$;
- classifier-free guidance for $y = +$ with $w = 3$: $0.048 + 3(-0.574 - 0.048) = -1.818$, three times as far from the unconditional prediction as the conditional one is;
- classifier guidance with the exact classifier: $\frac{d}{dx}\log p(+ \mid x) = 0.880$, and $0.048 - 1 \cdot \sqrt{0.5}\cdot 0.880 = -0.574$: exactly $\epsilon(\cdot \mid +)$.

the checks rebuild this mixture at a non-grid time and require your classifier-guided predictor to equal the conditional one to $10^{-5}$.

## on paper

1. **Bayes for scores.** show $\nabla_x\log p(x \mid y) = \nabla_x\log p(x) + \nabla_x\log p(y \mid x)$, and explain why the normalizer $p(y)$ doesn't appear.

<details><summary>solution</summary>

$p(x \mid y) = p(y \mid x)p(x)/p(y)$, so $\log p(x \mid y) = \log p(y \mid x) + \log p(x) - \log p(y)$; $p(y)$ doesn't depend on $x$, so it vanishes under $\nabla_x$. the same holds at every noise level, with $p_t$ in place of $p$.

</details>

2. **guidance overshoots.** let $p(x) = \mathcal{N}(0, v_0)$ and $p(x \mid y) = \mathcal{N}(m, v_1)$ with $v_1 < v_0$ (a class inside a wider population). show that $p(x)^{1 - w}p(x \mid y)^w$ is Gaussian with mean $w m/(w + (1 - w)v_1/v_0)$ and variance $1/\big((1 - w)/v_0 + w/v_1\big)$, and that for $w > 1$ the mean lies beyond $m$ and the variance below $v_1$.

<details><summary>solution</summary>

the log is a quadratic with precision $P = (1 - w)/v_0 + w/v_1$ and linear coefficient $w m/v_1$, so the mean is $(w m/v_1)/P = w m/(w + (1 - w)v_1/v_0)$. for $w > 1$, $(1 - w)v_1/v_0 \in (1 - w, 0)$, so the denominator is in $(1, w)$ and the mean exceeds $m$ (for $m > 0$). and $P - 1/v_1 = (w - 1)(1/v_1 - 1/v_0) > 0$, so the variance is below $v_1$. guided samples are both sharper than the class and pushed away from the rest of the population: at large $w$, outside the data.

</details>

3. **classifier guidance in noise units.** starting from $\tilde s = s + w\nabla\log p(y \mid x)$ and $\epsilon = -\sqrt{1 - \bar\alpha}\,s$, derive the classifier-guided noise prediction.

<details><summary>solution</summary>

$\tilde\epsilon = -\sqrt{1 - \bar\alpha}\,\tilde s = -\sqrt{1 - \bar\alpha}\,s - w\sqrt{1 - \bar\alpha}\,\nabla\log p(y \mid x) = \epsilon - w\sqrt{1 - \bar\alpha}\,\nabla\log p(y \mid x)$.

</details>

4. **guiding velocities.** for the straight path $x_t = (1 - t)x_0 + t x_1$ with Gaussian noise $x_0$, Tweedie gives $\mathbb{E}[x_0 \mid x_t] = -(1 - t)\nabla\log p_t(x_t)$. show that the marginal velocity is $v = (x + (1 - t)\nabla\log p_t(x))/t$, and conclude that classifier-free guidance applied to velocities equals guidance applied to scores.

<details><summary>solution</summary>

$x_1 = (x_t - (1 - t)x_0)/t$, so $v = \mathbb{E}[x_1 - x_0 \mid x_t] = (x - (1 - t)\mathbb{E}[x_0 \mid x])/t - \mathbb{E}[x_0 \mid x] = (x - \mathbb{E}[x_0 \mid x])/t = (x + (1 - t)\nabla\log p_t)/t$. the same holds conditionally on $y$. the term $x/t$ is the same for $v(\cdot, y)$ and $v(\cdot, \varnothing)$, and an affine combination with weights $(1 - w) + w = 1$ keeps it once, so $\tilde v = (x + (1 - t)\tilde s)/t$ with $\tilde s$ the guided score.

</details>

## experiment

`lab.py` trains a class-conditional flow on 2-D data with your `fit_conditional` and samples it with your `cfg_model`:

1. the guidance scale against fidelity and diversity.
2. what label dropout does, including $p_\text{uncond} = 0$.
3. classifier-free against classifier guidance on the same data.

## check your understanding

1. why does a single network with label dropout give a better unconditional model for guidance than training a separate unconditional network?

<details><summary>answer</summary>

it isn't necessarily better as an unconditional model, but it's cheaper (one network, one training run) and the two predictions share features and errors: their difference, which is what guidance amplifies, is then a cleaner estimate of the classifier gradient than the difference of two independently trained networks, whose unrelated errors would be amplified by $w$ too.

</details>

2. what does $w = 0$ produce? $w = 1$? $w < 0$?

<details><summary>answer</summary>

$w = 0$: unconditional samples (the label is ignored). $w = 1$: samples from the conditional model $p(x \mid y)$. $w < 0$: pushes samples *away* from the class ("negative guidance"); in text-to-image systems a variant of this is the "negative prompt".

</details>

3. why does strong guidance reduce diversity?

<details><summary>answer</summary>

the guided target $\propto p(x)p(y \mid x)^w$ weights each point by $p(y \mid x)^w$; as $w$ grows, only the points where the class is most certain survive, a shrinking region (the variance in on paper 2 drops below the class's). FID, which compares whole distributions, typically gets worse at large $w$ while per-sample fidelity (classifier confidence) keeps improving.

</details>

4. why must the classifier in classifier guidance be trained on noisy inputs?

<details><summary>answer</summary>

guidance evaluates it on $x_t$ at every noise level. a clean-image classifier has never seen such inputs; its predictions there are arbitrary and its gradients point nowhere useful. it must model $p(y \mid x_t, t)$, the posterior over classes given a noisy image, which is also why it takes $t$ as an input.

</details>

5. does the guided sampler draw exactly from $p(x)^{1-w}p(x \mid y)^w$ (normalized)?

<details><summary>answer</summary>

no. at each noise level the guided score is the score of $p_t(x)^{1-w}p_t(x \mid y)^w$, but these intermediate densities are not the noised versions of the final sharpened density (noising doesn't commute with taking powers), so the reverse process doesn't target that density exactly. it's a heuristic that works remarkably well, not an exact sampler of a known distribution.

</details>

## common bugs

- **null label off by one.** symptom: the null label collides with a real class (index $K - 1$ instead of $K$) or indexes past the table; the table has $K + 1$ rows and the null label is $K$.
- **two forward calls instead of one.** symptom: guidance doubles the sampling time; concatenate $[x, x]$ and $[y, \varnothing]$.
- **the batch halves swapped.** symptom: $w = 3$ gives *less* class-typical samples; check which half is conditional before taking the difference.
- **a guidance wrapper that isn't a module.** symptom: samplers can't switch it to eval mode, so dropout stays on inside the network during sampling.
- **modifying the labels in place.** symptom: after training, the dataset's labels are full of null labels; `drop_labels` must return a new tensor.
- **classifier gradients under `no_grad`.** symptom: "element 0 of tensors does not require grad" during sampling.

## going deeper

- Ho & Salimans (2021), "Classifier-free diffusion guidance", *NeurIPS workshop on deep generative models*. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Dhariwal & Nichol (2021), "Diffusion models beat GANs on image synthesis", *NeurIPS*. classifier guidance. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
- Nichol et al. (2022), "GLIDE: towards photorealistic image generation and editing with text-guided diffusion models", *ICML*. CFG for text. [arXiv:2112.10741](https://arxiv.org/abs/2112.10741)
- Kynkäänniemi, Aittala, Karras, Laine, Aila & Lehtinen (2024), "Applying guidance in a limited interval improves sample and distribution quality in diffusion models", *NeurIPS*. [arXiv:2404.07724](https://arxiv.org/abs/2404.07724)
- Chidambaram, Gatmiry, Chen, Lee & Lu (2024), "What does guidance do? A fine-grained analysis in a simple setting". [arXiv:2409.13074](https://arxiv.org/abs/2409.13074)

## connections

- ← builds on: scores and Bayes' rule (topic 02), noise prediction and samplers (topics 03–04), flow matching (topic 05), the null-label convention (topic 03), fidelity/diversity metrics (ch 21 · topic 01).
- → used in: the chapter capstone (class-conditional CIFAR-10 with CFG, DDPM vs flow matching), latent diffusion with a conditional DiT (topic 07), text conditioning with cross-attention in multimodal models (ch 25), and guidance-like reward steering in later chapters.
