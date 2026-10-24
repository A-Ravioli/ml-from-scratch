# attribution & probing

## why this matters

a classifier trained to recognise horses scored well on its test set. when Lapuschkin et al. (2019) asked it *which pixels* mattered, the answer was a copyright watermark in the corner of many horse photos: paste the watermark onto a car and the car became a horse. accuracy alone could never have told you this, because the shortcut works on any test set drawn the same way. an **attribution method** answers the question "which parts of this input made the model say that?", and when it works, it catches shortcuts, spurious correlations and bugs that no aggregate metric reveals.

the trouble is that attribution methods are themselves models of the model, and they can be wrong in ways that look convincing. Adebayo et al. (2018) showed that some popular saliency maps barely change when you replace the trained network's weights with random numbers: a map that survives the destruction of the model can't be explaining the model. the same worry applies one level deeper. a **probe** (a small classifier trained on a layer's activations) can tell you that some property is *decodable* from that layer, but a powerful enough probe decodes almost anything, and "decodable" isn't "used".

this topic builds the two basic tools, gradients and integrated gradients for inputs and linear probes for hidden layers, and then the controls that keep each of them honest: the completeness check, the control task and the randomization test. the habit to take away is the experimental one: an explanation is a hypothesis, and it needs a control.

## warm-up: predict first

1. a ReLU network classifies an image with very high confidence; its logit for the class is saturated in some pixels (making them brighter can't raise it further). what does the plain gradient say about those pixels?
2. for a linear model $f(x) = w^\top x + b$, what is the "right" attribution of $f(x) - f(0)$ to each input feature?
3. a linear probe on layer 5 of a language model predicts part-of-speech with 97% accuracy. does that mean layer 5 represents part of speech?
4. you replace every weight of a trained CNN with random values and recompute its saliency maps. will they still look like the input image?

## prerequisites check

- [ ] you can get the gradient of a scalar with respect to an input tensor with `torch.autograd.grad` (→ ch 14 · 03, ch 15 · 07)
- [ ] you know the trapezoid rule and its error (→ ch 01 · 04, `mlfs.numerics`)
- [ ] you can train and evaluate a linear probe on frozen features (→ ch 25 · 01, `mlfs.ssl.linear_probe`)
- [ ] you know what Spearman's rank correlation measures (→ ch 05)
- [ ] you know LeNet and what `model.eval()` changes (→ ch 16 · 03, ch 15 · 03)

## core ideas

### idea 1: the gradient as an explanation

near an input $x$, any differentiable model is approximately linear: $f_c(x + \delta) \approx f_c(x) + \nabla_x f_c(x)^\top \delta$, where $f_c$ is the logit of class $c$. the gradient says how much each input feature would move the logit if you nudged it, so its magnitude is a natural first answer to "which pixels matter". Simonyan et al. (2014) called the map $|\nabla_x f_c(x)|$ a **saliency map**. it costs one backward pass.

two details matter. first, explain a **logit**, not a softmax probability: the probability of a confident prediction is flat in almost every direction (its gradient is tiny), and it mixes in every other class. second, compute it in **eval mode**: dropout would make the explanation random, and batch norm in train mode would make one image's explanation depend on the others in the batch.

the gradient's weakness is that it's *local*. it describes the function in an infinitesimal neighbourhood of $x$, and a network that has already "decided" is often flat there: a ReLU that's off contributes nothing to the gradient, and a feature whose effect has saturated has zero gradient however important it was in getting there. (warm-up 1: the plain gradient calls a saturated pixel unimportant.) gradient maps of image classifiers are also visually noisy, with speckle all over the background, because the gradient of a piecewise-linear network jumps between linear regions. smoothing tricks like SmoothGrad (Smilkov et al., 2017) average the gradient over noisy copies of the input; they make prettier pictures, and the sanity checks of idea 4 apply to them too.

> **spec.** `saliency(model, x, target)`: `model` maps a batch `x` to logits `(B, C)`; `target` is an int or an int64 tensor `(B,)` (one class per row). returns $|\partial\, \text{model}(x)[b, \text{target}_b] / \partial x[b]|$, same shape as `x`. the model runs in eval mode, and its previous mode is restored afterwards; parameter `.grad`s are not touched (use `torch.autograd.grad`); `x` isn't modified; it works when called inside `torch.no_grad()` (use `torch.enable_grad()`).

> **▶ build: stage 1.** implement `saliency`.

### idea 2: integrated gradients

Sundararajan et al. (2017) took a different route: decide what properties an attribution should have, then find a method that has them. two of their axioms:

- **completeness.** the attributions $a_i(x)$ add up to the change in output between a **baseline** $x'$ (an input that means "nothing", like a black image) and the input: $\sum_i a_i(x) = f(x) - f(x')$. nothing is left unexplained.
- **sensitivity.** if changing one feature from the baseline changes the output, that feature gets a nonzero attribution. the gradient fails this (saturation).

the method that satisfies them integrates the gradient along the straight path from the baseline to the input:

$$\mathrm{IG}_i(x) = (x_i - x'_i) \int_0^1 \frac{\partial f}{\partial x_i}\big(x' + \alpha (x - x')\big)\, d\alpha.$$

completeness is the fundamental theorem of calculus for line integrals (on paper 1): summing over $i$ gives $\int_0^1 \frac{d}{d\alpha} f(x' + \alpha(x - x'))\, d\alpha = f(x) - f(x')$. a saturated pixel still gets credit, because the gradient was large *somewhere* on the way from black to the final image. for a linear model, IG is exactly $(x_i - x'_i) w_i$ (warm-up 2): the contribution of each feature to the change.

in code the integral becomes a sum. with $n$ intervals and the **trapezoid rule**, evaluate the gradient at $\alpha_k = k/n$ for $k = 0, \dots, n$ and weight the two end points by $\frac{1}{2n}$ and the rest by $\frac{1}{n}$. the error shrinks like $1/n^2$ for a smooth path. a ReLU network's gradient is piecewise *constant* along the path, jumping wherever a unit switches on or off; there the trapezoid rule's error shrinks only like $1/n$, so expect to need tens of steps, not three. the completeness gap $|\sum_i a_i - (f(x) - f(x'))|$ is a free diagnostic: compute it and increase $n$ until it's small.

the baseline is a modelling choice, not a detail. IG answers "what made the output differ from the output on $x'$?", so a black baseline can never attribute anything to black pixels ($x_i - x'_i = 0$), and a different baseline (a blurred image, the dataset mean, noise averaged over draws) asks a different question.

> **spec.** `integrated_gradients(model, x, target, *, baseline=None, n_steps=50)`: `baseline` is a tensor broadcastable to `x` (default zeros). returns $(x - x') \odot \sum_{k=0}^{n} w_k \nabla_x f_{\text{target}}(x' + \tfrac{k}{n}(x - x'))$ with trapezoid weights $w_0 = w_n = \frac{1}{2n}$, $w_k = \frac{1}{n}$ otherwise, $n$ = `n_steps`; same shape as `x`, signed. eval mode, previous mode restored, parameter grads untouched, works inside `torch.no_grad()`; `ValueError` if `n_steps < 1`.

> **🔮 predict:** for a model $f(x) = x_1^2 + x_1 x_2$ at $x = (2, 1)$ with a zero baseline, how should the output $f(x) - f(0) = 6$ be split between $x_1$ and $x_2$? (the worked example has the answer.)

> **▶ build: stage 2.** implement `integrated_gradients`.

### idea 3: probes, and the control task

attribution looks at the input. to ask what a model represents *inside*, train a **probe**: a simple classifier that reads a hidden layer's activations and predicts some property (part of speech, the colour of an object, whether a board position is legal). Alain & Bengio (2016) used linear probes on every layer of a network to watch class information become linearly decodable with depth.

a probe's accuracy is a statement about the *layer plus the probe*. two failure modes follow:

- **decodable isn't used.** information can be present in a layer and ignored by everything downstream. to show that the model *uses* it you need an intervention (topic 02): change the representation along the probe's direction and watch the output change.
- **the probe can do the work.** a probe with enough capacity learns the task itself from whatever the layer offers. if a layer simply identifies the word (one direction per vocabulary item), a probe can map each word to its usual part of speech by memorisation, and report 95% "part-of-speech accuracy" from a layer that knows nothing about syntax.

Hewitt & Liang (2019) proposed a control for the second one. a **control task** assigns every *type* (each distinct word, say) a label drawn at random, once, and keeps it fixed: every occurrence of "the" gets the same random label. (Hewitt & Liang draw each type's label from the real task's empirical label distribution; this topic draws it uniformly, which is the same thing for balanced labels and simpler to test.) the control task has the same label distribution and the same type structure as the real task, but no linguistic meaning, so the only way to solve it is to memorise types. **selectivity** = task accuracy − control accuracy. a high-accuracy, low-selectivity probe is reporting that the layer identifies types, not that it encodes the property. the random table is drawn once and shared between the train and test splits; draw two tables and the control task becomes impossible, and selectivity looks spuriously high.

> **spec.** `control_labels(types, n_classes, *, rng=None)`: `types` `(n,)` of hashable values; draws one label uniformly from `range(n_classes)` per distinct type with `mlfs.utils.as_rng(rng)`, and returns the `(n,)` int64 labels (equal types get equal labels). `probe_selectivity(train_feats, train_labels, test_feats, test_labels, train_types, test_types, *, rng=None)`: `n_classes` = 1 + the largest label in either split; ONE control table for the types of both splits; fits `mlfs.ssl.linear_probe` (default settings) on the task and on the control labels; returns `{"task": acc, "control": acc, "selectivity": task - control}` as floats.

> **▶ build: stage 3.** implement `control_labels` and `probe_selectivity`.

### idea 4: sanity checks for saliency maps

how do you test an explanation when you don't know the right answer? Adebayo et al. (2018) proposed two tests that need no ground truth, only the requirement that an explanation of the model should *depend on* the model:

- **model parameter randomization.** re-initialise the network's weights, from the top layer down (**cascading** randomization: first the output layer, then the top two, and so on), and recompute the map. if the map barely changes, it isn't telling you about the learned weights.
- **data randomization.** train the same architecture on randomly permuted labels (it can memorise them) and compare its maps with the real model's. a map that looks the same for a model that learned nothing about the task is explaining something else.

"barely changes" needs a number. Adebayo et al. used the **Spearman rank correlation** between the absolute values of the two maps (and SSIM, a perceptual similarity). ranks, because a map is read by which pixels are brightest, not by its scale; absolute values, because sign conventions differ between methods. their finding, which the lab reproduces: plain gradients change a lot under randomization, while methods that multiply the gradient by the input, like gradient × input and IG with a black baseline, keep looking like the input. the factor $x$ alone draws the digit, whatever the gradient is.

that doesn't make IG wrong (its axioms still hold for whatever model it explains), but it means that "the map looks like the object" is no evidence that the map is a good explanation. always compare against a randomized model, and read the numbers, not the pictures.

> **spec.** `rank_correlation(a, b)`: Spearman correlation of two arrays of equal size (flattened; ties get average ranks), a float; `ValueError` for different sizes. `randomize_layers(model, n, *, generator=None)`: a deep copy of `model` whose LAST `n` parameterised modules (modules with parameters of their own, in `model.modules()` order) are re-initialised with their `reset_parameters()`, seeded from `generator` so that the same seed gives the same weights; the original and torch's global RNG state are left unchanged; `ValueError` if `n` is negative or larger than the number of parameterised modules. `cascading_randomization(model, attribute, x, target, *, generator=None)`: `attribute(model, x, target)` returns a map; returns an ndarray `(L,)`, entry $k-1$ = `rank_correlation` of $|$map$|$ of `model` with $|$map$|$ of `randomize_layers(model, k)`, for $k = 1, \dots, L$ (L = number of parameterised modules).

> **🔮 predict:** you randomize every layer of LeNet. will the rank correlation of IG maps with the trained model's maps be near 0, near 0.5, or near 1?

> **▶ build: stage 4.** implement `rank_correlation`, `randomize_layers` and `cascading_randomization`.

## worked example

take the one-logit model $f(x) = x_1^2 + x_1 x_2$ at $x = (2, 1)$, baseline $x' = (0, 0)$. its gradient is $(2x_1 + x_2,\; x_1)$.

**saliency.** at $x$: $|(5, 2)| = (5, 2)$. the gradient says $x_1$ matters 2.5 times as much as $x_2$.

**integrated gradients.** on the path $\alpha x = (2\alpha, \alpha)$ the gradient is $(5\alpha, 2\alpha)$; its average over $\alpha \in [0, 1]$ is $(2.5, 1)$. multiply by $x - x' = (2, 1)$: $\mathrm{IG} = (5, 1)$, which sums to $6 = f(x) - f(0)$. the $x_1^2 = 4$ term goes entirely to $x_1$; the interaction $x_1 x_2 = 2$ is split equally, 1 each. the gradient is linear in $\alpha$, so the trapezoid rule is exact here even with `n_steps=1`: $\tfrac12((0, 0) + (5, 2)) = (2.5, 1)$. a left Riemann sum with 2 steps would use $\alpha = 0, \tfrac12$ and get $(2.5, 0.5)$, which sums to 3, only half the output.

**selectivity.** in the lab, a probe on LeNet's fc2 layer predicts "is the digit even?" with test accuracy 0.95, and the control task (a random binary label per digit class) with 0.94 on average: selectivity 0.01. the probe found the classes, not parity.

your `saliency` and `integrated_gradients` must reproduce $(5, 2)$ and $(5, 1)$.

## on paper

1. **completeness.** let $\gamma(\alpha) = x' + \alpha(x - x')$. show that $\sum_i \mathrm{IG}_i(x) = f(x) - f(x')$ for any differentiable $f$.

<details><summary>solution</summary>

$\frac{d}{d\alpha} f(\gamma(\alpha)) = \sum_i \frac{\partial f}{\partial x_i}(\gamma(\alpha))\, (x_i - x'_i)$ by the chain rule. integrate from 0 to 1: the left side gives $f(x) - f(x')$, the right side is $\sum_i (x_i - x'_i) \int_0^1 \partial_i f(\gamma(\alpha))\, d\alpha = \sum_i \mathrm{IG}_i$. (for ReLU networks $f$ is differentiable almost everywhere along the path and the same argument holds.)

</details>

2. **saturation.** $f(x) = 1 - \mathrm{ReLU}(1 - x)$ (it rises from 0 at $x = 0$ to 1 at $x = 1$ and stays there). compute the gradient and IG (baseline 0) at $x = 2$.

<details><summary>solution</summary>

the gradient at $x = 2$ is 0: $f$ is flat there, so saliency says $x$ doesn't matter. along the path $2\alpha$, $f' = 1$ for $2\alpha < 1$ and 0 after, so $\int_0^1 f'(2\alpha)\, d\alpha = \tfrac12$ and $\mathrm{IG} = 2 \cdot \tfrac12 = 1 = f(2) - f(0)$. IG gives the feature full credit for the change it caused.

</details>

3. **the trapezoid error for a ReLU path.** the gradient along the path is $g(\alpha) = 1$ for $\alpha < 0.3$ and 0 after (one unit switches off). what does the $n$-step trapezoid rule give for $\int_0^1 g$, for $n = 2$ and $n = 10$? how does the error scale?

<details><summary>solution</summary>

$n = 2$: points 0, 0.5, 1 with values 1, 0, 0 and weights $\tfrac14, \tfrac12, \tfrac14$: $\tfrac14$, error 0.05. $n = 10$: values 1 at 0, 0.1, 0.2 and 0 after: $\tfrac1{20} + \tfrac1{10} + \tfrac1{10} = 0.25$, error 0.05 again; but at $n = 7$ the error is $|0.3 - (1/14 + 2/7)| = 0.057$ and at $n = 100$ it's $0.005$. a jump costs up to $\tfrac{1}{2n}$ per unit of jump: error $O(1/n)$, not the $O(1/n^2)$ of smooth integrands. a network with many units switching along the path has many small jumps.

</details>

4. **why memorisation solves the control task.** features are one-hot vectors of $V$ word types. show that a linear probe can reach 100% training accuracy on *any* labelling that is a function of the type, and say why a held-out test set doesn't save you.

<details><summary>solution</summary>

with weight matrix $W \in \mathbb{R}^{k \times V}$, set $W_{c, v} = 1$ if type $v$ has label $c$ and 0 otherwise: the logits of a one-hot $e_v$ are column $v$, which picks the right label. a test set of new *tokens* of the same types has the same features, so test accuracy is 100% too. only types never seen in training would expose the memorisation, and a control task over the same types makes the gap measurable instead.

</details>

## experiment

the lab trains LeNet on MNIST and turns your tools on it: how many trapezoid steps IG needs before its attributions add up, what saliency, gradient × input and IG highlight, what survives the cascading randomization of every layer (and whether that survival is about the model at all), and what a linear probe for parity is really detecting in each layer once a control task is run beside it.

## check your understanding

1. why explain a logit rather than the softmax probability?

<details><summary>answer</summary>

a confident softmax output is saturated: its gradient is tiny in every direction, so the map is mostly numerical noise. the probability also depends on every other logit, so its gradient mixes "evidence for this class" with "evidence against the others".

</details>

2. IG with a black baseline gives exactly zero attribution to every black pixel. is that a bug?

<details><summary>answer</summary>

it's the definition: IG explains $f(x) - f(x')$, and pixels equal to their baseline didn't change. if black pixels matter (a missing stroke that distinguishes a 7 from a 9), a black baseline can't show it. pick the baseline that encodes the question you're asking.

</details>

3. your IG completeness gap is 30% at `n_steps=20`. what are the likely causes?

<details><summary>answer</summary>

too few steps for a network with many ReLU switches along the path (increase $n$ until the gap shrinks); a bug in the weights (Riemann instead of trapezoid, or dividing by $n+1$); dropout left on; or attributing a different output than the one you're comparing with (a probability instead of the logit).

</details>

4. a probe on layer 8 decodes sentiment with 90% accuracy and its control task with 88%. what can you conclude?

<details><summary>answer</summary>

very little about sentiment: the layer lets a linear probe memorise per-type labels almost as well, so the 90% may come from word identity. a more selective probe (lower capacity, more regularization), a control on types not seen in training, or an intervention would be needed.

</details>

5. why are the sanity checks phrased as "the map should change", not "the map should be correct"?

<details><summary>answer</summary>

there's no ground truth for what a network "really" uses on natural images. changing the model is a test that needs no ground truth: whatever a good explanation shows, it must depend on the model it explains. it's a necessary condition, not a sufficient one.

</details>

6. why does gradient × input survive randomization even when the gradient doesn't?

<details><summary>answer</summary>

$|x_i g_i|$ is zero wherever $x_i = 0$ and proportional to $|x_i|$ elsewhere, times a random-looking $|g_i|$. its rank structure is dominated by the input's own support and brightness, which no randomization of the model touches.

</details>

7. a layer-randomized model's IG still satisfies completeness. why doesn't that rescue IG in the sanity check?

<details><summary>answer</summary>

completeness is a property of the method relative to *whatever* model it's given: the random model's attributions add up to the random model's $f(x) - f(x')$. it says nothing about whether the picture tells a human which features the *trained* model relies on.

</details>

## common bugs

- **summing the wrong outputs.** `model(x)[:, target].sum()` with a `(B,)` target indexes a `(B, B)` block and mixes every row's class into every row. use `gather(1, target[:, None])`.
- **training mode.** dropout makes two calls disagree; batch norm in train mode makes one image's map depend on its batch. call `model.eval()`, and restore the mode afterwards.
- **polluting parameter gradients.** calling `.backward()` in an attribution leaves `.grad` on every weight; a later optimizer step uses it. use `torch.autograd.grad(out, x)`.
- **off-by-one in the path.** `torch.linspace(0, 1, n_steps)` gives $n - 1$ intervals; a left Riemann sum misses the end point. check completeness on a model where you know the answer.
- **two control tables.** drawing control labels separately for train and test turns the control task into noise and inflates selectivity.
- **correlating signed maps.** methods disagree on sign conventions; Adebayo et al. correlate absolute values.

## going deeper

- Simonyan, Vedaldi & Zisserman (2014), "Deep inside convolutional networks: visualising image classification models and saliency maps". [arXiv:1312.6034](https://arxiv.org/abs/1312.6034)
- Sundararajan, Taly & Yan (2017), "Axiomatic attribution for deep networks", *ICML* (integrated gradients). [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)
- Smilkov et al. (2017), "SmoothGrad: removing noise by adding noise". [arXiv:1706.03825](https://arxiv.org/abs/1706.03825)
- Adebayo et al. (2018), "Sanity checks for saliency maps", *NeurIPS*. [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)
- Kindermans et al. (2019), "The (un)reliability of saliency methods" (input-invariance). [arXiv:1711.00867](https://arxiv.org/abs/1711.00867)
- Lapuschkin et al. (2019), "Unmasking Clever Hans predictors and assessing what machines really learn", *Nature Communications*. [arXiv:1902.10178](https://arxiv.org/abs/1902.10178)
- Alain & Bengio (2016), "Understanding intermediate layers using linear classifier probes". [arXiv:1610.01644](https://arxiv.org/abs/1610.01644)
- Hewitt & Liang (2019), "Designing and interpreting probes with control tasks", *EMNLP*. [arXiv:1909.03368](https://arxiv.org/abs/1909.03368)
- Belinkov (2022), "Probing classifiers: promises, shortcomings, and advances", *Computational Linguistics*. [arXiv:2102.12452](https://arxiv.org/abs/2102.12452)

## connections

- ← builds on: autograd (ch 14–15), LeNet (ch 16 · 03), linear probes on frozen features (ch 25 · 01), model-agnostic attribution with permutation importance and KernelSHAP (ch 10, `mlfs.inspection`), the trapezoid rule (ch 01).
- → used in: topic 02 replaces "which input features" with "which internal components" and turns probes into interventions; topic 04 uses the same input gradient to *attack* the model (FGSM is a signed saliency map); the capstone reads a trained transformer's embeddings directly instead of probing them.
