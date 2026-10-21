# privacy & fairness

## why this matters

a model is a summary of its training data, and summaries leak. Shokri et al. (2017) showed that from a model's outputs alone an attacker can often tell whether a given record was in its training set (**membership inference**): for a hospital's model, membership itself can reveal a diagnosis. Carlini et al. (2021) extracted verbatim training text from GPT-2, including names, phone numbers and email addresses that appeared in its training data only a handful of times. "we removed the names" isn't protection either: the Netflix prize ratings were re-identified by matching them against public IMDb reviews.

the strongest answer we have is **differential privacy** (DP): a mathematical guarantee that the trained model would be almost equally likely whether or not any single person's data was used. its deep-learning form, **DP-SGD** (Abadi et al., 2016), bounds each example's influence on every gradient step by clipping, hides what's left under calibrated Gaussian noise, and uses an **accountant** to add up the privacy cost over thousands of steps.

the second half of the topic is about a different harm from the same source. a model trained on data from an unequal world can work worse, or decide differently, for some groups: commercial face-analysis systems misclassified darker-skinned women up to 35% of the time and lighter-skinned men under 1% (Buolamwini & Gebru, 2018), and a recidivism score used by US courts was found to make different kinds of errors for Black and white defendants. you'll build the standard group-fairness metrics, a post-processing fix, and then run into the result that makes fairness a choice rather than a fix: when base rates differ between groups, the common criteria can't all hold at once.

## warm-up: predict first

1. a model is trained on 1,000 photos for 30 epochs until it's 100% accurate on them and 33% on new photos. how well can an attacker who sees only its loss on a photo tell training photos from new ones?
2. DP-SGD adds noise to each gradient step. to protect individuals, why isn't noise alone enough: why also clip each example's gradient?
3. the privacy cost of 10,000 noisy steps, each touching a random 1% of the data: more or less than 100 times the cost of 100 steps?
4. a credit score is equally accurate for two groups, and one threshold is applied to everyone. will both groups be approved at the same rate?

## prerequisites check

- [ ] you know the Gaussian distribution, sums of independent Gaussians, and log-sum-exp (→ ch 04, ch 00 · 03)
- [ ] you know ROC AUC and true/false positive rates (→ ch 07 · 03, `mlfs.metrics`)
- [ ] you know how an optimizer reads `.grad` and what `optimizer.step()` does (→ ch 15 · 07)
- [ ] you know generalisation gaps and overfitting (→ ch 07 · 01, ch 13)
- [ ] you know `torch.func` exists; this topic is where you use `vmap` and `grad` (→ the PyTorch docs on `torch.func`)

## core ideas

### idea 1: one gradient per example

ordinary training only ever needs the *mean* gradient over a batch, and frameworks compute exactly that, never the individual terms. differential privacy needs the individual terms: to bound one person's influence you have to see and clip that person's gradient before it's mixed with anyone else's. the naive way is a loop of $B$ backward passes. the fast way treats "the gradient of the loss on one example, as a function of the parameters" as a function and **vectorises** it over the batch: `torch.func.grad` turns a scalar function into its gradient function, `torch.func.vmap` maps that over the batch dimension, and `torch.func.functional_call` runs the model with the parameters passed in as a dictionary. the result is a dictionary of tensors with an extra leading batch dimension, at the cost of about one batched backward pass (and $B$ times the gradient memory).

inside `vmap` each example is a single row without a batch dimension, so add one back (`x.unsqueeze(0)`) before calling the model and the loss.

> **spec.** `per_sample_gradients(model, loss_fn, x, y)`: returns `{name: (B, *param.shape)}` for every parameter with `requires_grad`, where row $i$ is the gradient of `loss_fn(model(x[i:i+1]), y[i:i+1])`; computed with `torch.func` (`functional_call`, `grad`, `vmap`), without touching any `.grad`.

> **▶ build: stage 1.** implement `per_sample_gradients`.

### idea 2: differential privacy and DP-SGD

a randomised algorithm $M$ is **$(\epsilon, \delta)$-differentially private** if for any two datasets $D, D'$ that differ in one person's record and any set of outcomes $S$,

$$P(M(D) \in S) \le e^\epsilon P(M(D') \in S) + \delta.$$

whatever an observer does with the output, their conclusions about whether you were in the data can shift by at most a factor of $e^\epsilon$ (plus a small slack $\delta$, which should be much smaller than $1/n$). two properties make it practical: anything computed from a DP output is still DP (**post-processing**), and running DP steps one after another adds up their costs (**composition**).

the basic tool is the **Gaussian mechanism**: to release a sum $f(D) = \sum_i v_i$ privately, add noise $\mathcal{N}(0, \sigma^2 C^2 I)$, where $C$ bounds how much one record can change $f$ (its **sensitivity**). DP-SGD makes a gradient step into such a sum: compute each example's gradient $g_i$, **clip** it to norm at most $C$ (`max_grad_norm`), $\bar g_i = g_i \cdot \min(1, C / \|g_i\|)$, so that adding or removing one example changes the sum by at most $C$; add Gaussian noise with standard deviation $\sigma C$ ($\sigma$ is the `noise_multiplier`); divide by the batch size; and step. the norm is the norm of the *whole* gradient, all parameters together: that's what bounds one example's influence. clipping the batch's mean gradient instead bounds nothing (one example can still dominate the mean before clipping).

noise alone wouldn't be enough (warm-up 2): without clipping a single outlier's gradient can be arbitrarily large and stand out through any fixed amount of noise. clipping alone isn't private either: a deterministic function of the data can still reveal it exactly.

> **spec.** `dp_sgd_step(model, loss_fn, optimizer, x, y, *, max_grad_norm=1.0, noise_multiplier=1.0, generator=None)`: with $C$ = `max_grad_norm`, $\sigma$ = `noise_multiplier` and $B$ = `len(x)`: per-sample gradients, each scaled by $\min(1, C / \|g_i\|)$ with $\|g_i\|$ the norm over all trainable parameters; summed; plus noise $\mathcal{N}(0, \sigma^2 C^2)$ per coordinate drawn from `generator`; divided by $B$; written into each trainable parameter's `.grad`; then `optimizer.step()`. returns the batch's mean loss (before the update) as a float.

> **▶ build: stage 2.** implement `dp_sgd_step`.

### idea 3: the privacy accountant

one DP-SGD step is a Gaussian mechanism applied to a random batch, and training takes thousands of them. naive composition (add up the $\epsilon$s) gives hopelessly large totals. two ideas make the numbers reasonable. **amplification by subsampling**: if each example is in the batch only with probability $q$, a step reveals much less about it. and **tight composition**, through **Rényi differential privacy** (Mironov, 2017): the privacy loss of a mechanism at order $\alpha > 1$ is measured by the Rényi divergence $D_\alpha$ between its output distributions on neighbouring datasets, written $\epsilon(\alpha)$, and under composition these simply add.

for one Poisson-subsampled Gaussian step with sensitivity 1 and noise multiplier $\sigma$, at integer orders $\alpha \ge 2$ (Mironov, Talwar & Zhang, 2019),

$$\epsilon(\alpha) = \frac{1}{\alpha - 1}\log \sum_{k=0}^{\alpha} \binom{\alpha}{k}(1-q)^{\alpha-k} q^k \exp\Big(\frac{k^2 - k}{2\sigma^2}\Big),$$

which for $q = 1$ reduces to the Gaussian mechanism's $\alpha / (2\sigma^2)$. after $T$ steps the RDP is $T\epsilon(\alpha)$, and it converts to $(\epsilon, \delta)$-DP by

$$\epsilon = \min_\alpha \Big[T\epsilon(\alpha) + \frac{\log(1/\delta)}{\alpha - 1}\Big].$$

the terms in the sum grow like $e^{k^2/(2\sigma^2)}$, so compute it in log space (log-sum-exp). Abadi et al.'s published example, $q = 0.01$, $\sigma = 4$, $T = 10{,}000$, $\delta = 10^{-5}$, gave $\epsilon \approx 1.26$ with their moments accountant; this RDP accountant reproduces it, $\epsilon = 1.2586$, because with integer orders it computes the same bound (the moments accountant's $\lambda$ is $\alpha - 1$). strong composition, the best general-purpose bound, gives about 9.3. (the analysis assumes each batch is a Poisson sample of the data; code that shuffles and takes fixed-size batches is common, but the accountant's guarantee doesn't strictly apply to it.)

> **spec.** `rdp_subsampled_gaussian(q, noise_multiplier, orders)`: an ndarray of $\epsilon(\alpha)$ for the integer `orders` ($\ge 2$, `ValueError` otherwise), with $q = 0$ giving 0, $\sigma = 0$ giving `inf`, and $q = 1$ the closed form. `dp_sgd_epsilon(q, noise_multiplier, steps, delta, *, orders=None)`: the minimum over `orders` (None means the integers 2 to 255) of `steps` $\times\, \epsilon(\alpha) + \log(1/\delta)/(\alpha - 1)$, a float.

> **🔮 predict:** with $q = 0.02$, how does $\epsilon$ grow from 100 to 1,000 steps: by 10×, by $\sqrt{10}$×, or something else?

> **▶ build: stage 3.** implement `rdp_subsampled_gaussian` and `dp_sgd_epsilon`.

### idea 4: membership inference

DP is a worst-case bound; an attack measures what actually leaks. the simplest **membership-inference attack** (Yeom et al., 2018) uses the fact that models fit their training data better than new data: flag an example as a member if the model's loss on it is low. with the generalisation gap as its only source of signal it's weak against models that generalise well, and devastating against models that memorise.

how to score an attack matters. the AUROC (the probability that a random member gets a lower loss than a random non-member) summarises average-case distinguishability. but privacy is about individuals: an attack that confidently identifies 5% of members while being right about nobody else is a serious breach with a mediocre AUROC. Carlini et al. (2022) argued for reporting the **true positive rate at a low false positive rate**, such as 1%: of the non-members, at most 1% may be wrongly accused; how many members are caught? DP gives the attacker a hard limit at every threshold, $\text{TPR} \le e^\epsilon \cdot \text{FPR} + \delta$ (Kairouz et al., 2015, on paper 3), so measuring TPR at low FPR is also a way to audit a DP claim.

> **spec.** `loss_threshold_attack(member_losses, nonmember_losses)`: the ROC AUC of scoring members (positives) by *lower* loss, ties counting 1/2, a float. `tpr_at_fpr(member_losses, nonmember_losses, fpr=0.01)`: with the non-member losses sorted ascending, $s_{(1)} \le \dots \le s_{(n)}$, and $m = \lfloor \text{fpr} \cdot n \rfloor$, the threshold is $\tau = s_{(m+1)}$ ($+\infty$ if $m \ge n$), and the result is the fraction of members with loss strictly below $\tau$ (so at most $m$ non-members are flagged).

> **▶ build: stage 4.** implement `loss_threshold_attack` and `tpr_at_fpr`.

### idea 5: group fairness, and why you can't have it all

for a binary decision $\hat y$ (approve a loan), a sensitive attribute $g$ (a group) and the true outcome $y$ (would repay), three common criteria:

- **demographic parity**: equal approval rates, $P(\hat y = 1 \mid g)$ the same for every group. it ignores $y$ entirely.
- **equalized odds** (Hardt, Price & Srebro, 2016): equal true positive rates *and* equal false positive rates across groups: people who would repay are approved equally often in every group, and so are people who wouldn't.
- **equal opportunity**: equal true positive rates only.

each is measured by its largest gap across groups. Hardt et al. showed equalized odds or equal opportunity can be reached by **post-processing** a score with group-specific thresholds: for equal opportunity at a target TPR, each group's threshold is the largest $t$ whose approved set ($\text{score} \ge t$) contains that fraction of the group's positives.

the uncomfortable part: when the groups' base rates $P(y = 1 \mid g)$ differ, an equally accurate score with one threshold violates demographic parity (warm-up 4), and enforcing demographic parity forces unequal error rates. Kleinberg, Mullainathan & Raghavan (2017) and Chouldechova (2017) proved that calibration within groups and equal false positive and false negative rates can't all hold unless base rates are equal or prediction is perfect. so choosing a fairness criterion is choosing *which* disparity to accept, a decision about values that belongs to the people affected, not to the metric.

> **spec.** `demographic_parity_difference(y_pred, groups)`: max − min over groups of $P(\hat y = 1 \mid g)$. `equalized_odds_difference(y_true, y_pred, groups)`: the larger of the TPR gap and the FPR gap (each max − min over groups); `ValueError` if a group has no positives or no negatives. `equal_opportunity_thresholds(scores, y_true, groups, target_tpr)`: `{group: threshold}`, each the $\lceil \text{target} \cdot n_+\rceil$-th largest score among the group's $n_+$ positives. groups may be any hashable labels.

> **🔮 predict:** you equalise approval rates between a group with 60% and a group with 40% repayers. what happens to the false positive rates?

> **▶ build: stage 5.** implement `demographic_parity_difference`, `equalized_odds_difference` and `equal_opportunity_thresholds`.

## worked example

**clipping.** a batch of two examples with gradients $g_1 = (3, 4)$ (norm 5) and $g_2 = (0.3, 0.4)$ (norm 0.5), $C = 1$. per-example clipping scales $g_1$ by $1/5$ to $(0.6, 0.8)$ and leaves $g_2$ alone; the sum is $(0.9, 1.2)$, plus noise of standard deviation $\sigma C$ per coordinate, divided by 2: $(0.45, 0.6) + \text{noise}/2$. clipping the *mean* $(1.65, 2.2)$ (norm 2.75) instead would give $(0.6, 0.8)$: almost all of it is $g_1$, the outlier the clipping was meant to contain.

**accounting.** a single step on the whole dataset ($q = 1$) with $\sigma = 2$: $\epsilon(\alpha) = \alpha/8$. with $\delta = 10^{-5}$, $\log(1/\delta) = 11.51$: at $\alpha = 10$, $1.25 + 1.28 = 2.53$; at $\alpha = 11$, $1.375 + 1.151 = 2.526$; at $\alpha = 12$, $1.5 + 1.047 = 2.547$. the minimum is $\epsilon = 2.53$ at $\alpha = 11$.

**TPR at 1% FPR.** 100 non-members with losses $0.01, 0.02, \dots, 1.00$ and members with losses $0.005, 0.015, 0.025, 0.5, 0.9$. at most $\lfloor 0.01 \cdot 100 \rfloor = 1$ non-member may be flagged, so $\tau = s_{(2)} = 0.02$: two of five members have loss below it, TPR 0.4.

**fairness.** groups $a$ (4 people) and $b$ (6), predictions $1, 1, 0, 0$ and $1, 0, 0, 0, 0, 0$: approval rates $\tfrac12$ and $\tfrac16$, demographic parity difference $\tfrac13$. with true labels $1, 0, 1, 0$ and $1, 1, 0, 0, 0, 1$: group $a$ has TPR $\tfrac12$ and FPR $\tfrac12$, group $b$ TPR $\tfrac13$ and FPR 0; equalized odds difference $\max(\tfrac16, \tfrac12) = \tfrac12$.

your functions must reproduce 2.53, 0.4, $\tfrac13$ and $\tfrac12$.

## on paper

1. **sensitivity of the clipped sum.** show that adding or removing one example changes $\sum_i \bar g_i$ by a vector of norm at most $C$, and explain why the Gaussian mechanism's noise is scaled by $C$.

<details><summary>solution</summary>

the sums over $D$ and $D' = D \cup \{x\}$ differ by exactly $\bar g_x$, and $\|\bar g_x\| = \|g_x\| \min(1, C/\|g_x\|) = \min(\|g_x\|, C) \le C$. the privacy of the Gaussian mechanism depends on the ratio of the noise scale to the sensitivity, so noise $\sigma C$ gives the same guarantee for any $C$; that's why the guarantee is stated in terms of the noise *multiplier* $\sigma$.

</details>

2. **RDP of the Gaussian mechanism.** for $\mathcal{N}(0, \sigma^2)$ versus $\mathcal{N}(1, \sigma^2)$, show $D_\alpha = \alpha / (2\sigma^2)$, using $D_\alpha(P\|Q) = \frac{1}{\alpha-1}\log \mathbb{E}_Q[(p/q)^\alpha]$.

<details><summary>solution</summary>

$p(z)/q(z) = \exp\big((2z - 1)/(2\sigma^2)\big)$ for $P = \mathcal{N}(1, \sigma^2)$, $Q = \mathcal{N}(0, \sigma^2)$. $\mathbb{E}_Q[e^{\alpha(2z-1)/(2\sigma^2)}] = e^{-\alpha/(2\sigma^2)} \mathbb{E}[e^{\alpha z/\sigma^2}] = e^{-\alpha/(2\sigma^2)} e^{\alpha^2/(2\sigma^2)} = e^{\alpha(\alpha-1)/(2\sigma^2)}$, using the Gaussian moment generating function. dividing the log by $\alpha - 1$ gives $\alpha/(2\sigma^2)$. the subsampled formula is the same expectation with $p$ replaced by the mixture $(1-q)q + q p$, expanded binomially.

</details>

3. **DP limits every attack.** an attacker decides "member" on a set $S$ of outputs. show that its true positive rate (the probability of $S$ when the target is in the data) and false positive rate (when it isn't) satisfy $\text{TPR} \le e^\epsilon \text{FPR} + \delta$.

<details><summary>solution</summary>

TPR $= P(M(D) \in S)$ with the target in $D$, and FPR $= P(M(D') \in S)$ for the neighbouring dataset without it. the DP inequality applied to the set $S$ is exactly the claim. at FPR = 1%, $\epsilon = 1.4$ allows TPR up to 4%; $\epsilon = 6$ allows anything. so small $\epsilon$ is a meaningful promise and large $\epsilon$ (common in practice) is mostly a promise about the noise, not about attacks.

</details>

4. **the impossibility, in one line.** for a group with base rate $p$, a classifier with positive predictive value PPV, false negative rate FNR and false positive rate FPR satisfies $\text{FPR} = \frac{p}{1-p}\cdot\frac{1 - \text{PPV}}{\text{PPV}}\cdot(1 - \text{FNR})$ (Chouldechova, 2017). use it to show that equal PPV and equal FNR across two groups with different base rates force different FPRs.

<details><summary>solution</summary>

with PPV and FNR the same in both groups, the FPR is proportional to $p/(1-p)$, which differs when the base rates differ. so a score that's equally "calibrated" (same PPV) and catches positives equally well in both groups must wrongly flag negatives at different rates. something has to give, and choosing what is a value judgement.

</details>

## experiment

the lab trains MNIST classifiers with your DP-SGD at several noise levels and prices their accuracy in $\epsilon$; attacks memorising and private CIFAR-10 models with your membership-inference attack and compares the attack with the DP bound; and applies three approval policies to a synthetic lending dataset to see which fairness criteria each one satisfies and which it breaks.

## check your understanding

1. why does DP-SGD use the norm of the *whole* per-example gradient, not one norm per parameter tensor?

<details><summary>answer</summary>

the guarantee needs a bound on how much one example can change the entire update. clipping each tensor to $C$ separately bounds the total by $C\sqrt{L}$ for $L$ tensors, so the noise would have to be calibrated to that larger sensitivity. clipping per tensor is possible (with adjusted noise) but it's a different mechanism.

</details>

2. why should $\delta$ be much smaller than $1/n$?

<details><summary>answer</summary>

a mechanism that publishes one random record in full is $(0, 1/n)$-DP. $\delta$ is the probability of a catastrophic failure; it has to be small compared with the chance of picking out any given individual.

</details>

3. the accountant reports $\epsilon = 2.8$ for your training run. you then tune the learning rate over 20 runs and keep the best. is the final model 2.8-DP?

<details><summary>answer</summary>

not on its own: the choice of run depends on the data too, so the tuning has a privacy cost. composing 20 runs naively costs about 20 times as much; there are better private-selection methods, and in practice hyperparameters are often tuned on public data.

</details>

4. a loss-threshold attack has AUROC 0.6 against your private model. what does that tell you, and what doesn't it?

<details><summary>answer</summary>

that this attack, on average, distinguishes members only slightly. it doesn't tell you about the worst-case individuals (look at TPR at low FPR), or about stronger attacks (per-example calibrated attacks like LiRA, or attacks with shadow models). attacks give lower bounds on leakage; DP gives the upper bound.

</details>

5. a classifier satisfies demographic parity. is it fair?

<details><summary>answer</summary>

it approves groups at equal rates, which can coexist with very unequal error rates (approving unqualified applicants in one group and rejecting qualified ones in another), and can be achieved by approving at random within a group. each criterion captures one notion of fairness and is blind to the others.

</details>

6. why might post-processing with group thresholds be legally or ethically problematic, even if it equalises error rates?

<details><summary>answer</summary>

it uses the sensitive attribute explicitly at decision time, which some laws forbid (disparate treatment), and it treats two people with the same score differently because of their group. alternatives change the training data or objective instead, with their own trade-offs. which is acceptable depends on the domain and the law, not on the mathematics.

</details>

## common bugs

- **batch clipping.** clipping the mean gradient (or calling `clip_grad_norm_` after a normal backward pass) bounds nothing about individuals.
- **noise after averaging at the wrong scale.** the noise has standard deviation $\sigma C$ on the *sum*; adding $\sigma C$ to the mean is $B$ times too much, and adding $\sigma$ without the $C$ is wrong whenever $C \ne 1$.
- **forgetting the batch dimension inside `vmap`.** the model sees a single example without a batch axis; `unsqueeze(0)` it.
- **overflow in the accountant.** $\exp(k^2/(2\sigma^2))$ overflows for large orders; sum in log space.
- **the wrong orientation of the attack.** members have *lower* loss; scoring by loss directly gives AUROC $1 - \text{AUROC}$.
- **rounding the FPR budget up.** at most $\lfloor \text{fpr} \cdot n \rfloor$ non-members may be flagged; $\lceil\cdot\rceil$ exceeds the FPR you claim.

## going deeper

- Dwork & Roth (2014), *The Algorithmic Foundations of Differential Privacy*, Foundations and Trends in TCS. [doi:10.1561/0400000042](https://doi.org/10.1561/0400000042)
- Abadi et al. (2016), "Deep learning with differential privacy", *CCS* (DP-SGD, the moments accountant). [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)
- Mironov (2017), "Rényi differential privacy", *CSF*. [arXiv:1702.07476](https://arxiv.org/abs/1702.07476)
- Mironov, Talwar & Zhang (2019), "Rényi differential privacy of the sampled Gaussian mechanism". [arXiv:1908.10530](https://arxiv.org/abs/1908.10530)
- Shokri et al. (2017), "Membership inference attacks against machine learning models", *IEEE S&P*. [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)
- Yeom et al. (2018), "Privacy risk in machine learning: analyzing the connection to overfitting", *CSF*. [arXiv:1709.01604](https://arxiv.org/abs/1709.01604)
- Carlini et al. (2022), "Membership inference attacks from first principles", *IEEE S&P* (LiRA, TPR at low FPR). [arXiv:2112.03570](https://arxiv.org/abs/2112.03570)
- Carlini et al. (2021), "Extracting training data from large language models", *USENIX Security*. [arXiv:2012.07805](https://arxiv.org/abs/2012.07805)
- Hardt, Price & Srebro (2016), "Equality of opportunity in supervised learning", *NeurIPS*. [arXiv:1610.02413](https://arxiv.org/abs/1610.02413)
- Kleinberg, Mullainathan & Raghavan (2017), "Inherent trade-offs in the fair determination of risk scores", *ITCS*. [arXiv:1609.05807](https://arxiv.org/abs/1609.05807)
- Chouldechova (2017), "Fair prediction with disparate impact: a study of bias in recidivism prediction instruments", *Big Data*. [arXiv:1703.00056](https://arxiv.org/abs/1703.00056)
- Buolamwini & Gebru (2018), "Gender shades: intersectional accuracy disparities in commercial gender classification", *FAT\**. [proceedings.mlr.press/v81/buolamwini18a](https://proceedings.mlr.press/v81/buolamwini18a.html)

## connections

- ← builds on: Gaussians and the moment generating function (ch 04), Rényi and KL divergences (ch 05 · 05), ROC curves (ch 07 · 03), generalisation gaps (ch 13), the training loop (ch 15).
- → used in: chapter 29's federated-learning elective applies the same clip-and-noise recipe to whole client updates; topic 07's evals face the same question as fairness metrics: which number should a model be held to, and who decides; chapter 30's projects can train with `dp_sgd_step` and audit with `tpr_at_fpr`.
