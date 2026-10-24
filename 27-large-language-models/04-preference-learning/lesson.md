# preference learning

## why this matters

after supervised fine-tuning (topic 03) a model imitates its demonstrations, including their flaws, and it can't be better than the people who wrote them. but people are much better at *judging* answers than at writing perfect ones: given two summaries, most of us can say which is better even if we couldn't have written either. **preference learning** turns those judgements into a training signal. InstructGPT (Ouyang et al., 2022) collected rankings of model outputs, trained a **reward model** to predict them, and optimized the language model against it with PPO, the **RLHF** recipe behind the first ChatGPT. labelers preferred its 1.3B-parameter model to the 175B GPT-3 base.

the catch is in the word *predict*. the reward model is a proxy for what people want, learned from a finite number of comparisons, and an optimizer pushing hard on a proxy finds its mistakes. Gao, Schulman & Hilton (2022) measured this precisely: as a policy is optimized against a proxy reward model, the proxy score keeps rising while the true ("gold") score rises, peaks and falls: Goodhart's law with a curve. two defenses run through everything in this topic: a **KL penalty** that keeps the policy near the model it started from, and **DPO** (Rafailov et al., 2023), which skips the explicit reward model and the RL loop altogether by solving for the optimal policy in closed form.

in this topic you build the pieces: sequence log-probabilities, a reward model and its Bradley–Terry loss, the DPO loss and a DPO update, and the KL-shaped rewards of RLHF. the lab then reproduces reward over-optimization end to end on a small model.

## warm-up: predict first

1. a reward model gives answer A a score of 2.0 and answer B a score of 0.5. under the Bradley–Terry model, how likely is a labeler to prefer A?
2. you pick the best of 64 samples according to a reward model. how far (in KL) has your "policy" moved from the original model? more or less than 1 nat?
3. RLHF adds a penalty $-\beta \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)}$ to the reward. what happens if $\beta = 0$?
4. DPO trains on preference pairs with a supervised-looking loss and no reward model. can it still over-optimize?

## prerequisites check

- [ ] you can compute log-probabilities of tokens from logits with pre-shifted targets (→ ch 18 · 04, topic 03)
- [ ] you know the logistic function and binary cross-entropy (→ ch 08)
- [ ] you know the PPO clipped objective and advantages (→ ch 24 · 03–04, `mlfs.deeprl.ppo_clip_loss`)
- [ ] you know KL divergence and that $\mathbb{E}_{y \sim \pi}[\log \pi(y) - \log \pi_\text{ref}(y)] = \mathrm{KL}(\pi \| \pi_\text{ref})$ (→ ch 05 · 05)

## core ideas

### idea 1: how likely is a response?

everything in preference learning compares how likely two models find the same response. for a prompt $x$ and response $y = (y_1, \dots, y_m)$, an autoregressive model gives

$$\log \pi(y \mid x) = \sum_{t=1}^{m} \log \pi(y_t \mid x, y_{<t}).$$

with the batch layout from topic 03 (prompt and response concatenated, targets pre-shifted, $-100$ on the prompt), that's one forward pass, a `log_softmax`, a `gather` of each target's log-probability, and a sum over the positions whose target isn't $-100$. keep the per-token version too: RLHF needs per-token log-probabilities for its per-token KL penalty and PPO ratios. compute it in float32 (the log-softmax of bf16 logits loses precision in exactly the small differences DPO cares about) and keep the graph: DPO backpropagates through it.

> **spec.** `token_logprobs(model, input_ids, targets)`: `model` follows the causal-LM protocol; returns a float `(B, T)` tensor with $\log p(\text{targets}[b, t] \mid \text{input\_ids}[b, \le t])$ and 0.0 where `targets == -100`; differentiable. `sequence_logprobs(model, input_ids, targets)`: its sum over $t$, shape `(B,)`.

> **▶ build: stage 1.** implement `token_logprobs` and `sequence_logprobs`.

### idea 2: reward models and Bradley–Terry

people don't give answers absolute scores reliably, but they can compare. the **Bradley–Terry** model (1952) connects comparisons to scores: if each answer has a latent reward $r$, the probability that $y_w$ is preferred to $y_l$ is

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big).$$

a reward model $r_\phi$ is fitted by maximum likelihood on comparisons $(x, y_w, y_l)$:

$$\mathcal{L}_\text{BT}(\phi) = -\mathbb{E}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big].$$

only differences matter, so the reward's offset (and, with finite data, much of its scale) is arbitrary: normalize it before using it as an RL reward. architecturally, a reward model is the language model itself with its head replaced: run the conversation through the transformer and apply a linear **value head** to the hidden state of the **last real token**, the only position that has seen the whole response. with right-padded batches that's the last True of the attention mask, not the last column. reward models are usually initialized from the SFT model, so they start out understanding the text they're judging.

> **spec.** `RewardModel(backbone, d_model)`: keeps `backbone` (anything with `hidden_states(idx)` → `(B, T, d_model)`) and `value_head = nn.Linear(d_model, 1)`. `forward(input_ids, attention_mask=None)` returns `(B,)` rewards: `value_head` of the hidden state at the last True position of each row of `attention_mask` (bool `(B, T)`), or at the last position if no mask is given. `bradley_terry_loss(chosen_rewards, rejected_rewards)` = $-\text{mean}(\log\sigma(\text{chosen} - \text{rejected}))$.

> **▶ build: stage 2.** implement `RewardModel` and `bradley_terry_loss`.

### idea 3: DPO

RLHF maximizes reward while staying close to a reference policy $\pi_\text{ref}$ (the SFT model):

$$\max_\pi\; \mathbb{E}_{x, y \sim \pi}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big(\pi(\cdot|x) \,\|\, \pi_\text{ref}(\cdot|x)\big).$$

this objective has a closed-form maximizer (on paper 1): $\pi^*(y|x) = \frac{1}{Z(x)}\pi_\text{ref}(y|x)\, e^{r(x,y)/\beta}$. Rafailov et al. (2023) turned it around: solve for the reward in terms of the optimal policy,

$$r(x, y) = \beta \log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x),$$

and substitute into the Bradley–Terry likelihood. the intractable $Z(x)$ cancels in the difference between two answers to the same prompt, leaving a loss on the policy alone:

$$\mathcal{L}_\text{DPO}(\pi) = -\mathbb{E}\Big[\log\sigma\Big(\beta\Big(\log\tfrac{\pi(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\tfrac{\pi(y_l|x)}{\pi_\text{ref}(y_l|x)}\Big)\Big)\Big].$$

**direct preference optimization** fits a reward model whose parameters *are* the policy: the implicit reward of an answer is $\beta \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)}$, and training on preference pairs with this loss is (in the idealized analysis) equivalent to RLHF, with no sampling, no reward model, no value function and no PPO. the reference log-probabilities are constants: compute them once, without gradients. $\beta$ plays the role of the KL coefficient: small $\beta$ lets the policy move far from the reference for a given margin.

> **spec.** `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, *, beta=0.1)`, all `(B,)` sequence log-probabilities: returns `(loss, chosen_rewards, rejected_rewards)` with `loss` the batch mean above, the reference log-probabilities treated as constants (detached), and the implicit rewards $\beta(\pi_c - \text{ref}_c)$ and $\beta(\pi_r - \text{ref}_r)$, detached.

> **🔮 predict:** at the first step of DPO the policy equals the reference. what is the loss, whatever the data?

> **▶ build: stage 3.** implement `dpo_loss`.

### idea 4: a DPO update

one update: compute the reference model's log-probabilities of the chosen and rejected responses **in eval mode and without gradients** (dropout would make the reference a moving target, and its parameters must never change), compute the policy's log-probabilities **in train mode** with gradients, take the DPO loss, and step the optimizer. report the implicit-reward margin and the fraction of pairs where the chosen response's implicit reward is higher: these are what DPO training curves plot. two facts to keep in mind: the loss can go down by pushing *both* responses' probabilities down, as long as the rejected one falls faster, and with a fixed dataset of pairs nothing stops the margin from growing forever. both are ways DPO drifts from the reference; the lab watches it happen.

> **spec.** `dpo_step(policy, reference, batch, optimizer, *, beta=0.1)`: `batch` is a dict with `chosen_ids`, `chosen_targets`, `rejected_ids`, `rejected_targets` (`(B, T)` tensors as from `collate_sft`). puts `reference` in eval mode and computes its sequence log-probabilities under `torch.no_grad()`; puts `policy` in train mode; computes `dpo_loss(..., beta=beta)`; `optimizer.zero_grad()`, backward, `optimizer.step()`. returns a dict of Python floats with at least `loss`, `accuracy` (fraction with chosen implicit reward > rejected) and `margin` (mean chosen − rejected implicit reward).

> **▶ build: stage 4.** implement `dpo_step`.

### idea 5: RLHF rewards, and the cost of best-of-n

in RLHF with PPO, the reward model scores a whole response, but PPO assigns credit per token. InstructGPT's shaping: each response token $t$ gets reward $-\beta\,(\log\pi(y_t|\cdot) - \log\pi_\text{ref}(y_t|\cdot))$, a per-token KL penalty, and the last token also gets the reward model's score. summed over the response, that's $r(x, y) - \beta \log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)}$, whose expectation is exactly the KL-regularized objective of idea 3. PPO then treats generation as an episode: actions are tokens, returns are sums of future rewards, and the clipped objective (chapter 24) keeps each update small.

the simplest optimizer of all is **best-of-n** sampling: draw $n$ responses from the reference policy and return the one the reward model likes best. no training, and a strong baseline. it moves the output distribution away from the reference by a known amount: for continuous rewards,

$$\mathrm{KL}(\pi_\text{BoN} \,\|\, \pi_\text{ref}) = \log n - \frac{n-1}{n}$$

(on paper 3), which is why Gao et al. plot every method against KL: best-of-64 spends only 3.2 nats, while an RL run can drift tens of nats.

> **spec.** `kl_penalized_rewards(scores, policy_logps, ref_logps, mask, *, beta=0.1)`: `scores` `(B,)`, per-token log-probabilities `(B, T)`, `mask` bool `(B, T)` marking response tokens; returns `(B, T)` with $-\beta(\text{policy} - \text{ref})$ on masked positions, 0 elsewhere, plus `scores[b]` added at the last True of row $b$. `best_of_n_kl(n)` = $\log n - (n-1)/n$, `ValueError` for $n < 1$.

> **🔮 predict:** optimizing a policy hard against the proxy reward model, with no KL penalty: what will its outputs look like after a few dozen PPO iterations?

> **▶ build: stage 5.** implement `kl_penalized_rewards` and `best_of_n_kl`.

## worked example

**Bradley–Terry.** rewards 1.5 (chosen) and 0.5 (rejected): $P = \sigma(1) = 0.731$, loss $\log(1 + e^{-1}) = 0.313$. equal rewards: $\log 2 = 0.693$.

**DPO.** $\beta = 0.1$; the policy's log-probabilities of the chosen and rejected responses are $-10$ and $-12$, the reference's $-11$ and $-11$. implicit rewards: $0.1 \cdot (-10 + 11) = 0.1$ and $0.1 \cdot (-12 + 11) = -0.1$; margin 0.2; loss $\log(1 + e^{-0.2}) = 0.598$. at the first step, policy = reference: margin 0, loss $\log 2$.

**KL-shaped rewards.** a response of two tokens, policy log-probs $(-1, -2)$, reference $(-1.5, -1)$, score 1, $\beta = 0.5$: per-token rewards $(-0.5 \cdot 0.5,\; -0.5 \cdot (-1) + 1) = (-0.25, 1.5)$, summing to $1 - 0.5 \cdot (-0.5) = 1.25$.

**best-of-n.** $n = 4$: $\log 4 - 3/4 = 0.636$ nats; $n = 64$: $4.159 - 0.984 = 3.17$. your functions must reproduce all of these.

## on paper

1. **the optimal KL-regularized policy.** show that for a fixed $x$, $\pi^*(y) = \pi_\text{ref}(y) e^{r(y)/\beta} / Z$ maximizes $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi\|\pi_\text{ref})$. (hint: write the objective as $-\beta\,\mathrm{KL}(\pi \| \pi^*) + \beta \log Z$.)

<details><summary>solution</summary>

$\mathbb{E}_\pi[r] - \beta\,\mathbb{E}_\pi[\log\pi - \log\pi_\text{ref}] = -\beta\,\mathbb{E}_\pi[\log\pi - \log\pi_\text{ref} - r/\beta] = -\beta\,\mathbb{E}_\pi[\log\pi - \log(\pi_\text{ref}e^{r/\beta}/Z)] + \beta\log Z = -\beta\,\mathrm{KL}(\pi\|\pi^*) + \beta\log Z$. $Z$ doesn't depend on $\pi$ and KL is minimized (at 0) by $\pi = \pi^*$.

</details>

2. **the DPO gradient.** with $u = \beta(\log\frac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \log\frac{\pi(y_l)}{\pi_\text{ref}(y_l)})$, show $\nabla_\theta \mathcal{L}_\text{DPO} = -\beta\,\sigma(-u)\,[\nabla\log\pi(y_w) - \nabla\log\pi(y_l)]$. interpret the weight $\sigma(-u)$.

<details><summary>solution</summary>

$\frac{d}{du}(-\log\sigma(u)) = -\sigma(-u)$, and $\nabla u = \beta(\nabla\log\pi(y_w) - \nabla\log\pi(y_l))$ since the reference terms are constants. the weight $\sigma(-u)$ is the probability the implicit reward model assigns to the *wrong* ordering: pairs it already ranks correctly with a large margin get little gradient, pairs it gets wrong get the most. note the gradient only cares about the difference: nothing in it keeps $\log\pi(y_w)$ from decreasing.

</details>

3. **the KL of best-of-n.** let $\pi_\text{ref}$ be a continuous distribution over responses whose rewards have CDF $F$. the best of $n$ samples has density $n F(y)^{n-1}\pi_\text{ref}(y)$ (in reward order). show $\mathrm{KL}(\pi_\text{BoN}\|\pi_\text{ref}) = \log n - (n-1)/n$.

<details><summary>solution</summary>

$\mathrm{KL} = \mathbb{E}_\text{BoN}[\log(nF^{n-1})] = \log n + (n-1)\,\mathbb{E}_\text{BoN}[\log F]$. under BoN, $U = F(y)$ has density $n u^{n-1}$ on $[0, 1]$, so $\mathbb{E}[\log U] = \int_0^1 n u^{n-1}\log u\,du = -1/n$. hence $\log n - (n-1)/n$. (with ties or discrete rewards it's an upper bound.)

</details>

## experiment

the lab pretrains and fine-tunes a small Llama to finish half-lines of Shakespeare, then plays both sides of preference learning with **synthetic** labels (there are no human preferences on this machine): a hidden "gold" rule labels pairs of the model's answers, with Bradley–Terry noise, and your reward model learns a proxy from them. it measures best-of-n against the proxy, runs PPO-based RLHF with and without a KL penalty, and runs DPO on the same pairs with two values of $\beta$, plotting the gold reward against the KL from the SFT model for each.

## check your understanding

1. why does the reward model read the hidden state at the last real token, and what goes wrong if you read the last column of a right-padded batch?

<details><summary>answer</summary>

with causal attention only the last real token has attended to the entire response. the last column of a shorter, padded row is a padding position: its hidden state depends on padding tokens, so the reward would depend on how much padding the batch needed.

</details>

2. a reward model's scores are all shifted by +5 after retraining. does it matter for best-of-n? for PPO?

<details><summary>answer</summary>

not for best-of-n (only the ranking matters) and not for PPO's advantages if they're normalized, since a constant per-sequence shift is removed by a baseline. it matters for the balance against a fixed KL coefficient if rewards are used raw; that's one reason to standardize rewards.

</details>

3. why is the reference model in eval mode and under `no_grad` in DPO? what if it shared parameters with the policy?

<details><summary>answer</summary>

the loss is defined relative to a *fixed* $\pi_\text{ref}$: with dropout its log-probabilities would be noisy, and with gradients (or shared parameters) the optimizer could lower the loss by moving the reference instead of the policy, so the implicit rewards would lose their meaning.

</details>

4. what does the KL penalty protect against, in terms of the reward model's training data?

<details><summary>answer</summary>

the reward model was trained on answers from (near) the SFT policy. far from that distribution its predictions are extrapolations, often confidently wrong. the KL penalty keeps the policy where the proxy is trustworthy.

</details>

5. Gao et al. found that larger reward models over-optimize later. why would more data or a bigger proxy help?

<details><summary>answer</summary>

a better proxy has fewer and smaller errors, and its errors stay uncorrelated with the gold reward further from the training distribution, so the optimizer must travel further (more KL) before the proxy's errors dominate the selection.

</details>

6. DPO's loss on a fixed dataset can go to zero. why is that a problem, and what does a larger $\beta$ do?

<details><summary>answer</summary>

the loss reaches zero only as the margin goes to infinity, which the policy can achieve by driving the rejected responses' probability to zero, often dragging the chosen ones down too and moving probability to responses in neither set. a larger $\beta$ reaches a given loss with a smaller log-ratio, so the policy stays closer to the reference (IPO and early stopping are other fixes).

</details>

## common bugs

- **summing log-probabilities over prompt or padding tokens.** the comparison must be over the response tokens only.
- **forgetting `no_grad`/`eval` for the reference**, or letting the reference share parameters with the policy (deep-copy it).
- **flipping the sign** in Bradley–Terry or DPO: the model learns to prefer the rejected answers; a sanity check is that the first DPO loss is $\log 2$ and decreases.
- **reading the reward at the wrong position** with padded batches.
- **unnormalized rewards with a fixed KL coefficient**: the effective strength of the penalty changes every time the reward model is retrained.
- **trusting the proxy curve.** plot something independent (a held-out judge, human ratings, here the gold rule) against KL.

## going deeper

- Christiano et al. (2017), "Deep reinforcement learning from human preferences", *NeurIPS*. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Stiennon et al. (2020), "Learning to summarize from human feedback", *NeurIPS*. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
- Ouyang et al. (2022), "Training language models to follow instructions with human feedback". [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Bradley & Terry (1952), "Rank analysis of incomplete block designs: I. the method of paired comparisons", *Biometrika*. [doi:10.2307/2334029](https://doi.org/10.2307/2334029)
- Rafailov et al. (2023), "Direct preference optimization: your language model is secretly a reward model", *NeurIPS*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Gao, Schulman & Hilton (2022), "Scaling laws for reward model overoptimization", *ICML 2023*. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
- Rafailov et al. (2024), "Scaling laws for reward model overoptimization in direct alignment algorithms". [arXiv:2406.02900](https://arxiv.org/abs/2406.02900)
- Azar et al. (2023), "A general theoretical paradigm to understand learning from human preferences" (IPO). [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)

## connections

- ← builds on: sequence log-probabilities and SFT batches (topic 03), PPO and advantages (ch 24 · 04), logistic regression and maximum likelihood (ch 08), KL divergence (ch 05).
- → used in: the capstone aligns its assistant with your `dpo_loss` and measures it against a gold rule; topic 06 asks how to evaluate the result; chapter 28 · 07 returns to reward hacking and specification gaming as alignment failures.
