# decoding & inference

## why this matters

a language model doesn't produce text; it produces a probability distribution over the next token. everything between that distribution and the words a user sees is **decoding**, and it matters more than people expect. the single most likely continuation, chosen greedily, is usually dull and often loops: "I will the shall the shall the shall...". sampling at temperature 1 is diverse but occasionally picks a junk token from the long tail, and once one junk token is in the context the model conditions on it and degrades further (Holtzman et al., 2020). the knobs you built in chapter 18 (temperature, top-k, top-p) and the two you add here (min-p and a repetition penalty) are how practitioners steer between those failures.

decoding is also where the cost is. generating is memory-bound (chapter 26): each new token needs a full pass over the weights, so a 70B model produces a few dozen tokens per second no matter how fast its arithmetic is. **speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) breaks the one-token-per-pass rule: a small draft model guesses several tokens cheaply, and the big model checks all of them in a single pass. a rejection-sampling correction makes the result an exact sample from the big model, not an approximation, and the reported speedups are 2–3× on large models.

and sometimes the output must have a shape: valid JSON, a number, one of a fixed set of labels. **constrained decoding** masks every token that would break a grammar. it guarantees validity, but, as the lab shows, it does not sample from the model conditioned on validity.

## warm-up: predict first

1. greedy decoding picks the most likely token every time. is the resulting text the most likely *sequence*? is it good text?
2. a draft model agrees with the target 60% of the time per token. with 4 drafted tokens per round, how many tokens does each target call produce on average?
3. can speculative decoding change the distribution of the output text, if the draft model is bad?
4. you force a model to answer with one of eight names by masking every other character. does it pick each name as often as the model would among valid answers?

## prerequisites check

- [ ] you built `sample_next_token` (temperature, top-k, top-p) and `generate_cached` (→ ch 18 · 04, 06)
- [ ] you can explain why decoding is memory-bound and what a KV cache saves (→ ch 18 · 06, ch 26 · 06)
- [ ] you know rejection sampling (→ ch 04)
- [ ] you have a `Llama` that follows the causal-LM protocol (→ topic 02)

## core ideas

### idea 1: shaping the next-token distribution

temperature divides the logits before the softmax: below 1 it sharpens towards the argmax, above 1 it flattens. top-k keeps the $k$ most likely tokens, top-p (nucleus) the smallest set with mass at least $p$. two more knobs are in common use.

**min-p** (Nguyen et al., 2024) keeps every token whose probability is at least $p_\text{min}$ times the top token's probability. the cut-off scales with the model's confidence: when the model is sure (top probability 0.9), only tokens above $0.9\,p_\text{min}$ survive; when it's unsure (a flat distribution), many tokens do. top-k can't adapt like that, and top-p adapts in a way that admits long tails when the distribution is flat. this is what lets min-p sample at high temperatures without falling into junk tokens.

the **repetition penalty** (introduced by Keskar et al., 2019, for CTRL, which divides the logits of repeated tokens by $\theta$) discourages tokens that already appear in the recent context. dividing a negative logit would make the token *more* likely, so the version in Hugging Face's `transformers`, built here, is sign-aware: a positive logit is divided by the penalty $\theta > 1$, a negative one multiplied by it, so the token becomes less likely either way. it's a blunt instrument (it penalizes "the" for appearing before, too), but it breaks the loops that greedy decoding falls into, and every serving library exposes it.

> **spec.** `apply_repetition_penalty(logits, prev_tokens, penalty)`: `logits` `(B, V)`, `prev_tokens` `(B, L)` int64; returns a new tensor where, in each row, every token id appearing in that row of `prev_tokens` (once or many times: it's penalized once) has its logit divided by `penalty` if positive and multiplied by it otherwise; `ValueError` for `penalty <= 0`; doesn't modify the input. `min_p_filter(logits, min_p)`: sets to $-\infty$ every logit whose softmax probability is below `min_p` times its row's largest probability (equal is kept); `ValueError` unless $0 \le$ `min_p` $\le 1$.

> **▶ build: stage 1.** implement `apply_repetition_penalty` and `min_p_filter`.

### idea 2: the acceptance step

speculative decoding rests on one small piece of probability. a draft model proposes a token $x \sim q$; the target's distribution at the same position is $p$. accept $x$ with probability $\min(1, p(x)/q(x))$; if it's rejected, sample a replacement from the **residual**

$$r(x) = \frac{\max(0,\, p(x) - q(x))}{\sum_{x'} \max(0,\, p(x') - q(x'))}.$$

the emitted token is then distributed exactly as $p$: the probability of emitting $x$ is $q(x)\min(1, p(x)/q(x)) + (1 - \alpha)\, r(x) = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)$, where $\alpha = \sum_x \min(p(x), q(x))$ is the overall acceptance probability (on paper 1). where the draft over-proposes a token ($q > p$) it's accepted only sometimes; where it under-proposes ($p > q$), the residual tops it up. $\alpha = 1 - \mathrm{TV}(p, q)$: a draft is useful exactly to the extent that it's close to the target in total variation.

with $k$ drafted tokens the check runs left to right: accept the first, then the second, ... and stop at the first rejection, replacing that token by a residual sample. tokens after a rejection were drafted from a context that's no longer true, so they're discarded. if all $k$ are accepted, the target's pass has already computed its distribution at the position after the last draft, so you get one more token, the **bonus**, sampled from $p_{k+1}$ for free.

> **spec.** `speculative_accept(p, q, draft_tokens, *, generator=None)`: `p` `(k+1, V)` target probabilities at the $k$ drafted positions and one more, `q` `(k, V)` draft probabilities, `draft_tokens` `(k,)`; for $i = 0, \dots, k-1$ accept `draft_tokens[i]` with probability $\min(1, p_i(x)/q_i(x))$ (uniforms from `torch.rand(..., generator=generator)`), else return `(i, a sample from the residual of p[i] and q[i])`; if all are accepted, return `(k, a sample from p[k])`. the token is a Python `int`.

> **🔮 predict:** with $p = (0.1, 0.5, 0.3, 0.1)$ and $q = (0.4, 0.2, 0.2, 0.2)$, how often is the draft accepted, and what does the residual look like?

> **▶ build: stage 2.** implement `speculative_accept`.

### idea 3: speculative decoding

a round: the draft model generates $k$ tokens autoregressively (cheap: it's small); the target runs **once** on the context plus all $k$ drafts, which gives its distribution at all $k+1$ positions in one pass, the same parallelism as training; `speculative_accept` decides how many drafts to keep and adds one corrected or bonus token. every round emits between 1 and $k+1$ tokens for one target pass. if each draft is accepted independently with probability $\alpha$, a round emits on average

$$\mathbb{E}[\text{tokens per target call}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}.$$

the speedup in wall-clock time is that, divided by the cost of a round relative to one plain target step: the $k$ draft passes plus one target pass over $k+1$ positions. in the memory-bound regime that pass costs about the same as a one-token pass, which is the whole trick. at temperature 0 the same structure gives exact greedy decoding: accept drafts while they equal the target's argmax, then emit the target's argmax.

two practical details. sample the draft at the same temperature as the target (any draft distribution keeps the output exact, but a mismatched one is rejected more). and a production implementation keeps KV caches for both models and rolls back the entries of rejected tokens; yours reruns both models on the whole context each step, which is simpler but gives away much of the speedup (the lab measures how much).

> **spec.** `speculative_decode(target, draft, idx, max_new_tokens, *, k=4, temperature=1.0, generator=None)`: `ValueError` unless `idx` has batch size 1 and $k \ge 1$; repeatedly drafts $\min(k, \text{remaining})$ tokens from `draft` (sampled from softmax(logits / temperature), or argmax at temperature 0), runs `target` once on the context plus the drafts, and accepts with `speculative_accept` (temperature > 0) or by matching the target's argmax (temperature 0); crops contexts to a model's `block_size` if it has one; puts both models in eval mode and restores their modes. returns `(tokens, acceptance_rate)`: `tokens` of shape `(1, T + max_new_tokens)`, and the fraction of proposed draft tokens that were accepted (a float).

> **▶ build: stage 3.** implement `speculative_decode`.

### idea 4: constrained decoding

to force an output format, compute at each step the set of tokens that can still lead to a valid output (from a grammar, a regular expression, a list of allowed strings) and set every other logit to $-\infty$ before sampling. libraries compile regexes and JSON schemas into automata over the tokenizer's vocabulary so the mask costs almost nothing (Willard & Louf, 2023). validity is guaranteed by construction.

but the result is **not** a sample from the model conditioned on producing valid output. masking renormalizes the model's choice *at each step*, among tokens that can still be completed, without asking how likely the model is to complete them. if a valid string starts with a character the model likes for other reasons, the mask funnels all of that character's probability into the one valid continuation. sampling from the true conditional needs the probability of finishing validly from each prefix, which is generally intractable; rejection sampling gets it exactly but wastes samples, and Park et al. (2024) give an approximation that converges to it. for a single short choice (a label, a yes/no) the distortion is small or zero; for long structured outputs it can be large.

> **spec.** `constrained_decode(model, idx, max_new_tokens, allowed, *, temperature=1.0, generator=None)`: batch size 1 (`ValueError` otherwise); at each step `allowed(generated)` receives the list of tokens generated so far (not the prompt) and returns a bool mask `(V,)`; if no token is allowed, stop early; otherwise mask the model's last logits to the allowed set and sample at `temperature` (argmax at 0); crop to `block_size`; restore the model's mode; return `(1, T + n_generated)`.

> **🔮 predict:** the model writes one of eight speaker names freely 8% of the time. will constrained decoding pick the names in the same proportions as the model does among its valid answers?

> **▶ build: stage 4.** implement `constrained_decode`.

## worked example

**repetition penalty.** logits $(2.0, -1.0, 0.5, 3.0)$, previous tokens $\{0, 1\}$, $\theta = 2$: $(1.0, -2.0, 0.5, 3.0)$.

**min-p.** probabilities $(0.5, 0.3, 0.15, 0.05)$, $p_\text{min} = 0.4$: the threshold is $0.2$, so the first two survive.

**acceptance.** $p = (0.1, 0.5, 0.3, 0.1)$, $q = (0.4, 0.2, 0.2, 0.2)$: $\alpha = 0.1 + 0.2 + 0.2 + 0.1 = 0.6$. the residual is $(0, 0.3, 0.1, 0)/0.4 = (0, 0.75, 0.25, 0)$. check token 1: $q_1 \min(1, p_1/q_1) + (1 - \alpha) r_1 = 0.2 + 0.4 \cdot 0.75 = 0.5 = p_1$.

**tokens per call.** $\alpha = 0.6$, $k = 4$: $(1 - 0.6^5)/(1 - 0.6) = 2.31$ tokens per target call. your functions must reproduce these.

## on paper

1. **exactness.** prove that the token emitted by the single-step acceptance procedure has distribution $p$, and that $\alpha = 1 - \mathrm{TV}(p, q)$.

<details><summary>solution</summary>

$P(\text{emit } x) = P(\text{draft } x, \text{accept}) + P(\text{reject})\, r(x) = q(x)\min(1, p(x)/q(x)) + (1 - \alpha)\frac{(p(x) - q(x))^+}{\sum_{x'}(p(x') - q(x'))^+}$. the first term is $\min(p(x), q(x))$. $1 - \alpha = 1 - \sum\min(p, q) = \sum (p - \min(p, q)) = \sum (p - q)^+$, so the second term is $(p(x) - q(x))^+$. their sum is $p(x)$. and $\mathrm{TV} = \frac12\sum|p - q| = \sum(p - q)^+ = 1 - \alpha$.

</details>

2. **expected tokens per call.** with independent acceptances of probability $\alpha$, the number of accepted drafts $N$ is the number of successes before the first failure, capped at $k$. show $\mathbb{E}[N + 1] = (1 - \alpha^{k+1})/(1 - \alpha)$.

<details><summary>solution</summary>

$P(N \ge j) = \alpha^j$ for $j \le k$, so $\mathbb{E}[N] = \sum_{j=1}^k \alpha^j$ and $\mathbb{E}[N + 1] = \sum_{j=0}^k \alpha^j = (1 - \alpha^{k+1})/(1 - \alpha)$. as $k \to \infty$ it tends to $1/(1 - \alpha)$: with $\alpha = 0.6$ no amount of drafting gets past 2.5 tokens per call.

</details>

3. **when does it pay?** let a draft pass cost $c$ times a target pass (both memory-bound, one token), and assume verifying $k+1$ positions costs one target pass. write the speedup over plain decoding and find the best $k$ for $\alpha = 0.8$, $c = 0.05$.

<details><summary>solution</summary>

speedup $= \frac{(1 - \alpha^{k+1})/(1 - \alpha)}{1 + kc}$. for $\alpha = 0.8$, $c = 0.05$: $k = 2$: $2.44/1.10 = 2.22$; $k = 4$: $3.36/1.20 = 2.80$; $k = 6$: $3.95/1.30 = 3.04$; $k = 8$: $4.33/1.40 = 3.09$; $k = 10$: $4.57/1.50 = 3.05$. the optimum is around $k = 8$, about 3×. a draft ten times more expensive ($c = 0.5$) gives at most about 1.2× (at $k = 2$: $2.44/2.0$): drafts must be cheap.

</details>

## experiment

the lab trains a character-level Llama on Shakespeare and compares six sampling recipes (greedy, greedy with a repetition penalty, three temperatures, and min-p) on repetitiveness and on the model's own likelihood of what it wrote. it measures speculative decoding with three drafts of increasing quality (bigram counts, a tiny Llama, a small Llama): the acceptance probability $\alpha$, tokens per target call against the formula, and tokens per second against the target alone, with and without its KV cache. last, it compares constrained decoding with rejection sampling for a choice among eight speaker names.

## check your understanding

1. why does greedy decoding loop?

<details><summary>answer</summary>

once a phrase has occurred, the context makes it more likely to occur again (models learn that text repeats), and greedy decoding always takes the most likely path, so a loop, once entered, reinforces itself. sampling escapes by chance; a repetition penalty breaks it by lowering the logits of recent tokens.

</details>

2. min-p with $p_\text{min} = 0.1$: how many tokens survive when the top probability is 0.95? when the distribution is uniform over 50 tokens?

<details><summary>answer</summary>

with top probability 0.95, only tokens with probability $\ge 0.095$, and the remaining 0.05 of mass can contain at most none (no other token can reach 0.095), so just the top token. uniform over 50: all 50 have $0.02 \ge 0.1 \times 0.02$, so all survive. the filter adapts to confidence.

</details>

3. why must tokens after the first rejected draft be discarded?

<details><summary>answer</summary>

they were drafted conditional on the rejected token, which has been replaced. the target's distributions at those positions were also computed with the wrong token in context. keeping them would sample from the wrong conditional.

</details>

4. a draft model is identical to the target. what's the acceptance rate, and is speculative decoding then fast?

<details><summary>answer</summary>

every draft is accepted, $k + 1$ tokens per target call, but each draft pass costs as much as a target pass, so a round of $k$ draft passes plus one verify pass yields $k+1$ tokens for $k+1$ passes: no speedup. the draft must be much cheaper than the target, not just accurate.

</details>

5. why is the acceptance rate (accepted / proposed) lower than $\alpha$?

<details><summary>answer</summary>

after the first rejection in a round, the remaining drafts are thrown away unexamined, but they still count as proposed. with $k = 4$ and $\alpha = 0.5$, a round proposes 4 and accepts on average $0.5 + 0.25 + 0.125 + 0.0625 = 0.94$, a rate of 0.23.

</details>

6. when is constrained decoding exactly equal to conditioning on validity?

<details><summary>answer</summary>

when every allowed prefix can be completed validly with probability 1 under the model's own continuation, e.g. a single-token choice, or a grammar where every allowed token leaves all continuations allowed. otherwise each step's renormalization ignores how likely the model is to complete each prefix.

</details>

## common bugs

- **penalizing a repeated token once per occurrence**, or modifying logits in place so the caller's tensor changes.
- **accepting with probability $p(x)$ instead of $\min(1, p(x)/q(x))$**: the output drifts towards the draft's distribution.
- **sampling the replacement from $p$ instead of the residual**: tokens the draft over-proposes come out too often.
- **an off-by-one in the target's positions**: the logits at position $t$ predict token $t + 1$; the distribution for draft $i$ is at the position of the token before it.
- **forgetting the bonus token**, or emitting $k + 1$ tokens past `max_new_tokens`.
- **passing the prompt to `allowed`** instead of the generated tokens, so the grammar starts in the wrong state.

## going deeper

- Holtzman, Buys, Du, Forbes & Choi (2020), "The curious case of neural text degeneration", *ICLR*. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Keskar et al. (2019), "CTRL: a conditional transformer language model for controllable generation" (repetition penalty). [arXiv:1909.05858](https://arxiv.org/abs/1909.05858)
- Nguyen et al. (2024), "Turning up the heat: min-p sampling for creative and coherent LLM outputs". [arXiv:2407.01082](https://arxiv.org/abs/2407.01082)
- Leviathan, Kalman & Matias (2023), "Fast inference from transformers via speculative decoding", *ICML*. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al. (2023), "Accelerating large language model decoding with speculative sampling". [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Willard & Louf (2023), "Efficient guided generation for large language models". [arXiv:2307.09702](https://arxiv.org/abs/2307.09702)
- Park et al. (2024), "Grammar-aligned decoding", *NeurIPS*. [arXiv:2405.21047](https://arxiv.org/abs/2405.21047)

## connections

- ← builds on: sampling and the KV cache (ch 18 · 04, 06), the memory-bound decoding argument (ch 26 · 01, 06), rejection sampling (ch 04), your `Llama` (topic 02).
- → used in: the capstone generates its assistant's answers with these samplers; topic 06 decodes greedily for exact-match evaluation and shows how decoding choices move benchmark scores; topic 07's agent loop uses constrained formats for tool calls.
