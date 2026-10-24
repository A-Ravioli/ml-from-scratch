# evaluation

## why this matters

every claim in this field ends in a number: model A scores 71.3 on a benchmark, model B 69.8. most of those numbers are less solid than they look. the same model scored on the same questions can move by tens of points when the prompt's formatting changes (Sclar et al., 2023). a small model that "fails" a multiple-choice benchmark may know the answers and fail only to say "B" (Gu et al., 2024). a benchmark posted on the web ends up in the next crawl, and a model that has seen the test set scores well for the wrong reason (Brown et al., 2020, built 13-gram overlap filters for exactly this). differences of a point or two between models are often inside the noise of a few hundred questions (Miller, 2024). and when the output is open-ended, the judge is now often another LLM, with biases of its own: it tends to prefer whichever answer it reads first (Zheng et al., 2023; Wang et al., 2023).

this topic builds the core of an evaluation harness and the tools to keep it honest: scoring a continuation by log-likelihood, multiple-choice accuracy with length normalization, bootstrap intervals for a score and for a *paired* difference between models, n-gram contamination checks, and a pairwise judge that cancels position bias. the lab then shows one model, one benchmark, and five different scores.

## warm-up: predict first

1. a model picks between "yes" and "yes, absolutely, without any doubt" by total log-likelihood. which does it favour, and why?
2. a model scores 70% on 100 questions. roughly how wide is a 95% interval for its true accuracy?
3. two models score 73% and 70% on the same 300 questions and disagree on only 15 of them (12 won by the first, 3 by the second). is the difference meaningful?
4. you ask an LLM "which answer is better, 1 or 2?" and it says 1. you swap the answers and ask again. what do you learn?

## prerequisites check

- [ ] you can compute log-probabilities of a continuation under a causal LM (→ topic 04)
- [ ] you know the bootstrap and percentile intervals (→ ch 05 · 03)
- [ ] you know why tokenizers can merge characters across a boundary (→ ch 17 · 02)
- [ ] you have shingles and n-gram sets from topic 01 (→ topic 01)

## core ideas

### idea 1: scoring a continuation

the workhorse of LLM evaluation is a single quantity: how likely does the model find a continuation $c$ given a context $x$?

$$\ell(c \mid x) = \sum_{t=1}^{|c|} \log p(c_t \mid x, c_{<t}),$$

plus a flag saying whether every $c_t$ is the model's argmax (an exact-match test that doesn't need generation). this is what lm-evaluation-harness calls `loglikelihood`. two details trip people up. **tokenize context and continuation separately** and concatenate the ids: encoding $x + c$ in one go can merge characters across the boundary (with BPE, "Answer:" + " B" may not split where you think), and then the "continuation tokens" aren't the continuation. and **truncate from the left**: if the sequence doesn't fit in the model's context, drop the oldest context tokens, never the continuation. evaluation runs in eval mode without gradients.

> **spec.** `loglikelihood(model, tokenizer, context, continuation, *, device='cpu')`: `ctx = tokenizer.encode(context)`, `cont = tokenizer.encode(continuation)`, `ValueError` if either is empty; the model reads `(ctx + cont)[:-1]` (left-truncated to `model.block_size` tokens if it has one; `ValueError` if the continuation alone doesn't fit) and the logits at the last `len(cont)` positions score `cont`. returns `(sum of log-probabilities as a float, is_greedy as a bool)`; no gradients; restores the model's mode.

> **▶ build: stage 1.** implement `loglikelihood`.

### idea 2: multiple choice by likelihood

a multiple-choice item is a question and $K$ answers. the standard way to score a base model (no instruction tuning) is to ask which answer it finds most likely as a continuation of the question: $\hat{k} = \arg\max_k \ell(c_k \mid x)$. no generation, no parsing of free text, no chance of an answer in the wrong format. but total log-likelihood is biased towards **short** answers: every extra token multiplies in another probability below 1. dividing by the number of tokens (or characters, or bytes; each harness picks one) compares average per-token log-likelihood instead. it's a fix, not a law: it can over-favour long answers made of easy tokens, and there's no single right normalization, which is one reason published numbers differ.

the other big choice is the **format**. in the *cloze* format the answers themselves are the continuations. in the *lettered* format (MMLU's, Hendrycks et al., 2021) the answers are listed as A–D in the prompt and the continuation is the letter. lettered scoring tests whether the model can map its knowledge to a symbol, a skill small models lack: Gu et al. (2024) recommend cloze for small models and lettered for large ones, because each format systematically under-rates one group.

> **spec.** `evaluate_multiple_choice(model, tokenizer, items, *, normalize=True, device='cpu')`: `items` is a list of dicts with `question` (the context, prompt template included), `choices` (list of continuation strings, used as given) and `answer` (index). for each item, score every choice with `loglikelihood(model, tokenizer, question, choice)`, divided by `len(tokenizer.encode(choice))` if `normalize`; the prediction is the argmax (first on ties). returns the accuracy as a float.

> **🔮 predict:** the same model and questions in five prompt formats. how far apart are the highest and lowest accuracy?

> **▶ build: stage 2.** implement `evaluate_multiple_choice`.

### idea 3: how sure are you?

a benchmark is a sample of questions from some population. with $n$ items and accuracy $\hat{p}$, the standard error is about $\sqrt{\hat{p}(1-\hat{p})/n}$: for 100 items at 70%, 4.6 points, so a 95% interval is about $\pm 9$ points. the **bootstrap** (chapter 05) gives intervals without the formula: resample $n$ items with replacement many times, recompute the accuracy each time, and read off the 2.5th and 97.5th percentiles.

comparing two models needs one more idea. their scores on the same items are strongly correlated: easy questions are easy for both. an unpaired comparison treats the two accuracies as independent and gets a wide interval for their difference; a **paired** bootstrap resamples the *same* items for both models in each replicate, so the shared difficulty cancels and only the items where they disagree contribute. if two models disagree on 15 of 300 items, the difference's standard error is set by those 15, not by all 300 (Miller, 2024, makes this case for LLM evals). report intervals (and the number of items) with every score.

> **spec.** `accuracy_ci(correct, *, n_boot=2000, alpha=0.05, rng=None)`: `correct` a 0/1 (or bool) array; returns `(accuracy, lo, hi)` with a percentile bootstrap over items, `n_boot` replicates, `rng` normalized with `mlfs.utils.as_rng`. `paired_accuracy_diff(correct_a, correct_b, *, n_boot=2000, alpha=0.05, rng=None)`: `(acc_a − acc_b, lo, hi)` resampling the same item indices for both; `ValueError` if the arrays' shapes differ.

> **▶ build: stage 3.** implement `accuracy_ci` and `paired_accuracy_diff`.

### idea 4: contamination

if test items appear in the training data, a score measures memory, not ability. GPT-3's authors (Brown et al., 2020) checked every benchmark item for 13-word n-grams shared with the training corpus and reported results on the clean subset; later models' reports do similar checks. the method is the set machinery of topic 01 applied to words: lowercase, strip punctuation, take all word $n$-grams of the training texts, and flag every test item that shares at least one. $n$ is a trade-off: too small and common phrases ("in the middle of the") flag everything; too large and a lightly reworded copy slips through. contamination checks need the training data, which closed models don't publish; then evaluators fall back to fresh items written after the training cutoff, private test sets, or probing whether the model can reproduce test items verbatim (Golchin & Surdeanu, 2023). the simplest defence is not to post test sets in plain text (Jacovi et al., 2023).

> **spec.** `ngrams(text, n=13)`: the set of word $n$-gram tuples, words being the runs of `[a-z0-9']` in the lowercased text (empty if fewer than $n$ words). `contamination_flags(test_texts, train_texts, *, n=13)`: a bool ndarray, True for each test text sharing at least one $n$-gram with any training text.

> **▶ build: stage 4.** implement `ngrams` and `contamination_flags`.

### idea 5: LLM-as-judge

open-ended answers (an essay, a chat reply) have no answer key. the scalable substitute is to ask a strong model to compare two answers: Chatbot Arena and MT-Bench (Zheng et al., 2023) showed that GPT-4's pairwise judgements agree with human preferences about as often as humans agree with each other. but judges have biases: **position** (preferring the first or second answer shown), **verbosity** (preferring longer answers), and **self-preference** (preferring their own style). position bias has a cheap partial fix: ask twice, swapping the order, and average:

$$P(a \succ b) = \tfrac12\Big(J(a, b) + 1 - J(b, a)\Big),$$

where $J(\text{first}, \text{second})$ is the judge's probability that the first is better. a judge that always prefers the first answer gets exactly 0.5, the "no information" it really carries. swapping cancels position bias, not the others.

> **spec.** `pairwise_judge(judge, prompt, response_a, response_b)`: calls `judge(prompt, response_a, response_b)` and `judge(prompt, response_b, response_a)` (each returns the probability that its first response is better) and returns $\tfrac12(J_1 + 1 - J_2)$ as a float.

> **🔮 predict:** a small Shakespeare model is asked "which line is better, 1 or 2?" for a real line and a shuffled one. how often will it pick the real one?

> **▶ build: stage 5.** implement `pairwise_judge`.

## worked example

**a continuation by hand.** a bigram model with $\log p(\text{b}\mid\text{a}) = -0.4$, $\log p(\text{c}\mid\text{b}) = -1.1$: context "xa", continuation "bc": $\ell = -0.4 + (-1.1) = -1.5$, two tokens, $-0.75$ per token. if "b" and "c" are both argmaxes, `is_greedy` is True.

**normalization.** a short answer of 2 tokens at total $-0.6$ against a long one of 5 tokens at total $-0.8$: unnormalized the short one wins; per token ($-0.30$ against $-0.16$) the long one does.

**an interval.** 70 correct of 100: standard error $\sqrt{0.7 \cdot 0.3 / 100} = 0.046$; 95% interval about $[0.61, 0.79]$. with 400 items, half as wide.

**the judge.** a judge that says 0.9 when the good answer is first and 0.3 when it's second: $\tfrac12(0.9 + 1 - 0.3) = 0.8$. its bias of +0.1 towards the first position cancels. your functions must reproduce these.

## on paper

1. **the paired standard error.** let $a_i, b_i \in \{0, 1\}$ be two models' correctness on item $i$, and $d_i = a_i - b_i$. show that $\mathrm{Var}(\bar{a} - \bar{b}) = (\mathrm{Var}(a) + \mathrm{Var}(b) - 2\,\mathrm{Cov}(a, b))/n$, and evaluate it for $n = 300$, both accuracies near 0.7, and disagreement on 15 items (12 where only $a$ is right, 3 where only $b$ is).

<details><summary>solution</summary>

$\bar{a} - \bar{b} = \bar{d}$, and $\mathrm{Var}(d) = \mathrm{Var}(a) + \mathrm{Var}(b) - 2\mathrm{Cov}(a, b)$, divided by $n$ for the mean. directly: $d$ is $+1$ on 12 items, $-1$ on 3, 0 on 285: mean $0.03$, $E[d^2] = 15/300 = 0.05$, $\mathrm{Var}(d) = 0.05 - 0.0009 = 0.049$, standard error $\sqrt{0.049/300} = 0.0128$, so the difference is $0.030 \pm 0.025$: significant. treated as independent: $\sqrt{(0.21 + 0.21)/300} = 0.037$, $\pm 0.073$: not significant. the correlation is what makes the comparison sensitive.

</details>

2. **why separate tokenization matters.** a tokenizer merges "ab" into one token. what does `loglikelihood(context="xa", continuation="b")` compute if you encode "xab" in one piece and take its last `len(encode("b"))` tokens as the continuation?

<details><summary>solution</summary>

"xab" encodes as [x, ab]: the "last 1 token" is "ab", so you'd score $p(\text{ab} \mid \text{x})$, a different event (and the context is now just "x"). encoding separately gives [x, a] + [b] and scores $p(\text{b} \mid \text{x}, \text{a})$, what was asked. real BPE tokenizers do this at every word boundary that starts with a space.

</details>

3. **how many items do you need?** to tell apart two models whose accuracies differ by 2 points with a standard error of 0.7 points (about 3 standard errors), roughly how many items do you need if their per-item correctness is (a) independent, (b) such that they disagree on 10% of items?

<details><summary>solution</summary>

(a) $\sqrt{(0.21 + 0.21)/n} = 0.007$ gives $n \approx 8{,}600$. (b) with $\mathrm{Var}(d) \approx 0.10$ (disagreement rate, minus a negligible squared mean), $\sqrt{0.10/n} = 0.007$ gives $n \approx 2{,}000$. pairing saves a factor of four here; and a benchmark of a few hundred items can't resolve 2-point differences either way.

</details>

## experiment

the lab trains a character-level Llama on tiny Shakespeare and builds a **synthetic** cloze benchmark from held-out lines (the first half of a line and its true ending among three others). it scores the same model and items in five formats (raw, unnormalized, instruction-style, after a speaker name, lettered A–D) with bootstrap intervals and a paired comparison; trains a second model with the benchmark leaked into its training text and checks it with your contamination flags; and uses the model as a judge of real against shuffled lines, both by asking it and by comparing likelihoods, in both orders.

## check your understanding

1. why score multiple choice by likelihood rather than by generating an answer and parsing it?

<details><summary>answer</summary>

likelihood scoring always produces a valid choice and measures what the model knows independently of whether it can follow an answer format. generation needs parsing (and fails when the model says "The answer is (B)" in an unexpected way), which mixes format-following into the score. for instruction-tuned models, generation with exact-match parsing is common anyway, because that's how they're used.

</details>

2. your harness uses per-character normalization and a paper used per-token. can you compare your number to theirs?

<details><summary>answer</summary>

not directly: the two normalizations can rank choices differently, so accuracies differ systematically. that's one of the main reasons the same model has different "official" scores in different papers (Biderman et al., 2024). compare models only under one harness configuration, and report it.

</details>

3. why is a paired bootstrap the right test for "is model A better than model B on this benchmark?"

<details><summary>answer</summary>

both models answer the same items, so their errors are correlated through item difficulty. resampling items jointly keeps that correlation, and the interval reflects the variability of the difference, driven by the items where they disagree. independent resampling double-counts the item-difficulty variance and gives far too wide an interval.

</details>

4. an 8-gram check finds no overlap between a benchmark and a model's training data. is the model uncontaminated?

<details><summary>answer</summary>

not necessarily: paraphrased or translated copies, different tokenization of numbers and formulas, or answers leaked without the questions all slip past exact n-gram matching; and the check is only as good as your access to the full training data. conversely, a match on a common phrase isn't contamination.

</details>

5. a judge has a position bias. why does averaging both orders remove it, and what bias remains?

<details><summary>answer</summary>

if the judge adds the same bias $\delta$ towards the first slot regardless of content, $J(a, b) = s + \delta$ and $J(b, a) = 1 - s + \delta$, and the average $\tfrac12(s + \delta + 1 - (1 - s + \delta)) = s$. biases that depend on content (length, style, self-preference) don't flip with the order and remain.

</details>

6. benchmark scores of a new model jump on one benchmark but not on similar fresh questions. what do you suspect, and how would you check?

<details><summary>answer</summary>

contamination. check overlap with the training data if available; test with fresh, post-cutoff items of the same kind; check whether the model can complete benchmark items verbatim from a prefix; compare performance on the benchmark's original versus perturbed items.

</details>

## common bugs

- **encoding context and continuation together**, so token boundaries move and the "continuation" isn't the continuation.
- **truncating the continuation instead of the context** when the input is too long.
- **summing over the context too**, or scoring the continuation from the wrong positions (the logit at position $t$ predicts token $t + 1$).
- **comparing scores from different harness settings** (format, normalization, few-shot examples) as if they were the same benchmark.
- **unpaired comparisons** of models on the same items, and scores without intervals.
- **asking a judge once**: the result mixes the judgement with the judge's position bias.

## going deeper

- Hendrycks et al. (2021), "Measuring massive multitask language understanding", *ICLR*. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
- Biderman et al. (2024), "Lessons from the trenches on reproducible evaluation of language models" (lm-evaluation-harness). [arXiv:2405.14782](https://arxiv.org/abs/2405.14782)
- Gu et al. (2024), "OLMES: a standard for language model evaluations". [arXiv:2406.08446](https://arxiv.org/abs/2406.08446)
- Sclar, Choi, Tsvetkov & Suhr (2023), "Quantifying language models' sensitivity to spurious features in prompt design", *ICLR 2024*. [arXiv:2310.11324](https://arxiv.org/abs/2310.11324)
- Miller (2024), "Adding error bars to evals: a statistical approach to language model evaluations". [arXiv:2411.00640](https://arxiv.org/abs/2411.00640)
- Brown et al. (2020), "Language models are few-shot learners" (appendix C: contamination). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Jacovi et al. (2023), "Stop uploading test data in plain text", *EMNLP*. [arXiv:2305.10160](https://arxiv.org/abs/2305.10160)
- Golchin & Surdeanu (2023), "Time travel in LLMs: tracing data contamination in large language models", *ICLR 2024*. [arXiv:2308.08493](https://arxiv.org/abs/2308.08493)
- Zheng et al. (2023), "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena", *NeurIPS*. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- Wang et al. (2023), "Large language models are not fair evaluators". [arXiv:2305.17926](https://arxiv.org/abs/2305.17926)

## connections

- ← builds on: sequence log-probabilities (topic 04), the bootstrap (ch 05 · 03), n-gram sets (topic 01), tokenization (ch 17 · 02).
- → used in: the capstone's model card reports every number with your `accuracy_ci` and compares checkpoints with `paired_accuracy_diff`; chapter 28 · 07 builds a small safety eval with the same machinery; chapter 30 asks you to evaluate a research claim honestly.
