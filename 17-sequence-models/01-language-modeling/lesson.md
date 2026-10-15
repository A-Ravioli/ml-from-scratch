# language modeling

## why this matters

your phone suggests the next word. a speech recognizer has to decide between "recognize speech" and "wreck a nice beach", which sound almost the same. a translation system has to choose between a dozen grammatical outputs, only one of which sounds like something a person would write. all three need the same thing: a number that says how plausible a sequence of words is. a **language model** is that number. it assigns a probability to every possible sequence, and the large language models of chapter 27 are, underneath everything else, exactly this: a probability distribution over the next token given the ones before.

chapter 04's capstone built one from character counts. this topic makes it into a tool you'll keep using: tokens of any kind (characters, words, the BPE pieces of topic 02), explicit sentence boundaries, words that never appeared in training, and the two ideas that made counting models work in practice for thirty years, **smoothing** and **interpolation**. you'll also learn to read **perplexity**, the number every language model in this course, from a bigram table to a GPT, is judged by. the n-gram model you build here is the baseline that the chapter's capstone LSTM has to beat, and its failure modes (contexts it has never seen, a memory of exactly $n - 1$ tokens) are what the rest of the chapter is designed to fix.

## warm-up: predict first

1. a model assigns probability $1/50$ to every token of a held-out text. what is its perplexity?
2. you train a word bigram model on a play. the test text contains "purple elephant", never seen in training. what probability does an unsmoothed (count-based) model give the whole test text?
3. you add 1 to every bigram count. the vocabulary has 10,000 words. the word "the" was followed by 2,000 different words in 5,000 occurrences. how much of P(· | the) now goes to the 8,000 words never seen after "the"?
4. as you increase $n$ from 2 to 8 in a character model, does the perplexity on held-out text keep falling?
5. one model has perplexity 200 per word, another 5.2 per character. which is better?

## prerequisites check

- [ ] you can apply the chain rule of probability and condition on events (→ ch 04 · topic 01)
- [ ] you know cross-entropy, entropy and bits vs nats (→ ch 05 · topic 05)
- [ ] you've built a character n-gram model with add-k smoothing and temperature sampling (→ ch 04 · capstone)
- [ ] you know EM for mixture models: responsibilities, then weighted averages (→ ch 11 · topic 02)
- [ ] you know why model choices are made on a validation set, never the test set (→ ch 07 · topic 02)

## core ideas

### idea 1: the chain rule, a Markov assumption, and counting

any distribution over sequences $w_1, \dots, w_T$ factorizes exactly by the chain rule:

$$P(w_1, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \dots, w_{t-1}).$$

so a language model only ever needs to answer one question: *given everything so far, how likely is each possible next token?* the trouble is that "everything so far" is a different string almost every time. an **n-gram model** makes a Markov assumption: only the last $n - 1$ tokens matter,

$$P(w_t \mid w_1, \dots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \dots, w_{t-1}),$$

and estimates each conditional by counting. the maximum-likelihood estimate (on paper 2 of chapter 04's capstone) is a ratio of counts:

$$P_\text{ML}(w \mid h) = \frac{c(h, w)}{c(h)},\qquad c(h) = \sum_{w'} c(h, w').$$

**boundaries.** a text is made of sequences (sentences, lines, documents), and their starts and ends matter: "The" is far more likely at the start of a line, and a model should know when a sentence is over. the standard trick is padding. every sequence gets $n - 1$ copies of a start symbol `<s>` in front and one end symbol `</s>` behind:

$$\texttt{the cat sat} \;\to\; \texttt{<s> the cat sat </s>}\quad(n = 2).$$

`</s>` is a token like any other: the model predicts it, it's counted, and it's scored in perplexity. it's the model's way of saying "this sentence is done", and without it the probabilities of all sentences of all lengths wouldn't sum to 1. `<s>` is only ever context, never predicted, so it isn't in the vocabulary.

**every order at once.** smoothing (ideas 2 and 3) needs not just the order-$n$ counts but every lower order too. so your `fit` walks every position $t \ge n - 1$ of each padded sequence and, for every $j = 0, \dots, n - 1$, adds one count to `counts_[j][context][token]` with the context of length $j$, `padded[t-j:t]`. order 0 has the single empty context `()`: plain unigram counts. alongside, `totals_[j][context]` is $c(h)$, so nothing has to sum a counter at query time, and `vocab_` holds every training token plus `</s>` and `<unk>` (idea 2), sorted.

> **▶ build: stage 1.** implement `NGramLM.__init__` (store the arguments, validate them) and `NGramLM.fit`.

### idea 2: zeros, and add-k smoothing

$P_\text{ML}$ has a fatal flaw: anything not in the training data gets probability zero. a held-out sentence containing one unseen bigram gets probability $0$ as a whole, perplexity $\infty$, and the model learns nothing from the comparison (warm-up 2). and it isn't rare: on tiny Shakespeare, more than a quarter of the 7-character contexts in held-out text never occur in the training text at all.

the simplest fix pretends every possible continuation was seen $k$ extra times:

$$P_{\text{add-}k}(w \mid h) = \frac{c(h, w) + k}{c(h) + k|V|}.$$

$k = 1$ is Laplace's rule of succession. for $k < 1$ it's often called Lidstone smoothing. an unseen context ($c(h) = 0$) gives the uniform distribution $1/|V|$. the price is that the pseudo-counts take mass from what was actually seen. a context seen $c$ times that was followed by $d$ distinct tokens gives the $|V| - d$ unseen ones a total of

$$\frac{k\,(|V| - d)}{c + k|V|}$$

of its probability (on paper 2). with a 10,000-word vocabulary that's most of the mass for almost every context (warm-up 3), which is why add-one is notorious for word models.

**unknown tokens.** a held-out text will contain tokens never seen in training: names, rare words, typos. the model's vocabulary includes a special token `<unk>` that is never counted but gets its pseudo-count like everything else, and every unknown token is scored as `<unk>`. the given `logprob` does this mapping (in the context too) and then calls your `prob_add_k`. one caution: a model can look better by mapping more words to `<unk>` (a vocabulary of one token has perplexity 1). compare perplexities only between models with the same vocabulary.

> **🔮 predict:** with add-1 smoothing, is a word *bigram* model better or worse than a word *unigram* model on held-out text? (the lab measures it.)

> **▶ build: stage 2.** implement `NGramLM.prob_add_k(ctx, token)`. `ctx` is already a tuple of exactly $n - 1$ tokens and `token` is already mapped into the vocabulary.

### idea 3: interpolation, and perplexity

add-k treats every unseen continuation of a context the same. but if "the purple" was never followed by "elephant" in training, you still know a lot: "elephant" is a fairly common word, and "purple" is often followed by nouns. lower-order statistics are estimated from far more data, so they're reliable when the high-order ones are missing. **linear interpolation** (Jelinek & Mercer, 1980) mixes all the orders:

$$P(w \mid h) = \sum_{j=0}^{n-1} \lambda_j\,\tilde P_j(w \mid h_{[j]}),\qquad \lambda_j \ge 0,\quad \sum_j\lambda_j = 1,$$

where $h_{[j]}$ is the last $j$ tokens of the context and `lambdas[j]` is the weight of the estimate that conditions on $j$ tokens. the components:

- $\tilde P_0(w) = \dfrac{c(w) + k}{N + k|V|}$, the add-k unigram, positive for every token including `<unk>`. that's what guarantees the mixture never gives zero.
- for $j \ge 1$: $\tilde P_j(w \mid h_{[j]}) = P_\text{ML}(w \mid h_{[j]})$ if the context $h_{[j]}$ was seen in training, and $\tilde P_{j-1}(w \mid h_{[j-1]})$ otherwise.

the second rule matters. an unseen context has no ML estimate at all ($0/0$). treating it as zero would leak probability mass (the conditional would sum to less than 1, on paper 4); passing the estimate of the order below up the chain keeps every component a proper distribution, so the mixture is one too. with `lambdas=None` the weights are uniform, $1/n$ each.

**perplexity.** how do you score a model on held-out text with $N$ tokens (every `</s>` counts as one)? the probability of the whole text shrinks exponentially with its length, so normalize per token. the **perplexity** is

$$\text{PP} = \exp\Big(-\frac1N\sum_{t=1}^{N}\log P(w_t \mid h_t)\Big) = P(w_1, \dots, w_N)^{-1/N},$$

the exponential of the average cross-entropy, in nats (chapter 05). it reads as an effective branching factor: a model that spreads its probability uniformly over $m$ tokens at every step has perplexity exactly $m$ (warm-up 1: 50). lower is better, 1 is perfect, $|V|$ is a uniform guess. dividing the log by $\ln 2$ gives **bits per token**, and that's how you compare models with different tokenizations: perplexity per word and perplexity per character measure different things (warm-up 5), but both convert to bits per character by multiplying the total bits by $N/(\text{number of characters})$ (on paper 1).

> **🔮 predict:** a character model with $n = 8$ and add-k smoothing versus the same with interpolation: on held-out text, which is better, and by how much? does more context help add-k at all past some $n$?

> **▶ build: stage 3.** implement `NGramLM.prob_interpolated(ctx, token)` and `NGramLM.perplexity(sequences)`.

### idea 4: choosing the weights with EM

uniform weights waste the model: with enough data, the high orders deserve most of the weight; with sparse data, the low ones. the weights should be fitted, but **not on the training data**: there, the highest order always looks best (it has seen every context, and $P_\text{ML}$ maximizes training likelihood), so it would take all the weight. fit them on a held-out set, the classic use of a validation split.

interpolation is a mixture model. each held-out token was "generated" by one of the $n$ components, chosen with probability $\lambda_j$, but you don't know which. that's exactly the latent-variable setup of EM (chapter 11), with the component estimates $\tilde P_j$ held fixed and only the mixing weights free. write $P_{tj} = \tilde P_j(w_t \mid h_t)$ for the $N$ held-out tokens. then iterate:

- **E step:** the responsibility of component $j$ for token $t$, $r_{tj} = \dfrac{\lambda_jP_{tj}}{\sum_i\lambda_iP_{ti}}$.
- **M step:** $\lambda_j \leftarrow \dfrac1N\sum_t r_{tj}$, the average responsibility.

each round never decreases the held-out log-likelihood $\sum_t\log\sum_j\lambda_jP_{tj}$ (on paper 3), and since that function is concave in $\lambda$, EM converges to its global maximum. starting from uniform weights, stop when an iteration improves the log-likelihood by less than `tol` per token.

you don't need a second implementation of the components to get $P_{tj}$: interpolation with all the weight on one order, `lambdas` $= e_j$, returns exactly $\tilde P_j$. so `tune_lambdas` sets `model.lambdas` to each unit vector in turn and calls your `prob_interpolated` on every held-out token (through the given `_context`, and with unknown tokens mapped to `'<unk>'`). call `prob_interpolated` directly, not `logprob`: $\tilde P_j$ can be zero, and $\log 0$ fails.

> **▶ build: stage 4.** implement `tune_lambdas(model, sequences, *, n_iters=100, tol=1e-6)`. it sets `model.lambdas` and returns the weights and the list of log-likelihoods, one per iteration.

### idea 5: generating text

a language model is a generative model: to write, draw the first token from $P(\cdot \mid \texttt{<s>}\dots)$, append it to the history, draw the next, and so on (ancestral sampling, as in chapter 04's capstone). your `sample` scores every vocabulary token with `logprob` given the current history, turns the log-probabilities into probabilities (subtract the maximum before exponentiating, chapter 00), and draws one with `rng.choice`. two details: `<unk>` is never sampled (it isn't a real token, so drop it from the candidates and renormalize), and when `</s>` is drawn it's emitted and the history restarts empty (the given `_context` pads it with `<s>` again), so a long sample is several sentences.

sampling is also a diagnostic. a model with a good perplexity can still write gibberish, and a model with an excellent *training* perplexity often writes verbatim copies of its training text: an order-$n$ model conditions on $n - 1$ tokens, and once a context is rare enough to have been seen only once, its only continuation is the one in the training text. the lab measures how much of what your models write is copied.

> **▶ build: stage 5.** implement `NGramLM.sample(n_tokens, *, context=(), rng=None)`.

## worked example

two training sentences, `the cat sat` and `the dog sat`, and a bigram model ($n = 2$). padded: `<s> the cat sat </s>` and `<s> the dog sat </s>`, 8 predicted tokens. the vocabulary is `</s>`, `<unk>`, `cat`, `dog`, `sat`, `the`: $|V| = 6$.

| context | continuations |
|---|---|
| `()` (order 0) | the 2, cat 1, dog 1, sat 2, `</s>` 2 (total 8) |
| `<s>` | the 2 |
| `the` | cat 1, dog 1 |
| `cat`, `dog` | sat 1 each |
| `sat` | `</s>` 2 |

**add-1:** $P(\text{cat} \mid \text{the}) = (1 + 1)/(2 + 6) = 0.25$; $P(\text{sat} \mid \text{the}) = 1/8 = 0.125$; any token after the unseen context `zebra`: $1/6$.

**interpolated, $\lambda = (0.5, 0.5)$, add-1 unigram:** $\tilde P_0(\text{cat}) = (1 + 1)/(8 + 6) = 1/7$ and $P_\text{ML}(\text{cat} \mid \text{the}) = 1/2$, so $P(\text{cat} \mid \text{the}) = 0.5/7 + 0.25 = 0.3214$. `<unk>` after `the` gets only the unigram's share, $0.5 \cdot 1/14 = 0.0357$.

**perplexity of `the cat sat`** under the interpolated model: four tokens, `the`, `cat`, `sat`, `</s>`, each with $\tilde P_0 = 3/14$ or $2/14$:

| token | context | $\tilde P_0$ | $P_\text{ML}$ | $P$ |
|---|---|---|---|---|
| the | `<s>` | 3/14 | 1 | 0.6071 |
| cat | the | 2/14 | 1/2 | 0.3214 |
| sat | cat | 3/14 | 1 | 0.6071 |
| `</s>` | sat | 3/14 | 1 | 0.6071 |

$\text{PP} = \exp\big(-\tfrac14(3\ln 0.6071 + \ln 0.3214)\big) = \exp(0.6580) = 1.931$. under add-1 the same sentence has perplexity 3.159: the pseudo-counts cost a lot on a vocabulary this small.

**backoff:** a trigram model with $\lambda = (0.2, 0.3, 0.5)$ asked for $P(\text{sat} \mid \text{dog cat})$: the context `dog cat` was never seen, so the trigram component falls back to $P_\text{ML}(\text{sat} \mid \text{cat}) = 1$, and $P = 0.2 \cdot 3/14 + 0.3 \cdot 1 + 0.5 \cdot 1 = 0.843$. the tests check every number here.

## on paper

1. **perplexity across tokenizations.** a text of 1,000 characters is 200 words. a word model has perplexity 300 on it; a character model has perplexity 4.5. convert both to bits per character. which is better?

<details><summary>solution</summary>

total bits = number of tokens × bits per token. word model: $200 \log_2 300 = 200 \cdot 8.23 = 1646$ bits, so $1.65$ bits per character. character model: $1000\log_2 4.5 = 1000 \cdot 2.17 = 2170$ bits, $2.17$ bits per character. the word model is better, despite its much larger perplexity: it predicts a whole word per step. (a caveat: the word model's `<unk>` makes it look better than it is, since a whole unknown word costs only $-\log P(\texttt{<unk>})$ instead of spelling it out.)

</details>

2. **where add-k puts its mass.** a context seen $c$ times was followed by $d$ distinct tokens. show that add-k gives the unseen continuations total probability $k(|V| - d)/(c + k|V|)$. evaluate it for "the" in warm-up 3, and for a word seen 3 times (followed by 3 distinct words).

<details><summary>solution</summary>

each unseen token gets $k/(c + k|V|)$, and there are $|V| - d$ of them. warm-up 3: $8000/(5000 + 10000) = 0.53$, so over half of P(· | the) goes to continuations never observed. for the rare word: $(10000 - 3)/(3 + 10000) = 0.9994$: the context is ignored almost entirely, and the 0.06% left for the 3 observed continuations is smaller than what a unigram model gives them. that's why add-one bigrams lose to unigrams.

</details>

3. **the EM update.** maximize $L(\lambda) = \sum_t\log\sum_j\lambda_jP_{tj}$ subject to $\sum_j\lambda_j = 1$. show that at a maximum with all $\lambda_j > 0$, $\lambda_j = \frac1N\sum_tr_{tj}$, so the optimum is a fixed point of the M step. then use Jensen's inequality to show one EM step never decreases $L$.

<details><summary>solution</summary>

lagrangian: $\partial/\partial\lambda_j\,[L - \mu(\sum\lambda - 1)] = \sum_tP_{tj}/m_t - \mu = 0$ with $m_t = \sum_i\lambda_iP_{ti}$. multiply by $\lambda_j$ and sum over $j$: $\sum_t\sum_j\lambda_jP_{tj}/m_t = N = \mu$. so $\sum_tP_{tj}/m_t = N$, and multiplying by $\lambda_j$: $\sum_tr_{tj} = N\lambda_j$. monotonicity: for the new weights $\lambda'$, $L(\lambda') - L(\lambda) = \sum_t\log\sum_jr_{tj}\frac{\lambda'_j}{\lambda_j} \ge \sum_t\sum_jr_{tj}\log\frac{\lambda'_j}{\lambda_j}$ (Jensen, $r_{t\cdot}$ sums to 1). the right side is $N\sum_j\lambda'_j\log(\lambda'_j/\lambda_j) = N\,\mathrm{KL}(\lambda'\,\|\,\lambda) \ge 0$, using $\sum_tr_{tj} = N\lambda'_j$.

</details>

4. **why unseen contexts must back off.** show that $\sum_w\tilde P_j(w \mid h_{[j]}) = 1$ for every context, seen or not, and hence that the interpolated conditional sums to 1. what would it sum to if unseen contexts contributed 0 instead?

<details><summary>solution</summary>

by induction on $j$. $\tilde P_0$ is add-k over $V$: it sums to $(N + k|V|)/(N + k|V|) = 1$. for $j \ge 1$, if $h_{[j]}$ was seen, $\sum_wc(h_{[j]}, w)/c(h_{[j]}) = 1$; if not, $\tilde P_j = \tilde P_{j-1}$, which sums to 1 by induction. the mixture sums to $\sum_j\lambda_j = 1$. with zeros instead, it would sum to $1 - \sum_{j \in U}\lambda_j$, where $U$ is the set of orders whose context is unseen: probability would leak, and perplexities would be too high (the model is penalized for mass it threw away) while samples could not even be drawn properly.

</details>

## experiment

`lab.py` runs your `NGramLM` and `tune_lambdas` on tiny Shakespeare, one line per sequence:

1. character perplexity against $n = 1, \dots, 8$, with add-k and with EM-tuned interpolation, on held-out lines and on training lines.
2. the same at the word level, orders 1–4, with add-1, add-0.01 and interpolation, with unknown words scored as `<unk>`.
3. samples from maximum-likelihood and interpolated character models, and how much of what they write is copied verbatim from the training text.

## check your understanding

1. why must `</s>` be predicted and scored, rather than just ignored like `<s>`?

<details><summary>answer</summary>

without it, a model gives probability to every prefix but never commits to a sentence being over, so the probabilities of complete sentences of all lengths sum to more than 1 (a sentence and all of its extensions each get their own mass). with `</s>` as a token, $P(\text{sentence})$ includes the probability of stopping, and the distribution over finite sequences is proper. it also lets a model learn sentence length, and lets `sample` end a sentence.

</details>

2. your interpolated model gives probability 0 to some held-out token. what went wrong?

<details><summary>answer</summary>

the unigram component should make that impossible: add-k with $k > 0$ is positive for every vocabulary token, and every unknown token maps to `<unk>`. so either the unigram isn't smoothed, the unigram weight $\lambda_0$ is exactly 0 (possible after EM if the unigram is never needed on held-out data, or set by hand), or the token isn't being mapped to `<unk>`.

</details>

3. EM puts almost no weight on the unigram, yet you can't remove it. why?

<details><summary>answer</summary>

the weight reflects how often the unigram is *the best explanation* of a held-out token, which is rare once there's context. but it's the only component that is never zero, so it's the safety net for tokens whose every higher-order estimate is zero (a known word in an unseen combination, or `<unk>`). with $\lambda_0 = 0$ exactly, one such token makes perplexity infinite. a tiny weight costs almost nothing elsewhere.

</details>

4. why would tuning the lambdas on the training data put all the weight on the highest order?

<details><summary>answer</summary>

on training data, every order-$(n-1)$ context was seen, and $P_\text{ML}$ of the highest order is the maximum-likelihood fit of that very data: it gives each training token at least as much probability as any lower-order estimate on average, and much more for rare contexts (a context seen once gives its continuation probability 1). EM would happily move all the weight there, which is exactly the overfitting the held-out set exists to detect.

</details>

5. two character models of different $n$ are scored on the same held-out lines. is it fair to compare their perplexities directly? what about a character model and a BPE-token model?

<details><summary>answer</summary>

same tokens, same number of predictions $N$: yes, directly. different tokenizations: no, $N$ differs and so does what one "step" predicts. convert both to bits per character (or per byte): total bits $= N\log_2\text{PP}$, divided by the number of characters. topic 02's lab does this.

</details>

6. the training perplexity of an order-8 character model is far below its held-out perplexity. is that a problem in itself?

<details><summary>answer</summary>

no: some gap is expected for any model fitted to data. the problem is choosing $n$ (or anything else) by training perplexity, which always prefers the largest model. what matters is held-out perplexity, and whether the gap grows (overfitting) as the model gets bigger. for count models, a big gap also predicts the copying you'll see in samples.

</details>

7. why is the perplexity of a model that predicts uniformly over $m$ tokens exactly $m$?

<details><summary>answer</summary>

every token has probability $1/m$, so the mean negative log-probability is $\log m$ and $\exp(\log m) = m$. this is the "effective branching factor" reading: a model with perplexity 5 is as uncertain, on average, as one choosing uniformly among 5 options at every step.

</details>

## common bugs

- **forgetting `</s>`, or counting `<s>` as a predicted token.** symptom: perplexities that don't match the tests by a small amount, a vocabulary with `<s>` in it, samples that never end a sentence.
- **padding with the wrong number of `<s>`.** symptom: counts for order $n - 1$ that miss the first tokens of every sequence, or contexts shorter than $n - 1$.
- **an unseen context contributing 0 to the mixture.** symptom: interpolated probabilities that sum to less than 1 (the normalization test fails), perplexities slightly too high.
- **tuning the lambdas on training data.** symptom: all the weight on the highest order, and a held-out perplexity worse than uniform weights.
- **taking the log of a zero component.** symptom: `math domain error` inside `tune_lambdas`. use the probabilities, not `logprob`, for the components.
- **summing a Counter at query time.** symptom: correct but painfully slow perplexities on real text. that's what `totals_` is for.

## going deeper

- Jurafsky & Martin (2025), *Speech and Language Processing*, 3rd ed. draft, chapter 3: n-gram language models, smoothing, interpolation and perplexity. [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/)
- Jelinek & Mercer (1980), "Interpolated estimation of Markov source parameters from sparse data", *Proceedings of the Workshop on Pattern Recognition in Practice*. the origin of interpolation with held-out EM.
- Chen & Goodman (1999), "An empirical study of smoothing techniques for language modeling", *Computer Speech & Language* 13(4): 359–394. the definitive comparison, and the case for modified Kneser–Ney. [doi:10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)
- Kneser & Ney (1995), "Improved backing-off for m-gram language modeling", *ICASSP*. lower orders that count contexts, not occurrences. [doi:10.1109/ICASSP.1995.479394](https://doi.org/10.1109/ICASSP.1995.479394)
- Brants, Popat, Xu, Och & Dean (2007), "Large language models in machine translation", *EMNLP*. n-grams on two trillion tokens, and "stupid backoff". [aclanthology.org/D07-1090](https://aclanthology.org/D07-1090/)
- Shannon (1951), "Prediction and entropy of printed English", *Bell System Technical Journal* 30(1): 50–64. [doi:10.1002/j.1538-7305.1951.tb01366.x](https://doi.org/10.1002/j.1538-7305.1951.tb01366.x)

## connections

- ← builds on: the chain rule and conditioning (ch 04 · topic 01), chapter 04's character n-gram capstone, cross-entropy and bits (ch 05 · topic 05), EM for mixtures (ch 11 · topic 02), validation splits (ch 07).
- → used in: tokenization, where your model compares tokenizers in bits per character (topic 02); the RNN and LSTM language models of topics 04–05; the capstone, where `mlfs.text.NGramLM` is the baseline the LSTM has to beat; perplexity for every language model in chapters 18 and 27.
