# 04 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass; $n$ and $k$ chosen on validation, test used once | all pass, plus your own extra checks (e.g. every `next_token_probs` sums to 1 on real contexts; your perplexity matches a direct computation on a tiny text; generated text has the training text's character frequencies) |
| **results** | no baseline | perplexity reported against the uniform baseline | test perplexity on tiny Shakespeare below the unigram and bigram baselines, in perplexity and bits per character, with all three baselines | plus an uncertainty estimate for the test perplexity (e.g. perplexity on 10 disjoint chunks of the test block, with mean ± standard error), and a clear statement of which differences between settings exceed it |
| **analysis** | none | describes the plots | explains the train/validation curve with the number of distinct contexts vs training size, and one ablation (k = 0, split strategy, or word level) with a reason | several ablations; samples at several temperatures with the greedy failure explained; the copying check used to discuss memorization; links back to the chapter (conditional distributions, Markov chains, expected log loss) |
| **clarity** | hard to follow | readable | clear figures with labeled axes, and a narrative that answers the six prompts | a peer could reproduce every number from the write-up alone (split, $n$, $k$, config, runtime) |

## what graders look for

- **choosing on the test set.** picking $(n, k)$ by test perplexity turns the test set into a validation set. choose on validation, report test once. a write-up that tunes on test scores at most 1 on results.
- **comparing on the same tokens.** an order-6 model can't score the first 5 characters. if each model scores a different set of positions, the comparison is unfair. the notebook passes the same `start` to every model; keep doing that in your own experiments.
- **the right explanation of the U-curve.** "higher $n$ overfits" is a description. the explanation counts: with $V = 65$ characters there are up to $65^5 \approx 10^9$ possible order-6 contexts, but only about 900,000 training positions, so most contexts are seen once or never, and their conditional distributions are mostly pseudo-counts. smaller $k$ helps (less smoothing toward uniform) until there's nothing left to smooth with.
- **honest uncertainty.** a perplexity is an exponentiated average of per-character losses. those losses are not independent (neighbouring characters share contexts), so the naive $\sigma/\sqrt n$ standard error is too optimistic. chunk-level error bars are the simple honest fix. and even honest error bars only describe sampling noise *within* one distribution: on tiny Shakespeare, validation and test differ by about 25% because the test block is different plays. a strong write-up notices that the gap is far outside any i.i.d. error bar and names the cause (distribution shift between contiguous blocks), rather than calling the test number "noise" or "overfitting to validation".
- **greedy decoding.** "temperature 0 gives the most likely text" is true and still produces "the the the the". the best write-ups connect this to topic 04's lab (the mode of a high-dimensional distribution is not a typical sample) and to the Markov chain view (greedy decoding is a deterministic chain, and a finite deterministic chain must eventually cycle).
