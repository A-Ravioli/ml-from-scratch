# decoding & inference · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, 4 threads), tiny Shakespeare characters from the cache, a target `Llama` of 4 layers and width 128 (734,464 parameters, 500 steps), one run. timings are best of two on a machine shared with other jobs: differences under ~10% are noise. offline the models are trained briefly on synthetic text and the numbers mean little.

## experiment 1: how to sample

**what people predict.** that greedy decoding is the "best" (most accurate) text and temperature trades quality for diversity smoothly.

**what happens.**

| recipe | distinct word pairs | log-prob per char (target) |
|---|---|---|
| greedy (T = 0) | **0.24** | **−0.88** |
| greedy + repetition penalty 1.3 | 0.89 | −0.87 |
| T = 0.7 | 0.99 | −1.24 |
| T = 1.0 | 1.00 | −1.55 |
| T = 1.5 | 1.00 | −2.57 |
| T = 1.5 + min-p 0.1 | 1.00 | −1.38 |

greedy decoding falls into a loop within a few words ("I will the shall the shall the shall ..."): only a quarter of its word pairs are distinct, and the target rates it as the most likely text of all. a repetition penalty of 1.3 breaks the loop at almost no cost in likelihood. at temperature 1.5 the text fills with misspelled words (log-probability −2.57 per character); adding min-p 0.1 brings it back to −1.38, better than plain temperature 1.0.

**why.** the probability of a sequence is the product of its token probabilities, and a repeated phrase becomes more probable each time it occurs (the model has learned that text repeats), so greedy decoding's local maximum is a self-reinforcing loop: highest likelihood, lowest quality. that's Holtzman et al.'s (2020) observation that likelihood and human judgement part ways at the top. high temperature lifts the long tail of individually unlikely characters; each one is rare, but at 1.5 there are so many that some are picked every few characters, and each mistake derails what follows. min-p cuts the tail relative to the top token, so it removes exactly the junk that temperature promoted, while keeping the flattened choice among plausible tokens.

## experiment 2: speculative decoding against draft quality

**what people predict.** that the best draft gives the largest speedup, and that speculative decoding beats ordinary decoding.

**what happens.** 150 characters at temperature 1, $k = 4$:

| draft | parameters | alpha | acceptance rate | tokens per target call | theory | tokens/s |
|---|---|---|---|---|---|---|
| bigram counts | 0 | 0.50 | 0.24 | 1.95 | 1.95 | **1,410** |
| tiny (1 layer, width 32) | 13,696 | 0.61 | 0.38 | 2.50 | 2.33 | 1,054 |
| small (2 layers, width 64) | 96,640 | 0.75 | 0.57 | **3.26** | 3.03 | 993 |

target alone: 818 tokens/s without a cache, **1,534 with it**. (an earlier run: 831 and 1,513, bigram 1,391.)

tokens per target call follow $(1 - \alpha^{k+1})/(1 - \alpha)$ closely, so measuring $\alpha$ on held-out text predicts how much verification work each draft saves. the measured values sit slightly above the formula because $\alpha$ isn't constant: it's high in predictable stretches (the rest of a common word, a newline after a speaker name) and low elsewhere, and since the formula is convex in $\alpha$, the average of $f(\alpha)$ over contexts exceeds $f$ of the average $\alpha$ (Jensen's inequality); acceptances also come in runs within a round. but the wall-clock order is reversed: the draft that saves the most target calls (small, 3.26 per call) is the slowest, and the worst draft (a table lookup) is the fastest, 1.7× the uncached target. and none of them beats the target with a KV cache.

**why.** the speedup is tokens per call divided by the cost of a round, and the round includes $k$ draft passes. the bigram draft costs nothing; the small Llama costs a noticeable fraction of the target (both are tiny, and on a CPU a forward pass of either is dominated by per-call overhead, not by weight bandwidth), so its better guesses are paid for by its own forward passes. Leviathan et al.'s 2–3× needs a target that's large and memory-bound, a draft 10–100× cheaper, and caches for both. your `speculative_decode` recomputes both models over the whole context at every step, which is what a KV cache avoids: here the cache alone is worth 1.9×, more than any draft gives without it. production systems (vLLM, TensorRT-LLM) combine them, keeping KV caches for both models and truncating the target's cache past a rejected token. the acceptance rate (accepted / proposed) is well below $\alpha$ because drafts after the first rejection in a round are counted as proposed and never examined.

## experiment 3: constrained decoding is not conditioning

**what people predict.** that constrained decoding picks each valid name about as often as the model would.

**what happens.** after a blank line in held-out text, free sampling produced one of the eight *Romeo and Juliet* names in 8% of 3,000 samples (225 valid). among those, Romeo (32%) and Juliet (25%) dominate; Tybalt and Nurse never appear. constrained decoding (300 samples) gives Romeo 12%, Juliet 6%, Tybalt 10%, Nurse 5%, Capulet 20%, Friar Laurence 18%. the total variation distance is **0.39**; two samples of these sizes from one distribution would differ by about 0.07.

**why.** in this layout every valid name is fixed by its first letter, so constrained decoding's choice is made entirely at the first character: the mask allows R, J, M, B, T, N, C and F and renormalizes the model's probabilities over them. measured separately on the same model and prompt, those first-letter probabilities are C 0.084, F 0.056, B 0.051, M 0.049, R 0.046, T 0.030, J 0.025, N 0.019; renormalized, C 23%, F 16%, B 14%, M 14%, R 13%, T 8%, J 7%, N 5%, which is almost exactly the constrained-decoding histogram (20, 18, 15, 13, 12, 10, 6, 5%). the model likes C because many speakers in the corpus start with C (Coriolanus, Camillo, Capulet...); among the valid names only Capulet does, so it inherits all of C's probability, while conditioning on validity weighs it by how often the model actually continues C into CAPULET. rejection sampling uses the probability of the whole string, $P(\text{R})\,P(\text{OMEO:\\n} \mid \text{R})$, and the model completes R into ROMEO far more readily than T into TYBALT. so constrained decoding over-weights names whose prefixes the model likes for other reasons. it's exactly right only when every allowed prefix is completed validly with the model's own probability 1 (for instance, a single-token label set). sampling the true conditional requires knowing, for each prefix, the probability that the model completes it validly: intractable in general, approximated by methods like grammar-aligned decoding (Park et al., 2024), or computed exactly here by rejection, at the cost of throwing away 92% of the samples.
