# language modeling · lab notes

numbers are for tiny Shakespeare (the cached file), measured with the reference solution: the first 80% of the text trains (31,497 lines, 892,000 characters), the second half of the next 10% is scored (2,015 lines, 55,774 characters counting each line's newline), and the first half of that 10% tunes the EM weights. offline (the synthetic stand-in that `verify.py` uses, independent random words), the curves have the same shapes but the context stops helping much sooner. the whole lab takes about 25 s on a laptop CPU.

## experiment 1: perplexity against n, characters

**what people usually predict:** more context is always better, so held-out perplexity falls steadily with $n$ for both models; or, from chapter 04, that it turns around for both.

**what happens:**

| n | add-k (k = 0.1), held-out | interpolated (EM), held-out | interpolated, training | EM weights, order 1 → n |
|---|---|---|---|---|
| 1 | 27.04 | 27.04 | 27.19 | 1.00 |
| 2 | 12.04 | 12.05 | 11.58 | 0.00 1.00 |
| 3 | 7.99 | 8.02 | 6.81 | 0.00 0.02 0.98 |
| 4 | 6.22 | 6.11 | 4.65 | 0.00 0.02 0.05 0.93 |
| 5 | 6.17 | 5.44 | 3.75 | 0.00 0.02 0.05 0.20 0.73 |
| 6 | 7.58 | 5.27 | 3.32 | 0.00 0.02 0.05 0.19 0.30 0.43 |
| 7 | 10.40 | 5.22 | 3.06 | 0.00 0.02 0.05 0.20 0.28 0.21 0.24 |
| 8 | 14.58 | 5.20 | 2.89 | 0.00 0.02 0.05 0.20 0.28 0.19 0.12 0.14 |

add-k is best at $n = 5$ and then more than doubles its perplexity by $n = 8$. the interpolated model never gets worse: it keeps improving slowly, to 5.20 at $n = 8$ (2.38 bits per character), already better than chapter 04's best add-k model on its validation block (5.68 at $n = 4$). the training perplexity falls all the way.

**why:** the number of distinct contexts explodes with $n$ (65 one-character contexts, 9,687 of length 3, 298,175 of length 7), so each is seen fewer times: at $n = 8$ the median 7-character context occurred with exactly one continuation in training, and 27.6% of the 7-character contexts in held-out text were never seen at all (1.6% at $n = 4$, 9.2% at $n = 6$). for an unseen context add-k returns the uniform distribution, $1/66$ for every character, which costs $\ln 66 = 4.2$ nats on that token; for a context seen once it puts most of its mass on the single observed continuation, and pays heavily when the held-out text continues differently. interpolation backs off to the longest context that *was* seen, so extra orders can only add information. the EM weights show where the trust goes: almost nothing on the unigram and bigram, most of the weight spread over orders 4–6, and steadily less on 7 and 8, whose estimates are too sparse to rely on.

## experiment 2: words, and the price of add-one

**what people usually predict:** that any context helps, so the add-1 bigram beats the unigram; and that interpolation is a modest improvement.

**what happens:** vocabulary 11,171 tokens; held-out 12,737 tokens, 5.7% of them out of vocabulary.

| smoothing | n = 1 | n = 2 | n = 3 | n = 4 |
|---|---|---|---|---|
| add-1 | 378.4 | 1022.4 | 4366.8 | 5492.7 |
| add-0.01 | 466.6 | 400.4 | 1925.9 | 3792.4 |
| interpolated (EM) | 378.4 | 211.2 | 203.6 | 203.1 |

the add-1 bigram is almost **three times worse than the unigram**, and every further order makes it worse. add-0.01 helps at $n = 2$ but collapses too. interpolation nearly halves the unigram's perplexity and stays flat after $n = 3$.

**why:** with $|V| = 11{,}171$, add-1 gives a context seen $c$ times with $d$ distinct continuations a total of $(|V| - d)/(c + |V|)$ for things it has never seen. "the" occurs 5,168 times with 1,959 distinct continuations, so 56% of P(· | the) goes to the 9,212 words never seen after it. a word seen 3 times gives 99.9% of its mass to the unseen: the observed continuations get less than the unigram would give them, so the bigram is worse than ignoring the context. the higher orders have even sparser contexts, and get worse still. note also the add-0.01 *unigram* (466.6 vs 378.4): with a tiny $k$, `<unk>` gets almost no probability, and the 5.7% of held-out tokens that are unknown each cost a fortune. the numbers can't be compared with experiment 1's: a word is a much bigger prediction than a character. converted to bits per character (total bits, 14,752 word tokens including one `</s>` per line, divided by 55,774 characters), the interpolated word model is at 2.03 bits per character against the character model's 2.38: the word model looks better, but the figure flatters it, since an unknown word costs one `<unk>` prediction instead of being spelled out.

## experiment 3: what the models write, and how much of it is copied

**what people usually predict:** a few percent at most for every model; or, for those who saw chapter 04's samples, a lot for high orders.

**what happens:** fraction of 12-character windows that occur verbatim in the training text (mean of 3 samples of 300 characters):

| order | maximum likelihood (all weight on order n) | EM interpolation |
|---|---|---|
| 2 | 0% | 0% |
| 3 | 0% | 0% |
| 4 | 1% | 1% |
| 6 | 9% | 3% |
| 8 | 31% | 5% |

the order-8 maximum-likelihood model writes almost-correct Shakespeare ("No better; I will flatter, you'll leave the grains: Was it more mock'd with ruthless senator speak."), and nearly a third of it is lifted directly from the training text. the interpolated order-8 model, which has the better held-out perplexity, copies only 5%, and reads like word salad with Shakespearean spelling ("O, God show the time brothyself and myself, for's patch my ble masks").

**why:** a 7-character context in this text has 1.4 distinct continuations on average, and most have exactly one, so a maximum-likelihood order-8 model, once it's inside a rare context, can only follow the training text. it writes fluently because it is reciting. the interpolated model mixes in lower orders at every step (about half its weight is on orders 1–5), so it leaves the training text often, but it then loses the thread, because no component knows anything beyond 7 characters. good held-out perplexity, local fluency and originality pull in different directions for count-based models: getting all three needs a model that *generalizes across contexts* (similar contexts should predict similarly, topic 03's embeddings) and that can carry information further back than a fixed window (topics 04–06).
