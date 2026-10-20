# 17 · sequence models · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. what is perplexity, and why does a model that spreads its probability uniformly over $m$ tokens have perplexity exactly $m$?

<details><summary>answer</summary>

$\text{PP} = \exp(-\frac1N\sum_t\log P(w_t \mid h_t))$, the exponential of the average negative log-likelihood per token (the cross-entropy in nats). a uniform model gives every token probability $1/m$, so the average is $\log m$ and the perplexity is $m$: an effective branching factor.

</details>

2. why does add-one smoothing make a word bigram model *worse* than a unigram model on held-out text, while it works fine for characters?

<details><summary>answer</summary>

a context seen $c$ times with $d$ distinct continuations gives the unseen ones a total of $(|V| - d)/(c + |V|)$. with $|V| = 10{,}000$ words that's most of the mass for nearly every context (99.9% for a word seen 3 times), so the observed continuations get less than the unigram would give them. with 65 characters, $|V|$ is small next to most context counts, and the pseudo-counts barely matter.

</details>

3. in interpolation, what happens to the weight of an order whose context was never seen, and why does it matter?

<details><summary>answer</summary>

that order's estimate is replaced by the estimate of the order below (backoff inside the mixture), so every component stays a proper distribution and the mixture sums to 1. treating the unseen context's estimate as 0 would leak probability and inflate perplexity.

</details>

4. why must interpolation weights be tuned on held-out data, and what does EM maximize?

<details><summary>answer</summary>

on training data the highest order fits best (it has seen every context; a context seen once gives its continuation probability 1), so all the weight would go there. EM maximizes the held-out log-likelihood $\sum_t\log\sum_j\lambda_jP_{tj}$ over the weights, treating the order that "generated" each token as a latent variable; the objective is concave in $\lambda$, so EM finds the global optimum.

</details>

5. why byte-level BPE rather than character-level BPE, and what does it cost?

<details><summary>answer</summary>

the base vocabulary is the 256 byte values, so every string in every language encodes (nothing is unknown). the cost: characters outside what the training text covered are spelled out as 2–4 byte tokens (Chinese takes 3 tokens per character with an English-trained tokenizer), and a token can be part of a character, so decoding must join the bytes before decoding UTF-8.

</details>

6. why does GPT-2-style BPE pre-tokenize the text before counting pairs?

<details><summary>answer</summary>

so merges never cross chunk boundaries (letters vs digits vs punctuation vs whitespace): otherwise frequent pairs that straddle words ("e t", "d.") would become tokens, wasting vocabulary on meaningless pieces. it also makes training fast: a text has far fewer distinct chunks than characters, so each distinct chunk is counted once, weighted by its frequency.

</details>

7. what does skip-gram with negative sampling optimize, and what does it converge to (Levy & Goldberg)?

<details><summary>answer</summary>

for each real (center, context) pair and $K$ noise words from the unigram$^{3/4}$ distribution, a logistic loss that scores real pairs high and noise pairs low with the dot product $u_c^\top v_w$. at the optimum, $u_c^\top v_w = \text{PMI}(w, c) - \log K$: the model implicitly factorizes a shifted PMI matrix.

</details>

8. why do antonyms like "love" and "hate" get similar word2vec vectors?

<details><summary>answer</summary>

the vectors encode which contexts a word predicts, and antonyms share almost all of their contexts ("I ___ thee"). distributional similarity is relatedness of usage, not sameness of meaning.

</details>

9. write the BPTT recursion for the gradient flowing into $h_t$, and explain why the weight gradients are sums over time.

<details><summary>answer</summary>

$g_t = \delta_t + (\partial h_{t+1}/\partial h_t)^\top g_{t+1}$ with $\partial h_{t+1}/\partial h_t = \operatorname{diag}(1 - h_{t+1}^2)W_{hh}$ for tanh. the same weights are used at every step, so by the multivariable chain rule their gradient is the sum of the gradients each use would get as a separate copy.

</details>

10. give the bound that explains vanishing gradients in an RNN. is $\|W_{hh}\| > 1$ enough for explosion?

<details><summary>answer</summary>

$\|\partial h_T/\partial h_t\| \le (\gamma\sigma_\max)^{T-t}$ with $|\phi'| \le \gamma$ and $\sigma_\max$ the largest singular value of $W_{hh}$: below 1 the gradient vanishes exponentially. $\sigma_\max > 1$ is necessary for explosion, not sufficient: the derivative factors can shrink the growing directions.

</details>

11. truncated BPTT carries the hidden state across chunks but detaches it. what can the model *use* from before the chunk, and what can it *learn*?

<details><summary>answer</summary>

the forward pass sees an unbroken stream, so the state can carry information from arbitrarily far back. but gradients stop at the chunk boundary, so the model can only learn to store information for about the truncation length; a dependency longer than that gets no training signal.

</details>

12. why does the LSTM's cell state let gradients survive many steps? what initialization helps, and why?

<details><summary>answer</summary>

the cell update $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ is additive, so along the cell path $\partial c_t/\partial c_{t-1} = \operatorname{diag}(f_t)$: a gate value, not a matrix times a squashing derivative, and it can sit near 1. initializing the forget-gate bias to 1 (or more) makes the cell remember from the start; with bias 0 the cell halves every step and the gradient path is as weak as an RNN's.

</details>

13. how does the GRU keep information without a separate cell state?

<details><summary>answer</summary>

$h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$: with the update gate $z_t \approx 1$, a unit copies its state exactly and ignores the candidate. one gate couples "keep" and "don't write", where the LSTM has separate forget and input gates.

</details>

14. in Bahdanau attention, how are padding positions handled, and why before the softmax?

<details><summary>answer</summary>

their scores are set to $-\infty$ before the softmax, so their weights are exactly 0 and the remaining weights sum to 1. multiplying the weights by the mask afterwards would leave weights that sum to less than 1, shrinking the context.

</details>

15. why does beam search normalize scores by length, and why doesn't it help a model that doesn't know the answer?

<details><summary>answer</summary>

every token adds a negative log-probability, so raw sums favour short outputs; dividing by $L^\alpha$ compares per-token log-probability. beam search finds outputs of higher *model* probability; if the model's distribution puts its mass on wrong answers, a better search finds a more probable wrong answer.

</details>

## 2. by hand

1. **an interpolated bigram.** train on two sequences, `a b a` and `b b`, with $n = 2$, $\lambda = (0.5, 0.5)$ and an add-1 unigram (vocabulary `</s>`, `<unk>`, `a`, `b`). compute $P(b \mid a)$, and the perplexity of the sequence `a b`.

<details><summary>solution</summary>

unigram counts (predicted tokens): a 2, b 3, `</s>` 2, total 7; add-1 over $|V| = 4$: $\tilde P_0(a) = 3/11$, $\tilde P_0(b) = 4/11$, $\tilde P_0(\texttt{</s>}) = 3/11$. bigram counts: after `<s>`: a 1, b 1; after a: b 1, `</s>` 1; after b: a 1, b 1, `</s>` 1. so $P(b \mid a) = 0.5 \cdot 4/11 + 0.5 \cdot 1/2 = 0.4318$. for `a b`: $P(a \mid \texttt{<s>}) = 0.5 \cdot 3/11 + 0.5 \cdot 1/2 = 0.3864$, $P(b \mid a) = 0.4318$, $P(\texttt{</s>} \mid b) = 0.5 \cdot 3/11 + 0.5 \cdot 1/3 = 0.3030$. product $0.05055$, perplexity $0.05055^{-1/3} = 2.704$.

</details>

2. **three BPE merges.** train a byte-level BPE with GPT-2 pre-tokenization on `"abab cab"` with `vocab_size=259` (a = 97, b = 98, c = 99, space = 32). give the merges in order, then encode `"cabab"`.

<details><summary>solution</summary>

chunks: `"abab"` and `" cab"`. pair counts: (97, 98) occurs 3 times (twice in the first chunk, once in the second), more than anything else → 256. now the chunks are `[256, 256]` and `[32, 99, 256]`: (256, 256), (32, 99) and (99, 256) each occur once; the tie goes to the smallest pair, (32, 99) → 257. then `[256, 256]` and `[257, 256]`: (256, 256) and (257, 256) tie, smallest is (256, 256) → 258. merges: (97, 98), (32, 99), (256, 256). `"cabab"` is one chunk: bytes 99 97 98 97 98 → merge rank 0 gives 99 256 256 → rank 2 gives `[99, 258]` ((99, 256) is not a merge).

</details>

3. **an LSTM step and a beam step.**
   a. a 1-unit LSTM with every input weight 1, every recurrent weight 0, all biases 0 except the forget gate's, which is 1. from $h = 0$, $c = 0.5$, the input is $x = 1$. compute $i, f, g, o$, the new $c$ and $h$.
   b. beam search with `beam_size=2` (tokens `eos` = 0, a = 1, b = 2, c = 3). after the first step the beam holds `a` (probability 0.5) and `b` (0.3). the next-token distributions are: after `a`: eos 0.2, a 0.2, b 0.3, c 0.3; after `b`: eos 0.7, a 0.1, b 0.1, c 0.1. which two extensions are kept, which are finished, and how many hypotheses are alive afterwards?

<details><summary>solution</summary>

a. $i = \sigma(1) = 0.7311$, $f = \sigma(1 + 1) = 0.8808$, $g = \tanh(1) = 0.7616$, $o = \sigma(1) = 0.7311$. $c' = 0.8808 \times 0.5 + 0.7311 \times 0.7616 = 0.4404 + 0.5568 = 0.9972$; $h' = 0.7311\tanh(0.9972) = 0.7311 \times 0.7604 = 0.5559$.

b. candidates (probabilities): `b eos` 0.21, `a b` 0.15, `a c` 0.15, `a eos` 0.10, `a a` 0.10, `b a`/`b b`/`b c` 0.03. the best two: `b eos` (0.21), finished; then `a b` and `a c` tie at 0.15, both from the same hypothesis, so the smaller token id wins: `a b`. one hypothesis is finished, one (`a b`) alive, and the next step keeps only $2 - 1 = 1$ extension.

</details>

## 3. blind rebuild (60 minutes, no notes)

open a fresh copy of `02-tokenization/exercise.py` and implement, from memory, `pair_counts`, `merge_pair`, `pretokenize` (you may look up the regular expression in the setup cell), and `BPETokenizer` (`__init__`, `train`, `encode_chunk`, `encode`, `decode`, `save`, `load`). the simple trainer that recounts every pair after every merge is fine. pass with

```bash
pytest 02-tokenization/test_exercise.py -k "stage2 or stage3 or stage4 or stage5"
```

the tests compare your merges, merge for merge, with a reference trainer, including the tie rule and overlapping pairs, and round-trip random Unicode through your encoder.

## 4. readiness

move on when you score at least 80% on sections 1 and 2 together and the blind rebuild passes. if you're using this as a placement test: the same thresholds let you skip the chapter; chapter 18 assumes you can train and evaluate a language model, read perplexity in bits per character, and use `mlfs.text.BPETokenizer`.
