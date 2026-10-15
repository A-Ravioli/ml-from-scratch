# seq2seq & attention

## why this matters

so far every model in this chapter predicts the next token of the same sequence. many problems instead map one sequence to *another*: a French sentence to an English one, speech to text, a question to an answer, "tuesday, march 3, 1987" to "1987-03-03". the input and output have different lengths, different vocabularies, and no fixed alignment between positions. the **encoder–decoder** (Sutskever, Vinyals & Le, 2014; Cho et al., 2014) handles this with two RNNs: one reads the input into a state, the other writes the output from it, one token at a time.

the first version had a flaw you can feel: the whole input had to squeeze through one fixed-size vector, the encoder's last state, and translation quality fell off a cliff for long sentences. **attention** (Bahdanau, Cho & Bengio, 2015) removed the bottleneck: at every output step, the decoder computes a weighted average of *all* the encoder's states, with weights it chooses itself, so it can look back at whichever input positions matter right now. that one idea, a learned, differentiable lookup, is the seed of the transformer (chapter 18), where attention replaces recurrence entirely. this topic builds the encoder–decoder with and without attention, then the decoding side: greedy decoding and **beam search**, which chapter 27 reuses for language models.

## warm-up: predict first

1. an encoder GRU has 128 units. how many numbers does a decoder without attention get to know about a 50-word input sentence?
2. attention weights come from a softmax over input positions. what should happen to the weights of padding positions, and how do you make sure?
3. during training the decoder is fed the *correct* previous token, during generation its *own* previous prediction. can that mismatch hurt?
4. greedy decoding takes the most likely token at every step. does that find the most likely output sequence?
5. a beam search scores outputs by their total log-probability. which outputs does that favour, short or long?

## prerequisites check

- [ ] you built the GRU and know `nn.GRU` and `nn.GRUCell` (now unlocked) (→ topic 05)
- [ ] you can write the softmax and its stable form (→ ch 08 · topic 05)
- [ ] you know embeddings and cross-entropy over a vocabulary (→ topics 03–04)
- [ ] you know masking with booleans and `masked_fill` in PyTorch (→ ch 15 · topic 07)
- [ ] you know the chain rule of probability for sequences (→ topic 01)

## core ideas

### idea 1: attention is a soft, learned lookup

you have a set of **keys** $k_1, \dots, k_T$ (here: the encoder's states, one per input position) and a **query** $q$ (here: the decoder's current state). attention scores how relevant each key is to the query, turns the scores into weights that sum to 1, and returns the weighted average:

$$e_j = \text{score}(q, k_j),\qquad \alpha_j = \frac{\exp(e_j)}{\sum_{l}\exp(e_l)},\qquad c = \sum_{j}\alpha_jk_j.$$

it's a dictionary lookup made differentiable: instead of returning the one value whose key matches, it returns a blend, weighted by how well each key matches. because every step is smooth, the network can *learn* what to look for by gradient descent, with no supervision about which positions are relevant.

Bahdanau et al. score with a one-hidden-layer network, **additive attention**:

$$e_j = v^\top\tanh(W_qq + W_kk_j),$$

with $W_q$ and $W_k$ mapping both to a common hidden size and $v$ a vector. chapter 18 uses the cheaper **dot-product** score $q^\top k_j/\sqrt{d}$; additive attention allows the query and keys to have different sizes and was the original.

**masks.** a batch of sources padded to the same length has keys at padding positions that mean nothing. a mask (`True` = may attend, the course convention) sets their scores to $-\infty$ before the softmax, so their weights are exactly 0 and the remaining weights renormalize over the real positions (warm-up 2). multiplying the weights by the mask *after* the softmax would leave weights that no longer sum to 1.

`BahdanauAttention(query_dim, key_dim, hidden_dim)` holds `W_q` (`nn.Linear(query_dim, hidden_dim, bias=False)`), `W_k` (`nn.Linear(key_dim, hidden_dim)`, with the one bias the tanh needs) and `v` (`nn.Linear(hidden_dim, 1, bias=False)`). `forward(query, keys, mask=None)` takes $(B, D_q)$, $(B, T, D_k)$ and an optional $(B, T)$ bool mask and returns `(context (B, D_k), weights (B, T))`. broadcast the projected query over positions with `unsqueeze(1)`, and get the context with one `torch.bmm`.

> **▶ build: stage 1.** implement `BahdanauAttention`.

### idea 2: the encoder–decoder, and teacher forcing

**the encoder** embeds the source tokens and runs a GRU over them (`nn.GRU`, batch-first: unlocked by topic 05): its outputs, one $H$-dimensional state per source position, are the attention keys. **the decoder** is a `nn.GRUCell` stepped once per output token. it starts from the encoder's state at the source's last *real* token (with padding at the end, `src_mask.sum(1) - 1` gives its index; for unpadded sources it's the last position).

one decoder step with attention, in Bahdanau's order:

1. attend with the **previous** decoder state $s_{t-1}$ as the query: $c_t, \alpha_t = \text{attn}(s_{t-1}, \text{keys}, \text{mask})$.
2. update the state from the previous output token and the context: $s_t = \text{GRUCell}([\text{emb}(y_{t-1}); c_t],\ s_{t-1})$.
3. predict: $\text{logits}_t = W_o[s_t; c_t] + b_o$.

without attention, the step is just $s_t = \text{GRUCell}(\text{emb}(y_{t-1}), s_{t-1})$ and $\text{logits}_t = W_os_t + b_o$: everything the decoder knows about the source came through the initial state (warm-up 1: 128 numbers, however long the input).

**teacher forcing.** to train, feed the decoder the *correct* previous token at every step, `tgt_in` = `[bos, y_1, ..., y_{T-1}]`, and score its predictions against `tgt_out` = `[y_1, ..., y_T]` (ending with `eos`) with cross-entropy. by the chain rule, the summed cross-entropy is exactly $-\log P(y \mid x)$ (on paper 1), and every step trains in parallel with its correct context. the logits at step $t$ depend only on `tgt_in[:, :t+1]`, never on later targets; the tests check that by changing future tokens.

`Seq2Seq(src_vocab_size, tgt_vocab_size, *, embed_dim=64, hidden_dim=128, attention=True)` has `src_embed`, `tgt_embed`, `encoder`, `decoder`, `attn` (`None` without attention) and `out`. split the work into two methods that decoding reuses: `encode(src, src_mask=None)` returns `(keys, initial state, mask)` (an all-`True` mask if none was given), and `decode_step(y_prev, state, keys, src_mask)` returns `(logits (B, V), new state, attention weights or None)`. `forward(src, tgt_in, *, src_mask=None)` encodes once, loops `decode_step` over the positions of `tgt_in`, returns the $(B, T, V)$ logits, and stores the attention weights as `last_attention`, $(B, T, S)$, detached (or `None` without attention).

> **🔮 predict:** in the lab's date task, the year comes last in most source formats and the day first. which field will a model *without* attention learn first?

> **▶ build: stage 2.** implement `Seq2Seq`: `__init__`, `encode`, `decode_step`, `forward`.

### idea 3: generating: greedy decoding

at test time there's no correct previous token. the decoder must feed its own predictions back in: start from `bos`, take the most likely token, feed it as the next input, and stop at `eos` or after `max_len` tokens. that's **greedy decoding**. `greedy_decode(model, src, bos_id, eos_id, *, max_len=50, src_mask=None)` runs it for a whole batch at once (encode once, then one `decode_step` per output position) and returns one list per source row, `[bos, ..., eos]` (no `eos` if `max_len` ran out). rows that have already produced `eos` simply stop collecting tokens while the others continue. run it under `torch.no_grad()` in eval mode, and restore the model's mode afterwards.

two caveats. first, the model is trained on correct prefixes and tested on its own, possibly wrong ones (**exposure bias**, warm-up 3): an early mistake puts it in a situation it never saw in training. for a well-trained model on a narrow task this rarely matters; for open-ended generation it's one reason outputs drift. second, greedy is not optimal (warm-up 4): the most likely *first* token can lead to a less likely *sequence* than a second-choice first token whose continuations are all confident. that's what beam search is for.

> **▶ build: stage 3.** implement `greedy_decode`.

### idea 4: beam search

**beam search** keeps the `beam_size` best partial outputs (hypotheses) instead of one. at every step it extends each alive hypothesis by every token, scores each extension by its total log-probability, and keeps the best. your `beam_search(step_fn, bos_id, eos_id, *, beam_size=5, max_len=50, length_penalty=1.0)` is model-agnostic: `step_fn(prefixes)` takes a list of token lists and returns next-token log-probabilities, one row per prefix (a tensor or an array). this contract is what lets chapter 27 decode a GPT with it. the exact rules, which the tests pin down:

1. start with one alive hypothesis, `[bos]`, with score 0, and no finished ones.
2. while some hypothesis is alive, fewer than `beam_size` have finished, and fewer than `max_len` tokens have been generated: call `step_fn` on the alive prefixes once; form every extension's cumulative log-probability; keep the best `beam_size - len(finished)` extensions (the beam **shrinks** as hypotheses finish); ties go to the earlier hypothesis, then the smaller token id (a stable sort of the flattened scores does this); skip extensions with log-probability $-\infty$. an extension ending in `eos` is finished; the others stay alive.
3. at the end, the finished hypotheses and any still alive (cut off at `max_len`) are ranked by the **length-normalized** score

$$\text{score} = \frac{\sum_t\log p(y_t \mid y_{<t}, x)}{L^{\alpha}},$$

where $L$ is the number of tokens after `bos` (`eos` included) and $\alpha$ is `length_penalty`. return the best `beam_size` as `(tokens, score)` pairs, best first.

why normalize? every token adds a negative log-probability, so the raw sum always prefers shorter outputs (warm-up 5), and a beam search on raw scores tends to stop early. dividing by $L^\alpha$ compares average per-token log-probabilities ($\alpha = 1$), $\alpha = 0$ recovers the raw sum, and values in between (Wu et al., 2016, use about 0.6) trade the two. with `beam_size=1`, beam search is greedy decoding; with a beam wide enough to never prune, it's exhaustive search, which is how the tests check it.

beam search finds sequences of higher probability, which isn't always what you want. on open-ended text it famously produces bland, repetitive output: the most probable continuation of a story is rarely a good one (Holtzman et al., 2020). it shines when the output is tightly constrained by the input, as in translation or speech recognition.

> **🔮 predict:** for a model that already gets 100% of the test dates right with greedy decoding, what can beam search add? and for a model that hasn't learned the day yet?

> **▶ build: stage 4.** implement `beam_search`.

## worked example

**attention by hand.** two keys, $k_1 = (1, 0)$ and $k_2 = (0, 1)$, with scores $(1, 0)$ after the $v^\top\tanh(\cdot)$: $\alpha = \big(\tfrac{e^1}{e^1 + e^0}, \tfrac{e^0}{e^1 + e^0}\big) = (0.7311, 0.2689)$ and $c = (0.7311, 0.2689)$. with a mask that forbids position 1, the score becomes $(-\infty, 0)$, so $\alpha = (0, 1)$ and $c = k_2$ exactly. with every weight of the attention module set to zero, every score is 0, the weights are uniform, and the context is the mean of the keys. the tests check the masked and the all-zero cases.

**beam search by hand.** tokens `eos` = 0, `a` = 1, `b` = 2. the first token is `a` with probability 0.6 or `b` with 0.4 (`eos` impossible). after `a`: `eos` 0.4, `a` 0.3, `b` 0.3. after `b`: `eos` 0.9, `a` 0.05, `b` 0.05.

- **greedy:** `a` (0.6), then `eos` (0.4): $P = 0.24$, score $\ln 0.24/2 = -0.714$.
- **beam 2:** after step 1 the beam holds `a` ($\ln 0.6$) and `b` ($\ln 0.4$). step 2 extends both: `b eos` $0.36$, `a eos` $0.24$, `a a` and `a b` $0.18$, `b a` and `b b` $0.02$. the best two are `b eos` and `a eos`, both finished, so the search stops. winner: `b eos` with $P = 0.36$, score $\ln 0.36/2 = -0.511$.

greedy committed to `a` because it was the better first step; the beam kept `b` alive long enough to see that its continuation is far more confident.

**a decoding loop.** with `bos` = 1 and `eos` = 2, a model whose output layer always favours `eos` returns `[[1, 2], [1, 2]]` for a batch of two; one that never produces `eos` returns `[1]` followed by exactly `max_len` tokens.

## on paper

1. **teacher forcing is maximum likelihood.** show that the sum over steps of the cross-entropy between the decoder's distribution and the target token, under teacher forcing, equals $-\log P(y_1, \dots, y_T \mid x)$.

<details><summary>solution</summary>

the decoder at step $t$, fed the correct prefix $y_{<t}$, outputs $P(\cdot \mid y_{<t}, x)$, and the cross-entropy against the one-hot target is $-\log P(y_t \mid y_{<t}, x)$. summing over $t$ and using the chain rule, $\sum_t -\log P(y_t \mid y_{<t}, x) = -\log\prod_tP(y_t \mid y_{<t}, x) = -\log P(y \mid x)$. so teacher forcing is exactly maximum-likelihood training of the sequence model; the catch is only at test time, when the prefix comes from the model itself.

</details>

2. **masking before the softmax.** with scores $e$ and a mask $m$, compare $\text{softmax}(e \text{ with masked entries} = -\infty)$ with $\text{softmax}(e) \odot m$. which sums to 1? show that the first equals the second renormalized.

<details><summary>solution</summary>

the first gives $\alpha_j = m_je^{e_j}/\sum_lm_le^{e_l}$, which sums to 1 over the allowed positions. the second gives $m_je^{e_j}/\sum_le^{e_l}$, which sums to $\sum_lm_le^{e_l}/\sum_le^{e_l} < 1$ whenever a masked position has finite score: the context would shrink towards 0 by however much weight the padding would have had. dividing the second by its sum gives the first.

</details>

3. **the cost of beam search.** with vocabulary $V$, beam $k$ and output length $L$, how many `step_fn` rows and how many candidate scores does the search evaluate? why is it much cheaper than exhaustive search?

<details><summary>solution</summary>

at most $k$ prefixes per step, so at most $kL$ rows of $V$ log-probabilities, and $kLV$ candidate scores to rank (a full sort costs $O(kV\log(kV))$ per step; a partial sort is cheaper). exhaustive search scores all $V^L$ sequences: with $V = 32{,}000$ and $L = 20$ that's $10^{90}$, against $5 \times 20 \times 32{,}000 = 3.2$ million for a beam of 5.

</details>

4. **why the raw sum prefers short outputs.** a model gives the output `eos` probability 0.2 straight away; a 10-token translation has per-token probability 0.8. compare their raw and length-normalized scores ($\alpha = 1$).

<details><summary>solution</summary>

raw: $\ln 0.2 = -1.61$ for the empty output against $10\ln 0.8 = -2.23$ for the translation: the empty output wins. normalized: $-1.61/1$ against $-2.23/10 = -0.22$: the translation wins. the raw sum compares total surprise, which can only grow with length; normalization compares surprise per token.

</details>

## experiment

`lab.py` trains your `Seq2Seq` to rewrite dates in six styles as ISO dates (20,000 training pairs):

1. with and without attention: learning curves for the year, the month and the day.
2. the attention maps of the trained model, for different source formats.
3. greedy decoding against beam search with beams of 3 and 10, for both models.

## check your understanding

1. why does the decoder attend with $s_{t-1}$ rather than $s_t$?

<details><summary>answer</summary>

because $s_t$ is computed *from* the context in Bahdanau's design: the decoder first decides where to look, given what it has written so far ($s_{t-1}$), then updates its state with what it found. Luong et al. (2015) use the other order (compute $s_t$ first, then attend with it and combine), which works as well; what matters is that the order is consistent between training and decoding.

</details>

2. the model without attention learns the year almost immediately and the day very slowly in the lab. why?

<details><summary>answer</summary>

the decoder only sees the encoder's final state. in most formats the year is the last thing the encoder reads, so it's still vivid in the final state; the day was read 10–20 steps earlier and has to survive all the later updates, and the gradient that would teach the encoder to keep it has to travel just as far back (topic 04's vanishing gradient again). attention gives every decoder step a direct path to every encoder state.

</details>

3. why must `encode` take the state at the last *valid* position when sources are padded?

<details><summary>answer</summary>

the GRU keeps running over the padding tokens, and its state at the final position has been updated by them: sources of different lengths would get initial decoder states that depend on how much padding they had. the tests check that a padded source gives exactly the output of the same source unpadded.

</details>

4. can attention weights be read as an explanation of the model's output?

<details><summary>answer</summary>

with care. in the lab they line up beautifully with the source characters each output depends on, but attention weights only say where the context vector came from, not how the decoder used it, and the encoder states themselves mix information from neighbouring positions (a spelled-out month is summarized at its last letter). Jain & Wallace (2019) show cases where very different attention patterns give the same predictions. chapter 28 treats attribution properly.

</details>

5. what does `length_penalty=0` do, and when would you want it?

<details><summary>answer</summary>

the score is the raw sum of log-probabilities, the true log-probability of the output: beam search then approximates the single most probable sequence, with its bias towards short outputs. when outputs have a known or fixed length (the lab's dates) the penalty doesn't matter; when you want calibrated sequence probabilities rather than a pleasant output, use 0.

</details>

6. why does `beam_search` stop as soon as `beam_size` hypotheses have finished, when a longer alive hypothesis might, after normalization, score better?

<details><summary>answer</summary>

it's a convention that bounds the work and makes the result well defined; with a shrinking beam, once `beam_size` hypotheses have finished there are no slots left anyway. a longer hypothesis could indeed overtake after normalization, which is why some implementations keep searching until no alive hypothesis can beat the best finished one. for the contract used across the course, the simpler rule is fixed and tested.

</details>

## common bugs

- **masking after the softmax.** symptom: weights on real positions that sum to less than 1 for padded sources; the padding test fails.
- **taking the encoder's last position as the initial state for padded sources.** symptom: outputs that change with the amount of padding.
- **feeding the decoder `tgt_out` instead of `tgt_in`.** symptom: training loss near 0 almost instantly (the model copies its input) and garbage at test time.
- **attending with $s_t$ after updating it, but with $s_{t-1}$ in `greedy_decode`** (or any other mismatch between training and decoding). symptom: perfect teacher-forced accuracy, poor decoding.
- **beam search that keeps finished hypotheses in the beam.** symptom: `step_fn` called on prefixes ending in `eos`, outputs with tokens after `eos`.
- **normalizing scores during the search instead of at the end.** symptom: results that differ from the reference; the course's contract prunes by the raw cumulative log-probability and normalizes only for the final ranking.

## going deeper

- Bahdanau, Cho & Bengio (2015), "Neural machine translation by jointly learning to align and translate", *ICLR*. additive attention. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Sutskever, Vinyals & Le (2014), "Sequence to sequence learning with neural networks", *NeurIPS*. the LSTM encoder–decoder, beam search, and reversing the source. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk & Bengio (2014), "Learning phrase representations using RNN encoder–decoder for statistical machine translation", *EMNLP*. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
- Luong, Pham & Manning (2015), "Effective approaches to attention-based neural machine translation", *EMNLP*. dot-product and local attention. [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
- Wu et al. (2016), "Google's neural machine translation system: bridging the gap between human and machine translation". length normalization in beam search. [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)
- Holtzman, Buys, Du, Forbes & Choi (2020), "The curious case of neural text degeneration", *ICLR*. why beam search is bad for open-ended text. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Jain & Wallace (2019), "Attention is not explanation", *NAACL*. [arXiv:1902.10186](https://arxiv.org/abs/1902.10186)

## connections

- ← builds on: GRUs and `nn.GRU`/`nn.GRUCell` (topic 05), embeddings (topic 03), the softmax and cross-entropy (ch 08), masking in PyTorch (ch 15), the chain rule for sequences (topic 01).
- → used in: self-attention and the transformer, where queries, keys and values all come from one sequence and recurrence disappears (ch 18 · topic 01); cross-attention in encoder–decoder transformers and diffusion models (ch 18, ch 22); beam search for language models (ch 27); attention maps as (imperfect) explanations (ch 28).
