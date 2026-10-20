# 17 · sequence models

> model sequences, from counting to recurrence to attention.

text, speech, music, code, DNA and sensor streams have one thing in common: what comes next depends on what came before. this chapter builds the models that learned to exploit that, in the order history discovered them. you'll start by counting: an n-gram language model with smoothing, interpolation weights fitted by EM, and perplexity, the number every language model since has been judged by. you'll build the byte-level BPE tokenizer that turns any string into integers and back, exactly, and that chapter 18's GPT will read its training text through. you'll learn word vectors from co-occurrence with word2vec, and find out how much of the famous vector arithmetic survives on 200,000 words of Shakespeare. then recurrence: an RNN with backpropagation through time written by hand, a measurement of how its gradient fades across time steps, and truncated BPTT; the LSTM and GRU cells that fixed it, checked against PyTorch to the last digit; and an encoder–decoder with Bahdanau attention and beam search. the capstone puts it together: a character-level LSTM language model on tiny Shakespeare, measured in bits per character with an honest error bar against the best n-gram you can build, and sampled with a temperature. on the way you'll meet an add-one bigram model that is three times *worse* than ignoring context, a Chinese sentence that a Shakespeare tokenizer makes three times longer than its character count, "love" and "hate" as nearest neighbours, a gradient that shrinks by a factor of several million through 50 steps of a length-preserving matrix, an LSTM whose memory is decided by one bias at initialization, and a model without attention that learns the year of a date in a hundred steps and the day in three thousand.

**you will build:** `NGramLM` with add-k and interpolated smoothing, `tune_lambdas` (EM), perplexity and sampling (topic 01); `CharTokenizer`, `pair_counts`, `merge_pair`, GPT-2-style `pretokenize`, and a byte-level `BPETokenizer` with train/encode/decode/save/load and special tokens (02); `skipgram_pairs`, `sample_negatives`, the `SkipGram` negative-sampling model, `most_similar`, `analogy`, `train_skipgram` (03); `rnn_forward` and a hand-written `rnn_backward` (BPTT) in NumPy, a PyTorch `RNN`, `hidden_grad_norms`, `batchify`, `tbptt_batches`, `CharRNN`, `train_tbptt` (04); `LSTMCell`, `GRUCell`, multi-layer `LSTM` and `GRU`, and a recall task with `recall_batch`, `SeqTagger`, `copy_accuracy` (05); `BahdanauAttention`, an attention `Seq2Seq` with teacher forcing, `greedy_decode`, and a model-agnostic `beam_search` (06). modules: **`mlfs.text`** (topics 01–02, pure Python/NumPy) and **`mlfs.seq`** (topics 03–06, PyTorch).
**prerequisites:** probability and the chain rule (ch 04), cross-entropy, entropy and PMI (ch 05), EM for mixtures (ch 11), backprop by hand (ch 14), PyTorch modules, training loops, gradient clipping and learning-rate schedules (ch 15: `mlfs.torchnn`, `mlfs.train`), the bootstrap (ch 05), and chapter 04's n-gram capstone.
**time · scaffolding:** about 75 hours · L3 (independent): every function and method comes with its signature and a one-line docstring; the conventions (padding symbols, gate orders, tie rules, the beam-search contract) are in the lessons. hints have two levels. the capstone gives you data loading and the n-gram baseline; the model, the training loop, the evaluation and the sampler are yours.
**newly unlocked library components:** from here on you may use, in PyTorch: `nn.Embedding` (topic 03; you built word2vec's tables), `nn.RNN` and `nn.RNNCell` (04), `nn.LSTM`, `nn.LSTMCell`, `nn.GRU` and `nn.GRUCell` (05); topic 06 unlocks nothing from PyTorch's library (`nn.MultiheadAttention` waits until chapter 18 has you build it), but its beam search replaces any library decoder. for text: tokenizer libraries such as Hugging Face `tokenizers` or `tiktoken` for byte-level BPE (02; your `mlfs.text.BPETokenizer` stays the course's tokenizer), `gensim`-style word2vec (03), and n-gram toolkits such as KenLM (01). inside `mlfs`, `mlfs.text.BPETokenizer` and `mlfs.seq.beam_search` are what chapters 18 and 27 build on.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [language modeling](01-language-modeling/) | `NGramLM` (add-k, interpolated, perplexity, sampling), `tune_lambdas` → `mlfs.text` | add-k's character perplexity more than doubles from $n = 5$ to $n = 8$ while EM interpolation keeps improving (5.20, 2.38 bits/char); the add-one word bigram is almost 3× worse than the unigram |
| 02 | [tokenization](02-tokenization/) | `CharTokenizer`, `pair_counts`, `merge_pair`, `pretokenize`, `BPETokenizer` → `mlfs.text` | the last 2,048 of 4,096 merges shorten the text by only 12%; Chinese takes 3 tokens per character; the same trigram model drops from 3.00 to 2.38 bits/char just by switching to BPE tokens |
| 03 | [embeddings](03-embeddings/) | `skipgram_pairs`, `sample_negatives`, `SkipGram`, `most_similar`, `analogy`, `train_skipgram` → `mlfs.seq` | "hate" is closer to "love" than "dear" is; without excluding the query words, 92% of analogy "answers" are one of the inputs; one SVD of a PMI matrix does as well as word2vec |
| 04 | [RNNs & BPTT](04-rnns-and-bptt/) | `rnn_forward`, `rnn_backward`, `RNN`, `hidden_grad_norms`, `batchify`, `tbptt_batches`, `CharRNN`, `train_tbptt` → `mlfs.seq` | with every singular value of $W_{hh}$ equal to 1, the gradient still shrinks to $2 \times 10^{-7}$ of its size over 50 steps; tanh at $\rho = 3$ grows over a millionfold; a 6-step memory is learnable only with truncation $\ge 6$ |
| 05 | [LSTMs & GRUs](05-lstms-and-grus/) | `LSTMCell`, `GRUCell`, `LSTM`, `GRU`, `recall_batch`, `SeqTagger`, `copy_accuracy` → `mlfs.seq` | the RNN can't learn a 5-step recall, the LSTM learns 20 but not 40, the GRU learns 80; the LSTM's reach is set by its forget-gate bias at initialization (0: none, 1: 20 steps, 3: 40) |
| 06 | [seq2seq & attention](06-seq2seq-and-attention/) | `BahdanauAttention`, `Seq2Seq`, `greedy_decode`, `beam_search` → `mlfs.seq` | without attention, the year (always written last) is learned at once and the day only after thousands of steps; with attention, everything by step 200, with sharp alignments nobody supervised; beam search adds nothing |

## capstone

*a character-level LSTM language model*: a 2-layer, 512-unit LSTM on tiny Shakespeare (same 80/10/10 split as chapter 04), trained with your warmup + cosine schedule, evaluated with its state carried across the whole validation and test blocks, with block-bootstrap intervals, against an interpolated order-7 n-gram from your `mlfs.text`, and sampled with a temperature. the reference reaches **2.35 bits per character on test** (95% block-bootstrap interval [2.31, 2.40]; 2.35–2.37 over three seeds) against 2.63 for the order-7 interpolated n-gram (paired difference −0.28 [−0.29, −0.27]) and 2.83 for chapter 04's add-k model, in about 10 minutes of training on a laptop GPU (`mps`). see [`capstone/brief.md`](capstone/brief.md). small CPU config: a few seconds on the offline synthetic stand-in.

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (an interpolated n-gram probability and perplexity, three BPE merges with a tie, one LSTM step and one beam-search step), and a 60-minute blind rebuild of the byte-level BPE tokenizer.

## notation

| symbol | meaning |
|---|---|
| $w_t$, $x_t$, $y_t$ | the token at position $t$ (input), and the target token |
| $h$, $h_{[j]}$ (n-grams) | a context; its last $j$ tokens |
| $n$ | n-gram order: the model conditions on $n - 1$ tokens |
| $c(h, w)$, $c(h)$ | counts of a context followed by a token, and of the context |
| $k$ (add-k), $\lambda_j$ | add-k pseudo-count; interpolation weight of the order-$(j+1)$ estimate |
| $\|V\|$, $V$ | vocabulary size |
| PP, bits/char | perplexity $\exp(-\frac1N\sum\log P)$; $\log_2$ of it per character |
| $v_w$, $u_w$ | word2vec input (center) and output (context) vectors |
| $K$ | number of negative samples; also the number of symbols in the recall task |
| $P_n$ | noise distribution, unigram counts to the power 3/4 |
| $h_t$, $c_t$ | RNN/LSTM hidden state and LSTM cell state |
| $W_{xh}$, $W_{hh}$, $b$ | input, recurrent weights and bias (nn.Linear layout: out × in) |
| $i, f, g, o$ | LSTM input, forget, candidate, output |
| $r, z, n$ | GRU reset gate, update gate, candidate |
| $\rho$ | the singular value(s) of $W_{hh}$ in topic 04's lab |
| $q$, $k_j$, $\alpha_j$, $c$ | attention query, keys, weights and context |
| $s_t$ | decoder state |
| $\alpha$ (beam search) | length penalty exponent |

## how this connects

- builds on: probability, Markov chains and chapter 04's n-gram capstone, information theory and the bootstrap (ch 05), EM (ch 11), backprop (ch 14), and the PyTorch toolkit and training recipes of chapter 15.
- used later in: chapter 18's transformers and GPT (trained on your BPE tokens; self-attention generalizes this chapter's attention), learning dynamics and scaling of language models (ch 19), contrastive objectives that generalize negative sampling (ch 25), the systems view of sequence models (ch 26), large language models with special tokens, perplexity-based evaluation and your beam search (ch 27), and attention maps and attribution (ch 28).

## core references

- Jurafsky & Martin (2025), *Speech and Language Processing*, 3rd ed. draft, chapters 3 (n-grams), 2 (tokenization), 6 (vector semantics), 8 (RNNs and LSTMs) and 13 (machine translation and attention). [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, chapter 10: sequence modeling, recurrent and recursive nets. [deeplearningbook.org](https://www.deeplearningbook.org)
- Hochreiter & Schmidhuber (1997), "Long short-term memory", *Neural Computation* 9(8): 1735–1780. [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
- Bahdanau, Cho & Bengio (2015), "Neural machine translation by jointly learning to align and translate", *ICLR*. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Sennrich, Haddow & Birch (2016), "Neural machine translation of rare words with subword units", *ACL*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Karpathy (2015), "The unreasonable effectiveness of recurrent neural networks". [karpathy.github.io/2015/05/21/rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
