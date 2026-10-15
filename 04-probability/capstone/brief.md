# 04 · capstone: an n-gram text generator

## goal

build a language model out of nothing but counting and conditioning, measure how good it is, and make it write. a character **n-gram model** predicts the next character from the previous $n - 1$:

$$P(x_t \mid x_1, \dots, x_{t-1}) \approx P(x_t \mid x_{t-n+1}, \dots, x_{t-1}) = \frac{\text{count}(x_{t-n+1}, \dots, x_t) + k}{\text{count}(x_{t-n+1}, \dots, x_{t-1}) + kV}.$$

every topic of the chapter shows up. the model is a table of conditional distributions (topic 01). generating text is sampling from them with your own `categorical` (topic 02). perplexity, the standard measure of a language model, is the exponential of an expected log loss (topic 03), estimated by an average whose reliability rests on the limit theorems (topics 05–06). and the generator is a Markov chain whose state is the last $n - 1$ characters (topic 07).

this is the oldest kind of statistical language model (Shannon estimated the entropy of English with n-gram statistics in 1951) and the direct ancestor of the models in chapters 17–18 and 27. modern language models are evaluated with exactly the perplexity you'll compute here, and they generate text with exactly the temperature sampling you'll write. what changes later is how $P(x_t \mid \text{context})$ is computed: a neural network instead of a count table, and a context of thousands of tokens instead of a handful.

## data

**tiny Shakespeare**, about 1.1 million characters of Shakespeare's plays in one plain-text file, assembled by Andrej Karpathy for his char-rnn experiments. load it with

```python
from mlfs import datasets
text = datasets.tiny_shakespeare()   # str, ~1.1M characters, 65 distinct
```

the first call downloads the file once and caches it under `~/.cache/mlfs/`. with `MLFS_OFFLINE=1`, or without a network, you get a synthetic stand-in: about 77,000 characters of random common English words (37 distinct characters), drawn independently from a Zipf-like distribution, so the notebook always runs. the tests never touch either: they generate text from a known order-2 Markov chain over six letters, whose true perplexity they compute exactly.

the notebook splits the text **by position**, not at random: the first 80% trains, the next 10% is validation (to choose $n$ and $k$), and the last 10% is the test set, used once. contiguous blocks keep the test text genuinely unseen. a random split of positions would put most of every test n-gram's neighbours in the training set.

## configurations

| config | data | orders fitted | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | synthetic stand-in (77k characters), or the tests' own chain | $n = 1, \dots, 5$; $k \in \{1, 0.1, 0.01\}$ | about a second |
| **full** (your write-up) | tiny Shakespeare (890k training characters) | $n = 1, \dots, 6$; same $k$ grid | about 4 s for the evaluation cell and 6 s for the whole notebook (reference, laptop CPU); the order-6 count table is about 50 MB |

the notebook picks the config automatically from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required pieces implemented and their checks green:
   - `build_vocab(text)`, `encode(text, vocab)`, `decode(ids, vocab)`
   - `NGramLM(n, vocab_size, k=1.0)` with `fit(ids)`, `next_token_probs(context)`, `perplexity(ids, start=None)`, and `generate(prompt, n_tokens, *, temperature=1.0, rng=None)`, which samples with your `mlfs.sampling.categorical`
2. the write-up section at the end of the notebook, answering its six prompts (results against baselines, the overfitting curve, one ablation, samples and memorization, validation vs test under distribution shift, and where each topic of the chapter entered).

choose $n$ and $k$ on the validation set only. the notebook reports the test perplexity once, for the chosen model.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| held-out perplexity of an order-3 model on the tests' order-2 chain | within 2% of the true chain's perplexity on the same held-out characters (enforced by `test_stage3_threshold_recovers_true_entropy`; the reference stays within 0.5% across 300 random chains) |
| text sampled from your model, scored by another model | within 5% (in log perplexity) of the true value (`test_stage4_generated_text_has_the_right_statistics`) |
| test perplexity on tiny Shakespeare (full config), at the $(n, k)$ chosen on validation | below both the unigram and the bigram baseline, reported in bits per character as well |
| comparison with baselines | report the uniform ($V$), unigram and bigram test perplexities next to your result |

what the reference gets (perplexities; train is measured on the first 200,000 training characters):

| config | chosen on validation | validation | test | unigram (test) | bigram (test) | uniform |
|---|---|---|---|---|---|---|
| full: tiny Shakespeare | $n = 4$, $k = 0.1$ | 5.68 | **7.11** (2.83 bits/char) | 28.44 | 12.22 | 65 |
| small: offline stand-in | $n = 4$, $k = 0.01$ | 2.65 | **2.72** | 15.22 | 4.05 | 37 |

on Shakespeare, order 5 with $k = 0.01$ is a near tie on validation (5.69), and order 6 overfits badly (training 2.83, validation 7.26 at $k = 0.01$). on the stand-in, order 5 already overfits (training 2.54, validation 2.82): its words are drawn independently, so context beyond the current word can't help.

notice the gap between validation and test on Shakespeare: 5.68 against 7.11, about 25%, and it's there at every order. that is not noise. with about 110,000 characters per block, an i.i.d. error bar on a perplexity is around ±1%. the blocks are contiguous, so the test block is *different plays* (other characters' names, other topics), and the model meets a shifted distribution. explaining that, and what it means for the error bars the limit theorems give you, is write-up prompt 5.

## stretch goals

- **interpolation.** mix orders: $P = \lambda_3 P_3 + \lambda_2 P_2 + \lambda_1 P_1$ with weights tuned on validation (Jelinek–Mercer smoothing). does it beat the best single order?
- **Kneser–Ney.** the smoothing method that dominated n-gram language modeling for two decades: lower-order distributions count in how many different contexts a character appears, not how often. compare with add-k (Chen & Goodman, 1999).
- **words instead of characters.** tokenize on whitespace, and watch the vocabulary explode and the counts thin out. what happens to perplexity, and why can't you compare it directly with character perplexity? (convert both to bits per character.)
- **the entropy of English.** plot bits per character against $n$ and compare with Shannon's (1951) estimates from human predictions.
- **sampling strategies.** implement top-$k$ or nucleus (top-$p$) sampling (Holtzman et al., 2020) and compare samples at matched diversity.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Shannon (1951), "Prediction and entropy of printed English", *Bell System Technical Journal* 30(1): 50–64. [doi:10.1002/j.1538-7305.1951.tb01366.x](https://doi.org/10.1002/j.1538-7305.1951.tb01366.x)
- Chen & Goodman (1999), "An empirical study of smoothing techniques for language modeling", *Computer Speech & Language* 13(4): 359–394. [doi:10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)
- Jurafsky & Martin (2025), *Speech and Language Processing*, 3rd ed. draft, chapter 3: n-gram language models, smoothing and perplexity. free at [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/).
- Holtzman, Buys, Du, Forbes & Choi (2020), "The curious case of neural text degeneration", ICLR 2020. why greedy decoding repeats itself, and nucleus sampling. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Karpathy (2015), "The unreasonable effectiveness of recurrent neural networks", blog post, and the char-rnn repository that tiny Shakespeare comes from. [karpathy.github.io/2015/05/21/rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
