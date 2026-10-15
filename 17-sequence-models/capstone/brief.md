# 17 · capstone: a character-level LSTM language model

## goal

train a recurrent neural network to predict the next character of Shakespeare, measure it the way language models are measured, and make it write. this is the model of Karpathy's "The unreasonable effectiveness of recurrent neural networks" (2015), and the direct descendant of chapter 04's capstone: the same data, the same split, the same question ("how surprised is the model by the next character?"), with the count table replaced by a two-layer LSTM that reads the whole past.

every topic of the chapter shows up. perplexity and bits per character are topic 01's measure, and your interpolated `NGramLM` is the baseline. characters become ids through your `CharTokenizer` (topic 02) and vectors through an embedding (topic 03). the model is trained by backpropagation through time on windows of the text, and evaluated with its state carried across a whole block, the stateful view of topic 04. its core is an LSTM (topic 05; `nn.LSTM` is now unlocked). and it writes by decoding one token at a time, with a temperature (topic 06). the honest-evaluation habits come from earlier: a validation set for every decision, the test set once, and an interval that respects how the data are correlated (the bootstrap, chapter 05).

## data

**tiny Shakespeare** (about 1.1 million characters, 65 distinct), from `mlfs.datasets.tiny_shakespeare()`, cached under `~/.cache/mlfs/`. the notebook splits it by position exactly like chapter 04: the first 80% (892,315 characters) trains, the next 10% (111,539) is validation, the last 10% (111,540) is the test set. offline (`MLFS_OFFLINE=1`) you get a synthetic stand-in: about 77,000 characters of common English words drawn independently, so the notebook always runs. the tests use neither: they sample from an order-1 Markov chain over 8 symbols whose entropy rate is known exactly.

## configurations

| config | model | training | data | runtime |
|---|---|---|---|---|
| **small** (tests, `verify.py`, offline, or `MLFS_CAPSTONE=small`) | 2 × 128 units, embeddings of 32, no dropout (221k parameters) | 300 steps, batches of 32 × 64 characters, CPU | the offline stand-in; order-4 n-gram | about 30 s for the whole notebook on a laptop CPU |
| **full** (your write-up) | 2 × 512 units, embeddings of 64, dropout 0.25 (3.32M parameters) | 2,500 steps, batches of 64 × 128 characters (about 23 passes over the training text), AdamW at $2 \times 10^{-3}$, 100 warmup steps, cosine to $2 \times 10^{-4}$, clipping at 1, validation every 250 steps, best checkpoint kept | tiny Shakespeare; order-7 n-gram | 9.7 minutes of training measured on an Apple M2 laptop GPU (`mps`, on a busy machine), plus well under a minute for the n-gram, the evaluation and the samples; on the CPU alone a step takes about five times longer (1.4 s against 0.26 s in our benchmark), so use `get_device()` |

the notebook picks the config automatically (`datasets.is_offline()`), and `get_device()` picks `cuda`, then `mps`, then `cpu`.

## deliverables

1. `capstone.py` with the required pieces implemented and their checks green:
   - `CharLSTM(vocab_size, *, embed_dim, hidden_dim, num_layers, dropout)`: embedding → `nn.LSTM` → dropout → linear; `forward(idx, state=None)` returns `(logits, state)`
   - `get_batch(ids, batch_size, seq_len, *, generator=None)`: random training windows and their next-character targets
   - `train_lm(model, train_ids, *, steps, ...)`: AdamW, your `CosineSchedule` with warmup, gradient clipping, validation every `eval_every` steps, and the best-validation weights restored at the end
   - `char_nll(model, ids, *, seq_len=256, device="cpu")`: the loss of every character, with the state carried through the whole text
   - `bpc_ci(nll, *, block_size=2000, ...)`: bits per character with a bootstrap interval over contiguous blocks
   - `sample(model, prompt_ids, n_tokens, *, temperature=1.0, generator=None, device="cpu")`: temperature sampling
2. the write-up section at the end of the notebook, answering its six prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass (`pytest test_capstone.py`) |
| learning a known source | on the tests' 8-symbol Markov chain, the median best validation loss over 5 seeds is within 0.08 nats of the chain's entropy rate after 100 steps (`test_stage3_learns_the_chain`) |
| the full config on tiny Shakespeare | test bits per character below the order-7 interpolated n-gram's, with a paired block-bootstrap interval for the difference that excludes 0 |
| reporting | validation and test bits per character and perplexity for both models with 95% block-bootstrap intervals, the paired difference, and chapter 04's add-k result next to them |

what the reference gets (full config, seed 0, `mps`; bits per character with 95% block-bootstrap intervals, blocks of 2,000 characters):

| model | validation | perplexity | **test** | perplexity |
|---|---|---|---|---|
| LSTM, 2 × 512 | 2.026 [1.968, 2.086] | 4.07 | **2.353** [2.310, 2.397] | 5.11 |
| interpolated n-gram, order 7 (EM weights on validation) | 2.266 [2.209, 2.324] | 4.81 | 2.631 [2.585, 2.679] | 6.19 |
| chapter 04's add-k n-gram, order 4, $k = 0.1$ | 2.51 | 5.68 | 2.83 | 7.11 |

paired difference on test, LSTM − n-gram: **−0.278** bits per character [−0.293, −0.265]. the interval of the difference is much narrower than either model's own interval: both models find the same passages hard, so most of the block-to-block variation cancels in the difference.

the seed matters less than the test block: two more full-config runs (seeds 1 and 2, same code) reached 2.035 and 2.032 on validation and 2.372 and 2.374 on test, so across three seeds the test bits per character are 2.353–2.374 (mean 2.366, spread ±0.01), a quarter of the width of the block-bootstrap interval. every run's validation minimum was at step 1,250.

the validation curve bottoms out around step 1,250–1,500 (best validation-window loss 2.04 bits per character) and then creeps up while the training loss keeps falling to about 1.55: the model starts to overfit after about a dozen passes, and the best-checkpoint rule returns the weights from the minimum. the samples at temperature 0.8 have Shakespeare's format (speaker names, verse lines, archaic forms: "Thou must know so? a word inever'd, in the strength / As we have weeping for this behellion.") and copy almost nothing verbatim (2% of 20-character chunks are found in the training text).

notice, as in chapter 04, the gap between validation and test: 2.03 against 2.35 bits for the LSTM, 2.27 against 2.63 for the n-gram, far outside either interval. the blocks are contiguous, so the test block is different plays, and both models meet a shifted distribution. the block bootstrap accounts for correlation *within* a block of text; it can't account for the test block being a different kind of text from the validation block. treating characters as independent would make things worse still: for the n-gram on the test block, an i.i.d. bootstrap gives ±0.016 bits per character, against ±0.047 with blocks of 2,000 (write-up prompt 2).

small config (offline stand-in, CPU): the LSTM reaches 1.487 bits per character on test [1.465, 1.515] against the order-4 n-gram's 1.432 [1.412, 1.456]. on text made of independently drawn words there is nothing beyond the current word to learn, so the counting model, near-optimal for that source, wins; it's a check that the pipeline runs, not a result.

## stretch goals

- **BPE tokens.** train the same LSTM on your `BPETokenizer` with 512 or 1,024 tokens, and compare in bits per character (not per token).
- **stateful training.** replace random windows by topic 04's truncated BPTT over contiguous streams, carrying the state between batches. does it help at the same number of steps?
- **a GRU** (`nn.GRU`) of the same parameter count.
- **sampling strategies.** top-$k$ and nucleus sampling (Holtzman et al., 2020), compared at matched diversity, and your `beam_search` from topic 06 (expect it to repeat itself).
- **what a cell tracks.** find a hidden unit that tracks something interpretable (quotes, line length, inside a speaker name), as Karpathy, Johnson & Fei-Fei (2015) did.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Karpathy (2015), "The unreasonable effectiveness of recurrent neural networks", blog post, and the char-rnn repository tiny Shakespeare comes from. [karpathy.github.io/2015/05/21/rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Karpathy, Johnson & Fei-Fei (2015), "Visualizing and understanding recurrent networks". [arXiv:1506.02078](https://arxiv.org/abs/1506.02078)
- Graves (2013), "Generating sequences with recurrent neural networks". character-level LSTM language models. [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)
- Merity, Keskar & Socher (2018), "Regularizing and optimizing LSTM language models", *ICLR*. what makes LSTM language models strong. [arXiv:1708.02182](https://arxiv.org/abs/1708.02182)
- Künsch (1989), "The jackknife and the bootstrap for general stationary observations", *Annals of Statistics* 17(3): 1217–1241. the block bootstrap. [doi:10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265)
- Holtzman, Buys, Du, Forbes & Choi (2020), "The curious case of neural text degeneration", *ICLR*. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
