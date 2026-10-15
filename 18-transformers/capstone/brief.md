# 18 · capstone: nanoGPT

## goal

train GPTs on tiny Shakespeare with your own `mlfs.transformer`, measure the best one exactly the way chapter 17 measured its LSTM, find out how the loss depends on model size, and make generation fast with your KV cache. this is the small-scale version of the experiment behind GPT-2 and GPT-3: one architecture, trained at several sizes on the same data with the same recipe, with the loss falling smoothly as the model grows. Karpathy's nanoGPT trains the same kind of model on the same data; yours is built from parts you wrote.

every topic of the chapter is in it. the model is your `GPT` (topic 04): your attention with its causal mask (01), learned positions (02), pre-LN blocks (03), weight tying, GPT-2 initialization and your `train_lm`. evaluation uses sliding windows, because a GPT can't carry a state the way the LSTM did. generation uses your `KVCache` and `generate_cached` (06). the encoder of topic 05 and the ViT of topic 07 aren't used, and question 6 of the write-up asks why. the honest-evaluation habits are older: a validation block for every decision, the test block once, and block-bootstrap intervals (chapter 05, and chapter 17's capstone).

## data

**tiny Shakespeare** (1,115,394 characters, 65 distinct), from `mlfs.datasets.tiny_shakespeare()`, cached under `~/.cache/mlfs/`, split by position exactly like chapters 04 and 17: the first 80% (892,315 characters) trains, the next 10% (111,539) validates, the last 10% (111,540) is the test set. characters are ids through your `CharTokenizer`, so bits per character compare directly with chapter 17's numbers. offline (`MLFS_OFFLINE=1`) you get a synthetic stand-in of random common words, so the notebook always runs. the tests use neither: they use small synthetic sequences, a context-free "bigram" model whose per-token losses are known, and exact power laws.

## configurations

| config | models | training | runtime |
|---|---|---|---|
| **small** (tests, `verify.py`, offline, or `MLFS_CAPSTONE=small`) | three GPTs of width 32–64, 1–2 layers, `block_size` 64 | 200 steps of 32 windows, CPU | under a minute for the whole notebook |
| **full** (your write-up) | **S**: $d$ 96, 3 layers, 4 heads (354,240 parameters); **M**: $d$ 192, 4 layers, 6 heads (1,816,896); **L**: $d$ 320, 6 layers, 8 heads (7,460,160); `block_size` 128, dropout 0.1, tied embeddings, learned positions | each 2,000 steps of 64 windows of 128 characters (about 18 passes over the training text), your `train_lm`: AdamW at $10^{-3}$ ($\beta = 0.9, 0.95$, weight decay 0.1 on matrices), 100 warmup steps, cosine to $10^{-4}$, clipping at 1, evaluation every 250 steps | **25.4 minutes** for the whole notebook, measured once on an Apple-silicon laptop GPU (`mps`) shared with other jobs: 2.9, 6.0 and 16.1 minutes of training for S, M and L, the rest evaluation, timing and sampling. use `get_device()`; on a CPU expect several times longer |

`block_size` 128 rather than nanoGPT's 256 keeps the full run inside a 30–45-minute budget on a shared laptop; a longer context is the first thing to try with more compute (stretch goals).

## deliverables

1. `capstone.py` with the five functions implemented and their checks green:
   - `split_ids(ids, *, train_frac=0.8, val_frac=0.1)`: the split by position, three int64 arrays
   - `token_nll(model, ids, *, block_size, stride=None, batch_size=16, device="cpu")`: the loss of every token after the first, from windows of `block_size` that start at 0, `stride`, `2·stride`, ... (default `block_size // 2`) plus one window ending at the last token; the first window scores all its positions, each later one only the positions not yet scored, so every token past the first window has at least `block_size − stride` characters of context. eval mode, no gradients, mode restored; a float64 array of length `len(ids) − 1`
   - `bpc_ci(nll, *, chars_per_token=1.0, block_size=2000, n_boot=2000, alpha=0.05, rng=None)`: bits per character (mean nll / ln 2 / `chars_per_token`) with a bootstrap interval over contiguous blocks (`np.array_split` into `len // block_size` blocks, at least 2; resample blocks and recompute the ratio of summed losses to summed counts with your `bootstrap_ci`)
   - `fit_scaling(n_params, losses)`: $(a, b)$ of $L = aN^{-b}$ by least squares on $\log L$ against $\log N$
   - `generation_speed(model, prompt, n_new, *, cached=True, device="cpu")`: new tokens per second for greedy generation, with your `generate_cached` (`cached=True`) or the model's own `generate`, synchronizing the device around the timing
2. the full-config run, and the write-up section at the end of the notebook, answering its six prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass (`pytest test_capstone.py`) |
| the scaling run | validation loss strictly decreasing in parameters across the three sizes, with the fitted exponent and its plot |
| the best model on the test block | test bits per character below the order-7 n-gram's 2.631, with a 95% block-bootstrap interval that excludes it |
| cached inference | greedy cached and uncached generation produce the same tokens, and the cached version is faster on the CPU |
| reporting | validation and test bits per character with intervals next to chapter 17's LSTM and n-gram; the three sizes' losses; the speedups on your device and the CPU |

what the reference gets (full config, seed 0, `mps`):

| model | parameters | train bits/char (random windows) | validation bits/char (random windows) | minutes |
|---|---|---|---|---|
| S | 354,240 | 2.406 | 2.567 | 2.9 |
| M | 1,816,896 | 1.966 | 2.190 | 6.0 |
| L | 7,460,160 | 1.639 | 2.062 | 16.1 |

the validation curves are still falling at step 2,000 for all three sizes, slowly for L; none has turned upward, though L's train–validation gap (0.42 bits) is the largest. a single power law through the three final validation losses (nats per character, from `train_lm`'s random windows) gives $L \approx a N^{-0.073}$, but the curve bends: the local slope is 0.097 from S to M and 0.043 from M to L. with three points, one seed and the same 2,000 steps for every size, the larger models are the furthest from convergence and closest to the limits of 892,000 characters of data, so the flattening says as much about the fixed budget as about the models; don't read the fitted exponent as a scaling law (write-up prompt 2, and chapter 19).

the best model, L, on every character of the validation and test blocks (sliding windows of 128 with stride 64, so each character has 64–127 characters of context; 95% block-bootstrap intervals with blocks of 2,000 characters):

| model | validation bits/char | **test** bits/char |
|---|---|---|
| GPT L, 7.5M parameters, 128-character context | 2.033 [1.966, 2.104] | **2.457** [2.400, 2.514] |
| chapter 17's char-LSTM, 2 × 512, state carried through the whole text | 2.026 [1.968, 2.086] | 2.353 [2.310, 2.397] |
| chapter 17's interpolated n-gram, order 7 | 2.266 [2.209, 2.324] | 2.631 [2.585, 2.679] |

the GPT beats the n-gram clearly (0.17 bits on test) and ties the LSTM on validation, but it is 0.10 bits *worse* than the LSTM on test, with intervals that barely don't overlap. this isn't the result most people expect from "transformers replaced LSTMs", and it has reasons you can investigate: the LSTM was evaluated with its state carried across the whole test block (unlimited context, in effect), the GPT sees at most 127 characters; the LSTM trained for 2,500 steps of 64 × 128 characters with dropout 0.25, the GPT for 2,000 steps with dropout 0.1, and L's train–validation gap suggests it would benefit from more regularization; and the test block is a different part of the text from validation (chapter 17 found the same 0.3-bit shift for both models), so a 0.10 difference on test after a tie on validation is partly about which model generalizes across that shift. the notebook's comparison is unpaired (the LSTM's per-character losses aren't saved by chapter 17), which is weaker than chapter 17's paired one. a reviewer paired the two models' per-character losses from separate runs of both capstones: GPT − LSTM is +0.107 bits per character on test [0.085, 0.129] and +0.010 on validation [−0.005, 0.024] (block bootstrap, blocks of 2,000): a real gap on test, none on validation (write-up prompt 1). the validation numbers from sliding windows (2.033) are lower than `train_lm`'s random-window estimate (2.062) because every scored character has at least 64 characters of context, while a random window scores early positions with almost none.

cached generation, 121 new tokens after `ROMEO:\n`, greedy: on `mps` 20 tokens per second without the cache and 40 with it (2.0×); on the CPU 39 and 161 (4.1×). on this small model the laptop GPU generates *slower* than the CPU: each step is a few tiny kernels, and launching them costs more than computing them (write-up prompt 4). samples at temperature 0.8 have the format and much of the vocabulary of the play ("ROMEO: / Thou must know the field and good womanicely from / how to give thee receive thy soul's deed, / But then thou love thee and t"), and copy nothing verbatim (0% of 20-character chunks are in the training text).

## stretch goals

- **a longer context.** `block_size` 256 (nanoGPT's), with stride 128 for evaluation. does the test gap to the LSTM close? at what cost per step?
- **regularization for L.** dropout 0.2, or weight decay 0.2, at the same step budget; and a second seed of the best configuration.
- **a paired comparison.** rerun chapter 17's LSTM evaluation, save its per-character test losses, and report the paired difference with a block-bootstrap interval.
- **BPE tokens.** train on your `BPETokenizer` with 512 tokens and report bits per *character* with `chars_per_token`. the context then covers several times more text.
- **a fourth size, and compute-optimal training.** add a size and train each to a fixed amount of *compute* ($6ND$, chapter 19) instead of a fixed number of steps.
- **rotary positions.** swap learned positions for your `RotaryEmbedding` inside the attention, and check topic 02's extrapolation result at this scale.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Karpathy (2022), nanoGPT. [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Radford, Wu, Child, Luan, Amodei & Sutskever (2019), "Language models are unsupervised multitask learners" (GPT-2). [cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu & Amodei (2020), "Scaling laws for neural language models". [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022), "Training compute-optimal large language models" (Chinchilla). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Al-Rfou, Choe, Constant, Guo & Jones (2019), "Character-level language modeling with deeper self-attention", *AAAI*. transformers against LSTMs at the character level. [arXiv:1808.04444](https://arxiv.org/abs/1808.04444)
- Künsch (1989), "The jackknife and the bootstrap for general stationary observations", *Annals of Statistics* 17(3): 1217–1241. [doi:10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265)
