# tokenization · lab notes

numbers are for tiny Shakespeare (the cached file), measured with the reference solution: the tokenizer trains on the first 80% (892,315 characters) and is scored on the next 10% (111,539 characters). the whole lab takes about 10 s with the reference; training the 4,096-token vocabulary takes 0.6 s. a trainer that recounts every pair after every merge needs about 30 ms per merge at the start on this text, a minute or two for the 3,840 merges. offline, the synthetic stand-in has only 120 distinct chunks (common words, with and without a leading space or capital), so training runs out of pairs after 166 merges and every vocabulary from 512 up is the same 422 tokens.

## experiment 1: vocabulary size against sequence length

**what people usually predict:** that sequence length shrinks roughly in proportion to the vocabulary, or that a 4,096-token vocabulary covers about one word (5–6 characters) per token.

**what happens:**

| vocabulary | held-out tokens | characters per token | last tokens learned |
|---|---|---|---|
| 256 | 111,539 | 1.00 | (bytes) |
| 320 | 74,710 | 1.49 | `ut`, `'s`, `ith`, `im` |
| 512 | 58,542 | 1.91 | `--`, ` sw`, `LO`, ` love` |
| 1024 | 47,491 | 2.35 | `erc`, `ons`, ` le`, ` peace` |
| 2048 | 41,380 | 2.70 | ` something`, ` battle`, ` beauty`, ` spirit` |
| 4096 | 36,225 | 3.08 | `Nothing`, `Pre`, `Rem`, `Set` |

the first 64 merges cut a third of the tokens; the last 2,048 cut only 12% (41,380 → 36,225). on a log axis the curve is close to a straight line: each doubling of the vocabulary buys roughly the same, shrinking, absolute gain.

**why:** each merge removes one token per occurrence of its pair, and BPE takes the most frequent pair first, so the savings per merge fall as fast as the pair frequencies do. the first merges are ` t` (19,514 occurrences in the training text when it was merged), `he` (14,708) and ` a` (10,572); merge 64 (`im`) had 1,645, merge 1,000 (` Marcius`) 73, merge 3,000 (` former`) only 14. late tokens are whole words, usually with their leading space (` battle`, ` spirit`), because the pre-tokenizer attaches a word's preceding space to it, plus capitalized line-initial words and fragments of names (`Nothing`, `Pre`). for a transformer, sequence length costs compute (quadratically in attention) while vocabulary costs parameters (the embedding and output layers), and the diminishing returns are why practical vocabularies stop in the tens of thousands even for huge corpora.

## experiment 2: text the tokenizer has never seen

**what people usually predict:** that everything not in the training data is encoded badly but somewhere near one token per character, or that the tokenizer fails on Chinese.

**what happens** (2,048-token Shakespeare tokenizer):

| text | characters | UTF-8 bytes | tokens | bytes per token | tokens per character |
|---|---|---|---|---|---|
| Shakespeare (held-out) | 400 | 400 | 134 | 2.99 | 0.34 |
| modern English | 84 | 84 | 31 | 2.71 | 0.37 |
| French | 77 | 78 | 37 | 2.11 | 0.48 |
| Python code | 58 | 58 | 37 | 1.57 | 0.64 |
| digits | 62 | 62 | 50 | 1.24 | 0.81 |
| Chinese | 18 | 54 | 54 | 1.00 | 3.00 |
| emoji | 6 | 24 | 24 | 1.00 | 4.00 |

every string round-trips. modern English is nearly as cheap as Shakespeare, French costs 40% more per character, and code and numbers are spelled out almost character by character. Chinese takes **3 tokens per character** and emoji 4: more tokens than characters. the digits line is cut into single digits (`4`·`8`·`2`·`1`·`3`), with words split into odd pieces (`Or`·`d`·`er`, ` s`·`hi`·`pp`·`ed`).

**why:** byte-level BPE can always fall back to bytes, so nothing is unknown, but it only has merges for what was frequent in training. Shakespeare contains no CJK characters and no emoji, so they stay as raw UTF-8 bytes (3 per CJK character, 4 per emoji). Shakespeare has few digits either, so numbers become one token per digit; a model trained on such tokens has to learn arithmetic over an arbitrary, frequency-driven segmentation, a known weakness of BPE language models (GPT-2's tokenizer splits numbers into irregular multi-digit chunks, which is arguably worse). the lesson: train the tokenizer on the same mix of languages and domains the model will see, or the model pays for it in sequence length and in harder-to-learn units.

## experiment 3: the same n-gram model on different tokens

**what people usually predict:** that the character model wins (it's the "finest" representation), or that bigger vocabularies always win (more context per token).

**what happens** (interpolated order-3 `NGramLM`, EM weights, 55,774 held-out characters scored):

| tokens | held-out tokens | perplexity per token | bits per character | EM weights (orders 1, 2, 3) |
|---|---|---|---|---|
| characters | 55,774 | 8.02 | 3.003 | 0.00 0.02 0.98 |
| BPE 320 | 37,369 | 14.09 | 2.557 | 0.01 0.12 0.88 |
| BPE 512 | 29,107 | 24.97 | 2.423 | 0.03 0.31 0.65 |
| BPE 1024 | 23,864 | 47.24 | **2.380** | 0.11 0.43 0.47 |
| BPE 2048 | 20,829 | 85.23 | 2.395 | 0.19 0.46 0.35 |

the same counting model, with the same order, goes from 3.00 to 2.38 bits per character just by changing what a token is. the best is 1,024 tokens; 2,048 is slightly worse. measured per token, the ranking is exactly reversed: perplexity rises more than tenfold from characters to BPE 2048.

**why:** an order-3 model always sees 2 tokens of context. two characters is very little; two BPE tokens are 4–6 characters, so the model effectively becomes a higher-order model, and topic 01 showed how much that helps when it's done without sparsity blowing up. but contexts get sparser as the vocabulary grows (more distinct tokens, each seen less), and EM's weights show the model trusting the full trigram less and less (0.98 → 0.35) and leaning on the bigram and unigram instead; past about 1,024 tokens the sparsity cost cancels the extra context. per-token perplexity is not comparable across tokenizations because a BPE token is a much bigger prediction than a character. that is why language-model papers report bits per byte or per character when tokenizers differ.
